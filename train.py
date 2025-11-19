import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import editdistance
import numpy as np
from collections import defaultdict

import vocab_model
import collate
import model
import GPU



#读取数据
data = torch.load('ctc_dataset.pt')

def train_model(
    net,             #模型
    data,              #直接读取的pt数据，后续待初始化
    num_gpus = 8,
    num_epochs=10,
    batch_size = 128,
    lr =  1e-4,
    ):
    
    #定义GPU多卡训练
    devices = [GPU.try_gpu(i) for i in range(num_gpus)]
    device = devices[0]
    net.to(device)
    net = nn.DataParallel(net, device_ids=devices)
    nn.init.constant_(net.module.fc.bias[0], 2.0)
    #加载数据
    dataset = collate.CTCDataset(data)
    loader = DataLoader(dataset,
                    batch_size=batch_size,   # 这里直接设置 batch_size
                    shuffle=True,
                    collate_fn=collate.collate_ctc_batch)
    
    #提前将vocab中blank预留了0，zero_infinity避免梯度爆炸
    loss = nn.CTCLoss(blank=0, zero_infinity=True) 
    optimizer = optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        start = time.time()
        total_loss = 0
        net.train()
        for X,y,X_len,y_len in loader:
            batch_features = X.to(device)
            batch_labels   = y.to(device)
            input_lengths = X_len
            label_lengths = y_len
            
            optimizer.zero_grad()
            outputs = net(batch_features)
            
            #CTCLoss要求 shape -> [max_T, batch_size, num_classes]
            outputs = outputs.transpose(0,1)  
            l = loss(outputs, batch_labels, input_lengths, label_lengths)
            l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += l.item()
        avg_loss = total_loss / len(loader)
        # ⚡ 调试输出
        print("logits sample:", outputs[0, :5, :5])          # 第一条样本，前5帧、前5类
        print("predicted indices:", outputs.argmax(2)[0, :20]) # 贪心预测前20帧
        end = time.time()
        print(f"Epoch {epoch}/{num_epochs}, Avg Loss: {avg_loss:.4f},spend time:{end-start}")
    torch.save(net.state_dict(), "ctc_model.pth") 
    return net
        

#训练
# net = model.Model(input_size=26, hidden_size=128, num_layers=3, num_classes=61)
# net = train_model(net,data,num_gpus = 7,num_epochs= 400,batch_size = 32,lr = 4e-4)


def logsumexp(a, b):
    """安全求 log(e^a + e^b)"""
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))

def ctc_beam_search(log_probs, beam_width, blank_id, idx_to_token):
    """
    log_probs: [T, C] torch tensor, log_softmax 输出
    beam_width: beam size
    blank_id: blank 索引
    idx_to_token: 索引到字符的映射
    """
    T, C = log_probs.shape
    beam = [( (), 0.0 )] # (prefix, log_prob)
    
    for t in range(T):
        new_beam = {}
        for prefix, score in beam:
            for c in range(C):
                log_p = log_probs[t, c].item()  # 直接使用 log 概率
                if c == blank_id:
                    new_prefix = prefix
                else:
                    token = idx_to_token[c]
                    # CTC 合并重复
                    if len(prefix) > 0 and prefix[-1] == token:
                        new_prefix = prefix
                    else:
                        new_prefix = prefix + (token,)
                # 累积 log 概率
                if new_prefix in new_beam:
                    new_beam[new_prefix] = logsumexp(new_beam[new_prefix], score + log_p)
                else:
                    new_beam[new_prefix] = score + log_p

        # 选 top-k
        beam = sorted(new_beam.items(), key=lambda x: x[1], reverse=True)[:beam_width]

    return list(beam[0][0])

def ctc_prefix_beam_search(log_probs, beam_width, blank_id, idx_to_token):
    """
    完整 CTC prefix beam search
    log_probs: [T, C] torch tensor, log_softmax 输出
    beam_width: beam size
    blank_id: blank 索引
    idx_to_token: idx -> token 映射
    """
    T, C = log_probs.shape
    beam = {(): (0.0, float('-inf'))}  # prefix -> (log_p_blank, log_p_non_blank)

    for t in range(T):
        new_beam = defaultdict(lambda: (float('-inf'), float('-inf')))
        for prefix, (p_b, p_nb) in beam.items():
            for c in range(C):
                log_p = log_probs[t, c].item()

                if c == blank_id:
                    # 更新 p_blank: 可以从任何前缀转入 blank
                    nb_new, bb_new = new_beam[prefix]
                    new_beam[prefix] = (
                        logsumexp(bb_new, logsumexp(p_b + log_p, p_nb + log_p)),
                        nb_new
                    )
                else:
                    token = idx_to_token[c]
                    new_prefix = prefix + (token,)

                    if len(prefix) > 0 and prefix[-1] == token:
                        # 重复字符，必须经过 blank
                        p_nb_new = logsumexp(new_beam[new_prefix][1], p_b + log_p)
                    else:
                        p_nb_new = logsumexp(new_beam[new_prefix][1], logsumexp(p_b + log_p, p_nb + log_p))

                    bb_new = new_beam[new_prefix][0]
                    new_beam[new_prefix] = (bb_new, p_nb_new)

        # 保留 top-k 前缀
        beam = dict(sorted(new_beam.items(),
                           key=lambda x: logsumexp(x[1][0], x[1][1]),
                           reverse=True)[:beam_width])

    # 返回概率最高的前缀
    best_prefix = max(beam.items(), key=lambda x: logsumexp(x[1][0], x[1][1]))[0]
    return list(best_prefix)


# CER 计算函数
def cer(pred_seq, true_seq):
    """
    pred_seq, true_seq: list of int
    返回字符错误率
    """
    if len(true_seq) == 0:
        return 0 if len(pred_seq) == 0 else 1
    return editdistance.eval(pred_seq, true_seq) / len(true_seq)


def evaluate_ctc_model(net, data, device='cuda', batch_size=128, idx_to_token=None, beam_width=5):
    """
    net: 已训练模型（最后层 LogSoftmax）
    data: 数据字典
    device: 'cuda' 或 'cpu'
    batch_size: DataLoader batch size
    idx_to_token: list, 索引->字符
    beam_width: Beam Search 宽度
    """
    net.to(device)
    net.eval()
    dataset = collate.CTCDataset(data)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=collate.collate_ctc_batch)

    total_cer = 0
    num_samples = 0

    with torch.no_grad():
        for X, y, X_len, y_len in loader:
            X = X.to(device)
            y = y.to(device)

            outputs = net(X)              # [batch, T, C], log_softmax 输出
            probs = torch.exp(outputs)    # ← 从 log 概率恢复为概率，不再 softmax

            batch_decoded_strs = []

            for b in range(X.size(0)):
                prob_seq = probs[b]  # [T, C]
                decoded_str = ctc_beam_search(prob_seq, beam_width, blank_id=0, idx_to_token=idx_to_token)
                batch_decoded_strs.append(decoded_str)

            # 遍历 batch 计算 CER
            for pred_str, true_seq, seq_len in zip(batch_decoded_strs, y, y_len):
                true_seq = true_seq[:seq_len].cpu().numpy().tolist()
                true_str = ''.join([idx_to_token[i] for i in true_seq])
                total_cer += cer(pred_str, true_str)
                num_samples += 1

            break  # 只跑一个 batch 进行测试

    avg_cer = total_cer / num_samples
    return avg_cer, batch_decoded_strs


def test_output(net, data, device='cuda', batch_size=128 ,beam_width=5):
    net.to(device)
    net.eval()
    dataset = collate.CTCDataset(data)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=collate.collate_ctc_batch)
    
    with torch.no_grad():
        for X, y, X_len, y_len in loader:
            X = X.to(device)
            y = y.to(device)

            outputs = net(X)  # [batch, T, C]
            break
    return outputs.softmax(dim = 2)
# result = test_output(net,data,GPU.try_gpu(0),1,5)

# probs_np = result[0,:,:].cpu().numpy()  # [T, C]
# np.savetxt("probs.txt", probs_np, fmt="%.6f", delimiter=" ")
# preds_np = probs_np.argmax(1)
# np.savetxt("preds.txt", preds_np)

#avg_cer , result = evaluate_ctc_model(net,data,GPU.try_gpu(0),128,vocab_model.vocab.idx_to_token,5)

# 先测试一个样本，看看效果
# decoded_str = ctc_beam_search(probs_np, 20, 0, vocab_model.vocab.idx_to_token)
# print(decoded_str)

def collapse_repeats(seq):
    collapsed = []
    for i, s in enumerate(seq):
        if i == 0 or s != seq[i-1]:
            collapsed.append(s)
    return collapsed

vocab = torch.load("vocab.pt")
with open("/home/funrk25/code/research_projects/CTC_Graves2006/probs.txt", "r") as f:
    lines = f.readlines()

probs = [list(map(float, line.strip().split())) for line in lines]
probs = torch.tensor(probs)  # [T, C]
probs[:, 0] *= 0.5 
probs = probs / probs.sum(dim=-1, keepdim=True)
log_probs = torch.log(probs + 1e-8)
decoded_str = ctc_beam_search(log_probs, 50, 0, vocab.idx_to_token)
print(decoded_str)
temp = [44, 23,  6, 10, 44,  1, 16, 26, 27, 15, 20,  3, 54, 10, 19,  1, 39, 34,
        52, 25, 17, 33, 40,  5, 50, 20, 32, 37, 13, 31, 23, 25,  1, 10, 44]
print([vocab.idx_to_token[i] for i in temp])