import os

import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F

import vocab_model

data_dir = "timit/TIMIT/TRAIN/"

rows = []
for dr in sorted(os.listdir(data_dir)):
    dr_path = os.path.join(data_dir, dr)
    if not os.path.isdir(dr_path):
        continue
    for root, dirs, files in os.walk(dr_path):
        for f in files:
            if f.lower().endswith(".wav"):
                file_path = os.path.join(root, f)
                # 对应音素标注文件 .PHN
                phn_path = file_path[:-4] + ".PHN"
                speaker_id = os.path.basename(root)
                rows.append({
                    "wav_path": file_path,
                    "phn_path": phn_path,
                })
def load_audio(wav_path):
    waveform, sr = torchaudio.load(wav_path)  # [1, T]
    return waveform
def load_phonemes(phn_path):
    phonemes = []
    with open(phn_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                start, end, phn = parts
                phonemes.append(phn)
    if phonemes and phonemes[0] == "h#":
        phonemes = phonemes[1:]
    if phonemes and phonemes[-1] == "h#":
        phonemes = phonemes[:-1]
    return phonemes

df = pd.DataFrame(rows)
audio_list = []
phoneme_list = []
for idx, row in df.iterrows():
    wav = load_audio(row['wav_path'])
    phns = load_phonemes(row['phn_path'])
    audio_list.append(wav)       # [1, T_i]
    phoneme_list.append(phns)    # list of phonemes



print('--------数据读取------------')
print("音频条数:", len(audio_list))
print("音素条数:", len(phoneme_list))
print("示例音频形状:", audio_list[0].shape)



vocab = vocab_model.build_vocab(phoneme_list) #初始化词表
'''
1. 帧分割+ 加窗
长度：10ms
重叠：5ms
frame_step = (帧长- 重叠)*采样率

2. 滤波器 + MFCC
3. 归一化 + 加噪音
'''
frame_length = int(0.01 * 16000) # 160
frame_step = int(0.005 * 16000)  #帧移
std_noise = 0

mfcc_transform = T.MFCC(
    sample_rate=16000,
    n_mfcc=26,
    melkwargs={
        'n_fft': 512,  
        #512/16000s = 32ms，每隔hop_length采样32ms做一次FFT(时域->频域)
        #输出的频谱图维度 = [1 + n_fft/2, num_frames]：
        #num_frames: 音频长度(采样点数)/hop_length
        'n_mels': 40, 
        # 用40个Mel滤波器组模仿人耳听觉特性，输出40维的梅尔频谱  
        # (20-40语音识别，80-128音乐识别)
         # 输出的频谱图维度 = [n_mels,num_frames]：
        'hop_length': frame_step,  #帧移
        'win_length': frame_length #帧长
    }
)

for i in range(len(audio_list)):
    audio_list[i] = mfcc_transform(audio_list[i]) #计算MFCC特征
    mean = audio_list[i].mean(dim = 1,keepdim = True)
    std = audio_list[i].std(dim = 1, keepdim = True)
    mfcc_norm = (audio_list[i] - mean) / (std + 1e-6) #归一化
    noise = torch.randn_like(mfcc_norm) * std_noise #高斯噪音
    audio_list[i] = mfcc_norm + noise
print("mfcc示例音频形状:", audio_list[0].shape)


x_list_padded = [x.squeeze(0).transpose(0,1) for x in audio_list]  
input_lengths = torch.tensor([f.size(0) for f in x_list_padded], dtype=torch.long)#每条序列真实长度
x_padded = pad_sequence(x_list_padded, batch_first=True)  # shape -> [batch_size, max_T, 26]

label_lengths = torch.tensor([len(l) for l in phoneme_list], dtype=torch.long)#每条标签序列真实长度

y_labels = []
for seq in phoneme_list:
    idx_seq = [vocab[p] for p in seq]  # 转成整数
    y_labels.append(torch.tensor(idx_seq, dtype=torch.long))  # 转成tensor
print(y_labels[0])

# pad_sequence 自动将不同长度的序列 pad 到 max_len
y_labels = pad_sequence(y_labels, batch_first=True, padding_value=0)


print('--------数据封装前------------')
print("音频X.shape:",x_padded.shape)
print("标签y.shape:",y_labels.shape)
print("音频X有效长度:",input_lengths.shape)
print("标签y有效长度:",label_lengths.shape)

#封装
# 音频X.shape: torch.Size([4620, 1558, 26])
# 标签y.shape: torch.Size([4620])
# 音频X有效长度: torch.Size([4620, 73])
# 标签y有效长度: torch.Size([4620])
data_dict = {
    'x_padded': x_padded,           
    'input_lengths': input_lengths, 
    'y_labels': y_labels,           
    'label_lengths': label_lengths  
}


print('--------数据封装后------------')
# 音频X.shape: torch.Size([4620, 1558, 26])   [B,T,N]
# 标签y.shape: torch.Size([4620])
# 音频X有效长度: torch.Size([4620, 73])
# 标签y有效长度: torch.Size([4620])
print("音频X.shape:",data_dict[ 'x_padded'].shape)
print("标签y.shape:",data_dict['input_lengths'].shape)
print("音频X有效长度:",data_dict['y_labels'].shape)
print("标签y有效长度:",data_dict['label_lengths'].shape)
torch.save(data_dict, 'ctc_dataset.pt')
torch.save(vocab,'vocab.pt')