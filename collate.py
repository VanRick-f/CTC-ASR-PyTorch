import torch
from torch.utils.data import  Dataset
from torch.nn.utils.rnn import pad_sequence

'''
因为y_label是拼接的一维张量，无法与xpadded对齐，无法直接load
现在定义一个类，以便加载数据时可以自定义batch_size
'''

def collate_ctc_batch(batch):
    """
    batch: list of tuples (x, y, x_len, y_len)
    """
    xs, ys, x_lens, y_lens = zip(*batch)  # 分别是tuple

    # X 已经 pad 好，直接 stack
    xs = torch.stack(xs, dim=0)           # [batch, max_T, feat_dim]
    x_lens = torch.tensor(x_lens, dtype=torch.long)
    
    # Y 已经 pad 好，但需要根据 y_lens 切有效部分再拼成一维 tensor
    ys_cat = torch.cat([y[:l] for y, l in zip(ys, y_lens)])
    y_lens = torch.tensor(y_lens, dtype=torch.long)

    return xs, ys_cat, x_lens, y_lens
class CTCDataset(Dataset):
    def __init__(self, data_dict):
        """
        x_padded: [num_samples, max_T, feat_dim] tensor
        y_padded: [num_samples, max_label_len] tensor
        x_lengths: [num_samples] tensor/list 每条音频有效帧数
        y_lengths: [num_samples] tensor/list 每条标签有效长度
        """
        self.x_padded = data_dict['x_padded']
        self.y_padded = data_dict['y_labels']
        self.x_lengths = data_dict['input_lengths']
        self.y_lengths = data_dict['label_lengths']

    def __len__(self):
        return len(self.x_padded)

    def __getitem__(self, idx):
        x = self.x_padded[idx]         # [max_T, feat_dim]
        y = self.y_padded[idx]         # [max_label_len]
        x_len = self.x_lengths[idx]    # 标量
        y_len = self.y_lengths[idx]    # 标量
        return x, y, x_len, y_len