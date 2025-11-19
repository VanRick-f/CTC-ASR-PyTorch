import torch.nn as nn



#网络模型
class Model(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, num_classes):
        '''
         input_size: 输入特征维度 (26)
         hidden_size: LSTM 隐藏状态维度 (这里选择100) 最后输出是[batch,T,100]
         num_layers: LSTM 层数
         num_classes 输出类别数 (初始化时注意+1, CTC空白标签)
        '''
        super().__init__()
        self.rnn = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,     #输入数据格式：(batch, seq_len, feature_dim)
            bidirectional = True,        #双向LSTM
        )
        self.fc = nn.Linear(hidden_size * 2 , num_classes) #双向LSTM
        
    def forward(self,x):
        self.rnn.flatten_parameters()
        run_out , _ = self.rnn(x)
        '''
        run_out_每一帧的ht拼起来：
            [batch, seq_len, hidden_size*directions]
        '''
        out = self.fc(run_out)
        return out

