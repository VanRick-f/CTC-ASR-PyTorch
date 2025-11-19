import collections # 提供一些更强大的容器类型和操作
import numpy as np

"""统计词元的频率"""
def count_corpus(vocab_list):
    return collections.Counter(vocab_list)


'''
Vocab类：
    注意：Vocab写好之后全局使用，包括训练，验证，测试，不能进行修改
'''
class Vocab():
    def __init__(self,vocab_list):
        counter = count_corpus(vocab_list)
        
        #按频次排序，传入counter.items(),排序规则key=lambda x: x[1],reverse降序
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],reverse=True)
        
        #列表，根据idx_to_token[idx]直接获取词元，这里给空白标签留了位置0
        self.idx_to_token = ['blank']  
        
        # 词元：索引
        #先初始化，把预留的元素先加进去，这里是空白标签'blank'
        self.token_to_idx = {
            token: idx for idx, token in enumerate(self.idx_to_token)
        }
        
        '''
        将次元映射到索引
        '''
        for token, _ in self._token_freqs:  #遍历按频次排序的列表{token:freq}
            if token not in self.token_to_idx:
                self.idx_to_token.append(token) #按顺序添加到列表，这样频次高的索引靠前
                self.token_to_idx[token] = len(self.idx_to_token) - 1 #token对应的索引就是当前长度-1

    def __len__(self):
        return len(self.idx_to_token)
    
    #可以直接通过索引获取词元，例vocab['pau','y'],vocab['pau']
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, -1) 
        return [self.__getitem__(token) for token in tokens]
    
    @property
    def token_freqs(self):
        return self._token_freqs

def build_vocab(phoneme_list):
    flat_phoneme_list = np.concatenate(phoneme_list).tolist()
    return Vocab(flat_phoneme_list)

