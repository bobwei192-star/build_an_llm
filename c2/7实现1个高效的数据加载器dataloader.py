

# dataloader 会遍历数据集， 并将输入和目标以pytorch张量的形式返回
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids=[]
        self.target_ids=[]
        token_ids=tokenizer.encode(txt) #对全部文本进行分词
        

