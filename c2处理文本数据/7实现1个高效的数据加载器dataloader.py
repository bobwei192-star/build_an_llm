

# dataloader 会遍历数据集， 并将输入和目标以pytorch张量的形式返回
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

#用于批处理输入和目标的数据集
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids=[]
        self.target_ids=[]
        token_ids=tokenizer.encode(txt) #对全部文本进行分词

        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk=token_ids[i:i+max_length]
            target_chunk=token_ids[i+1:i+max_length+1]
            self.input_ids.append(input_chunk)
            self.target_ids.append(target_chunk)

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        # 返回时转换为 tensor，确保类型一致且能被 DataLoader 正确批处理
        return torch.tensor(self.input_ids[idx], dtype=torch.long), torch.tensor(self.target_ids[idx], dtype=torch.long)


#用于批量生成输入-目标的数据加载器
def create_dataloader_v1(
                    txt, 
                    batch_size=4, 
                    max_length=256, 
                    stride=128,
                    shuffle=True,
                    drop_last=True,
                    num_workers=0
                    ): 
    tokenizer=tiktoken.get_encoding("gpt2")

    dataset=GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader=DataLoader(
                        dataset, 
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        drop_last=drop_last, 
                        num_workers=num_workers)#用于预处理的cpu进程数
    return dataloader
#如果drop_last=True， 且批次大小小于指定的batch_size， 
# �则会删除最后一个批次，以防止在训练期间出现损失剧增


# 用批次大小为1 的DataLoader 对上下文长度为4的大预言模型进行测试，来直观的理解GPTdatasetV1 和
# create_dataloader_v1函数是如何协同工作的
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_txt=f.read()
    dataloader=create_dataloader_v1(raw_txt, batch_size=1, max_length=4, stride=1,shuffle=False)
    data_iter=iter(dataloader)
    first_batch=next(data_iter)
    
    # 分离输入和目标
    input_batch, target_batch = first_batch
    
    print("批次结构:")
    print(f"输入张量形状: {input_batch.shape}")  # 应该是 [batch_size, seq_len] = [1, 4]
    print(f"目标张量形状: {target_batch.shape}")
    print(f"\n输入批次:\n{input_batch}")
    print(f"\n目标批次:\n{target_batch}")
    print("first_batch:")
    print(first_batch)
    
    second_batch=next(data_iter)
    
    print("second_batch:")
    print(second_batch)




# 如何以大于1的批次小小使用数据加载器进行采样：
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_txt=f.read()
    dataloader=create_dataloader_v1(raw_txt, batch_size=8, max_length=4, stride=4,shuffle=False)
    data_iter=iter(dataloader)
    inputs,targets=next(data_iter)
    print("Inputs:\n", inputs)
    print("Targets:\n", targets)
    