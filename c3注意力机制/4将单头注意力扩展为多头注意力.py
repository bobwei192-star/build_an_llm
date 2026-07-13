
import torch
import torch.nn as nn
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, '3利用因果注意力隐藏未来词汇.py'), 'r', encoding='utf-8') as f:
    code = f.read()
    exec(code)

##################### 一个实现多头注意力的封装类###############################
##################### 一个实现多头注意力的封装类###############################

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                d_in, 
                d_out, 
                context_length,
                dropout,
                num_heads,
                qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
                d_in,
                d_out,
                context_length,
                dropout,
                qkv_bias
            )
            for _ in range(num_heads)
            ]
        )

    def forward(self,x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


#像之前使用CausalAttention 类一样使用MultiHeadAttention 类

torch.manual_seed(123)
context_length = batch.shape[1]
d_in, d_out = 3, 2
mha = MultiHeadAttention(
    d_in, d_out, context_length, 0.0, num_heads = 2
)
context_vecs= mha(batch)
print("==========================================")
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


# 结果中的context_vecs 是一个2 x 6 x 4 的张量
# 因为有2个输入文本
# /home/worker/build_an_llm/c3注意力机制/3利用因果注意力隐藏未来词汇.py
# 为简单起见可以通过复制输入文本示例来模拟批量输入
# batch = torch.stack((inputs, inputs), dim = 0) #2个输入每个输入有6个词元 
# 输入文本是重复的 所以这些上下文向量完全相同

################## 一个高效的多头注意力类 ##################################
################## 一个高效的多头注意力类 ##################################
prin("一个高效的多头注意力类")
class MultiHeadAttention(nn.Module):
        def __init__(self, 
                    d_in, 
                    d_out,
                    context_length,
                    dropout,
                    nums_heads,
                    qkv_bias=False):
        super().__init__()
        assert (d_out % nums_heads ==0 ), \
            "d_out must be divisible by nums_heads"
        self.d_out =d_out
        self.num_heads = nums_heads
        self.head_dim = d_out // nums_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        




