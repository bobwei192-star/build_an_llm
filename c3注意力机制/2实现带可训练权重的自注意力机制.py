import torch


#逐步计算注意力权重
# 输入数据：6个词元的嵌入向量（3维）
inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # Your   (索引0)
    [0.55, 0.87, 0.66],  # journey (索引1)
    [0.57, 0.85, 0.64],  # starts (索引2)
    [0.22, 0.58, 0.33],  # with   (索引3)
    [0.77, 0.25, 0.10],  # one    (索引4)
    [0.05, 0.80, 0.55]   # step   (索引5)
])


x_2= inputs[1]

d_in= inputs.shape[1] #输入嵌入维度 d_in = 3
d_out = 2 #输出嵌入维度 d_out =2 

# 初始化 3个权重矩阵  Wq  Wk  Wv
torch.manual_seed(123)
W_query= torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key= torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value= torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

print("W_query:", W_query)
print("W_key:", W_key)
print("W_value:", W_value)

#接下来 计算 查询向量 键向量 值向量
# (1 * 2)=(1 * 3) @ (3 * 2)
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2=x_2 @ W_value

print("query_2:", query_2)



#虽然目标只是计算1个上下文向量z2 但仍然需要所有输入元素的键向量和值向量 因为他们参与了
#计算相对于查询q2 的注意力权重

# (6 * 3)=(6 * 3) @ (3 * 2 )
keys=inputs @ W_key
values=inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)


#首先计算注意力分数 ω22
keys_2=keys[1]

print("query_2:", query_2)
print("keys_2:", keys_2)
attn_scores_22= query_2.dot(keys_2)
print(attn_scores_22)
# query_2: tensor([0.4306, 1.4551])
# keys_2: tensor([0.4433, 1.1419])
# tensor(1.8524)

# 同样通过矩阵乘法计算所有注意力分数
attn_scores_2= query_2 @ keys.T
print(attn_scores_2)
# tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440])

# 现在将注意力分数转换为注意力权重 我们通过缩放注意力分数并应用softmax函数来计算注意力权重
# 不过 此时是通过将注意力分数除以键向量的嵌入维度的平方根来进行缩放

# 负索引：-1 表示取张量的最后一个维度
# keys.shape = (6, 2)，其中：
#   第一个维度（索引0）= 6，代表词元数量
#   第二个维度（索引1 或 -1）= 2，代表键向量的特征维度
# 使用 -1 的好处：即使张量维度变化（如添加批次维度），也能正确获取最后一维
d_k = keys.shape[-1]  # 键向量维度 d_k = 2
# 计算注意力权重：先缩放再应用 softmax
# attn_scores_2 / d_k ** 0.5：将注意力分数除以键向量维度的平方根（缩放操作）
# dim=-1：在最后一个维度上应用 softmax，确保每个查询的注意力权重总和为1
# 例如：attn_scores_2 = [1.27, 1.85, 1.81, 1.08, 0.56, 1.54]
#       经过 softmax(dim=-1) 后 → [0.12, 0.22, 0.21, 0.10, 0.07, 0.16]，总和=1
attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim=-1)
print("attn_weights_2:", attn_weights_2)

# 计算上下文向量 通过对值向量进行加权求和来计算上下文向量 在这里 注意力权重作为加权因子 用于权衡每个值向量
#的重要性 
context_vec_2 = attn_weights_2 @ values
print("context_vec_2:", context_vec_2)


# ================================================
# 自注意力机制类实现
# ================================================
# PyTorch nn.Module 类继承语法说明：
# 1. class ClassName(nn.Module):  - 继承自 nn.Module
# 2. super().__init__():          - 必须调用父类构造函数，初始化模块状态
# 3. nn.Parameter():              - 将张量标记为可训练参数，会被优化器更新
# 4. forward(self, x):            - 定义前向传播逻辑，调用 model(x) 时自动执行

import torch.nn as nn

# ------------------------------------------------
# 版本1：手动创建权重矩阵
# ------------------------------------------------
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()  # 调用父类构造函数
        # 使用 nn.Parameter 创建可训练参数（等价于手动权重矩阵）
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))  # 查询权重矩阵 (d_in × d_out)
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))    # 键权重矩阵 (d_in × d_out)
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))  # 值权重矩阵 (d_in × d_out)
    
    def forward(self, x):
        # 手动矩阵乘法计算 Q/K/V
        # x: (seq_len × d_in) @ W_key: (d_in × d_out) = keys: (seq_len × d_out)
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        # 计算注意力分数：Q @ K^T (seq_len × seq_len)
        attn_scores = queries @ keys.T
        
        # 缩放 + softmax：确保权重总和为1
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        
        # 计算上下文向量：注意力权重 @ 值向量
        context_vec = attn_weights @ values
        return context_vec

# 使用示例
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print("SelfAttention_v1 输出:", sa_v1(inputs))

# ------------------------------------------------
# 版本2：使用 PyTorch 线性层（更推荐）
# ------------------------------------------------
# nn.Linear(d_in, d_out, bias=bool) 说明：
# - 这是 PyTorch 内置的线性变换层
# - 内部包含：权重矩阵 W (d_out × d_in) + 偏置向量 b (d_out,)（如果 bias=True）
# - 前向传播：output = input @ W^T + b（注意：权重矩阵形状与手动方式相反）
# - 为什么可以用 self.W_key(x) 调用？
#   → nn.Module 的子类都实现了 __call__ 方法，调用时自动执行 forward()
#   → self.W_key(x) 等价于 self.W_key.forward(x)

qkv_bias = False  # 是否使用偏置向量，GPT等模型通常设置为 False

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        # 使用 nn.Linear 替代手动权重矩阵
        # nn.Linear(d_in, d_out) 内部自动创建可训练的权重和偏置
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        # 使用线性层计算 Q/K/V
        # self.W_key(x) 调用 nn.Linear 的 __call__ 方法，内部执行：
        # keys = x @ self.W_key.weight.T + self.W_key.bias（如果 bias=True）
        # 等价于手动方式的：x @ W_key（当 bias=False 时）
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # 以下逻辑与版本1完全相同
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec

# 使用示例
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print("SelfAttention_v2 输出:", sa_v2(inputs))

# ------------------------------------------------
# 版本对比总结
# ------------------------------------------------
# SelfAttention_v1: 手动矩阵乘法，只有权重，无偏置
# SelfAttention_v2: 使用 nn.Linear，可选择是否添加偏置
# 两种方式的核心逻辑相同，但 nn.Linear 更规范，支持偏置，且与 PyTorch 生态更兼容



    










