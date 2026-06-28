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










