import torch

# 输入数据：6个词元的嵌入向量（3维）
inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # Your   (索引0)
    [0.55, 0.87, 0.66],  # journey (索引1)
    [0.57, 0.85, 0.64],  # starts (索引2)
    [0.22, 0.58, 0.33],  # with   (索引3)
    [0.77, 0.25, 0.10],  # one    (索引4)
    [0.05, 0.80, 0.55]   # step   (索引5)
])

# 提取查询向量：选择 "journey" 作为查询词元
query = inputs[1]
print("查询向量 (journey):")
print(query)
# 输出: tensor([0.5500, 0.8700, 0.6600])

# 创建空的注意力分数数组（未初始化，节省性能）
attn_scores_2 = torch.empty(inputs.shape[0])
print("\n初始化前的注意力分数（未初始化值）:")
print(attn_scores_2)

# 计算注意力分数：遍历每个词元，计算与查询词元的点积
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

print("\n计算后的注意力分数:")
print(attn_scores_2)
# 输出: tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])


# ================================================
# 自注意力机制原理说明
# ================================================
# 核心概念：使用向量点积计算词元之间的相似度

# 点积公式（以 Your vs journey 为例）:
# attn_scores_2[0] = 0.43*0.55 + 0.15*0.87 + 0.89*0.66
#                  = 0.2365 + 0.1305 + 0.5874
#                  = 0.9544

# 注意力分数排序：
# 1. journey → 1.4950（与自己最相似）
# 2. starts  → 1.4754（动作与旅程相关）  
# 3. step    → 1.0865（步骤与旅程相关）
# 4. Your    → 0.9544（一般相关）
# 5. with    → 0.8434（一般相关）
# 6. one     → 0.7070（不太相关）

# 自注意力机制核心原理：
# 让模型学会根据当前词元（查询），动态关注输入序列中最相关的词元

#归一化的主要目的是获得总和为1 的注意力权重
# 以下是实现归一化步骤的简单方法
attn_weights_2_tmp=attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())


#实际应用中使用softmax函数进行归一化更常见
def softmax(x):
    return torch.exp(x) / torch.exp(x).sum(dim = 0)

attn_weights_2_native=softmax(attn_scores_2)
print("Attention weights:", attn_weights_2_native)
print("Sum:", attn_weights_2_native.sum())

#实践中建议使用softmax的pytorch实现
attn_weights_2=torch.softmax(attn_scores_2, dim = 0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

#接下来进入最后一步 计算上下文向量
query=inputs[2]

context_vec_2=torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[i] * x_i

print("Context vector:", context_vec_2)


#计算所有输入词元的注意力权重
attn_scores=torch.empty(6,6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)

#for循环慢 用矩阵乘法计算
attn_scores = inputs @ inputs.T
print(attn_scores)

# 对每一行进行归一化 确保每行的值总和为1
attn_weights = torch.softmax(attn_scores, dim =1)
print("attn_weights:", attn_weights)

# 用上述注意力权重 通过矩阵乘法计算所有上下文向量
# attn_weights= 
# attn_weights=torch.tensor([
#         [0.2098, 0.2006, 0.1981, 0.1242, 0.1220, 0.1452],
#         [0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581],
#         [0.1390, 0.2369, 0.2326, 0.1242, 0.1108, 0.1565],
#         [0.1435, 0.2074, 0.2046, 0.1462, 0.1263, 0.1720],
#         [0.1526, 0.1958, 0.1975, 0.1367, 0.1879, 0.1295],
#         [0.1385, 0.2184, 0.2128, 0.1420, 0.0988, 0.1896]])
# inputs = torch.tensor([
#     [0.43, 0.15, 0.89],  # Your   (索引0)
#     [0.55, 0.87, 0.66],  # journey (索引1)
#     [0.57, 0.85, 0.64],  # starts (索引2)
#     [0.22, 0.58, 0.33],  # with   (索引3)
#     [0.77, 0.25, 0.10],  # one    (索引4)
#     [0.05, 0.80, 0.55]   # step   (索引5)
# ])

all_context_vecs = attn_weights @ inputs
print("\n所有上下文向量（每行对应一个词元）：")
print(all_context_vecs)

# ========================================
# 输出结果解读
# ========================================
# 输出是一个 6×3 的矩阵，每一行对应一个词元的上下文向量：
#
# tensor([[0.4421, 0.5931, 0.5790],  ← Your   的上下文向量（第0行）
#         [0.4419, 0.6515, 0.5683],  ← journey 的上下文向量（第1行）
#         [0.4431, 0.6496, 0.5671],  ← starts  的上下文向量（第2行）
#         [0.4304, 0.6298, 0.5510],  ← with    的上下文向量（第3行）
#         [0.4671, 0.5910, 0.5266],  ← one     的上下文向量（第4行）
#         [0.4177, 0.6503, 0.5645]]) ← step    的上下文向量（第5行）
#
# ========================================
# 上下文向量含义说明
# ========================================
# 每个词元的上下文向量是所有词元嵌入向量的加权平均
# 权重来自注意力权重矩阵的对应行
#
# 例如：Your 的上下文向量计算过程：
# context_vec[Your] = attn_weights[0] @ inputs
#                  = [0.2098, 0.2006, 0.1981, 0.1242, 0.1220, 0.1452] @ [[0.43,0.15,0.89], ...]
#                  = 0.2098*[0.43,0.15,0.89] + 0.2006*[0.55,0.87,0.66] + ...
#                  = [0.4421, 0.5931, 0.5790]
#
# 注意力权重越大，该词元对上下文向量的贡献越大


