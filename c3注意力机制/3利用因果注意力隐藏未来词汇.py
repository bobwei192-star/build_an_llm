
import torch

# 由于 Python 不支持以数字开头的模块名，使用 exec() 执行文件导入类
with open('2实现带可训练权重的自注意力机制.py', 'r', encoding='utf-8') as f:
    code = f.read()
    exec(code)

# 原原理和过程请看书第 67页
print("c3注意力机制/3利用因果注意力隐藏未来词汇.py")



#**************************** 3个步骤实现因果注意力掩码 **************************************

print("****************** 3个步骤实现因果注意力掩码 ******************")


################################## 步骤1 ##########################
# 输入数据：6个词元的嵌入向量（3维）
inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # Your   (索引0)
    [0.55, 0.87, 0.66],  # journey (索引1)
    [0.57, 0.85, 0.64],  # starts (索引2)
    [0.22, 0.58, 0.33],  # with   (索引3)
    [0.77, 0.25, 0.10],  # one    (索引4)
    [0.05, 0.80, 0.55]   # step   (索引5)
])

# 设置相同的随机种子，确保与 2实现带可训练权重的自注意力机制.py 中一致
# nn.Linear 使用 Kaiming 初始化，需要相同种子才能得到相同权重
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)


queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)

attn_scores=queries @ keys.T
attn_weights=torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim =-1)
print("attn_weights: ")
print(attn_weights)


################################### 步骤2 ##########################
context_length=attn_scores.shape[0]
mask_simple=torch.tril(torch.ones(context_length, context_length))
print("mask_simple: ")
print(mask_simple)

#现在可以把这个掩码矩阵和注意力权重矩阵相乘 使对角线上方的值变为0
masked_simple=attn_weights * mask_simple
print("masked_simple")
print(masked_simple)

################################### 步骤3 ##########################
# 重新归一化注意力权重  使得每行的总和再次为1 通过将每行中的每个元素除以每行中的和实现这一点
row_sums=masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print("masked_simple_norm")
print(masked_simple_norm)




#**************************** 2个步骤实现因果注意力掩码 **************************************
#更高效的掩码的方法

print("****************** 2个步骤实现因果注意力掩码 ******************")
mask=torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked=attn_scores.masked_fill(mask.bool(), -torch.inf)
print("masked:")
print(masked)

#现在只要对掩码结果应用softmax函数 就可以完成整个过程
attn_weights=torch.softmax(masked / keys.shape[-1] ** 0.5 ,dim=1)
print("attn_weights")
print(attn_weights)



#**************************** 利用drop out 掩码额外的注意力权重 **************************************
#更高效的掩码的方法

print("****************** 利用drop out 掩码额外的注意力权重 ******************")

torch.manual_seed(123)
dropout=torch.nn.Dropout(0.5) #选择50%的dropout率
example=torch.ones(6,6) #创建一个6x6的矩阵 全为1
print("大概有一半的值会被dropout 因为dropout率是0.5")
print(dropout(example))

# 现在对注意力权重矩阵进行dropout 操作
torch.manual_seed(123)
print("现在对注意力权重矩阵进行dropout 操作")
print(dropout(attn_weights))
现在对注意力权重矩阵进行dropout 操作
# tensor([[2.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.8966, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.0000, 0.6206, 0.0000, 0.0000, 0.0000],
#         [0.5517, 0.4921, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.4350, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.3327, 0.0000, 0.0000, 0.0000, 0.0000]],
#        grad_fn=<MulBackward0>)
# worker@master:~/build_an_llm/c3注意力机制$ 
#这里和原书 不一样 



#**************************** 实现一个简化的因果注意力类 **********************************
# 在开始之前 确保代码可以处理包含多个样本的批次 以便CausalAttention类可支持第2章中实现的数据加载器产生的
# 批量输出
# 为简单起见可以通过复制输入文本示例来模拟批量输入
batch = torch.stack((inputs, inputs), dim = 0) #2个输入每个输入有6个词元 每个词元的嵌入维度是3
print("batch.shape:", batch.shape)
#将生成1个三维张量 其中包含2个输入文本，每个文本6个词元，每个词元是1个3维的嵌入向量

