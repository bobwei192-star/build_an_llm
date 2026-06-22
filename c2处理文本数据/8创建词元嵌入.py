import torch

#通过实际例子演示词元id转换为嵌入向量的工作原理
input_ids=torch.tensor([2,3,5,1])
vocab_size=6
output_dim=3

#  在 pytorch中实例化一个嵌入层，用于将词元id转换为嵌入向量
torch.manual_seed(123)
embedding_layer=torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)

# Parameter containing: 6行3列
# tensor([[ 0.3374, -0.1778, -0.1690],
#         [ 0.9178,  1.5810,  1.3010],
#         [ 1.2753, -0.2010, -0.1606],
#         [-0.4015,  0.9666, -1.1481],
#         [-1.1589,  0.3255, -0.6315],
#         [-2.8400, -0.7849, -1.4096]], requires_grad=True)


#将其应用到1个词元id上
print("=============================")
print(embedding_layer(torch.tensor([3])))
# tensor([[-0.4015,  0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>)


# 接下来 将其应用到多个词元id上
print("=============================")
print(embedding_layer(input_ids))