
import torch


vocab_size=50257
output_dim=256
token_embedding_layer=torch.nn.Embedding(vacab_size, output_dim)

#使用上述的token_embedding_layer 当我们从数据加载器中采样数据时，每个批次中的每
# 个词元都将被嵌入为256维向量，如果设定批次大小为8， 且每个批次包含4个词元，则结果
# 将是一个 8 * 4 * 256 的张量

#首先实例化 数据加载器

max_length=4
dataloader=create_dataloader_v1(
    raw_txt,
    batch_size=8,
    max_length=max_length,
    stride=max_length,
    shuffle=False
)

data_iter=iter(dataloader)
inputs, targets=next(data_iter)

print("Token IDs: \n", inputs)
print("\nInputs shape:\n", inputs.shape)

# 现在 使用嵌入层将这些词元ID 嵌入256维的向量中
token_embeddings=token_embedding_layer(inputs)
print(token_embeddings.shape)



# 为了获取GPT模型所采用的绝对位置嵌入，只需创建一个维度与token_embeddings_layer相同的嵌入层即可，
context_length = max_length
pos_embedding_layer=torch.nn.Embedding(context_length, output_dim)
pos_embeddings=pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

input_embeddings=token_embeddings+pos_embeddings
print(input_embeddings.shape)
# 并将位置信息编码到该张量中即可








