# 解决方案：使用 exec() 动态执行以数字开头的文件
# Python 不支持直接导入以数字开头的模块

# 读取并执行 '1分割工具.py' 文件
with open('1分割工具.py', 'r', encoding='utf-8') as f:
    code = f.read()
    exec(code)

# 从执行后的命名空间中获取 preprocessed 变量
# 注意：需要确保 '1分割工具.py' 文件中定义了 preprocessed 变量

# 生成词汇表：排序后的唯一词列表
all_tokens = sorted(list(set(preprocessed)))

# 添加特殊标记
# <|endoftext|> - 文本结束标记，用于分隔多个文本片段
# <|unk|> - 未知词标记，用于表示词汇表中不存在的单词
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

# 创建词汇表字典（单词→ID）
vocab = {token: integer for integer, token in enumerate(all_tokens)}

# 打印词汇表大小
print("词汇表大小:", len(vocab))

# 打印词汇表最后5个
print("\n词汇表最后5个:")
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)