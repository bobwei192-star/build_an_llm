# 将词汇表作为类属性存储，以便在 encode 和 decode 方法中访问
import re

class SimpleTokenizer:
    # 初始化分词器，接收词汇表字典（字符串到ID的映射）
    def __init__(self, vocab: dict[str, int]):
        # 字符串到整数的映射（词→ID）
        self.str_to_int = vocab
        # 整数到字符串的反向映射（ID→词）
        self.int_to_str = {i: s for s, i in vocab.items()}

    # 编码方法：将文本转换为token ID列表
    def encode(self, text: str) -> list[int]:
        # 使用正则表达式分割文本，保留分隔符（标点和空白）
        # 分隔符包括：逗号、句号、问号、下划线、感叹号、引号、括号、单引号、破折号、空白字符
        preprocessed: list[str] = re.split(r'([,.?_!"()\']|--|\s)', text)
        # 过滤掉空白字符，并去除每个元素的首尾空白
        preprocessed: list[str] = [
            item.strip() for item in preprocessed if item.strip()
        ]
        # 将每个词转换为对应的ID
        ids: list[int] = [self.str_to_int[s] for s in preprocessed]
        return ids 
    
    # 解码方法：将token ID列表转换回文本
    def decode(self, ids: list[int]) -> str:
        # 将每个ID转换为对应的词，并用空格连接
        # 修复：移除多余的方括号，避免生成嵌套列表
        text = " ".join(self.int_to_str[i] for i in ids)
        # 移除标点符号前的空格（如 "hello ," → "hello,"）
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


# 生成词汇表（从 the-verdict.txt 文件）
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text: str = f.read()

# 预处理：分割文本并过滤空白
preprocessed: list[str] = re.split(r'([,.?:;?_!"()\']|--|\s)', raw_text)
preprocessed: list[str] = [item for item in preprocessed if item.strip()]

# 生成词汇表：排序后的唯一词列表
all_words: list[str] = sorted(set(preprocessed))
vocab: dict[str, int] = {token: integer for integer, token in enumerate(all_words)}

# 测试分词器
tokenizer = SimpleTokenizer(vocab)
text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print("Encoded IDs:", ids)

# 测试解码
decoded_text = tokenizer.decode(ids)
print("Decoded text:", decoded_text)


# 将这个分词器应用于训练集之外的新样本
text="Hello, do you like tea?"
# print(tokenizer.encode(text)) #KeyError: 'Hello'




