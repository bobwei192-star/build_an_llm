with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text: str = f.read()
print(raw_text)
print("Total number of characters:", len(raw_text))
print(raw_text[:99])

print("====================================")
import re
text: str = "Hello, world. This, is a test."
result: list[str] = re.split(r'(\s)', text)
print(result)

result: list[str] = re.split(r'([,.]|\s)', text)
print(result)


# .strip() 是字符串方法，用于 去除字符串首尾的空白字符 。
# - item.strip() 会去除每个元素的首尾空白
# - if item.strip() 判断去除空白后是否为非空字符串
# - 空字符串 "" 在布尔上下文中为 False ，非空字符串为 True
# - 因此这行代码 过滤掉纯空白字符的元素
result: list[str] = [item for item in result if item.strip()]
print(result)

text: str="Hello, world. Is this-- a test?"
result: list[str]= re.split('([,.:?_!"()\']|--|\s)', text)
result: list[str] = [item for item in result if item.strip()]
print(result)

# \' 是为了在单引号字符串中表示单引号字符本身
preprocessed: list[str] = re.split(r'([,.:;?_!"()\']|--|\s)',raw_text)
preprocessed: list[str]=  [item for item in preprocessed if item.strip()]
print(len(preprocessed))

print(preprocessed[:30])

print("**************************************************************")
# Python 的 sorted() 按 ASCII 码顺序 排序：
# 标点符号 → 数字 → 大写字母(A-Z) → 小写字母(a-z)

# - 标点符号的 ASCII 码 < 大写字母 < 小写字母
# - 字典序：先比较第一个字符，相同则比较第二个，以此类推
# - 所以 'A' 在 'Ah' 前面， 'Ah' 在 'Among' 前面
all_words: list[str] = sorted(set(preprocessed))
print(all_words)
vocab_size: int = len(all_words)
print(vocab_size)

print("-------------------------------------------------------------------")
vocab={token:integer for integer, token  in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break
print("============================================================")
print(vocab)

