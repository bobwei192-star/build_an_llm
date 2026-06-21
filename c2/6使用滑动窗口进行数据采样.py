
# 使用 exec() 执行 5基于BPE概念的分词方案.py，导入 tokenizer 变量
import os

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
tokenizer_script = os.path.join(current_dir, "5基于BPE概念的分词方案.py")

# 执行脚本，将 tokenizer 导入到当前命名空间
with open(tokenizer_script, "r", encoding="utf-8") as f:
    code = f.read()
    exec(code)

# 现在可以直接使用 tokenizer 变量
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text: str = f.read()

enc_text: list[int] = tokenizer.encode(raw_text)

print("编码后的 token 数量:", len(enc_text))





enc_sample=enc_text[50:]
print(enc_sample[0:10])
context_size=4 #上下文大小决定了输入中包含多少个词元
x=enc_sample[:context_size]
y=enc_sample[1:context_size+1]
print("x:",x)
print("y:",y)

for i in range(1, context_size+1):
    context=enc_sample[:i]
    desired=enc_sample[i]
    print(context, "----->", desired)

# [290] -----> 4920
# [290, 4920] -----> 2241
# [290, 4920, 2241] -----> 287
# [290, 4920, 2241, 287] -----> 257
# 箭头左侧表示大模型接受的输入，右侧表示大模型应该预测的目标词元id

# 更直观的表示
for i in range(1, context_size+1):
    context=enc_sample[:i]
    desired=enc_sample[i]
    # tokenizer.decode() 需要列表参数，所以将单个 ID 包装在列表中
    print(tokenizer.decode(context), "----->", tokenizer.decode([desired]))

# 这样可以清楚地看到：
# - 左侧是模型看到的输入文本（上下文）
# - 右侧是模型应该预测的下一个词元就创建好了