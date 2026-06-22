
# tiktoken 0.7.0 API 使用 tiktoken.get_encoding()
# 安装命令: pip install tiktoken==0.7.0
import os
import urllib.request

# 禁用所有代理（解决 SSL 连接问题）
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('all_proxy', None)
os.environ.pop('ALL_PROXY', None)

# 确保缓存目录存在
CACHE_DIR = os.path.expanduser("~/.cache/tiktoken")
os.makedirs(CACHE_DIR, exist_ok=True)

VOCAB_FILE = os.path.join(CACHE_DIR, "vocab.bpe")

# 检查 vocab.bpe 文件是否存在，不存在则自动下载
if not os.path.exists(VOCAB_FILE):
    print("正在下载 vocab.bpe 文件...")
    VOCAB_URL = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe"
    try:
        urllib.request.urlretrieve(VOCAB_URL, VOCAB_FILE)
        print(f"vocab.bpe 已下载到: {VOCAB_FILE}")
    except Exception as e:
        print(f"下载失败: {e}")
        print("请手动下载: wget --no-proxy " + VOCAB_URL)
        raise
else:
    print(f"vocab.bpe 文件已存在: {VOCAB_FILE}")

# 导入 tiktoken 并测试
from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))

# 创建分词器
tokenizer = tiktoken.get_encoding("gpt2")

# 测试文本
text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the someunknownPalace."

# 编码文本
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print("编码结果:", integers)

strigs=tokenizer.decode(integers)
print("解码结果:", strigs)
