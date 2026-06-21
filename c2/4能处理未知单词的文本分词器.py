# ============================================================================
# SimpleTokenizerV2 - 支持未知单词处理的文本分词器
# ============================================================================
# Use Case (使用场景):
# 1. 文本预处理：将自然语言文本转换为模型可理解的 token ID 序列
# 2. 未知词处理：遇到词汇表外的单词时，用 <|unk|> 标记替换
# 3. 文本重建：将 token ID 序列还原为可读文本，保持标点符号正确格式
# 4. 适用于小型语言模型、文本分类、文本生成等任务

import re  # 正则表达式模块，用于文本分割


class SimpleTokenizerV2:
    """
    改进版简单分词器，支持处理词汇表外的未知单词
    
    核心特性：
    - 使用正则表达式进行文本分割
    - 自动处理未知单词（替换为 <|unk|>）
    - 支持编码（文本→ID）和解码（ID→文本）
    """
    
    # 初始化方法 - 创建分词器实例
    def __init__(self, vocab: dict[str, int]):
        """
        构造函数
        
        参数:
            vocab: dict[str, int] - 词汇表字典，键为单词，值为对应的整数ID
        
        类属性:
            str_to_int: 单词到ID的映射
            int_to_str: ID到单词的反向映射（通过字典推导式生成）
        """
        self.str_to_int = vocab  # 字符串→整数 映射表
        # 使用字典推导式创建反向映射：遍历 vocab 的键值对，交换键和值
        self.int_to_str = {i: s for s, i in vocab.items()}
    
    # 编码方法 - 文本转 token ID 列表
    def encode(self, text: str) -> list[int]:
        """
        将文本编码为 token ID 列表
        
        参数:
            text: str - 输入的原始文本
        
        返回:
            list[int] - 对应的 token ID 列表
        
        处理流程:
            1. 使用正则分割文本，保留分隔符
            2. 过滤空白字符
            3. 处理未知单词（替换为 <|unk|>）
            4. 转换为 token ID
        """
        # 正则表达式分割：保留标点符号和空白作为独立 token
        # 分隔符包括：逗号、句号、冒号、分号、问号、下划线、感叹号、双引号、括号、单引号、破折号、空白
        preprocessed: list[str] = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        
        # 列表推导式：过滤掉空白字符，并去除首尾空白
        # 条件 item.strip() 确保非空字符串才保留
        preprocessed: list[str] = [
            item.strip() for item in preprocessed if item.strip()
        ]
        
        # 关键改进：处理未知单词
        # 如果单词在词汇表中则保留，否则替换为 <|unk|>（未知标记）
        # 三元表达式：condition ? value_if_true : value_if_false
        preprocessed: list[str] = [
            item if item in self.str_to_int else "<|unk|>" 
            for item in preprocessed
        ]

        # 将处理后的单词转换为对应的 ID
        ids: list[int] = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    # 解码方法 - token ID 列表转文本
    def decode(self, ids: list[int]) -> str:
        """
        将 token ID 列表解码为文本
        
        参数:
            ids: list[int] - token ID 列表
        
        返回:
            str - 还原后的文本
        
        处理流程:
            1. 将每个 ID 转换为对应的单词
            2. 用空格连接所有单词
            3. 移除标点前的多余空格，恢复正确格式
        """
        # 使用生成器表达式将 ID 转换为单词，并用空格连接
        # 生成器表达式比列表推导式更节省内存
        text: str = " ".join(self.int_to_str[i] for i in ids)
        
        # 正则替换：移除标点符号前的多余空格
        # r'\s+([,.?!"()\'])' 匹配一个或多个空格后跟标点
        # r'\1' 引用捕获组，只保留标点
        # 示例："hello ," → "hello,"
        text: str = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        
        return text










# 测试SimpleTokenizerV2





# 读取并执行 '3将特殊词元添加到词汇表中.py' 文件
with open('3将特殊词元添加到词汇表中.py', 'r', encoding='utf-8') as f:
    code = f.read()
    exec(code)


text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

# 使用 <|endoftext|> 特殊标记连接两个文本
# 
# 详细解释：
# 1. (text1, text2) - 双括号创建元组（tuple），包含两个字符串元素
#    元组是不可变的序列，类似列表，但用圆括号包裹
#    作用：告诉 join() 方法要连接哪些元素
#
# 2. " <|endoftext|> " - 分隔符字符串，两端有空格
#    <|endoftext|> 是特殊标记，表示文本片段的结束
#
# 3. .join() - 字符串方法
#    语法：separator.join(iterable)
#    作用：将可迭代对象中的元素用分隔符连接成新字符串
#
# 示例：
#   text1 = "Hello"
#   text2 = "World"
#   text = "||".join((text1, text2))
#   结果: "Hello||World"
#
# 本例中：
#   输入: text1="Hello, do you like tea?", text2="In the sunlit terraces of the palace."
#   输出: "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))

print(text)



tokenizer=SimpleTokenizerV2(vocab)
ids=tokenizer.encode(text)
print(ids)
text=tokenizer.decode(ids)
print(text)

