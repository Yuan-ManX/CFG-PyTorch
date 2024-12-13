# take from https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py
# to give users a quick easy start to training DALL-E without doing BPE

import torch

import html
import os
import ftfy
import regex as re
from functools import lru_cache
from pathlib import Path


# OpenAI simple tokenizer

@lru_cache() # 使用缓存装饰器缓存函数结果
def default_bpe():
    """
    获取默认的 BPE 词汇表文件路径。

    返回:
        str: BPE 词汇表文件的完整路径。
    """
    # 返回默认的 BPE 词汇表文件路径
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/bpe_simple_vocab_16e6.txt")


@lru_cache() # 使用缓存装饰器缓存函数结果
def bytes_to_unicode():
    """
    生成字节到 Unicode 字符的映射字典。

    返回:
        dict: 字节到 Unicode 字符的映射字典。
    """
    # 生成 ASCII 可打印字符的字节列表
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    # 复制字节列表作为 Unicode 字符列表
    cs = bs[:]

    # 初始化计数器
    n = 0
    # 遍历所有 256 个可能的字节值
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b) # 将不在 bs 列表中的字节添加到 bs 列表中
            cs.append(2 ** 8 + n) # 为该字节分配一个新的 Unicode 字符代码点（从 256 开始）
            n += 1 # 计数器加一
    # 将 Unicode 代码点转换为对应的字符
    cs = [chr(n) for n in cs]

    # 返回字节到 Unicode 字符的映射字典
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    获取字符串中相邻字符对。

    参数:
        word (str): 输入字符串。

    返回:
        set: 相邻字符对的集合。
    """
    # 初始化一个空集合，用于存储字符对
    pairs = set()
    # 获取第一个字符作为前一个字符
    prev_char = word[0]

    # 遍历字符串中的每个字符（从第二个字符开始）
    for char in word[1:]:
        # 添加前一个字符和当前字符组成的元组到集合中
        pairs.add((prev_char, char))
        # 更新前一个字符为当前字符
        prev_char = char
    # 返回字符对集合
    return pairs


def basic_clean(text):
    """
    对文本进行基本清理，包括修复文本编码和解码 HTML 实体。

    参数:
        text (str): 输入文本。

    返回:
        str: 清理后的文本。
    """
    # 使用 ftfy 库修复文本编码问题
    text = ftfy.fix_text(text)
    # 解码 HTML 实体两次，确保所有实体都被正确解析
    text = html.unescape(html.unescape(text))
    # 去除文本首尾的空白字符并返回
    return text.strip()


def whitespace_clean(text):
    """
    清理文本中的空白字符。

    参数:
        text (str): 输入文本。

    返回:
        str: 清理后的文本。
    """
    # 使用正则表达式将所有连续的空白字符（包括空格、制表符、换行符等）替换为一个空格
    text = re.sub(r'\s+', ' ', text)
    # 去除文本首尾的空白字符
    text = text.strip()
    # 返回清理后的文本
    return text


class SimpleTokenizer(object):
    """
    SimpleTokenizer 类，用于对文本进行分词和编码。

    参数:
        bpe_path (str, 可选): BPE 词汇表文件的路径。默认值为 default_bpe() 返回的路径。
    """
    def __init__(self, bpe_path = default_bpe()):
        # 初始化字节编码器和解码器
        # 生成字节到 Unicode 字符的映射字典
        self.byte_encoder = bytes_to_unicode()
        # 生成 Unicode 字符到字节的映射字典
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # 读取 BPE 词汇表文件并处理
        # 读取 BPE 词汇表文件，按行分割
        merges = Path(bpe_path).read_text(encoding='utf8').split('\n')
        # 截取特定范围的词汇表行（跳过前几行和后几行）
        merges = merges[1:49152 - 256 - 2 + 1]
        # 将每行拆分为元组，并存储到 merges 列表中
        merges = [tuple(merge.split()) for merge in merges]

        # 生成词汇表
        # 获取所有 Unicode 字符作为基础词汇
        vocab = list(bytes_to_unicode().values())
        # 为每个字符添加 '</w>' 后缀，表示单词结束
        vocab = vocab + [v + '</w>' for v in vocab]
        for merge in merges:
            # 将所有 BPE 合并的词汇添加到词汇表中
            vocab.append(''.join(merge))
        
        # 添加开始和结束标记到词汇表中
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])

        # 设置词汇表大小
        self.vocab_size = 49408

        # 创建编码器字典，将词汇映射到索引
        self.encoder = dict(zip(vocab, range(len(vocab))))
        # 创建解码器字典，将索引映射回词汇
        self.decoder = {v: k for k, v in self.encoder.items()}

        # 创建 BPE 合并规则的排名字典
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        # 初始化缓存字典，用于存储常用词汇的 BPE 合并结果
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}

        # 定义正则表达式模式，用于分词
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE)

    def bpe(self, token):
        """
        对单个 token 应用 BPE 合并规则。

        参数:
            token (str): 输入的 token。

        返回:
            str: 应用 BPE 合并规则后的 token。
        """
        if token in self.cache:
            # 如果 token 在缓存中，则直接返回缓存结果
            return self.cache[token]
        # 将 token 转换为元组，并在最后一个字符后添加 '</w>' 标记
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        # 获取相邻字符对
        pairs = get_pairs(word)

        if not pairs:
            # 如果没有相邻字符对，则返回添加 '</w>' 的 token
            return token + '</w>'

        while True:
            # 找到当前 pairs 中排名最低的双字符对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                # 如果双字符对不在 BPE 排名中，则退出循环
                break

            # 解包双字符对
            first, second = bigram
            # 初始化新的单词列表
            new_word = []

            i = 0
            while i < len(word):
                try:
                    # 查找第一个字符的位置
                    j = word.index(first, i)
                    # 添加从当前位置到第一个字符之间的字符
                    new_word.extend(word[i:j])
                    # 更新索引
                    i = j
                except:
                    # 如果找不到第一个字符，则添加剩余的字符
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    # 如果找到双字符对，则添加合并后的字符
                    new_word.append(first + second)
                    # 更新索引，跳过第二个字符
                    i += 2
                else:
                    # 否则，添加当前字符
                    new_word.append(word[i])
                    i += 1
            
            # 将新的单词转换为元组
            new_word = tuple(new_word)
            # 更新单词
            word = new_word
            if len(word) == 1:
                # 如果单词长度为 1，则退出循环
                break
            else:
                # 否则，更新相邻字符对
                pairs = get_pairs(word)
        
        # 将单词元组转换为字符串
        word = ' '.join(word)
        # 将结果添加到缓存中
        self.cache[token] = word
        # 返回 BPE 合并后的 token
        return word

    def encode(self, text):
        """
        对文本进行编码，返回编码后的 token 索引列表。

        参数:
            text (str): 输入的文本。

        返回:
            List[int]: 编码后的 token 索引列表。
        """
        # 初始化 BPE 编码后的 token 列表
        bpe_tokens = []
        # 对文本进行基本清理和空白字符清理，并转换为小写
        text = whitespace_clean(basic_clean(text)).lower()
        # 使用正则表达式对文本进行分词
        for token in re.findall(self.pat, text):
            # 将 token 转换为字节，并映射为对应的 Unicode 字符
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # 对 BPE 合并后的 token 进行编码，并添加到列表中
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        # 返回编码后的 token 索引列表
        return bpe_tokens

    def decode(
        self,
        tokens,
        remove_start_end = True,
        pad_tokens = set()
    ):
        """
        对编码后的 token 索引列表进行解码，返回原始文本。

        参数:
            tokens (List[int]): 编码后的 token 索引列表。
            remove_start_end (bool, 可选): 是否移除开始和结束标记。默认值为 True。
            pad_tokens (set, 可选): 需要移除的填充 token。默认值为空集合。

        返回:
            str: 解码后的文本。
        """
        if torch.is_tensor(tokens):
            # 如果 tokens 是张量，则转换为列表
            tokens = tokens.tolist()

        if remove_start_end:
            # 移除开始和结束标记的索引
            tokens = [token for token in tokens if token not in (49406, 40407, 0)]
        # 将 token 索引映射回词汇，并拼接成字符串
        text = ''.join([self.decoder[token] for token in tokens if token not in pad_tokens])
        # 将字节转换为字符串，并替换 '</w>' 为空格
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        # 返回解码后的文本
        return text

    def tokenize(
        self,
        texts,
        context_length = 256,
        truncate_text = False
    ):
        """
        对输入文本进行分词和编码，返回 token 索引张量和最大上下文长度。

        参数:
            texts (Union[str, List[str]]): 输入的文本或文本列表。
            context_length (int, 可选): 上下文长度。默认值为 256。
            truncate_text (bool, 可选): 是否截断文本。默认值为 False。

        返回:
            Tuple[Tensor, int]: 返回 token 索引张量和最大上下文长度。
        """
        if isinstance(texts, str):
            # 如果输入是字符串，则转换为列表
            texts = [texts]
        # 对每个文本进行编码
        all_tokens = [self.encode(text) for text in texts]
        # 获取最大上下文长度
        max_context_length = max([len(tokens) for tokens in all_tokens])

        # 初始化结果张量
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate_text:
                    # 如果文本长度超过上下文长度，则截断
                    tokens = tokens[:context_length]
                else:
                    # 否则，抛出错误
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            # 将 token 索引填充到结果张量中
            result[i, :len(tokens)] = torch.tensor(tokens)
        # 返回结果张量和最大上下文长度
        return result, max_context_length


tokenizer = SimpleTokenizer()
