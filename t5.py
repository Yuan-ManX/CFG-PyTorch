from typing import List
from beartype import beartype

import torch
import transformers
from transformers import T5Tokenizer, T5EncoderModel, T5Config


# 设置 transformers 库的日志级别为错误，避免过多的日志输出
transformers.logging.set_verbosity_error()


# helpers
# 辅助函数

def exists(val):
    """
    检查一个值是否存在（不为 None）。

    参数:
        val: 需要检查的值。

    返回:
        bool: 如果 val 不为 None，则返回 True；否则返回 False。
    """
    return val is not None


def default(val, d):
    """
    返回可选值或默认值。

    参数:
        val: 需要检查的可选值。
        d: 默认值。

    返回:
        Any: 如果 val 存在，则返回 val；否则返回 d。
    """
    return val if exists(val) else d


# config
# 配置

# 设置最大序列长度为 256
MAX_LENGTH = 256

# 设置默认的 T5 模型名称为 'google/t5-v1_1-base'
DEFAULT_T5_NAME = 'google/t5-v1_1-base'

# 初始化一个空字典，用于存储 T5 模型的配置
T5_CONFIGS = {}


# singleton globals
# 全局变量，用于缓存 T5 模型和 tokenizer

def get_tokenizer(name):
    """
    获取指定名称的 T5 tokenizer。

    参数:
        name (str): T5 模型的名称。

    返回:
        T5Tokenizer: 预训练的 T5 tokenizer。
    """
    # 从预训练模型加载 T5 tokenizer
    tokenizer = T5Tokenizer.from_pretrained(name)
    return tokenizer


def get_model(name):
    """
    获取指定名称的 T5 模型。

    参数:
        name (str): T5 模型的名称。

    返回:
        T5EncoderModel: 预训练的 T5 模型。
    """
    # 从预训练模型加载 T5 模型
    model = T5EncoderModel.from_pretrained(name)
    return model


def get_model_and_tokenizer(name):
    """
    获取指定名称的 T5 模型和 tokenizer。如果已经加载过，则从缓存中获取。

    参数:
        name (str): T5 模型的名称。

    返回:
        Tuple[T5EncoderModel, T5Tokenizer]: 预训练的 T5 模型和 tokenizer。
    """
    # 声明使用全局变量 T5_CONFIGS
    global T5_CONFIGS

    if name not in T5_CONFIGS:
        # 如果模型名称不在缓存中，则初始化一个空字典
        T5_CONFIGS[name] = dict()
    if "model" not in T5_CONFIGS[name]:
        # 如果模型不在缓存中，则加载模型并缓存
        T5_CONFIGS[name]["model"] = get_model(name)
    if "tokenizer" not in T5_CONFIGS[name]:
        # 如果 tokenizer 不在缓存中，则加载 tokenizer 并缓存
        T5_CONFIGS[name]["tokenizer"] = get_tokenizer(name)

    # 返回模型和 tokenizer
    return T5_CONFIGS[name]['model'], T5_CONFIGS[name]['tokenizer']


def get_encoded_dim(name):
    """
    获取指定名称的 T5 模型的编码维度。

    参数:
        name (str): T5 模型的名称。

    返回:
        int: 模型的编码维度。
    """
    if name not in T5_CONFIGS:
        # avoids loading the model if we only want to get the dim
        # 如果只需要获取维度而不加载模型，则避免加载整个模型
        # 从预训练模型加载配置
        config = T5Config.from_pretrained(name)
        # 将配置缓存到 T5_CONFIGS 中
        T5_CONFIGS[name] = dict(config=config)
    elif "config" in T5_CONFIGS[name]:
        # 如果配置已经在缓存中，则直接使用
        config = T5_CONFIGS[name]["config"]
    elif "model" in T5_CONFIGS[name]:
        # 如果模型在缓存中，则从模型中获取配置
        config = T5_CONFIGS[name]["model"].config
    else:
        # 如果以上条件都不满足，则断言失败
        assert False
    # 返回模型的编码维度
    return config.d_model


# encoding text
# 文本编码

def t5_encode_text(texts, name = DEFAULT_T5_NAME, output_device = None):
    """
    使用 T5 模型对输入文本进行编码。

    参数:
        texts (List[str]): 输入的文本列表。
        name (str, 可选): T5 模型的名称。默认值为 DEFAULT_T5_NAME ('google/t5-v1_1-base')。
        output_device (Optional[torch.device], 可选): 输出设备。默认值为 None。

    返回:
        Tuple[Tensor, Tensor]: 返回编码后的文本张量和注意力掩码。
    """
    # 获取 T5 模型和 tokenizer
    t5, tokenizer = get_model_and_tokenizer(name)

    if torch.cuda.is_available():
        # 如果 CUDA 可用，则将模型移动到 CUDA 设备
        t5 = t5.cuda()

    # 获取模型所在的设备
    device = next(t5.parameters()).device

    # 对文本进行编码，返回包含 input_ids 和 attention_mask 的字典
    encoded = tokenizer.batch_encode_plus(
        texts, # 输入文本
        return_tensors = "pt", # 返回 PyTorch 张量
        padding = 'longest', # 使用最长序列长度进行填充
        max_length = MAX_LENGTH, # 设置最大序列长度
        truncation = True # 启用截断
    )

    # 将 input_ids 移动到模型所在的设备
    input_ids = encoded.input_ids.to(device)
    # 将 attention_mask 移动到模型所在的设备
    attn_mask = encoded.attention_mask.to(device)

    # 将模型设置为评估模式
    t5.eval()

    with torch.no_grad():
        # 前向传播，获取模型输出
        output = t5(input_ids = input_ids, attention_mask = attn_mask)
        # 获取编码后的文本（最后一个隐藏层的输出）并分离计算图
        encoded_text = output.last_hidden_state.detach()

    # 将 attention_mask 转换为布尔类型
    attn_mask = attn_mask.bool()

    if not exists(output_device):
        # 如果未指定输出设备，则返回编码后的文本和 attention_mask
        return encoded_text, attn_mask

    # 将编码后的文本移动到输出设备
    encoded_text.to(output_device)
    # 将 attention_mask 移动到输出设备
    attn_mask.to(output_device)

    # 返回编码后的文本和 attention_mask
    return encoded_text, attn_mask


class T5Adapter():
    """
    T5Adapter 类，用于适配 T5 模型，并提供文本嵌入功能。

    参数:
        name (str): T5 模型的名称。
        text_embed_pad_value (float, 可选): 文本嵌入填充值。默认值为 0.0。
    """
    def __init__(
        self,
        name,
        text_embed_pad_value = 0.
    ):
        # 如果未提供 name，则使用默认的 T5 模型名称
        name = default(name, DEFAULT_T5_NAME)
        # 获取 T5 模型和 tokenizer
        t5, tokenizer = get_model_and_tokenizer(name)

        if torch.cuda.is_available():
            # 如果 CUDA 可用，则将模型移动到 CUDA 设备
            t5 = t5.cuda()

        # 存储模型名称
        self.name =  name
        # 存储 T5 模型
        self.t5 = t5
        # 存储 tokenizer
        self.tokenizer = tokenizer
        # 存储文本嵌入填充值
        self.text_embed_pad_value = text_embed_pad_value

    @property
    def dim_latent(self):
        """
        获取潜在空间的维度。

        返回:
            int: 潜在空间的维度。
        """
        return get_encoded_dim(self.name)

    @property
    def max_text_len(self):
        """
        获取最大文本长度。

        返回:
            int: 最大文本长度。
        """
        return MAX_LENGTH

    @torch.no_grad()
    @beartype # 使用 beartype 进行类型检查
    def embed_text(
        self,
        texts: List[str],
        return_text_encodings = False,
        output_device = None
    ):
        """
        嵌入文本，返回文本嵌入或文本编码。

        参数:
            texts (List[str]): 输入的文本列表。
            return_text_encodings (bool, 可选): 是否返回文本编码。默认值为 False。
            output_device (Optional[torch.device], 可选): 输出设备。默认值为 None。

        返回:
            Union[Tensor, Tuple[Tensor, Tensor]]: 返回文本嵌入或文本编码。
        """
        # 获取 T5 模型所在的设备
        device = next(self.t5.parameters()).device

        # 对文本进行编码，返回包含 input_ids 和 attention_mask 的字典
        encoded = self.tokenizer.batch_encode_plus(
            texts, # 输入文本
            return_tensors = "pt", # 返回 PyTorch 张量
            padding = 'longest', # 使用最长序列长度进行填充
            max_length = MAX_LENGTH, # 设置最大序列长度
            truncation = True # 启用截断
        )

        # 将 input_ids 移动到模型所在的设备
        input_ids = encoded.input_ids.to(device)
        # 将 attention_mask 移动到模型所在的设备
        attn_mask = encoded.attention_mask.to(device)

        # 将模型设置为评估模式
        self.t5.eval()

        with torch.no_grad():
            # 前向传播，获取模型输出
            output = self.t5(input_ids = input_ids, attention_mask = attn_mask)
            # 获取编码后的文本（最后一个隐藏层的输出）并分离计算图
            encoded_text = output.last_hidden_state.detach()

        # 将 attention_mask 转换为布尔类型
        attn_mask = attn_mask.bool()

        # 使用填充值填充掩码外的部分
        encoded_text.masked_fill_(~attn_mask[..., None], self.text_embed_pad_value)

        if not return_text_encodings:
            # 如果不需要返回文本编码，则计算均值编码
            # 对文本嵌入进行求和
            numer = encoded_text.sum(dim = -2)
            # 计算注意力掩码的和
            denom = attn_mask.sum(dim = -1)[..., None]
            # 如果注意力掩码的和为 0，则将对应的位置填充为 0
            numer.masked_fill_(denom == 0, 0.)
            # 计算均值编码，并避免除以零
            mean_encodings = numer / denom.clamp(min = 1e-3)
            # 返回均值编码
            return mean_encodings

        # 返回编码后的文本
        return encoded_text.to(output_device)
