from __future__ import annotations
from collections import namedtuple
from functools import wraps, partial, cache

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, pack, unpack

from beartype.door import is_bearable
from beartype.typing import Callable, Tuple, List, Literal, Dict, Any

from inspect import signature

from typing import typecheck, beartype_isinstance
from t5 import T5Adapter
from open_clip import OpenClipAdapter
from attend import Attend
from bge import BGEAdapter


# constants
# 常量

# 条件 dropout 概率的键名
COND_DROP_KEY_NAME = 'cond_drop_prob'

# 文本的键名
TEXTS_KEY_NAME = 'texts'
# 文本嵌入的键名
TEXT_EMBEDS_KEY_NAME = 'text_embeds'
# 文本条件器的名称
TEXT_CONDITIONER_NAME = 'text_conditioner'
# 条件函数的键名
CONDITION_FUNCTION_KEY_NAME = 'cond_fns'

# 定义 TextCondReturn 命名元组，用于存储文本条件返回的结果
TextCondReturn = namedtuple('TextCondReturn', [
    'embed', # 文本嵌入
    'mask' # 掩码
])


# helper functions
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


def is_empty(l):
    """
    检查列表是否为空。

    参数:
        l (list): 需要检查的列表。

    返回:
        bool: 如果列表为空，则返回 True；否则返回 False。
    """
    return len(l) == 0


def default(*values):
    """
    返回第一个存在（不为 None）的值。

    参数:
        *values: 需要检查的可选值。

    返回:
        Any: 返回第一个存在（不为 None）的值。如果所有值都不存在，则返回 None。
    """
    for value in values:
        if exists(value):
            return value
    return None


def cast_tuple(val, length = 1):
    """
    将输入值转换为元组。如果输入已经是元组，则直接返回；否则，将其重复指定次数并转换为元组。

    参数:
        val: 需要转换的值。
        length (int, 可选): 如果输入不是元组，则重复的次数。默认值为 1。

    返回:
        tuple: 转换后的元组。
    """
    return val if isinstance(val, tuple) else ((val,) * length)


def pack_one(x, pattern):
    """
    将单个张量 x 按照指定的 pattern 打包。

    参数:
        x (Tensor): 需要打包的张量。
        pattern (str): 打包的模式，例如 'b *' 表示批次维度和其他维度。

    返回:
        Tensor: 打包后的张量。
    """
    return pack([x], pattern)


def unpack_one(x, ps, pattern):
    """
    将打包后的张量 x 按照指定的 pattern 和打包参数 ps 进行解包，并返回第一个解包后的张量。

    参数:
        x (Tensor): 需要解包的张量。
        ps: 打包参数。
        pattern (str): 解包的模式。

    返回:
        Tensor: 解包后的第一个张量。
    """
    return unpack(x, ps, pattern)[0]


def pack_one_with_inverse(x, pattern):
    """
    将单个张量 x 按照指定的 pattern 打包，并返回一个反向操作函数，用于解包。

    参数:
        x (Tensor): 需要打包的张量。
        pattern (str): 打包的模式。

    返回:
        Tuple[Tensor, callable]: 返回打包后的张量和反向操作函数。
    """
    # 打包张量，并获取打包后的形状
    packed, packed_shape = pack_one(x, pattern)

    def inverse(x, inverse_pattern = None):
        """
        反向操作函数，将打包后的张量 x 解包回原始张量。

        参数:
            x (Tensor): 打包后的张量。
            inverse_pattern (Optional[str], 可选): 反向解包的 pattern。如果未提供，则使用原始的 pattern。

        返回:
            Tensor: 解包后的原始张量。
        """
        # 解包并返回原始张量
        return unpack_one(x, packed_shape, default(inverse_pattern, pattern))
    # 返回打包后的张量和反向操作函数
    return packed, inverse


# tensor helpers

def project(x, y):
    """
    将张量 x 投影到张量 y 上，并返回平行分量和正交分量。

    参数:
        x (Tensor): 输入张量。
        y (Tensor): 投影目标张量。

    返回:
        Tuple[Tensor, Tensor]: 返回平行分量和正交分量的张量。
    """
    # 将张量 x 和 y 按照 'b *' 模式打包，并获取反向操作函数
    x, inverse = pack_one_with_inverse(x, 'b *')
    y, _ = pack_one_with_inverse(y, 'b *')

    # 获取张量 x 的数据类型
    dtype = x.dtype

    # 将张量 x 和 y 转换为双精度浮点数类型
    x, y = x.double(), y.double()
    # 对张量 y 在最后一个维度上进行 L2 归一化，得到单位向量
    unit = F.normalize(y, dim = -1)

    # 计算平行分量：x 在 y 方向上的投影
    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    # 计算正交分量：x 中垂直于 y 的部分
    orthogonal = x - parallel

    # 将平行分量和正交分量反向解包回原始数据类型
    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)


def prob_mask_like(shape, prob, device):
    """
    根据给定的形状和概率生成一个布尔掩码张量。

    参数:
        shape (Tuple[int, ...]): 掩码张量的形状。
        prob (float): 掩码中元素为 True 的概率。
        device (torch.device): 掩码张量所在的设备。

    返回:
        Tensor: 生成的布尔掩码张量。
    """
    if prob == 1:
        # 如果概率为 1，则生成全为 True 的掩码
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        # 如果概率为 0，则生成全为 False 的掩码
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        # 否则，生成一个均匀分布的随机张量，并根据概率生成布尔掩码
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob


# classifier free guidance with automatic text conditioning

@typecheck
def classifier_free_guidance(
    fn: Callable,
    cond_drop_prob_keyname = COND_DROP_KEY_NAME,
    texts_key_name = TEXTS_KEY_NAME,
    text_embeds_key_name = TEXT_EMBEDS_KEY_NAME,
    cond_fns_keyname = CONDITION_FUNCTION_KEY_NAME,
    text_conditioner_name = TEXT_CONDITIONER_NAME
):
    """
    对给定的函数应用 classifier-free guidance（无分类器引导）。

    参数:
        fn (Callable): 需要应用 classifier-free guidance 的函数。
        cond_drop_prob_keyname (str, 可选): 条件 dropout 概率的键名。默认值为 COND_DROP_KEY_NAME。
        texts_key_name (str, 可选): 文本的键名。默认值为 TEXTS_KEY_NAME。
        text_embeds_key_name (str, 可选): 文本嵌入的键名。默认值为 TEXT_EMBEDS_KEY_NAME。
        cond_fns_keyname (str, 可选): 条件函数的键名。默认值为 CONDITION_FUNCTION_KEY_NAME。
        text_conditioner_name (str, 可选): 文本条件器的名称。默认值为 TEXT_CONDITIONER_NAME。

    返回:
        Callable: 应用了 classifier-free guidance 的函数。
    """
    # 获取函数 fn 的参数
    fn_params = signature(fn).parameters

    # 判断是否需要自动处理文本条件
    auto_handle_text_condition = texts_key_name not in fn_params and text_embeds_key_name not in fn_params

    @wraps(fn) # 保留原函数的元数据
    def inner(
        self,
        *args,
        cond_scale: float = 1., # 条件缩放因子，默认值为 1.0
        rescale_phi: float = 0., # 重缩放因子，默认值为 0.0
        return_unconditioned: bool = False, # 是否返回无条件的预测结果，默认值为 False
        remove_parallel_component: bool = False, # 是否移除平行分量，默认值为 False
        keep_parallel_frac: float = 0., # 保留平行分量的比例，默认值为 0.0（论文中建议完全移除）
        cfg_routed_kwargs: Dict[str, Tuple[Any, Any]] = dict(), # 用于传递不同的参数给前向传播和 null 前向传播（用于处理使用 CFG 时 transformer 解码的缓存）
        **kwargs
    ):
        """
        应用 classifier-free guidance 的内部函数。

        参数:
            self: 类的实例。
            *args: 位置参数。
            cond_scale (float, 可选): 条件缩放因子。默认值为 1.0。
            rescale_phi (float, 可选): 重缩放因子。默认值为 0.0。
            return_unconditioned (bool, 可选): 是否返回无条件的预测结果。默认值为 False。
            remove_parallel_component (bool, 可选): 是否移除平行分量。默认值为 False。
            keep_parallel_frac (float, 可选): 保留平行分量的比例。默认值为 0.0。
            cfg_routed_kwargs (Dict[str, Tuple[Any, Any]], 可选): 用于传递不同的参数给前向传播和 null 前向传播。默认值为空字典。
            **kwargs: 其他关键字参数。

        返回:
            Any: 应用 classifier-free guidance 后的结果。
        """
        @wraps(fn) # 保留原函数的元数据
        def fn_maybe_with_text(self, *args, **kwargs):
            if auto_handle_text_condition:
                # 如果需要自动处理文本条件，则从 kwargs 中弹出文本和文本嵌入
                texts = kwargs.pop('texts', None)
                text_embeds = kwargs.pop('text_embeds', None)

                # 确保文本和文本嵌入不同时存在
                assert not (exists(texts) and exists(text_embeds))

                # 初始化原始文本条件和条件函数
                raw_text_cond = cond_fns = None

                # 获取文本条件器
                text_conditioner = getattr(self, text_conditioner_name, None)

                # 获取条件 dropout 概率
                cond_drop_prob = kwargs.pop(cond_drop_prob_keyname, None)

                # 确保条件 dropout 概率在 [0, 1] 之间
                assert not exists(cond_drop_prob) or 0. <= cond_drop_prob <= 1.

                # auto convert texts -> conditioning functions
                # 自动将文本转换为条件函数

                if exists(texts) ^ exists(text_embeds):

                    assert is_bearable(texts, List[str] | None), f'keyword `{texts_key_name}` must be a list of strings'

                    assert exists(text_conditioner) and is_bearable(text_conditioner, Conditioner), 'text_conditioner must be set on your network with the correct hidden dimensions to be conditioned on'

                    text_condition_input = dict(texts = texts) if exists(texts) else dict(text_embeds = text_embeds)

                    cond_fns, raw_text_cond = text_conditioner(**text_condition_input, cond_drop_prob = cond_drop_prob)

                elif isinstance(text_conditioner, NullConditioner):
                    assert cond_drop_prob == 0., 'null conditioner has nothing to dropout'

                    cond_fns, raw_text_cond = text_conditioner()

                if 'cond_fns' in fn_params:
                    kwargs.update(cond_fns = cond_fns)

                if 'raw_text_cond' in fn_params:
                    kwargs.update(raw_text_cond = raw_text_cond)

            return fn(self, *args, **kwargs)

        # main classifier free guidance logic
        # classifier-free guidance 的主要逻辑

        if self.training:
            assert cond_scale == 1, 'you cannot do condition scaling when in training mode'

            return fn_maybe_with_text(self, *args, **kwargs)

        assert cond_scale >= 1, 'invalid conditioning scale, must be greater or equal to 1'

        kwargs_without_cond_dropout = {**kwargs, cond_drop_prob_keyname: 0.}
        kwargs_with_cond_dropout = {**kwargs, cond_drop_prob_keyname: 1.}

        # handle kwargs to be routed to forward and nulled forward separately
        # for handling caching of both calls
        # 处理分别传递给前向传播和 null 前向传播的参数
        # 以处理使用 CFG 时 transformer 解码的缓存

        fn_kwargs = {k: v[0] for k, v in cfg_routed_kwargs.items()}
        null_fn_kwargs = {k: v[1] for k, v in cfg_routed_kwargs.items()}

        # non-null forward

        outputs = fn_maybe_with_text(self, *args, **fn_kwargs, **kwargs_without_cond_dropout)

        if cond_scale == 1:
            return outputs

        logits, *rest = cast_tuple(outputs)

        # nulled forward

        null_outputs = fn_maybe_with_text(self, *args, **null_fn_kwargs, **kwargs_with_cond_dropout)

        null_logits, *null_rest = cast_tuple(null_outputs)

        zipped_rest = tuple(zip(rest, null_rest))

        update = logits - null_logits

        if remove_parallel_component:
            update_parallel, update_orthog = project(update, logits)
            update = update_orthog + update_parallel * keep_parallel_frac

        scaled_logits = logits + update * (cond_scale - 1.)

        if rescale_phi <= 0:
            logit_output = scaled_logits
        else:
            # proposed in https://arxiv.org/abs/2305.08891
            # as a way to prevent over-saturation with classifier free guidance
            # works both in pixel as well as latent space as opposed to the solution from imagen

            dims = tuple(range(1, logits.ndim - 1))
            rescaled_logits = scaled_logits * (logits.std(dim = dims, keepdim = True) / scaled_logits.std(dim = dims, keepdim= True))
            logit_output = rescaled_logits * rescale_phi + scaled_logits * (1. - rescale_phi)

        # can return unconditioned prediction
        # for use in CFG++ https://arxiv.org/abs/2406.08070
        # 可以返回无条件的预测结果
        # 用于 CFG++

        output = logit_output

        if return_unconditioned:
            output = (output, null_logits)

        # handle multiple outputs from original function
        # 处理原始函数返回的多个输出

        if is_empty(zipped_rest):
            return output

        return (output, *zipped_rest)

    return inner


# class decorator

@typecheck # 使用 typecheck 装饰器进行类型检查
def classifier_free_guidance_class_decorator(
    orig_class,
    cond_drop_prob_keyname = COND_DROP_KEY_NAME,
    texts_key_name = TEXTS_KEY_NAME,
    text_embeds_key_name = TEXT_EMBEDS_KEY_NAME,
    cond_fns_keyname = CONDITION_FUNCTION_KEY_NAME,
    text_conditioner_name = TEXT_CONDITIONER_NAME
):
    """
    对给定的类应用 classifier-free guidance（无分类器引导）。

    参数:
        orig_class (Module): 需要应用 classifier-free guidance 的原始类。
        cond_drop_prob_keyname (str, 可选): 条件 dropout 概率的键名。默认值为 COND_DROP_KEY_NAME。
        texts_key_name (str, 可选): 文本的键名。默认值为 TEXTS_KEY_NAME。
        text_embeds_key_name (str, 可选): 文本嵌入的键名。默认值为 TEXT_EMBEDS_KEY_NAME。
        cond_fns_keyname (str, 可选): 条件函数的键名。默认值为 CONDITION_FUNCTION_KEY_NAME。
        text_conditioner_name (str, 可选): 文本条件器的名称。默认值为 TEXT_CONDITIONER_NAME。

    返回:
        Callable: 应用了 classifier-free guidance 的类。
    """
    assert issubclass(orig_class, Module)

    # decorate init
    
    # 获取原始类的 __init__ 方法
    orig_init = orig_class.__init__

    @wraps(orig_init) # 保留原始 __init__ 方法的元数据
    @typecheck # 使用 typecheck 装饰器进行类型检查
    def __init__(
        self,
        *args,
        text_condition_type: Literal['film', 'attention', 'null', 'raw'] = 'film', # 文本条件类型，可选值为 'film', 'attention', 'null', 'raw'，默认值为 'film'
        text_condition_model_types: Tuple[str, ...] = ('t5',), # 文本条件模型类型，默认值为 ('t5',)
        text_condition_hidden_dims: Tuple[int, ...], # 文本条件隐藏层维度
        text_condition_cond_drop_prob: float, # 文本条件 dropout 概率
        **kwargs
    ):
        # 调用原始类的 __init__ 方法
        orig_init(self, *args, **kwargs)

        # 根据文本条件类型选择相应的条件器类
        if text_condition_type == 'film':
            condition_klass = TextConditioner
        elif text_condition_type == 'attention':
            condition_klass = AttentionTextConditioner
        elif text_condition_type == 'raw':
            condition_klass = TextEmbeddingReturner
        else:
            condition_klass = NullConditioner

        # 实例化文本条件器
        self.text_conditioner = condition_klass(
            model_types = text_condition_model_types, # 模型类型
            hidden_dims = text_condition_hidden_dims, # 隐藏层维度
            cond_drop_prob = text_condition_cond_drop_prob # dropout 概率
        )

    # 将装饰后的 __init__ 方法赋值给原始类
    orig_class.__init__ = __init__

    # decorate forward
    # 装饰类的 forward 方法
    decorated_forward = classifier_free_guidance(
        orig_class.forward, # 需要装饰的 forward 方法
        cond_drop_prob_keyname = cond_drop_prob_keyname, # 条件 dropout 概率的键名
        texts_key_name = texts_key_name, # 文本的键名
        text_embeds_key_name = text_embeds_key_name, # 文本嵌入的键名
        cond_fns_keyname = cond_fns_keyname, # 条件函数的键名
        text_conditioner_name = text_conditioner_name # 文本条件器的名称
    )

    # 将装饰后的 forward 方法赋值给原始类
    orig_class.forward = decorated_forward

    # forward `embed_texts` to the `text_conditioner.embed_texts`
    # 将 `embed_texts` 方法转发到 `text_conditioner.embed_texts`

    @typecheck # 使用 typecheck 装饰器进行类型检查
    def embed_texts(self, texts: List[str]):
        """
        嵌入文本，返回文本嵌入。

        参数:
            texts (List[str]): 输入的文本列表。

        返回:
            Tensor: 嵌入后的文本张量。
        """
        return self.text_conditioner.embed_texts(texts)

    @property
    @cache
    def max_cond_text_len(self):
        """
        获取最大条件文本长度。

        返回:
            int: 最大条件文本长度。
        """
        # 计算所有文本模型的最大文本长度之和
        total_cond_text_len = sum([text_model.max_text_len for text_model in self.text_conditioner.text_models])
        # 返回最大条件文本长度
        return total_cond_text_len
    
    # 将 `max_cond_text_len` 属性赋值给原始类（如果不存在）
    if not hasattr(orig_class, 'max_cond_text_len'):
        orig_class.max_cond_text_len = max_cond_text_len

    # 将 `embed_texts` 方法赋值给原始类（如果不存在）
    if not hasattr(orig_class, 'embed_texts'):
        orig_class.embed_texts = embed_texts

    # 添加一个标记，表示类已经被装饰
    orig_class.__decorated_with_cfg = True
    # 返回装饰后的类
    return orig_class


# attention

class Attention(Module):
    """
    Attention 模块，用于实现多头自注意力机制。

    参数:
        dim (int): 输入和输出的维度。
        dim_head (int, 可选): 每个注意力头的维度。默认值为 64。
        heads (int, 可选): 注意力头的数量。默认值为 8。
        dim_context (int, 可选): 上下文向量的维度。如果未提供，则默认为 dim。
        norm_context (bool, 可选): 是否对上下文向量进行归一化。默认值为 False。
        num_null_kv (int, 可选): 额外的 null key-value 对的数量，用于填充。默认值为 0。
        flash (bool, 可选): 是否使用 Flash Attention。默认值为 False。
    """
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dim_context = None,
        norm_context = False,
        num_null_kv = 0,
        flash = False
    ):
        super().__init__()
        # 存储注意力头的数量
        self.heads = heads
        # 计算缩放因子
        self.scale = dim_head ** -0.5
        # 计算多头注意力后的内部维度
        inner_dim = dim_head * heads

        # 如果未提供上下文维度，则默认为输入维度
        dim_context = default(dim_context, dim)

        # 对输入进行层归一化
        self.norm = nn.LayerNorm(dim)
        # 如果需要，对上下文进行层归一化；否则，使用恒等变换
        self.context_norm = nn.LayerNorm(dim_context) if norm_context else nn.Identity()

        # 创建 Attend 实例，用于计算注意力
        self.attend = Attend(flash = flash)        

        # 存储额外的 null key-value 对的数量
        self.num_null_kv = num_null_kv
        # 初始化 null key-value 对
        self.null_kv = nn.Parameter(torch.randn(2, num_null_kv, dim_head))

        # 线性变换，将输入转换为查询向量
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        # 线性变换，将上下文转换为键和值向量
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias = False)
        # 线性变换，将多头注意力输出转换回原始维度
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        context = None,
        mask = None
    ):
        """
        前向传播方法，计算多头自注意力。

        参数:
            x (Tensor): 输入张量，形状为 (batch_size, sequence_length, dim)。
            context (Optional[Tensor], 可选): 上下文张量，形状为 (batch_size, context_length, dim_context)。默认值为 None。
            mask (Optional[Tensor], 可选): 注意力掩码，形状为 (batch_size, sequence_length)。默认值为 None。

        返回:
            Tensor: 注意力输出，形状为 (batch_size, sequence_length, dim)。
        """
        # 获取批次大小
        b = x.shape[0]

        if exists(context):
            # 对上下文进行层归一化
            context = self.context_norm(context)

        # 如果存在上下文，则使用上下文作为键和值输入；否则，使用输入 x
        kv_input = default(context, x)

        # 对输入进行层归一化
        x = self.norm(x)

        # 计算查询、键和值向量
        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)

        if self.num_null_kv > 0:
            # 重复 null key-value 对以匹配批次大小
            null_k, null_v = repeat(self.null_kv, 'kv n d -> kv b n d', b = b).unbind(dim = 0)
            # 将 null key 拼接到键向量中
            k = torch.cat((null_k, k), dim = -2)
            # 将 null value 拼接到值向量中
            v = torch.cat((null_v, v), dim = -2)

        if exists(mask):
            # 对掩码进行填充，以匹配 null key-value 对
            mask = F.pad(mask, (self.num_null_kv, 0), value = True)
            # 重塑掩码形状为 (batch_size, 1, 1, sequence_length)
            mask = rearrange(mask, 'b j -> b 1 1 j')

        # 重塑查询向量形状为 (batch_size, heads, sequence_length, dim_head)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # 计算注意力输出
        out = self.attend(q, k, v, mask = mask)

        # 重塑注意力输出形状为 (batch_size, sequence_length, inner_dim)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 通过线性变换将输出转换回原始维度
        return self.to_out(out)


# dimension adapters

def rearrange_channel_last(fn):
    """
    装饰器：将输入张量的通道维度从第一个维度移动到最后一个维度，并调用被装饰的函数。

    参数:
        fn (callable): 需要被装饰的函数，该函数接受形状为 (batch_size, ..., dim) 的张量作为输入。

    返回:
        callable: 装饰后的函数，返回形状为 (batch_size, ..., dim) 的张量。
    """
    @wraps(fn)
    def inner(hiddens):
        # 将输入张量打包为 (batch_size, ..., dim) 的格式
        hiddens, ps = pack_one(hiddens, 'b * d')
        # 调用被装饰的函数
        conditioned = fn(hiddens)
        # 解包并返回结果，保持原始形状
        return unpack_one(conditioned, ps, 'b * d')
    # 返回装饰后的函数
    return inner


def rearrange_channel_first(fn):
    """ will adapt shape of (batch, feature, ...) for conditioning """
    """
    装饰器：将输入张量的通道维度从最后一个维度移动到第一个维度，并调用被装饰的函数。

    参数:
        fn (callable): 需要被装饰的函数，该函数接受形状为 (batch_size, dim, ...) 的张量作为输入。

    返回:
        callable: 装饰后的函数，返回形状为 (batch_size, dim, ...) 的张量。
    """

    @wraps(fn)
    def inner(hiddens):
        # 将输入张量打包为 (batch_size, dim, ...) 的格式
        hiddens, ps = pack_one(hiddens, 'b d *')
        # 重塑张量形状，将通道维度移动到最后一个维度
        hiddens = rearrange(hiddens, 'b d n -> b n d')
        # 调用被装饰的函数
        conditioned =  fn(hiddens)
        # 重塑张量形状，将通道维度移动回第一个维度
        conditioned = rearrange(conditioned, 'b n d -> b d n')
        # 解包并返回结果，保持原始形状
        return unpack_one(conditioned, ps, 'b d *')
    # 返回装饰后的函数
    return inner


# conditioning modules

class FiLM(Module):
    """
    FiLM (Feature-wise Linear Modulation) 条件化模块，用于对隐藏层进行特征级别的线性调制。

    参数:
        dim (int): 输入条件的维度。
        hidden_dim (int): 隐藏层的维度。
    """
    def __init__(
        self,
        dim,
        hidden_dim
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * 4), # 线性变换，将条件转换为更高维度的特征
            nn.SiLU(), # SiLU 激活函数
            nn.Linear(hidden_dim * 4, hidden_dim * 2) # 线性变换，将特征转换为缩放和偏移参数
        )

        # 初始化最后一层线性变换的权重为 0
        nn.init.zeros_(self.net[-1].weight)
        # 初始化最后一层线性变换的偏置为 0
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, conditions, hiddens):
        """
        前向传播方法，应用 FiLM 调制。

        参数:
            conditions (Tensor): 条件输入，形状为 (batch_size, dim)。
            hiddens (Tensor): 隐藏层输入，形状为 (batch_size, ..., hidden_dim)。

        返回:
            Tensor: 调制后的隐藏层输出。
        """
        # 将网络输出拆分为缩放因子和偏移量
        scale, shift = self.net(conditions).chunk(2, dim = -1)
        # 确保缩放因子和隐藏层维度匹配
        assert scale.shape[-1] == hiddens.shape[-1], f'unexpected hidden dimesion {hiddens.shape[-1]} used for conditioning'
        # 重塑缩放因子和偏移量的形状为 (batch_size, 1, hidden_dim)
        scale, shift = map(lambda t: rearrange(t, 'b d -> b 1 d'), (scale, shift))
        # 应用 FiLM 调制，返回调制后的隐藏层输出
        return hiddens * (scale + 1) + shift


class CrossAttention(Module):
    """
    交叉注意力模块，用于将条件信息融入隐藏层。

    参数:
        dim (int): 条件输入的维度。
        hidden_dim (int): 隐藏层的维度。
        heads (int, 可选): 注意力头的数量。默认值为 8。
        dim_head (int, 可选): 每个注意力头的维度。默认值为 64。
        flash (bool, 可选): 是否使用 Flash Attention。默认值为 False。
    """
    def __init__(
        self,
        dim,
        hidden_dim,
        heads = 8,
        dim_head = 64,
        flash = False
    ):
        super().__init__()
        self.attn = Attention(
            dim = hidden_dim, # 注意力模块的隐藏层维度
            dim_context = dim, # 上下文向量的维度为条件输入的维度
            norm_context = True, # 对上下文进行归一化
            num_null_kv = 1, # 额外的 null key-value 对数量
            dim_head = dim_head, # 每个注意力头的维度
            heads = heads, # 注意力头的数量
            flash = flash # 是否使用 Flash Attention
        )

    def forward(
        self,
        condition, # 条件输入，形状为 (batch_size, ..., dim)
        hiddens, # 隐藏层输入，形状为 (batch_size, ..., hidden_dim)
        mask = None # 注意力掩码，形状为 (batch_size, ..., sequence_length)
    ):
        """
        前向传播方法，应用交叉注意力。

        参数:
            condition (Tensor): 条件输入。
            hiddens (Tensor): 隐藏层输入。
            mask (Optional[Tensor], 可选): 注意力掩码。默认值为 None。

        返回:
            Tensor: 应用交叉注意力后的隐藏层输出。
        """
        # 应用交叉注意力并添加残差连接
        return self.attn(hiddens, condition, mask = mask) + hiddens


# film text conditioning

# 条件化配置字典，定义不同类型的文本条件器
CONDITION_CONFIG = dict(
    t5 = T5Adapter, # 使用 T5Adapter 作为 't5' 类型的条件器
    clip = OpenClipAdapter, # 使用 OpenClipAdapter 作为 'clip' 类型的条件器
    bge = BGEAdapter # 使用 BGEAdapter 作为 'bge' 类型的条件器
)


# 模型类型列表，包含所有可用的条件器类型
# 获取 CONDITION_CONFIG 字典的所有键，即 ['t5', 'clip', 'bge']
MODEL_TYPES = CONDITION_CONFIG.keys()


class Conditioner(Module):
    """
    Conditioner 基类，用于定义文本条件器的接口。
    """
    pass


# null conditioner

class Identity(Module):
    """
    恒等变换模块，用于返回输入张量不变。
    """
    def forward(self, t, *args, **kwargs):
        """
        前向传播方法，返回输入张量 t。

        参数:
            t (Tensor): 输入张量。
            *args: 其他位置参数。
            **kwargs: 其他关键字参数。

        返回:
            Tensor: 输入张量 t。
        """
        return t


class NullConditioner(Conditioner):
    """
    NullConditioner 类，用于实现空条件器，不进行任何文本嵌入或条件化。

    参数:
        hidden_dims (Tuple[int, ...]): 隐藏层的维度。
        **kwargs: 其他关键字参数。
    """
    @typecheck # 使用 typecheck 装饰器进行类型检查
    def __init__(
        self,
        *,
        hidden_dims: Tuple[int, ...],
        **kwargs
    ):
        super().__init__()
        # 获取隐藏层维度的数量
        num_null_conditioners = len(hidden_dims)
        # 初始化一个包含多个恒等变换的元组，作为条件函数
        self.cond_fns = tuple(Identity() for _ in range(num_null_conditioners))

        # 注册一个缓冲区，用于跟踪设备信息
        self.register_buffer('_device_param', torch.tensor(0), persistent = False)

    @property
    def device(self):
        """
        获取设备信息。

        返回:
            torch.device: 当前设备。
        """
        return next(self.buffers()).device

    @typecheck
    def embed_texts(self, texts: List[str]):
        """
        嵌入文本。

        参数:
            texts (List[str]): 输入的文本列表。

        断言:
            如果调用此方法，则断言失败，因为 null conditioner 不能嵌入文本。
        """
        assert False, 'null conditioner cannot embed text'

    def forward(self, *args, **kwarg):
        """
        前向传播方法，返回条件函数和原始文本条件。

        参数:
            *args: 其他位置参数。
            **kwarg: 其他关键字参数。

        返回:
            Tuple[Tuple[Identity, ...], None]: 返回一个包含多个恒等变换的元组和 None。
        """
        return self.cond_fns, None


# text conditioner with FiLM

class TextConditioner(Conditioner):
    """
    TextConditioner 类，使用 FiLM（Feature-wise Linear Modulation）进行文本条件化。

    参数:
        hidden_dims (Tuple[int, ...]): 隐藏层的维度，用于每个条件函数。
        model_types (str | Tuple[str, ...], 可选): 条件器类型，可以是 't5', 'clip', 'bge' 中的一个或多个。默认值为 't5'。
        model_names (str | Tuple[str, ...], 可选): 预训练模型的名称。默认值为 None。
        cond_drop_prob (float, 可选): 条件 dropout 概率。默认值为 0.0。
        hiddens_channel_first (bool | Tuple[bool, ...], 可选): 隐藏层是否为通道优先。默认值为 True。
        text_embed_stem_dim_mult (int, 可选): 文本嵌入主干 MLP 的维度倍数。默认值为 2。
        text_embed_pad_value (float, 可选): 文本嵌入填充值。默认值为 0.0。
    """
    @typecheck
    def __init__(
        self,
        *,
        hidden_dims: Tuple[int, ...],
        model_types = 't5',
        model_names = None,
        cond_drop_prob = 0.,
        hiddens_channel_first = True,
        text_embed_stem_dim_mult = 2,
        text_embed_pad_value = 0.
    ):
        super().__init__()
        # 将 model_types 转换为元组
        model_types = cast_tuple(model_types)
        # 将 model_names 转换为元组，长度与 model_types 相同
        model_names = cast_tuple(model_names, length = len(model_types))

        # 确保 model_types 和 model_names 的长度相同
        assert len(model_types) == len(model_names)
        # 确保所有 model_types 都是有效的类型
        assert all([model_type in MODEL_TYPES for model_type in model_types])

        # 初始化文本模型列表
        text_models = []

        # 遍历 model_types 和 model_names，加载相应的文本模型
        for model_type, model_name in zip(model_types, model_names):
            # 根据 model_type 获取条件器类
            klass = CONDITION_CONFIG.get(model_type)
            # 实例化条件器
            model = klass(model_name, text_embed_pad_value = text_embed_pad_value)
            # 将条件器添加到文本模型列表中
            text_models.append(model)

        # 存储文本模型列表
        self.text_models = text_models
        # 获取每个文本模型的潜在维度
        self.latent_dims = [model.dim_latent for model in text_models]

        # 初始化条件函数列表
        self.conditioners = ModuleList([])

        # 存储隐藏层维度
        self.hidden_dims = hidden_dims
        # 条件函数的数量
        self.num_condition_fns = len(hidden_dims)
        # 将 hiddens_channel_first 转换为元组，长度与条件函数数量相同
        self.hiddens_channel_first = cast_tuple(hiddens_channel_first, self.num_condition_fns) # whether hiddens to be conditioned is channel first or last

        # 确保 hiddens_channel_first 的长度与条件函数数量相同
        assert len(self.hiddens_channel_first) == self.num_condition_fns

        # 存储条件 dropout 概率
        self.cond_drop_prob = cond_drop_prob

        # 计算总潜在维度
        total_latent_dim = sum(self.latent_dims)

        # 存储总潜在维度
        self.dim_latent = total_latent_dim

        # 计算文本嵌入主干 MLP 的输出维度
        mlp_stem_output_dim = total_latent_dim * text_embed_stem_dim_mult

        # 定义文本嵌入主干 MLP
        self.text_embed_stem_mlp = nn.Sequential(
            nn.Linear(total_latent_dim, mlp_stem_output_dim), # 线性变换
            nn.SiLU()
        )

        for hidden_dim in hidden_dims:
            # 为每个隐藏层维度添加 FiLM 条件函数
            self.conditioners.append(FiLM(mlp_stem_output_dim, hidden_dim))

        # 初始化 null 文本嵌入参数
        self.null_text_embed = nn.Parameter(torch.randn(total_latent_dim))

        # 注册一个缓冲区，用于跟踪设备信息
        self.register_buffer('_device_param', torch.tensor(0.), persistent = False)

    @property
    def device(self):
        """
        获取设备信息。

        返回:
            torch.device: 当前设备。
        """
        return next(self.buffers()).device

    @typecheck
    def embed_texts(self, texts: List[str]):
        """
        嵌入文本，返回嵌入后的文本张量。

        参数:
            texts (List[str]): 输入的文本列表。

        返回:
            Tensor: 嵌入后的文本张量。
        """
        device = self.device

        # 初始化文本嵌入列表
        text_embeds = []
        # 遍历每个文本模型，嵌入文本
        for text_model in self.text_models: 
            text_embed = text_model.embed_text(texts) # 嵌入文本
            text_embeds.append(text_embed.to(device)) # 将嵌入后的文本移动到设备，并添加到列表中
        
        # 将所有文本嵌入拼接起来，返回
        return torch.cat(text_embeds, dim = -1)

    @typecheck
    def forward(
        self,
        texts: List[str] | None = None,
        text_embeds: Tensor | None = None,
        cond_drop_prob = None,
        repeat_batch = 1,               # for robotic transformer edge case
    ) -> Tuple[
        Tuple[Callable, ...],
        TextCondReturn
    ]:
        """
        前向传播方法，应用文本条件化。

        参数:
            texts (Optional[List[str]], 可选): 输入的文本列表。默认值为 None。
            text_embeds (Optional[Tensor], 可选): 输入的文本嵌入张量。默认值为 None。
            cond_drop_prob (Optional[float], 可选): 条件 dropout 概率。默认值为 None。
            repeat_batch (int, 可选): 重复批次，用于 transformer 解码器的特殊情况。默认值为 1。

        返回:
            Tuple[Tuple[Callable, ...], TextCondReturn]: 返回条件函数元组和文本条件返回对象。
        """
        # 确保提供了 texts 或 text_embeds 中的一个
        assert exists(texts) ^ exists(text_embeds)

        if self.training:
            # 在训练模式下，使用默认的 cond_drop_prob
            cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        else:
            # 在非训练模式下，必须明确设置 cond_drop_prob
            assert exists(cond_drop_prob), 'when not training, cond_drop_prob must be explicitly set'

        if exists(texts):
            # 获取批次大小
            batch = len(texts)
        elif exists(text_embeds):
            # 获取批次大小
            batch = text_embeds.shape[0]

        if not exists(text_embeds):
            # 如果未提供 text_embeds，则嵌入文本
            text_embeds = self.embed_texts(texts)

        if cond_drop_prob > 0.:
            # 生成概率掩码
            prob_keep_mask = prob_mask_like((batch, 1), 1. - cond_drop_prob, device = self.device)
            # 重塑 null 文本嵌入
            null_text_embeds = rearrange(self.null_text_embed, 'd -> 1 d')

            text_embeds = torch.where(
                prob_keep_mask,
                text_embeds,
                null_text_embeds # 应用概率掩码，替换部分文本嵌入为 null 文本嵌入
            )

        # text embed mlp stem, as done in unet conditioning in guided diffusion
        # 文本嵌入主干 MLP，作为在 guided diffusion 中对 unet 进行条件化的步骤

        # 通过主干 MLP 处理文本嵌入
        text_embeds = self.text_embed_stem_mlp(text_embeds)

        # prepare the conditioning functions
        # 准备条件函数

        # 将 repeat_batch 转换为元组，长度与条件函数数量相同
        repeat_batch = cast_tuple(repeat_batch, self.num_condition_fns)

        # 初始化条件函数列表
        cond_fns = []

        # 遍历每个条件函数，生成条件函数元组
        for cond, cond_hiddens_channel_first, cond_repeat_batch in zip(self.conditioners, self.hiddens_channel_first, repeat_batch):
            # 重复文本嵌入
            cond_text_embeds = repeat(text_embeds, 'b ... -> (b r) ...', r = cond_repeat_batch)
            # 创建条件函数
            cond_fn = partial(cond, cond_text_embeds)

            # 选择通道优先或通道最后
            wrapper_fn = rearrange_channel_first if cond_hiddens_channel_first else rearrange_channel_last

            # 将条件函数添加到列表中
            cond_fns.append(wrapper_fn(cond_fn))

        # 返回条件函数元组和文本条件返回对象
        return tuple(cond_fns), TextCondReturn(text_embeds, None)


# cross attention text conditioner

@typecheck
class AttentionTextConditioner(Conditioner):
    """
    AttentionTextConditioner 类，使用交叉注意力机制进行文本条件化。

    参数:
        hidden_dims (Tuple[int, ...]): 隐藏层的维度，用于每个条件函数。
        model_types (str | Tuple[str, ...], 可选): 条件器类型，可以是 't5', 'clip', 'bge' 中的一个或多个。默认值为 't5'。
        model_names (str | Tuple[str, ...], 可选): 预训练模型的名称。默认值为 None。
        cond_drop_prob (float, 可选): 条件 dropout 概率。默认值为 0.0。
        hiddens_channel_first (bool | Tuple[bool, ...], 可选): 隐藏层是否为通道优先。默认值为 True。
        dim_latent (int, 可选): 潜在空间的维度。如果未提供，则默认为所有文本模型中最大的潜在维度。
        attn_dim_head (int, 可选): 交叉注意力模块中每个注意力头的维度。默认值为 64。
        attn_heads (int, 可选): 交叉注意力模块中注意力头的数量。默认值为 8。
        flash (bool, 可选): 是否使用 Flash Attention。默认值为 True。
        text_embed_pad_value (float, 可选): 文本嵌入填充值。默认值为 0.0。
    """
    def __init__(
        self,
        *,
        hidden_dims: Tuple[int, ...],
        model_types = 't5',
        model_names = None,
        cond_drop_prob = 0.,
        hiddens_channel_first = True,
        dim_latent = None,
        attn_dim_head = 64,
        attn_heads = 8,
        flash = True,
        text_embed_pad_value = 0.
    ):
        super().__init__()
        # 将 model_types 转换为元组
        model_types = cast_tuple(model_types)
        # 将 model_names 转换为元组，长度与 model_types 相同
        model_names = cast_tuple(model_names, length = len(model_types))

        # 确保 model_types 和 model_names 的长度相同
        assert len(model_types) == len(model_names)
        # 确保所有 model_types 都是有效的类型
        assert all([model_type in MODEL_TYPES for model_type in model_types])

        # 初始化文本模型列表
        text_models = []

        # 遍历 model_types 和 model_names，加载相应的文本模型
        for model_type, model_name in zip(model_types, model_names):
            # 根据 model_type 获取条件器类
            klass = CONDITION_CONFIG.get(model_type)
            # 实例化条件器
            model = klass(model_name, text_embed_pad_value = text_embed_pad_value)
            # 将条件器添加到文本模型列表中
            text_models.append(model)

        # 存储文本模型列表
        self.text_models = text_models

        # 初始化线性变换模块列表，用于将文本模型的输出转换为潜在空间
        self.to_latent_dims = ModuleList([])

        # 如果未提供 dim_latent，则设置为所有文本模型中最大的潜在维度
        dim_latent = default(dim_latent, max([model.dim_latent for model in text_models]))

        # 存储潜在空间的维度
        self.dim_latent = dim_latent

        # 为每个文本模型添加线性变换，将文本嵌入转换为潜在空间
        for model in text_models:
            self.to_latent_dims.append(nn.Linear(model.dim_latent, dim_latent))

        # 初始化条件函数列表
        self.conditioners = ModuleList([])

        # 存储隐藏层维度
        self.hidden_dims = hidden_dims
        # 条件函数的数量
        self.num_condition_fns = len(hidden_dims)
        # 将 hiddens_channel_first 转换为元组，长度与条件函数数量相同
        self.hiddens_channel_first = cast_tuple(hiddens_channel_first, self.num_condition_fns) # whether hiddens to be conditioned is channel first or last

        # 确保 hiddens_channel_first 的长度与条件函数数量相同
        assert len(self.hiddens_channel_first) == self.num_condition_fns

        # 存储文本嵌入填充值
        self.text_embed_pad_value = text_embed_pad_value
        # 存储条件 dropout 概率
        self.cond_drop_prob = cond_drop_prob

        # 为每个隐藏层维度添加交叉注意力条件函数
        for hidden_dim in hidden_dims:
            self.conditioners.append(CrossAttention(dim_latent, hidden_dim, flash = flash))

        # 注册一个缓冲区，用于跟踪设备信息
        self.register_buffer('_device_param', torch.tensor(0), persistent = False)

    @property
    def device(self):
        """
        获取设备信息。

        返回:
            torch.device: 当前设备。
        """
        return next(self.buffers()).device

    def embed_texts(self, texts: List[str]):
        """
        嵌入文本，返回嵌入后的文本张量。

        参数:
            texts (List[str]): 输入的文本列表。

        返回:
            Tensor: 嵌入后的文本张量。
        """
        device = self.device

        # 初始化文本嵌入列表
        text_embeds = []

        # 遍历每个文本模型和线性变换模块，嵌入文本并转换为潜在空间
        for text_model, to_latent in zip(self.text_models, self.to_latent_dims):
            # 嵌入文本
            text_embed = text_model.embed_text(texts, return_text_encodings = True)
            # 将嵌入后的文本移动到设备
            text_embed = text_embed.to(device)
            # 创建掩码，标记非填充位置
            mask = (text_embed != self.text_embed_pad_value).any(dim = -1)
            # 将文本嵌入转换为潜在空间
            text_embed = to_latent(text_embed)
            # 使用填充值填充掩码外的部分
            text_embed = text_embed.masked_fill(~mask[..., None], self.text_embed_pad_value)
            # 将嵌入后的文本添加到列表中
            text_embeds.append(text_embed)
        # 将所有文本嵌入拼接起来，返回
        return torch.cat(text_embeds, dim = -2)

    @typecheck
    def forward(
        self,
        texts: List[str] | None = None,
        text_embeds: Tensor | None = None,
        cond_drop_prob = None,
        repeat_batch = 1,  # for robotic transformer edge case
    ) -> Tuple[
        Tuple[Callable, ...],
        TextCondReturn
    ]:
        """
        前向传播方法，应用文本条件化。

        参数:
            texts (Optional[List[str]], 可选): 输入的文本列表。默认值为 None。
            text_embeds (Optional[Tensor], 可选): 输入的文本嵌入张量。默认值为 None。
            cond_drop_prob (Optional[float], 可选): 条件 dropout 概率。默认值为 None。
            repeat_batch (int, 可选): 重复批次，用于 transformer 解码器的特殊情况。默认值为 1。

        返回:
            Tuple[Tuple[Callable, ...], TextCondReturn]: 返回条件函数元组和文本条件返回对象。
        """
        # 确保提供了 texts 或 text_embeds 中的一个
        assert exists(texts) or exists(text_embeds)

        if exists(text_embeds) and exists(texts):
            # 如果同时存在 texts 和 text_embeds，则优先使用 text_embeds
            # 以便在第一个 epoch 中缓存文本嵌入
            texts = None

        if self.training:
            # 在训练模式下，使用默认的 cond_drop_prob
            cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        else:
            # 在非训练模式下，必须明确设置 cond_drop_prob
            assert exists(cond_drop_prob), 'when not training, cond_drop_prob must be explicitly set'

        if exists(texts):
            # 获取批次大小
            batch = len(texts)

        elif exists(text_embeds):
            # 获取批次大小
            batch = text_embeds.shape[0]

        if not exists(text_embeds):
            # 如果未提供 text_embeds，则嵌入文本
            text_embeds = self.embed_texts(texts)

        # 创建掩码，标记非填充位置
        mask = (text_embeds != self.text_embed_pad_value).any(dim = -1)

        if cond_drop_prob > 0.:
            # 生成概率掩码
            prob_keep_mask = prob_mask_like((batch, 1), 1. - cond_drop_prob, device = self.device)
            # 应用概率掩码
            mask = mask & prob_keep_mask

        # prepare the conditioning functions

        # 将 repeat_batch 转换为元组，长度与条件函数数量相同
        repeat_batch = cast_tuple(repeat_batch, self.num_condition_fns)

        # 初始化条件函数列表
        cond_fns = []

        # 遍历每个条件函数，生成条件函数元组
        for cond, cond_hiddens_channel_first, cond_repeat_batch in zip(self.conditioners, self.hiddens_channel_first, repeat_batch):
            # 重复文本嵌入
            cond_text_embeds = repeat(text_embeds, 'b ... -> (b r) ...', r = cond_repeat_batch)
            # 重复掩码
            cond_mask = repeat(mask, 'b ... -> (b r) ...', r = cond_repeat_batch) if exists(mask) else None
            # 创建条件函数
            cond_fn = partial(cond, cond_text_embeds, mask = cond_mask)
            # 选择通道优先或通道最后
            wrapper_fn = rearrange_channel_first if cond_hiddens_channel_first else rearrange_channel_last
            # 将条件函数添加到列表中
            cond_fns.append(wrapper_fn(cond_fn))
        # 返回条件函数元组和文本条件返回对象
        return tuple(cond_fns), TextCondReturn(text_embeds, mask)


# return raw text embedding

class TextEmbeddingReturner(Conditioner):
    """
    TextEmbeddingReturner 类，用于返回原始的文本嵌入，不进行任何条件化操作。

    参数:
        dim_latent (int, 可选): 潜在空间的维度。如果未提供，则默认为所有文本模型中最大的潜在维度。
        hidden_dims (Tuple[int, ...], 可选): 隐藏层的维度。默认值为空元组。
        model_types (str | Tuple[str, ...], 可选): 条件器类型，可以是 't5', 'clip', 'bge' 中的一个或多个。默认值为 't5'。
        model_names (str | Tuple[str, ...], 可选): 预训练模型的名称。默认值为 None。
        model_kwargs (dict, 可选): 其他传递给文本模型的参数。默认值为空字典。
        cond_drop_prob (float, 可选): 条件 dropout 概率。默认值为 0.0。
        text_embed_pad_value (float, 可选): 文本嵌入填充值。默认值为 0.0。
    """
    @typecheck
    def __init__(
        self,
        *,
        dim_latent = None,
        hidden_dims: Tuple[int, ...] = (),
        model_types = 't5',
        model_names = None,
        model_kwargs: dict = dict(),
        cond_drop_prob = 0.,
        text_embed_pad_value = 0.
    ):
        super().__init__()
        # 将 model_types 转换为元组
        model_types = cast_tuple(model_types)
        # 将 model_names 转换为元组，长度与 model_types 相同
        model_names = cast_tuple(model_names, length = len(model_types))
        # 将 model_kwargs 转换为元组，长度与 model_types 相同
        model_kwargs = cast_tuple(model_kwargs, length = len(model_types))

        # 确保 model_types, model_names 和 model_kwargs 的长度相同
        assert len(model_types) == len(model_names) == len(model_kwargs)
        # 确保所有 model_types 都是有效的类型
        assert all([model_type in MODEL_TYPES for model_type in model_types])

        # 初始化文本模型列表
        text_models = []

        # 遍历 model_types, model_names 和 model_kwargs，加载相应的文本模型
        for model_type, model_name, model_kwarg in zip(model_types, model_names, model_kwargs):
            # 根据 model_type 获取条件器类
            klass = CONDITION_CONFIG.get(model_type)
            # 实例化条件器
            model = klass(model_name, text_embed_pad_value = text_embed_pad_value, **model_kwarg)
            # 将条件器添加到文本模型列表中
            text_models.append(model)

        # 存储文本模型列表
        self.text_models = text_models
        # 存储文本嵌入填充值
        self.text_embed_pad_value = text_embed_pad_value

        # 初始化线性变换模块列表，用于将文本模型的输出转换为潜在空间
        self.to_latent_dims = ModuleList([])

        # 如果未提供 dim_latent，则设置为所有文本模型中最大的潜在维度
        dim_latent = default(dim_latent, max([model.dim_latent for model in text_models]))

        # 存储潜在空间的维度
        self.dim_latent = dim_latent

        # 为每个文本模型添加线性变换，将文本嵌入转换为潜在空间
        for model in text_models:
            self.to_latent_dims.append(nn.Linear(model.dim_latent, dim_latent))

        # 初始化条件函数列表
        self.conditioners = ModuleList([])

        # 存储条件 dropout 概率
        self.cond_drop_prob = cond_drop_prob

        for hidden_dim in hidden_dims:
            # 为每个隐藏层维度添加恒等变换条件函数
            self.conditioners.append(nn.Identity())

        # 注册一个缓冲区，用于跟踪设备信息
        self.register_buffer('_device_param', torch.tensor(0), persistent = False)

    @property
    def device(self):
        """
        获取设备信息。

        返回:
            torch.device: 当前设备。
        """
        return next(self.buffers()).device

    @typecheck
    def embed_texts(self, texts: List[str]):
        """
        嵌入文本，返回嵌入后的文本张量。

        参数:
            texts (List[str]): 输入的文本列表。

        返回:
            Tensor: 嵌入后的文本张量。
        """
        device = self.device

        # 初始化文本嵌入列表
        text_embeds = []
        # 遍历每个文本模型和线性变换模块，嵌入文本并转换为潜在空间
        for text_model, to_latent in zip(self.text_models, self.to_latent_dims):
            # 嵌入文本
            text_embed = text_model.embed_text(texts, return_text_encodings = True)
            # 将嵌入后的文本移动到设备
            text_embed = text_embed.to(device)
            # 创建掩码，标记非填充位置
            mask = (text_embed != self.text_embed_pad_value).any(dim = -1)
            # 将文本嵌入转换为潜在空间
            text_embed = to_latent(text_embed)
            # 使用填充值填充掩码外的部分
            text_embed = text_embed.masked_fill(~mask[..., None], self.text_embed_pad_value)
            # 将嵌入后的文本添加到列表中
            text_embeds.append(text_embed)
        # 将所有文本嵌入拼接起来，返回
        return torch.cat(text_embeds, dim = -2)

    @typecheck
    def forward(
        self,
        texts: List[str] | None = None,
        text_embeds: Tensor | None = None,
        cond_drop_prob = None
    ) -> Tuple[
        Tuple[Callable, ...],
        TextCondReturn
    ]:
        """
        前向传播方法，返回原始的文本嵌入。

        参数:
            texts (Optional[List[str]], 可选): 输入的文本列表。默认值为 None。
            text_embeds (Optional[Tensor], 可选): 输入的文本嵌入张量。默认值为 None。
            cond_drop_prob (Optional[float], 可选): 条件 dropout 概率。默认值为 None。

        返回:
            Tuple[Tuple[Callable, ...], TextCondReturn]: 返回条件函数元组和文本条件返回对象。
        """
        # 确保提供了 texts 或 text_embeds 中的一个
        assert exists(texts) ^ exists(text_embeds)

        if self.training:
            # 在训练模式下，使用默认的 cond_drop_prob
            cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        else:
            # 在非训练模式下，必须明确设置 cond_drop_prob
            assert exists(cond_drop_prob), 'when not training, cond_drop_prob must be explicitly set'

        if exists(texts):
            # 获取批次大小
            batch = len(texts)

        elif exists(text_embeds):
            # 获取批次大小
            batch = text_embeds.shape[0]

        if not exists(text_embeds):
            # 如果未提供 text_embeds，则嵌入文本
            text_embeds = self.embed_texts(texts)

        # 创建掩码，标记非填充位置
        mask = (text_embeds != self.text_embed_pad_value).any(dim = -1)

        if cond_drop_prob > 0.:
            # 生成概率掩码
            prob_keep_mask = prob_mask_like((batch, 1), 1. - cond_drop_prob, device = self.device)
            # 应用概率掩码
            mask = mask & prob_keep_mask

        # 返回条件函数元组和文本条件返回对象
        return tuple(self.conditioners), TextCondReturn(text_embeds, mask)
