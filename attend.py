from collections import namedtuple
from functools import wraps
from packaging import version

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat


# constants
# 常量

# 定义 EfficientAttentionConfig 命名元组，用于配置高效注意力机制
EfficientAttentionConfig = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])
# 参数说明：
# - enable_flash: 是否启用 Flash Attention（一种高效的注意力机制实现）。
# - enable_math: 是否启用数学运算优化。
# - enable_mem_efficient: 是否启用内存高效模式。


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


def once(fn):
    """
    装饰器，用于确保函数只被调用一次。

    参数:
        fn (callable): 需要装饰的函数。

    返回:
        callable: 装饰后的函数。
    """
    # 初始化调用标志
    called = False

    @wraps(fn)  # 保留原函数的元数据
    def inner(x):
        # 使用 nonlocal 关键字声明 called 为外部变量
        nonlocal called
        if called:
            # 如果已经调用过，则直接返回，不执行函数体
            return
        # 设置调用标志为 True
        called = True
        # 调用原函数
        return fn(x)
    # 返回装饰后的函数
    return inner

# 创建一个只打印一次的 print 函数
print_once = once(print)


# main class

class Attend(nn.Module):
    """
    Attend 类，实现了高效的注意力机制，包括 Flash Attention 和其他优化。

    参数:
        dropout (float, 可选): Dropout 概率。默认值为 0.0。
        causal (bool, 可选): 是否使用因果掩码。默认值为 False。
        flash (bool, 可选): 是否启用 Flash Attention。默认值为 False。
    """
    def __init__(
        self,
        dropout = 0.,
        causal = False,
        flash = False
    ):
        super().__init__()
        # 存储 Dropout 概率
        self.dropout = dropout
        # 定义 Dropout 层
        self.attn_dropout = nn.Dropout(dropout)

        # 存储因果掩码标志
        self.causal = causal
        # 注册一个缓冲区，用于存储掩码
        self.register_buffer("mask", None, persistent=False)

        # 存储 Flash Attention 标志
        self.flash = flash
        # 确保使用 PyTorch 2.0 或更高版本以支持 Flash Attention
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu
        # 确定 CPU 和 CUDA 的高效注意力配置

        # CPU 上启用 Flash Attention、数学运算优化和内存高效模式
        self.cpu_config = EfficientAttentionConfig(True, True, True)
        # 初始化 CUDA 配置
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            # 如果 CUDA 不可用或未启用 Flash Attention，则跳过
            return

        # 获取 CUDA 设备属性
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            # 如果是 A100 GPU，则启用 Flash Attention
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            # A100 上启用 Flash Attention，禁用数学运算优化和内存高效模式
            self.cuda_config = EfficientAttentionConfig(True, False, False)
        else:
            # 如果不是 A100 GPU，则启用数学运算优化和内存高效模式
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            # 非 A100 上启用数学运算优化和内存高效模式
            self.cuda_config = EfficientAttentionConfig(False, True, True)

    def get_mask(self, n, device):
        """
        获取因果掩码。

        参数:
            n (int): 序列长度。
            device (torch.device): 张量设备。

        返回:
            Tensor: 因果掩码，形状为 (n, n)。
        """
        if exists(self.mask) and self.mask.shape[-1] >= n:
            # 如果已有掩码且长度足够，则返回子掩码
            return self.mask[:n, :n]

        # 生成上三角掩码
        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        # 注册掩码到缓冲区
        self.register_buffer("mask", mask, persistent=False)
        # 返回掩码
        return mask

    def flash_attn(self, q, k, v, mask = None):
        """
        使用 Flash Attention 计算注意力。

        参数:
            q (Tensor): 查询张量，形状为 (batch_size, heads, q_len, d_k)。
            k (Tensor): 键张量，形状为 (batch_size, heads, k_len, d_k)。
            v (Tensor): 值张量，形状为 (batch_size, heads, k_len, d_v)。
            mask (Optional[Tensor], 可选): 掩码张量。默认值为 None。

        返回:
            Tensor: 注意力输出，形状为 (batch_size, heads, q_len, d_v)。
        """
        # 获取张量形状和设备信息
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        # Recommended for multi-query single-key-value attention by Tri Dao
        # 推荐用于多查询单键值注意力（由 Tri Dao 推荐）
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if k.ndim == 3:
            # 如果键张量维度为 3，则重复键张量以匹配多头注意力
            k = repeat(k, 'b ... -> b h ...', h = heads)

        if v.ndim == 3:
            # 如果值张量维度为 3，则重复值张量以匹配多头注意力
            v = repeat(v, 'b ... -> b h ...', h = heads)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L
        # 检查掩码是否存在并扩展到兼容的形状
        # 掩码的形状为 B L，因此需要扩展到 B H N L

        if exists(mask):
            if mask.ndim == 2:
                # 重塑掩码形状为 (batch_size, 1, 1, j)
                mask = rearrange(mask, 'b j -> b 1 1 j')
            # 扩展掩码以匹配多头注意力
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention
        # 检查是否存在兼容的设备以使用 Flash Attention

        # 根据设备选择配置
        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        # 使用 SDD 内核进行 Flash Attention
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v, # 查询、键和值张量
                attn_mask = mask, # 掩码
                dropout_p = self.dropout if self.training else 0.,  # Dropout 概率
                is_causal = self.causal # 是否为因果注意力
            )
        # 返回注意力输出
        return out

    def forward(self, q, k, v, mask = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        """
        前向传播方法，计算注意力。

        参数:
            q (Tensor): 查询张量，形状为 (batch_size, heads, q_len, d_k)。
            k (Tensor): 键张量，形状为 (batch_size, heads, k_len, d_k)。
            v (Tensor): 值张量，形状为 (batch_size, heads, k_len, d_v)。
            mask (Optional[Tensor], 可选): 掩码张量。默认值为 None。

        返回:
            Tensor: 注意力输出，形状为 (batch_size, heads, q_len, d_v)。
        """
        # 获取序列长度和设备
        n, device = q.shape[-2], q.device
        # 计算缩放因子
        scale = q.shape[-1] ** -0.5

        if self.flash:
            # 如果启用 Flash Attention，则使用 Flash Attention
            return self.flash_attn(q, k, v, mask = mask)
        
        # 定义键和值张量的 Einstein 符号
        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        # similarity
        # 计算相似度
        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # key padding mask
        # 键填充掩码
        if exists(mask):
            if mask.ndim == 2:
                # 重塑掩码形状为 (batch_size, 1, 1, j)
                mask = rearrange(mask, 'b j -> b 1 1 j')
            # 应用掩码
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # causal mask
        # 因果掩码
        if self.causal:
            # 获取因果掩码
            causal_mask = self.get_mask(n, device)
            # 使用因果掩码将未来的注意力分数设为负无穷大
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention
        # 计算注意力权重

        # 对相似度矩阵进行 softmax 操作，得到注意力权重
        attn = sim.softmax(dim=-1)
        # 对注意力权重应用 Dropout
        attn = self.attn_dropout(attn)

        # aggregate values
        # 使用注意力权重聚合值，得到输出
        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        return out
