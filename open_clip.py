from beartype import beartype
from typing import List

import torch
from torch import nn, einsum
import torch.nn.functional as F

import open_clip


# constants
# 常量

# 默认的 CLIP 模型名称，使用 ViT-B-32 模型
DEFAULT_CLIP_NAME = 'ViT-B-32'
# 默认预训练的 CLIP 模型，使用在 LAION-400M 数据集上训练的 e32 版本
DEFAULT_PRETRAINED_CLIP = 'laion400m_e32'


# helper functions
# 辅助函数

# 检查一个值是否存在（不为 None）
def exists(val):
    """
    检查一个值是否存在（不为 None）。

    参数:
        val: 需要检查的值。

    返回:
        bool: 如果 val 不为 None，则返回 True；否则返回 False。
    """
    return val is not None


# 返回可选值或默认值
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


# 对张量 t 进行 L2 归一化
def l2norm(t):
    """
    对张量 t 进行 L2 归一化。

    参数:
        t (Tensor): 输入张量。

    返回:
        Tensor: 归一化后的张量。
    """
    return F.normalize(t, dim = -1)


# adapter
# OpenCLIP 适配器

class OpenClipAdapter():
    """
    OpenClipAdapter 类，用于适配 OpenCLIP 模型，并提供文本嵌入功能。

    参数:
        name (str, 可选): CLIP 模型的名称。默认值为 DEFAULT_CLIP_NAME ('ViT-B-32')。
        pretrained (str, 可选): 预训练的 CLIP 模型名称。默认值为 DEFAULT_PRETRAINED_CLIP ('laion400m_e32')。
        text_embed_pad_value (float, 可选): 文本嵌入填充值。默认值为 0.0。
        auto_move_clip_cuda (bool, 可选): 是否自动将 CLIP 模型移动到 CUDA 设备（如果可用）。默认值为 True。
    """
    def __init__(
        self,
        name = DEFAULT_CLIP_NAME,
        pretrained = DEFAULT_PRETRAINED_CLIP,
        text_embed_pad_value = 0.,
        auto_move_clip_cuda = True
    ):
        # 如果未提供 name，则使用默认的 CLIP 模型名称
        name = default(name, DEFAULT_CLIP_NAME)
        # 如果未提供 pretrained，则使用默认的预训练 CLIP 模型名称
        pretrained = default(pretrained, DEFAULT_PRETRAINED_CLIP)

        # 创建 CLIP 模型和预处理变换
        clip, _, preprocess = open_clip.create_model_and_transforms(name, pretrained = pretrained)

        if auto_move_clip_cuda and torch.cuda.is_available():
            # 如果启用了自动移动到 CUDA，并且 CUDA 可用，则将 CLIP 模型移动到 CUDA 设备
            clip = clip.cuda()

        # 存储 CLIP 模型
        self.clip = clip
        # 将 CLIP 模型设置为评估模式
        clip.eval()

        # 获取 CLIP 模型的 tokenizer
        self.tokenizer = open_clip.get_tokenizer(name)
        # 存储文本嵌入填充值
        self.text_embed_pad_value = text_embed_pad_value

        # 结束符 ID
        self.eos_id = 49407

        # 查找 'ln_final' 层
        text_attention_final = self.find_layer('ln_final')
        # 获取潜在空间的维度
        self._dim_latent = text_attention_final.weight.shape[0]

        # 注册前向钩子，用于获取文本编码
        self.handle = text_attention_final.register_forward_hook(self._hook)
        # 获取 CLIP 模型的最后一个预处理变换
        self.clip_normalize = preprocess.transforms[-1]
        # 初始化 cleared 标志
        self.cleared = False

    def find_layer(self,  layer):
        """
        查找指定的层。

        参数:
            layer (str): 层名称。

        返回:
            Optional[nn.Module]: 找到的层或 None。
        """
        # 获取 CLIP 模型的所有命名模块
        modules = dict([*self.clip.named_modules()])
        # 返回指定名称的层
        return modules.get(layer, None)

    def clear(self):
        """
        清除钩子。
        """
        if self.cleared:
            # 如果已经清除，则返回
            return
        # 清除钩子
        self.handle()

    def _hook(self, _, inputs, outputs):
        """
        前向钩子，用于存储文本编码。
        """
        # 存储文本编码
        self.text_encodings = outputs

    @property
    def dim_latent(self):
        """
        获取潜在空间的维度。

        返回:
            int: 潜在空间的维度。
        """
        return self._dim_latent

    @property
    def max_text_len(self):
        """
        获取最大文本长度。

        返回:
            int: 最大文本长度。
        """
        return 77

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
        # 获取 CLIP 模型所在的设备
        clip_device = next(self.clip.parameters()).device

        if not exists(output_device):
            # 如果未提供输出设备，则使用 CLIP 模型的设备
            output_device = clip_device

        if output_device != clip_device:
            # 如果输出设备与 CLIP 模型的设备不同，则将 CLIP 模型移动到输出设备
            self.clip = self.clip.to(output_device) 
        
        # 对文本进行 tokenize，并移动到输出设备
        texts = self.tokenizer(texts).to(output_device)
        # 计算最大文本长度
        max_length = (texts != 0).sum(dim=1).max().item()
        # 截断文本到最大文本长度
        texts = texts[..., :self.max_text_len]

        # 使用 CLIP 模型编码文本
        text_embeds = self.clip.encode_text(texts)

        # 截断文本到实际长度
        texts = texts[..., :max_length]

        if not return_text_encodings:
            # 如果不需要返回文本编码，则返回归一化后的文本嵌入
            return l2norm(text_embeds).to(output_device)

        # 找到结束符的位置
        is_eos_id = (texts == self.eos_id)
        # 创建一个掩码，排除结束符
        text_mask_excluding_eos = is_eos_id.cumsum(dim = -1) == 0
        # 填充掩码
        text_mask = F.pad(text_mask_excluding_eos, (1, -1), value = True)
        # 应用掩码
        text_mask = text_mask & (texts != 0)

        # 确保钩子未被清除
        assert not self.cleared

        # 获取文本编码
        text_encodings = self.text_encodings[:, :max_length]
        # 使用填充值填充掩码外的部分
        text_encodings = text_encodings.masked_fill(~text_mask[..., None], self.text_embed_pad_value)
        # 删除文本编码
        del self.text_encodings

        # 返回文本编码
        return text_encodings.float().to(output_device)
