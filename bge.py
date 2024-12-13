from typing import List
from beartype import beartype

import torch
import transformers 
from transformers import AutoTokenizer, AutoModel, AutoConfig
transformers.logging.set_verbosity_error()

 
class BGEAdapter():
    """
    BGEAdapter 类，用于适配 BAAI 的 BGE 模型，并提供文本嵌入功能。

    参数:
        name (str): BGE 模型的名称。默认值为 'BAAI/bge-base-en-v1.5'。
        text_embed_pad_value (float, 可选): 文本嵌入填充值。默认值为 0.0。
    """
    def __init__(
        self,
        name,
        text_embed_pad_value = 0.
    ):
        # 设置默认的 BGE 模型名称
        name = 'BAAI/bge-base-en-v1.5'
        # 从预训练模型加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(name)
        # 从预训练模型加载 BGE 模型
        model = AutoModel.from_pretrained(name)
        # 从预训练模型加载配置
        self.Config = AutoConfig.from_pretrained(name)
        
        if torch.cuda.is_available():
            model = model.to("cuda")  
        
        # 存储模型名称
        self.name =  name
        # 存储模型
        self.model = model
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
        # 返回模型的隐藏层大小
        return self.Config.hidden_size

    @property
    def max_text_len(self):
        """
        获取最大文本长度。

        返回:
            int: 最大文本长度。
        """
        # 返回最大文本长度
        return 512

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
        if output_device is None:
            # 如果未提供输出设备，则使用模型的设备
            output_device = self.model.device
        elif output_device != self.model.device:
            # 如果输出设备与模型的设备不同，则将模型移动到输出设备
            self.model = self.model.to(output_device) 
        
        # 对文本进行编码，并移动到输出设备
        encoded_input  = self.tokenizer(texts, 
                                        padding=True, # 填充文本
                                        truncation=True, # 截断文本
                                        return_tensors='pt' # 返回 PyTorch 张量
                                        ).to(output_device)
        
        # 将模型设置为评估模式
        self.model.eval()
         
        with torch.no_grad():
            # 前向传播，获取模型输出
            model_output = self.model(**encoded_input)  
            
        if not return_text_encodings: 
            # 如果不需要返回文本编码，则返回 CLS 嵌入，并进行 L2 归一化
            # 获取 CLS 嵌入
            sentence_embeddings = model_output[0][:, 0]
            # L2 归一化
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            # 返回归一化后的 CLS 嵌入
            return sentence_embeddings  # Return normalized CLS embedding
        
        # 如果需要返回文本编码
        # 获取最后一个隐藏层的输出
        encoded_text = model_output.last_hidden_state.to(output_device)
        # 获取注意力掩码
        attn_mask = encoded_input.attention_mask.bool()
        
        # 使用填充值填充掩码外的部分
        encoded_text = encoded_text.masked_fill_(~attn_mask[..., None], self.text_embed_pad_value)

        # 返回文本编码
        return encoded_text
