import torch
from torch import nn

from cfg import TextConditioner, AttentionTextConditioner, classifier_free_guidance_class_decorator


# 初始化 TextConditioner 实例，用于文本条件化
text_conditioner = TextConditioner(
    model_types = 't5', # 使用 't5' 类型的文本模型   
    hidden_dims = (256, 512), # 隐藏层的维度，分别为 256 和 512
    hiddens_channel_first = False, # 隐藏层是否为通道优先，此处设置为 False，表示通道在后
    cond_drop_prob = 0.2  # 条件 dropout 概率为 20%，即有 20% 的概率会进行条件 dropout
).cuda()


# pass in your text as a List[str], and get back a List[callable]
# each callable function receives the hiddens in the dimensions listed at init (hidden_dims)
# 输入文本列表，并获取条件函数列表
# 每个条件函数接收隐藏层张量，隐藏层的维度与初始化时指定的 hidden_dims 对应

# 输入文本，获取两个条件函数
first_condition_fn, second_condition_fn = text_conditioner(['a cat chasing after a ball'])


# these hiddens will be in the direct flow of your model, say in a unet
# 这些隐藏层张量将直接用于你的模型中，例如在 UNet 中

# 生成第一个隐藏层张量，形状为 (1, 16, 256)，并移动到 GPU
first_hidden = torch.randn(1, 16, 256).cuda()
# 生成第二个隐藏层张量，形状为 (1, 32, 512)，并移动到 GPU
second_hidden = torch.randn(1, 32, 512).cuda()


# conditioned features
# 应用条件函数，获取条件化后的特征

# 使用第一个条件函数对第一个隐藏层张量进行条件化
first_conditioned = first_condition_fn(first_hidden)
# 使用第二个条件函数对第二个隐藏层张量进行条件化
second_conditioned = second_condition_fn(second_hidden)


# 输出条件化后的特征
print(first_conditioned.shape)  # 输出第一个条件化后的隐藏层张量的形状
print(second_conditioned.shape)  # 输出第二个条件化后的隐藏层张量的形状


# 使用交叉注意力进行文本条件化的条件器实例化
text_conditioner = AttentionTextConditioner(
    # 使用 't5' 和 'clip' 两种类型的文本模型
    model_types = ('t5', 'clip'), # 在 eDiff 论文中，他们同时使用了 T5 和 CLIP 模型以获得更好的结果（Balaji 等人）
    hidden_dims = (256, 512), # 隐藏层的维度，分别为 256 和 512
    cond_drop_prob = 0.2 # 条件 dropout 概率为 20%，即有 20% 的概率进行条件 dropout
)



# 定义一个简单的多层感知机（MLP）类
class MLP(nn.Module):
    """
    MLP 类，定义了一个简单的多层感知机模型。

    参数:
        dim (int): 输入数据的维度。
    """
    def __init__(
        self,
        dim
    ):
        """
        初始化 MLP 模型。

        参数:
            dim (int): 输入数据的维度。
        """
        super().__init__()
        # 定义输入投影层，使用线性变换将输入维度映射到 2 倍的维度，并应用 ReLU 激活函数
        self.proj_in = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU()) 
        # 定义中间投影层，使用线性变换将 2 倍的维度映射回原始维度，并应用 ReLU 激活函数
        self.proj_mid = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU())
        # 定义输出层，使用线性变换将原始维度映射到 1 维（用于二分类）
        self.proj_out = nn.Linear(dim, 1)

    def forward(
        self,
        data
    ):
        """
        前向传播方法，计算 MLP 的输出。

        参数:
            data (Tensor): 输入数据，形状为 (batch_size, dim)。

        返回:
            Tensor: MLP 的输出，形状为 (batch_size, 1)。
        """
        # 通过输入投影层，得到隐藏层输出 hiddens1
        hiddens1 = self.proj_in(data)
        # 通过中间投影层，得到隐藏层输出 hiddens2
        hiddens2 = self.proj_mid(hiddens1)
        # 通过输出层，得到最终的预测结果
        return self.proj_out(hiddens2)


# instantiate model and pass in some data, get (in this case) a binary prediction
# 实例化 MLP 模型，并传入一些数据，得到二分类预测结果

# 实例化 MLP 模型，输入维度为 256
model = MLP(dim = 256)

# 生成随机数据，形状为 (2, 256)
data = torch.randn(2, 256)

# 前向传播，得到预测结果
pred = model(data)

# 输出预测结果
print(pred)  



# 使用 classifier-free guidance 类装饰器装饰 MLP 类
@classifier_free_guidance_class_decorator
class MLP(nn.Module):
    """
    MLP 类，定义了一个多层感知机模型，并使用 classifier-free guidance 进行文本条件化。

    参数:
        dim (int): 输入数据的维度。
    """
    def __init__(self, dim):
        """
        初始化 MLP 模型，并设置文本条件化参数。

        参数:
            dim (int): 输入数据的维度。
        """
        super().__init__()

        # 定义输入投影层，使用线性变换将输入维度映射到 2 倍的维度，并应用 ReLU 激活函数
        self.proj_in = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU())
        # 定义中间投影层，使用线性变换将 2 倍的维度映射回原始维度，并应用 ReLU 激活函数
        self.proj_mid = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU())
        # 定义输出层，使用线性变换将原始维度映射到 1 维（用于二分类）
        self.proj_out = nn.Linear(dim, 1)

    def forward(
        self,
        inp,
        cond_fns # List[Callable] - (1) your forward function now receives a list of conditioning functions, which you invoke on your hidden tensors
    ):
        """
        前向传播方法，应用文本条件化并计算 MLP 的输出。

        参数:
            inp (Tensor): 输入数据。
            cond_fns (List[Callable]): 条件函数列表，按照 `hidden_dims` 设置的顺序返回。

        返回:
            Tensor: MLP 的输出，形状为 (batch_size, 1)。
        """
        # 解包条件函数列表，顺序与 TextConditioner 中设置的 hidden_dims 对应
        cond_hidden1, cond_hidden2 = cond_fns # conditioning functions are given back in the order of the `hidden_dims` set on the text conditioner

        # 通过输入投影层，得到隐藏层输出 hiddens1
        hiddens1 = self.proj_in(inp)
        # 使用第一个条件函数对 hiddens1 进行条件化 (使用 FiLM)
        hiddens1 = cond_hidden1(hiddens1) # (2) condition the first hidden layer with FiLM

        # 通过中间投影层，得到隐藏层输出 hiddens2
        hiddens2 = self.proj_mid(hiddens1)
        # 使用第二个条件函数对 hiddens2 进行条件化 (使用 FiLM)
        hiddens2 = cond_hidden2(hiddens2) # condition the second hidden layer with FiLM

        # 通过输出层，得到最终的预测结果
        return self.proj_out(hiddens2)


# instantiate your model - extra keyword arguments will need to be defined, prepended by `text_condition_`
# 实例化模型 - 需要定义额外的关键字参数，参数名以 `text_condition_` 开头

model = MLP(
    dim = 256, # 输入数据的维度
    text_condition_type = 'film', # 文本条件化类型，可以是 'film', 'attention', 或 'null'（无条件化）
    text_condition_model_types = ('t5', 'clip'), # 在本例中，使用 T5 和 OpenCLIP 进行文本条件化
    text_condition_hidden_dims = (512, 256), # 设置需要条件化的隐藏层维度，这里有两个隐藏层维度（dim * 2 和 dim，分别对应第一个和第二个投影层）
    text_condition_cond_drop_prob = 0.25  # 条件 dropout 概率，用于 classifier-free guidance。可以设置为 0.0 如果不需要条件 dropout，而只需要文本条件化
)


# now you have your input data as well as corresponding free text as List[str]
# 现在你有了输入数据以及对应的自由文本作为 List[str]

# 生成随机输入数据，形状为 (2, 256)
data = torch.randn(2, 256)
# 输入文本列表
texts = ['a description', 'another description']


# train your model, passing in your list of strings as 'texts'
# 训练你的模型，将文本列表作为 'texts' 参数传入

# 前向传播，得到预测结果
pred  = model(data, texts = texts)


# after much training, you can now do classifier free guidance by passing in a condition scale of > 1. !
# 经过大量训练后，你现在可以通过设置 cond_scale > 1 来进行 classifier-free guidance！

# 将模型设置为评估模式
model.eval()
# cond_scale 表示 classifier-free guidance 论文中的条件缩放因子
guided_pred = model(data, texts = texts, cond_scale = 3., remove_parallel_component = True)  

