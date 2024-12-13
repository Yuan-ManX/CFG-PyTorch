from environs import Env
from beartype import beartype
from beartype.door import is_bearable


# 环境配置

env = Env()  # 创建环境实例
env.read_env()  # 读取环境变量


# 函数

def always(value):
    """
    创建一个总是返回固定值的函数。

    参数:
        value: 需要返回的固定值。

    返回:
        callable: 一个总是返回固定值的函数。
    """
    def inner(*args, **kwargs):
        return value  # 无论输入如何，总是返回固定值
    return inner  # 返回内部函数


def identity(t):
    """
    恒等函数，返回输入值不变。

    参数:
        t: 输入值。

    返回:
        Any: 输入值 t。
    """
    return t  # 返回输入值


should_typecheck = env.bool('TYPECHECK', False)  # 从环境变量中读取 'TYPECHECK'，如果未设置则默认为 False


# 根据环境变量决定是否启用类型检查
# 如果 should_typecheck 为 True，则使用 beartype 进行类型检查
# 否则，使用 identity 函数，即不进行任何类型检查
typecheck = beartype if should_typecheck else identity


# 根据环境变量决定是否启用 beartype 的 isinstance 检查
# 如果 should_typecheck 为 True，则使用 is_bearable 进行类型检查
# 否则，使用 always(True) 函数，即总是返回 True，不进行类型检查
beartype_isinstance = is_bearable if should_typecheck else always(True)


# 导出接口

__all__ = [
    typecheck,  # 导出类型检查函数
    beartype_isinstance  # 导出 beartype 的 isinstance 检查函数
]
