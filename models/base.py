from .types_ import * # 为了方便，作者直接将所有用到的类型在``type_.py``文件中导入
from torch import nn
from abc import abstractmethod


class BaseVAE(nn.Module):
    """
    建立VAE基类，从成员函数可以看出，VAE主要由编码器``encode(r)``、解码器``decode(r)``、
    采样器``sample(r)``、生成器``generate``四个部分组成。
    """
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass



