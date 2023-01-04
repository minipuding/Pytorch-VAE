import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class VanillaVAE(BaseVAE):
    """
    这是最简单的VAE实现，也就是最原始VAE，原文链接：
    https://arxiv.org/pdf/1312.6114.pdf
    """

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        """
        初始化，主要完成模型的建立
        Args:
            in_channel: 输入通道数
            latent_dim: 隐藏层维度，这个指的是编码器输出的线性层的通道数；
            hidden_dims: 隐藏层维度，这个指的是卷积层中的隐藏层通道数
        """
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        # 默认的卷积层通道数为[32, 64, 128, 256, 512]
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        # 编码器由len(hidden_dims)个3x3卷积块组成，
        # 注意，每次卷积的stride=2，说明每一次卷积的同时完成一次降采样，扩大感受野。
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        # 科普：这里的``module``是一个list变量, ``*module``表示把list拆解开作为独立参数传递给``nn.Sequential``
        self.encoder = nn.Sequential(*modules)
        # 建立线性层，用于映射数据的分布，即均值和方差
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []
        # 建立线性层，将编码器的输出重整化后(重整化后面会描述)投射回原始的维度
        # 注意，输出维度是hidden_dims[-1]*4, 这里默认特征图降采样至[-1, hidden_dims[-1], 2, 2]的大小
        # 前面编码器映射分布中维度``*4``也是这个原因
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        # 将卷积隐藏层通道数的list反转，作为解码器的输出通道数
        hidden_dims.reverse()

        # 前len(hidden_dims) - 1层使用用反卷积，stride=2
        # 输出层在后面单独定义，所以编码器的卷积层数也是len(hidden_dims)
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        # 单独定义最后一层的输出层，由一次反卷积模块和一次卷积组成。
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=3,
                                      kernel_size=3, padding=1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # 假设输入的图像size是64x64，batch_size = 64, 卷积隐藏层通道数是默认的[32, 64, 128, 256, 512],
        # 经过``self.encoder``前向传播后输出的result形状为[64, 512, 2, 2]
        # 经过flatten后result形状为[64, 512*2*2]=[64, 2048]
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        # 假设``self.latent_dim``为128(这些假设的数值参考.configs/vae.yaml配置文件)
        # 则mu和log_var形状均为[64, 128]
        # 注意，这里希望投射到的不是方差var，而是log(var)，后面会通过exp恢复为var，
        # 原因是var必须是正值，但是经神经网络出来的数可能有负数，因此计算exp可以保证是正数。
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        # 按照原先``encode``中形状的假设，这里的z形状应为[64, 128]
        # z表示从编码器得到的分布中采样而来的数据，采样过程要用到参数重整化技巧，参见``reparameterize``
        # z经过``self.decoder_input``投射层后重新变成[64, 2048]
        result = self.decoder_input(z)
        # 将[64, 2048]的result reshape成[64, 512, 2, 2]
        result = result.view(-1, 512, 2, 2)
        # 经过decoder以后得到的输出应该是[64, 64, 32, 32]
        result = self.decoder(result)
        # 再经过最后一层输出得到[64, 1, 64, 64]
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param log_var: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        # 参数重整化，目的是解决采样步骤不可导的问题
        # 标准差通过exp(0.5 * log_var)还原，这里的0.5对应的其实是方差(var)开根号，具体来说就是：
        # exp(0.5 * log_var) = exp(0.5 * log(var)) = exp(log(\sqrt(var))) = exp(log(std)) = std
        std = torch.exp(0.5 * log_var)
        # 生成和std长度相等的随机向量，向量服从均值为0方差为1的标准高斯分布
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """
        前向传播过程：input -> encode -> reparameterize -> decode
        输出包括生成的结果、input, 均值， log方差
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        # 获取四个输出结果以及kl散度权重
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        # 计算重建损失，也就是mse损失，衡量输入和输出的差异
        recons_loss =F.mse_loss(recons, input)

        # 根据上面公式计算kl散度损失，用于衡量分布差异
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        # 计算总损失，由重建损失和kl损失组成
        loss = recons_loss + kld_weight * kld_loss
        # 返回所有损失函数
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        # 从标准高斯分布中采样，送入解码器生成图像
        # VAE的编码器只在训练时使用，作用是将数据集分布映射到标准高斯分布；在推理阶段，只需要解码器即可
        # 理论上一个训练好的解码器，只需要标准高斯分布的随机噪声作为输入即可。
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        # 生成器的作用就是做一次前向推理，只需要输出的生成图像即可。
        return self.forward(x)[0]