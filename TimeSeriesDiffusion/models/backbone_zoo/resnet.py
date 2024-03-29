import torch
from torch import nn, Tensor
from einops import rearrange

from TimeSeriesDiffusion.models.backbone_zoo.resnet_ingridients import ConvBlock


class ResnetBlock(nn.Module):
    """
    class for convolution net with residual connection and two convolution
    blocks (each conv -> group norm -> sigmoid linear unit)
    """

    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 1):
        """
        initialize class
        """
        super().__init__()

        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))

        self.conv_block1 = ConvBlock(dim, dim_out, groups=groups)
        self.conv_block2 = ConvBlock(dim_out, dim_out, groups=groups)

        if dim != dim_out:
            self.res_conv = nn.Conv1d(dim, dim_out, 1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        pass tensor
        """
        h = self.conv_block1(x)

        time_emb = self.mlp(t)

        h += rearrange(time_emb, "b c -> b c 1")

        h = self.conv_block2(h)

        return h + self.res_conv(x)
