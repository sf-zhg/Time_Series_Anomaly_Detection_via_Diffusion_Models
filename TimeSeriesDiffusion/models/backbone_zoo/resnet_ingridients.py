import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    class for 1d convolution as a block consisting of a convolution,
    group normalization and activation layer.
    """

    def __init__(self, dim, dim_out, groups=1):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor):
        """
        pass 1d series through convolution block. output channels depends on
        the dim_out
        """
        x = self.proj(x)
        x = self.norm(x)

        x = self.act(x)
        return x
