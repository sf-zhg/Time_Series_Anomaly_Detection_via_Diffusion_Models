import math
from torch import Tensor
import torch
from torch import nn
from typing import Callable


def exists(x: any) -> bool:
    """
    functino to check whether a variable x exists and return true if x is not
    none, false otherwise
    """
    return x is not None


def default(val: any, d: any) -> any:
    """
    get value of val if val is not none, use it as to check values in network
    """
    if exists(val):
        return val
    return d() if callable(d) else d


def num_to_groups(num: int, divisor: int) -> list:
    """
    convert number into groups with a divisor, returns a
    group list
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Residual(nn.Module):
    """
    wrapper for residual block, returns sum of mapped inputs and
    original inputs in resnet fashion
    """

    def __init__(self, fn: Callable):
        """
        initialize class with a mapping function

        """
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        pass tensor
        """
        return self.fn(x, *args, **kwargs) + x


def Downsample1D(channels: int) -> nn.Conv1d:
    """
    get down sampling conv layer. halves dims of input series
    """
    return nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)


def Upsample1D(channels: int) -> nn.ConvTranspose1d:
    """
    up sampling conv layer, doubles input dims
    """
    return nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)


class SinusoidalPositionEmbeddings(nn.Module):
    """
    sinusoidal embeddings
    """

    def __init__(self, dim):
        """
        initialize class
        """
        super().__init__()
        self.dim = dim

    def forward(self, time: Tensor) -> Tensor:
        """
        pass tensor
        """
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class PreNorm(nn.Module):
    """
    prenorm class
    """

    def __init__(self, dim, fn):
        """
        initialize class
        """
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        pass x, norm shit
        """
        x = self.norm(x)
        return self.fn(x)
