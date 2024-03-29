import torch.nn as nn
from torch import einsum
from einops import rearrange
from torch import Tensor


class Attention1D(nn.Module):
    """
    class for attention in 1d
    """

    def __init__(self, dim, heads=4, dim_head=32):
        """
        initialize class
        """
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        pass x through module, get output of same shape as inout tensor
        """
        b, c, t = x.shape

        # lin trans
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) i -> b h c i", h=self.heads), qkv)
        q = q * self.scale

        # get attn score
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # apply attn to shit
        out = einsum("b h i j, b h d j -> b h i d", attn, v)

        # little data engineering because the shape is always fucked up
        out = rearrange(out, "b h i d -> b (h d) i", i=t)

        return self.to_out(out)


class LinearAttention1D(nn.Module):
    """
    linear class attention for 1d data
    """

    def __init__(self, dim, heads=4, dim_head=32):
        """
        initialize class
        """
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        pass x through module get tensor of same shape as input
        """
        b, c, t = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) i -> b h c i", h=self.heads), qkv)

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = einsum("b h d n, b h e n -> b h d e", k, v)

        out = einsum("b h d e, b h d n -> b h e n", context, q)

        out = rearrange(out, "b h c i -> b (h c) i")
        return self.to_out(out)
