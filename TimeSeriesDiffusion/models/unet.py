from functools import partial

import torch
from torch import nn, Tensor

from TimeSeriesDiffusion.models.backbone_zoo.resnet import ResnetBlock

from TimeSeriesDiffusion.models.backbone_zoo.attention import (
    LinearAttention1D,
    Attention1D,
)
from TimeSeriesDiffusion.models.unet_utils import (
    SinusoidalPositionEmbeddings,
    Residual,
    PreNorm,
    Downsample1D,
    Upsample1D,
)

from typing import Tuple


class Unet(nn.Module):
    """
    class for u net, returns tensor of same shape as input
    """

    def __init__(
        self,
        dim: int,
        init_dim: int = None,
        out_dim: int = None,
        dim_mults: Tuple = (1, 2, 4, 8),
        channels: int = 1,
        self_condition: bool = False,
        resnet_block_groups: int = 1,
    ):
        """
        initialize unet
        """
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = init_dim if init_dim is not None else dim // 3 * 2
        self.init_conv = nn.Conv1d(input_channels, init_dim, 1, padding=0)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention1D(dim_out))),
                        (
                            Downsample1D(dim_out)
                            if not is_last
                            else nn.Conv1d(dim_out, dim_out, 3, padding=1)
                        ),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention1D(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention1D(dim_in))),
                        (
                            Upsample1D(dim_in)
                            if not is_last
                            else nn.Conv1d(dim_in, dim_out, 3, padding=1)
                        ),
                    ]
                )
            )

        self.out_dim = out_dim if out_dim is not None else channels

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x: Tensor, time: Tensor, x_self_cond: Tensor = None) -> Tensor:
        """
        pass through unet
        """
        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)

            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        # print('t shape:', t.shape)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            # h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # print(f"size of x: {x.size()}")
        # print(f"size of h tensor: {h[-1].size()}")

        for block1, block2, attn, upsample in self.ups:
            # print(f"size of x: {x.size()}")
            # print(f"size of h tensor: {h[-1].size()}")
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            # x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
