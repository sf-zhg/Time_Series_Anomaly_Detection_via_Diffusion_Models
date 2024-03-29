import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from TimeSeriesDiffusion.models.unet_utils import SinusoidalPositionEmbeddings


class ConvNetWithTimeEmbeddings(nn.Module):
    """
    class for a convolutional network with time embeddings.
    """

    def __init__(
        self,
        dim: int,
        in_channels: int = 1,
        out_channels: int = 32,
        conv_channels: Tuple[int] = (32, 64, 128),
    ):
        """
        initialize the convolutional network with time embeddings.
        """
        super(ConvNetWithTimeEmbeddings, self).__init__()
        self.time_dim = dim * 4
        self.conv1 = nn.Conv1d(in_channels + self.time_dim, out_channels, 1)
        self.conv_layers = nn.ModuleList()
        for out_channel in conv_channels:
            self.conv_layers.append(nn.Conv1d(out_channels, out_channel, 1))
            out_channels = out_channel
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, self.time_dim),
            nn.ReLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.final_conv = nn.Conv1d(self.time_dim, in_channels, 1)

    def forward(self, x: Tensor, time: Tensor) -> Tensor:
        """
        pass in the convolutional network with time embeddings.
        """
        t_emb = self.time_mlp(time)
        t_emb = t_emb.unsqueeze(2).expand(-1, -1, x.size(2))

        x = torch.cat((x, t_emb), dim=1)

        x = self.conv1(x)
        for conv_layer in self.conv_layers:
            x = nn.functional.relu(conv_layer(x))

        x = self.final_conv(x)

        return x
