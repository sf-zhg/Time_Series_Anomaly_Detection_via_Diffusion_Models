import pytest
import torch as torch

from TimeSeriesDiffusion.models.unet import Unet
from TimeSeriesDiffusion.models.convnet import ConvNetWithTimeEmbeddings


class ShapeTestNetworks:

    def setup_method(self):
        self.x_in = torch.randn((4, 8)).unsqueeze(1)
        self.time = torch.randint(0, 5, (4,))

    def test_unet_shape(self):
        model = Unet(dim=8)
        x_out = model(self.x_in, self.time)
        assert self.x_in.shape == x_out.shape

    def test_convnet_shape(self):
        model = ConvNetWithTimeEmbeddings(dim=8)
        x_out = model(self.x_in, self.time)
        assert self.x_in.shape == x_out.shape
