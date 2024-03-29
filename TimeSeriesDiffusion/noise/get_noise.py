from TimeSeriesDiffusion.noise.simplex_noise import SimplexNoise
from TimeSeriesDiffusion.noise.brownian_noise import BrownianNoise

import torch
from torch import Tensor


def get_noise(x: Tensor, noise: str) -> Tensor:
    """
    get tensor of noise of choice
    """
    if noise == "gaussian":
        return torch.randn_like(x)

    elif noise == "simplex":
        Noise = SimplexNoise(
            octaves=6, persistence=0.8, frequency=1 / 64, lacunarity=2.0
        )
        return Noise(x)

    elif noise == "brownian":
        Noise = BrownianNoise(
            octaves=6, persistence=0.8, frequency=1 / 64, lacunarity=2.0
        )
        return Noise(x)

    else:
        raise ValueError(
            "choice of noise does not exist in code, you fucked up. implement or choose between existing noises"
        )
