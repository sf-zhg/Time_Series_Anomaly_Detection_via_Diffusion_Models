from typing import Optional, Tuple

import numpy as np
import opensimplex
import torch
from numba import njit
from torch import Tensor

MAX_SEED = np.iinfo(np.int32).max
MIN_SEED = np.iinfo(np.int32).min


class SimplexNoise:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, x_like: Tensor) -> Tensor:
        return self._get_noise(x_like.shape, **self.kwargs).to(x_like.device)

    @staticmethod
    def _get_noise(shape: Tuple, **kwargs) -> Tensor:
        bs = shape[0]
        signal_shape = shape[-1]
        return torch.from_numpy(
            batch_octavate1d(signal_shape, int(bs), **kwargs)[:, None, :]
        ).float()


@njit
def rnd1(x: int) -> np.ndarray:
    return np.random.randint(MIN_SEED, MAX_SEED, size=x)


@njit
def rnd2(x: int) -> np.ndarray:
    return np.arange(x)


def batch_octavate1d(
    signal_length: int,
    bs: int,
    octaves: int = 6,
    persistence: float = 0.5,
    frequency: float = 1 / 32,
    amplitude: float = 1,
    lacunarity: float = 0.5,
    perms: Optional[np.ndarray] = None,
) -> np.ndarray:

    rndf = rnd2
    if perms is None:
        rndf = rnd1
        seeds = rndf(octaves)
        perms = np.array([opensimplex.internals._init(seed) for seed in seeds])

    @njit(cache=True, parallel=True)
    def octavate(
        signal_length, bs, octaves, persistence, frequency, amplitude, lacunarity, perms
    ):

        noise = np.zeros((bs, signal_length))

        x = np.arange(signal_length)
        z = rndf(bs)

        xr = rndf(octaves)

        frequencies = np.ones(octaves) * frequency
        frequencies[1:] = frequency * np.cumprod(lacunarity * np.ones(octaves - 1))

        amplitudes = np.ones(octaves) * amplitude
        amplitudes[1:] = amplitude * np.cumprod(persistence * np.ones(octaves - 1))

        for i in range(octaves):
            noise += amplitudes[i] * opensimplex.internals._noise2a(
                (x + xr[i]) * frequencies[i],
                z,
                perms[i][0],
                perms[i][1],
            )
        return noise

    return octavate(
        signal_length, bs, octaves, persistence, frequency, amplitude, lacunarity, perms
    )
