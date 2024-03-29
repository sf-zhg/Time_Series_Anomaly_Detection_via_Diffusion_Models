import torch
import numpy as np
from numba import njit
from torch import Tensor
from typing import Tuple, Optional

MAX_SEED = np.iinfo(np.int32).max
MIN_SEED = np.iinfo(np.int32).min


@njit
def rnd1(x: int) -> np.ndarray:
    return np.random.randint(MIN_SEED, MAX_SEED, size=x)


@njit
def rnd2(x: int) -> np.ndarray:
    return np.arange(x)


def brownian_motion(
    shape: Tuple,
    bs: int,
    octaves: int = 1,
    persistence: float = 0.5,
    amplitude: float = 1,
    lacunarity: float = 0.5,
    perms: Optional[np.ndarray] = None,
) -> np.ndarray:

    rndf = rnd2
    if perms is None:
        rndf = rnd1
        perms = np.array([np.random.permutation(512) for _ in range(octaves)])

    bs, signal_shape = shape[0], shape[-2:]

    @njit(cache=True, parallel=True)
    def brownian(bs, signal_shape, octaves, persistence, amplitude, lacunarity, perms):

        noise = np.zeros((bs, signal_shape[0], signal_shape[1]))

        x = np.arange(signal_shape[0])
        y = np.arange(signal_shape[1])

        # Further decorrelation through x, y offset
        xr = rndf(octaves)
        yr = rndf(octaves)

        frequencies = np.ones(octaves) * (1 / signal_shape[0])
        frequencies[1:] = (1 / signal_shape[0]) * np.cumprod(
            lacunarity * np.ones(octaves - 1)
        )

        amplitudes = np.ones(octaves) * amplitude
        amplitudes[1:] = amplitude * np.cumprod(persistence * np.ones(octaves - 1))

        for i in range(octaves):
            noise += (
                amplitudes[i]
                * np.interp(
                    x,
                    (np.linspace(0, 1, perms[i].size) + xr[i]) % 1,
                    perms[i],
                    left=perms[i][-1],
                    right=perms[i][0],
                )[:, None]
            )

        for i in range(octaves):
            noise += (
                amplitudes[i]
                * np.interp(
                    y,
                    (np.linspace(0, 1, perms[i].size) + yr[i]) % 1,
                    perms[i],
                    left=perms[i][-1],
                    right=perms[i][0],
                )[None, :]
            )

        return noise

    return brownian(
        bs, signal_shape, octaves, persistence, amplitude, lacunarity, perms
    )


class BrownianNoise:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, x_like: Tensor) -> Tensor:
        return self._get_noise(x_like.shape, **self.kwargs).to(x_like.device)

    @staticmethod
    def _get_noise(shape: Tuple, **kwargs) -> Tensor:
        bs = shape[0]
        signal_shape = shape[-2:]
        return torch.from_numpy(
            brownian_motion(tuple(signal_shape), int(bs), **kwargs)[:, None, :, :]
        ).float()
