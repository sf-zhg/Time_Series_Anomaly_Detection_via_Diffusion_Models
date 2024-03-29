from torch import Tensor

from typing import Tuple


def extract(a: Tensor, t: Tensor, x_shape: Tuple) -> Tensor:
    """get timestep values from tensor and reshape to target dim"""

    batch_size = t.shape[0]

    out = a.gather(-1, t)

    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
