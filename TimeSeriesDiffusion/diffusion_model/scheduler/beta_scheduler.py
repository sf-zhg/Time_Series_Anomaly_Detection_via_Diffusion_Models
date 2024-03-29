from abc import ABC
import torch
from torch import Tensor


class LinScheduler(ABC):
    """
    linear noise scheduler
    """

    def __init__(self, time_steps: int, beta_lower: float, beta_upper: float):

        self._t = time_steps
        self.beta_lower = beta_lower
        self.beta_upper = beta_upper

    def forward(self):
        return torch.linspace(self.beta_lower, self.beta_upper, self._t)


class CosScheduler(ABC):
    """
    cosine noise scheduler
    """

    def __init__(self, time_steps: int, beta_lower: float, beta_upper: float):

        self._t = time_steps
        self.beta_lower = beta_lower
        self.beta_upper = beta_upper

        self.steps = self._t + 1
        self._s = 0.008
        self._x = torch.linspace(0, self._t, self.steps)
        self.alpha_prod = (
            torch.cos(((self._x / self._t) + self._s) / (1 + self._s) * torch.pi * 0.5)
            ** 2
        )
        self.alpha_prod_vec = self.alpha_prod / self.alpha_prod[0]

        self.beta_vec = 1 - (self.alpha_prod_vec[1:] / self.alpha_prod_vec[:-1])

    def forward(self) -> Tensor:
        return torch.clip(self.beta_vec, 0.0001, 0.9999)


class QuadScheduler(ABC):
    """
    quadratic noise scheduler
    """

    def __init__(self, time_steps: int, beta_lower: float, beta_upper: float):

        self._t = time_steps
        self.beta_lower = beta_lower
        self.beta_upper = beta_upper

    def forward(self) -> Tensor:
        return torch.linspace(self.beta_lower**0.5, self.beta_upper**0.5, self._t) ** 2


class SigScheduler(ABC):
    """
    sigmoid noise scheduler
    """

    def __init__(self, time_steps: int, beta_lower: float, beta_upper: float):

        self._t = time_steps
        self.beta_lower = beta_lower
        self.beta_upper = beta_upper
        self.beta_vec = torch.linspace(-6, 6, self._t)

    def forward(self) -> Tensor:
        return (
            torch.sigmoid(self.beta_vec) * (self.beta_upper - self.beta_lower)
            + self.beta_upper
        )
