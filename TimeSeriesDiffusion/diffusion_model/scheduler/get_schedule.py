import torch
import torch.nn.functional as F

from TimeSeriesDiffusion.diffusion_model.scheduler import beta_scheduler


class Schedule:
    """
    get scheduler and scheduling variables for easy access
    """

    def __init__(
        self, time_steps: int, beta_lower: float, beta_upper: float, scheduler: str
    ):

        self.time_steps = time_steps
        self.beta_lower = beta_lower
        self.beta_upper = beta_upper
        self.scheduler = scheduler
        self.beta_schedule = None

        if self.scheduler == "linear":
            self.beta_schedule = beta_scheduler.LinScheduler(
                time_steps=self.time_steps,
                beta_lower=self.beta_lower,
                beta_upper=self.beta_upper,
            )
        elif self.scheduler == "cosine":
            self.beta_schedule = beta_scheduler.CosScheduler(
                time_steps=self.time_steps,
                beta_lower=self.beta_lower,
                beta_upper=self.beta_upper,
            )
        elif self.scheduler == "quadratic":
            self.beta_schedule = beta_scheduler.QuadScheduler(
                time_steps=self.time_steps,
                beta_lower=self.beta_lower,
                beta_upper=self.beta_upper,
            )
        elif self.scheduler == "sigmoid":
            self.beta_schedule = beta_scheduler.SigScheduler(
                time_steps=self.time_steps,
                beta_lower=self.beta_lower,
                beta_upper=self.beta_upper,
            )
        else:
            raise ValueError(
                "scheduler not existing, choose linear, cosine, quadratic, sgimoid or implement new one"
            )

        self.betas = self.beta_schedule.forward()

        self.alphas = torch.ones_like(self.betas) - self.betas

        self.root_alphas = torch.sqrt(self.alphas)

        self.prod_alphas = torch.cumprod(self.alphas, axis=0)

        self.prod_alphas_prev = F.pad(self.prod_alphas[:-1], (1, 0), value=1.0)

        self.root_inv_alphas = torch.sqrt(1.0 / self.alphas)

        self.root_prod_alphas = torch.sqrt(self.prod_alphas)

        self.root_one_minus_prod_alphas = torch.sqrt(1.0 - self.prod_alphas)

        self.root_inv_prod_alphas = torch.sqrt(1.0 / self.prod_alphas)

        self.root_inv_prod_alphas_minus_one = torch.sqrt(1.0 / self.prod_alphas - 1)

        self.posterior_var = (
            self.betas * (1.0 - self.prod_alphas_prev) / (1.0 - self.prod_alphas)
        )

        self.posterior_log_var = torch.log(
            torch.maximum(self.posterior_var, torch.tensor(0.001))
        )

        self.posterior_mean_1 = (
            self.betas * torch.sqrt(self.prod_alphas_prev) / (1.0 - self.prod_alphas)
        )

        self.posterior_mean_2 = (
            (1.0 - self.prod_alphas_prev) * self.root_alphas / (1.0 - self.prod_alphas)
        )
