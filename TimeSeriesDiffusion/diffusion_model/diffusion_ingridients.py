import torch

from torch import Tensor, nn
from typing import Tuple


from TimeSeriesDiffusion.diffusion_model.diff_utils import extract
from TimeSeriesDiffusion.diffusion_model.scheduler.get_schedule import Schedule
from TimeSeriesDiffusion.noise.get_noise import get_noise
from TimeSeriesDiffusion.get_loss import get_loss


class DiffusionIngridients:
    def __init__(
        self,
        model: nn.Module,
        time_steps: int,
        beta_lower: float,
        beta_upper: float,
        scheduler: str,
        noise: str,
        loss: str,
        device: torch.device,
    ) -> None:
        """
        Cache diffusion input components for convenience and
        initialize diffusion processes as functions
        """

        self.model = model
        self.time_steps = time_steps
        self.beta_lower = beta_lower
        self.beta_upper = beta_upper
        self.scheduler = scheduler
        self.noise = noise
        self.loss = loss

        self.schedule = Schedule(
            self.time_steps, self.beta_lower, self.beta_upper, self.scheduler
        )

        self.device = device

    def q_sample(self, x_start: Tensor, t: Tensor, base_noise: Tensor) -> Tensor:
        """
        forward diffusion process. introduce noise to original series according
        to the appropriate strength dependent on the time step and the
        respective values at a certain timestep (in a batch obvs). returns
        a batch of diffused serieses (series what the fuck is the plural?) at
        the timesteps t.
        x, t --> x_t
        """

        root_prod_alphas_t = extract(self.schedule.root_prod_alphas, t, x_start.shape)

        root_one_minus_prod_alphas_t = extract(
            self.schedule.root_one_minus_prod_alphas, t, x_start.shape
        )

        return root_prod_alphas_t * x_start + root_one_minus_prod_alphas_t * base_noise

    def recon_from_noise(self, x_t: Tensor, t: Tensor, base_noise: Tensor) -> Tensor:
        """
        take noisy series x_t and reconstruct x by substracting the noise
        according to variance schedule. backward diffusion process
        """

        root_inv_prod_alphas_t = extract(
            self.schedule.root_inv_prod_alphas, t, x_t.shape
        )

        root_inv_prod_alphas_t_minus_one = extract(
            self.schedule.root_inv_prod_alphas_minus_one, t, x_t.shape
        )

        return (
            root_inv_prod_alphas_t * x_t - root_inv_prod_alphas_t_minus_one * base_noise
        )

    def q_posterior(
        self, x_start: Tensor, x_t: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        calc posterior mean, varriance and log variance for further
        computation down the line
        """

        posterior_var_t = extract(self.schedule.posterior_var, t, x_t.shape)

        posterior_log_var_t = extract(self.schedule.posterior_log_var, t, x_t.shape)

        posterior_mean_1_t = extract(self.schedule.posterior_mean_1, t, x_t.shape)

        posterior_mean_2_t = extract(self.schedule.posterior_mean_2, t, x_t.shape)

        post_mean_t = posterior_mean_1_t * x_start + posterior_mean_2_t * x_t

        return (post_mean_t, posterior_var_t, posterior_log_var_t)

    def p_mean_var(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        calc mean, variance and log variance for backward diffusion process
        """

        pred_noise = self.model(x=x, time=t)

        recon_x = self.recon_from_noise(x_t=x, t=t, base_noise=pred_noise)

        return self.q_posterior(x_start=recon_x, x_t=x, t=t)

    @torch.no_grad()
    def p_sample(self, x: Tensor, t: Tensor) -> Tensor:
        """
        get sample of size batch_size from backward diffusion process
        """

        mean, _, log_var = self.p_mean_var(x=x, t=t)

        base_noise = get_noise(x=x, noise=self.noise)

        if t[0] == 0:
            return mean
        else:
            return mean + torch.exp((0.5 * log_var)) * base_noise

    def p_loss(self, model: nn.Module, x_start: Tensor, t: Tensor) -> Tensor:
        """
        training with loss in mini_batch
        """

        base_noise = get_noise(x=x_start, noise=self.noise)

        noisy_x = self.q_sample(x_start=x_start, t=t, base_noise=base_noise)

        pred_noise = self.model(x=noisy_x, time=t)

        losses = get_loss(loss=self.loss)

        return losses(base_noise, pred_noise)
