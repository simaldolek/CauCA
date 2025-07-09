from abc import ABC
from torch.distributions import Uniform
import torch
from torch import Tensor


class MultiEnvNoise(ABC):
    def __init__(
        self,
        latent_dim: int,
        intervention_targets_per_env: Tensor,
        mean: float = 0.0,
        std: float = 1.0,
        shift: bool = False,
        shift_type: str = "mean",
    ) -> None:
        self.latent_dim = latent_dim
        self.intervention_targets = intervention_targets_per_env
        self.mean = mean
        self.std = std
        self.shift = shift
        assert shift_type in ["mean", "std", "both"], f"Invalid shift type: {shift_type}"
        self.shift_type = shift_type

    def sample(self, e: int, size: int = 1) -> Tensor:
        raise NotImplementedError()

    def log_prob(self, u: Tensor, e: int) -> Tensor:
        raise NotImplementedError()


class GaussianNoise(MultiEnvNoise):
    def __init__(
        self,
        latent_dim: int,
        intervention_targets_per_env: Tensor,
        shift: bool = False,
        shift_type: str = "mean",
        env_noise_scale: torch.Tensor = None
    ):
        super().__init__(
            latent_dim=latent_dim,
            intervention_targets_per_env=intervention_targets_per_env,
            shift=shift,
            shift_type=shift_type,
        )
        self.env_noise_scale = env_noise_scale
        self.mean = 0.0
        self.std = 1.0
        self.means_per_env, self.stds_per_env = self.setup_params(intervention_targets_per_env)

    def setup_params(self, intervention_targets_per_env: Tensor) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        means_per_env = {}
        stds_per_env = {}
        for e in range(intervention_targets_per_env.shape[0]):
            means = torch.ones(self.latent_dim) * self.mean
            stds = torch.ones(self.latent_dim) * self.std

            if self.shift and intervention_targets_per_env is not None:
                for i in range(self.latent_dim):
                    if intervention_targets_per_env[e][i] == 1:
                        if self.shift_type == "mean":
                            coin_flip = torch.randint(0, 2, (1,)).item()
                            factor = 2
                            means[i] = self.mean + factor * self.std if coin_flip else self.mean - factor * self.std
                        elif self.shift_type == "std":
                            coin_flip = torch.randint(0, 2, (1,)).item()
                            std_scaling = Uniform(0.25, 0.75).sample((1,)) if coin_flip == 0 else Uniform(1.25, 1.75).sample((1,))
                            stds[i] *= std_scaling

            means_per_env[e] = means
            stds_per_env[e] = stds

        return means_per_env, stds_per_env

    def sample(self, env: int, size: int) -> Tensor:
        base_noise = torch.randn(size, self.latent_dim)

        if not self.shift:
            return base_noise

        int_targets = self.intervention_targets[env]
        scale = self.env_noise_scale[env] if self.env_noise_scale is not None else 1.0

        if self.shift_type == "mean":
            mean_shift = int_targets.float()
            return base_noise + mean_shift

        elif self.shift_type == "std":
            std_shift = 1.0 + int_targets.float() * (scale - 1.0)
            return base_noise * std_shift

        elif self.shift_type == "both":
            mean_shift = int_targets.float()
            std_shift = 1.0 + int_targets.float() * (scale - 1.0)
            return (base_noise * std_shift) + mean_shift

        else:
            raise ValueError(f"Unknown shift_type: {self.shift_type}")

    def log_prob(self, u: Tensor, e: int) -> Tensor:
        return torch.distributions.Normal(
            self.means_per_env[e].unsqueeze(0).repeat(u.shape[0], 1),
            self.stds_per_env[e].unsqueeze(0).repeat(u.shape[0], 1),
        ).log_prob(u)

    

  #def sample(self, e: int, size: int = 1) -> Tensor:
    #    return torch.normal(
    #        self.means_per_env[e].unsqueeze(0).repeat(size, 1),
    #        self.stds_per_env[e].unsqueeze(0).repeat(size, 1),
    #    )
    
