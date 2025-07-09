import numpy as np
import torch
from torch import Tensor

from data_generator.graph_sampler import sample_random_dag
from data_generator.mixing_function import NonlinearMixing, MixingFunction, LinearMixing
from data_generator.noise_generator import GaussianNoise, MultiEnvNoise
from data_generator.scm import LinearSCM, LocationScaleSCM, MultiEnvLatentSCM
from .mixing_function import LinearMixing, MixingFunction, NonlinearMixing
from .noise_generator import GaussianNoise, MultiEnvNoise
from .scm import LinearSCM, MultiEnvLatentSCM, LocationScaleSCM


class MultiEnvDGP:
    """
    Multi-environment data generating process (DGP).

    The DGP is defined by a latent structural causal model (SCM), a noise generator and a mixing function.
    This class is used to generate data from those three components.

    The latent SCM is a multi-environment SCM, i.e. it generates data for multiple environments which
    differ by interventions on some of the variables. The noise generator is also multi-environmental,
    i.e. it generates noise for multiple environments. The mixing function is a function that maps the
    latent variables to the observed variables. The mixing function is the same for all environments.

    Attributes
    ----------
    mixing_function: MixingFunction
        Mixing function.
    latent_scm: MultiEnvLatentSCM
        Multi-environment latent SCM.
    noise_generator: MultiEnvNoise
        Multi-environment noise generator.

    Methods
    -------
    sample(num_samples_per_env, intervention_targets_per_env) -> tuple[Tensor, ...]
        Sample from the DGP.
    """

    def __init__(
        self,
        mixing_function: MixingFunction,
        latent_scm: MultiEnvLatentSCM,
        noise_generator: MultiEnvNoise,
    ) -> None:
        self.mixing_function = mixing_function
        self.latent_scm = latent_scm
        self.noise_generator = noise_generator
        self.adjacency_matrix = self.latent_scm.adjacency_matrix

    def sample(
        self,
        num_samples_per_env: int,
        intervention_targets_per_env: Tensor,
    ) -> tuple[Tensor, ...]:
        """
        Sample from the DGP.
        """
        num_envs = intervention_targets_per_env.shape[0]
        shape = (
            num_samples_per_env,
            num_envs,
            self.latent_scm.latent_dim,
        )
        u = torch.zeros(shape)
        v = torch.zeros(shape)
        intervention_targets_out = torch.zeros(shape)
        e = torch.zeros((num_samples_per_env, num_envs, 1), dtype=torch.long)
        log_prob = torch.zeros((num_samples_per_env, num_envs, 1))

        for env in range(num_envs):
            int_targets_env = intervention_targets_per_env[env, :]

            noise_samples_env = self.noise_generator.sample(env, size=num_samples_per_env)
            noise_log_prob_env = self.noise_generator.log_prob(noise_samples_env, env)

            latent_samples_env = self.latent_scm.push_forward(noise_samples_env, env)
            log_det_scm = self.latent_scm.log_inverse_jacobian(latent_samples_env, noise_samples_env, env)

            intervention_targets_out[:, env, :] = int_targets_env
            u[:, env, :] = noise_samples_env
            v[:, env, :] = latent_samples_env
            e[:, env, :] = env
            log_prob[:, env, :] = (log_det_scm + noise_log_prob_env.sum(dim=1)).unsqueeze(1)

        # Flatten
        flattened_shape = (num_samples_per_env * num_envs, self.latent_scm.latent_dim)
        intervention_targets_out = intervention_targets_out.reshape(flattened_shape)
        u = u.reshape(flattened_shape)
        v = v.reshape(flattened_shape)
        e = e.reshape(num_samples_per_env * num_envs, 1)
        log_prob = log_prob.reshape(num_samples_per_env * num_envs, 1)

        # Apply mixing function
        x = self.mixing_function(v)

        # Skip this part – non-square mixing matrix
        # unmixing_jacobian = self.mixing_function.unmixing_jacobian(v)
        # log_det_unmixing_jacobian = torch.slogdet(unmixing_jacobian).logabsdet.unsqueeze(1)
        # log_prob += log_det_unmixing_jacobian

        return (
            x,
            v,
            u,
            e,
            intervention_targets_out,
            log_prob,
        )



#def make_multi_env_dgp(
#    latent_dim: int,
#    observation_dim: int,
#    adjacency_matrix: np.ndarray,
#    intervention_targets_per_env: Tensor,
#    shift_noise: bool = True,
#    noise_shift_type: str = "mean",
#    mixing: str = "nonlinear",
#    scm: str = "linear",
#    n_nonlinearities: int = 1,
#    scm_coeffs_low: float = -1,
#    scm_coeffs_high: float = 1,
#    coeffs_min_abs_value: float = None,
#    edge_prob: float = None,
#    snr: float = 1.0,
#) -> MultiEnvDGP:

def make_multi_env_dgp(
    latent_dim: int,
    observation_dim: int,
    adjacency_matrix: np.ndarray,
    intervention_targets_per_env: Tensor,
    shift_noise: bool = True,
    noise_shift_type: str = "mean",
    env_noise_scale: torch.Tensor = None,
    mixing: str = "nonlinear",
    scm: str = "linear",
    n_nonlinearities: int = 1,
    scm_coeffs_low: float = -1,
    scm_coeffs_high: float = 1,
    coeffs_min_abs_value: float = None,
    edge_prob: float = None,
    snr: float = 1.0,
    mixing_matrix: np.ndarray = None,  # ✅ FIXED
) -> MultiEnvDGP:

    # Choose mixing function
    if mixing == "nonlinear":
        mixing_function = NonlinearMixing(
            latent_dim=latent_dim,
            observation_dim=observation_dim,
            n_nonlinearities=n_nonlinearities,
            mixing_matrix=mixing_matrix  # ✅ uses your custom matrix if provided
        )
    elif mixing == "linear":
        mixing_function = LinearMixing(
            latent_dim=latent_dim,
            observation_dim=observation_dim
        )
    else:
        raise ValueError(f"Unknown mixing function {mixing}")

    # Validate or sample adjacency matrix
    if not isinstance(adjacency_matrix, np.ndarray):
        assert edge_prob is not None, "edge_prob must be given if no adjacency_matrix is given"
        adjacency_matrix = sample_random_dag(latent_dim, edge_prob)

    # Choose latent SCM
    if scm == "linear":
        latent_scm = LinearSCM(
            adjacency_matrix=adjacency_matrix,
            latent_dim=latent_dim,
            intervention_targets_per_env=intervention_targets_per_env,
            coeffs_low=scm_coeffs_low,
            coeffs_high=scm_coeffs_high,
            coeffs_min_abs_value=coeffs_min_abs_value,
        )
    elif scm == "location-scale":
        latent_scm = LocationScaleSCM(
            adjacency_matrix=adjacency_matrix,
            latent_dim=latent_dim,
            intervention_targets_per_env=intervention_targets_per_env,
            snr=snr,
        )
    else:
        raise ValueError(f"Unknown SCM {scm}")

    # Set up noise generator
    noise_generator = GaussianNoise(
        latent_dim=latent_dim,
        intervention_targets_per_env=intervention_targets_per_env,
        shift=shift_noise,
        shift_type=noise_shift_type,
        env_noise_scale=env_noise_scale
    )

    # Construct full DGP
    return MultiEnvDGP(
        latent_scm=latent_scm,
        noise_generator=noise_generator,
        mixing_function=mixing_function,
    )
