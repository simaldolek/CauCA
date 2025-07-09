from abc import ABC
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .utils import leaky_tanh, sample_invertible_matrix


class MixingFunction(ABC):
    """
    Base class for mixing functions.

    The mixing function is the function that maps from the latent space to the observation space.

    Parameters
    ----------
    latent_dim: int
        Dimension of the latent space.
    observation_dim: int
        Dimension of the observation space.
    """

    def __init__(self, latent_dim: int, observation_dim: int) -> None:
        self.latent_dim = latent_dim
        self.observation_dim = observation_dim
     

    def __call__(self, v: Tensor) -> Tensor:
        """
        Apply the mixing function to the latent variables.

        Parameters
        ----------
        v: Tensor, shape (num_samples, latent_dim)
            Latent variables.

        Returns
        -------
        x: Tensor, shape (num_samples, observation_dim)
            Observed variables.
        """
        raise NotImplementedError()

    def save_coeffs(self, path: Path) -> None:
        """
        Save the coefficients of the mixing function to disk.

        Parameters
        ----------
        path: Path
            Path to save the coefficients to.
        """
        raise NotImplementedError()

    def unmixing_jacobian(self, v: Tensor) -> Tensor:
        """
        Compute the jacobian of the inverse mixing function using autograd and the inverse function theorem.

        Parameters
        ----------
        v: Tensor, shape (num_samples, latent_dim)
            Latent variables.

        Returns
        -------
        unmixing_jacobian: Tensor, shape (num_samples, observation_dim, latent_dim)
            Jacobian of the inverse mixing function.

        References
        ----------
        https://en.wikipedia.org/wiki/Inverse_function_theorem
        https://discuss.pytorch.org/t/computing-batch-jacobian-efficiently/80771/7
        """
        func = self.__call__
        inputs = v

        mixing_jacobian = torch.vmap(torch.func.jacrev(func))(inputs)
        unmixing_jacobian = torch.inverse(mixing_jacobian)

        return unmixing_jacobian


class LinearMixing(MixingFunction):
    """
    Linear mixing function. The coefficients are sampled from a uniform distribution.

    Parameters
    ----------
    latent_dim: int
        Dimension of the latent space.
    observation_dim: int
        Dimension of the observation space.
    """

    def __init__(self, latent_dim: int, observation_dim: int) -> None:
        super().__init__(latent_dim, observation_dim)
        self.coeffs = torch.rand((latent_dim, observation_dim))

    def __call__(self, v: Tensor) -> Tensor:
        return torch.matmul(v, self.coeffs.to(v.device))

    def save_coeffs(self, path: Path) -> None:
        # save matrix coefficients
        torch.save(self.coeffs, path / "matrix.pt")
        matrix_np = self.coeffs.numpy()  # convert to Numpy array
        df = pd.DataFrame(matrix_np)  # convert to a dataframe
        df.to_csv(path / "matrix.csv", index=False)  # save as csv


class NonlinearMixing(MixingFunction):
    """
    Nonlinear mixing function for latent -> observed mapping.

    Step 1: Apply a predefined [obs_dim x latent_dim] mixing matrix (e.g., 170×7) to project latent values into 170 ROI space.
    Step 2: Apply a series of invertible transformations and nonlinearities to simulate more complex mappings.
    """

    def __init__(
        self,
        latent_dim: int,
        observation_dim: int,
        n_nonlinearities: int = 1,
        mixing_matrix: np.ndarray = None
    ) -> None:
        super().__init__(latent_dim, observation_dim)

        # Validate and load custom mixing matrix
        if mixing_matrix is None:
            raise ValueError("You must supply a [observation_dim x latent_dim] mixing matrix.")
        else:
            assert mixing_matrix.shape == (observation_dim, latent_dim), \
                f"Expected shape ({observation_dim}, {latent_dim}), got {mixing_matrix.shape}"
            self.mixing_matrix = torch.tensor(mixing_matrix, dtype=torch.float32)

        self.n_nonlinearities = n_nonlinearities

        # Sample invertible [obs_dim x obs_dim] matrices
        self.matrices = [sample_invertible_matrix(observation_dim) for _ in range(n_nonlinearities)]
        self.nonlinearities = [leaky_tanh for _ in range(n_nonlinearities)]

    def __call__(self, v: Tensor) -> Tensor:
        """
        Apply the mixing: first the linear ROI projection, then nonlinear mixing layers.
        """
        # Step 1: Latent → Observed (linear projection)
        x = torch.matmul(v, self.mixing_matrix.T.to(v.device))  # Shape: [batch_size, observation_dim]

        # Step 2: Optional Nonlinear Layers
        for i in range(self.n_nonlinearities):
            mat = self.matrices[i].to(v.device)  # Shape: [obs_dim, obs_dim]
            x = self.nonlinearities[i](torch.matmul(x, mat))    # Shape preserved

        return x

    def save_coeffs(self, path: Path) -> None:
        """
        Save matrices and their determinants to disk.
        """
        path.mkdir(parents=True, exist_ok=True)

        # Save each matrix
        for i, mat in enumerate(self.matrices):
            torch.save(mat, path / f"matrix_{i}.pt")
            pd.DataFrame(mat.numpy()).to_csv(path / f"matrix_{i}.csv", index=False)

        # Save determinants of the nonlinear mixing matrices
        dets = torch.stack([torch.det(m) for m in self.matrices]).numpy()
        pd.DataFrame(dets, columns=["Determinant"]).to_csv(path / "matrix_determinants.csv", index=False)

