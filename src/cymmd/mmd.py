from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float

from .cymmd import calculate_mmd, median_trick


class CythonMMD:
    """
    CythonMMD is a class that provides methods for calculating the Maximum Mean Discrepancy (MMD) between two distributions.
    """

    @staticmethod
    def median_trick(
        x: Float[npt.NDArray[np.floating] | torch.Tensor, "num dim"],
    ) -> float:
        """
        The median trick, used to automatically determine the sigma parameter of the Gaussian kernel

        Args:
            x (Float[npt.NDArray[np.floating] | torch.Tensor, "num dim"]): The input samples for which the median is calculated

        Returns:
            float: The median of the pairwise distances between samples
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return median_trick(x)

    @staticmethod
    def calculate(
        x: Float[npt.NDArray[np.floating] | torch.Tensor, "num dim"],
        y: Float[npt.NDArray[np.floating] | torch.Tensor, "num dim"],
        sigma: Optional[float] = None,
        block_size: int = 1000,
        use_median_trick: bool = True,
    ) -> float:
        """
        Calculate the Maximum Mean Discrepancy (MMD) between two distributions using the Gaussian kernel.

        Args:
            x (Float[npt.NDArray[np.floating] | torch.Tensor, "num dim"]): The first input samples
            y (Float[npt.NDArray[np.floating] | torch.Tensor, "num dim"]): The second input samples
            sigma (Optional[float], optional): The bandwidth of the Gaussian kernel. If None, it will be calculated using the median trick. Defaults to None.
            block_size (int, optional): The block size for the calculation. Defaults to 1000.
            use_median_trick (bool, optional): Whether to use the median trick to calculate sigma. Defaults to True.

        Returns:
            float: The MMD value between the two distributions
        """
        # Ensure the input is a numpy array
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        # Ensure the data is two-dimensional
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        return calculate_mmd(
            x.astype(np.float64),
            y.astype(np.float64),
            sigma,
            block_size,
            use_median_trick,
        )
