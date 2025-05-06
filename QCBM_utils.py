# This file is based on the code from
# Amazon Braket Algorithm Library, specifically the QCBM algorithm:
# https://github.com/amazon-braket/amazon-braket-algorithm-library/blob/main/src/braket/experimental/algorithms/quantum_circuit_born_machine/qcbm.py


import numpy as np
from typing import List


def _compute_kernel(px: np.ndarray, py: np.ndarray, sigma_list: List[float] = [0.1, 1]) -> float:
    r"""Gaussian radial basis function (RBF) kernel.

    .. math::
        K(x, y) = sum_\sigma exp(-|x-y|^2/(2\sigma^2 ))

    Args:
        px (ndarray): Probability distribution
        py (ndarray): Target probability distribution
        sigma_list (List[float]): Standard deviations of distribution. Defaults to [0.1, 1].

    Returns:
        float: Value of the Gaussian RBF function for kernel(px, py).
    """
    x = np.arange(len(px))
    y = np.arange(len(py))
    K = sum(np.exp(-(np.abs(x[:, None] - y[None, :])
            ** 2) / (2 * s**2)) for s in sigma_list)
    kernel = px @ K @ py
    return kernel


def mmd_loss(px: np.ndarray, py: np.ndarray, sigma_list: List[float] = [0.1, 1]) -> float:
    r"""Maximum Mean Discrepancy loss (MMD).

    MMD determines if two distributions are equal by looking at the difference between
    their means in feature space.

    .. math::
        MMD(x, y) = | \sum_{j=1}^N \phi(y_j) - \sum_{i=1}^N \phi(x_i) |_2^2

    With a RBF kernel, we apply the kernel trick to expand MMD to

    .. math::
        MMD(x, y) = \sum_{j=1}^N \sum_{j'=1}^N k(y_j, y_{j'})
                + \sum_{i=1}^N \sum_{i'=1}^N k(x_i, x_{i'})
                - 2 \sum_{j=1}^N \sum_{i=1}^N k(y_j, x_i)

    For the RBF kernel, MMD is zero if and only if the distributions are identical.

    Args:
        px (ndarray): Probability distribution
        py (ndarray): Target probability distribution
        sigma_list (List[float]):  Standard deviations of distribution. Defaults to [0.1, 1].

    Returns:
        float: Value of the MMD loss
    """

    mmd_xx = _compute_kernel(px, px, sigma_list)
    mmd_yy = _compute_kernel(py, py, sigma_list)
    mmd_xy = _compute_kernel(px, py, sigma_list)
    return mmd_xx + mmd_yy - 2 * mmd_xy
