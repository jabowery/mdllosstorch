import numpy as np
import torch
import pytest

from mdllosstorch import MDLLoss

@pytest.mark.parametrize("delta", [1e-3, 1e-2, 0.1])
def test_gauss_nml_ordering_and_sign(delta):
    # Synthetic data to avoid I/O; this is a fast unit test
    N, D = 512, 32
    X = torch.randn(N, D, dtype=torch.float32)
    perfect = X.clone()
    noisy = X + 0.01 * torch.randn_like(X)

    # Absolute coder should be default and should behave well
    loss = MDLLoss(coder="gauss_nml", data_resolution=delta)
    bits_perfect = loss(X, perfect, torch.nn.Identity()).item()
    bits_noisy = loss(X, noisy, torch.nn.Identity()).item()

    assert np.isfinite(bits_perfect) and np.isfinite(bits_noisy)
    assert bits_noisy > bits_perfect, "Noisy reconstruction must cost more bits than perfect"
    # With quantization constant + sigma penalty, absolute bits should not be highly negative
    assert bits_perfect > 0.0, "Absolute MDL under gauss_nml should be non-negative for reasonable Δ"
