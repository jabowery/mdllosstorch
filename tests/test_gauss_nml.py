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
    # Use larger noise for delta=0.1 to ensure detectability above quantization floor
    noise_scale = max(0.01, delta * 0.5)  # Scale noise with delta
    noisy = X + noise_scale * torch.randn_like(X)

    # Absolute coder should be default and should behave well
    loss = MDLLoss(coder="gauss_nml", data_resolution=delta)
    bits_perfect = loss(X, perfect, torch.nn.Identity()).item()
    bits_noisy = loss(X, noisy, torch.nn.Identity()).item()

    assert np.isfinite(bits_perfect) and np.isfinite(bits_noisy)
    assert bits_noisy > bits_perfect, f"Noisy reconstruction must cost more bits than perfect. Perfect: {bits_perfect:.6f}, Noisy: {bits_noisy:.6f}, Delta: {delta}, Noise scale: {noise_scale}"
    # With quantization constant + sigma penalty, absolute bits should not be highly negative
    assert bits_perfect > 0.0, "Absolute MDL under gauss_nml should be non-negative for reasonable δ"
