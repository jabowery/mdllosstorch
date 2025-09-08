# test_census_integration.py - Census data tests with current API
import numpy as np
import pytest

pytest.importorskip(
    "pandas", reason="pandas required for census integration tests (install with '.[census]')"
)
import torch

from mdllosstorch import MDLLoss
from tests.util_census import load_census_df


@pytest.mark.slow
def test_census_mdl_ordering():
    """Test MDL ordering properties on real census data."""

    # Load manageable subset of census data
    df = load_census_df(max_rows=1000, max_cols=64).astype(np.float32)
    assert df.shape[0] > 0 and df.shape[1] > 0

    X = df.values
    x_t = torch.from_numpy(X)

    # Create models and reconstructions of different quality
    model = torch.nn.Identity()
    perfect_recon = x_t.clone()

    # Add controlled noise for worse reconstruction
    torch.manual_seed(42)
    noisy_recon = x_t + 0.01 * x_t.std() * torch.randn_like(x_t)

    loss = MDLLoss()
    bits_perfect = loss(x_t, perfect_recon, model).item()
    bits_noisy = loss(x_t, noisy_recon, model).item()

    print(f"Census data shape: {X.shape}")
    print(f"Data range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"Perfect reconstruction: {bits_perfect:.3f} bits")
    print(f"Noisy reconstruction: {bits_noisy:.3f} bits")
    print(f"Difference: {bits_noisy - bits_perfect:.3f} bits")

    # Verify basic properties
    assert np.isfinite(bits_perfect), f"Perfect recon non-finite: {bits_perfect}"
    assert np.isfinite(bits_noisy), f"Noisy recon non-finite: {bits_noisy}"
    assert bits_noisy > bits_perfect, (
        f"Noisy ({bits_noisy:.3f}) should be > perfect ({bits_perfect:.3f})"
    )


@pytest.mark.slow
def test_census_model_vs_baseline():
    """Test that a reasonable model beats a no-model baseline on census data."""

    # Load census data
    df = load_census_df(max_rows=800, max_cols=32).astype(np.float32)
    X = df.values
    x_t = torch.from_numpy(X)
    B, F = x_t.shape

    # Baseline: no model (predict zeros)
    baseline_model = torch.nn.Identity()
    baseline_recon = torch.zeros_like(x_t)

    # Reasonable model: capture some structure with linear transformation
    reasonable_model = torch.nn.Linear(F, F, bias=False)

    # Initialize with some structure (not random)
    with torch.no_grad():
        # Start with identity then add small perturbation to create imperfect but reasonable reconstruction
        reasonable_model.weight.copy_(torch.eye(F))
        reasonable_model.weight.add_(0.01 * torch.randn_like(reasonable_model.weight))

    reasonable_recon = reasonable_model(x_t)

    loss = MDLLoss()
    baseline_bits = loss(x_t, baseline_recon, baseline_model).item()
    reasonable_bits = loss(x_t, reasonable_recon, reasonable_model).item()

    print(f"Census baseline (no model): {baseline_bits:.3f} bits")
    print(f"Reasonable model: {reasonable_bits:.3f} bits")
    print(f"Improvement: {baseline_bits - reasonable_bits:.3f} bits")

    # Verify finite results
    assert np.isfinite(baseline_bits), f"Baseline produced non-finite: {baseline_bits}"
    assert np.isfinite(reasonable_bits), f"Model produced non-finite: {reasonable_bits}"

    # The reasonable model should beat the baseline
    assert reasonable_bits < baseline_bits, (
        f"Reasonable model ({reasonable_bits:.3f}) should beat baseline ({baseline_bits:.3f}) "
        f"on census data shape {X.shape}"
    )


@pytest.mark.slow
def test_census_scale_robustness():
    """Test that MDL handles census data's extreme scale differences robustly."""

    df = load_census_df(max_rows=500, max_cols=20).astype(np.float32)
    X = df.values

    # Verify we have scale differences (census data should have this naturally)
    feature_scales = np.std(X, axis=0)
    scale_ratio = np.max(feature_scales) / np.min(feature_scales[feature_scales > 0])
    print(f"Feature scale ratio: {scale_ratio:.2f}")
    print(f"Data range: [{X.min():.6f}, {X.max():.6f}]")
    print(f"Feature scale range: [{feature_scales.min():.6f}, {feature_scales.max():.6f}]")

    assert scale_ratio > 10, "Census data should have significant scale differences"

    x_t = torch.from_numpy(X)
    model = torch.nn.Linear(X.shape[1], X.shape[1])
    yhat = model(x_t)

    loss = MDLLoss()
    bits = loss(x_t, yhat, model)

    assert torch.isfinite(bits), f"MDL should handle scale differences, got {bits}"
    assert bits.item() > 0, f"MDL should be positive with scale differences, got {bits.item()}"
    print(f"MDL with scale differences: {bits.item():.3f} bits")


@pytest.mark.slow
def test_census_gradient_flow():
    """Test gradient flow on census data."""

    df = load_census_df(max_rows=400, max_cols=16).astype(np.float32)
    X = df.values
    x_t = torch.from_numpy(X)

    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 8), torch.nn.ReLU(), torch.nn.Linear(8, X.shape[1])
    )
    model.train()

    yhat = model(x_t)
    loss = MDLLoss()
    bits = loss(x_t, yhat, model)

    # Backward pass
    bits.backward()

    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            grad_norms.append(grad_norm)
            assert torch.isfinite(param.grad).all(), f"Non-finite gradients in {name}"

    total_grad_norm = sum(grad_norms)
    assert len(grad_norms) > 0, "No parameters received gradients"
    assert total_grad_norm > 1e-8, f"Gradients too small: {total_grad_norm}"

    print(f"Census gradient flow test - Total grad norm: {total_grad_norm:.6f}")


@pytest.mark.slow
def test_census_computational_efficiency():
    """Test that MDL computation completes in reasonable time on census data."""
    import time

    df = load_census_df(max_rows=1500, max_cols=100).astype(np.float32)
    X = df.values
    x_t = torch.from_numpy(X)

    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 50), torch.nn.ReLU(), torch.nn.Linear(50, X.shape[1])
    )

    yhat = model(x_t)

    # Time the computation
    loss = MDLLoss(sa_max_steps=500)  # Reduced for efficiency test
    start_time = time.time()
    bits = loss(x_t, yhat, model)
    elapsed = time.time() - start_time

    assert torch.isfinite(bits), f"Efficiency test produced non-finite: {bits}"
    print(f"Census data {X.shape}: {bits.item():.3f} bits in {elapsed:.2f}s")

    # Should complete in reasonable time (generous bound)
    assert elapsed < 60, f"Computation took too long: {elapsed:.2f}s"
