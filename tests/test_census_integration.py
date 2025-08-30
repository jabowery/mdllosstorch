import os
import numpy as np
import pytest
pytest.importorskip("pandas", reason="pandas required for census integration tests (install with '.[census]')")
import torch

from mdllosstorch import MDLLoss
from tests.util_census import load_census_df, estimate_global_resolution

@pytest.mark.slow
def test_census_mdl_ordering():
    path = os.getenv("LOTC_CENSUS_BZ2")
    if not path or not os.path.exists(path):
        pytest.skip("County dataset not available")

    # Load a manageable slice and center columns
    df = load_census_df(max_rows=5000, max_cols=512).astype(np.float32)
    assert df.shape[0] > 0 and df.shape[1] > 0

    data_res = estimate_global_resolution(df)
    X = df.values
    X = X - np.nanmean(X, axis=0, keepdims=True)

    x_t = torch.from_numpy(X)
    model = torch.nn.Identity()
    yhat_perfect = x_t.clone()

    # Slightly worse reconstruction: add small noise with fixed seed for determinism
    rng = np.random.default_rng(123)
    noise = torch.from_numpy(rng.normal(loc=0.0, scale=0.01, size=X.shape).astype(np.float32))
    yhat_noisy = x_t + noise

    loss = MDLLoss(method="yeo-johnson", data_resolution=data_res, param_resolution=1e-6)

    bits_perfect = loss(x_t, yhat_perfect, model).item()
    bits_noisy = loss(x_t, yhat_noisy, model).item()

    # 1) Finite numbers
    assert np.isfinite(bits_perfect), f"Perfect recon produced non-finite bits: {bits_perfect}"
    assert np.isfinite(bits_noisy), f"Noisy recon produced non-finite bits: {bits_noisy}"

    # 2) Ordering: worse recon should have higher MDL than perfect recon
    assert bits_noisy > bits_perfect, (
        f"Ordering violated: noisy ({bits_noisy:.3f}) <= perfect ({bits_perfect:.3f}); "
        f"data_res={data_res:.6g}, shape={X.shape}"
    )
