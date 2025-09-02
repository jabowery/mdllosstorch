import os

import numpy as np
import pytest

pytest.importorskip(
    "pandas", reason="pandas required for census integration tests (install with '.[census]')"
)
import torch

from mdllosstorch import MDLLoss
from tests.util_census import estimate_global_resolution, load_census_df


@pytest.mark.slow
def test_mdl_beats_raw_baseline():
    """
    Sanity check: a reasonable model should yield a *smaller* MDL than
    encoding the raw dataset directly with the same residual coder and
    zero parameter bits (i.e., 'no model' baseline).
    """
    path = os.getenv("LOTC_CENSUS_BZ2")
    if not path or not os.path.exists(path):
        pytest.skip("County dataset not available")

    # Load a manageable slice and center columns
    df = load_census_df(max_rows=2000, max_cols=128).astype(np.float32)
    assert df.shape[0] > 0 and df.shape[1] > 0

    data_res = estimate_global_resolution(df)
    X = df.values
    X = X - np.nanmean(X, axis=0, keepdims=True)  # match other tests’ centering

    x_t = torch.from_numpy(X)

    # --- Baseline: "no model"
    # Encode raw data as residuals by setting reconstructed = 0,
    # and use an Identity module (no parameters -> 0 parameter bits).
    baseline_model = torch.nn.Identity()
    baseline_recon = torch.zeros_like(x_t)

    loss = MDLLoss(
        method="yeo-johnson", data_resolution=data_res, param_resolution=1e-6, coder="gauss_nml"
    )
    baseline_bits = loss(x_t, baseline_recon, baseline_model).item()

    # --- Reasonable model: exact reconstruction with a parametrized linear layer
    # This forces residuals ~ 0 while incurring a (small) parameter-bit cost.
    d = X.shape[1]
    model = torch.nn.Linear(d, d, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(d, dtype=x_t.dtype))  # yhat = x_t

    yhat = model(x_t)
    model_bits = loss(x_t, yhat, model).item()

    # We expect the model (perfect recon + param bits) to beat the raw baseline
    print(f"Expected MDL(model) < MDL(raw baseline); got model={model_bits:.3f}")
    print(f"baseline={baseline_bits:.3f} (data_res={data_res:.6g}, shape={X.shape})")
    assert model_bits < baseline_bits, (
        f"Expected MDL(model) < MDL(raw baseline); got model={model_bits:.3f}, "
        f"baseline={baseline_bits:.3f} (data_res={data_res:.6g}, shape={X.shape})"
    )
