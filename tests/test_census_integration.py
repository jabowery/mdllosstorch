
import os
import numpy as np
import pytest
import torch

from mdllosstorch import MDLLoss
from tests.util_census import load_census_df, estimate_global_resolution

@pytest.mark.slow
def test_census_mdl_smoketest():
    path = (
        os.getenv("LOTC_CENSUS_TSV_BZ2")
        or os.getenv("LOTC_CENSUS_BZ2")
        or os.getenv("LOTC_CENSUS_NPZ")
    )
    if not path or not os.path.exists(path):
        pytest.skip("County dataset not available")

    df = load_census_df(max_rows=5000, max_cols=512).astype(np.float32)
    assert df.shape[0] > 0 and df.shape[1] > 0

    data_res = estimate_global_resolution(df)
    X = df.values
    X = X - np.nanmean(X, axis=0, keepdims=True)

    x_t = torch.from_numpy(X)
    model = torch.nn.Identity()
    yhat = x_t.clone()

    loss = MDLLoss(method="yeo-johnson", data_resolution=data_res, param_resolution=1e-6)
    bits = loss(x_t, yhat, model)

    n = X.size
    baseline = n * float(np.log2(max(1.0 / max(data_res, 1e-12), 1.0)))
    rel_err = abs(bits.item() - baseline) / max(baseline, 1.0)
    assert rel_err < 0.30, f"Residual bits deviate too much: got {bits.item()}, baseline {baseline}"
