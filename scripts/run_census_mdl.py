#!/usr/bin/env python
import argparse
import os
import numpy as np
import torch

from mdllosstorch import MDLLoss
from mdllosstorch.mdlloss import compute_mdl, report_mdl
from tests.util_census import load_census_df, estimate_global_resolution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000, help="max rows to sample")
    parser.add_argument("--cols", type=int, default=512, help="max cols to sample")
    parser.add_argument(
        "--method", type=str, default="yeo-johnson", choices=["yeo-johnson", "box-cox"],
        help="transform method"
    )
    # Default to env LOTC_CODER if present, else "gauss_nml"
    default_coder = os.getenv("LOTC_CODER", "gauss_nml")
    parser.add_argument("--coder", type=str, default=default_coder, choices=["gauss_nml", "legacy"],
                        help="residual coder; 'gauss_nml' is absolute & quantization-aware (default)")
    parser.add_argument("--report", action="store_true", help="print detailed MDL breakdown")
    args = parser.parse_args()

    path = os.getenv("LOTC_CENSUS_BZ2")
    if not path or not os.path.exists(path):
        raise FileNotFoundError("Set LOTC_CENSUS_BZ2 to a valid .tsv.bz2 file path")

    df = load_census_df(max_rows=args.rows, max_cols=args.cols).astype(np.float32)
    data_res = estimate_global_resolution(df)

    X = df.values
    X = X - np.nanmean(X, axis=0, keepdims=True)

    x_t = torch.from_numpy(X)
    model = torch.nn.Identity()
    yhat = x_t.clone()

    if args.report:
        rpt = report_mdl(x_t, yhat, model, method=args.method, data_resolution="auto")
        loss = MDLLoss(method=args.method, data_resolution=data_res, param_resolution=1e-6, coder=args.coder)
        total = loss(x_t, yhat, model).item()
        rpt["total_bits(coded)"] = total
        for k, v in rpt.items():
            print(f"{k:25s} : {v}")
    else:
        loss = MDLLoss(method=args.method, data_resolution=data_res, param_resolution=1e-6, coder=args.coder)
        bits = loss(x_t, yhat, model)
        print(bits.item())

if __name__ == "__main__":
    main()
