
#!/usr/bin/env python
import os, argparse, numpy as np, torch
from mdllosstorch import MDLLoss
from tests.util_census import load_census_df, estimate_global_resolution

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--cols", type=int, default=512)
    ap.add_argument("--method", type=str, default="yeo-johnson", choices=["yeo-johnson", "box-cox"])
    args = ap.parse_args()

    mr = None if args.rows == -1 else args.rows
    mc = None if args.cols == -1 else args.cols

    df = load_census_df(max_rows=mr or 0, max_cols=mc or 0).astype(np.float32)
    data_res = estimate_global_resolution(df)
    X = df.values
    X = X - np.nanmean(X, axis=0, keepdims=True)

    x_t = torch.from_numpy(X)
    model = torch.nn.Identity()
    yhat = x_t.clone()

    loss = MDLLoss(method=args.method, data_resolution=data_res, param_resolution=1e-6)
    bits = loss(x_t, yhat, model).item()

    n = X.size
    baseline = n * float(np.log2(max(1.0 / max(data_res, 1e-12), 1.0)))
    bps = bits / n

    print("=== MDL Residual Bits (Identity reconstruction) ===")
    print(f"shape: {X.shape}, n={n}")
    print(f"estimated data_resolution: {data_res:.6g}")
    print(f"total bits: {bits:.3f}")
    print(f"baseline n*log2(1/res): {baseline:.3f}")
    print(f"bits per entry: {bps:.6f}")

if __name__ == "__main__":
    main()
