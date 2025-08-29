
import os
import numpy as np
import pandas as pd

def _sniff_delimiter(path: str) -> str:
    if path.endswith(".tsv.bz2"):
        return "\t"
    if path.endswith(".csv.bz2") or path.endswith(".csv"):
        return ","
    return ","

def load_census_df(max_rows: int = 5000, max_cols: int = 512) -> pd.DataFrame:
    path = (
        os.getenv("LOTC_CENSUS_TSV_BZ2")
        or os.getenv("LOTC_CENSUS_BZ2")
        or os.getenv("LOTC_CENSUS_NPZ")
    )
    if not path or not os.path.exists(path):
        raise FileNotFoundError("Set LOTC_CENSUS_TSV_BZ2 or LOTC_CENSUS_BZ2 or LOTC_CENSUS_NPZ to a valid file path")

    if path.endswith(".npz"):
        arrs = np.load(path, allow_pickle=False)
        for key in ("X", "data", "array"):
            if key in arrs:
                X = arrs[key]
                break
        else:
            first = list(arrs.keys())[0]
            X = arrs[first]
        df = pd.DataFrame(X)
    else:
        delim = _sniff_delimiter(path)
        df = pd.read_csv(path, sep=delim, compression="bz2", low_memory=False)

    df = df.select_dtypes(include=[np.number])

    if max_cols and df.shape[1] > max_cols:
        step = max(df.shape[1] // max_cols, 1)
        cols = df.columns[::step][:max_cols]
        df = df.loc[:, cols]

    if max_rows and df.shape[0] > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    return df

def estimate_global_resolution(df: pd.DataFrame, sample_per_col: int = 20000, min_positive: float = 1e-12) -> float:
    resolutions = []
    for col in df.columns:
        s = df[col].dropna().values
        if s.size == 0:
            continue
        if sample_per_col > 0 and s.size > sample_per_col:
            rng = np.random.default_rng(42)
            s = rng.choice(s, size=sample_per_col, replace=False)
        s = np.sort(np.unique(s))
        if s.size < 2:
            continue
        diffs = np.diff(s)
        pos = diffs[diffs > min_positive]
        if pos.size:
            resolutions.append(np.min(pos))
    if not resolutions:
        return 1e-6
    return float(np.median(resolutions))
