import os
import numpy as np
import pandas as pd

def _sniff_delimiter(path: str) -> str:
    # For now we only support TSV (.tsv.bz2)
    return "\t"

def load_census_df(max_rows: int = 5000, max_cols: int = 512) -> pd.DataFrame:
    """Load county dataset strictly from LOTC_CENSUS_BZ2 (.tsv.bz2). Optionally use tests/county_data.getdf()."""
    path = os.getenv("LOTC_CENSUS_BZ2")
    if not path or not os.path.exists(path):
        raise FileNotFoundError("Set LOTC_CENSUS_BZ2 to a valid .tsv.bz2 file path")
    if not path.endswith(".tsv.bz2"):
        raise ValueError(f"Only .tsv.bz2 is supported for now, got: {path}")

    # Optional normalized path via your tests/county_data.py
    use_getdf = os.getenv("LOTC_USE_GETDF", "0").lower() in ("1", "true", "yes", "on")
    if use_getdf:
        try:
            from county_data import getdf  # lives in ./tests
            df = getdf()                  # getdf reads LOTC_CENSUS_BZ2 internally
        except Exception as e:
            raise RuntimeError("LOTC_USE_GETDF=1 but county_data.getdf() failed") from e
    else:
        df = pd.read_csv(path, sep="\t", compression="bz2", index_col="STCOU", low_memory=False)

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])

    # Optional column subsample
    if max_cols is not None and max_cols > 0 and df.shape[1] > max_cols:
        step = max(df.shape[1] // max_cols, 1)
        cols = df.columns[::step][:max_cols]
        df = df.loc[:, cols]

    # Optional row subsample
    if max_rows is not None and max_rows > 0 and df.shape[0] > max_rows:
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
        s = np.asarray(s, dtype=np.float64)
        s = np.sort(np.unique(s))
        if s.size < 2:
            continue
        diffs = np.diff(s)
        pos = diffs[diffs > min_positive]
        if pos.size:
            resolutions.append(float(np.min(pos)))
    # If *no* column yielded a positive gap, fall back
    if not resolutions:
        return 1e-6
    # Median across columns, but clamp to keep MDL baseline meaningful on coarse data
    res = float(np.median(resolutions))
    return float(np.clip(res, 1e-12, 2.5e-1))
