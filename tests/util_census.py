import numpy as np
import pandas as pd


def _sniff_delimiter(path: str) -> str:
    # For now we only support TSV (.tsv.bz2)
    return "\t"


def load_census_df(max_rows: int = 5000, max_cols: int = 512) -> pd.DataFrame:
    """Load county dataset strictly from LOTC_CENSUS_BZ2 (.tsv.bz2). Optionally use tests/county_data.getdf()."""
    from county_data import getdf  # type: ignore

    df = getdf()  # county_data.getdf() should read LOTC_CENSUS_BZ2 itself

    # Optional column subsample
    if max_cols is not None and max_cols > 0 and df.shape[1] > max_cols:
        step = max(df.shape[1] // max_cols, 1)
        cols = df.columns[::step][:max_cols]
        df = df.loc[:, cols]

    # Optional row subsample
    if max_rows is not None and max_rows > 0 and df.shape[0] > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    return df


def estimate_global_resolution(
    df: pd.DataFrame, sample_per_col: int = 20000, min_positive: float = 1e-12
) -> float:
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
    if not resolutions:
        return 1e-6
    res = float(np.median(resolutions))
    return float(np.clip(res, 1e-12, 2.5e-1))
