# MDL Modes: Absolute vs. Legacy

`mdllosstorch` supports two residual coding modes:

- **`gauss_nml` (default)**: An absolute, quantization-aware coder based on a Gaussian-with-unknown-variance MDL approximation.
  - Adds a quantization constant per entry: `log2(1/Δ)`
  - Floors variance at the quantization noise level: `σ²_min = Δ² / 12`
  - Adds an unknown-σ penalty of about `½ log2(n)` **per feature**
  - Produces stable, interpretable *absolute* codelengths (in bits) that do not dive to −∞ at perfect reconstruction.

- **`legacy`**: The original transform-based path (Box–Cox/Yeo–Johnson + variance).
  - Behaves like a **differential entropy** estimate; may show strongly **negative totals** when residual variance is small.
  - Useful for **relative comparisons** (ordering models) but less suitable as an absolute MDL.

## Choosing a Mode

- For *model selection* and *reporting absolute MDL*, prefer **`gauss_nml`** (now the default).
- For continuity with older experiments, use `coder="legacy"`.

## Resolution (Δ)

Absolute MDL for continuous data depends on a discretization resolution `Δ`. You can:
- Provide a scalar `data_resolution` explicitly, or
- Use the library’s estimator (`"auto"`) to infer a global resolution.

## CLI and Environment Variables

The CLI (`scripts/run_census_mdl.py`) reads:
- `LOTC_CODER` to choose the residual coder (`gauss_nml` or `legacy`) when `--coder` is not provided.
- `LOTC_CENSUS_BZ2` for the path to the TSV dataset.
- `LOTC_USE_GETDF=1` to use the normalized `tests/county_data.getdf()` path when available.

Examples:
```bash
# Absolute MDL (default)
export LOTC_CODER=gauss_nml
python scripts/run_census_mdl.py --rows 5000 --cols 512 --method yeo-johnson --report

# Legacy path (differential-like, may be negative)
export LOTC_CODER=legacy
python scripts/run_census_mdl.py --rows 5000 --cols 512 --method yeo-johnson --report