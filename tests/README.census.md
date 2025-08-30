
# County Census Integration Harness (mdllosstorch)

This harness lets you run `MDLLoss` against the large county dataset you described,
without committing the raw data. It supports `.tsv.bz2`, `.csv.bz2`, and `.npz`.

## How to run (locally)

1. Place your file somewhere accessible, then export one of these env vars:

```bash
export LOTC_CENSUS_BZ2=/path/to/LaboratoryOfTheCounties.csv.bz2
```

2. Run the CLI script (no training, just evaluation on a sample):

```bash
python scripts/run_census_mdl.py --rows 5000 --cols 512 --method yeo-johnson
```

3. Or run the pytest integration (skips if dataset env var not set):

```bash
pytest -q tests/test_census_integration.py
```

4. Run with and without normalization of the census data:

Without normalization:
```bash
# Raw TSV (default)
export LOTC_CENSUS_BZ2=/absolute/path/to/LaboratoryOfTheCounties.tsv.bz2
pytest -q tests/test_census_integration.py
```

With normalization (dividing by population and removing some variables not compatible with normalization):
```bash
# Normalized via getdf (requires tests/county_data.py and Ref/*)
export LOTC_USE_GETDF=1
pytest -q tests/test_census_integration.py
```

## Notes

- The script estimates a sensible `data_resolution` from the data (per column) and uses
  the **median** across columns to avoid outliers. This greatly reduces the additive
  discretization constant that can otherwise dominate residual bits.
- The pytest test uses a *sample* of rows/cols to keep runtime reasonable.
- If you want to test full data locally, pass `--rows -1 --cols -1` to the script.

