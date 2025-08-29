
# GitHub Release & Branch Protection

## Release workflow
- `.github/workflows/release.yml` builds, tests, and publishes to PyPI on tags matching `v*`.
- Configure the repo secret **PYPI_API_TOKEN** (an API token from PyPI).

## Nightly canary
- `.github/workflows/nightly-canary.yml` runs tests daily on latest dependencies.

## Branch protection
Use the helper script to set protection on **main** (requires repo admin and gh CLI):

```bash
OWNER=yourname REPO=mdllosstorch ./scripts/apply_branch_protection.sh
```

Adjust the required status checks to match your CI job names. You can also set protection via GitHub UI:
Settings → Branches → Branch protection rules.
