
# mdllosstorch – Instructions for Maintainers

You are a **maintainer** (release manager/packager). You need to *publish the library* to PyPI.

## Meta-Test `pyproject.toml`
The design philosophy of this repository is to treat `pyproject.toml` as its `single source of truth` regarding virtually everything release-related except the README files.

To this end there exists a script that validates the `pyproject.tom` file:

./scripts/soak_pyproject.sh

## Release Process
1. Bump version in:
   - `pyproject.toml`
   - `src/mdllosstorch/__init__.py`

   Or run the helper script:
   ```bash
   ./scripts/release_bump.sh 0.1.1
   ```

2. Build the package:
   ```bash
   python -m build
   ```

3. Validate the build:
   ```bash
   twine check dist/*
   ```

4. Upload to PyPI:
   ```bash
   twine upload dist/*
   ```

   For TestPyPI:
   ```bash
   twine upload --repository testpypi dist/*
   ```

## Notes
- CI must be green before release.
- Keep dependencies in `pyproject.toml` **minimal and permissive**.
- Update constraints (`constraints.txt`) only for reproducible dev/test, not for published deps.
