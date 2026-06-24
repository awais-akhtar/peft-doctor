# Publishing

This repository is ready for GitHub Actions CI and PyPI publishing.

## CI

The CI workflow runs on pushes and pull requests:

```text
.github/workflows/ci.yml
```

It installs the package in editable mode, runs Ruff, runs tests, and builds the wheel and source distribution.

## PyPI API Token Publishing

The publish workflow can publish with the repository secret `PYPI_API_TOKEN1`.

Workflow file:

```text
.github/workflows/publish.yml
```

Add this GitHub Actions secret:

```text
PYPI_API_TOKEN1
```

The token value should be the full PyPI token, starting with `pypi-`.

Pushes to `main` build the package and attempt a PyPI upload. The workflow uses
`skip-existing: true`, so a repeated push with the same package version will not
upload a duplicate release.

## Publish to TestPyPI

Open GitHub Actions, run `Publish Python Package`, and choose `testpypi`.

## Publish to PyPI With a Tag

Update the version in:

```text
pyproject.toml
src/peft_doctor/_version.py
```

Commit the change, tag it, and push the tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The `pypi` environment should require manual approval before release.
