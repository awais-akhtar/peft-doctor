# Publishing

This repository is ready for GitHub Actions CI and PyPI publishing.

## CI

The CI workflow runs on pushes and pull requests:

```text
.github/workflows/ci.yml
```

It installs the package in editable mode, runs Ruff, runs tests, and builds the wheel and source distribution.

## Trusted Publishing

The publish workflow uses PyPI trusted publishing. That means GitHub Actions can publish without storing a long-lived PyPI token in repository secrets.

Workflow file:

```text
.github/workflows/publish.yml
```

Register these trusted publishers:

PyPI:

- project name: `peft-doctor`
- owner: `awais-akhtar`
- repository: `peft-doctor`
- workflow name: `publish.yml`
- environment: `pypi`

TestPyPI:

- project name: `peft-doctor`
- owner: `awais-akhtar`
- repository: `peft-doctor`
- workflow name: `publish.yml`
- environment: `testpypi`

## Publish to TestPyPI

Open GitHub Actions, run `Publish Python Package`, and choose `testpypi`.

## Publish to PyPI

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
