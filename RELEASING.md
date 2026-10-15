# Releasing sparc-xai

Releases are published to PyPI by the `release` workflow through trusted publishing: PyPI trusts GitHub Actions of this repository for the project `sparc-xai`, so no API token is stored anywhere. Publishing happens when a GitHub release is published.

## One time setup on PyPI (done by the project owner)

1. Sign in to https://pypi.org, open *Your projects*, then *Publishing* (https://pypi.org/manage/account/publishing/).
2. Under *Add a new pending publisher*, choose GitHub and fill in:
   - PyPI project name: `sparc-xai`
   - Owner: `charis-ntak`
   - Repository name: `sparc`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. Save. The entry stays "pending" until the first release creates the project; afterwards it is listed under the project's publishers.

The GitHub side is already in place: the workflow `.github/workflows/release.yml` and the environment `pypi` of the repository. To require a manual approval before every upload, add a required reviewer to that environment in the repository settings.

## Every release

1. Set the version in `pyproject.toml`, `src/sparc/__init__.py` and `CITATION.cff` (with the release date), and date the section in `CHANGELOG.md`.
2. Commit and push to `main`; the `tests` and `docs` workflows must pass.
3. Create and publish the GitHub release with the tag `v<version>`, for example:

   ```
   gh release create v0.1.0 --title "sparc-xai 0.1.0" --notes-file <(sed -n '/^## 0.1.0/,/^## /p' CHANGELOG.md | sed '$d')
   ```

4. The `release` workflow builds the source distribution and the wheel, checks their metadata and that the tag matches the version, and publishes them. The run appears under *Actions*; the package appears at https://pypi.org/project/sparc-xai/ within a minute.
5. Check the installation from PyPI in a fresh environment: `pip install sparc-xai` followed by `python -c "import sparc; print(sparc.__version__)"`.

## Zenodo DOI (optional, once)

Sign in to https://zenodo.org with GitHub, enable the repository `charis-ntak/sparc` under *GitHub*, and publish a GitHub release: Zenodo archives the release and mints a DOI, which then goes into `CITATION.cff` and the README.
