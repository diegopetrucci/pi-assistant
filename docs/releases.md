# Release Process

Versioning is tracked via the `[project].version` field in `pyproject.toml`. The repository is currently on `0.1.0`; bump this value in every release PR so the merge commit clearly identifies what will ship when the tag is created.

## Steps to Cut a Release
- Branch from `main` (example: `release/v0.1.1`) and bump the version in `pyproject.toml`.
- Run the usual quality gates locally (`uv run ruff check .` and `uv run pytest -v`) and open a PR describing the release.
- After the PR merges, create an annotated tag that matches the new version (e.g. `git tag -a v0.1.1 -m "Release v0.1.1"` on the merge commit) and push it with `git push origin v0.1.1`.
- Tag pushes that match `v*` automatically trigger `.github/workflows/release.yml`, which installs dependencies, runs `uv build`, uploads the `dist/` artifacts, and publishes a GitHub Release with auto-generated notes.
- If a tag was pushed by mistake, delete it (`git push --delete origin v0.1.1`) so the workflow does not rerun on stale bits.

Following this flow keeps `main` deployable at all times while making releases fully reproducible from signed tags.
