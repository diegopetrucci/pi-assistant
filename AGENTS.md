# Repository Guidelines

## Project Structure & Module Organization
`src/pi_assistant/` houses the CLI, audio capture, wake-word logic, and websocket client; keep new runtime code under the matching subpackage instead of the root. Assistant-specific glue (Responses API client, reasoning controls, transcript aggregation, and speech playback helpers) belongs under `src/pi_assistant/assistant/` rather than mixing it into the CLI. Configuration defaults live in `config/defaults.toml`, while hardware calibration helpers and provisioning scripts belong in `scripts/`. Bundled wake-word artifacts are stored in `models/` and should not be regenerated at runtime. Tests reside in `tests/`, with manual diagnostics under `tests/manual/`. Store long-form documentation or diagrams in `docs/` so README stays concise.

## Build, Test, and Development Commands
* Use `uv sync --group dev` to provision the managed `.venv/` plus lint/test tooling.
* Run the CLI via `uv run pi-assistant` and pass flags such as `--force-always-on`, `--assistant-audio-mode local-tts`, or `--simulate-query` depending on the scenario.
* Format and lint with `uv run ruff format .` and `uv run ruff check --fix .`, respectively.
* Execute the fast wake-word regression with `uv run python -m unittest tests/test_wake_word.py`, and prefer `uv run pytest -v` (optionally `--cov`) for the full suite.
* When invoking the interpreter directly, call `python3` (not `python`) to avoid hitting the system stub.
* After completing a change, always run `uv run pyright && uv run pytest`.

## Coding Style & Naming Conventions
Target Python 3.9+ and follow Ruff’s formatter—never hand-tune spacing after running it. Stick to 4-space indentation and type-annotated functions. Modules, packages, and files use `snake_case`; CLIs expose kebab-case flags (e.g., `--assistant-audio-mode`). Keep public functions documented with concise docstrings that explain side effects or I/O expectations, and colocate constants next to the features they configure to avoid sprawling globals.

## Testing Guidelines
* Pytest is the primary harness, while targeted regressions may still use `unittest`.
* Mirror production behavior by loading fixtures from `tests/` (for example `tests/hey_jarvis.wav`) rather than embedding byte blobs.
* Name tests `test_<feature>.py` and organize helper coroutines inside `tests/utils.py` if you need reuse.
* Aim for coverage above 85% in critical modules (`audio`, `wake_word`, `network`) and gate new features on at least one positive and one failure-path assertion.
* Always write unit tests to confirm new logic, or changes to existing one

## Commit & Pull Request Guidelines
Write commits in the form `component: one-line summary` (e.g., `audio: harden portaudio fallback`) and keep bodies under 72 columns with “why” over “what”. Each PR should describe the user-facing behavior, link relevant issues, and include verification steps (`uv run pi-assistant`, `uv run pytest`). Attach screenshots or logs when the change impacts CLI output, note config additions in `README.md`, and seek a second review for hardware-touching changes.

## Security & Configuration Tips
Never commit `.env`, API keys, or artifacts generated under `.uv/` or `.venv/`. Validate that `OPENAI_API_KEY` is set before invoking network code and prefer `config/defaults.toml` for new knobs so agents running headless Pis inherit sane defaults. When adding binaries or models, document their license in `docs/` and update `.gitignore` if they should remain local-only.

## Documentation Expectations
* Whenever you add or change CLI flags, configuration keys, assistant behavior, or setup steps, mirror the change in `README.md` (usage + configuration sections) and drop any long-form writeups in `docs/`.
* Keep this `AGENTS.md` file in sync with repo conventions—update it when workflows, required tools, or testing expectations evolve.

## PRs
* Use github's `gh` command line tool to check out the prs when asked; see `.ai/gh-cli.md` for the full command list
