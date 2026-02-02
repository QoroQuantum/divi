# AGENTS.md

## Project overview

- Divi is a Python library for generating and executing quantum programs at scale.
- Core source lives in `divi/`, tests in `tests/`, docs in `docs/`.
- Tutorials are in `tutorials/`; UI/demo tooling is in `visualizations/`.

## Dev environment

- Python: `>=3.11,<3.13` (see `pyproject.toml`).
- Always use the virtual environment in `.venv/` or `venv/` when running commands.
- If neither exists, do not run any Python code until the human specifies which Python executable to use.
- Poetry is the primary workflow:
  - Install: `poetry install`
  - Shell: `poetry shell`
- Alternative editable install for dev: `pip install -e .[dev]`

## Code style and formatting

- Use `black` and `isort` for formatting; `isort` must use the Black profile; `autoflake` is used to remove unused imports.
- Recommended checks:
  - `poetry run black .`
  - `poetry run isort .` (uses Black profile, see `pyproject.toml`)
  - `pre-commit run -a`
- New/updated `.py` files (outside `docs/`) should include the license header from `LICENSES/.license-header` (pre-commit enforces this).
- Hook configuration lives in `/.pre-commit-config.yaml`; keep any new files compatible with these hooks.
- If the human is asking an inquisitive or brainstorm-style question, do not change code; respond with analysis or ideas only.

## Testing

- Run full suite: `poetry run pytest`
- Coverage: `poetry run pytest --cov=divi`
- Parallel (CI/large suites): `poetry run pytest -n auto`
- Write spec-driven tests first (behavior-focused) before adding critical low-level mocking.
- Use pytest ecosystem tools (e.g., `pytest-mock`) only; avoid `unittest` unless no alternative exists and the human agrees.
- Markers:
  - `requires_api_key` for cloud API tests (do not run these as an agent unless explicitly asked)
  - `algo` for algorithm tests
  - `e2e` for slow integration tests (avoid during development; run only when explicitly requested)
- API tests require a Qoro API key:
  - `QORO_API_KEY=... poetry run pytest --run-api-tests`
  - or `poetry run pytest --run-api-tests --api-key your-key-here`

## Documentation

- Install doc deps: `poetry install --with docs`
- Build: `cd docs` once, then run `make build`
- Live reload: `cd docs` once, then run `make dev`
- Serve built docs: `cd docs` once, then run `make serve`

## Commit conventions

- Commit messages follow Conventional Commits (enforced by commit-msg hook).
- Run relevant tests for the areas you change; add or update tests for new behavior.
- Release automation is configured in `release-please-config.json`; avoid manual changes unless explicitly requested.
