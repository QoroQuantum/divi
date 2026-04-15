# AGENTS.md

## Project overview

- Divi is a Python library for generating and executing quantum programs at scale.
- Core source lives in `divi/`, tests in `tests/`, docs in `docs/`.
- Tutorials are in `tutorials/`; UI/demo tooling is in `visualizations/`.

## Dev environment

- Python: `>=3.11,<3.13` (see `pyproject.toml`).
- Always use the virtual environment in `.venv/` or `venv/` when running commands.
- If neither exists, do not run any Python code until the human specifies which Python executable to use.
- uv is the primary workflow:
  - Install: `uv sync`

## Code style and formatting

- Use `black` and `isort` for formatting; `isort` must use the Black profile; `autoflake` is used to remove unused imports.
- Recommended checks:
  - `uv run black .`
  - `uv run isort .` (uses Black profile, see `pyproject.toml`)
  - `pre-commit run -a`
- New/updated `.py` files (outside `docs/`) should include the license header from `LICENSES/.license-header` (pre-commit enforces this).
- Hook configuration lives in `/.pre-commit-config.yaml`; keep any new files compatible with these hooks.
- If the human is asking an inquisitive or brainstorm-style question, do not change code; respond with analysis or ideas only.

## Testing

- Run full suite: `uv run pytest`
- Coverage: `uv run pytest --cov=divi`
- Parallel (CI/large suites): `uv run pytest -n auto`
- Write spec-driven tests first (behavior-focused) before adding critical low-level mocking.
- Use pytest ecosystem tools (e.g., `pytest-mock`) only; avoid `unittest` unless no alternative exists and the human agrees.
- Markers:
  - `requires_api_key` for cloud API tests (do not run these as an agent unless explicitly asked)
  - `algo` for algorithm tests
  - `e2e` for slow integration tests (avoid during development; run only when explicitly requested)
- API tests require a Qoro API key:
  - `QORO_API_KEY=... uv run pytest --run-api-tests`
  - or `uv run pytest --run-api-tests --api-key your-key-here`
- Never inline imports nor classes in test functions. If you are facing errors due to circular imports, ask for approval before inlining imports to fix it.
- Use simulator fixtures from `tests/conftest.py`: `dummy_simulator` when a real backend is not needed (returns fake counts), `default_test_simulator` for actual circuit execution (QASM→shots), `dummy_expval_backend` for expval-mode tests. Do not create new `QiskitSimulator` instances in tests.

## Documentation

- **Always use the Makefile** in `docs/` for documentation tasks; do not invoke `sphinx-build` or other Sphinx commands directly.
- From the repo root: `cd docs` then run the desired target (e.g. `make build`, `make dev`).
- Install doc deps: `make install` (from `docs/`) or `uv sync --group docs`
- Build: `cd docs` then `make build`
- Live reload: `cd docs` then `make dev`
- Serve built docs: `cd docs` then `make serve`
- Always `make clean` before `make build` — stale generated stubs and build state can hide or spuriously produce warnings.
- Nitpick mode is always on (`nitpicky = True` in `conf.py`). No need to pass `-n`.

### API reference pages

API pages use `sphinx-automodapi` — one `.. automodapi::` directive per submodule, not hand-rolled `autoclass`/`autofunction` blocks. Every submodule that automodapi documents must have an `__all__` list controlling which names are public (otherwise stdlib imports leak into the generated stubs).

### Cross-reference conventions (avoid nitpick warnings)

- **Use the submodule path, not the top-level re-export.** Write `:class:\`~divi.qprog.algorithms.VQE\`` (resolves), not `:class:\`~divi.qprog.VQE\`` (doesn't). The `~` prefix still renders just `VQE` in the output.
- **Don't put types in `Attributes:` / `Args:` docstring sections.** `sphinx-autodoc-typehints` reads types from the Python signature; duplicating them in the docstring causes Napoleon to emit unresolvable cross-references. Write `cost_history: Optimization cost history.` not `cost_history (list[dict]): ...`.
- **Napoleon parses colons in attribute docstrings as type annotations.** `"How to sample: uniform or weighted."` makes Sphinx treat `uniform or weighted` as a type. Reword or use em-dashes.
- **Continuation lines in Google-style `Args:` blocks must be indented** relative to the argument name. Otherwise Napoleon parses each wrapped line as a new argument.
- **Parameterized generics (`list[str]`, `Sequence[float]`) don't resolve** — only the outer class does. These are covered by a catch-all in `nitpick_ignore_regex`.

### When editing `nitpick_ignore_regex`

- Add an entry only after confirming the target really cannot be resolved any other way (intersphinx, `__all__`, fully-qualified cross-reference).
- Comment *why* the rule exists — not just *what* it matches.
- The current list has ~8 entries, each justified. Keep it small.

### RST code snippets

- Run `cd docs && make test-snippets` after changing user-guide examples; CI runs this in the docs workflow.
- Conventions for RST snippets and Sybil: `docs/source/development/building_docs.rst` (section *Documentation code snippets (Sybil)*).

## Commit conventions

- Commit messages follow Conventional Commits (enforced by commit-msg hook).
- Run relevant tests for the areas you change; add or update tests for new behavior.
- Release automation is configured in `release-please-config.json`; avoid manual changes unless explicitly requested.
