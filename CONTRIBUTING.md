# Contributing to Divi

Thank you for considering contributing to Divi!

Divi is an open-source Python library designed for distributed quantum program execution. Whether you're here to fix bugs, improve documentation, build new features, or test our code on different platforms, your contributions are highly appreciated.

---

## What You Can Contribute

- Examples written in the Divi framework ([divi-examples](https://github.com/QoroQuantum/divi-examples))
- Bug fixes
- New features or enhancements
- Benchmarking and testing improvements
- Backend integrations
- Local simulator integrations

## Getting Started

### 1. Fork the Repository

Click "Fork" at the top right of [the main repository](https://github.com/qoroquantum/divi) and clone your fork:

```bash
git clone https://github.com/your-username/divi.git
cd divi
```

### 2. Install uv

Divi uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Install Dependencies

This installs the `dev`, `testing`, and `docs` groups by default:

```bash
uv sync
```

For AI work, add the extra:

```bash
uv sync --extra ai            # divi-ai dependencies
```

### 4. Set Up Pre-Commit Hooks

We use pre-commit hooks to enforce formatting and license headers automatically:

```bash
pre-commit install
```

You can run all hooks manually with:

```bash
pre-commit run -a
```

### 5. Workflow

1. Create a new branch with a descriptive name, e.g. `git checkout -b feature/implementation-of-qaoa`
2. Make your changes
3. Add or update tests for new behaviour
4. Ensure all tests pass (see [Testing](#testing) below)
5. Format the code (see [Code Style](#code-style) below)
6. Push your code and create a pull request

## Code Style

We use the following tools for formatting:

- **[Black](https://github.com/psf/black)** for code formatting
- **[isort](https://pycqa.github.io/isort/)** for import sorting (configured with the Black profile)
- **[autoflake](https://github.com/PyCQA/autoflake)** for removing unused imports (runs via pre-commit)

Run all formatters before committing:

```bash
uv run black .
uv run isort .
```

### Optional Dependencies

Packages from the optional extras (`chem`, `pennylane`, `aer`, `qubo-decompose`) are never imported at module level. Go through `divi._optional` instead:

- **Using the package:** call `import_optional` where the package is needed and work with the module it returns. If the extra is missing, it raises an `ImportError` naming the extra to install; `hint=` appends extra advice.
- **Recognising an input:** call `module_if_imported` and check the input with `isinstance`. It reads `sys.modules` and never imports, since an instance of an optional package's class can only exist once that package is loaded.
- **Type hints:** import under `if TYPE_CHECKING:`.
- **Lazy public exports:** a package `__getattr__` calls `import_optional`, then imports the internal module that depends on the package. Code behind that gate, or behind an `isinstance` check on the package's own objects, uses ordinary imports.

```python
from divi._optional import import_optional, module_if_imported


def build_mean_field(mol):
    scf = import_optional("pyscf.scf", extra="chem", capability="Building a mean field")
    return scf.RHF(mol)


def is_qnode(candidate):
    qp = module_if_imported("pennylane")
    return qp is not None and isinstance(candidate, qp.QNode)
```

Avoid module-level `try`/`except ImportError` fallbacks and hand-written "requires the extra" messages.

### License Headers

All new or updated `.py` files (outside `docs/`) must include the license header from `LICENSES/.license-header`. This is enforced by pre-commit hooks.

## Testing

Run the full test suite with:

```bash
uv run pytest
```

### Parallel Execution

For faster runs:

```bash
uv run pytest -n auto
```

### Test Markers

| Marker | Description |
|---|---|
| `requires_api_key` | Cloud API tests (need a Qoro API key) |
| `algo` | Algorithm-level tests |
| `e2e` | Slow integration tests (run only when explicitly requested) |

### Running API Tests

API tests require a Qoro API key. Set the `QORO_API_KEY` environment variable or use the `--api-key` option:

```bash
QORO_API_KEY=your-key uv run pytest --run-api-tests
# or
uv run pytest --run-api-tests --api-key your-key
```

### Coverage

```bash
uv run pytest --cov=divi
```

## Documentation

### Install Doc Dependencies

```bash
uv sync --group docs
```

### Build Docs

```bash
cd docs
make build
```

### Live Reload for Development

```bash
cd docs
make dev
```

### Serve Built Docs

```bash
cd docs
make serve
```

## Commit Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/). This is enforced by a commit-msg hook.

Examples:

```
feat: add beam search aggregation strategy
fix: resolve duplicate object warnings in docs
docs: update backend execution guide with job configuration
test: add auth token resolution tests
```

## Questions?

If you have questions or want to discuss a feature before starting work, feel free to open an issue or start a discussion on the repository.
