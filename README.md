<p align="center">
  <h1 align="center">Divi</h1>
  <p align="center">
    <em>Generate, parallelize, and execute quantum programs at scale.</em>
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/qoro-divi/"><img src="https://img.shields.io/pypi/v/qoro-divi?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/qoro-divi/"><img src="https://img.shields.io/pypi/pyversions/qoro-divi" alt="Python"></a>
  <a href="https://divi.readthedocs.io"><img src="https://img.shields.io/badge/docs-readthedocs-blue" alt="Docs"></a>
  <a href="LICENSES/Apache-2.0.txt"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000" alt="Code style: black"></a>
</p>

---

**Divi** is a Python library by [Qoro Quantum](https://qoroquantum.net) for building and running quantum programs at scale. It handles circuit generation, job parallelization, and cloud execution — with built-in support for variational algorithms, custom workflows, and more — so you can focus on the quantum problem, not the plumbing.

> [!IMPORTANT]
> Divi is under active development. Expect breaking changes between minor versions.

## ⚡ Quick Start

```bash
pip install qoro-divi
```

### Nightly Builds

To install the latest development build (published daily from `main`):

```bash
pip install qoro-divi --pre
```

Run a VQE energy minimization in a few lines:

```python
import numpy as np
import pennylane as qml
from divi.qprog import VQE, HartreeFockAnsatz
from divi.backends import QiskitSimulator
from divi.qprog.optimizers import ScipyOptimizer, ScipyMethod

# Define an H₂ molecule
molecule = qml.qchem.Molecule(
    symbols=["H", "H"],
    coordinates=np.array([(0, 0, 0), (0, 0, 0.5)]),
)

vqe = VQE(
    molecule=molecule,
    ansatz=HartreeFockAnsatz(),
    n_layers=2,
    backend=QiskitSimulator(shots=5000),
    optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
    seed=42,
)

vqe.run()
print(f"Ground state energy: {vqe.best_loss:.6f}")
```

## 🌐 Cloud Execution with Qoro Service

Run the same programs on Qoro's cloud platform with tensor-network simulators — no code changes needed:

```python
from divi.backends import QoroService

service = QoroService()  # reads QORO_API_KEY from .env or environment
vqe = VQE(molecule=molecule, backend=service)
vqe.run()
```

**Get started for free** → Sign up at [dash.qoroquantum.net](https://dash.qoroquantum.net/) and receive **$100 worth of credits** to run your first quantum programs on our cloud.

## 🤖 divi-ai: AI Coding Assistant

Ask questions about Divi directly in your terminal — no API keys, no internet required after setup.

```bash
pip install qoro-divi[ai]
divi-ai
```

Answers questions about Divi APIs, generates code examples, and explains concepts — powered by a local LLM that runs entirely on your machine. See the [full documentation](https://divi.readthedocs.io/tools/divi_ai.html) for model options and usage.

## 🧩 Key Features

| Feature | Description |
|---|---|
| **VQE & QAOA** | Built-in variational algorithms with pluggable ansätze and optimizers |
| **Circuit Pipelines** | Expand → execute → reduce pattern for complex circuit workflows |
| **Program Ensembles** | Parallel execution of multiple quantum programs with automatic scheduling |
| **Dual Backends** | Local `QiskitSimulator` for dev, `QoroService` for cloud production |
| **Execution Config** | Control bond dimension, simulator type, and simulation method per job |
| **Live Reporting** | Real-time dashboards and convergence tracking via callbacks |

## 🏗️ Architecture

```
divi/
├── qprog/        # Quantum programs: VQE, QAOA, base classes, optimizers
├── backends/     # Execution backends: QiskitSimulator, QoroService
├── circuits/     # MetaCircuit templates and Circuit instances
├── pipeline/     # Circuit pipeline stages (expand, execute, reduce)
├── hamiltonians  # Molecular Hamiltonian generation
├── reporting/    # Live reporting and visualization callbacks
└── ai/           # Offline documentation chatbot (divi-ai)
```

## 📚 Documentation

Full documentation, user guides, and API reference: **[divi.readthedocs.io](https://divi.readthedocs.io)**

Hands-on examples are in the [`tutorials/`](tutorials/) folder.

## � Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and code style guidelines.

## 📄 License

Apache 2.0 — see [LICENSE](LICENSES/Apache-2.0.txt) for details.
