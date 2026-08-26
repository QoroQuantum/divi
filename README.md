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
  <a href="https://github.com/facebook/pyrefly"><img src="https://img.shields.io/endpoint?url=https://pyrefly.org/badge.json" alt="Checked with pyrefly"></a>
</p>

---

**Divi** is a Python library by [Qoro Quantum](https://qoroquantum.net) for building and running quantum programs at scale. It handles circuit generation, job parallelization, and cloud execution — with built-in support for variational algorithms, custom workflows, and more — so you can focus on the quantum problem, not the plumbing.

> [!IMPORTANT]
> Divi is under active development. Expect breaking changes between minor versions.

<!-- -->

> [!TIP]
> **Using Claude Code, Cursor, or another LLM coding agent?** Divi is indexed on [Context7](https://context7.com/qoroquantum/divi) — point your agent at `/qoroquantum/divi` to pull current, version-specific Divi docs and snippets directly into its context.

## ⚡ At-Scale Example

For the shortest introduction, start with the
**[five-minute tutorial](https://divi.readthedocs.io/en/latest/quickstart.html)**.

```bash
pip install qoro-divi
```

That covers local and cloud execution, VQE, QAOA, and program ensembles.
Heavier features live behind optional extras — `aer` (Qiskit Aer backends),
`qubo-decompose` (partitioned QUBO solving), `chem` (PySCF/OpenFermion
chemistry), `ai`, and `jupyter` — or `pip install "qoro-divi[all]"` for
everything. See the
**[installation guide](https://divi.readthedocs.io/en/latest/installation.html)**
for what each unlocks and what keeps working without it.

### Nightly Builds

To install the latest development build (published daily from `main`):

```bash
pip install qoro-divi --pre
```

Split a graph into quantum-sized MaxCut problems, solve the partitions, and
stitch their candidates into a global solution:

```python
import networkx as nx

from divi.backends import MaestroSimulator
from divi.qprog import BeamSearchStrategy
from divi.qprog.optimizers import ScipyOptimizer, ScipyMethod
from divi.qprog.problems import GraphPartitioningConfig, MaxCutProblem
from divi.qprog.workflows import PartitioningProgramEnsemble

graph = nx.barbell_graph(4, 0)
problem = MaxCutProblem(
    graph,
    config=GraphPartitioningConfig(
        max_n_nodes_per_cluster=4,
        partitioning_algorithm="kernighan_lin",
    ),
)

backend = MaestroSimulator()
ensemble = PartitioningProgramEnsemble(
    problem=problem,
    n_layers=1,
    backend=backend,
    optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
    max_iterations=10,
    seed=42,
)

ensemble.run()
cut, _ = ensemble.aggregate_results(
    strategy=BeamSearchStrategy(beam_width=3, n_partition_candidates=5)
)
print(f"Cut edges: {nx.cut_size(graph, cut)}")
print(f"Circuits executed: {ensemble.total_circuit_count}")
```

``PartitioningProgramEnsemble`` handles decomposition, parallel execution, and
candidate aggregation while each partition remains small enough for the chosen
backend.

## 🌐 Cloud Execution with Qoro Service

Run the same workflow on Qoro's cloud platform by swapping only the backend:

```python
from divi.backends import QoroService

backend = QoroService()  # reads QORO_API_KEY from .env or environment
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
| **Program Ensembles** | Parallel execution of multiple quantum programs with automatic scheduling, over one round or many adaptive ones |
| **Flexible Backends** | `MaestroSimulator` for local simulation, `QiskitSimulator` for Qiskit-native noise models (extra: `aer`), `QoroService` for cloud execution |
| **Execution Config** | Control bond dimension, simulator type, and simulation method per job |
| **Live Reporting** | Real-time dashboards and convergence tracking via callbacks |

## 🏗️ Architecture

```
divi/
├── qprog/        # Quantum programs: VQE, QAOA, base classes, optimizers
├── backends/     # Execution backends: MaestroSimulator, QiskitSimulator (extra: aer), QoroService
├── circuits/     # MetaCircuit templates and Circuit instances
├── pipeline/     # Circuit pipeline stages (expand, execute, reduce)
├── hamiltonians  # Molecular Hamiltonian generation
├── reporting/    # Live reporting and visualization callbacks
└── ai/           # Offline documentation chatbot (divi-ai)
```

## 📚 Documentation

Algorithm guides, execution guides, and API reference: **[divi.readthedocs.io](https://divi.readthedocs.io)**

Hands-on examples are in the [`tutorials/`](tutorials/) folder.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and code style guidelines.

## 📄 License

Apache 2.0 — see [LICENSE](LICENSES/Apache-2.0.txt) for details.
