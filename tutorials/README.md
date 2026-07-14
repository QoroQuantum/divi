# Divi Tutorials

This directory contains runnable Python examples demonstrating Divi's capabilities. Each example is self-contained and can be run independently.

## Quick Start

To run any example, point Python at the file from the repo root:

```bash
python tutorials/optimization/qaoa_graph_problems.py
```

### Backend flags

Tutorials that use `get_backend()` from `_backend.py` accept these CLI flags:

| Flag | Backend | Description |
| ------ | --------- | ------------- |
| `--local-qiskit` (default) | `QiskitSimulator` | Local Qiskit Aer simulation |
| `--local-maestro` | `MaestroSimulator` | Local Maestro orchestration |
| `--cloud-maestro` | `QoroService` | Qoro cloud backend |
| `--force-sampling` | — | Disable exact expectation values; use shot-based sampling instead |

```bash
python tutorials/optimization/qaoa_graph_problems.py --local-qiskit
python tutorials/optimization/qaoa_graph_problems.py --cloud-maestro
python tutorials/optimization/qaoa_graph_problems.py --local-qiskit --force-sampling
```

## Layout

```
tutorials/
├── optimization/         QAOA / PCE / partitioning on QUBO and graph problems
├── routing/              Constraint-Enhanced QAOA on TSP and CVRP
├── chemistry/            VQE on molecular Hamiltonians
├── dynamics/             Hamiltonian time evolution and trajectories
├── error_mitigation/     ZNE / QuEPP wrapped around a VQA loop
├── visualization/        Loss-landscape and parameter-space analysis tools
├── advanced/             BYO circuits, custom optimization, checkpointing
└── backends/             Qoro cloud and Qiskit backend integration
```

### `optimization/`

Two scaling axes: solver choice (QAOA vs PCE) and problem size (single program vs partitioned ensemble).

- **`qubo_qaoa_vs_pce.py`** — Solving the same QUBO with QAOA and PCE side by side
- **`qaoa_hubo.py`** — QAOA for Higher-Order Binary Optimization (HUBO) problems
- **`qaoa_partitioning.py`** — Partition a large problem into many sub-programs (graph, QUBO, and edge-based entry points)
- **`qaoa_graph_problems.py`** — QAOA on max clique and max-weight matching
- **`qaoa_qdrift.py`** — QAOA with QDrift randomized Trotterization
- **`iterative_qaoa.py`** — Iterative QAOA with parameter interpolation vs standard QAOA

### `routing/`

- **`ce_qaoa_routing.py`** — Constraint-Enhanced QAOA on TSP (grid search → parameter transfer → repair) and CVRP (one-hot vs binary encoding, qubit projections, VRP file parser)

### `chemistry/`

- **`vqe_h2_molecule.py`** — VQE on H2: basic run plus grouping-strategy and shot-allocation comparisons
- **`vqe_hyperparameter_sweep.py`** — Hyperparameter sweeps across multiple molecules

### `dynamics/`

- **`time_evolution.py`** — Hamiltonian time evolution: probabilities, observables, multi-observable groups, QDrift, and trajectories over many time points

### `error_mitigation/`

- **`error_mitigation.py`** — VQE with Zero Noise Extrapolation (ZNE) and probabilistic error amplification

### `visualization/`

- **`viz_qaoa_pce_comparison.py`** — Compare QAOA and PCE loss landscapes: 1D scans, 2D scans, PCA scans with trajectory overlay
- **`viz_advanced_analysis.py`** — Interpolation scans, Hessian eigenvalue analysis, Fourier power spectra, gradient overlays, 3D surface plots, NEB minimum-energy paths

### `advanced/`

Escape hatches and specialized algorithms beyond the core QAOA/VQE flow: run circuits without a `QuantumProgram`, plug your own circuit into a VQA loop, train a quantum neural network, or save and resume long-running optimizations.

- **`standalone_pipeline.py`** — One-shot execution of PennyLane / Qiskit circuits through `CircuitPipeline` directly (no `QuantumProgram` wrapper, no optimization loop)
- **`custom_vqa.py`** — Bring your own circuit (PennyLane `QNode`/`QuantumScript` or Qiskit) to a VQA loop via `CustomVQA`, building from a toy observable up to data binding (`data_param_indices`) and multi-argument QNNs (template feature maps + ansatz via `arg_shapes`/`data_arg`, including nonlinear `IQPEmbedding`)
- **`qnn_classifier.py`** — Train a supervised quantum classifier with `QNN`: an `AngleEmbedding` feature map plus a `GenericLayerAnsatz`, scored by mean-squared error over a labeled feature batch
- **`checkpointing.py`** — Save and resume optimization runs

### `backends/`

- **`qasm_thru_service.py`** — Submit raw QASM circuits to the Qoro cloud
- **`characterize_maxcut_qubo.py`** — Get a regime/certificate + classical baseline before running QAOA, and skip optimization via the Qoro Characterization Service
- **`backend_properties_conversion.py`** — Converting Qiskit `BackendProperties` to a `BackendV2`

## Requirements

All examples require Divi to be installed:

```bash
pip install qoro-divi
```

## Documentation

For comprehensive documentation and detailed explanations, see the [User Guide](https://divi.readthedocs.io/en/latest/user_guide/).

## Contributing

When adding new examples:

1. Drop the file into the folder that matches its primary topic (or open a new folder if it's a genuinely new category that maps to a user-guide page).
2. Keep examples focused and self-contained.
3. Add clear docstrings explaining the example's purpose.
4. Include expected output in comments.
5. Update this README with a brief description under the right folder section.
6. **Register in `_ci_runner.py`**: All tutorials are executed in CI to catch regressions — if a code change breaks a tutorial, the pipeline fails. To keep CI fast, tutorials run with reduced parameters (fewer iterations, smaller problems, lower shots). The runner copies tutorials to a temp directory, applies string patches, and runs them in parallel with per-tutorial timeouts.

   Every non-underscore `.py` file must appear in exactly one of these three lists (validated at startup), keyed by its **path relative to `tutorials/`** (e.g. `"optimization/qaoa_hubo.py"`):
   - `SKIP` — tutorials that cannot run in CI (e.g. need API keys)
   - `NO_PATCHES` — tutorials that run as-is (shots capped by `DIVI_CI_MAX_SHOTS`)
   - `TUTORIALS` — tutorials that need source patches (e.g. fewer iterations, smaller problems) and/or custom timeouts
