# Divi Tutorials

This directory contains runnable Python examples demonstrating Divi's capabilities. Each example is self-contained and can be run independently.

## Quick Start

To run any example:

```bash
cd tutorials
python <example_name>.py
```

### Backend flags

Tutorials that use `get_backend()` from `_backend.py` accept these CLI flags:

| Flag | Backend | Description |
|------|---------|-------------|
| `--local-qiskit` (default) | `QiskitSimulator` | Local Qiskit Aer simulation |
| `--local-maestro` | `MaestroSimulator` | Local Maestro orchestration |
| `--cloud-maestro` | `QoroService` | Qoro cloud backend |
| `--force-sampling` | — | Disable exact expectation values; use shot-based sampling instead |

```bash
python qaoa_max_clique.py --local-qiskit
python qaoa_max_clique.py --cloud-maestro
python qaoa_max_clique.py --local-qiskit --force-sampling
```

## Examples by Category

### Chemistry

- **`vqe_h2_molecule_local.py`** - Basic VQE calculation for H2 molecule
- **`vqe_h2_with_grouping.py`** - VQE with wire grouping for efficiency
- **`vqe_hyperparameter_sweep.py`** - Hyperparameter sweeps across multiple molecules

### Optimization (QUBO & Binary Models)

- **`qaoa_qubo.py`** - QAOA for QUBO optimization problems
- **`qaoa_hubo.py`** - QAOA for Higher-Order Binary Optimization (HUBO) problems
- **`qaoa_binary_quadratic_model.py`** - QAOA for BinaryQuadraticModel inputs
- **`qaoa_qubo_partitioning.py`** - QUBO partitioning for scalability
- **`pce_qubo.py`** - PCE-VQE for a QUBO problem

### Optimization (Graph Problems)

- **`qaoa_max_clique_local.py`** - Basic QAOA for maximum clique problem
- **`qaoa_max_weight_matching.py`** - Maximum-weight matching with QAOA (standalone and partitioned)
- **`qaoa_graph_partitioning.py`** - Large graph partitioning with QAOA
- **`qaoa_qdrift_local.py`** - QAOA with QDrift randomized Trotterization
- **`iterative_qaoa.py`** - Iterative QAOA with parameter interpolation vs standard QAOA

### Optimization (Routing)

- **`ce_qaoa_tsp.py`** - TSP with Constraint-Enhanced QAOA: grid search, parameter transfer, feasibility stats, repair
- **`ce_qaoa_cvrp.py`** - CVRP with CE-QAOA: one-hot vs binary encoding, qubit projections, VRP file parser

### Quadratic Programming

- **`qaoa_quadratic_program.py`** - QAOA for quadratic programming

### Error Mitigation

- **`zne_local.py`** - VQE with Zero Noise Extrapolation error mitigation

### Dynamics

- **`time_evolution_local.py`** - Hamiltonian time evolution with probs, observables, and QDrift

### Custom Workflows

- **`custom_vqa.py`** - CustomVQA with QuantumScript and Qiskit inputs

### Checkpointing

- **`checkpointing.py`** - Save and resume optimization runs with checkpointing

### Backends and Services

- **`qasm_thru_service.py`** - Using QoroService cloud backend
- **`backend_properties_conversion.py`** - Converting `BackendProperties` to A Qiskit `BackendV2`

## Requirements

All examples require Divi to be installed:

```bash
pip install qoro-divi
```

## Documentation

For comprehensive documentation and detailed explanations, see the [User Guide](https://divi.readthedocs.io/en/latest/user_guide/).

## Contributing

When adding new examples:

1. Keep examples focused and self-contained
2. Add clear docstrings explaining the example's purpose
3. Include expected output in comments
4. Update this README with a brief description
5. **Register in `_ci_runner.py`**: All tutorials are executed in CI to catch regressions — if a code change breaks a tutorial, the pipeline fails. To keep CI fast, tutorials run with reduced parameters (fewer iterations, smaller problems, lower shots). The runner copies tutorials to a temp directory, applies string patches, and runs them in parallel with per-tutorial timeouts.

   Every non-underscore `.py` file must appear in exactly one of these three lists (validated at startup):
   - `SKIP` — tutorials that cannot run in CI (e.g. need API keys)
   - `NO_PATCHES` — tutorials that run as-is (shots capped by `DIVI_CI_MAX_SHOTS`)
   - `TUTORIALS` — tutorials that need source patches (e.g. fewer iterations, smaller problems) and/or custom timeouts
