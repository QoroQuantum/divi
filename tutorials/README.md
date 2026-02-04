# Divi Tutorials

This directory contains runnable Python examples demonstrating Divi's capabilities. Each example is self-contained and can be run independently.

## Quick Start

To run any example:

```bash
cd tutorials
python <example_name>.py
```

## Examples by Category

### Chemistry

- **`vqe_h2_molecule_local.py`** - Basic VQE calculation for H2 molecule
- **`vqe_h2_with_grouping.py`** - VQE with wire grouping for efficiency
- **`vqe_hyperparameter_sweep.py`** - Hyperparameter sweeps across multiple molecules

### Optimization (QUBO & Binary Models)

- **`qaoa_qubo.py`** - QAOA for QUBO optimization problems
- **`qaoa_binary_quadratic_model.py`** - QAOA for BinaryQuadraticModel inputs
- **`qaoa_qubo_partitioning.py`** - QUBO partitioning for scalability
- **`pce_qubo.py`** - PCE-VQE for a QUBO problem

### Optimization (Graph Problems)

- **`qaoa_max_clique_local.py`** - Basic QAOA for maximum clique problem
- **`qaoa_graph_partitioning.py`** - Large graph partitioning with QAOA
- **`qaoa_qdrift_local.py`** - QAOA with QDrift randomized Trotterization

### Quadratic Programming

- **`qaoa_quadratic_program.py`** - QAOA for quadratic programming

### Error Mitigation

- **`zne_local.py`** - VQE with Zero Noise Extrapolation error mitigation

### Custom Workflows

- **`custom_vqa.py`** - CustomVQA with QuantumScript and Qiskit inputs

### Checkpointing

- **`checkpointing.py`** - Save and resume optimization runs with checkpointing

### Backends and Services

- **`qasm_thru_service.py`** - Using QoroService cloud backend
- **`circuit_cutting.py`** - Circuit cutting with cloud backends
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
