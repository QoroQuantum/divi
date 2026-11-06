# Divi Demos

Example programs that use the [Divi](https://github.com/QoroQuantum/divi) quantum-programming framework. Single-file examples are in [`tutorials/`](../tutorials).

## Installation

```bash
pip install "qoro-divi[jupyter,qubo-decompose]"
```

An example that needs further packages lists them in a `requirements.txt` in its own directory.

Examples use `MaestroSimulator` unless stated. A section that uses `QoroService` needs service credentials.

## Examples

- [Economic Load Dispatch](./economic_load_dispatch) — formulate constrained generation dispatch as a `BinaryOptimizationProblem` with linear constraints and run it with PCE.
- [Minimum Birkhoff Decomposition](./minimum_birkhoff_decomposition) — combine a standalone `CircuitPipeline` with a classical CPLEX post-processor.
- [Portfolio Optimisation](./portfolio_optimization) — build a Markowitz QUBO and partition it before solving with QAOA or PCE.
- [Quantum-Guided Cluster Algorithm](./quantum_guided_cluster) — use QAOA or PCE correlations to guide Max-Cut cluster moves.
- [Spin Dynamics](./spin_dynamics) — evolve a transverse-field Ising chain with `TimeEvolutionTrajectory`.
- [Travelling Salesman Problem](./travelling_salesman) — run direct QAOA, partitioned QAOA, and PCE on a TSP.

## Working Through a Demo

Each demo directory has a README. A demo contains scripts, a notebook or both.

## License

See [LICENSE](../LICENSE) for details.
