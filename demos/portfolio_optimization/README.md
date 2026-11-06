# Portfolio Optimisation with Quantum Algorithms

This notebook builds a Markowitz risk-return objective as a QUBO, partitions it into communities, and solves the partitions with QAOA.

It illustrates `CommunityDecomposer`, Divi's `PartitioningProgramEnsemble`, and beam-search aggregation of partition candidates. A snippet shows how to switch the partition solver to PCE. The supplied data and default notebook path use `MaestroSimulator` locally.

## Run

```bash
jupyter notebook portfolio_optimization.ipynb
```

`utils.py` provides the QUBO construction and solution-evaluation helpers.
