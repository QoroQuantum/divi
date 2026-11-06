# Economic Load Dispatch with Prohibited Operating Zones

This example formulates fuel-cost minimisation under a demand constraint and prohibited operating zones.

It demonstrates how to state a constrained dispatch problem as a `BinaryOptimizationProblem` with linear constraints, solve it with PCE, decode the sampled assignments, and apply a repair step to the decoded dispatches.

## Run locally

```bash
python economic_load_dispatch.py
```

The notebook provides the same walkthrough:

```bash
jupyter notebook economic_load_dispatch.ipynb
```

## Concepts

- Four binary variables encode the output of each generator.
- The demand is an equality constraint and each prohibited zone an inequality constraint; the demand adds no qubits and each zone adds one slack qubit.
- PCE encodes each variable as the parity of one or two qubits (`PolyEncoding`).
- The repair step moves decoded generator powers out of prohibited zones, then adjusts them towards the demand.
- Exhaustive enumeration is the reference for the three-generator instance; simulated annealing is the reference for the six-generator variant.

The script starts with a three-generator instance and also includes a six-generator variant for experimenting with the same formulation. Both can use a local simulator.
