# Quantum-Guided Cluster Algorithm for Max-Cut

Based on [arXiv:2508.10656](https://arxiv.org/abs/2508.10656). Two-point correlations from QAOA or PCE drive a cluster Monte Carlo for Max-Cut.

`main.py` runs simulated annealing, a cluster algorithm with coupling-constant correlations, and cluster algorithms with correlations from QAOA; `pce_encodings` adds PCE extractors.

## Run locally

```bash
python main.py
```

Or open `quantum_guided_cluster.ipynb`.

## Parameters to explore

- `n_nodes` and `degree`: the random regular Max-Cut graph.
- `qaoa_depths`: QAOA depths for correlation extraction.
- `pce_encodings`: optional `dense` or `poly` PCE extractors.
- `lambda_scale`: scales the cluster-link probability.
