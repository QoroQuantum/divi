# Travelling Salesman Problem with QAOA and PCE

Three ways to run the same cyclic-tour problem on randomly placed cities:

1. Direct QAOA on `TSPProblem`, with a W-state start and an XY mixer on each city's block of qubits.
2. Partitioned QAOA: `TSPProblem(max_cities_per_partition=...)` clusters the cities, and `PartitioningProgramEnsemble` runs each cluster's sub-tour and patches the sub-tours into one tour.
3. PCE on a `BinaryOptimizationProblem` whose objective is the tour length and whose constraints are one city per step and one step per city.

Sampled solutions that are not valid tours are repaired with the Hungarian algorithm.

## Run locally

```bash
python travelling_salesman.py
```

Or open `travelling_salesman.ipynb` for the guided version.

`N_CITIES_SMALL` sets the direct-QAOA and PCE instance, and `N_CITIES_LARGE` sets the partitioned instance. The brute-force tour is printed as a reference.
