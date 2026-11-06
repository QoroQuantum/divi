# Spin Dynamics of the Transverse-Field Ising Model

These scripts simulate a one-dimensional transverse-field Ising chain,

$$ H = -J \sum_i Z_i Z_{i+1} - h \sum_i X_i. $$

`spin_dynamics.py` compares exact Trotterisation and QDrift across several time points using `TimeEvolutionTrajectory`. `neel_dynamics.py` additionally demonstrates QASM export, optional external circuit processing, and reloading the circuits for execution.

## Run locally

```bash
python spin_dynamics.py
python neel_dynamics.py export --n-qubits 15
python neel_dynamics.py run --n-qubits 15
```

`neel_dynamics.py run --cloud` runs the comparisons on QoroService and requires configured credentials. Without it, every comparison runs on `MaestroSimulator`.
