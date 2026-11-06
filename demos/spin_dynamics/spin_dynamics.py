# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""
Transverse-Field Ising Model (TFIM) spin dynamics via Divi.

``TimeEvolutionTrajectory`` creates one ``TimeEvolution`` program per time point
and runs them through ``ProgramEnsemble``.

Tracks the magnetisation ⟨Z₀⟩ for J=1.0 with h=0.2, 2.0 and 0.5. The first two
start in |0…0⟩ and the last in a Néel product state. Each is run with exact
Trotterisation and with QDrift.
"""

import time
from dataclasses import dataclass

import numpy as np
from plotting import compute_trajectory_error, plot_dynamics, plot_trajectory_error
from qiskit.quantum_info import SparsePauliOp

from divi.backends import JobConfig, MaestroSimulator, QoroService
from divi.hamiltonians.trotterization import ExactTrotterization, QDrift
from divi.qprog.ensembles import TimeEvolutionTrajectory
from divi.qprog.initial_states import CustomPerQubitState, InitialState, ZerosState


def build_tfim_hamiltonian(n_qubits: int, J: float, h: float) -> SparsePauliOp:
    """Build the Transverse-Field Ising Model (TFIM) Hamiltonian.

    H = -J Σ_i Z_i Z_{i+1}  -  h Σ_i X_i
    """
    zz_terms = [("ZZ", [i, i + 1], -J) for i in range(n_qubits - 1)]
    x_terms = [("X", [i], -h) for i in range(n_qubits)]
    return SparsePauliOp.from_sparse_list(zz_terms + x_terms, num_qubits=n_qubits)


def run_trajectory(
    hamiltonian,
    time_points,
    strategy,
    backend,
    n_steps=6,
    initial_state: InitialState | None = None,
):
    """Create and run a TimeEvolutionTrajectory, return sorted (times, mags)."""
    trajectory = TimeEvolutionTrajectory(
        hamiltonian=hamiltonian,
        time_points=time_points,
        observable=SparsePauliOp.from_sparse_list(
            [("Z", [0], 1.0)], num_qubits=hamiltonian.num_qubits
        ),
        backend=backend,
        trotterization_strategy=strategy,
        n_steps=n_steps,
        order=1,
        initial_state=initial_state,
    )
    trajectory.run()

    results = trajectory.aggregate_results()
    sorted_t = sorted(results.keys())
    return sorted_t, [results[t] for t in sorted_t]


@dataclass
class Experiment:
    name: str
    J: float
    h: float
    initial_state: InitialState
    filename: str


def run_experiment(
    exp: Experiment, n_qubits: int, n_steps: int, time_points, backend
) -> None:
    """Build H, run Exact + QDrift trajectories, save the comparison and error plots."""
    print(f"\n=== {exp.name} (J={exp.J}) ===")
    H = build_tfim_hamiltonian(n_qubits=n_qubits, J=exp.J, h=exp.h)

    t0 = time.time()
    t_exact, m_exact = run_trajectory(
        H,
        time_points,
        ExactTrotterization(),
        backend,
        n_steps=n_steps,
        initial_state=exp.initial_state,
    )
    print(f"  Exact   ({len(time_points)} points): {time.time() - t0:.1f}s")

    t0 = time.time()
    t_qdrift, m_qdrift = run_trajectory(
        H,
        time_points,
        QDrift(sampling_budget=10),
        backend,
        n_steps=n_steps,
        initial_state=exp.initial_state,
    )
    print(f"  QDrift  ({len(time_points)} points): {time.time() - t0:.1f}s")

    plot_dynamics(
        [
            ("Exact Trotterisation", t_exact, m_exact, "o-", "#3b82f6"),
            ("QDrift (stochastic)", t_qdrift, m_qdrift, "s--", "#f97316"),
        ],
        title=f"Spin Dynamics: {exp.name} (J={exp.J})",
        filename=exp.filename,
    )
    plot_trajectory_error(
        t_exact,
        compute_trajectory_error(m_exact, m_qdrift),
        exp.name,
        exp.filename.replace(".png", "_error.png"),
    )


if __name__ == "__main__":
    N_QUBITS = 6
    N_STEPS = 5
    T_MAX = 3.0
    N_POINTS = 6
    SHOTS = 5_000
    USE_CLOUD = False

    backend = (
        QoroService(job_config=JobConfig(shots=SHOTS))
        if USE_CLOUD
        else MaestroSimulator(shots=SHOTS)
    )

    time_points = np.linspace(0, T_MAX, N_POINTS).tolist()

    experiments = [
        Experiment(
            name="h=0.2, zeros state",
            J=1.0,
            h=0.2,
            initial_state=ZerosState(),
            filename="dynamics_h0p2_zeros.png",
        ),
        Experiment(
            name="h=2.0, zeros state",
            J=1.0,
            h=2.0,
            initial_state=ZerosState(),
            filename="dynamics_h2p0_zeros.png",
        ),
        Experiment(
            name="h=0.5, Néel state",
            J=1.0,
            h=0.5,
            initial_state=CustomPerQubitState(
                "10" * (N_QUBITS // 2) + ("1" if N_QUBITS % 2 else "")
            ),
            filename="dynamics_h0p5_neel.png",
        ),
    ]

    for exp in experiments:
        run_experiment(exp, N_QUBITS, N_STEPS, time_points, backend)
