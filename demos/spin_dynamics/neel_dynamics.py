# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""
Néel State Dynamics — QASM Export / Reload Demo.

Two-phase workflow:

  Phase 1 (export):  Build Trotter circuits and save them as QASM files.
     python neel_dynamics.py export --n-qubits 15

  Run:               Run the exact Trotter circuits on the local backend (and on
                     QoroService with --cloud), reload externally processed
                     QASM circuits if a directory is given, run QDrift, and
                     plot all series.
     python neel_dynamics.py run --n-qubits 15 --compressed-dir compressed_circuits

``run`` reads ``times.npy`` and the QASM files written by ``export``; the QASM
files can be processed externally before ``run``.
"""

import argparse
import os

import numpy as np
from plotting import plot_dynamics
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp
from spin_dynamics import build_tfim_hamiltonian

from divi.backends import JobConfig, MaestroSimulator, QoroService
from divi.circuits import dag_to_qasm_body
from divi.hamiltonians.trotterization import ExactTrotterization, QDrift, TrotterSpec
from divi.qprog.ensembles import BatchConfig, BatchMode, TimeEvolutionTrajectory
from divi.qprog.initial_states import CustomPerQubitState

BASIS_GATES = ["h", "x", "s", "sdg", "rx", "rz", "cx"]


def run_trajectory(
    hamiltonian, observable, times, strategy, backend, neel_state, n_steps, order
):
    """Return ⟨observable⟩ at every time point from one trajectory run."""
    trajectory = TimeEvolutionTrajectory(
        hamiltonian=hamiltonian,
        time_points=list(times),
        observable=observable,
        backend=backend,
        trotterization_strategy=strategy,
        n_steps=n_steps,
        order=order,
        initial_state=CustomPerQubitState(neel_state),
    )
    trajectory.run(batch_config=BatchConfig(mode=BatchMode.OFF))
    results = trajectory.aggregate_results()
    return [results[t] for t in times]


def build_neel_qasm_body(hamiltonian, neel_state, time, n_steps, order):
    """Return the QASM body of the Néel-state Trotter evolution."""
    n_qubits = hamiltonian.num_qubits
    circuit = QuantumCircuit(n_qubits)
    circuit.compose(
        CustomPerQubitState(neel_state).build(range(n_qubits)), inplace=True
    )
    result = ExactTrotterization().process_hamiltonian(TrotterSpec(hamiltonian))
    circuit = result.synthesize_evolution(
        circuit,
        time=time,
        n_steps=n_steps,
        order=order,
        qubits=list(range(n_qubits)),
        basis_gates=BASIS_GATES,
    )
    return dag_to_qasm_body(circuit_to_dag(circuit), precision=10)


def load_and_execute_qasm(qasm_dir, times, backend, neel_state):
    """Load QASM files from a directory, execute them, and return magnetisations.

    Returns:
        list of <Z_0> values, one per time point (including t=0).
    """
    circuits = {}
    time_by_label = {}
    for i, t in enumerate(times):
        if t == 0.0:
            continue
        filepath = os.path.join(qasm_dir, f"neel_t{i:03d}.qasm")
        if not os.path.exists(filepath):
            print(f"   Missing: {filepath}")
            continue
        label = f"neel_t{i:03d}"
        with open(filepath, "r") as f:
            circuits[label] = f.read()
        time_by_label[label] = t

    print(f"   Loaded {len(circuits)} QASM circuits from {qasm_dir}/")
    print("   Submitting to backend...")

    result = backend.submit_circuits(circuits)

    if result.is_async():
        print(f"   Job {result.job_id} submitted, polling for completion...")
        backend.poll_job_status(result, loop_until_complete=True, verbose=True)
        result = backend.get_job_results(result)

    if result.results is None:
        raise RuntimeError(f"Backend returned no results for job {result.job_id}.")

    mag_by_time = {}
    for entry in result.results:
        label = entry["label"]
        counts = entry["results"]
        t = time_by_label[label]

        total_shots = sum(counts.values())
        p0, p1 = 0, 0
        for bitstring, count in counts.items():
            # Qubit 0 is the rightmost character of the bitstring.
            if bitstring[-1] == "0":
                p0 += count
            else:
                p1 += count
        mag_by_time[t] = (p0 - p1) / total_shots

    magnetizations = []
    for t in times:
        if t == 0.0:
            magnetizations.append(-1.0 if neel_state[0] == "1" else 1.0)
        else:
            magnetizations.append(mag_by_time.get(t, 0.0))

    return magnetizations


def export_circuits(args):
    """Build one Exact-Trotter circuit per time step and save as QASM."""
    N_QUBITS = args.n_qubits
    N_STEPS = args.n_steps
    ORDER = args.order
    T_MAX = args.t_max
    N_POINTS = args.n_points
    output_dir = args.output_dir

    times = np.linspace(0, T_MAX, N_POINTS)
    neel_state = "10" * (N_QUBITS // 2) + ("1" if N_QUBITS % 2 else "")

    H = build_tfim_hamiltonian(n_qubits=N_QUBITS, J=1.0, h=0.5)

    print(f"   Hamiltonian: {N_QUBITS}-qubit TFIM (J=1.0, h=0.5)")
    print(f"   Néel state:  |{neel_state}⟩")
    print(
        f"   Time range:  0 to {T_MAX} "
        f"({N_POINTS} points, {N_STEPS} Trotter steps, order {ORDER})"
    )

    os.makedirs(output_dir, exist_ok=True)

    for f in os.listdir(output_dir):
        if f.endswith(".qasm"):
            os.remove(os.path.join(output_dir, f))

    times_file = os.path.join(output_dir, "times.npy")
    np.save(times_file, times)

    count = 0
    for i, t in enumerate(times):
        if t == 0.0:
            continue

        body_qasm = build_neel_qasm_body(H, neel_state, t, N_STEPS, ORDER)

        measure_lines = "".join(f"measure q[{w}] -> c[{w}];\n" for w in range(N_QUBITS))
        qasm_header = (
            "OPENQASM 2.0;\n"
            'include "qelib1.inc";\n'
            f"qreg q[{N_QUBITS}];\n"
            f"creg c[{N_QUBITS}];\n"
        )
        full_qasm = qasm_header + body_qasm + measure_lines

        filepath = os.path.join(output_dir, f"neel_t{i:03d}.qasm")
        with open(filepath, "w") as f:
            f.write(full_qasm)
        count += 1

    print(f"\n   Saved {count} QASM circuits to {output_dir}/")
    print(f"   Time metadata saved to {times_file}")
    print(f"\n   Next step: python neel_dynamics.py run --n-qubits {N_QUBITS}")


def run_comparison(args):
    """Run exact trajectories, reload compressed circuits, run QDrift, and plot."""
    N_QUBITS = args.n_qubits
    N_STEPS = args.n_steps
    ORDER = args.order
    SHOTS = args.shots
    SAMPLING_BUDGET = args.sampling_budget
    original_dir = args.output_dir
    compressed_dir = args.compressed_dir

    neel_state = "10" * (N_QUBITS // 2) + ("1" if N_QUBITS % 2 else "")

    local_backend = MaestroSimulator(shots=SHOTS, track_depth=True)
    print("Local backend: MaestroSimulator")
    if args.cloud:
        cloud_backend = QoroService(
            job_config=JobConfig(shots=SHOTS, force_sampling=True), track_depth=True
        )
        print("Cloud backend: QoroService")
    else:
        cloud_backend = None

    times_file = os.path.join(original_dir, "times.npy")
    if not os.path.exists(times_file):
        print(f"   Time metadata not found at {times_file}")
        print(f"   Run 'python neel_dynamics.py export --n-qubits {N_QUBITS}' first.")
        return
    times = np.load(times_file)

    H = build_tfim_hamiltonian(n_qubits=N_QUBITS, J=1.0, h=0.5)
    observable = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=N_QUBITS)
    hw_backend = cloud_backend if cloud_backend else local_backend
    hw_label = "QoroService" if cloud_backend else "local MaestroSimulator"

    print("\n" + "=" * 70)
    print("  1. Exact Trotterisation — local MaestroSimulator")
    print("=" * 70)

    mag_local = run_trajectory(
        H,
        observable,
        times,
        ExactTrotterization(),
        local_backend,
        neel_state,
        N_STEPS,
        ORDER,
    )

    print("   Local exact Trotterisation complete")
    print(
        f"   Avg circuit depth: {local_backend.average_depth():.0f} "
        f"(std: {local_backend.std_depth():.0f})"
    )
    local_backend.clear_depth_history()

    step = 2
    mag_exact_hw = None
    if cloud_backend:
        print("\n" + "=" * 70)
        print(f"  {step}. Exact Trotterisation — {hw_label}")
        print("=" * 70)

        mag_exact_hw = run_trajectory(
            H,
            observable,
            times,
            ExactTrotterization(),
            hw_backend,
            neel_state,
            N_STEPS,
            ORDER,
        )

        print(f"   Exact Trotterisation ({hw_label}) complete")
        print(
            f"   Avg circuit depth: {hw_backend.average_depth():.0f} "
            f"(std: {hw_backend.std_depth():.0f})"
        )
        hw_backend.clear_depth_history()
        step += 1

    mag_compressed = None
    if compressed_dir and os.path.isdir(compressed_dir):
        print("\n" + "=" * 70)
        print(f"  {step}. Compressed circuits — {hw_label} ({compressed_dir}/)")
        print("=" * 70)
        step += 1

        mag_compressed = load_and_execute_qasm(
            compressed_dir, times, hw_backend, neel_state
        )
        print("   Compressed circuits complete")
        print(
            f"   Avg circuit depth: {hw_backend.average_depth():.0f} "
            f"(std: {hw_backend.std_depth():.0f})"
        )
        hw_backend.clear_depth_history()
    else:
        print("\n   No compressed circuit directory provided, skipping.")
        print("      Use --compressed-dir <path> to include compressed results.")

    print("\n" + "=" * 70)
    print(f"  {step}. QDrift — {hw_label} (sampling_budget={SAMPLING_BUDGET})")
    print("=" * 70)

    mag_qdrift = run_trajectory(
        H,
        observable,
        times,
        QDrift(
            sampling_budget=SAMPLING_BUDGET,
            keep_fraction=0.75,
            sampling_strategy="weighted",
        ),
        hw_backend,
        neel_state,
        N_STEPS,
        1,
    )

    print("   QDrift complete")
    print(
        f"   Avg circuit depth: {hw_backend.average_depth():.0f} "
        f"(std: {hw_backend.std_depth():.0f})"
    )
    hw_backend.clear_depth_history()

    series = [
        (
            "Trotter, MaestroSimulator expectation value",
            times,
            mag_local,
            "o-",
            "#6366f1",
        ),
    ]
    if mag_exact_hw is not None:
        series.append(
            (f"Exact Trotter ({hw_label})", times, mag_exact_hw, "D-", "#3b82f6")
        )
    if mag_compressed is not None:
        series.append(
            (f"Compressed ({hw_label})", times, mag_compressed, "^-", "#10b981")
        )
    series.append(
        (
            f"QDrift ({hw_label}, budget={SAMPLING_BUDGET})",
            times,
            mag_qdrift,
            "s--",
            "#f97316",
        )
    )
    plot_dynamics(
        series,
        f"Néel State |{neel_state}⟩ Dynamics (J=1.0, h=0.5)",
        "dynamics_neel.png",
    )

    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Néel State Dynamics — QASM Export / Reload Demo"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--n-qubits", type=int, default=15, help="Number of qubits (default: 15)"
    )
    shared.add_argument(
        "--n-steps", type=int, default=5, help="Trotter steps (default: 5)"
    )
    shared.add_argument(
        "--order", type=int, default=2, help="Suzuki-Trotter order (default: 2)"
    )
    shared.add_argument(
        "--t-max", type=float, default=3.0, help="Max evolution time (default: 3.0)"
    )
    shared.add_argument(
        "--n-points", type=int, default=15, help="Number of time points (default: 15)"
    )
    shared.add_argument(
        "--output-dir",
        type=str,
        default="qasm_circuits",
        help="QASM output directory (default: qasm_circuits)",
    )

    export_parser = subparsers.add_parser(
        "export", parents=[shared], help="Build circuits and save as QASM"
    )
    export_parser.set_defaults(func=export_circuits)

    run_parser = subparsers.add_parser(
        "run",
        parents=[shared],
        help="Run exact and QDrift trajectories, reload compressed QASM, and plot",
    )
    run_parser.add_argument(
        "--shots", type=int, default=10_000, help="Measurement shots (default: 10000)"
    )
    run_parser.add_argument(
        "--sampling-budget",
        type=int,
        default=50,
        help="QDrift sampling budget (default: 50)",
    )
    run_parser.add_argument(
        "--compressed-dir",
        type=str,
        default=None,
        help="Directory with compressed QASM circuits (optional)",
    )
    run_parser.add_argument(
        "--cloud",
        action="store_true",
        help="Run the comparisons on QoroService (otherwise on the local backend)",
    )
    run_parser.set_defaults(func=run_comparison)

    args = parser.parse_args()
    args.func(args)
