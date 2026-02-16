# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import math

import pennylane as qml
import pytest

from divi.backends import ExecutionResult, ParallelSimulator
from divi.hamiltonians import ExactTrotterization, QDrift
from divi.qprog import TimeEvolution

# Tolerance for probability checks (5000 shots: ~0.02 std for p=0.5)
_PROB_TOL = 0.05


@pytest.fixture
def two_qubit_hamiltonian():
    return 0.5 * qml.PauliZ(0) + 0.3 * qml.PauliZ(1)


class TestTimeEvolutionInitialization:
    def test_initialization_valid(self, two_qubit_hamiltonian, default_test_simulator):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            time=1.0,
            backend=default_test_simulator,
        )
        assert te.time == 1.0
        assert te.n_qubits == 2
        assert te.initial_state == "Zeros"
        assert te.observable is None

    def test_initialization_requires_backend(self, two_qubit_hamiltonian):
        with pytest.raises(ValueError, match="requires a backend"):
            TimeEvolution(hamiltonian=two_qubit_hamiltonian, backend=None)

    def test_initialization_constant_hamiltonian_fails(self, default_test_simulator):
        with pytest.raises(ValueError, match="only constant terms"):
            TimeEvolution(
                hamiltonian=qml.Identity(0) * 1.0,
                backend=default_test_simulator,
            )

    def test_initialization_invalid_initial_state(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        with pytest.raises(ValueError, match="initial_state"):
            TimeEvolution(
                hamiltonian=two_qubit_hamiltonian,
                initial_state="Invalid",
                backend=default_test_simulator,
            )

    def test_default_trotterization_strategy(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            backend=default_test_simulator,
        )
        assert isinstance(te.trotterization_strategy, ExactTrotterization)


class TestTimeEvolutionGenerateCircuits:
    def test_pipeline_exact_trotterization_one_circuit(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            trotterization_strategy=ExactTrotterization(),
            backend=default_test_simulator,
        )
        env = te._build_pipeline_env()
        trace = te._pipeline.run_forward_pass(te._hamiltonian, env)
        # ExactTrotterization: 1 Hamiltonian sample → 1 circuit
        assert len(trace.final_batch) >= 1

    def test_pipeline_qdrift_multiple_circuits(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            trotterization_strategy=QDrift(
                sampling_budget=2,
                seed=42,
                n_hamiltonians_per_iteration=3,
            ),
            backend=default_test_simulator,
        )
        env = te._build_pipeline_env()
        trace = te._pipeline.run_forward_pass(te._hamiltonian, env)
        # QDrift with 3 samples → at least 3 circuits
        assert len(trace.final_batch) >= 3


class TestTimeEvolutionRun:
    def test_run_probs_mode(self, two_qubit_hamiltonian, default_test_simulator):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            time=0.5,
            initial_state="Zeros",
            backend=default_test_simulator,
        )
        count, runtime = te.run()
        assert count >= 1
        assert runtime >= 0
        assert "probs" in te.results
        probs = te.results["probs"]
        assert isinstance(probs, dict)
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.1  # Approximate due to shots

    def test_run_initial_state_superposition(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            time=0.5,
            initial_state="Superposition",
            backend=default_test_simulator,
        )
        count, _ = te.run()
        assert count >= 1
        assert "probs" in te.results

    def test_run_initial_state_ones(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            time=0.5,
            initial_state="Ones",
            backend=default_test_simulator,
        )
        count, _ = te.run()
        assert count >= 1

    def test_single_term_hamiltonian_fallback(self, default_test_simulator):
        """ExactTrotterization with keep_top_n=1 yields single-term; use evolve not TrotterProduct."""
        h = 0.5 * qml.PauliX(0) + 0.3 * qml.PauliY(0)
        te = TimeEvolution(
            hamiltonian=h,
            trotterization_strategy=ExactTrotterization(keep_top_n=1),
            time=0.5,
            backend=default_test_simulator,
        )
        count, _ = te.run()
        assert count >= 1
        assert "probs" in te.results


class TestTimeEvolutionObservable:
    def test_run_with_observable_shot_backend(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            time=0.5,
            observable=qml.PauliZ(0),
            backend=default_test_simulator,
        )
        count, _ = te.run()
        assert count >= 1
        assert "expval" in te.results
        assert -1.1 <= te.results["expval"] <= 1.1

    def test_run_with_observable_expval_backend_multi_term(
        self, two_qubit_hamiltonian, dummy_expval_backend, mocker
    ):
        def _deterministic_submit(circuits, **kwargs):
            ham_ops = kwargs.get("ham_ops", "")
            ops = ham_ops.split(";") if ham_ops else []
            payload = {op: 1.0 for op in ops}
            return ExecutionResult(
                results=[
                    {"label": label, "results": payload.copy()}
                    for label in circuits.keys()
                ]
            )

        mocker.patch.object(
            dummy_expval_backend,
            "submit_circuits",
            side_effect=_deterministic_submit,
        )

        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            time=0.5,
            observable=qml.PauliZ(0) + qml.PauliZ(1),
            backend=dummy_expval_backend,
        )
        count, _ = te.run()
        assert count >= 1
        assert "expval" in te.results
        assert te.results["expval"] == pytest.approx(2.0)


class TestTimeEvolutionQDrift:
    def test_qdrift_multi_sample_averages_probs(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            trotterization_strategy=QDrift(
                sampling_budget=2,
                seed=42,
                n_hamiltonians_per_iteration=3,
            ),
            time=0.5,
            backend=default_test_simulator,
        )
        count, _ = te.run()
        assert count >= 3
        assert "probs" in te.results
        probs = te.results["probs"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.2

    def test_qdrift_expval_backend_raises(self, two_qubit_hamiltonian, mocker):
        """Multi-sample QDrift + expval backend is not supported — run() raises."""
        backend = mocker.Mock()
        backend.supports_expval = True
        backend.shots = 100
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            observable=qml.PauliZ(0),
            trotterization_strategy=QDrift(
                sampling_budget=2,
                seed=42,
                n_hamiltonians_per_iteration=2,
            ),
            backend=backend,
        )
        with pytest.raises(ValueError, match="Multi-sample QDrift"):
            te.run()

    def test_expval_shot_backend_qdrift_aggregation(self, default_test_simulator):
        """QDrift with observable on shot backend averages expvals across samples."""
        te = TimeEvolution(
            hamiltonian=qml.PauliX(0),
            observable=qml.PauliZ(0),
            trotterization_strategy=QDrift(
                sampling_budget=1,
                seed=42,
                n_hamiltonians_per_iteration=2,
            ),
            backend=default_test_simulator,
        )
        count, _ = te.run()
        assert count >= 2
        assert "expval" in te.results


@pytest.mark.e2e
class TestTimeEvolutionE2E:
    """E2E tests with well-known Hamiltonians and quantitative checks against analytic results."""

    def test_h_x0_plus_x1_zeros_evolves_to_11(self, default_test_simulator):
        """H=X₀+X₁ (2-qubit, commuting): |00⟩ at t=π/2 → |11⟩, P(11)=1."""
        te = TimeEvolution(
            hamiltonian=qml.PauliX(0) + qml.PauliX(1),
            time=math.pi / 2,
            initial_state="Zeros",
            backend=default_test_simulator,
        )
        te.run()
        probs = te.results["probs"]
        assert probs.get("11", 0.0) >= 1.0 - _PROB_TOL
        for key in ("00", "01", "10"):
            assert probs.get(key, 0.0) <= _PROB_TOL

    def test_h_x_plus_z_full_rotation(self, default_test_simulator):
        """H=X+Z (1-qubit, non-commuting): |0⟩ at t=π/√2 → |0⟩ (full Bloch rotation)."""
        te = TimeEvolution(
            hamiltonian=qml.PauliX(0) + qml.PauliZ(0),
            time=math.pi / math.sqrt(2),
            initial_state="Zeros",
            n_steps=10,
            backend=default_test_simulator,
        )
        te.run()
        probs = te.results["probs"]
        assert probs.get("0", 0.0) >= 1.0 - _PROB_TOL
        assert probs.get("1", 0.0) <= _PROB_TOL

    def test_heisenberg_xx_superposition_uniform(self, default_test_simulator):
        """Heisenberg XX (X₀X₁+Y₀Y₁, non-commuting): |++⟩ stays uniform, P(·)=0.25."""
        te = TimeEvolution(
            hamiltonian=qml.PauliX(0) @ qml.PauliX(1) + qml.PauliY(0) @ qml.PauliY(1),
            time=math.pi / 4,
            initial_state="Superposition",
            backend=default_test_simulator,
        )
        te.run()
        probs = te.results["probs"]
        total = sum(probs.values())
        assert abs(total - 1.0) <= _PROB_TOL
        for key in ("00", "01", "10", "11"):
            assert 0.25 - _PROB_TOL <= probs.get(key, 0.0) <= 0.25 + _PROB_TOL

    def test_heisenberg_xxx_superposition_uniform(self, default_test_simulator):
        """Heisenberg XXX (X₀X₁+Y₀Y₁+Z₀Z₁): |++⟩ stays uniform, P(·)=0.25."""
        te = TimeEvolution(
            hamiltonian=(
                qml.PauliX(0) @ qml.PauliX(1)
                + qml.PauliY(0) @ qml.PauliY(1)
                + qml.PauliZ(0) @ qml.PauliZ(1)
            ),
            time=math.pi / 4,
            initial_state="Superposition",
            backend=default_test_simulator,
        )
        te.run()
        probs = te.results["probs"]
        total = sum(probs.values())
        assert abs(total - 1.0) <= _PROB_TOL
        for key in ("00", "01", "10", "11"):
            assert 0.25 - _PROB_TOL <= probs.get(key, 0.0) <= 0.25 + _PROB_TOL

    def test_heisenberg_xxx_ones_stays_eigenstate(self, default_test_simulator):
        """Heisenberg XXX: |11⟩ is triplet eigenstate, P(11)=1."""
        te = TimeEvolution(
            hamiltonian=(
                qml.PauliX(0) @ qml.PauliX(1)
                + qml.PauliY(0) @ qml.PauliY(1)
                + qml.PauliZ(0) @ qml.PauliZ(1)
            ),
            time=math.pi / 2,
            initial_state="Ones",
            backend=default_test_simulator,
        )
        te.run()
        probs = te.results["probs"]
        assert probs.get("11", 0.0) >= 1.0 - _PROB_TOL
        for key in ("00", "01", "10"):
            assert probs.get(key, 0.0) <= _PROB_TOL

    def test_qdrift_commuting_stays_eigenstate(self, default_test_simulator):
        """QDrift H=Z₀+Z₁: |00⟩ is eigenstate, multi-sample average P(00)=1."""
        te = TimeEvolution(
            hamiltonian=0.5 * qml.PauliZ(0) + 0.3 * qml.PauliZ(1),
            time=0.5,
            initial_state="Zeros",
            trotterization_strategy=QDrift(
                sampling_budget=2,
                seed=42,
                n_hamiltonians_per_iteration=3,
            ),
            backend=default_test_simulator,
        )
        te.run()
        assert te.total_circuit_count >= 3
        probs = te.results["probs"]
        assert probs.get("00", 0.0) >= 1.0 - _PROB_TOL

    def test_qdrift_keep_top_n_evolves_correctly(self, default_test_simulator):
        """QDrift with keep_top_n=1: H=X₀+X₁, |00⟩ at t=π/2 → P(11)=1."""
        te = TimeEvolution(
            hamiltonian=qml.PauliX(0) + qml.PauliX(1),
            time=math.pi / 2,
            initial_state="Zeros",
            trotterization_strategy=QDrift(
                keep_top_n=1,
                sampling_budget=1,
                seed=42,
                n_hamiltonians_per_iteration=3,
            ),
            backend=default_test_simulator,
        )
        te.run()
        assert te.total_circuit_count >= 3
        probs = te.results["probs"]
        assert probs.get("11", 0.0) >= 1.0 - _PROB_TOL

    def test_qdrift_heisenberg_xx_superposition_uniform(self, default_test_simulator):
        """QDrift Heisenberg XX (product operators): |++⟩ stays uniform, P(·)=0.25."""
        te = TimeEvolution(
            hamiltonian=qml.PauliX(0) @ qml.PauliX(1) + qml.PauliY(0) @ qml.PauliY(1),
            time=math.pi / 4,
            initial_state="Superposition",
            trotterization_strategy=QDrift(
                sampling_budget=2,
                seed=42,
                n_hamiltonians_per_iteration=5,
            ),
            backend=default_test_simulator,
        )
        te.run()
        assert te.total_circuit_count >= 5
        probs = te.results["probs"]
        total = sum(probs.values())
        assert abs(total - 1.0) <= _PROB_TOL
        for key in ("00", "01", "10", "11"):
            assert 0.25 - _PROB_TOL <= probs.get(key, 0.0) <= 0.25 + _PROB_TOL

    def test_qdrift_heisenberg_xxx_ones_stays_eigenstate(self, default_test_simulator):
        """QDrift Heisenberg XXX: |11⟩ triplet eigenstate, P(11)=1."""
        te = TimeEvolution(
            hamiltonian=(
                qml.PauliX(0) @ qml.PauliX(1)
                + qml.PauliY(0) @ qml.PauliY(1)
                + qml.PauliZ(0) @ qml.PauliZ(1)
            ),
            time=math.pi / 2,
            initial_state="Ones",
            trotterization_strategy=QDrift(
                sampling_budget=3,
                seed=42,
                n_hamiltonians_per_iteration=5,
            ),
            backend=default_test_simulator,
        )
        te.run()
        assert te.total_circuit_count >= 5
        probs = te.results["probs"]
        assert probs.get("11", 0.0) >= 1.0 - _PROB_TOL

    def test_qdrift_reduces_circuit_op_count(self):
        """QDrift with sampling_budget < n_terms produces shallower circuits than exact."""

        # 6-term, 3-qubit Hamiltonian
        hamiltonian = (
            qml.PauliX(0) @ qml.PauliX(1)
            + qml.PauliY(0) @ qml.PauliY(1)
            + qml.PauliZ(0) @ qml.PauliZ(1)
            + qml.PauliX(1) @ qml.PauliX(2)
            + qml.PauliY(1) @ qml.PauliY(2)
            + qml.PauliZ(1) @ qml.PauliZ(2)
        )

        # Exact: evolves with all 6 terms across 4 Trotter steps
        backend_exact = ParallelSimulator(
            shots=1000, track_depth=True, _deterministic_execution=True
        )
        te_exact = TimeEvolution(
            hamiltonian=hamiltonian,
            time=0.5,
            n_steps=4,
            trotterization_strategy=ExactTrotterization(),
            backend=backend_exact,
        )
        te_exact.run()
        exact_depth = backend_exact.average_depth()

        # QDrift: samples only 2 of the 6 terms
        backend_qdrift = ParallelSimulator(
            shots=1000, track_depth=True, _deterministic_execution=True
        )
        te_qdrift = TimeEvolution(
            hamiltonian=hamiltonian,
            time=0.5,
            n_steps=4,
            trotterization_strategy=QDrift(sampling_budget=2, seed=42),
            backend=backend_qdrift,
        )
        te_qdrift.run()
        qdrift_depth = backend_qdrift.average_depth()

        assert qdrift_depth < exact_depth, (
            f"QDrift circuit depth ({qdrift_depth}) should be less "
            f"than exact ({exact_depth}) when sampling_budget < n_terms"
        )
