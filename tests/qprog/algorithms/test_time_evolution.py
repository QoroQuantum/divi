# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import math
import warnings

import pennylane as qp
import pytest

from divi.backends import ExecutionResult, QiskitSimulator
from divi.circuits import DEFAULT_PRECISION
from divi.circuits.quepp import QuEPP
from divi.circuits.zne import ZNE, RichardsonExtrapolator
from divi.hamiltonians import ExactTrotterization, QDrift
from divi.pipeline.stages import MeasurementStage
from divi.qprog import OnesState, SuperpositionState, TimeEvolution, ZerosState
from tests.qprog._program_contracts import ObservableMeasuringContractsBase

# Tolerance for probability checks (5000 shots: ~0.02 std for p=0.5)
_PROB_TOL = 0.05

# Tolerance for QDrift expval comparisons (sampling noise + shot noise)
_QDRIFT_EXPVAL_TOL = 0.15


@pytest.fixture
def two_qubit_hamiltonian():
    return 0.5 * qp.PauliZ(0) + 0.3 * qp.PauliZ(1)


class TestTimeEvolutionInitialization:
    def test_initialization_valid(self, two_qubit_hamiltonian, default_test_simulator):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            time=1.0,
            backend=default_test_simulator,
        )
        assert te.time == 1.0
        assert te.n_qubits == 2
        assert isinstance(te.initial_state, ZerosState)
        assert te.observable is None

    def test_initialization_requires_backend(self, two_qubit_hamiltonian):
        with pytest.raises(ValueError, match="requires a backend"):
            TimeEvolution(hamiltonian=two_qubit_hamiltonian, backend=None)

    def test_initialization_constant_hamiltonian_fails(self, default_test_simulator):
        with pytest.raises(ValueError, match="only constant terms"):
            TimeEvolution(
                hamiltonian=qp.Identity(0) * 1.0,
                backend=default_test_simulator,
            )

    def test_initialization_invalid_initial_state(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        with pytest.raises(TypeError):
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
            backend=default_test_simulator,
        )
        te.run()
        assert te.total_circuit_count >= 1
        assert te.total_run_time >= 0
        probs = te.results
        assert isinstance(probs, dict)
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.1  # Approximate due to shots

    def test_run_initial_state_superposition(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            time=0.5,
            initial_state=SuperpositionState(),
            backend=default_test_simulator,
        )
        te.run()
        assert te.total_circuit_count >= 1
        assert te.results is not None

    def test_run_initial_state_ones(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            time=0.5,
            initial_state=OnesState(),
            backend=default_test_simulator,
        )
        te.run()
        assert te.total_circuit_count >= 1

    def test_single_term_hamiltonian_fallback(self, default_test_simulator):
        """ExactTrotterization with keep_top_n=1 yields single-term; use evolve not TrotterProduct."""
        h = 0.5 * qp.PauliX(0) + 0.3 * qp.PauliY(0)
        te = TimeEvolution(
            hamiltonian=h,
            trotterization_strategy=ExactTrotterization(keep_top_n=1),
            time=0.5,
            backend=default_test_simulator,
        )
        te.run()
        assert te.total_circuit_count >= 1
        assert te.results is not None


class TestTimeEvolutionObservable:
    def test_run_with_observable_shot_backend(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            time=0.5,
            observable=qp.PauliZ(0),
            backend=default_test_simulator,
        )
        te.run()
        assert te.total_circuit_count >= 1
        assert te.results is not None
        assert -1.1 <= te.results <= 1.1

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
            observable=qp.PauliZ(0) + qp.PauliZ(1),
            backend=dummy_expval_backend,
        )
        te.run()
        assert te.total_circuit_count >= 1
        assert te.results is not None
        assert te.results == pytest.approx(2.0)


class TestTimeEvolutionMultiObservable:
    """Multiple ``observable`` entries → results is a list[float] in input order."""

    def test_list_observables_normalised_to_tuple(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        """``observable=[...]`` is normalised to a tuple so downstream
        ``isinstance(..., tuple)`` checks fire."""
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            time=0.5,
            observable=[qp.PauliZ(0), qp.PauliZ(1)],
            backend=default_test_simulator,
        )
        assert isinstance(te.observable, tuple)
        assert len(te.observable) == 2

    def test_tuple_observable_returns_list_of_floats(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            time=0.5,
            observable=(qp.PauliZ(0), qp.PauliZ(1)),
            backend=default_test_simulator,
        )
        te.run()
        assert isinstance(te.results, list)
        assert len(te.results) == 2
        for v in te.results:
            assert isinstance(v, float)
            assert -1.1 <= v <= 1.1

    def test_single_observable_returns_float(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        """Single (non-tuple) observable keeps the scalar result contract."""
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            time=0.5,
            observable=qp.PauliZ(0),
            backend=default_test_simulator,
        )
        te.run()
        assert isinstance(te.results, float)

    def test_none_observable_returns_probs_dict(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        """No observable → probs dict (unchanged contract)."""
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            time=0.5,
            backend=default_test_simulator,
        )
        te.run()
        assert isinstance(te.results, dict)

    def test_multi_observable_matches_per_observable_runs(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        """For commuting observables on a noiseless backend, the multi
        run produces (within shot noise) the same per-observable values
        as one TimeEvolution per observable."""
        common = dict(
            hamiltonian=two_qubit_hamiltonian,
            time=0.5,
        )
        te_multi = TimeEvolution(
            **common,
            observable=(qp.PauliZ(0), qp.PauliZ(1)),
            backend=default_test_simulator,
        )
        te_multi.run()

        te_solo_0 = TimeEvolution(
            **common, observable=qp.PauliZ(0), backend=default_test_simulator
        )
        te_solo_0.run()
        te_solo_1 = TimeEvolution(
            **common, observable=qp.PauliZ(1), backend=default_test_simulator
        )
        te_solo_1.run()

        # Loose tolerance: shot noise + any small numerical drift from QWC
        # grouping vs single-observable measurement.
        assert te_multi.results[0] == pytest.approx(te_solo_0.results, abs=0.1)
        assert te_multi.results[1] == pytest.approx(te_solo_1.results, abs=0.1)


class TestTimeEvolutionExpvalAccessor:
    """``.expval()`` mirrors ``.results`` for both single- and multi-observable
    runs and rejects only the probability-mode case."""

    def test_expval_returns_scalar_for_single_observable(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            time=0.5,
            observable=qp.PauliZ(0),
            backend=default_test_simulator,
        )
        te.run()
        out = te.expval()
        assert isinstance(out, float)
        assert out == te.results

    def test_expval_returns_list_for_multi_observable(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            time=0.5,
            observable=[qp.PauliZ(0), qp.PauliZ(1)],
            backend=default_test_simulator,
        )
        te.run()
        out = te.expval()
        assert isinstance(out, list)
        assert len(out) == 2
        assert all(isinstance(v, float) for v in out)
        assert out == te.results

    def test_expval_rejects_probability_mode(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            time=0.5,
            backend=default_test_simulator,
        )
        te.run()
        with pytest.raises(RuntimeError, match="probability mode"):
            te.expval()

    def test_expval_returns_list_via_expval_native_backend(
        self, two_qubit_hamiltonian, dummy_expval_backend, mocker
    ):
        """Multi-obs expval() works through an expval-native backend, not just
        shot-based simulators. Pins the contract for QoroService and Maestro
        analytical paths."""

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
            observable=[qp.PauliZ(0), qp.PauliZ(1)],
            backend=dummy_expval_backend,
        )
        te.run()
        out = te.expval()
        assert isinstance(out, list)
        assert len(out) == 2
        assert all(isinstance(v, float) for v in out)

    def test_expval_raises_before_run(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        """``expval()`` before ``run()`` raises a clear runtime error."""
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            observable=qp.PauliZ(0),
            backend=default_test_simulator,
        )
        with pytest.raises(RuntimeError):
            te.expval()


class TestTimeEvolutionSingleItemListPreserved:
    """``observable=[O]`` (an explicit single-item list) opts the user
    into the multi-observable API: results stay as a length-1 ``list``
    rather than being squeezed to a scalar."""

    def test_single_item_list_returns_length_one_list(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            time=0.5,
            observable=[qp.PauliZ(0)],
            backend=default_test_simulator,
        )
        te.run()
        assert isinstance(te.results, list)
        assert len(te.results) == 1
        assert isinstance(te.results[0], float)

    def test_bare_observable_returns_scalar(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            time=0.5,
            observable=qp.PauliZ(0),
            backend=default_test_simulator,
        )
        te.run()
        assert isinstance(te.results, float)

    def test_single_item_tuple_returns_length_one_list(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            time=0.5,
            observable=(qp.PauliZ(0),),
            backend=default_test_simulator,
        )
        te.run()
        assert isinstance(te.results, list)
        assert len(te.results) == 1


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
        te.run()
        assert te.total_circuit_count >= 3
        assert te.results is not None
        probs = te.results
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.2

    def test_expval_shot_backend_qdrift_aggregation(self, default_test_simulator):
        """QDrift with observable on shot backend averages expvals across samples."""
        te = TimeEvolution(
            hamiltonian=qp.PauliX(0),
            observable=qp.PauliZ(0),
            trotterization_strategy=QDrift(
                sampling_budget=1,
                seed=42,
                n_hamiltonians_per_iteration=2,
            ),
            backend=default_test_simulator,
        )
        te.run()
        assert te.total_circuit_count == 2
        assert te.results is not None

    def test_multi_sample_qdrift_expval_vs_sampling(self, default_test_simulator):
        """Multi-sample QDrift: expval backend and sampling backend agree."""
        hamiltonian = qp.sum(
            -(qp.PauliZ(0) @ qp.PauliZ(1)),
            -qp.PauliX(0),
            -qp.PauliX(1),
        )
        qdrift_kwargs = dict(
            sampling_budget=50,
            seed=42,
            n_hamiltonians_per_iteration=10,
            sampling_strategy="weighted",
        )
        common = dict(
            hamiltonian=hamiltonian,
            time=0.8,
            n_steps=1,
            observable=hamiltonian,
        )

        te_expval = TimeEvolution(
            **common,
            trotterization_strategy=QDrift(**qdrift_kwargs),
            backend=default_test_simulator,
        )
        te_expval.run()

        te_sampling = TimeEvolution(
            **common,
            trotterization_strategy=QDrift(**qdrift_kwargs),
            backend=QiskitSimulator(
                shots=5000, force_sampling=True, _deterministic_execution=True
            ),
        )
        te_sampling.run()

        assert abs(te_expval.results - te_sampling.results) < _QDRIFT_EXPVAL_TOL


@pytest.mark.e2e
class TestTimeEvolutionE2E:
    """E2E tests with well-known Hamiltonians and quantitative checks against analytic results."""

    def test_h_x0_plus_x1_zeros_evolves_to_11(self, default_test_simulator):
        """H=X₀+X₁ (2-qubit, commuting): |00⟩ at t=π/2 → |11⟩, P(11)=1."""
        te = TimeEvolution(
            hamiltonian=qp.PauliX(0) + qp.PauliX(1),
            time=math.pi / 2,
            backend=default_test_simulator,
        )
        te.run()
        probs = te.results
        assert probs.get("11", 0.0) >= 1.0 - _PROB_TOL
        for key in ("00", "01", "10"):
            assert probs.get(key, 0.0) <= _PROB_TOL

    def test_h_x_plus_z_full_rotation(self, default_test_simulator):
        """H=X+Z (1-qubit, non-commuting): |0⟩ at t=π/√2 → |0⟩ (full Bloch rotation)."""
        te = TimeEvolution(
            hamiltonian=qp.PauliX(0) + qp.PauliZ(0),
            time=math.pi / math.sqrt(2),
            n_steps=10,
            backend=default_test_simulator,
        )
        te.run()
        probs = te.results
        assert probs.get("0", 0.0) >= 1.0 - _PROB_TOL
        assert probs.get("1", 0.0) <= _PROB_TOL

    def test_heisenberg_xx_superposition_uniform(self, default_test_simulator):
        """Heisenberg XX (X₀X₁+Y₀Y₁, non-commuting): |++⟩ stays uniform, P(·)=0.25."""
        te = TimeEvolution(
            hamiltonian=qp.PauliX(0) @ qp.PauliX(1) + qp.PauliY(0) @ qp.PauliY(1),
            time=math.pi / 4,
            initial_state=SuperpositionState(),
            backend=default_test_simulator,
        )
        te.run()
        probs = te.results
        total = sum(probs.values())
        assert abs(total - 1.0) <= _PROB_TOL
        for key in ("00", "01", "10", "11"):
            assert 0.25 - _PROB_TOL <= probs.get(key, 0.0) <= 0.25 + _PROB_TOL

    def test_heisenberg_xxx_superposition_uniform(self, default_test_simulator):
        """Heisenberg XXX (X₀X₁+Y₀Y₁+Z₀Z₁): |++⟩ stays uniform, P(·)=0.25."""
        te = TimeEvolution(
            hamiltonian=(
                qp.PauliX(0) @ qp.PauliX(1)
                + qp.PauliY(0) @ qp.PauliY(1)
                + qp.PauliZ(0) @ qp.PauliZ(1)
            ),
            time=math.pi / 4,
            initial_state=SuperpositionState(),
            backend=default_test_simulator,
        )
        te.run()
        probs = te.results
        total = sum(probs.values())
        assert abs(total - 1.0) <= _PROB_TOL
        for key in ("00", "01", "10", "11"):
            assert 0.25 - _PROB_TOL <= probs.get(key, 0.0) <= 0.25 + _PROB_TOL

    def test_heisenberg_xxx_ones_stays_eigenstate(self, default_test_simulator):
        """Heisenberg XXX: |11⟩ is triplet eigenstate, P(11)=1."""
        te = TimeEvolution(
            hamiltonian=(
                qp.PauliX(0) @ qp.PauliX(1)
                + qp.PauliY(0) @ qp.PauliY(1)
                + qp.PauliZ(0) @ qp.PauliZ(1)
            ),
            time=math.pi / 2,
            initial_state=OnesState(),
            backend=default_test_simulator,
        )
        te.run()
        probs = te.results
        assert probs.get("11", 0.0) >= 1.0 - _PROB_TOL
        for key in ("00", "01", "10"):
            assert probs.get(key, 0.0) <= _PROB_TOL

    def test_qdrift_commuting_stays_eigenstate(self, default_test_simulator):
        """QDrift H=Z₀+Z₁: |00⟩ is eigenstate, multi-sample average P(00)=1."""
        te = TimeEvolution(
            hamiltonian=0.5 * qp.PauliZ(0) + 0.3 * qp.PauliZ(1),
            time=0.5,
            trotterization_strategy=QDrift(
                sampling_budget=2,
                seed=42,
                n_hamiltonians_per_iteration=3,
            ),
            backend=default_test_simulator,
        )
        te.run()
        assert te.total_circuit_count >= 3
        probs = te.results
        assert probs.get("00", 0.0) >= 1.0 - _PROB_TOL

    def test_qdrift_keep_top_n_evolves_correctly(self, default_test_simulator):
        """QDrift with keep_top_n=1: H=X₀+X₁, |00⟩ at t=π/2 → P(11)=1."""
        te = TimeEvolution(
            hamiltonian=qp.PauliX(0) + qp.PauliX(1),
            time=math.pi / 2,
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
        probs = te.results
        assert probs.get("11", 0.0) >= 1.0 - _PROB_TOL

    def test_qdrift_heisenberg_xx_superposition_uniform(self, default_test_simulator):
        """QDrift Heisenberg XX (product operators): |++⟩ stays uniform, P(·)=0.25."""
        te = TimeEvolution(
            hamiltonian=qp.PauliX(0) @ qp.PauliX(1) + qp.PauliY(0) @ qp.PauliY(1),
            time=math.pi / 4,
            initial_state=SuperpositionState(),
            trotterization_strategy=QDrift(
                sampling_budget=2,
                seed=42,
                n_hamiltonians_per_iteration=5,
            ),
            backend=default_test_simulator,
        )
        te.run()
        assert te.total_circuit_count >= 5
        probs = te.results
        total = sum(probs.values())
        assert abs(total - 1.0) <= _PROB_TOL
        for key in ("00", "01", "10", "11"):
            assert 0.25 - _PROB_TOL <= probs.get(key, 0.0) <= 0.25 + _PROB_TOL

    def test_qdrift_heisenberg_xxx_ones_stays_eigenstate(self, default_test_simulator):
        """QDrift Heisenberg XXX: |11⟩ triplet eigenstate, P(11)=1."""
        te = TimeEvolution(
            hamiltonian=(
                qp.PauliX(0) @ qp.PauliX(1)
                + qp.PauliY(0) @ qp.PauliY(1)
                + qp.PauliZ(0) @ qp.PauliZ(1)
            ),
            time=math.pi / 2,
            initial_state=OnesState(),
            trotterization_strategy=QDrift(
                sampling_budget=3,
                seed=42,
                n_hamiltonians_per_iteration=5,
            ),
            backend=default_test_simulator,
        )
        te.run()
        assert te.total_circuit_count >= 5
        probs = te.results
        assert probs.get("11", 0.0) >= 1.0 - _PROB_TOL

    def test_qdrift_tfim_non_commuting_expval(self):
        """QDrift TFIM 4q (non-commuting ZZ+X): Campbell's protocol matches exact."""
        backend = QiskitSimulator(shots=10000, _deterministic_execution=True)
        hamiltonian = qp.sum(
            -(qp.PauliZ(0) @ qp.PauliZ(1)),
            -(qp.PauliZ(1) @ qp.PauliZ(2)),
            -(qp.PauliZ(2) @ qp.PauliZ(3)),
            -qp.PauliX(0),
            -qp.PauliX(1),
            -qp.PauliX(2),
            -qp.PauliX(3),
        )

        te_exact = TimeEvolution(
            hamiltonian=hamiltonian,
            time=0.8,
            n_steps=10,
            observable=hamiltonian,
            trotterization_strategy=ExactTrotterization(),
            backend=backend,
        )
        te_exact.run()

        te_qdrift = TimeEvolution(
            hamiltonian=hamiltonian,
            time=0.8,
            n_steps=1,
            observable=hamiltonian,
            trotterization_strategy=QDrift(
                sampling_budget=500,
                seed=42,
                n_hamiltonians_per_iteration=20,
                sampling_strategy="weighted",
            ),
            backend=backend,
        )
        te_qdrift.run()

        assert abs(te_exact.results - te_qdrift.results) < _QDRIFT_EXPVAL_TOL

    def test_qdrift_x_plus_z_non_commuting_expval(self):
        """QDrift H=X+Z (non-commuting): Campbell's protocol matches exact."""
        backend = QiskitSimulator(shots=10000, _deterministic_execution=True)
        hamiltonian = qp.PauliX(0) + qp.PauliZ(0)

        te_exact = TimeEvolution(
            hamiltonian=hamiltonian,
            time=0.8,
            n_steps=10,
            observable=hamiltonian,
            trotterization_strategy=ExactTrotterization(),
            backend=backend,
        )
        te_exact.run()

        te_qdrift = TimeEvolution(
            hamiltonian=hamiltonian,
            time=0.8,
            n_steps=1,
            observable=hamiltonian,
            trotterization_strategy=QDrift(
                sampling_budget=50,
                seed=42,
                n_hamiltonians_per_iteration=20,
                sampling_strategy="weighted",
            ),
            backend=backend,
        )
        te_qdrift.run()

        assert abs(te_exact.results - te_qdrift.results) < _QDRIFT_EXPVAL_TOL

    def test_qdrift_reduces_circuit_op_count(self):
        """QDrift with sampling_budget < n_terms produces shallower circuits than exact."""

        # 6-term, 3-qubit Hamiltonian
        hamiltonian = (
            qp.PauliX(0) @ qp.PauliX(1)
            + qp.PauliY(0) @ qp.PauliY(1)
            + qp.PauliZ(0) @ qp.PauliZ(1)
            + qp.PauliX(1) @ qp.PauliX(2)
            + qp.PauliY(1) @ qp.PauliY(2)
            + qp.PauliZ(1) @ qp.PauliZ(2)
        )

        # Exact: evolves with all 6 terms across 4 Trotter steps
        backend_exact = QiskitSimulator(
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
        backend_qdrift = QiskitSimulator(
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


@pytest.mark.usefixtures("suppress_quepp_warnings")
class TestTimeEvolutionQEM:
    """Tests for QEM integration in TimeEvolution."""

    def test_no_qem_protocol_unchanged(self, default_test_simulator):
        """Without qem_protocol, TimeEvolution pipeline has no QEM stages."""
        te = TimeEvolution(
            hamiltonian=qp.PauliX(0) + qp.PauliZ(0),
            observable=qp.PauliZ(0),
            backend=default_test_simulator,
        )
        stage_names = [type(s).__name__ for s in te._pipeline.stages]
        assert "QEMStage" not in stage_names
        assert "PauliTwirlStage" not in stage_names

    def test_quepp_adds_qem_stage(self, default_test_simulator):
        """QuEPP with n_twirls=0 adds QEMStage only."""
        te = TimeEvolution(
            hamiltonian=qp.PauliX(0) + qp.PauliZ(0),
            observable=qp.PauliZ(0),
            backend=default_test_simulator,
            qem_protocol=QuEPP(truncation_order=1, n_twirls=0),
        )
        stage_names = [type(s).__name__ for s in te._pipeline.stages]
        assert "QEMStage" in stage_names
        assert "PauliTwirlStage" not in stage_names

    def test_quepp_adds_twirl_stage(self, default_test_simulator):
        """QuEPP with n_twirls>0 adds both QEMStage and PauliTwirlStage."""
        te = TimeEvolution(
            hamiltonian=qp.PauliX(0) + qp.PauliZ(0),
            observable=qp.PauliZ(0),
            backend=default_test_simulator,
            qem_protocol=QuEPP(truncation_order=1, n_twirls=5),
        )
        stage_names = [type(s).__name__ for s in te._pipeline.stages]
        assert "QEMStage" in stage_names
        assert "PauliTwirlStage" in stage_names

    def test_zne_adds_qem_stage(self, default_test_simulator):
        """ZNE protocol adds QEMStage without PauliTwirlStage."""
        scale_factors = [1.0, 3.0, 5.0]
        te = TimeEvolution(
            hamiltonian=qp.PauliX(0) + qp.PauliZ(0),
            observable=qp.PauliZ(0),
            backend=default_test_simulator,
            qem_protocol=ZNE(
                scale_factors=scale_factors,
                extrapolator=RichardsonExtrapolator(),
            ),
        )
        stage_names = [type(s).__name__ for s in te._pipeline.stages]
        assert "QEMStage" in stage_names
        assert "PauliTwirlStage" not in stage_names

    def test_zne_excluded_in_sampling_mode(self, default_test_simulator):
        """With no observable the pipeline samples probabilities; ZNE (expval-only)
        must not ride it."""
        te = TimeEvolution(
            hamiltonian=qp.PauliX(0) + qp.PauliZ(0),
            observable=None,
            backend=default_test_simulator,
            qem_protocol=ZNE(scale_factors=[1.0, 3.0]),
        )
        stage_names = [type(s).__name__ for s in te._pipeline.stages]
        assert "QEMStage" not in stage_names

    def test_non_variational_pipeline_has_no_param_binding(
        self, default_test_simulator
    ):
        """A plain (non-templated) TimeEvolution binds no parameters: the base
        assembler default is off for non-variational programs."""
        te = TimeEvolution(
            hamiltonian=qp.PauliX(0) + qp.PauliZ(0),
            observable=qp.PauliZ(0),
            backend=default_test_simulator,
        )
        stage_names = [type(s).__name__ for s in te._pipeline.stages]
        assert "ParameterBindingStage" not in stage_names

    def test_quepp_run_produces_mitigated_result(self):
        """QuEPP on TimeEvolution produces a scalar result."""
        backend = QiskitSimulator(shots=5000, _deterministic_execution=True)
        hamiltonian = qp.PauliX(0) + qp.PauliZ(0)

        te = TimeEvolution(
            hamiltonian=hamiltonian,
            time=0.5,
            observable=qp.PauliZ(0),
            backend=backend,
            qem_protocol=QuEPP(truncation_order=1, n_twirls=0),
        )
        te.run()
        assert te.total_circuit_count >= 1
        assert te.total_run_time >= 0
        assert isinstance(te.results, float)
        assert -1.1 <= te.results <= 1.1

    def test_dry_run_with_qem(self, default_test_simulator):
        """TimeEvolution.dry_run works and reports QEM stages."""
        te = TimeEvolution(
            hamiltonian=qp.PauliX(0) + qp.PauliZ(0),
            observable=qp.PauliZ(0),
            backend=default_test_simulator,
            qem_protocol=QuEPP(truncation_order=1, n_twirls=5),
        )
        reports = te.dry_run()
        assert "evolution" in reports
        report = reports["evolution"]
        stage_names = [s.name for s in report.stages]
        assert any("QEM" in name or "qem" in name.lower() for name in stage_names)

    def test_dry_run_without_qem(self, default_test_simulator):
        """TimeEvolution.dry_run works even without QEM."""
        te = TimeEvolution(
            hamiltonian=qp.PauliX(0) + qp.PauliZ(0),
            observable=qp.PauliZ(0),
            backend=default_test_simulator,
        )
        reports = te.dry_run()
        assert "evolution" in reports


@pytest.mark.filterwarnings(
    "ignore:Backend supports analytic expectation values:UserWarning"
)
class TestTimeEvolutionMeasurementConfig:
    """``grouping_strategy`` and ``shot_distribution`` kwargs flow into
    the :class:`~divi.pipeline.stages.MeasurementStage` of the program's
    evolution pipeline."""

    @staticmethod
    def _measurement_stage(te: TimeEvolution):
        return next(
            stage
            for stage in te._pipeline.stages
            if isinstance(stage, MeasurementStage)
        )

    def test_grouping_strategy_threaded_to_measurement_stage(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            observable=qp.PauliZ(0),
            grouping_strategy="wires",
            backend=default_test_simulator,
        )
        assert self._measurement_stage(te)._grouping_strategy == "wires"

    def test_shot_distribution_threaded_to_measurement_stage(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            observable=qp.PauliZ(0),
            shot_distribution="weighted",
            backend=default_test_simulator,
        )
        assert self._measurement_stage(te)._shot_distribution == "weighted"

    def test_explicit_backend_expval_rejected(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        """``"_backend_expval"`` is not a valid user-facing strategy."""
        with pytest.raises(ValueError, match="Invalid grouping_strategy"):
            TimeEvolution(
                hamiltonian=two_qubit_hamiltonian,
                observable=qp.PauliZ(0),
                grouping_strategy="_backend_expval",
                backend=default_test_simulator,
            )

    def test_sampling_mode_accepts_grouping_kwargs(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        """Sampling mode (``observable=None``) stores the kwargs verbatim
        and runs; MeasurementStage's probs branch ignores them."""
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            observable=None,
            grouping_strategy="wires",
            backend=default_test_simulator,
        )
        assert te._grouping_strategy == "wires"
        te.run()
        assert isinstance(te.results, dict)

    def test_multi_observable_with_expval_backend(
        self, two_qubit_hamiltonian, dummy_expval_backend, mocker
    ):
        """Multi-observable circuits stay on ``"qwc"`` and run end-to-end
        on an analytic-expval backend (no auto-flip to ``"_backend_expval"``,
        which only supports single-observable batches)."""

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
            observable=(qp.PauliZ(0), qp.PauliZ(1)),
            backend=dummy_expval_backend,
        )
        te.run()
        assert isinstance(te.results, list)
        assert len(te.results) == 2

    def test_qwc_with_single_obs_auto_flips_at_runtime(
        self, two_qubit_hamiltonian, dummy_expval_backend, mocker
    ):
        """Regression guard for the runtime auto-flip: explicit
        ``grouping_strategy="qwc"`` + single observable + expval-capable
        backend must execute through the backend's analytic path
        (``ham_ops`` kwarg present on submit), not via shot-based
        measurement QASMs."""

        submitted_kwargs = []

        def _capture_submit(circuits, **kwargs):
            submitted_kwargs.append(kwargs)
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
            side_effect=_capture_submit,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            te = TimeEvolution(
                hamiltonian=two_qubit_hamiltonian,
                observable=qp.PauliZ(0),
                grouping_strategy="qwc",
                backend=dummy_expval_backend,
            )
            te.run()

        assert te._grouping_strategy == "qwc"
        assert any("ham_ops" in kw for kw in submitted_kwargs)


class TestTimeEvolutionPrecision:
    """``precision`` is read from :class:`~divi.qprog.QuantumProgram` and
    propagated into the produced ``MetaCircuit``."""

    def test_precision_defaults_to_module_default(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian, backend=default_test_simulator
        )
        assert te._precision == DEFAULT_PRECISION
        assert te.precision == DEFAULT_PRECISION

    def test_explicit_precision_stored(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            backend=default_test_simulator,
            precision=4,
        )
        assert te._precision == 4
        assert te.precision == 4

    def test_precision_propagates_to_meta_circuit(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        te = TimeEvolution(
            hamiltonian=two_qubit_hamiltonian,
            backend=default_test_simulator,
            precision=3,
        )
        meta = te._meta_circuit_factory(te._hamiltonian, ham_id=0)
        assert meta.precision == 3


class TestObservableMeasuringContracts(ObservableMeasuringContractsBase):
    @pytest.fixture
    def make_program(self, two_qubit_hamiltonian, dummy_simulator):
        def _make(**kwargs):
            return TimeEvolution(
                hamiltonian=two_qubit_hamiltonian,
                backend=dummy_simulator,
                **kwargs,
            )

        return _make
