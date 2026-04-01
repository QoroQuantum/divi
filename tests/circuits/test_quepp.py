# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the QuEPP error mitigation protocol."""

import warnings

import cirq
import numpy as np
import pennylane as qml
import pytest
from qiskit_aer.noise import NoiseModel

from divi.backends import QiskitSimulator
from divi.circuits import MetaCircuit
from divi.circuits.qem import QEMContext, QEMProtocol, _NoMitigation
from divi.circuits.quepp import (
    QuEPP,
    _build_clifford_tableaus,
    _enumerate_paths_dfs,
    _extract_rotation_gates,
    _is_pauli_rotation,
    _obs_to_stim_terms,
    _sample_paths_montecarlo,
    _simulate_clifford_ensemble,
)
from divi.pipeline import CircuitPipeline, PipelineEnv
from divi.pipeline.stages import CircuitSpecStage, MeasurementStage, QEMStage
from tests.pipeline.helpers import DummySpecStage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_circuit():
    """Single-qubit Rx(0.5) circuit."""
    q = cirq.LineQubit(0)
    return cirq.Circuit(cirq.rx(0.5)(q))


@pytest.fixture
def bell_circuit():
    """Two-qubit Bell circuit (fully Clifford)."""
    q = cirq.LineQubit.range(2)
    return cirq.Circuit(cirq.H(q[0]), cirq.CNOT(q[0], q[1]))


@pytest.fixture
def mixed_circuit():
    """Two-qubit circuit with Clifford and non-Clifford gates."""
    q = cirq.LineQubit.range(2)
    return cirq.Circuit(
        cirq.H(q[0]),
        cirq.rx(0.3)(q[0]),
        cirq.CNOT(q[0], q[1]),
        cirq.ry(0.7)(q[1]),
    )


def _rx_circuit(angle: float) -> cirq.Circuit:
    """Helper for tests that need a specific rotation angle."""
    return cirq.Circuit(cirq.rx(angle)(cirq.LineQubit(0)))


def _exact_expval(circuit: cirq.Circuit, observable, n_qubits: int) -> float:
    """Compute exact expectation value via statevector simulation."""
    sv = cirq.Simulator().simulate(circuit).final_state_vector
    mat = np.array(
        qml.matrix(observable, wire_order=list(range(n_qubits))), dtype=complex
    )
    return float(np.real(sv.conj() @ mat @ sv))


# ---------------------------------------------------------------------------
# Gate identification tests
# ---------------------------------------------------------------------------


class TestIsPauliRotation:
    def test_rx_detected(self):
        result = _is_pauli_rotation(cirq.rx(0.5))
        assert result[0] == "x"
        assert result[1] == pytest.approx(0.5)

    def test_ry_detected(self):
        result = _is_pauli_rotation(cirq.ry(1.2))
        assert result[0] == "y"
        assert result[1] == pytest.approx(1.2)

    def test_rz_detected(self):
        result = _is_pauli_rotation(cirq.rz(0.8))
        assert result[0] == "z"
        assert result[1] == pytest.approx(0.8)

    def test_clifford_gate_returns_none(self):
        assert _is_pauli_rotation(cirq.H) is None
        assert _is_pauli_rotation(cirq.S) is None
        assert _is_pauli_rotation(cirq.CNOT) is None

    def test_clifford_rotation_still_detected(self):
        """Rx(π/2) is detected to ensure consistent path count across bindings."""
        assert _is_pauli_rotation(cirq.rx(np.pi / 2)) is not None


class TestExtractRotationGates:
    def test_fully_clifford_circuit(self, bell_circuit):
        qubits = sorted(bell_circuit.all_qubits())
        gates = _extract_rotation_gates(bell_circuit, qubits)
        assert len(gates) == 0

    def test_single_rotation(self, simple_circuit):
        qubits = sorted(simple_circuit.all_qubits())
        gates = _extract_rotation_gates(simple_circuit, qubits)
        assert len(gates) == 1
        assert gates[0].axis == "x"

    def test_mixed_circuit(self, mixed_circuit):
        qubits = sorted(mixed_circuit.all_qubits())
        gates = _extract_rotation_gates(mixed_circuit, qubits)
        assert len(gates) == 2


# ---------------------------------------------------------------------------
# CPT correctness tests
# ---------------------------------------------------------------------------


class TestCPTExpansion:
    """Verify that the Heisenberg CPT expansion recovers exact expectation values."""

    def test_single_rx(self):
        """Rx(θ) with Z observable → cos(θ)."""
        angle = 0.8
        c = _rx_circuit(angle)
        protocol = QuEPP(sampling="exhaustive", truncation_order=5)
        _, ctx = protocol.expand(c, observable=qml.Z(0))
        cpt = float(ctx["weights"] @ ctx["classical_values"])
        assert cpt == pytest.approx(np.cos(angle), abs=1e-6)

    def test_h_rx_h_ry(self):
        """Multi-gate single-qubit circuit."""
        q = cirq.LineQubit(0)
        c = cirq.Circuit(cirq.H(q), cirq.rx(0.3)(q), cirq.H(q), cirq.ry(0.5)(q))
        protocol = QuEPP(sampling="exhaustive", truncation_order=5)
        _, ctx = protocol.expand(c, observable=qml.Z(0))
        cpt = float(ctx["weights"] @ ctx["classical_values"])
        assert cpt == pytest.approx(_exact_expval(c, qml.Z(0), 1), abs=1e-4)

    def test_two_qubit_circuit(self, mixed_circuit):
        """Two-qubit circuit with ZZ observable."""
        obs = qml.Z(0) @ qml.Z(1)
        protocol = QuEPP(sampling="exhaustive", truncation_order=5)
        _, ctx = protocol.expand(mixed_circuit, observable=obs)
        cpt = float(ctx["weights"] @ ctx["classical_values"])
        assert cpt == pytest.approx(_exact_expval(mixed_circuit, obs, 2), abs=1e-4)

    def test_commuting_gate_no_branch(self):
        """When observable commutes with rotation generator, no branching occurs.

        Rx with X observable — X commutes with X generator, so the gate
        is transparent.  The back-propagated observable stays X, which is
        not diagonal, so the path has zero contribution.
        """
        q = cirq.LineQubit(0)
        c = cirq.Circuit(cirq.rx(0.5)(q))
        protocol = QuEPP(sampling="exhaustive", truncation_order=5)
        circuits, ctx = protocol.expand(c, observable=qml.X(0))
        # No surviving paths (X is not diagonal on |0⟩)
        assert ctx["n_paths"] == 0
        # CPT sum = 0, matching exact ⟨0|X|0⟩ = 0
        assert float(ctx["weights"] @ ctx["classical_values"]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Path enumeration tests
# ---------------------------------------------------------------------------


class TestEnumeratePathsDFS:
    def test_clifford_circuit_single_path(self, bell_circuit):
        qubits = sorted(bell_circuit.all_qubits())
        rots = _extract_rotation_gates(bell_circuit, qubits)
        tabs = _build_clifford_tableaus(bell_circuit, rots, qubits)
        terms = _obs_to_stim_terms(qml.Z(0) @ qml.Z(1), 2)
        paths = _enumerate_paths_dfs(rots, tabs, terms, max_order=5)
        assert len(paths) == 1
        assert paths[0].order == 0

    def test_coefficient_threshold_prunes(self, mixed_circuit):
        qubits = sorted(mixed_circuit.all_qubits())
        rots = _extract_rotation_gates(mixed_circuit, qubits)
        tabs = _build_clifford_tableaus(mixed_circuit, rots, qubits)
        terms = _obs_to_stim_terms(qml.Z(0) @ qml.Z(1), 2)
        all_paths = _enumerate_paths_dfs(rots, tabs, terms, max_order=5)
        filtered = _enumerate_paths_dfs(
            rots, tabs, terms, max_order=5, coefficient_threshold=0.5
        )
        assert len(filtered) <= len(all_paths)


class TestSamplePathsMonteCarlo:
    def test_deterministic_with_seed(self, mixed_circuit):
        qubits = sorted(mixed_circuit.all_qubits())
        rots = _extract_rotation_gates(mixed_circuit, qubits)
        tabs = _build_clifford_tableaus(mixed_circuit, rots, qubits)
        terms = _obs_to_stim_terms(qml.Z(0) @ qml.Z(1), 2)
        p1 = _sample_paths_montecarlo(rots, tabs, terms, 50, np.random.default_rng(0))
        p2 = _sample_paths_montecarlo(rots, tabs, terms, 50, np.random.default_rng(0))
        assert len(p1) == len(p2)


# ---------------------------------------------------------------------------
# Clifford simulation tests
# ---------------------------------------------------------------------------


class TestSimulateCliffordEnsemble:
    def test_bell_state_zz(self, bell_circuit):
        vals = _simulate_clifford_ensemble(
            [bell_circuit], qml.Z(0) @ qml.Z(1), n_qubits=2
        )
        assert vals[0] == pytest.approx(1.0)

    def test_bell_state_xx(self, bell_circuit):
        vals = _simulate_clifford_ensemble(
            [bell_circuit], qml.X(0) @ qml.X(1), n_qubits=2
        )
        assert vals[0] == pytest.approx(1.0)

    def test_batch_returns_correct_count(self, bell_circuit):
        vals = _simulate_clifford_ensemble(
            [bell_circuit, bell_circuit, bell_circuit], qml.Z(0) @ qml.Z(1), n_qubits=2
        )
        assert len(vals) == 3


# ---------------------------------------------------------------------------
# QuEPP protocol tests
# ---------------------------------------------------------------------------


class TestQuEPPProtocol:
    def test_is_qem_protocol(self):
        assert isinstance(QuEPP(truncation_order=1), QEMProtocol)
        assert QuEPP().name == "quepp"

    def test_invalid_truncation_order(self):
        with pytest.raises(ValueError, match="truncation_order"):
            QuEPP(truncation_order=-1)

    def test_invalid_sampling(self):
        with pytest.raises(ValueError, match="sampling"):
            QuEPP(sampling="bogus")

    def test_montecarlo_requires_positive_n_samples(self):
        with pytest.raises(ValueError, match="n_samples"):
            QuEPP(sampling="montecarlo", n_samples=0)

    def test_expand_requires_observable(self, simple_circuit):
        with pytest.raises(ValueError, match="requires an observable"):
            QuEPP().expand(simple_circuit, observable=None)

    def test_expand_returns_circuits_and_context(self, simple_circuit):
        circuits, ctx = QuEPP(truncation_order=1).expand(
            simple_circuit, observable=qml.Z(0)
        )
        assert len(circuits) >= 2
        assert isinstance(ctx, QEMContext)
        assert ctx["target_idx"] == 0
        assert ctx["ensemble_start"] == 1

    def test_expand_clifford_circuit(self, bell_circuit):
        circuits, ctx = QuEPP(truncation_order=5).expand(
            bell_circuit, observable=qml.Z(0) @ qml.Z(1)
        )
        assert len(circuits) == 2  # target + 1 path
        assert ctx["n_rotations"] == 0

    def test_reduce_clifford_circuit_exact(self, bell_circuit):
        _, ctx = QuEPP(truncation_order=5).expand(
            bell_circuit, observable=qml.Z(0) @ qml.Z(1)
        )
        qr = [1.0]  # target
        qr.extend(ctx["classical_values"])
        assert QuEPP().reduce(qr, ctx) == pytest.approx(1.0)

    def test_full_round_trip_single_qubit(self):
        """expand → reduce with exact quantum results recovers ideal value."""
        angle = 0.8
        c = _rx_circuit(angle)
        exact = np.cos(angle)
        protocol = QuEPP(sampling="exhaustive", truncation_order=10)
        _, ctx = protocol.expand(c, observable=qml.Z(0))
        qr = [exact]
        qr.extend(ctx["classical_values"])
        assert protocol.reduce(qr, ctx) == pytest.approx(exact, abs=1e-6)

    def test_noise_correction(self):
        """QuEPP corrects a globally-scaled noise bias."""
        angle = 0.8
        c = _rx_circuit(angle)
        exact = np.cos(angle)
        protocol = QuEPP(sampling="exhaustive", truncation_order=10)
        _, ctx = protocol.expand(c, observable=qml.Z(0))
        noise_factor = 0.9
        qr = [exact * noise_factor]
        qr.extend(ctx["classical_values"] * noise_factor)
        assert protocol.reduce(qr, ctx) == pytest.approx(exact, abs=1e-4)

    def test_montecarlo_expand(self, mixed_circuit):
        protocol = QuEPP(sampling="montecarlo", n_samples=50, seed=42)
        circuits, ctx = protocol.expand(mixed_circuit, observable=qml.Z(0) @ qml.Z(1))
        assert len(circuits) >= 2

    def test_pipeline_integration(self, dummy_pipeline_env):
        """QuEPP integrates correctly with QEMStage in a pipeline."""
        qscript = qml.tape.QuantumScript(
            ops=[qml.RX(0.5, wires=0)],
            measurements=[qml.expval(qml.Z(0))],
        )
        meta = MetaCircuit(source_circuit=qscript, symbols=np.array([], dtype=object))
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                MeasurementStage(),
                QEMStage(protocol=QuEPP(truncation_order=1)),
            ],
        )
        trace = pipeline.run_forward_pass("ignored", dummy_pipeline_env)
        assert len(trace.final_batch) == 1
        final_meta = next(iter(trace.final_batch.values()))
        assert len(final_meta.circuit_body_qasms) >= 2

    def test_effectiveness_with_readout_noise(self):
        """QuEPP mitigates uniform readout noise on a real backend."""
        qscript = qml.tape.QuantumScript(
            ops=[
                qml.Hadamard(0),
                qml.Hadamard(1),
                qml.CNOT(wires=[0, 1]),
                qml.RY(0.8, wires=0),
                qml.RX(0.5, wires=1),
                qml.CNOT(wires=[0, 1]),
                qml.Hadamard(0),
                qml.Hadamard(1),
            ],
            measurements=[qml.expval(qml.Z(0))],
        )
        meta = MetaCircuit(source_circuit=qscript, symbols=np.array([], dtype=object))

        noise = NoiseModel()
        noise.add_all_qubit_readout_error([[0.95, 0.05], [0.05, 0.95]])

        shared = dict(shots=200000, simulation_seed=42, _deterministic_execution=True)

        exact = list(
            CircuitPipeline(stages=[CircuitSpecStage(), MeasurementStage()])
            .run(meta, PipelineEnv(backend=QiskitSimulator(**shared)))
            .values()
        )[0]

        noisy = list(
            CircuitPipeline(stages=[CircuitSpecStage(), MeasurementStage()])
            .run(
                meta, PipelineEnv(backend=QiskitSimulator(noise_model=noise, **shared))
            )
            .values()
        )[0]

        quepp_val = list(
            CircuitPipeline(
                stages=[
                    CircuitSpecStage(),
                    MeasurementStage(),
                    QEMStage(protocol=QuEPP(truncation_order=2)),
                ]
            )
            .run(
                meta, PipelineEnv(backend=QiskitSimulator(noise_model=noise, **shared))
            )
            .values()
        )[0]

        noisy_err = abs(noisy - exact)
        quepp_err = abs(quepp_val - exact)
        assert quepp_err < noisy_err / 2, (
            f"QuEPP error ({quepp_err:.4f}) should be less than half "
            f"of noisy error ({noisy_err:.4f})"
        )


class TestQuEPPSignalDestruction:
    """Tests for signal-destruction detection and post_reduce warning."""

    def _make_context(self, classical_values, weights=None):
        cv = np.array(classical_values)
        w = np.array(weights) if weights is not None else np.ones(len(cv)) / len(cv)
        return {
            "classical_values": cv,
            "weights": w,
            "target_idx": 0,
            "ensemble_start": 1,
            "n_rotations": len(cv),
            "n_paths": len(cv),
        }

    def test_signal_destroyed_flag_set_when_eta_below_threshold(self):
        """reduce() flags _signal_destroyed when eta < min_eta."""
        ctx = self._make_context([0.5, 0.3])
        # Ensemble noisy near zero → eta ≈ 0.02/0.5 ≈ 0.04 < 0.1
        quantum_results = [0.5, 0.01, 0.01]
        QuEPP(truncation_order=1, n_twirls=0).reduce(quantum_results, ctx)
        assert ctx.get("_signal_destroyed") is True

    def test_signal_destroyed_flag_not_set_when_eta_valid(self):
        """reduce() does NOT flag when eta is above threshold."""
        ctx = self._make_context([0.5, 0.3])
        # Ensemble noisy close to classical → eta ≈ 1.0
        quantum_results = [0.5, 0.48, 0.29]
        QuEPP(truncation_order=1, n_twirls=0).reduce(quantum_results, ctx)
        assert "_signal_destroyed" not in ctx

    def test_signal_destroyed_flag_not_set_for_near_zero_classical(self):
        """reduce() does NOT flag when classical values are all near zero."""
        ctx = self._make_context([1e-15, 1e-15])
        quantum_results = [0.5, 0.01, 0.01]
        QuEPP(truncation_order=1, n_twirls=0).reduce(quantum_results, ctx)
        assert "_signal_destroyed" not in ctx

    def test_post_reduce_warns_on_destroyed_signal(self):
        """post_reduce() emits a UserWarning when contexts have destroyed signals."""
        destroyed = {"_signal_destroyed": True}
        healthy = {}
        protocol = QuEPP(truncation_order=1, n_twirls=0)

        with pytest.warns(UserWarning, match=r"signal destroyed for 1/2"):
            protocol.post_reduce([destroyed, healthy])

    def test_post_reduce_silent_when_no_destruction(self):
        """post_reduce() does not warn when all groups are healthy."""
        healthy1 = {}
        healthy2 = {}
        protocol = QuEPP(truncation_order=1, n_twirls=0)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            protocol.post_reduce([healthy1, healthy2])

    def test_post_reduce_default_noop_on_base_class(self):
        """QEMProtocol.post_reduce() is a no-op that does not raise."""
        ctx = {"_signal_destroyed": True}
        _NoMitigation().post_reduce([ctx])  # should not raise

    def test_qem_stage_calls_post_reduce(self, mocker):
        """QEMStage.reduce() calls protocol.post_reduce() with all contexts."""
        protocol = QuEPP(truncation_order=1, n_twirls=0)
        spy = mocker.spy(protocol, "post_reduce")
        stage = QEMStage(protocol=protocol)

        ctx1 = {"_signal_destroyed": True}
        ctx2 = {}
        token = {"key1": ctx1, "key2": ctx2}

        mocker.patch.object(stage, "_reduce_grouped", return_value={})
        mocker.patch.object(stage, "_detect_per_obs", return_value=False)
        mocker.patch(
            "divi.pipeline.stages._qem_stage.group_by_base_key", return_value={}
        )

        stage.reduce({}, PipelineEnv(backend=mocker.MagicMock()), token)

        spy.assert_called_once()
        contexts_passed = spy.call_args[0][0]
        assert len(contexts_passed) == 2
        assert any(c.get("_signal_destroyed") for c in contexts_passed)
