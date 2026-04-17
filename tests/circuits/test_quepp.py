# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.circuits.quepp (DAG-native QuEPP implementation)."""

import warnings

import numpy as np
import pennylane as qml
import pytest
import stim
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from qiskit_aer.noise import NoiseModel

from divi.backends import QiskitSimulator
from divi.circuits import qscript_to_meta
from divi.circuits.qem import _NoMitigation
from divi.circuits.quepp import (
    QuEPP,
    _build_clifford_tableaus,
    _build_path_dag,
    _decompose_controlled_rotations,
    _enumerate_paths_dfs,
    _extract_rotation_gates,
    _has_symbolic_angles,
    _is_pauli_rotation,
    _normalize_angle,
    _normalize_circuit,
    _obs_to_stim_terms,
    _qiskit_clifford_to_stim,
    _sample_paths_montecarlo,
    _simulate_clifford_ensemble,
)
from divi.pipeline import CircuitPipeline, PipelineEnv
from divi.pipeline.stages import CircuitSpecStage, MeasurementStage, QEMStage
from tests.pipeline.helpers import DummySpecStage


@pytest.fixture
def bell_qc():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


@pytest.fixture
def simple_qc():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.rx(0.3, 0)
    qc.cx(0, 1)
    return qc


@pytest.fixture
def mixed_qc():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.rx(0.3, 0)
    qc.cx(0, 1)
    qc.rz(0.7, 1)
    return qc


class TestIsPauliRotation:
    def test_rx_detected(self):
        qc = QuantumCircuit(1)
        qc.rx(0.5, 0)
        axis, angle = _is_pauli_rotation(qc.data[0].operation)
        assert axis == "x"
        assert angle == pytest.approx(0.5)

    def test_ry_detected(self):
        qc = QuantumCircuit(1)
        qc.ry(1.2, 0)
        axis, angle = _is_pauli_rotation(qc.data[0].operation)
        assert axis == "y"
        assert angle == pytest.approx(1.2)

    def test_rz_detected(self):
        qc = QuantumCircuit(1)
        qc.rz(-0.7, 0)
        axis, angle = _is_pauli_rotation(qc.data[0].operation)
        assert axis == "z"
        assert angle == pytest.approx(-0.7)

    def test_non_rotation_returns_none(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        assert _is_pauli_rotation(qc.data[0].operation) is None

    def test_symbolic_angle_returns_parameter_expression(self):
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rx(2 * theta, 0)
        axis, angle = _is_pauli_rotation(qc.data[0].operation)
        assert axis == "x"
        assert isinstance(angle, ParameterExpression)
        assert "theta" in str(angle)


class TestNormalizeAngle:
    def test_small_angle_unchanged(self):
        n, theta_prime = _normalize_angle(0.2)
        assert n == 0
        assert theta_prime == pytest.approx(0.2)

    def test_pi_over_2(self):
        n, theta_prime = _normalize_angle(np.pi / 2)
        assert n == 1
        assert abs(theta_prime) < 1e-12

    def test_large_angle_normalized(self):
        n, theta_prime = _normalize_angle(1.2)
        # 1.2 is closer to π/2 (≈1.5708) than to 0 → n=1.
        assert n == 1
        assert abs(theta_prime) <= np.pi / 4 + 1e-12

    def test_negative_angle(self):
        n, theta_prime = _normalize_angle(-np.pi / 2 - 0.1)
        assert n == -1
        assert abs(theta_prime) <= np.pi / 4 + 1e-12


class TestNormalizeCircuit:
    def test_small_angles_unchanged(self, mixed_qc):
        normalized = _normalize_circuit(mixed_qc)
        assert Operator(normalized).equiv(Operator(mixed_qc))

    def test_pi_over_2_becomes_clifford(self):
        qc = QuantumCircuit(1)
        qc.rx(np.pi / 2, 0)
        normalized = _normalize_circuit(qc)
        rotations = [i for i in normalized.data if i.operation.name == "rx"]
        assert len(rotations) == 0
        assert Operator(normalized).equiv(Operator(qc))

    def test_large_angle_decomposed(self):
        qc = QuantumCircuit(1)
        qc.rx(1.2, 0)
        normalized = _normalize_circuit(qc)
        assert Operator(normalized).equiv(Operator(qc))

    def test_symbolic_angles_passed_through(self):
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        normalized = _normalize_circuit(qc)
        names = [i.operation.name for i in normalized.data]
        assert "rx" in names


class TestDecomposeControlledRotations:
    @pytest.mark.parametrize("method,axis", [("crx", "x"), ("cry", "y"), ("crz", "z")])
    def test_unitary_preserved(self, method, axis):
        qc = QuantumCircuit(2)
        getattr(qc, method)(0.6, 0, 1)
        decomposed = _decompose_controlled_rotations(qc)
        assert Operator(decomposed).equiv(Operator(qc))
        names = {i.operation.name for i in decomposed.data}
        assert method not in names

    def test_non_controlled_unchanged(self, mixed_qc):
        out = _decompose_controlled_rotations(mixed_qc)
        assert Operator(out).equiv(Operator(mixed_qc))


class TestExtractRotationGates:
    def test_fully_clifford(self, bell_qc):
        assert _extract_rotation_gates(bell_qc) == []

    def test_single_rotation(self, simple_qc):
        rots = _extract_rotation_gates(simple_qc)
        assert len(rots) == 1
        assert rots[0].axis == "x"
        assert rots[0].angle == pytest.approx(0.3)

    def test_mixed_circuit(self, mixed_qc):
        rots = _extract_rotation_gates(mixed_qc)
        assert [r.axis for r in rots] == ["x", "z"]
        assert [r.qubit_idx for r in rots] == [0, 1]


class TestQiskitCliffordToStim:
    def test_basic_cliffords(self, bell_qc):
        sc = _qiskit_clifford_to_stim(bell_qc)
        assert sc.num_qubits == 2
        # Tableau builds successfully (no exception).
        tab = stim.Tableau.from_circuit(sc)
        assert len(tab) == 2

    def test_clifford_rotation(self):
        qc = QuantumCircuit(1)
        qc.rx(np.pi / 2, 0)
        sc = _qiskit_clifford_to_stim(qc)
        assert "SQRT_X" in str(sc)

    def test_non_clifford_raises(self):
        qc = QuantumCircuit(1)
        qc.rx(0.3, 0)
        with pytest.raises(ValueError, match="Non-Clifford angle"):
            _qiskit_clifford_to_stim(qc)

    def test_parametric_raises(self):
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        with pytest.raises(ValueError, match="parametric"):
            _qiskit_clifford_to_stim(qc)


class TestObsToStimTerms:
    def test_single_pauli_qubit_0(self):
        obs = SparsePauliOp.from_list([("IZ", 1.0)])  # Z on qubit 0
        terms = _obs_to_stim_terms(obs, 2)
        assert len(terms) == 1
        coeff, ps = terms[0]
        assert coeff == pytest.approx(1.0)
        # big-endian stim label: qubit 0 on the left → "Z_"
        assert str(ps) == "+Z_"

    def test_multi_term(self):
        obs = SparsePauliOp.from_list([("IZ", 0.5), ("ZI", -0.3)])
        terms = _obs_to_stim_terms(obs, 2)
        coeffs = sorted(c for c, _ in terms)
        assert coeffs == pytest.approx([-0.3, 0.5])


class TestEnumeratePathsDFS:
    def test_no_rotations_single_identity_path(self, bell_qc):
        obs = SparsePauliOp.from_list([("ZZ", 1.0)])
        rots = _extract_rotation_gates(bell_qc)
        tabs = _build_clifford_tableaus(bell_qc, rots)
        obs_terms = _obs_to_stim_terms(obs, 2)
        paths = _enumerate_paths_dfs(rots, tabs, obs_terms, max_order=2)
        assert len(paths) == 1
        assert paths[0].branches == ()
        assert paths[0].weight == pytest.approx(1.0)
        assert paths[0].order == 0

    def test_coefficient_threshold_prunes(self, mixed_qc):
        obs = SparsePauliOp.from_list([("IZ", 1.0)])
        rots = _extract_rotation_gates(mixed_qc)
        tabs = _build_clifford_tableaus(mixed_qc, rots)
        obs_terms = _obs_to_stim_terms(obs, 2)
        paths_all = _enumerate_paths_dfs(rots, tabs, obs_terms, max_order=2)
        paths_pruned = _enumerate_paths_dfs(
            rots, tabs, obs_terms, max_order=2, coefficient_threshold=0.5
        )
        assert len(paths_pruned) <= len(paths_all)


class TestSamplePathsMonteCarlo:
    def test_deterministic_with_seed(self, mixed_qc):
        obs = SparsePauliOp.from_list([("IZ", 1.0)])
        rots = _extract_rotation_gates(mixed_qc)
        tabs = _build_clifford_tableaus(mixed_qc, rots)
        obs_terms = _obs_to_stim_terms(obs, 2)
        # This small circuit + observable triggers the MC fallback (all samples
        # non-diagonal).  Assert the warning is emitted and results are still
        # deterministic across identical seeds.
        with pytest.warns(UserWarning, match="non-diagonal Pauli strings"):
            rng1 = np.random.default_rng(42)
            paths1 = _sample_paths_montecarlo(rots, tabs, obs_terms, 100, rng1)
            rng2 = np.random.default_rng(42)
            paths2 = _sample_paths_montecarlo(rots, tabs, obs_terms, 100, rng2)
        assert sorted(p.branches for p in paths1) == sorted(p.branches for p in paths2)


class TestSimulateCliffordEnsemble:
    def test_bell_state_zz(self, bell_qc):
        obs = SparsePauliOp.from_list([("ZZ", 1.0)])
        vals = _simulate_clifford_ensemble([bell_qc], obs, 2)
        assert vals[0] == pytest.approx(1.0)

    def test_bell_state_xx(self, bell_qc):
        obs = SparsePauliOp.from_list([("XX", 1.0)])
        vals = _simulate_clifford_ensemble([bell_qc], obs, 2)
        assert vals[0] == pytest.approx(1.0)

    def test_batch_returns_correct_count(self, bell_qc):
        obs = SparsePauliOp.from_list([("ZZ", 1.0)])
        vals = _simulate_clifford_ensemble([bell_qc, bell_qc, bell_qc], obs, 2)
        assert vals.shape == (3,)


class TestHasSymbolicAngles:
    def test_concrete_angles(self, mixed_qc):
        assert _has_symbolic_angles(mixed_qc) is False

    def test_symbolic_angle(self):
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        assert _has_symbolic_angles(qc) is True

    def test_mixed_symbolic_and_concrete(self):
        theta = Parameter("theta")
        qc = QuantumCircuit(2)
        qc.rx(0.5, 0)
        qc.rz(theta, 1)
        assert _has_symbolic_angles(qc) is True

    def test_non_rotation_gates_ignored(self, bell_qc):
        assert _has_symbolic_angles(bell_qc) is False


class TestQuEPPProtocol:
    def test_expand_returns_circuits_and_context(
        self, mixed_qc, suppress_quepp_warnings
    ):
        obs = SparsePauliOp.from_list([("IZ", 0.5), ("ZI", -0.3)])
        p = QuEPP(truncation_order=2, sampling="exhaustive", n_twirls=0)
        dags, ctx = p.expand(circuit_to_dag(mixed_qc), obs)
        assert len(dags) == ctx["n_paths"] + 1
        assert "classical_values" in ctx
        assert ctx["target_idx"] == 0
        assert ctx["ensemble_start"] == 1

    def test_expand_clifford_circuit(self, bell_qc):
        obs = SparsePauliOp.from_list([("ZZ", 1.0)])
        p = QuEPP(truncation_order=0, sampling="exhaustive", n_twirls=0)
        dags, ctx = p.expand(circuit_to_dag(bell_qc), obs)
        assert ctx["n_rotations"] == 0
        assert ctx["n_paths"] == 1
        assert len(dags) == 2

    def test_reduce_clifford_circuit_exact(self, bell_qc):
        obs = SparsePauliOp.from_list([("ZZ", 1.0)])
        p = QuEPP(truncation_order=0, sampling="exhaustive", n_twirls=0)
        _, ctx = p.expand(circuit_to_dag(bell_qc), obs)
        assert ctx["classical_values"][0] == pytest.approx(1.0)
        # No rotations → reduce returns weights @ classical_values.
        result = p.reduce([1.0, 1.0], ctx)
        assert result == pytest.approx(1.0)

    def test_missing_observable_raises(self, bell_qc):
        p = QuEPP()
        with pytest.raises(ValueError, match="observable"):
            p.expand(circuit_to_dag(bell_qc), None)

    def test_wrong_observable_type_raises(self, bell_qc):
        p = QuEPP()
        with pytest.raises(TypeError, match="SparsePauliOp"):
            p.expand(circuit_to_dag(bell_qc), "not an observable")

    def test_montecarlo_expand(self, mixed_qc, suppress_quepp_warnings):
        obs = SparsePauliOp.from_list([("IZ", 1.0)])
        p = QuEPP(sampling="montecarlo", n_samples=100, seed=42, n_twirls=0)
        _, ctx = p.expand(circuit_to_dag(mixed_qc), obs)
        assert ctx["n_paths"] >= 1


class TestQuEPPSignalDestruction:
    def test_low_eta_triggers_fallback(self):
        """When noisy/classical ratio falls below min_eta, reduce returns
        the raw target and marks ``_signal_destroyed`` so post_reduce can warn.
        """
        # Non-zero classical values (so "valid" mask has entries) but the
        # ensemble_noisy values are ~0 ⇒ η ≈ 0 < min_eta (0.1) ⇒ fallback.
        ctx = {
            "classical_values": np.array([1.0, 0.5]),
            "weights": np.array([0.5, 0.5]),
            "target_idx": 0,
            "ensemble_start": 1,
            "n_rotations": 1,
            "n_paths": 2,
        }
        p = QuEPP(n_twirls=0)
        result = p.reduce([0.3, 0.0, 0.0], ctx)
        assert result == pytest.approx(0.3)
        assert ctx.get("_signal_destroyed") is True


class TestSymbolicExpand:
    def test_expand_marks_symbolic(self, suppress_quepp_warnings):
        theta = Parameter("theta")
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rx(theta, 0)
        qc.cx(0, 1)
        obs = SparsePauliOp.from_list([("IZ", 1.0)])
        p = QuEPP(sampling="exhaustive", truncation_order=1, n_twirls=0)
        _, ctx = p.expand(circuit_to_dag(qc), obs)
        assert ctx.get("symbolic") is True
        assert [str(s) for s in ctx["weight_symbols"]] == ["theta"]

    def test_weights_are_parameter_expressions(self, suppress_quepp_warnings):
        theta = Parameter("theta")
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rx(theta, 0)
        qc.cx(0, 1)
        obs = SparsePauliOp.from_list([("IZ", 1.0)])
        p = QuEPP(sampling="exhaustive", truncation_order=1, n_twirls=0)
        _, ctx = p.expand(circuit_to_dag(qc), obs)
        assert ctx["weights"].dtype == object
        for w in ctx["weights"]:
            assert isinstance(w, (ParameterExpression, int, float))

    def test_montecarlo_falls_back_to_exhaustive(self, suppress_quepp_warnings):
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        p = QuEPP(sampling="montecarlo", n_samples=10, truncation_order=1, n_twirls=0)
        with pytest.warns(UserWarning, match="Monte Carlo"):
            _, ctx = p.expand(circuit_to_dag(qc), obs)
        assert ctx.get("symbolic") is True


class TestEvaluateSymbolicWeights:
    def test_substitutes_concrete_values(self):
        theta = Parameter("theta_eval")
        # Build ParameterExpression weights: cos(theta) and sin(theta).
        cos_w = theta.cos()
        sin_w = theta.sin()
        ctx = {
            "weights": np.array([cos_w, sin_w], dtype=object),
            "symbolic": True,
        }
        QuEPP.evaluate_symbolic_weights(ctx, [theta], np.array([0.0]))
        assert ctx["weights"][0] == pytest.approx(1.0)
        assert ctx["weights"][1] == pytest.approx(0.0)
        assert ctx["symbolic"] is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rx_qc(angle: float) -> QuantumCircuit:
    """Single-qubit Rx(angle) circuit."""
    qc = QuantumCircuit(1)
    qc.rx(angle, 0)
    return qc


def _exact_expval(qc: QuantumCircuit, obs: SparsePauliOp) -> float:
    """Exact expectation value via statevector."""
    sv = Statevector.from_instruction(qc)
    return float(np.real(sv.expectation_value(obs)))


# ---------------------------------------------------------------------------
# CPT correctness tests
# ---------------------------------------------------------------------------


class TestCPTExpansion:
    """Verify that the Heisenberg CPT expansion recovers exact expectation values."""

    def test_single_rx(self, suppress_quepp_warnings):
        """Rx(θ) with Z observable → cos(θ)."""
        angle = 0.8
        qc = _rx_qc(angle)
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        _, ctx = QuEPP(sampling="exhaustive", truncation_order=5, n_twirls=0).expand(
            circuit_to_dag(qc), obs
        )
        cpt = float(ctx["weights"] @ ctx["classical_values"])
        assert cpt == pytest.approx(np.cos(angle), abs=1e-6)

    def test_h_rx_h_ry(self, suppress_quepp_warnings):
        """Multi-gate single-qubit circuit."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rx(0.3, 0)
        qc.h(0)
        qc.ry(0.5, 0)
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        _, ctx = QuEPP(sampling="exhaustive", truncation_order=5, n_twirls=0).expand(
            circuit_to_dag(qc), obs
        )
        cpt = float(ctx["weights"] @ ctx["classical_values"])
        assert cpt == pytest.approx(_exact_expval(qc, obs), abs=1e-4)

    def test_two_qubit_circuit(self, mixed_qc, suppress_quepp_warnings):
        """Two-qubit circuit with ZZ observable."""
        obs = SparsePauliOp.from_list([("ZZ", 1.0)])
        _, ctx = QuEPP(sampling="exhaustive", truncation_order=5, n_twirls=0).expand(
            circuit_to_dag(mixed_qc), obs
        )
        cpt = float(ctx["weights"] @ ctx["classical_values"])
        assert cpt == pytest.approx(_exact_expval(mixed_qc, obs), abs=1e-4)

    def test_commuting_gate_no_branch(self, suppress_quepp_warnings):
        """When observable commutes with rotation generator, no branching occurs.

        Rx with X observable — X commutes with X generator, so the gate
        is transparent.  The back-propagated observable stays X, which is
        not diagonal, so the path has zero contribution.
        """
        qc = _rx_qc(0.5)
        obs = SparsePauliOp.from_list([("X", 1.0)])
        _, ctx = QuEPP(sampling="exhaustive", truncation_order=5, n_twirls=0).expand(
            circuit_to_dag(qc), obs
        )
        assert ctx["n_paths"] == 0
        assert float(ctx["weights"] @ ctx["classical_values"]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Additional decomposition tests
# ---------------------------------------------------------------------------


class TestDecomposeControlledRotationsExtended:
    """Additional controlled-rotation decomposition tests."""

    def test_clifford_cry_produces_no_rotations(self):
        """CRY(π) is Clifford — after decomposition and normalization, no rotations."""
        qc = QuantumCircuit(2)
        qc.cry(np.pi, 0, 1)
        dc = _decompose_controlled_rotations(qc)
        nc = _normalize_circuit(dc)
        rots = _extract_rotation_gates(nc)
        assert len(rots) == 0

    def test_non_clifford_cry_produces_rotations(self):
        """CRY(0.7) decomposes into two Ry rotations (θ/2 and -θ/2)."""
        qc = QuantumCircuit(2)
        qc.cry(0.7, 0, 1)
        dc = _decompose_controlled_rotations(qc)
        rots = _extract_rotation_gates(dc)
        assert len(rots) == 2
        assert rots[0].axis == "y"
        assert rots[1].axis == "y"


# ---------------------------------------------------------------------------
# Normalization accuracy
# ---------------------------------------------------------------------------


class TestNormalizeCircuitExtended:
    def test_cpt_accuracy_with_normalization(self, suppress_quepp_warnings):
        """CPT expansion on normalized circuit still recovers exact value."""
        angle = 1.2  # > π/4, so normalization kicks in
        qc = _rx_qc(angle)
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        _, ctx = QuEPP(sampling="exhaustive", truncation_order=5, n_twirls=0).expand(
            circuit_to_dag(qc), obs
        )
        cpt = float(ctx["weights"] @ ctx["classical_values"])
        assert cpt == pytest.approx(np.cos(angle), abs=1e-6)


# ---------------------------------------------------------------------------
# Monte Carlo weights
# ---------------------------------------------------------------------------


class TestMCWeightsConvergence:
    def test_mc_weights_are_cpt_coefficients(self, suppress_quepp_warnings):
        """MC IS-weighted paths converge to the correct CPT estimate."""
        angle = 0.5
        qc = _rx_qc(angle)
        nc = _normalize_circuit(qc)
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        rots = _extract_rotation_gates(nc)
        tabs = _build_clifford_tableaus(nc, rots)
        obs_terms = _obs_to_stim_terms(obs, 1)
        paths = _sample_paths_montecarlo(
            rots, tabs, obs_terms, 1000, np.random.default_rng(42)
        )
        weights = np.array([p.weight for p in paths])
        nc_dag = circuit_to_dag(nc)
        rotation_positions = [(rot.inst_idx, rot) for rot in rots]
        path_dags = [
            _build_path_dag(nc_dag, rotation_positions, p.branches) for p in paths
        ]
        cv = _simulate_clifford_ensemble(path_dags, obs, 1)
        mc_estimate = float(weights @ cv)
        assert mc_estimate == pytest.approx(np.cos(angle), abs=0.05)


# ---------------------------------------------------------------------------
# Full protocol round-trip and noise correction
# ---------------------------------------------------------------------------


class TestQuEPPRoundTrip:
    def test_full_round_trip_single_qubit(self, suppress_quepp_warnings):
        """expand → reduce with exact quantum results recovers ideal value."""
        angle = 0.8
        qc = _rx_qc(angle)
        exact = np.cos(angle)
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        protocol = QuEPP(sampling="exhaustive", truncation_order=10, n_twirls=0)
        _, ctx = protocol.expand(circuit_to_dag(qc), obs)
        qr = [exact]
        qr.extend(ctx["classical_values"])
        assert protocol.reduce(qr, ctx) == pytest.approx(exact, abs=1e-6)

    def test_noise_correction(self, suppress_quepp_warnings):
        """QuEPP corrects a globally-scaled noise bias."""
        angle = 0.8
        qc = _rx_qc(angle)
        exact = np.cos(angle)
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        protocol = QuEPP(sampling="exhaustive", truncation_order=10, n_twirls=0)
        _, ctx = protocol.expand(circuit_to_dag(qc), obs)
        noise_factor = 0.9
        qr = [exact * noise_factor]
        qr.extend(ctx["classical_values"] * noise_factor)
        assert protocol.reduce(qr, ctx) == pytest.approx(exact, abs=1e-4)

    def test_expand_with_controlled_rotation(self, suppress_quepp_warnings):
        """Full QuEPP expand works on a circuit with controlled rotations."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cry(0.5, 0, 1)
        qc.ry(0.3, 1)
        obs = SparsePauliOp.from_list([("ZZ", 1.0)])
        protocol = QuEPP(sampling="exhaustive", truncation_order=5, n_twirls=0)
        _, ctx = protocol.expand(circuit_to_dag(qc), obs)
        cpt = float(ctx["weights"] @ ctx["classical_values"])
        assert cpt == pytest.approx(_exact_expval(qc, obs), abs=1e-4)


# ---------------------------------------------------------------------------
# Signal destruction (extended)
# ---------------------------------------------------------------------------


class TestQuEPPSignalDestructionExtended:
    """Additional signal-destruction detection and post_reduce tests."""

    @staticmethod
    def _make_context(classical_values, weights=None):
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
        with pytest.warns(UserWarning, match=r"signal destroyed"):
            protocol.post_reduce([destroyed, healthy])

    def test_post_reduce_silent_when_no_destruction(self):
        """post_reduce() does not warn when all groups are healthy."""
        protocol = QuEPP(truncation_order=1, n_twirls=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            protocol.post_reduce([{}, {}])

    def test_post_reduce_default_noop_on_base_class(self):
        """QEMProtocol.post_reduce() is a no-op that does not raise."""
        ctx = {"_signal_destroyed": True}
        _NoMitigation().post_reduce([ctx])  # should not raise


# ---------------------------------------------------------------------------
# Shallow circuit warning
# ---------------------------------------------------------------------------


class TestShallowCircuitWarning:
    def test_shallow_circuit_warning_in_expand(self):
        """expand() warns when K / n_rotations > 0.33 (shallow circuit)."""
        qc = QuantumCircuit(2)
        qc.rx(0.3, 0)
        qc.cx(0, 1)
        qc.ry(0.7, 1)
        obs = SparsePauliOp.from_list([("IZ", 1.0)])
        protocol = QuEPP(sampling="exhaustive", truncation_order=2, n_twirls=0)
        with pytest.warns(UserWarning, match=r"large fraction"):
            protocol.expand(circuit_to_dag(qc), obs)

    def test_no_shallow_circuit_warning_for_deep_circuits(self):
        """expand() does NOT warn when K / n_rotations is small."""
        qc = QuantumCircuit(2)
        for i in range(10):
            qc.rx(0.1 * (i + 1), i % 2)
        obs = SparsePauliOp.from_list([("IZ", 1.0)])
        protocol = QuEPP(sampling="exhaustive", truncation_order=1, n_twirls=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            protocol.expand(circuit_to_dag(qc), obs)


# ---------------------------------------------------------------------------
# Symbolic hybrid normalization
# ---------------------------------------------------------------------------


class TestSymbolicHybridNormalization:
    def test_hybrid_normalization(self, suppress_quepp_warnings):
        """Concrete rotations are normalized; symbolic ones are kept as-is."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        # Rx(π/2) is concrete Clifford → normalized away; Rx(theta) is symbolic → kept
        qc.rx(np.pi / 2, 0)
        qc.rx(theta, 0)
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        _, ctx = QuEPP(sampling="exhaustive", truncation_order=1, n_twirls=0).expand(
            circuit_to_dag(qc), obs
        )
        # Only the symbolic rotation should remain
        assert ctx["n_rotations"] == 1


# ---------------------------------------------------------------------------
# Bind-before-mitigation flag
# ---------------------------------------------------------------------------


class TestBindBeforeMitigation:
    def test_default_is_false(self):
        assert QuEPP().bind_before_mitigation is False

    def test_stored_when_true(self):
        assert QuEPP(bind_before_mitigation=True).bind_before_mitigation is True


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


class TestQuEPPPipelineIntegration:
    def test_pipeline_integration(self, dummy_pipeline_env, suppress_quepp_warnings):
        """QuEPP integrates correctly with QEMStage in a pipeline."""
        qscript = qml.tape.QuantumScript(
            ops=[qml.RX(0.5, wires=0)],
            measurements=[qml.expval(qml.Z(0))],
        )
        meta = qscript_to_meta(qscript)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                QEMStage(protocol=QuEPP(truncation_order=1, n_twirls=0)),
                MeasurementStage(),
            ],
        )
        trace = pipeline.run_forward_pass("ignored", dummy_pipeline_env)
        assert len(trace.final_batch) == 1
        final_meta = next(iter(trace.final_batch.values()))
        assert len(final_meta.circuit_bodies) >= 2

    @pytest.mark.e2e
    def test_effectiveness_with_readout_noise(self, suppress_quepp_warnings):
        """QuEPP mitigates uniform readout noise on a real backend."""
        qscript = qml.tape.QuantumScript(
            ops=[qml.RX(0.8, wires=0)],
            measurements=[qml.expval(qml.Z(0))],
        )
        meta = qscript_to_meta(qscript)

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
                meta,
                PipelineEnv(backend=QiskitSimulator(noise_model=noise, **shared)),
            )
            .values()
        )[0]

        quepp_val = list(
            CircuitPipeline(
                stages=[
                    CircuitSpecStage(),
                    QEMStage(
                        protocol=QuEPP(
                            sampling="exhaustive",
                            truncation_order=5,
                            n_twirls=0,
                        )
                    ),
                    MeasurementStage(),
                ]
            )
            .run(
                meta,
                PipelineEnv(backend=QiskitSimulator(noise_model=noise, **shared)),
            )
            .values()
        )[0]

        noisy_err = abs(noisy - exact)
        quepp_err = abs(quepp_val - exact)
        assert quepp_err < noisy_err / 2, (
            f"QuEPP error ({quepp_err:.4f}) should be less than half "
            f"of noisy error ({noisy_err:.4f})"
        )
