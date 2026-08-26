# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.circuits.quepp (DAG-native QuEPP implementation)."""

import warnings

import numpy as np
import pytest
import stim
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from qiskit_aer.noise import NoiseModel

from divi.backends import QiskitSimulator
from divi.circuits import MetaCircuit
from divi.circuits.qem import _NoMitigation
from divi.circuits.quepp import (
    QuEPP,
    SymbolicAngleWarning,
    _all_cos_paths,
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
    _ObservableCPT,
    _PreprocResult,
    _qiskit_clifford_to_stim,
    _sample_paths_montecarlo,
    _simulate_clifford_ensemble,
)
from divi.pipeline import CircuitPipeline, PipelineEnv
from divi.pipeline.stages import CircuitSpecStage, MeasurementStage, QEMStage
from tests.pipeline._helpers import DummySpecStage, meta_from_circuit

_Z0 = SparsePauliOp.from_list([("Z", 1.0)])
_Z0_2Q = SparsePauliOp.from_list([("IZ", 1.0)])
_Z0Z1 = SparsePauliOp.from_list([("ZZ", 1.0)])


def _rx_expval_meta(angle: float) -> MetaCircuit:
    """Single ``RX(angle)`` measured as ``<Z0>``."""
    qc = QuantumCircuit(1)
    qc.rx(angle, 0)
    return meta_from_circuit(qc, observable=_Z0)


def _entangled_two_qubit_circuit() -> QuantumCircuit:
    """The shared body of the end-to-end QuEPP pipeline tests."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.rx(0.3, 0)
    qc.cx(0, 1)
    qc.rz(0.7, 1)
    return qc


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


class TestPathDagConstruction:
    def test_rotation_indices_align_with_working_dag_topology(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rx(0.2, 0)
        qc.cx(0, 1)
        qc.rz(-0.4, 1)

        obs = SparsePauliOp.from_list([("ZZ", 1.0)])
        prep = QuEPP._preprocess(circuit_to_dag(qc), obs)
        working_dag = circuit_to_dag(prep.working)
        topo_nodes = list(working_dag.topological_op_nodes())

        for rot in prep.rotations:
            node = topo_nodes[rot.inst_idx]
            assert node.op.name == f"r{rot.axis}"
            assert working_dag.find_bit(node.qargs[0]).index == rot.qubit_idx

    def test_single_rotation_branch_semantics(self):
        qc = QuantumCircuit(1)
        qc.rx(0.3, 0)
        working_dag = circuit_to_dag(qc)
        rotations = _extract_rotation_gates(qc)
        rotation_positions = [(rot.inst_idx, rot) for rot in rotations]

        skip_dag = _build_path_dag(working_dag, rotation_positions, (0,))
        replace_dag = _build_path_dag(working_dag, rotation_positions, (1,))

        identity_qc = QuantumCircuit(1)
        clifford_qc = QuantumCircuit(1)
        clifford_qc.rx(np.pi / 2, 0)

        assert Operator(dag_to_circuit(skip_dag)).equiv(Operator(identity_qc))
        assert Operator(dag_to_circuit(replace_dag)).equiv(Operator(clifford_qc))

    def test_branch_tuple_order_matches_rotation_order(self):
        qc = QuantumCircuit(1)
        qc.rx(0.3, 0)
        qc.rz(0.4, 0)
        working_dag = circuit_to_dag(qc)
        rotations = _extract_rotation_gates(qc)
        rotation_positions = [(rot.inst_idx, rot) for rot in rotations]

        first_only = _build_path_dag(working_dag, rotation_positions, (1, 0))
        second_only = _build_path_dag(working_dag, rotation_positions, (0, 1))

        first_expected = QuantumCircuit(1)
        first_expected.rx(np.pi / 2, 0)
        second_expected = QuantumCircuit(1)
        second_expected.rz(np.pi / 2, 0)

        assert Operator(dag_to_circuit(first_only)).equiv(Operator(first_expected))
        assert Operator(dag_to_circuit(second_only)).equiv(Operator(second_expected))


def test_deterministic_with_seed(mixed_qc):
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


def _all_cos_weight_of(rots, inv_tabs, obs_terms, term_idx=0) -> float:
    """The all-cos path weight for one term, or 0.0 when it has no such path."""
    path = next(
        (
            p
            for p in _all_cos_paths(rots, inv_tabs, obs_terms)
            if p.term_idx == term_idx
        ),
        None,
    )
    return 0.0 if path is None else path.weight


class TestAllCosPaths:
    """Spec: ``_all_cos_paths`` returns the branches=(0,)*K path per observable term."""

    def test_matches_exhaustive_dfs_all_zero_branch(self):
        """Deterministic fallback weight matches the DFS-enumerated all-zero path."""
        angle = 0.7
        qc = _rx_qc(angle)  # Rx(θ) on qubit 0
        nc = _normalize_circuit(qc)
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        rots = _extract_rotation_gates(nc)
        tabs = _build_clifford_tableaus(nc, rots)
        inv_tabs = [t.inverse() for t in tabs]
        obs_terms = _obs_to_stim_terms(obs, 1)

        fallback_w = _all_cos_weight_of(rots, inv_tabs, obs_terms)

        # Cross-check with the exhaustive enumeration's all-zero branch.
        dfs_paths = _enumerate_paths_dfs(rots, tabs, obs_terms, max_order=10)
        zero_branch = next(p for p in dfs_paths if all(b == 0 for b in p.branches))
        assert fallback_w == pytest.approx(zero_branch.weight, rel=1e-12)
        assert fallback_w == pytest.approx(np.cos(angle), abs=1e-12)

    def test_emits_no_path_when_no_term_diagonal(self):
        """Observable that never propagates to a diagonal Pauli yields no path."""
        qc = _rx_qc(0.4)
        nc = _normalize_circuit(qc)
        obs = SparsePauliOp.from_list([("X", 1.0)])  # X commutes with Rx
        rots = _extract_rotation_gates(nc)
        tabs = _build_clifford_tableaus(nc, rots)
        inv_tabs = [t.inverse() for t in tabs]
        obs_terms = _obs_to_stim_terms(obs, 1)

        # X commutes with Rx so no cos factor accumulates, but the final
        # Pauli is still X — non-diagonal — so the term contributes no path.
        assert _all_cos_paths(rots, inv_tabs, obs_terms) == []

    def test_weights_are_coefficient_free_and_per_term(self):
        """Only the Z term propagates diagonally, and its weight excludes its coeff.

        The coefficient is applied later, to the measured value the weight
        pairs with; folding it in here would double-count it.
        """
        angle = 0.5
        qc = _rx_qc(angle)
        nc = _normalize_circuit(qc)
        obs = SparsePauliOp.from_list([("Z", 0.8), ("X", 0.2)])  # X term → no path
        rots = _extract_rotation_gates(nc)
        tabs = _build_clifford_tableaus(nc, rots)
        inv_tabs = [t.inverse() for t in tabs]
        obs_terms = _obs_to_stim_terms(obs, 1)

        paths = _all_cos_paths(rots, inv_tabs, obs_terms)

        assert [p.term_idx for p in paths] == [0]
        assert paths[0].weight == pytest.approx(np.cos(angle), abs=1e-12)

    def test_mc_fallback_returns_computed_weight(self, mixed_qc):
        """When every MC sample is discarded, the returned path carries the all-cos weight."""
        obs = SparsePauliOp.from_list([("IZ", 1.0)])
        rots = _extract_rotation_gates(mixed_qc)
        tabs = _build_clifford_tableaus(mixed_qc, rots)
        inv_tabs = [t.inverse() for t in tabs]
        obs_terms = _obs_to_stim_terms(obs, 2)
        expected = _all_cos_paths(rots, inv_tabs, obs_terms)

        with pytest.warns(UserWarning, match="non-diagonal Pauli strings"):
            paths = _sample_paths_montecarlo(
                rots, tabs, obs_terms, 100, np.random.default_rng(42)
            )

        if not expected:
            # Fallback correctly declines to fabricate a path with bogus weight.
            assert paths == []
        else:
            assert len(paths) == len(expected)
            assert paths[0].branches == (0,) * len(rots)
            assert paths[0].weight == pytest.approx(expected[0].weight, rel=1e-12)


def _single_term_specs(obs: SparsePauliOp, n_qubits: int, n_circuits: int):
    """``entry_terms`` measuring ``obs``'s only Pauli on each of *n_circuits*."""
    (term,) = _obs_to_stim_terms(obs, n_qubits)
    return [term] * n_circuits


@pytest.mark.usefixtures("suppress_quepp_warnings")
class TestPathCircuitsMatchTargetStructure:
    """Path circuits must stay structurally comparable to the target.

    η measures how much noise the ensemble suffers relative to the target,
    so a path circuit that is cheaper than the target decoheres less, η
    over-reports the surviving signal, and the correction comes out too
    small. Per the paper's Eq. (2), a cos branch substitutes the identity
    gate — it does not drop the instruction.
    """

    @staticmethod
    def _expanded(truncation_order=2):
        qc = QuantumCircuit(3)
        qc.h(range(3))
        for _ in range(2):
            qc.cz(0, 1)
            qc.cz(1, 2)
            for q in range(3):
                qc.rx(0.6, q)
        obs = SparsePauliOp.from_list([("ZZZ", 1.0)])
        protocol = QuEPP(
            sampling="exhaustive", truncation_order=truncation_order, n_twirls=0
        )
        dags, _ = protocol.expand(circuit_to_dag(qc), (obs,))
        return [dag_to_circuit(d) for d in dags]

    def test_cos_branch_substitutes_identity_rather_than_dropping_the_gate(self):
        """Every replaced rotation leaves an idle slot behind."""
        target, *paths = self._expanded()
        n_rotations = target.count_ops()["rx"]

        for path in paths:
            ops = path.count_ops()
            # Each rotation became either an identity or a Clifford rotation.
            substituted = ops.get("id", 0) + ops.get("sx", 0) + ops.get("sxdg", 0)
            assert substituted == n_rotations
            assert "rx" not in ops

    def test_path_circuits_preserve_target_depth(self):
        """A deleted rotation would shorten the circuit; an idle slot does not."""
        target, *paths = self._expanded()
        assert paths
        for path in paths:
            assert path.depth() == target.depth()

    def test_two_qubit_gate_count_is_untouched(self):
        """Only rotations are replaced — the entangling structure is shared."""
        target, *paths = self._expanded()
        for path in paths:
            assert path.count_ops().get("cz", 0) == target.count_ops()["cz"]


class TestSimulateCliffordEnsemble:
    def test_bell_state_zz(self, bell_qc):
        obs = SparsePauliOp.from_list([("ZZ", 1.0)])
        vals = _simulate_clifford_ensemble([bell_qc], _single_term_specs(obs, 2, 1))
        assert vals[0] == pytest.approx(1.0)

    def test_bell_state_xx(self, bell_qc):
        obs = SparsePauliOp.from_list([("XX", 1.0)])
        vals = _simulate_clifford_ensemble([bell_qc], _single_term_specs(obs, 2, 1))
        assert vals[0] == pytest.approx(1.0)

    def test_batch_returns_correct_count(self, bell_qc):
        obs = SparsePauliOp.from_list([("ZZ", 1.0)])
        vals = _simulate_clifford_ensemble([bell_qc] * 3, _single_term_specs(obs, 2, 3))
        assert vals.shape == (3,)

    def test_applies_each_entrys_own_coefficient(self, bell_qc):
        """One circuit, two terms: each value carries only its own coefficient."""
        terms = _obs_to_stim_terms(
            SparsePauliOp.from_list([("ZZ", 0.5), ("XX", -2.0)]), 2
        )
        vals = _simulate_clifford_ensemble([bell_qc, bell_qc], terms)
        assert vals == pytest.approx([0.5, -2.0])


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
    @pytest.mark.usefixtures("suppress_quepp_warnings")
    def test_expand_returns_circuits_and_context(self, mixed_qc):
        obs = SparsePauliOp.from_list([("IZ", 0.5), ("ZI", -0.3)])
        p = QuEPP(truncation_order=2, sampling="exhaustive", n_twirls=0)
        dags, ctx = p.expand(circuit_to_dag(mixed_qc), obs)
        assert len(dags) == ctx["n_paths"] + 1
        assert isinstance(ctx["per_obs"][0], _ObservableCPT)
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
        assert ctx["per_obs"][0].classical_values[0] == pytest.approx(1.0)
        # No rotations → reduce returns weights @ classical_values.
        result = p.reduce([1.0, 1.0], ctx)
        assert result == pytest.approx([1.0])

    def test_missing_observable_raises(self, bell_qc):
        p = QuEPP()
        with pytest.raises(ValueError, match="observable"):
            p.expand(circuit_to_dag(bell_qc), None)

    def test_wrong_observable_type_raises(self, bell_qc):
        p = QuEPP()
        with pytest.raises(TypeError, match="SparsePauliOp"):
            p.expand(circuit_to_dag(bell_qc), "not an observable")

    @pytest.mark.usefixtures("suppress_quepp_warnings")
    def test_montecarlo_expand(self, mixed_qc):
        # ZZ (rather than IZ) picks an observable that propagates to a
        # diagonal Pauli under mixed_qc, so MC sampling returns real
        # diagonal paths and does not fall back.
        obs = SparsePauliOp.from_list([("ZZ", 1.0)])
        p = QuEPP(sampling="montecarlo", n_samples=100, seed=42, n_twirls=0)
        _, ctx = p.expand(circuit_to_dag(mixed_qc), obs)
        assert ctx["n_paths"] >= 1

    @pytest.mark.usefixtures("suppress_quepp_warnings")
    def test_montecarlo_all_discarded_returns_empty_when_fallback_is_zero(
        self, mixed_qc
    ):
        """When all MC samples are non-diagonal *and* the all-cos fallback has
        no diagonal contribution, the protocol correctly produces zero paths
        rather than fabricating one with bogus weight."""
        obs = SparsePauliOp.from_list([("IZ", 1.0)])
        p = QuEPP(sampling="montecarlo", n_samples=100, seed=42, n_twirls=0)
        with pytest.warns(UserWarning, match="non-diagonal Pauli strings"):
            _, ctx = p.expand(circuit_to_dag(mixed_qc), obs)
        assert ctx["n_paths"] == 0


def _reduce_entry(classical_values, weights) -> _ObservableCPT:
    """A ``per_obs`` entry whose N entries read DAGs 1..N of a one-term observable."""
    n = len(weights)
    return _ObservableCPT(
        weights=np.asarray(weights),
        classical_values=np.asarray(classical_values),
        dag_indices=list(range(1 + n)),
        entry_slots=[0] * n,
        target_slots=[0],
        n_paths=n,
    )


def _flagged_entry(*, eta_rejection=None, eta_amplifying=None) -> _ObservableCPT:
    """A pathless entry carrying only the η diagnostics ``post_reduce`` reads."""
    entry = _reduce_entry(np.array([]), np.array([]))
    entry.eta_rejection = eta_rejection
    entry.eta_amplifying = eta_amplifying
    return entry


def _reduce_ctx(per_obs, *, n_paths, n_rotations=1) -> dict:
    """A minimal QuEPP reduce context wrapping *per_obs*."""
    return {
        "per_obs": per_obs,
        "target_idx": 0,
        "ensemble_start": 1,
        "n_rotations": n_rotations,
        "n_paths": n_paths,
    }


def test_low_eta_triggers_fallback():
    """When noisy/classical ratio falls below min_eta, reduce returns
    the raw target and records the rejection so post_reduce can warn.
    """
    # Non-zero classical values (so "valid" mask has entries) but the
    # ensemble_noisy values are ~0 ⇒ η ≈ 0 < min_eta (0.1) ⇒ fallback.
    per_obs = [_reduce_entry(np.array([1.0, 0.5]), np.array([0.5, 0.5]))]
    ctx = _reduce_ctx(per_obs, n_paths=2)
    p = QuEPP(n_twirls=0)
    result = p.reduce([0.3, 0.0, 0.0], ctx)
    assert result == pytest.approx([0.3])
    assert per_obs[0].eta_rejection == "below_floor"


def test_negative_eta_is_distinguished_from_a_small_one():
    """A sign-inverted noisy ensemble is a different failure than a decayed one.

    Rescaling cannot repair an inverted sign, so it must not be reported as
    merely-weak signal.
    """
    per_obs = [_reduce_entry(np.array([1.0, 0.5]), np.array([0.5, 0.5]))]
    ctx = _reduce_ctx(per_obs, n_paths=2)
    result = QuEPP(n_twirls=0).reduce([0.3, -0.9, -0.45], ctx)
    assert result == pytest.approx([0.3])
    assert per_obs[0].eta_rejection == "negative"


def test_no_classical_signal_is_distinguished_from_a_small_eta():
    """All-negligible classical values leave η undefined, not merely small."""
    per_obs = [_reduce_entry(np.array([0.0, 0.0]), np.array([0.5, 0.5]))]
    ctx = _reduce_ctx(per_obs, n_paths=2)
    result = QuEPP(n_twirls=0).reduce([0.3, 0.2, 0.2], ctx)
    assert result == pytest.approx([0.3])
    assert per_obs[0].eta_rejection == "no_signal"


def test_small_but_accepted_eta_records_amplification():
    """η above the floor still amplifies (T - N); that has to be visible."""
    # eta = median(0.15/1.0, 0.075/0.5) = 0.15 -> 1/eta = 6.7 > 5.
    per_obs = [_reduce_entry(np.array([1.0, 0.5]), np.array([0.5, 0.5]))]
    ctx = _reduce_ctx(per_obs, n_paths=2)
    QuEPP(n_twirls=0).reduce([0.3, 0.15, 0.075], ctx)
    assert per_obs[0].eta_amplifying == pytest.approx(1 / 0.15, rel=1e-9)


class TestQuEPPNoDiagonalPathsWarning:
    """Spec: when path enumeration yields zero diagonal-final paths for an
    observable, ``QuEPP.reduce`` silently returns the noisy target unchanged
    (mitigation is a no-op). The expand-time warning surfaces this so the
    user does not consume noisy results believing they were mitigated.
    """

    @staticmethod
    def _h_then_rz_qc() -> QuantumCircuit:
        # Single non-Clifford rotation (RZ) preceded by an H Clifford. Walking
        # back-to-front, observable Z passes through RZ unchanged (commute),
        # then conjugates through H to land as X — non-diagonal, so every
        # candidate path is rejected by the diagonal-final filter.
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(0.5, 0)
        return qc

    def test_warns_on_zero_diagonal_paths_in_expand(self):
        proto = QuEPP(sampling="exhaustive", truncation_order=1, n_twirls=0)
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        with warnings.catch_warnings():
            # The truncation-ratio warning also fires on this tiny circuit
            # (K/n_rot = 100%); silence it so pytest.warns sees the target.
            warnings.filterwarnings("ignore", message=r"QuEPP:.*shallow circuits")
            with pytest.warns(UserWarning, match=r"zero diagonal Pauli paths"):
                proto.expand(circuit_to_dag(self._h_then_rz_qc()), (obs,))

    def test_warns_on_zero_diagonal_paths_in_dry_expand(self):
        proto = QuEPP(sampling="exhaustive", truncation_order=1, n_twirls=0)
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"QuEPP:.*shallow circuits")
            with pytest.warns(UserWarning, match=r"zero diagonal Pauli paths"):
                proto.dry_expand(circuit_to_dag(self._h_then_rz_qc()), (obs,))

    def test_warns_lists_all_offending_observables_once(self):
        # Two failing observables in one tuple → ONE batched warning that
        # mentions both indices, not two separate warnings.
        proto = QuEPP(sampling="exhaustive", truncation_order=1, n_twirls=0)
        bad1 = SparsePauliOp.from_list([("Z", 1.0)])
        bad2 = SparsePauliOp.from_list([("Z", 0.5)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            proto.expand(circuit_to_dag(self._h_then_rz_qc()), (bad1, bad2))
        zero_path_warnings = [
            w for w in caught if "zero diagonal Pauli paths" in str(w.message)
        ]
        assert len(zero_path_warnings) == 1
        assert "[0, 1]" in str(zero_path_warnings[0].message)

    def test_no_warning_when_paths_exist(self):
        # RX(θ) with observable Z keeps Z as the back-propagated Pauli on the
        # cos branch (diagonal) → at least one path survives → no warning.
        qc = QuantumCircuit(1)
        qc.rx(0.7, 0)
        proto = QuEPP(sampling="exhaustive", truncation_order=1, n_twirls=0)
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            warnings.filterwarnings("ignore", message=r"QuEPP:.*shallow circuits")
            proto.expand(circuit_to_dag(qc), (obs,))

    def test_no_warning_for_clifford_only_circuit(self):
        # n_rotations == 0 guard: a Clifford-only circuit has nothing to
        # mitigate by construction, so the warning is suppressed even when
        # no diagonal paths would survive in principle.
        qc = QuantumCircuit(1)
        qc.h(0)
        proto = QuEPP(sampling="exhaustive", truncation_order=1, n_twirls=0)
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            proto.expand(circuit_to_dag(qc), (obs,))


class TestSymbolicExpand:
    @pytest.mark.usefixtures("suppress_quepp_warnings")
    def test_expand_marks_symbolic(self):
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

    @pytest.mark.usefixtures("suppress_quepp_warnings")
    def test_weights_are_parameter_expressions(self):
        theta = Parameter("theta")
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rx(theta, 0)
        qc.cx(0, 1)
        obs = SparsePauliOp.from_list([("IZ", 1.0)])
        p = QuEPP(sampling="exhaustive", truncation_order=1, n_twirls=0)
        _, ctx = p.expand(circuit_to_dag(qc), obs)
        weights = ctx["per_obs"][0].weights
        assert weights.dtype == object
        for w in weights:
            assert isinstance(w, (ParameterExpression, int, float))

    @pytest.mark.usefixtures("suppress_quepp_warnings")
    def test_montecarlo_falls_back_to_exhaustive(self):
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
        entry = _reduce_entry(
            np.array([1.0, 1.0]), np.array([cos_w, sin_w], dtype=object)
        )
        QuEPP.evaluate_symbolic_weights(entry, [theta], np.array([0.0]))
        assert entry.weights[0] == pytest.approx(1.0)
        assert entry.weights[1] == pytest.approx(0.0)

    def test_rejects_full_context(self):
        theta = Parameter("theta_full_ctx")
        ctx = {
            "per_obs": [_reduce_entry(np.array([1.0]), np.array([theta.cos()]))],
            "symbolic": True,
        }
        with pytest.raises(TypeError, match="per_obs entry"):
            QuEPP.evaluate_symbolic_weights(ctx, [theta], np.array([0.0]))


def _rx_qc(angle: float) -> QuantumCircuit:
    """Single-qubit Rx(angle) circuit."""
    qc = QuantumCircuit(1)
    qc.rx(angle, 0)
    return qc


def _exact_expval(qc: QuantumCircuit, obs: SparsePauliOp) -> float:
    """Exact expectation value via statevector."""
    sv = Statevector.from_instruction(qc)
    return float(np.real(sv.expectation_value(obs)))


def _two_qubit_qc() -> QuantumCircuit:
    """A 2-qubit circuit with several non-commuting rotations."""
    qc = QuantumCircuit(2)
    qc.ry(0.9, 0)
    qc.rz(0.37, 1)
    qc.cx(0, 1)
    qc.rx(0.21, 0)
    return qc


def _cpt_estimate(qc: QuantumCircuit, obs: SparsePauliOp) -> float:
    """The CPT reconstruction ``weights @ classical_values`` for one observable."""
    _, ctx = QuEPP(sampling="exhaustive", truncation_order=5, n_twirls=0).expand(
        circuit_to_dag(qc), obs
    )
    entry = ctx["per_obs"][0]
    return float(np.asarray(entry.weights) @ np.asarray(entry.classical_values))


class TestCPTMultiTermObservables:
    """The CPT sum runs over ``(term, path)`` pairs: each term's weight against that
    term's own Pauli. Collapsing it to paths alone applies every coefficient twice
    and multiplies each term's weight by every other term's expectation.
    """

    @pytest.mark.usefixtures("suppress_quepp_warnings")
    @pytest.mark.parametrize("coeff", [1.0, 2.0, 0.5, -3.0])
    def test_cpt_is_linear_in_the_observable_coefficient(self, coeff):
        """``⟨cP⟩ = c⟨P⟩``. A coefficient folded into both the path weight and the
        Clifford expectation scales the estimate by ``c²``, which unit coefficients
        hide."""
        qc = _two_qubit_qc()
        obs = SparsePauliOp.from_list([("ZI", coeff)])
        assert _cpt_estimate(qc, obs) == pytest.approx(_exact_expval(qc, obs), rel=1e-9)

    @pytest.mark.usefixtures("suppress_quepp_warnings")
    def test_cpt_of_a_sum_is_the_sum_of_the_terms(self):
        """Linearity, with unit coefficients so only the cross terms can break it:
        paths from one term must not be weighted by another term's expectation."""
        qc = _two_qubit_qc()
        terms = [("ZI", 1.0), ("IZ", 1.0)]
        whole = _cpt_estimate(qc, SparsePauliOp.from_list(terms))
        per_term = sum(
            _cpt_estimate(qc, SparsePauliOp.from_list([term])) for term in terms
        )
        assert whole == pytest.approx(per_term, rel=1e-9)
        assert whole == pytest.approx(
            _exact_expval(qc, SparsePauliOp.from_list(terms)), rel=1e-9
        )

    @pytest.mark.usefixtures("suppress_quepp_warnings")
    def test_cpt_recovers_a_realistic_hamiltonian(self):
        """Several terms, none of unit magnitude — the shape of every chemistry and
        QUBO Hamiltonian, and the case no accuracy test covered."""
        qc = _two_qubit_qc()
        obs = SparsePauliOp.from_list(
            [("ZI", 0.7), ("IZ", -0.4), ("ZZ", 0.9), ("XI", 0.25)]
        )
        assert _cpt_estimate(qc, obs) == pytest.approx(_exact_expval(qc, obs), rel=1e-9)


class TestCPTExpansion:
    """Verify that the Heisenberg CPT expansion recovers exact expectation values."""

    @pytest.mark.usefixtures("suppress_quepp_warnings")
    def test_single_rx(self):
        """Rx(θ) with Z observable → cos(θ)."""
        angle = 0.8
        qc = _rx_qc(angle)
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        _, ctx = QuEPP(sampling="exhaustive", truncation_order=5, n_twirls=0).expand(
            circuit_to_dag(qc), obs
        )
        entry = ctx["per_obs"][0]
        cpt = float(entry.weights @ entry.classical_values)
        assert cpt == pytest.approx(np.cos(angle), rel=1e-9)

    @pytest.mark.usefixtures("suppress_quepp_warnings")
    def test_h_rx_h_ry(self):
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
        entry = ctx["per_obs"][0]
        cpt = float(entry.weights @ entry.classical_values)
        assert cpt == pytest.approx(_exact_expval(qc, obs), rel=1e-9)

    @pytest.mark.usefixtures("suppress_quepp_warnings")
    def test_two_qubit_circuit(self, mixed_qc):
        """Two-qubit circuit with ZZ observable."""
        obs = SparsePauliOp.from_list([("ZZ", 1.0)])
        _, ctx = QuEPP(sampling="exhaustive", truncation_order=5, n_twirls=0).expand(
            circuit_to_dag(mixed_qc), obs
        )
        entry = ctx["per_obs"][0]
        cpt = float(entry.weights @ entry.classical_values)
        assert cpt == pytest.approx(_exact_expval(mixed_qc, obs), rel=1e-9)

    @pytest.mark.usefixtures("suppress_quepp_warnings")
    def test_commuting_gate_no_branch(self):
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
        entry = ctx["per_obs"][0]
        assert float(entry.weights @ entry.classical_values) == pytest.approx(0.0)


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


@pytest.mark.usefixtures("suppress_quepp_warnings")
def test_cpt_accuracy_with_normalization():
    """CPT expansion on normalized circuit still recovers exact value."""
    angle = 1.2  # > π/4, so normalization kicks in
    qc = _rx_qc(angle)
    obs = SparsePauliOp.from_list([("Z", 1.0)])
    _, ctx = QuEPP(sampling="exhaustive", truncation_order=5, n_twirls=0).expand(
        circuit_to_dag(qc), obs
    )
    entry = ctx["per_obs"][0]
    cpt = float(entry.weights @ entry.classical_values)
    assert cpt == pytest.approx(np.cos(angle), rel=1e-9)


@pytest.mark.usefixtures("suppress_quepp_warnings")
def test_mc_weights_are_cpt_coefficients():
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
    path_dags = [_build_path_dag(nc_dag, rotation_positions, p.branches) for p in paths]
    cv = _simulate_clifford_ensemble(path_dags, [obs_terms[p.term_idx] for p in paths])
    mc_estimate = float(weights @ cv)
    assert mc_estimate == pytest.approx(np.cos(angle), abs=0.05)


class TestQuEPPRoundTrip:
    @pytest.mark.usefixtures("suppress_quepp_warnings")
    def test_full_round_trip_single_qubit(self):
        """expand → reduce with exact quantum results recovers ideal value."""
        angle = 0.8
        qc = _rx_qc(angle)
        exact = np.cos(angle)
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        protocol = QuEPP(sampling="exhaustive", truncation_order=10, n_twirls=0)
        _, ctx = protocol.expand(circuit_to_dag(qc), obs)
        qr = [exact]
        qr.extend(ctx["per_obs"][0].classical_values)
        assert protocol.reduce(qr, ctx) == pytest.approx([exact], rel=1e-9)

    @pytest.mark.usefixtures("suppress_quepp_warnings")
    def test_noise_correction(self):
        """QuEPP corrects a globally-scaled noise bias."""
        angle = 0.8
        qc = _rx_qc(angle)
        exact = np.cos(angle)
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        protocol = QuEPP(sampling="exhaustive", truncation_order=10, n_twirls=0)
        _, ctx = protocol.expand(circuit_to_dag(qc), obs)
        noise_factor = 0.9
        qr = [exact * noise_factor]
        qr.extend(ctx["per_obs"][0].classical_values * noise_factor)
        assert protocol.reduce(qr, ctx) == pytest.approx([exact], rel=1e-9)

    @pytest.mark.usefixtures("suppress_quepp_warnings")
    def test_expand_with_controlled_rotation(self):
        """Full QuEPP expand works on a circuit with controlled rotations."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cry(0.5, 0, 1)
        qc.ry(0.3, 1)
        obs = SparsePauliOp.from_list([("ZZ", 1.0)])
        protocol = QuEPP(sampling="exhaustive", truncation_order=5, n_twirls=0)
        _, ctx = protocol.expand(circuit_to_dag(qc), obs)
        entry = ctx["per_obs"][0]
        cpt = float(entry.weights @ entry.classical_values)
        assert cpt == pytest.approx(_exact_expval(qc, obs), rel=1e-9)


class TestQuEPPSignalDestructionExtended:
    """Additional signal-destruction detection and post_reduce tests."""

    @staticmethod
    def _make_context(classical_values, weights=None):
        cv = np.array(classical_values)
        w = np.array(weights) if weights is not None else np.ones(len(cv)) / len(cv)
        return _reduce_ctx([_reduce_entry(cv, w)], n_paths=len(cv), n_rotations=len(cv))

    def test_eta_not_rejected_when_valid(self):
        """reduce() does NOT flag when eta is above threshold."""
        ctx = self._make_context([0.5, 0.3])
        # Ensemble noisy close to classical → eta ≈ 1.0
        quantum_results = [0.5, 0.48, 0.29]
        QuEPP(truncation_order=1, n_twirls=0).reduce(quantum_results, ctx)
        assert ctx["per_obs"][0].eta_rejection is None

    def test_near_zero_classical_reports_no_signal_not_destruction(self):
        """All-negligible classical values are undefined η, not decayed signal."""
        ctx = self._make_context([1e-15, 1e-15])
        quantum_results = [0.5, 0.01, 0.01]
        QuEPP(truncation_order=1, n_twirls=0).reduce(quantum_results, ctx)
        assert ctx["per_obs"][0].eta_rejection == "no_signal"

    def test_post_reduce_warns_on_destroyed_signal(self):
        """post_reduce() emits a UserWarning when contexts have destroyed signals."""
        destroyed = {"per_obs": [_flagged_entry(eta_rejection="below_floor")]}
        healthy = {"per_obs": [_flagged_entry()]}
        protocol = QuEPP(truncation_order=1, n_twirls=0)
        with pytest.warns(UserWarning, match=r"signal destroyed"):
            protocol.post_reduce([destroyed, healthy])

    def test_post_reduce_warns_separately_on_no_signal(self):
        """An undefined η reads differently from a decayed one."""
        protocol = QuEPP(truncation_order=1, n_twirls=0)
        with pytest.warns(UserWarning, match=r"no Pauli path with a non-negligible"):
            protocol.post_reduce(
                [{"per_obs": [_flagged_entry(eta_rejection="no_signal")]}]
            )

    def test_post_reduce_warns_separately_on_negative_eta(self):
        """A sign inversion is not a rescaling problem and must say so."""
        protocol = QuEPP(truncation_order=1, n_twirls=0)
        with pytest.warns(UserWarning, match=r"negative η"):
            protocol.post_reduce(
                [{"per_obs": [_flagged_entry(eta_rejection="negative")]}]
            )

    def test_post_reduce_warns_on_noise_amplification(self):
        """A small-but-accepted η amplifies (T - N); post_reduce reports it."""
        protocol = QuEPP(truncation_order=1, n_twirls=0)
        with pytest.warns(UserWarning, match=r"amplify the noisy residual"):
            protocol.post_reduce([{"per_obs": [_flagged_entry(eta_amplifying=8.0)]}])

    def test_post_reduce_silent_when_no_destruction(self):
        """post_reduce() does not warn when all groups are healthy."""
        protocol = QuEPP(truncation_order=1, n_twirls=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            protocol.post_reduce([{"per_obs": [_flagged_entry()]}] * 2)

    def test_post_reduce_default_noop_on_base_class(self):
        """QEMProtocol.post_reduce() is a no-op that does not raise."""
        ctx = {"per_obs": [_flagged_entry(eta_rejection="below_floor")]}
        _NoMitigation().post_reduce([ctx])  # should not raise


class TestComputeEta:
    def test_uses_median_ratio_with_valid_mask(self):
        classical = np.array([1.0, 0.0, -2.0, 4.0])
        noisy = np.array([0.8, 999.0, -1.0, 2.4])
        eta, reason = QuEPP.compute_eta(classical, noisy, min_eta=0.1)
        assert eta == pytest.approx(0.6)
        assert reason is None

    def test_returns_none_when_all_classical_values_are_near_zero(self):
        classical = np.array([0.0, 1e-14, -1e-15])
        noisy = np.array([0.5, 0.2, -0.1])
        assert QuEPP.compute_eta(classical, noisy, min_eta=0.1) == (None, "no_signal")

    def test_returns_none_when_eta_is_below_threshold(self):
        classical = np.array([1.0, -2.0, 4.0])
        noisy = np.array([0.09, -0.18, 0.36])
        assert QuEPP.compute_eta(classical, noisy, min_eta=0.1) == (
            None,
            "below_floor",
        )


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


@pytest.mark.usefixtures("suppress_quepp_warnings")
def test_hybrid_normalization():
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


def test_symbolic_fallback_warnings_carry_their_own_category():
    """Callers that expect symbolic angles need to silence exactly these — matching
    on message text stops suppressing, silently, the moment the wording changes."""
    qc = QuantumCircuit(1)
    qc.rx(Parameter("theta"), 0)
    proto = QuEPP(sampling="montecarlo", coefficient_threshold=0.1, n_twirls=0)
    prep = _PreprocResult(
        working=qc,
        n_qubits=1,
        rotations=_extract_rotation_gates(qc),
        tableaus=_build_clifford_tableaus(qc, _extract_rotation_gates(qc)),
        obs_terms=_obs_to_stim_terms(SparsePauliOp("Z"), 1),
        symbolic=True,
    )
    with pytest.warns(SymbolicAngleWarning) as record:
        proto._select_paths(prep)
    assert len(record) == 2

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.simplefilter("ignore", SymbolicAngleWarning)
        proto._select_paths(prep)


class TestBindBeforeMitigation:
    def test_monte_carlo_sampling_keeps_symbolic_weights(self):
        assert QuEPP().requires_bound_params is False

    def test_exhaustive_sampling_binds_before_mitigation(self):
        assert QuEPP(sampling="exhaustive").requires_bound_params is True


class TestQuEPPPipelineIntegration:
    @pytest.mark.usefixtures("suppress_quepp_warnings")
    def test_pipeline_integration(self, dummy_pipeline_env):
        """QuEPP integrates correctly with QEMStage in a pipeline."""
        meta = _rx_expval_meta(0.5)
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
    @pytest.mark.usefixtures("suppress_quepp_warnings")
    def test_effectiveness_with_readout_noise(self):
        """QuEPP mitigates uniform readout noise on a real backend."""
        meta = _rx_expval_meta(0.8)

        noise = NoiseModel()
        noise.add_all_qubit_readout_error([[0.95, 0.05], [0.05, 0.95]])

        shared = dict(shots=200000, simulation_seed=42, _deterministic_execution=True)

        exact = list(
            CircuitPipeline(stages=[CircuitSpecStage(), MeasurementStage()])
            .run(meta, PipelineEnv(backend=QiskitSimulator(**shared)))
            .values()
        )[0][0]

        noisy = list(
            CircuitPipeline(stages=[CircuitSpecStage(), MeasurementStage()])
            .run(
                meta,
                PipelineEnv(backend=QiskitSimulator(noise_model=noise, **shared)),
            )
            .values()
        )[0][0]

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
                ],
                suppress_performance_warnings=True,
            )
            .run(
                meta,
                PipelineEnv(backend=QiskitSimulator(noise_model=noise, **shared)),
            )
            .values()
        )[0][0]

        noisy_err = abs(noisy - exact)
        quepp_err = abs(quepp_val - exact)
        assert quepp_err < noisy_err / 2, (
            f"QuEPP error ({quepp_err:.4f}) should be less than half "
            f"of noisy error ({noisy_err:.4f})"
        )


class TestQuEPPMultiObservable:
    """QuEPP with a tuple of observables (shared target + deduped paths)."""

    @pytest.fixture
    def qc_two_rotations(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rx(0.3, 0)
        qc.cx(0, 1)
        qc.rz(0.7, 1)
        return qc

    def test_expand_returns_per_obs_entries(self, qc_two_rotations):
        obs1 = SparsePauliOp.from_list([("IZ", 1.0)])
        obs2 = SparsePauliOp.from_list([("ZZ", 1.0)])
        protocol = QuEPP(sampling="exhaustive", truncation_order=2, n_twirls=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, ctx = protocol.expand(circuit_to_dag(qc_two_rotations), (obs1, obs2))
        per_obs = ctx["per_obs"]
        assert isinstance(per_obs, list)
        assert len(per_obs) == 2
        for entry in per_obs:
            assert isinstance(entry, _ObservableCPT)
            # Target shared across all observables, always at merged index 0.
            assert entry.dag_indices[0] == 0

    def test_classical_values_match_independent_runs(self, qc_two_rotations):
        """Multi-observable expand produces the same per-observable
        classical values as N independent single-observable expand calls.
        """
        obs1 = SparsePauliOp.from_list([("IZ", 1.0)])
        obs2 = SparsePauliOp.from_list([("ZZ", 1.0)])
        protocol = QuEPP(sampling="exhaustive", truncation_order=2, n_twirls=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, ctx1 = protocol.expand(circuit_to_dag(qc_two_rotations.copy()), obs1)
            _, ctx2 = protocol.expand(circuit_to_dag(qc_two_rotations.copy()), obs2)
            _, ctx_multi = protocol.expand(
                circuit_to_dag(qc_two_rotations.copy()), (obs1, obs2)
            )
        np.testing.assert_allclose(
            sorted(ctx_multi["per_obs"][0].classical_values),
            sorted(ctx1["per_obs"][0].classical_values),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            sorted(ctx_multi["per_obs"][1].classical_values),
            sorted(ctx2["per_obs"][0].classical_values),
            atol=1e-9,
        )

    def test_target_dag_is_shared_across_observables(self, qc_two_rotations):
        obs1 = SparsePauliOp.from_list([("IZ", 1.0)])
        obs2 = SparsePauliOp.from_list([("ZZ", 1.0)])
        protocol = QuEPP(sampling="exhaustive", truncation_order=2, n_twirls=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dags, ctx = protocol.expand(circuit_to_dag(qc_two_rotations), (obs1, obs2))
        per_obs = ctx["per_obs"]
        target_dag_for_obs1 = dags[per_obs[0].dag_indices[0]]
        target_dag_for_obs2 = dags[per_obs[1].dag_indices[0]]
        assert target_dag_for_obs1 is target_dag_for_obs2

    def test_path_dag_dedup_across_observables(self):
        qc = QuantumCircuit(1)
        qc.rx(0.4, 0)
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        protocol = QuEPP(sampling="exhaustive", truncation_order=1, n_twirls=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dags_solo, _ = protocol.expand(circuit_to_dag(qc.copy()), obs)
            dags_multi, ctx_multi = protocol.expand(
                circuit_to_dag(qc.copy()), (obs, obs)
            )
        per_obs = ctx_multi["per_obs"]
        assert len(dags_multi) == len(dags_solo)
        assert per_obs[0].dag_indices == per_obs[1].dag_indices

    def test_n_identical_observables_share_path_dags(self):
        """N identical observables produce the same number of DAGs as 1
        observable solo — path enumeration is not duplicated."""
        qc = QuantumCircuit(1)
        qc.rx(0.4, 0)
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        protocol = QuEPP(sampling="exhaustive", truncation_order=1, n_twirls=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dags_solo, _ = protocol.expand(circuit_to_dag(qc.copy()), obs)
            dags_multi, _ = protocol.expand(circuit_to_dag(qc.copy()), (obs, obs, obs))
        assert len(dags_multi) == len(dags_solo)

    def test_reduce_returns_list_for_multi_obs_context(self, qc_two_rotations):
        obs1 = SparsePauliOp.from_list([("IZ", 1.0)])
        obs2 = SparsePauliOp.from_list([("ZZ", 1.0)])
        protocol = QuEPP(sampling="exhaustive", truncation_order=2, n_twirls=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dags, ctx = protocol.expand(circuit_to_dag(qc_two_rotations), (obs1, obs2))
        per_dag_per_obs = [[0.5, 0.3] for _ in dags]
        out = protocol.reduce(per_dag_per_obs, ctx)
        assert isinstance(out, list)
        assert len(out) == 2
        assert all(isinstance(v, float) for v in out)

    def test_reduce_per_observable_matches_independent_runs(self, qc_two_rotations):
        obs1 = SparsePauliOp.from_list([("IZ", 1.0)])
        obs2 = SparsePauliOp.from_list([("ZZ", 1.0)])
        protocol = QuEPP(sampling="exhaustive", truncation_order=2, n_twirls=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, ctx1 = protocol.expand(circuit_to_dag(qc_two_rotations.copy()), obs1)
            _, ctx2 = protocol.expand(circuit_to_dag(qc_two_rotations.copy()), obs2)
            dags_multi, ctx_multi = protocol.expand(
                circuit_to_dag(qc_two_rotations.copy()), (obs1, obs2)
            )

        rng = np.random.default_rng(0)
        noisy_solo_1 = np.concatenate(
            [
                [float(rng.uniform(-1, 1))],
                np.array(ctx1["per_obs"][0].classical_values),
            ]
        )
        noisy_solo_2 = np.concatenate(
            [
                [float(rng.uniform(-1, 1))],
                np.array(ctx2["per_obs"][0].classical_values),
            ]
        )
        out1 = protocol.reduce(noisy_solo_1.tolist(), ctx1)
        out2 = protocol.reduce(noisy_solo_2.tolist(), ctx2)

        per_obs = ctx_multi["per_obs"]
        n_dags = len(dags_multi)
        rows = [[float("nan"), float("nan")] for _ in range(n_dags)]
        for slot, d in enumerate(per_obs[0].dag_indices):
            rows[d][0] = noisy_solo_1[slot]
        for slot, d in enumerate(per_obs[1].dag_indices):
            rows[d][1] = noisy_solo_2[slot]

        out_multi = protocol.reduce(rows, ctx_multi)
        assert out_multi[0] == pytest.approx(out1[0], abs=1e-9)
        assert out_multi[1] == pytest.approx(out2[0], abs=1e-9)

    def test_empty_observables_tuple_rejected(self, qc_two_rotations):
        protocol = QuEPP(sampling="exhaustive", truncation_order=2, n_twirls=0)
        with pytest.raises(ValueError, match="at least one observable"):
            protocol.expand(circuit_to_dag(qc_two_rotations), ())

    def test_pipeline_e2e_matches_independent_runs(self, suppress_quepp_warnings):
        """End-to-end pipeline (CircuitSpecStage → QEMStage(QuEPP) →
        MeasurementStage) on a noiseless backend with two QWC observables
        produces the same per-observable mitigated values as running each
        observable through its own pipeline.
        """
        # One circuit read out three ways: both QWC observables together, then
        # each on its own.
        qc = _entangled_two_qubit_circuit()
        multi_meta = meta_from_circuit(qc, observable=(_Z0_2Q, _Z0Z1))
        single_meta_1 = meta_from_circuit(qc, observable=_Z0_2Q)
        single_meta_2 = meta_from_circuit(qc, observable=_Z0Z1)

        backend_kwargs = dict(
            shots=200000, simulation_seed=42, _deterministic_execution=True
        )

        def _run(meta):
            return list(
                CircuitPipeline(
                    stages=[
                        CircuitSpecStage(),
                        QEMStage(
                            protocol=QuEPP(
                                sampling="exhaustive",
                                truncation_order=2,
                                n_twirls=0,
                            )
                        ),
                        MeasurementStage(),
                    ],
                    suppress_performance_warnings=True,
                )
                .run(meta, PipelineEnv(backend=QiskitSimulator(**backend_kwargs)))
                .values()
            )[0]

        multi_out = _run(multi_meta)
        solo_1 = _run(single_meta_1)
        solo_2 = _run(single_meta_2)

        assert isinstance(multi_out, list)
        assert len(multi_out) == 2
        # Tolerances are loose because the noiseless QuEPP path still passes
        # through finite-shot measurement; QWC grouping plus the η rescale
        # produce numbers that agree between modes only up to statistical
        # noise (and tiny numerical drift from the path-DAG dedup).
        assert multi_out[0] == pytest.approx(solo_1[0], abs=5e-3)
        assert multi_out[1] == pytest.approx(solo_2[0], abs=5e-3)

    def test_pipeline_e2e_on_a_multi_term_hamiltonian(self, suppress_quepp_warnings):
        """A Pauli sum has to survive the real measurement stage, not just expand.

        QuEPP declares single-term observables so the noisy side is
        term-resolved; the measurement stage has to be measuring *those* for
        ``reduce`` to find its per-term slots. Driving ``expand``/``reduce``
        directly cannot show that, because it never builds the fan-out the
        stage would.
        """
        hamiltonian = SparsePauliOp.from_list([("IZ", 0.7), ("ZI", -0.4), ("ZZ", 0.9)])
        meta = meta_from_circuit(_entangled_two_qubit_circuit(), observable=hamiltonian)
        (spo,) = meta.observable
        assert len(spo.paulis) > 1, "fixture must be genuinely multi-term"

        out = list(
            CircuitPipeline(
                stages=[
                    CircuitSpecStage(),
                    QEMStage(
                        protocol=QuEPP(
                            sampling="exhaustive", truncation_order=2, n_twirls=0
                        )
                    ),
                    MeasurementStage(),
                ],
                suppress_performance_warnings=True,
            )
            .run(
                meta,
                PipelineEnv(
                    backend=QiskitSimulator(
                        shots=200000, simulation_seed=42, _deterministic_execution=True
                    )
                ),
            )
            .values()
        )[0]

        # One value per *requested* observable, however many terms it holds.
        value = out[0] if isinstance(out, list) else out
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rx(0.3, 0)
        qc.cx(0, 1)
        qc.rz(0.7, 1)
        assert value == pytest.approx(_exact_expval(qc, spo), abs=2e-2)


@pytest.mark.usefixtures("suppress_quepp_warnings")
class TestQuEPPMultiTermUnderNonUniformNoise:
    """The estimator end-to-end on a realistic Hamiltonian, with noise that
    differs per ensemble circuit.

    Uniform damping is the one regime where a coefficient mishandled on both
    the classical and the noisy side cancels itself out, so a test that damps
    every circuit equally cannot see it. These drive each circuit with its own
    factor and compare against a statevector reference.
    """

    @staticmethod
    def _hamiltonian() -> SparsePauliOp:
        """Several terms, mixed signs, none of unit magnitude."""
        return SparsePauliOp.from_list(
            [("ZI", 0.7), ("IZ", -0.4), ("ZZ", 0.9), ("XI", 0.25)]
        )

    @staticmethod
    def _mitigated_under_noise(qc, obs, noise_factors) -> float:
        """Run expand → reduce with per-circuit noise applied to exact values.

        Damps each declared single-term value on each emitted circuit by that
        circuit's own factor, standing in for a backend whose noise varies
        across the ensemble.
        """
        protocol = QuEPP(sampling="exhaustive", truncation_order=5, n_twirls=0)
        dags, ctx = protocol.expand(circuit_to_dag(qc), (obs,))
        declared = ctx["observable_override"]
        rows = []
        for dag_idx, dag in enumerate(dags):
            exact = [
                _exact_expval(dag_to_circuit(dag), term_obs) for term_obs in declared
            ]
            factor = noise_factors[dag_idx % len(noise_factors)]
            rows.append([value * factor for value in exact])
        return protocol.reduce(rows, ctx)[0]

    def test_recovers_exact_energy_when_ensemble_is_noiseless(self):
        """With exact inputs the estimator must return the exact value.

        ``T - N`` vanishes, so this isolates the classical CPT reconstruction
        from the η rescale.
        """
        qc = _two_qubit_qc()
        obs = self._hamiltonian()
        mitigated = self._mitigated_under_noise(qc, obs, [1.0])
        assert mitigated == pytest.approx(_exact_expval(qc, obs), rel=1e-9)

    def test_improves_on_the_unmitigated_value_under_non_uniform_noise(self):
        """Mitigation must move the estimate toward exact, not away from it."""
        qc = _two_qubit_qc()
        obs = self._hamiltonian()
        exact = _exact_expval(qc, obs)
        factors = [0.88, 0.94, 0.9, 0.85, 0.92, 0.87, 0.95]

        mitigated = self._mitigated_under_noise(qc, obs, factors)
        unmitigated = exact * factors[0]

        assert abs(mitigated - exact) < abs(unmitigated - exact)

    def test_is_linear_in_a_hamiltonian_rescaling(self):
        """Scaling H scales the mitigated energy by the same factor.

        A coefficient applied twice makes this quadratic; an absolute
        coefficient threshold makes it non-monotonic.
        """
        qc = _two_qubit_qc()
        obs = self._hamiltonian()
        factors = [0.88, 0.94, 0.9, 0.85]
        base = self._mitigated_under_noise(qc, obs, factors)

        for scale in (0.01, 3.0):
            scaled = self._mitigated_under_noise(qc, scale * obs, factors)
            assert scaled == pytest.approx(scale * base, rel=1e-9)
