# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.circuits.quepp (DAG-native QuEPP implementation)."""

import subprocess
import sys
import warnings

import maestro
import numpy as np
import pytest
import stim
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector

from divi.backends import MaestroConfig
from divi.circuits import MetaCircuit, quepp
from divi.circuits.qem import _NoMitigation
from divi.circuits.quepp import (
    QuEPP,
    SymbolicAngleWarning,
    _all_cos_paths,
    _build_clifford_tableaus,
    _build_path_dag,
    _coerce_angle,
    _decompose_controlled_rotations,
    _enumerate_paths_dfs,
    _extract_rotation_gates,
    _has_symbolic_angles,
    _is_pauli_rotation,
    _merge_paths_by_branch,
    _normalize_angle,
    _normalize_circuit,
    _obs_to_stim_terms,
    _ObservableCPT,
    _PauliPath,
    _PreprocResult,
    _qiskit_clifford_to_stim,
    _sample_paths_montecarlo,
    _simulate_clifford_ensemble,
    _warn_on_term_starvation,
)
from divi.pipeline import CircuitPipeline, PipelineEnv
from divi.pipeline.stages import CircuitSpecStage, MeasurementStage, QEMStage
from tests._helpers import exact_match
from tests.pipeline._helpers import DummySpecStage, meta_from_circuit

_Z0 = SparsePauliOp("Z")
_Z0_2Q = SparsePauliOp.from_list([("IZ", 1.0)])
_Z0Z1 = SparsePauliOp.from_list([("ZZ", 1.0)])


@pytest.fixture
def quepp_backend(make_maestro_simulator):
    """A seeded maestro backend at the shot count the end-to-end tests need."""

    def _make(*, force_sampling: bool = False, **config_kwargs):
        return make_maestro_simulator(
            shots=200000,
            force_sampling=force_sampling,
            maestro_config=MaestroConfig(seed=42, **config_kwargs),
        )

    return _make


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


def _prep(qc: QuantumCircuit, obs: SparsePauliOp):
    """``(rotations, tableaus, obs_terms)`` for ``obs`` back-propagated through ``qc``."""
    rots = _extract_rotation_gates(qc)
    tabs = _build_clifford_tableaus(qc, rots)
    return rots, tabs, _obs_to_stim_terms(obs, qc.num_qubits)


def _two_rx_qc() -> QuantumCircuit:
    """``RX(0.5)`` then ``RX(0.4)`` on one qubit: under ``Z``, paths (0,0) and (1,1)."""
    qc = QuantumCircuit(1)
    qc.rx(0.5, 0)
    qc.rx(0.4, 0)
    return qc


_CONCURRENT_FIRST_TO_NUMPY = """
import threading
import divi.circuits.quepp
import stim
barrier = threading.Barrier(8)
def work():
    barrier.wait()
    stim.PauliString("XZ").to_numpy()
threads = [threading.Thread(target=work) for _ in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()
"""


def test_importing_quepp_makes_concurrent_stim_to_numpy_safe():
    """A fresh interpreter, so no earlier test has already warmed stim."""
    subprocess.run(
        [sys.executable, "-c", _CONCURRENT_FIRST_TO_NUMPY], check=True, timeout=60
    )


def _mixed_rx_qc(angle, angle_first: bool) -> QuantumCircuit:
    """``RX(angle)`` and a fixed non-Clifford ``RX(0.3)`` on one qubit, in either order."""
    qc = QuantumCircuit(1)
    for a in (angle, 0.3) if angle_first else (0.3, angle):
        qc.rx(a, 0)
    return qc


_TWO_RX_PATHS = {
    (0, (0, 0), 0): np.cos(0.5) * np.cos(0.4),
    (0, (1, 1), 2): np.sin(0.5) * np.sin(0.4),
}


def _assert_paths(paths, expected, tol=1e-12):
    """``paths`` are exactly ``expected``'s ``(term_idx, branches, order) → weight``."""
    got = {(p.term_idx, p.branches, p.order): p.weight for p in paths}
    assert len(got) == len(paths)
    assert got == pytest.approx(expected, abs=tol)


def test_coerce_angle_rejects_non_numeric_angles():
    with pytest.raises(
        TypeError, match=exact_match("Unsupported angle type for QuEPP: str")
    ):
        _coerce_angle("0.5")


class TestIsPauliRotation:
    @pytest.mark.parametrize(
        "axis, angle",
        [
            pytest.param("x", 0.5, id="rx"),
            pytest.param("y", 1.2, id="ry"),
            pytest.param("z", -0.7, id="rz"),
        ],
    )
    def test_rotation_detected(self, axis, angle):
        qc = QuantumCircuit(1)
        getattr(qc, f"r{axis}")(angle, 0)
        got_axis, got_angle = _is_pauli_rotation(qc.data[0].operation)
        assert got_axis == axis
        assert got_angle == pytest.approx(angle)

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


@pytest.mark.parametrize(
    "theta, expected_n",
    [
        pytest.param(0.2, 0, id="small_angle_unchanged"),
        pytest.param(np.pi / 2, 1, id="pi_over_2"),
        pytest.param(1.2, 1, id="closer_to_pi_over_2_than_0"),
        pytest.param(-np.pi / 2 - 0.1, -1, id="negative"),
    ],
)
def test_normalise_angle(theta, expected_n):
    """θ = n·(π/2) + θ' with |θ'| ≤ π/4."""
    n, theta_prime = _normalize_angle(theta)
    assert n == expected_n
    assert theta_prime == pytest.approx(theta - n * np.pi / 2, abs=1e-12)
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


@pytest.mark.parametrize(
    "rewrite", [_decompose_controlled_rotations, _normalize_circuit]
)
def test_rewrites_keep_classical_bits(rewrite):
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    out = rewrite(qc)
    assert out.num_clbits == 1
    assert [(i.operation.name, len(i.clbits)) for i in out.data] == [
        ("h", 0),
        ("measure", 1),
    ]


@pytest.mark.usefixtures("suppress_quepp_warnings")
def test_path_dags_keep_the_targets_classical_register():
    qc = QuantumCircuit(1, 1)
    qc.rx(0.3, 0)
    dags, _ = QuEPP(sampling="exhaustive", n_twirls=0).expand(circuit_to_dag(qc), _Z0)
    assert [d.num_clbits() for d in dags] == [1, 1]


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


def test_one_tableau_per_clifford_layer():
    rots, tabs, _ = _prep(_two_rx_qc(), _Z0)
    assert len(tabs) == len(rots) + 1 == 3


class TestQiskitCliffordToStim:
    def test_basic_cliffords(self, bell_qc):
        sc = _qiskit_clifford_to_stim(bell_qc)
        assert sc.num_qubits == 2
        # Tableau builds successfully (no exception).
        tab = stim.Tableau.from_circuit(sc)
        assert len(tab) == 2

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

    @pytest.mark.parametrize(
        "gates,expected",
        [
            ([("rx", np.pi / 2)], "I 0\nSQRT_X 0"),
            ([("rz", np.pi / 2)], "I 0\nS 0"),
            ([("rx", -np.pi / 2), ("ry", 2 * np.pi)], "I 0\nSQRT_X_DAG 0"),
            ([], "I 0"),
        ],
        ids=[
            "rx-quarter-turn",
            "rz-quarter-turn",
            "negative-and-full-turn",
            "idle-qubit-padded",
        ],
    )
    def test_exact_conversion(self, gates, expected):
        qc = QuantumCircuit(1)
        for name, angle in gates:
            getattr(qc, name)(angle, 0)
        assert str(_qiskit_clifford_to_stim(qc)) == expected

    def test_empty_register_converts_to_an_empty_circuit(self):
        assert _qiskit_clifford_to_stim(QuantumCircuit(0)) == stim.Circuit()

    def test_unrecognised_gate_raises(self):
        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        with pytest.raises(
            ValueError,
            match=exact_match(
                "Gate 'measure' is not recognised as Clifford by QuEPP's stim "
                "converter."
            ),
        ):
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

    def test_pads_a_narrower_observable_to_the_circuit_width(self):
        obs = SparsePauliOp.from_list([("XZ", 0.5), ("ZI", -2.0)])
        terms = _obs_to_stim_terms(obs, 3)
        assert [(c, str(ps)) for c, ps in terms] == [(0.5, "+ZX_"), (-2.0, "+_Z_")]


class TestEnumeratePathsDFS:
    def test_no_rotations_single_identity_path(self, bell_qc):
        obs = SparsePauliOp.from_list([("ZZ", 1.0)])
        paths = _enumerate_paths_dfs(*_prep(bell_qc, obs), max_order=2)
        assert len(paths) == 1
        assert paths[0].branches == ()
        assert paths[0].weight == pytest.approx(1.0)
        assert paths[0].order == 0

    @pytest.mark.parametrize(
        "max_order,expected",
        [
            (2, _TWO_RX_PATHS),
            (1, {k: w for k, w in _TWO_RX_PATHS.items() if k[2] <= 1}),
        ],
    )
    def test_truncation_order_caps_the_sine_branches(self, max_order, expected):
        paths = _enumerate_paths_dfs(*_prep(_two_rx_qc(), _Z0), max_order=max_order)
        _assert_paths(paths, expected)

    @pytest.mark.parametrize(
        "obs,threshold,expected",
        [
            (_Z0, float(np.cos(0.5)), {(0, (0,), 0): np.cos(0.5)}),
            (SparsePauliOp("Y"), float(np.sin(0.5)), {(0, (1,), 1): np.sin(0.5)}),
            (_Z0, 0.99, {}),
        ],
        ids=["cos-at-threshold", "sin-at-threshold", "both-below"],
    )
    def test_coefficient_threshold_keeps_weights_equal_to_it(
        self, obs, threshold, expected
    ):
        paths = _enumerate_paths_dfs(
            *_prep(_rx_qc(0.5), obs), max_order=2, coefficient_threshold=threshold
        )
        _assert_paths(paths, expected)

    @pytest.mark.parametrize(
        "obs,branches,trig", [(_Z0, (0,), np.cos), (SparsePauliOp("Y"), (1,), np.sin)]
    )
    def test_symbolic_weights_are_the_trigonometric_factor(self, obs, branches, trig):
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        (path,) = _enumerate_paths_dfs(*_prep(qc, obs), max_order=2)
        assert path.branches == branches
        assert float(path.weight.bind({theta: 0.3})) == pytest.approx(trig(0.3))


@pytest.mark.parametrize(
    "select",
    [
        lambda terms: _enumerate_paths_dfs([], [stim.Tableau(1)], terms, max_order=2),
        lambda terms: _sample_paths_montecarlo(
            [], [stim.Tableau(1)], terms, 10, np.random.default_rng(0)
        ),
    ],
    ids=["exhaustive", "montecarlo"],
)
def test_clifford_only_circuit_yields_one_unit_path_per_term(select):
    terms = _obs_to_stim_terms(SparsePauliOp.from_list([("Z", 1.0), ("X", 0.5)]), 1)
    _assert_paths(select(terms), {(0, (), 0): 1.0, (1, (), 0): 1.0})


def test_merge_sums_weights_per_term_and_branches():
    merged = _merge_paths_by_branch(
        [
            _PauliPath((1, 0), 0.25, 1, term_idx=1),
            _PauliPath((1, 0), 0.5, 1, term_idx=1),
            _PauliPath((1, 0), 0.125, 1, term_idx=0),
        ]
    )
    assert set(merged) == {
        _PauliPath((1, 0), 0.75, 1, term_idx=1),
        _PauliPath((1, 0), 0.125, 1, term_idx=0),
    }


class TestPathDagConstruction:
    def test_rotation_indices_align_with_working_dag_topology(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rx(0.2, 0)
        qc.cx(0, 1)
        qc.rz(-0.4, 1)

        obs = SparsePauliOp.from_list([("ZZ", 1.0)])
        (prep,) = QuEPP._preprocess(circuit_to_dag(qc), (obs,))
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


def _all_cos_of(qc: QuantumCircuit, obs: SparsePauliOp):
    """``_all_cos_paths`` for ``obs`` back-propagated through ``qc``."""
    rots, tabs, obs_terms = _prep(qc, obs)
    return _all_cos_paths(rots, [t.inverse() for t in tabs], obs_terms)


class TestAllCosPaths:
    """Spec: ``_all_cos_paths`` returns the branches=(0,)*K path per observable term."""

    def test_matches_exhaustive_dfs_all_zero_branch(self):
        """Deterministic fallback weight matches the DFS-enumerated all-zero path."""
        angle = 0.7
        rots, tabs, obs_terms = _prep(_normalize_circuit(_rx_qc(angle)), _Z0)
        inv_tabs = [t.inverse() for t in tabs]

        fallback_w = _all_cos_weight_of(rots, inv_tabs, obs_terms)

        # Cross-check with the exhaustive enumeration's all-zero branch.
        dfs_paths = _enumerate_paths_dfs(rots, tabs, obs_terms, max_order=10)
        zero_branch = next(p for p in dfs_paths if all(b == 0 for b in p.branches))
        assert fallback_w == pytest.approx(zero_branch.weight, rel=1e-12)
        assert fallback_w == pytest.approx(np.cos(angle), abs=1e-12)

    def test_emits_no_path_when_no_term_diagonal(self):
        """Observable that never propagates to a diagonal Pauli yields no path."""
        # X commutes with Rx so no cos factor accumulates, but the final
        # Pauli is still X — non-diagonal — so the term contributes no path.
        assert _all_cos_of(_normalize_circuit(_rx_qc(0.4)), SparsePauliOp("X")) == []

    def test_weights_are_coefficient_free_and_per_term(self):
        """Only the Z term propagates diagonally, and its weight excludes its coeff.

        The coefficient is applied later, to the measured value the weight
        pairs with; folding it in here would double-count it.
        """
        angle = 0.5
        obs = SparsePauliOp.from_list([("Z", 0.8), ("X", 0.2)])  # X term → no path

        paths = _all_cos_of(_normalize_circuit(_rx_qc(angle)), obs)

        assert [p.term_idx for p in paths] == [0]
        assert paths[0].weight == pytest.approx(np.cos(angle), abs=1e-12)

    def test_commuting_rotations_contribute_no_cos_factor(self):
        qc = QuantumCircuit(2)
        qc.rz(0.2, 0)
        qc.rx(0.3, 0)
        qc.rx(0.4, 0)
        obs = SparsePauliOp.from_list([("IZ", 1.0), ("ZI", 0.5)])
        _assert_paths(
            _all_cos_of(qc, obs),
            {(0, (0, 0, 0), 0): np.cos(0.3) * np.cos(0.4), (1, (0, 0, 0), 0): 1.0},
        )

    def test_mc_fallback_returns_the_all_cos_path(self):
        # default_rng(0)'s single draw takes the non-diagonal sin branch.
        with pytest.warns(
            UserWarning,
            match=exact_match(
                "QuEPP Monte Carlo: all 1 samples produced non-diagonal Pauli "
                "strings.  Falling back to the deterministic all-cos path of 1 "
                "observable term(s).  Consider increasing n_samples or using "
                "exhaustive enumeration."
            ),
        ):
            paths = _sample_paths_montecarlo(
                *_prep(_rx_qc(0.7), _Z0), 1, np.random.default_rng(0)
            )
        _assert_paths(paths, {(0, (0,), 0): np.cos(0.7)})


def _rz_on_qubit_0_qc() -> QuantumCircuit:
    """``RZ(0.3)`` on qubit 0 of two: commutes with every Z-type term."""
    qc = QuantumCircuit(2)
    qc.rz(0.3, 0)
    return qc


@pytest.mark.parametrize(
    "qc,obs,n_samples,seed,expected,tol",
    [
        (
            _rz_on_qubit_0_qc(),
            SparsePauliOp.from_list([("IZ", 0.9), ("ZI", 0.1)]),
            8000,
            0,
            {(0, (0,), 0): 1.0, (1, (0,), 0): 1.0},
            0.15,
        ),
        (_two_rx_qc(), _Z0, 20000, 1, _TWO_RX_PATHS, 0.02),
    ],
    ids=["terms-drawn-by-coefficient", "weights-multiply-across-rotations"],
)
def test_mc_weights_converge_to_the_cpt_weights(
    qc, obs, n_samples, seed, expected, tol
):
    paths = _sample_paths_montecarlo(
        *_prep(qc, obs), n_samples, np.random.default_rng(seed)
    )
    _assert_paths(paths, expected, tol=tol)


def test_mc_with_all_zero_coefficients_returns_no_paths():
    obs = SparsePauliOp.from_list([("Z", 0.0)])
    with pytest.warns(
        UserWarning,
        match=exact_match(
            "QuEPP Monte Carlo: every coefficient of a 1-term observable is zero, "
            "so it has no paths to sample and mitigation is a no-op. Check whether "
            "the observable was built as intended (a coefficient that optimises to "
            "zero reaches this too)."
        ),
    ):
        paths = _sample_paths_montecarlo(
            *_prep(_two_rx_qc(), obs), 10, np.random.default_rng(0)
        )
    assert paths == []


_STARVATION_AT_200_SAMPLES = (
    "QuEPP Monte Carlo: with n_samples=200 across 2 Pauli terms, the "
    "smallest-coefficient term expects only 2.0 samples (terms are "
    "drawn in proportion to |coefficient|). Its contribution will be "
    "poorly estimated, and it may draw none at all. Raise n_samples "
    "to at least 2000 or use sampling='exhaustive'."
)


class TestTermStarvationWarning:
    def test_warns_below_twenty_expected_samples(self):
        with pytest.warns(UserWarning, match=exact_match(_STARVATION_AT_200_SAMPLES)):
            _warn_on_term_starvation(np.array([0.99, 0.01]), 200)

    def test_silent_at_twenty_expected_samples(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_on_term_starvation(np.array([0.5, 0.5]), 40)


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
    @pytest.mark.parametrize("pauli", ["ZZ", "XX"])
    def test_bell_state_stabiliser(self, bell_qc, pauli):
        obs = SparsePauliOp.from_list([(pauli, 1.0)])
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

    def test_simulates_each_distinct_circuit_once(self, bell_qc, mocker):
        flipped = QuantumCircuit(2)
        flipped.x(0)
        dag_a, dag_b = circuit_to_dag(bell_qc), circuit_to_dag(flipped)
        spy = mocker.spy(quepp, "_qiskit_clifford_to_stim")
        terms = _obs_to_stim_terms(
            SparsePauliOp.from_list([("ZZ", 0.5), ("ZZ", -2.0), ("ZZ", 3.0)]), 2
        )
        vals = _simulate_clifford_ensemble([dag_a, dag_a, dag_b], terms)
        assert spy.call_count == 2
        assert vals == pytest.approx([0.5, -2.0, -3.0])


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
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _, ctx = p.expand(circuit_to_dag(mixed_qc), obs)
        assert not any("non-diagonal Pauli strings" in str(w.message) for w in record)
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


@pytest.mark.parametrize(
    "ensemble_noisy", [[0.3, 0.15], [0.2, 0.1]], ids=["eta_0.3", "eta_0.2_at_limit"]
)
def test_eta_within_the_amplification_limit_is_not_flagged(ensemble_noisy):
    """``1/η`` of at most 5 is reported as no amplification."""
    per_obs = [_reduce_entry([1.0, 0.5], [0.5, 0.5])]
    QuEPP(n_twirls=0).reduce([0.3, *ensemble_noisy], _reduce_ctx(per_obs, n_paths=2))
    assert per_obs[0].eta_amplifying is None


def test_reduce_rescales_the_noisy_residual_by_eta():
    """η = 0.5: classical 0.75 + (T 0.3 - N 0.375) / 0.5."""
    ctx = _reduce_ctx([_reduce_entry([1.0, 0.5], [0.5, 0.5])], n_paths=2)
    assert QuEPP(n_twirls=0).reduce([0.3, 0.5, 0.25], ctx) == pytest.approx([0.6])


def test_reduce_without_rotations_returns_each_exact_value():
    per_obs = [_reduce_entry([1.0], [1.0]), _reduce_entry([1.0], [1.0])]
    ctx = _reduce_ctx(per_obs, n_paths=1, n_rotations=0)
    assert QuEPP(n_twirls=0).reduce([0.5, 0.8], ctx) == pytest.approx([1.0, 1.0])


@pytest.mark.usefixtures("suppress_quepp_warnings")
@pytest.mark.parametrize(
    "method, angle, error, message",
    [
        (
            "expand",
            Parameter("theta"),
            ValueError,
            "QuEPP weights are still symbolic — parameter values were never "
            "substituted. Add ParameterBindingStage to the pipeline or use "
            "QuEPP(sampling='exhaustive') to bind parameters before mitigation.",
        ),
        (
            "dry_expand",
            0.4,
            RuntimeError,
            "QuEPP.reduce: context has no per_obs entries (was this a dry-run "
            "context?).",
        ),
    ],
    ids=["symbolic", "dry_run"],
)
def test_reduce_refuses_an_unevaluable_context(method, angle, error, message):
    protocol = QuEPP(n_twirls=0)
    _, ctx = getattr(protocol, method)(circuit_to_dag(_rx_qc(angle)), _Z0)
    with pytest.raises(error, match=exact_match(message)):
        protocol.reduce([1.0] * (1 + ctx["n_paths"]), ctx)


@pytest.mark.parametrize(
    "n_classical, n_slots, dag_indices, message",
    [
        (
            1,
            2,
            [0, 1, 2],
            "_ObservableCPT: weights (2), classical_values (1) and entry_slots "
            "(2) must run in parallel over (term, path) pairs.",
        ),
        (
            2,
            1,
            [0, 1, 2],
            "_ObservableCPT: weights (2), classical_values (2) and entry_slots "
            "(1) must run in parallel over (term, path) pairs.",
        ),
        (
            2,
            2,
            [0, 1],
            "_ObservableCPT: dag_indices has 2 entries; expected 3 (the target "
            "slot plus one per (term, path) pair).",
        ),
    ],
    ids=["classical_values", "entry_slots", "dag_indices"],
)
def test_observable_cpt_rejects_misaligned_fields(
    n_classical, n_slots, dag_indices, message
):
    with pytest.raises(ValueError, match=exact_match(message)):
        _ObservableCPT(
            weights=np.zeros(2),
            classical_values=np.zeros(n_classical),
            dag_indices=dag_indices,
            entry_slots=[0] * n_slots,
            target_slots=[0],
        )


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

    @pytest.mark.parametrize("method", ["expand", "dry_expand"])
    def test_warning_text(self, method):
        proto = QuEPP(sampling="exhaustive", truncation_order=1, n_twirls=0)
        with pytest.warns(UserWarning) as record:
            getattr(proto, method)(circuit_to_dag(self._h_then_rz_qc()), (_Z0,))
        assert (
            "QuEPP: observable(s) at index/indices [0] produced zero diagonal "
            "Pauli paths (truncation_order=1, 1 non-Clifford rotation(s)). The "
            "Heisenberg back-propagation terminates in a non-diagonal basis, so "
            "mitigation will be a no-op for these observables and the raw noisy "
            "expectation will be returned. Consider rebasing the observable or "
            "restructuring the circuit's final Clifford layer."
        ) in [str(w.message) for w in record]

    def test_warns_lists_all_offending_observables_once(self):
        # Two failing observables in one tuple → ONE batched warning that
        # mentions both indices, not two separate warnings.
        proto = QuEPP(sampling="exhaustive", truncation_order=1, n_twirls=0)
        bad1 = SparsePauliOp("Z")
        bad2 = 0.5 * SparsePauliOp("Z")
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
        obs = SparsePauliOp("Z")
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
        obs = SparsePauliOp("Z")
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
        obs = SparsePauliOp("Z")
        p = QuEPP(sampling="montecarlo", n_samples=10, truncation_order=1, n_twirls=0)
        with pytest.warns(UserWarning, match="Monte Carlo"):
            _, ctx = p.expand(circuit_to_dag(qc), obs)
        assert ctx.get("symbolic") is True

    @pytest.mark.usefixtures("suppress_quepp_warnings")
    def test_dry_expand_marks_symbolic(self):
        _, ctx = QuEPP(n_twirls=0).dry_expand(
            circuit_to_dag(_rx_qc(Parameter("theta"))), _Z0
        )
        assert ctx.get("symbolic") is True

    @pytest.mark.usefixtures("suppress_quepp_warnings")
    @pytest.mark.parametrize(
        "symbolic_first", [True, False], ids=["symbolic-first", "concrete-first"]
    )
    def test_coefficient_threshold_is_disabled(self, symbolic_first):
        qc = _mixed_rx_qc(Parameter("theta"), symbolic_first)
        n_paths = [
            QuEPP(
                truncation_order=2, coefficient_threshold=threshold, n_twirls=0
            ).expand(circuit_to_dag(qc), _Z0)[1]["n_paths"]
            for threshold in (None, 0.5)
        ]
        assert n_paths == [2, 2]

    @pytest.mark.usefixtures("suppress_quepp_warnings")
    @pytest.mark.parametrize(
        "symbolic_first", [True, False], ids=["symbolic-first", "concrete-first"]
    )
    def test_bound_symbolic_weights_match_the_bound_circuit(self, symbolic_first):
        theta = Parameter("theta")
        quepp = QuEPP(sampling="exhaustive", truncation_order=2, n_twirls=0)
        _, symbolic_ctx = quepp.expand(
            circuit_to_dag(_mixed_rx_qc(theta, symbolic_first)), _Z0
        )
        (entry,) = symbolic_ctx["per_obs"]
        QuEPP.evaluate_symbolic_weights(entry, symbolic_ctx["weight_symbols"], [0.5])
        _, bound_ctx = quepp.expand(
            circuit_to_dag(_mixed_rx_qc(0.5, symbolic_first)), _Z0
        )
        np.testing.assert_allclose(
            sorted(entry.weights), sorted(bound_ctx["per_obs"][0].weights)
        )


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

    def test_passes_concrete_weights_through(self):
        theta = Parameter("theta_mixed")
        entry = _reduce_entry(
            np.array([1.0, 1.0]), np.array([0.5, theta.cos()], dtype=object)
        )
        QuEPP.evaluate_symbolic_weights(entry, [theta], np.array([0.0]))
        np.testing.assert_allclose(entry.weights, [0.5, 1.0])

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
    @pytest.mark.parametrize(
        "angle",
        [
            pytest.param(0.8, id="below_pi_over_4"),
            pytest.param(1.2, id="above_pi_over_4_normalised"),
        ],
    )
    def test_single_rx(self, angle):
        """Rx(θ) with Z observable → cos(θ), including when normalisation kicks in."""
        qc = _rx_qc(angle)
        obs = SparsePauliOp("Z")
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
        obs = SparsePauliOp("Z")
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
        obs = SparsePauliOp("X")
        _, ctx = QuEPP(sampling="exhaustive", truncation_order=5, n_twirls=0).expand(
            circuit_to_dag(qc), obs
        )
        assert ctx["n_paths"] == 0
        entry = ctx["per_obs"][0]
        assert float(entry.weights @ entry.classical_values) == pytest.approx(0.0)


class TestDecomposeControlledRotationsExtended:
    """Additional controlled-rotation decomposition tests."""

    def test_clifford_cry_produces_no_rotations(self):
        """CRY(π) is Clifford — after decomposition and normalisation, no rotations."""
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


class TestQuEPPRoundTrip:
    @pytest.mark.usefixtures("suppress_quepp_warnings")
    @pytest.mark.parametrize(
        "noise_factor",
        [
            pytest.param(1.0, id="exact_results"),
            pytest.param(0.9, id="global_noise_bias"),
        ],
    )
    def test_full_round_trip_single_qubit(self, noise_factor):
        """expand → reduce recovers the ideal value from exact results and corrects
        a globally-scaled noise bias."""
        angle = 0.8
        qc = _rx_qc(angle)
        exact = np.cos(angle)
        obs = SparsePauliOp("Z")
        protocol = QuEPP(sampling="exhaustive", truncation_order=10, n_twirls=0)
        _, ctx = protocol.expand(circuit_to_dag(qc), obs)
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

    def test_post_reduce_warns_on_noise_amplification(self):
        """A small-but-accepted η amplifies (T - N); post_reduce reports it."""
        protocol = QuEPP(truncation_order=1, n_twirls=0)
        with pytest.warns(UserWarning, match=r"amplify the noisy residual"):
            protocol.post_reduce([{"per_obs": [_flagged_entry(eta_amplifying=8.0)]}])

    @pytest.mark.parametrize(
        "rejection, message",
        [
            (
                "no_signal",
                "QuEPP: an observable had no Pauli path with a non-negligible "
                "classical expectation value, so η is undefined and the raw noisy "
                "value was returned unmitigated. Check that the observable's "
                "coefficients are not all negligible, and that the circuit's final "
                "Clifford layer leaves the back-propagated Pauli diagonal. If you "
                "also saw the zero-diagonal-paths warning, this is that same cause "
                "surfacing at reduction time.",
            ),
            (
                "below_floor",
                "QuEPP: signal destroyed — η fell below the safety threshold and "
                "mitigation fell back to the raw noisy value. Consider increasing "
                "shots or reducing noise.",
            ),
            (
                "negative",
                "QuEPP: an observable produced a negative η — the noisy Clifford "
                "ensemble came back with the opposite sign to the exact one, which "
                "rescaling cannot repair, so the raw noisy value was returned. "
                "Raise the shot count first: a sign flip on a weak signal is often "
                "statistical. If it persists, the noise is past this protocol's "
                "usable range — use ZNE instead.",
            ),
        ],
        ids=["no_signal", "below_floor", "negative"],
    )
    def test_post_reduce_rejection_warning_text(self, rejection, message):
        with pytest.warns(UserWarning) as record:
            QuEPP(n_twirls=0).post_reduce(
                [
                    {"per_obs": [_flagged_entry(eta_rejection=rejection)]},
                    {"per_obs": [_flagged_entry()]},
                ]
            )
        assert [str(w.message) for w in record] == [message]

    def test_post_reduce_silent_when_no_destruction(self):
        """post_reduce() does not warn when all groups are healthy."""
        protocol = QuEPP(truncation_order=1, n_twirls=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            protocol.post_reduce([{"per_obs": [_flagged_entry()]}] * 2)

    def test_post_reduce_default_noop_on_base_class(self):
        """QEMProtocol.post_reduce() is a no-op that does not raise."""
        ctx = {"per_obs": [_flagged_entry(eta_rejection="below_floor")]}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _NoMitigation().post_reduce([ctx])


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

    @pytest.mark.parametrize(
        "classical, noisy, expected",
        [(1e-12, 1e-12, (None, "no_signal")), (1.0, 0.1, (None, "below_floor"))],
        ids=["classical_at_cutoff", "eta_at_floor"],
    )
    def test_boundaries_are_rejected(self, classical, noisy, expected):
        result = QuEPP.compute_eta(np.array([classical]), np.array([noisy]), 0.1)
        assert result == expected


def _shallow_circuit_warning(k: int, n_rotations: int, ratio: str) -> str:
    return (
        f"QuEPP: truncation order K={k} replaces a large fraction of the "
        f"{n_rotations} non-Clifford rotations ({ratio}). Mitigation quality "
        f"may degrade on shallow circuits — consider reducing "
        f"truncation_order or using a deeper circuit."
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

    @pytest.mark.parametrize("n_rotations, ratio", [(1, "100%"), (2, "50%")])
    def test_warning_text(self, n_rotations, ratio):
        with pytest.warns(UserWarning) as record:
            QuEPP(truncation_order=1)._warn_on_truncation_ratio(n_rotations)
        assert [str(w.message) for w in record] == [
            _shallow_circuit_warning(1, n_rotations, ratio)
        ]

    def test_ratio_at_the_limit_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            QuEPP(truncation_order=33)._warn_on_truncation_ratio(100)


@pytest.mark.usefixtures("suppress_quepp_warnings")
def test_hybrid_normalisation():
    """Concrete rotations are normalised; symbolic ones are kept as-is."""
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    # Rx(π/2) is concrete Clifford → normalised away; Rx(theta) is symbolic → kept
    qc.rx(np.pi / 2, 0)
    qc.rx(theta, 0)
    obs = SparsePauliOp("Z")
    _, ctx = QuEPP(sampling="exhaustive", truncation_order=1, n_twirls=0).expand(
        circuit_to_dag(qc), obs
    )
    # Only the symbolic rotation should remain
    assert ctx["n_rotations"] == 1


@pytest.mark.usefixtures("suppress_quepp_warnings")
def test_barriers_do_not_change_the_expansion():
    plain, fenced = QuantumCircuit(1), QuantumCircuit(1)
    for qc in (plain, fenced):
        qc.rx(0.3, 0)
        if qc is fenced:
            qc.barrier()
        qc.h(0)
    quepp = QuEPP(sampling="exhaustive", n_twirls=0)
    (expected,), (actual,) = (
        quepp.expand(circuit_to_dag(qc), SparsePauliOp("X"))[1]["per_obs"]
        for qc in (plain, fenced)
    )
    np.testing.assert_allclose(actual.weights, expected.weights)
    np.testing.assert_allclose(actual.classical_values, expected.classical_values)


def _single_rx_prep(angle) -> _PreprocResult:
    """Preprocessed ``RX(angle)`` measured as ``<Z0>``, symbolic iff ``angle`` is."""
    qc = QuantumCircuit(1)
    qc.rx(angle, 0)
    rotations, tableaus, obs_terms = _prep(qc, _Z0)
    return _PreprocResult(
        working=qc,
        rotations=rotations,
        tableaus=tableaus,
        obs_terms=obs_terms,
        symbolic=isinstance(angle, Parameter),
    )


def test_symbolic_fallback_warnings_carry_their_own_category():
    """Callers that expect symbolic angles need to silence exactly these — matching
    on message text stops suppressing, silently, the moment the wording changes."""
    proto = QuEPP(sampling="montecarlo", coefficient_threshold=0.1, n_twirls=0)
    prep = _single_rx_prep(Parameter("theta"))
    with pytest.warns(SymbolicAngleWarning) as record:
        proto._select_paths(prep)
    assert [str(w.message) for w in record] == [
        "QuEPP: Monte Carlo sampling requires concrete angles. Falling back to "
        "exhaustive enumeration for symbolic circuit.",
        "QuEPP: coefficient_threshold pruning disabled for symbolic circuit "
        "(angle magnitudes unknown).",
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.simplefilter("ignore", SymbolicAngleWarning)
        proto._select_paths(prep)


@pytest.mark.parametrize(
    "sampling,requires_bound_params",
    [("auto", False), ("montecarlo", False), ("exhaustive", True)],
)
def test_requires_bound_params_by_sampling(sampling, requires_bound_params):
    assert QuEPP(sampling=sampling).requires_bound_params is requires_bound_params


def test_default_options():
    assert QuEPP().n_twirls == 10

    with pytest.warns(UserWarning) as record:
        QuEPP(sampling="exhaustive", n_twirls=0).expand(
            circuit_to_dag(_two_rx_qc()), _Z0
        )
    assert _shallow_circuit_warning(2, 2, "100%") in [str(w.message) for w in record]

    starved = SparsePauliOp.from_list([("ZZ", 0.99), ("ZI", 0.01)])
    with pytest.warns(UserWarning) as record:
        QuEPP(sampling="montecarlo", n_twirls=0).dry_expand(
            circuit_to_dag(_two_qubit_qc()), starved
        )
    assert _STARVATION_AT_200_SAMPLES in [str(w.message) for w in record]


@pytest.mark.parametrize(
    "kwargs", [dict(sampling="exhaustive", n_samples=0), dict(n_samples=1)]
)
def test_accepts_valid_sample_counts(kwargs):
    QuEPP(**kwargs)


_N_SAMPLES_ERROR = "n_samples must be a positive integer for montecarlo sampling."


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(sampling="montecarlo", n_samples=0), _N_SAMPLES_ERROR),
        (dict(sampling="montecarlo", n_samples=None), _N_SAMPLES_ERROR),
        (dict(truncation_order=-1), "truncation_order must be non-negative."),
        (
            dict(sampling="sobol"),
            "sampling must be 'auto', 'exhaustive' or 'montecarlo', got 'sobol'",
        ),
    ],
    ids=["zero_samples", "no_samples", "negative_order", "unknown_sampling"],
)
def test_rejects_invalid_options(kwargs, message):
    with pytest.raises(ValueError, match=exact_match(message)):
        QuEPP(**kwargs)


def _montecarlo_weights(seed: int) -> np.ndarray:
    """Path weights of a seeded Monte Carlo expansion of :func:`_two_qubit_qc`."""
    protocol = QuEPP(sampling="montecarlo", n_samples=30, seed=seed, n_twirls=0)
    _, ctx = protocol.expand(circuit_to_dag(_two_qubit_qc()), _Z0Z1)
    return ctx["per_obs"][0].weights


@pytest.mark.usefixtures("suppress_quepp_warnings")
def test_seed_makes_montecarlo_expansion_reproducible():
    np.testing.assert_array_equal(_montecarlo_weights(3), _montecarlo_weights(3))
    assert not np.array_equal(_montecarlo_weights(3), _montecarlo_weights(4))


def _layered_two_qubit_qc() -> QuantumCircuit:
    """Eight layers of non-commuting rotations, so sampled path counts vary."""
    qc = QuantumCircuit(2)
    for i in range(8):
        qc.ry(0.5 + 0.1 * i, 0)
        qc.rx(0.6 + 0.07 * i, 1)
        qc.cx(0, 1)
    return qc


@pytest.mark.usefixtures("suppress_quepp_warnings")
def test_dry_expand_preview_path_count_is_deterministic():
    dag = circuit_to_dag(_layered_two_qubit_qc())
    counts = {
        QuEPP(sampling="montecarlo", n_samples=20, n_twirls=0).dry_expand(dag, _Z0Z1)[
            1
        ]["n_paths"]
        for _ in range(10)
    }
    assert len(counts) == 1


def test_dry_expand_preview_keeps_montecarlo_warnings():
    obs = SparsePauliOp.from_list([("ZZ", 1.0), ("ZI", 0.001)])
    protocol = QuEPP(sampling="montecarlo", n_samples=50, seed=1, n_twirls=0)
    with pytest.warns(UserWarning) as record:
        protocol.dry_expand(circuit_to_dag(_two_qubit_qc()), obs)
    assert any(
        str(w.message).startswith(
            "QuEPP Monte Carlo: with n_samples=50 across 2 Pauli terms"
        )
        for w in record
    )


@pytest.mark.usefixtures("suppress_quepp_warnings")
@pytest.mark.parametrize("threshold, n_paths", [(0.0, 2), (0.1, 1)])
def test_exhaustive_coefficient_threshold_prunes_small_paths(threshold, n_paths):
    """The sin·sin path weighs sin²(0.3) ≈ 0.087."""
    qc = QuantumCircuit(1)
    qc.rx(0.3, 0)
    qc.rx(0.3, 0)
    protocol = QuEPP(
        sampling="exhaustive",
        truncation_order=2,
        coefficient_threshold=threshold,
        n_twirls=0,
    )
    _, ctx = protocol.expand(circuit_to_dag(qc), _Z0)
    assert ctx["n_paths"] == n_paths


class TestAutoSampling:
    """The default enumerates symbolic circuits silently and samples concrete ones."""

    def test_enumerates_a_symbolic_circuit_without_warning(self):
        prep = _single_rx_prep(Parameter("theta"))

        with warnings.catch_warnings():
            warnings.simplefilter("error", SymbolicAngleWarning)
            paths = QuEPP(n_twirls=0)._select_paths(prep)

        expected = QuEPP(sampling="exhaustive", n_twirls=0)._select_paths(prep)
        assert [p.branches for p in paths] == [p.branches for p in expected]

    def test_samples_a_concrete_circuit(self):
        prep = _single_rx_prep(0.4)

        paths = QuEPP(n_twirls=0)._select_paths(prep, rng=np.random.default_rng(7))

        expected = QuEPP(sampling="montecarlo", n_twirls=0)._select_paths(
            prep, rng=np.random.default_rng(7)
        )
        assert [p.branches for p in paths] == [p.branches for p in expected]


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
    def test_effectiveness_with_readout_noise(self, quepp_backend):
        """QuEPP mitigates uniform readout noise on a real backend."""
        meta = _rx_expval_meta(0.8)

        noise = maestro.NoiseModel()
        noise.set_all_readout_error(1, 0.05)

        # Readout error is applied after measurement, so maestro's analytical
        # estimate never sees it — every arm here has to sample.
        exact = list(
            CircuitPipeline(stages=[CircuitSpecStage(), MeasurementStage()])
            .run(meta, PipelineEnv(backend=quepp_backend(force_sampling=True)))
            .values()
        )[0][0]

        noisy = list(
            CircuitPipeline(stages=[CircuitSpecStage(), MeasurementStage()])
            .run(
                meta,
                PipelineEnv(
                    backend=quepp_backend(force_sampling=True, noise_model=noise)
                ),
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
                PipelineEnv(
                    backend=quepp_backend(force_sampling=True, noise_model=noise)
                ),
            )
            .values()
        )[0][0]

        noisy_err = abs(noisy - exact)
        quepp_err = abs(quepp_val - exact)
        assert quepp_err < noisy_err / 2, (
            f"QuEPP error ({quepp_err:.4f}) should be less than half "
            f"of noisy error ({noisy_err:.4f})"
        )


def _quepp_pipeline_values(meta: MetaCircuit, backend) -> list[float]:
    """Mitigated values of ``meta`` through spec → QuEPP → measurement on ``backend``."""
    pipeline = CircuitPipeline(
        stages=[
            CircuitSpecStage(),
            QEMStage(
                protocol=QuEPP(sampling="exhaustive", truncation_order=2, n_twirls=0)
            ),
            MeasurementStage(),
        ],
        suppress_performance_warnings=True,
    )
    return list(pipeline.run(meta, PipelineEnv(backend=backend)).values())[0]


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

    @pytest.mark.parametrize("n_copies", [2, 3])
    def test_path_dag_dedup_across_observables(self, n_copies):
        qc = QuantumCircuit(1)
        qc.rx(0.4, 0)
        obs = SparsePauliOp("Z")
        protocol = QuEPP(sampling="exhaustive", truncation_order=1, n_twirls=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dags_solo, _ = protocol.expand(circuit_to_dag(qc.copy()), obs)
            dags_multi, ctx_multi = protocol.expand(
                circuit_to_dag(qc.copy()), (obs,) * n_copies
            )
        assert len(dags_multi) == len(dags_solo)
        indices = [entry.dag_indices for entry in ctx_multi["per_obs"]]
        assert indices == [indices[0]] * n_copies

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

    def test_pipeline_e2e_matches_independent_runs(
        self, suppress_quepp_warnings, quepp_backend
    ):
        """End-to-end pipeline (CircuitSpecStage → QEMStage(QuEPP) →
        MeasurementStage) on a noiseless backend with two QWC observables
        produces the same per-observable mitigated values as running each
        observable through its own pipeline.
        """
        qc = _entangled_two_qubit_circuit()
        multi_out = _quepp_pipeline_values(
            meta_from_circuit(qc, observable=(_Z0_2Q, _Z0Z1)), quepp_backend()
        )
        solo_1, solo_2 = (
            _quepp_pipeline_values(
                meta_from_circuit(qc, observable=obs), quepp_backend()
            )
            for obs in (_Z0_2Q, _Z0Z1)
        )

        assert multi_out == [
            pytest.approx(solo_1[0], abs=1e-9),
            pytest.approx(solo_2[0], abs=1e-9),
        ]

    def test_pipeline_e2e_on_a_multi_term_hamiltonian(
        self, suppress_quepp_warnings, quepp_backend
    ):
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

        assert _quepp_pipeline_values(meta, quepp_backend()) == [
            pytest.approx(_exact_expval(_entangled_two_qubit_circuit(), spo), abs=1e-9)
        ]


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
