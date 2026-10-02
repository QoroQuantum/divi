# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.circuits.zne."""

import copy
import functools
import random

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager

from divi.circuits.zne import (
    ZNE,
    GlobalFoldPass,
    LinearExtrapolator,
    LocalFoldPass,
    RichardsonExtrapolator,
    global_fold,
    local_fold,
)
from divi.pipeline.abc import ResultFormat
from tests._helpers import exact_match


def _op_names(dag):
    return [inst.operation.name for inst in dag_to_circuit(dag).data]


def _fold_to_requested_scale(dag, scale):
    """A custom fold that mutates its input and realises the requested scale."""
    dag.apply_operation_back(dag.op_nodes()[0].op, dag.op_nodes()[0].qargs)
    return dag, scale


@pytest.fixture
def bell_dag():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return circuit_to_dag(qc)


@pytest.fixture
def h_rz_cx_dag():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.rz(0.3, 1)
    qc.cx(0, 1)
    return circuit_to_dag(qc)


@pytest.fixture
def stub_rng(mocker):
    return mocker.Mock(spec=random.Random)


class TestGlobalFoldPass:
    """Spec: GlobalFoldPass returns U · (U† · U)^k for scale = 1 + 2k."""

    @pytest.fixture
    def two_qubit_qc(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.5, 1)
        return qc

    def test_scale_1_is_identity(self, two_qubit_qc):
        out = PassManager([GlobalFoldPass(1.0)]).run(two_qubit_qc)
        assert Operator(out).equiv(Operator(two_qubit_qc))
        assert out.size() == two_qubit_qc.size()

    def test_scale_3_triples_gate_count(self, two_qubit_qc):
        folded = PassManager([GlobalFoldPass(3.0)]).run(two_qubit_qc)
        assert folded.size() == 3 * two_qubit_qc.size()

    def test_barriers_not_duplicated_in_fold(self):
        # Non-unitary instructions are excluded from the inverse and the tail;
        # the k-fold forward pass must exclude them too, or the barrier gets
        # re-applied inside every fold body.
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier()
        qc.cx(0, 1)

        folded = PassManager([GlobalFoldPass(3.0)]).run(qc)

        n_barriers = sum(
            1 for instr in folded.data if instr.operation.name == "barrier"
        )
        n_unitary = sum(1 for instr in folded.data if instr.operation.name != "barrier")
        assert n_barriers == 1
        assert n_unitary == 6  # two unitary gates folded to 3x
        assert Operator(folded).equiv(Operator(qc))

    @pytest.mark.parametrize("scale", [1.0, 3.0, 5.0, 7.0])
    def test_folded_unitary_equals_original(self, scale):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rz(0.7, 0)
        qc.cx(0, 1)
        qc.ry(-0.3, 1)
        folded = PassManager([GlobalFoldPass(scale)]).run(qc)
        assert Operator(folded).equiv(Operator(qc))

    def test_parametric_circuit_preserves_unitary(self):
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2)
        qc.rx(theta, 0)
        qc.cx(0, 1)
        qc.rz(2 * phi, 1)

        folded = PassManager([GlobalFoldPass(3.0)]).run(qc)

        assert {p.name for p in folded.parameters} == {"theta", "phi"}
        binding = {theta: 0.4, phi: 0.2}
        assert Operator(folded.assign_parameters(binding)).equiv(
            Operator(qc.assign_parameters(binding))
        )

    def test_rejects_below_one_scale(self):
        with pytest.raises(ValueError, match=">= 1"):
            GlobalFoldPass(0.5)

    @pytest.mark.parametrize(
        "scale,expected", [(1.0, 1.0), (1.25, 1.0), (1.5, 1.5), (3.0, 3.0)]
    )
    def test_effective_scale_snaps_to_the_gate_grid(self, scale, expected):
        qc = QuantumCircuit(1)
        for angle in (0.1, 0.2, 0.3, 0.4):
            qc.rx(angle, 0)
        assert GlobalFoldPass(scale).effective_scale(circuit_to_dag(qc)) == expected

    @pytest.mark.parametrize(
        "scale,d,expected_size",
        [
            (1.5, 4, 6),
            (2.0, 4, 8),
            (2.5, 4, 10),
            (3.5, 4, 14),
            (4.0, 4, 16),
        ],
    )
    def test_fractional_scale_gate_count(self, scale, d, expected_size):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.5, 1)
        qc.ry(-0.3, 0)
        assert qc.size() == d
        folded = PassManager([GlobalFoldPass(scale)]).run(qc)
        assert folded.size() == expected_size

    @pytest.mark.parametrize("scale", [1.25, 1.5, 2.0, 2.5, 3.5, 4.0])
    def test_fractional_scale_preserves_unitary(self, scale):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rx(0.4, 0)
        qc.cx(0, 1)
        qc.ry(-0.2, 1)
        folded = PassManager([GlobalFoldPass(scale)]).run(qc)
        assert Operator(folded).equiv(Operator(qc))

    def test_fractional_scale_folds_the_tail(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(0.5, 0)
        qc.rx(0.4, 0)
        qc.ry(0.3, 0)
        folded = PassManager([GlobalFoldPass(1.5)]).run(qc)
        names = [inst.operation.name for inst in folded.data]
        assert names == ["h", "rz", "rx", "ry", "ry", "ry"]
        assert Operator(folded).equiv(Operator(qc))

    def test_fractional_parametric_circuit_preserves_unitary(self):
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2)
        qc.rx(theta, 0)
        qc.cx(0, 1)
        qc.rz(2 * phi, 1)

        folded = PassManager([GlobalFoldPass(2.5)]).run(qc)

        assert {p.name for p in folded.parameters} == {"theta", "phi"}
        binding = {theta: 0.4, phi: 0.2}
        assert Operator(folded.assign_parameters(binding)).equiv(
            Operator(qc.assign_parameters(binding))
        )


class TestLocalFoldPass:
    """Spec: LocalFoldPass replaces each gate with G·(G†·G)^k."""

    @pytest.fixture
    def four_gate_qc(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.5, 1)
        qc.ry(-0.3, 0)
        return qc

    def test_scale_1_is_identity(self, four_gate_qc):
        out = PassManager([LocalFoldPass(1.0)]).run(four_gate_qc)
        assert out.size() == four_gate_qc.size()
        assert Operator(out).equiv(Operator(four_gate_qc))

    @pytest.mark.parametrize("scale,expected_mult", [(3.0, 3), (5.0, 5), (7.0, 7)])
    def test_odd_integer_scale_multiplies_gate_count(
        self, four_gate_qc, scale, expected_mult
    ):
        folded = PassManager([LocalFoldPass(scale)]).run(four_gate_qc)
        assert folded.size() == expected_mult * four_gate_qc.size()

    @pytest.mark.parametrize("scale", [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0])
    def test_folded_unitary_equals_original(self, four_gate_qc, scale):
        rng = random.Random(0)
        folded = PassManager([LocalFoldPass(scale, rng=rng)]).run(four_gate_qc)
        assert Operator(folded).equiv(Operator(four_gate_qc))

    @pytest.mark.parametrize(
        "scale,d,expected_size",
        [
            (1.5, 4, 6),
            (2.0, 4, 8),
            (2.5, 4, 10),
            (3.5, 4, 14),
        ],
    )
    def test_fractional_scale_gate_count(self, scale, d, expected_size):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.5, 1)
        qc.ry(-0.3, 0)
        assert qc.size() == d
        folded = PassManager([LocalFoldPass(scale, rng=random.Random(0))]).run(qc)
        assert folded.size() == expected_size

    def test_parametric_circuit_preserves_unitary(self):
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2)
        qc.rx(theta, 0)
        qc.cx(0, 1)
        qc.rz(2 * phi, 1)

        folded = PassManager([LocalFoldPass(2.0, rng=random.Random(0))]).run(qc)

        assert {p.name for p in folded.parameters} == {"theta", "phi"}
        binding = {theta: 0.4, phi: 0.2}
        assert Operator(folded.assign_parameters(binding)).equiv(
            Operator(qc.assign_parameters(binding))
        )

    def test_non_unitary_ops_are_skipped(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.barrier()
        qc.measure([0, 1], [0, 1])

        folded = PassManager([LocalFoldPass(3.0)]).run(qc)
        op_counts = folded.count_ops()
        assert op_counts.get("barrier", 0) == 1
        assert op_counts.get("measure", 0) == 2
        assert op_counts["h"] == 3
        assert op_counts["cx"] == 3

    def test_rng_reproducibility(self, four_gate_qc):
        a = PassManager([LocalFoldPass(1.5, rng=random.Random(7))]).run(four_gate_qc)
        b = PassManager([LocalFoldPass(1.5, rng=random.Random(7))]).run(four_gate_qc)
        assert [inst.operation.name for inst in a.data] == [
            inst.operation.name for inst in b.data
        ]

    def test_rejects_below_one_scale(self):
        with pytest.raises(ValueError, match=">= 1"):
            LocalFoldPass(0.5)

    def test_rejects_unknown_selection(self):
        with pytest.raises(ValueError, match="selection must be one of"):
            LocalFoldPass(2.0, selection="middle")

    @pytest.mark.parametrize("selection", ["from_left", "from_right", "random"])
    def test_selection_preserves_unitary_and_size(self, four_gate_qc, selection):
        folded = PassManager(
            [LocalFoldPass(2.0, selection=selection, rng=random.Random(0))]
        ).run(four_gate_qc)
        assert folded.size() == 8
        assert Operator(folded).equiv(Operator(four_gate_qc))

    def test_from_left_folds_prefix(self):
        qc = QuantumCircuit(1)
        for _ in range(4):
            qc.rx(0.3, 0)
        folded = PassManager([LocalFoldPass(1.5, selection="from_left")]).run(qc)
        names = [inst.operation.name for inst in folded.data]
        assert len(names) == 6
        assert names[:3] == ["rx", "rx", "rx"]
        assert names[3:] == ["rx", "rx", "rx"]

    def test_from_right_folds_suffix(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(1)
        qc.cx(0, 1)
        qc.rz(0.2, 0)
        folded = PassManager([LocalFoldPass(1.5, selection="from_right")]).run(qc)
        names = [inst.operation.name for inst in folded.data]
        assert names == ["h", "x", "cx", "rz", "rz", "rz"]

    def test_exclude_by_name_leaves_gates_untouched(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.3, 1)
        qc.cx(1, 0)
        folded = PassManager([LocalFoldPass(3.0, exclude={"cx"})]).run(qc)
        counts = folded.count_ops()
        assert counts["cx"] == 2
        assert counts["h"] == 3
        assert counts["rz"] == 3
        assert Operator(folded).equiv(Operator(qc))

    def test_exclude_by_arity_shorthand(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.cz(0, 1)
        qc.rz(0.2, 1)
        folded = PassManager([LocalFoldPass(3.0, exclude={"double"})]).run(qc)
        counts = folded.count_ops()
        assert counts["cx"] == 1
        assert counts["cz"] == 1
        assert counts["h"] == 3
        assert counts["rz"] == 3
        assert Operator(folded).equiv(Operator(qc))

    def test_exclude_affects_effective_scale(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.3, 1)
        qc.cx(1, 0)
        folded = PassManager(
            [LocalFoldPass(2.0, exclude={"cx"}, rng=random.Random(0))]
        ).run(qc)
        assert folded.size() == 6
        assert Operator(folded).equiv(Operator(qc))

    def test_exclude_unknown_name_is_harmless(self, four_gate_qc):
        folded = PassManager([LocalFoldPass(3.0, exclude={"nonexistent_gate"})]).run(
            four_gate_qc
        )
        assert folded.size() == 3 * four_gate_qc.size()
        assert Operator(folded).equiv(Operator(four_gate_qc))

    def test_exclude_all_gates_returns_untouched(self, four_gate_qc):
        folded = PassManager([LocalFoldPass(3.0, exclude={"single", "double"})]).run(
            four_gate_qc
        )
        assert folded.size() == four_gate_qc.size()
        assert Operator(folded).equiv(Operator(four_gate_qc))

    def test_deterministic_selections_ignore_rng(self, four_gate_qc):
        for selection in ("from_left", "from_right"):
            a = PassManager(
                [LocalFoldPass(1.5, selection=selection, rng=random.Random(0))]
            ).run(four_gate_qc)
            b = PassManager(
                [LocalFoldPass(1.5, selection=selection, rng=random.Random(999))]
            ).run(four_gate_qc)
            assert [i.operation.name for i in a.data] == [
                i.operation.name for i in b.data
            ]

    def test_empty_circuit_is_untouched(self):
        qc = QuantumCircuit(2)
        folded = PassManager([LocalFoldPass(3.0)]).run(qc)
        assert folded.size() == 0

    def test_random_selection_without_rng_uses_a_default_source(self):
        qc = QuantumCircuit(1)
        for _ in range(4):
            qc.rx(0.3, 0)
        folded = PassManager([LocalFoldPass(1.5)]).run(qc)
        assert folded.size() == 6


@pytest.mark.parametrize("fold_pass", [GlobalFoldPass, LocalFoldPass])
def test_single_gate_triples_at_scale_3(fold_pass):
    qc = QuantumCircuit(1)
    qc.rx(0.3, 0)
    pass_ = fold_pass(3.0)
    assert pass_.effective_scale(circuit_to_dag(qc)) == 3.0
    assert PassManager([pass_]).run(qc).size() == 3


@pytest.mark.parametrize("exclude", [{"cx"}, {"double"}])
def test_local_fold_exclude_sets_the_effective_scale(h_rz_cx_dag, exclude):
    folded, effective = local_fold(
        h_rz_cx_dag, 2.0, selection="from_left", exclude=exclude
    )
    assert _op_names(folded) == ["h", "h", "h", "rz", "cx"]
    assert effective == 2.0


def test_local_fold_from_left_ignores_rng(h_rz_cx_dag, stub_rng):
    stub_rng.sample.return_value = [0, 2]
    folded, effective = local_fold(
        h_rz_cx_dag, 2.0, selection="from_left", rng=stub_rng
    )
    assert _op_names(folded) == ["h", "h", "h", "rz", "rz", "rz", "cx"]
    assert effective == pytest.approx(7 / 3)
    stub_rng.sample.assert_not_called()


@pytest.mark.parametrize("seed", range(4))
def test_local_fold_defaults_to_random_selection_from_rng(h_rz_cx_dag, seed):
    default, default_scale = local_fold(
        copy.deepcopy(h_rz_cx_dag), 2.0, rng=random.Random(seed)
    )
    explicit, explicit_scale = local_fold(
        h_rz_cx_dag, 2.0, selection="random", rng=random.Random(seed)
    )
    assert _op_names(default) == _op_names(explicit)
    assert default_scale == explicit_scale == pytest.approx(7 / 3)


class TestZNEProtocol:
    def test_applies_to_expvals_only(self):
        zne = ZNE([1.0, 3.0])
        assert zne.applies_to(ResultFormat.EXPVALS) is True
        assert zne.applies_to(ResultFormat.PROBS) is False
        assert zne.applies_to(ResultFormat.COUNTS) is False

    def test_twirl_and_bind_defaults(self):
        zne = ZNE([1.0, 3.0])
        assert zne.n_twirls == 0
        assert zne.requires_bound_params is False


class TestZNE:
    def test_valid_initialization(self):
        zne = ZNE(scale_factors=[1.0, 3.0, 5.0])
        assert zne.name == "zne"
        assert list(zne.scale_factors) == [1.0, 3.0, 5.0]
        assert isinstance(zne.extrapolator, RichardsonExtrapolator)

    def test_accepts_explicit_extrapolator(self):
        extrapolator = LinearExtrapolator()
        zne = ZNE(scale_factors=[1.0, 3.0], extrapolator=extrapolator)
        assert zne.extrapolator is extrapolator

    @pytest.mark.parametrize(
        "bad_scale",
        [
            "not_a_sequence",
            [1.0, "foo"],
        ],
    )
    def test_rejects_invalid_scale_factor_types(self, bad_scale):
        with pytest.raises(
            ValueError, match=r"^scale_factors must be a sequence of real numbers\.$"
        ):
            ZNE(scale_factors=bad_scale)

    def test_rejects_scale_factor_below_one(self):
        with pytest.raises(ValueError, match=r"^All scale factors must be ≥ 1\.0\.$"):
            ZNE(scale_factors=[0.5, 1.0])

    @pytest.mark.parametrize("bad_scale", [[], [1.0]])
    def test_rejects_fewer_than_two_scale_factors(self, bad_scale):
        expected = (
            "scale_factors must contain at least two points to extrapolate to the "
            f"zero-noise limit; got {bad_scale}."
        )
        with pytest.raises(ValueError, match=exact_match(expected)):
            ZNE(scale_factors=bad_scale)

    def test_rejects_duplicate_scale_factors(self):
        with pytest.raises(
            ValueError,
            match=r"^scale_factors must be unique; got duplicates in \[1\.0, 1\.0, 3\.0\]\.$",
        ):
            ZNE(scale_factors=[1.0, 1.0, 3.0])

    def test_rejects_non_extrapolator(self):
        with pytest.raises(
            ValueError, match=r"^extrapolator must be a ZNEExtrapolator, got str\.$"
        ):
            ZNE(scale_factors=[1.0, 3.0], extrapolator="not an extrapolator")

    def test_expand_effective_scales_ignore_non_unitary_ops(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier()
        qc.cx(0, 1)
        dags, ctx = ZNE(scale_factors=[1.0, 2.0]).expand(circuit_to_dag(qc))
        assert ctx["effective_scales"] == (1.0, 2.0)
        assert [_op_names(d) for d in dags] == [
            ["h", "barrier", "cx"],
            ["h", "barrier", "cx", "cx", "cx"],
        ]

    def test_reduce_rejects_empty_results(self):
        with pytest.raises(
            RuntimeError, match=r"^ZNE received an empty results sequence\.$"
        ):
            ZNE(scale_factors=[1.0, 3.0]).reduce([], {})

    def test_expand_folds_one_dag_per_scale(self, bell_dag):
        dags, ctx = ZNE(scale_factors=[1.0, 3.0, 5.0]).expand(bell_dag)
        assert ctx["effective_scales"] == (1.0, 3.0, 5.0)
        assert [_op_names(d) for d in dags] == [
            ["h", "cx"],
            ["h", "cx", "cx", "h", "h", "cx"],
            ["h", "cx", "cx", "h", "h", "cx", "cx", "h", "h", "cx"],
        ]

    def test_expand_leaves_the_callers_dag_untouched(self, bell_dag):
        """``expand`` receives a DAG it does not own — a program's seed circuit,
        reused by every pipeline built from it and by every later call. Folding it
        in place leaves it permanently scaled: sibling pipelines that mitigation
        does not apply to inherit the fold, and repeated calls compound it."""
        before = bell_dag.size()
        zne = ZNE(scale_factors=[1.0, 3.0, 5.0])

        first, _ = zne.expand(bell_dag)
        assert bell_dag.size() == before
        assert Operator(dag_to_circuit(bell_dag)).equiv(
            Operator(dag_to_circuit(first[0]))
        )

        second, _ = zne.expand(bell_dag)
        assert [d.size() for d in second] == [d.size() for d in first]

    def test_expand_preserves_unitary(self, bell_dag):
        zne = ZNE(scale_factors=[1.0, 3.0, 5.0])
        dags, _ = zne.expand(bell_dag)
        u_orig = Operator(dag_to_circuit(bell_dag))
        for d in dags:
            assert Operator(dag_to_circuit(d)).equiv(u_orig)

    @pytest.mark.parametrize(
        "ctx",
        [
            pytest.param({"effective_scales": (1.0, 3.0, 5.0)}, id="effective_scales"),
            pytest.param({}, id="falls_back_to_requested_scales"),
        ],
    )
    def test_reduce_extrapolates_to_zero(self, ctx):
        zne = ZNE(scale_factors=[1.0, 3.0, 5.0], extrapolator=LinearExtrapolator())
        assert zne.reduce([1.0, -1.0, -3.0], ctx) == pytest.approx([2.0])

    def test_expand_forwards_effective_scales_to_reduce(self, bell_dag):
        zne = ZNE(scale_factors=[1.0, 3.0, 5.0], extrapolator=LinearExtrapolator())
        _, ctx = zne.expand(bell_dag)
        assert zne.reduce([1.0, -1.0, -3.0], ctx) == pytest.approx([2.0])

    def test_expand_warns_when_scales_collapse(self, bell_dag):
        zne = ZNE(scale_factors=[1.5, 2.5, 3.0])
        with pytest.warns(UserWarning, match="collapse to effective scales"):
            zne.expand(bell_dag)

    def test_reduce_with_per_obs_list(self, bell_dag):
        zne = ZNE(scale_factors=[1.0, 3.0], extrapolator=LinearExtrapolator())
        _, ctx = zne.expand(bell_dag, observable=("a", "b"))
        rows = [[1.0, 2.0], [-1.0, -2.0]]
        out = zne.reduce(rows, ctx)
        assert isinstance(out, list)
        assert out[0] == pytest.approx(2.0)
        assert out[1] == pytest.approx(4.0)


class TestZNEDryExpand:
    """ZNE's analytic dry path: same fan-out/context as expand, zero mutation."""

    def test_emits_one_dag_per_scale_without_folding(self, bell_dag):
        zne = ZNE(scale_factors=[1.0, 3.0, 5.0])
        base = bell_dag.size()
        dags, ctx = zne.dry_expand(bell_dag)
        assert len(dags) == 3
        assert all(d is bell_dag for d in dags)
        assert bell_dag.size() == base
        assert len(ctx["effective_scales"]) == 3

    @pytest.mark.parametrize(
        "folding_fn",
        [
            global_fold,
            local_fold,
            functools.partial(local_fold, selection="from_left", exclude={"cx"}),
            _fold_to_requested_scale,
            functools.partial(_fold_to_requested_scale),
        ],
        ids=["global", "local", "local-excluding-cx", "custom", "custom-partial"],
    )
    def test_effective_scales_match_expand(self, h_rz_cx_dag, folding_fn):
        zne = ZNE(scale_factors=[1.0, 2.0, 3.0], folding_fn=folding_fn)
        before = copy.deepcopy(h_rz_cx_dag)
        _, dry_ctx = zne.dry_expand(h_rz_cx_dag)
        assert h_rz_cx_dag == before
        _, real_ctx = zne.expand(h_rz_cx_dag)
        assert dry_ctx["effective_scales"] == real_ctx["effective_scales"]

    def test_aliased_batch_does_not_compound(self, bell_dag):
        zne = ZNE(scale_factors=[1.0, 3.0, 5.0])
        base = bell_dag.size()
        shared_entries = [bell_dag] * 10
        for dag in shared_entries:
            zne.dry_expand(dag)
        assert bell_dag.size() == base

    def test_warns_when_scales_collapse(self, bell_dag):
        zne = ZNE(scale_factors=[1.5, 2.5, 3.0])
        with pytest.warns(UserWarning, match="collapse to effective scales"):
            zne.dry_expand(bell_dag)

    def test_circuit_without_foldable_gates_stays_at_unit_scale(self):
        zne = ZNE(scale_factors=[1.0, 3.0])
        with pytest.warns(UserWarning, match="collapse to effective scales"):
            _, ctx = zne.dry_expand(circuit_to_dag(QuantumCircuit(1)))
        assert ctx["effective_scales"] == (1.0, 1.0)


class TestLinearExtrapolator:
    def test_fits_line_through_two_points(self):
        e = LinearExtrapolator()
        assert e.extrapolate([1.0, 3.0], [3.0, 7.0]) == pytest.approx(1.0)

    def test_intercept_from_three_points(self):
        e = LinearExtrapolator()
        sfs = [1.0, 3.0, 5.0]
        ys = [4.5, 3.5, 2.5]
        assert e.extrapolate(sfs, ys) == pytest.approx(5.0)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="lengths disagree"):
            LinearExtrapolator().extrapolate([1.0, 3.0], [1.0])

    def test_rejects_single_point(self):
        with pytest.raises(
            ValueError, match=r"^LinearExtrapolator requires at least 2 data points\.$"
        ):
            LinearExtrapolator().extrapolate([1.0], [2.0])


class TestRichardsonExtrapolator:
    def test_interpolates_polynomial_exactly(self):
        e = RichardsonExtrapolator()
        sfs = [1.0, 2.0, 3.0]
        ys = [4.0, 4.0, 2.0]
        assert e.extrapolate(sfs, ys) == pytest.approx(2.0)

    def test_linear_through_two_points_matches_linear(self):
        sfs = [1.0, 3.0]
        ys = [0.5, -0.5]
        richardson = RichardsonExtrapolator().extrapolate(sfs, ys)
        linear = LinearExtrapolator().extrapolate(sfs, ys)
        assert richardson == pytest.approx(linear)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="lengths disagree"):
            RichardsonExtrapolator().extrapolate([1.0, 3.0], [1.0])

    def test_rejects_duplicate_scale_factors(self):
        with pytest.raises(
            ValueError,
            match=(
                r"^RichardsonExtrapolator requires unique scale factors; "
                r"got duplicates in \[1\.0, 3\.0, 3\.0\]\.$"
            ),
        ):
            RichardsonExtrapolator().extrapolate([1.0, 3.0, 3.0], [0.5, 0.3, 0.3])

    def test_single_point_returns_its_value(self):
        assert RichardsonExtrapolator().extrapolate([2.0], [0.7]) == 0.7

    def test_rejects_empty_input(self):
        with pytest.raises(
            ValueError,
            match=r"^RichardsonExtrapolator requires at least 1 data point\.$",
        ):
            RichardsonExtrapolator().extrapolate([], [])


@pytest.mark.parametrize("extrapolator", [LinearExtrapolator, RichardsonExtrapolator])
@pytest.mark.parametrize(
    "scale_factors,results,field",
    [
        ([1.0, 3.0], [float("nan"), 1.0], "results"),
        ([1.0, float("inf")], [1.0, 2.0], "scale_factors"),
    ],
)
def test_extrapolators_reject_non_finite_input(
    extrapolator, scale_factors, results, field
):
    expected = f"{extrapolator.__name__}: {field} contains NaN or Inf values."
    with pytest.raises(ValueError, match=exact_match(expected)):
        extrapolator().extrapolate(scale_factors, results)
