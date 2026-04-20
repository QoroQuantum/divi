# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.circuits.qem (DAG-native QEM protocols)."""

import pytest
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Operator

from divi.circuits.qem import (
    ZNE,
    LinearExtrapolator,
    QEMProtocol,
    RichardsonExtrapolator,
    _NoMitigation,
)


@pytest.fixture
def bell_dag():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return circuit_to_dag(qc)


class TestQEMProtocol:
    def test_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            QEMProtocol()

    def test_concrete_implementations_can_be_instantiated(self):
        assert isinstance(_NoMitigation(), QEMProtocol)
        assert isinstance(ZNE([1.0, 3.0]), QEMProtocol)


class TestNoMitigation:
    def test_name(self):
        assert _NoMitigation().name == "NoMitigation"

    def test_expand_is_identity(self, bell_dag):
        dags, ctx = _NoMitigation().expand(bell_dag)
        assert len(dags) == 1
        assert dags[0] is bell_dag
        assert ctx == {}

    def test_reduce_returns_single_value(self, bell_dag):
        p = _NoMitigation()
        assert p.reduce([1.23], {}) == 1.23
        assert p.reduce([-0.5], {}) == -0.5

    def test_reduce_raises_on_multi_results(self, bell_dag):
        with pytest.raises(RuntimeError, match="multiple partial results"):
            _NoMitigation().reduce([0.1, 0.2], {})

    def test_reduce_raises_on_empty_results(self, bell_dag):
        with pytest.raises(RuntimeError, match="empty results sequence"):
            _NoMitigation().reduce([], {})


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
            [1.0, "foo"],  # non-numeric
        ],
    )
    def test_rejects_invalid_scale_factor_types(self, bad_scale):
        with pytest.raises(ValueError, match="sequence of real numbers"):
            ZNE(scale_factors=bad_scale)

    def test_rejects_scale_factor_below_one(self):
        with pytest.raises(ValueError, match="≥ 1"):
            ZNE(scale_factors=[0.5, 1.0])

    def test_rejects_non_extrapolator(self):
        with pytest.raises(ValueError, match="ZNEExtrapolator"):
            ZNE(scale_factors=[1.0, 3.0], extrapolator="not an extrapolator")

    def test_expand_returns_one_dag_per_scale(self, bell_dag):
        zne = ZNE(scale_factors=[1.0, 3.0, 5.0])
        dags, ctx = zne.expand(bell_dag)
        assert len(dags) == 3
        # Effective scales are reported in the context for extrapolation.
        assert ctx["effective_scales"] == (1.0, 3.0, 5.0)

    def test_expand_preserves_unitary(self, bell_dag):
        zne = ZNE(scale_factors=[1.0, 3.0, 5.0])
        dags, _ = zne.expand(bell_dag)
        u_orig = Operator(dag_to_circuit(bell_dag))
        for d in dags:
            assert Operator(dag_to_circuit(d)).equiv(u_orig)

    def test_expand_scales_gate_count(self, bell_dag):
        zne = ZNE(scale_factors=[1.0, 3.0, 5.0])
        base = bell_dag.size()  # capture before expand (which consumes the input)
        dags, _ = zne.expand(bell_dag)
        assert [d.size() for d in dags] == [base, 3 * base, 5 * base]

    def test_reduce_extrapolates_to_zero(self, bell_dag):
        # y = 2 - s → intercept at s=0 is 2.
        zne = ZNE(scale_factors=[1.0, 3.0, 5.0], extrapolator=LinearExtrapolator())
        extrapolated = zne.reduce(
            [1.0, -1.0, -3.0], {"effective_scales": (1.0, 3.0, 5.0)}
        )
        assert extrapolated == pytest.approx(2.0)

    def test_reduce_falls_back_to_requested_scales_without_context(self, bell_dag):
        """If a legacy context lacks effective_scales, reduce uses the requested values."""
        zne = ZNE(scale_factors=[1.0, 3.0, 5.0], extrapolator=LinearExtrapolator())
        assert zne.reduce([1.0, -1.0, -3.0], {}) == pytest.approx(2.0)

    def test_expand_forwards_effective_scales_to_reduce(self, bell_dag):
        """Effective scales survive the expand→reduce roundtrip unbiased."""
        zne = ZNE(scale_factors=[1.0, 3.0, 5.0], extrapolator=LinearExtrapolator())
        _, ctx = zne.expand(bell_dag)
        # y = 2 - s evaluated at effective scales (1, 3, 5).
        assert zne.reduce([1.0, -1.0, -3.0], ctx) == pytest.approx(2.0)

    def test_expand_warns_when_scales_collapse(self, bell_dag):
        """Small-d circuits can snap distinct requested scales to the same effective value."""
        # bell_dag has d=2; requested 1.5 → eff=1.0, 2.5 → eff=3.0, 3.0 → eff=3.0.
        zne = ZNE(scale_factors=[1.5, 2.5, 3.0])
        with pytest.warns(UserWarning, match="collapse to effective scales"):
            zne.expand(bell_dag)


class TestLinearExtrapolator:
    def test_fits_line_through_two_points(self):
        e = LinearExtrapolator()
        # y = 1 + 2*s → intercept = 1.
        assert e.extrapolate([1.0, 3.0], [3.0, 7.0]) == pytest.approx(1.0)

    def test_intercept_from_three_points(self):
        e = LinearExtrapolator()
        # Noisy y = 5 - 0.5*s; best linear fit intercept close to 5.
        sfs = [1.0, 3.0, 5.0]
        ys = [4.5, 3.5, 2.5]
        assert e.extrapolate(sfs, ys) == pytest.approx(5.0)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="lengths disagree"):
            LinearExtrapolator().extrapolate([1.0, 3.0], [1.0])

    def test_rejects_single_point(self):
        with pytest.raises(ValueError, match="at least 2"):
            LinearExtrapolator().extrapolate([1.0], [2.0])

    def test_rejects_nan_input(self):
        with pytest.raises(ValueError, match="NaN or Inf"):
            LinearExtrapolator().extrapolate([1.0, 3.0], [float("nan"), 1.0])

    def test_rejects_inf_input(self):
        with pytest.raises(ValueError, match="NaN or Inf"):
            LinearExtrapolator().extrapolate([1.0, float("inf")], [1.0, 2.0])


class TestRichardsonExtrapolator:
    def test_interpolates_polynomial_exactly(self):
        e = RichardsonExtrapolator()
        # y = 2 + 3*s - s**2 evaluated at s=1,2,3 → y=4,4,2. At s=0, y=2.
        sfs = [1.0, 2.0, 3.0]
        ys = [4.0, 4.0, 2.0]
        assert e.extrapolate(sfs, ys) == pytest.approx(2.0)

    def test_linear_through_two_points_matches_linear(self):
        # For N=2 points, Richardson reduces to linear extrapolation.
        sfs = [1.0, 3.0]
        ys = [0.5, -0.5]
        richardson = RichardsonExtrapolator().extrapolate(sfs, ys)
        linear = LinearExtrapolator().extrapolate(sfs, ys)
        assert richardson == pytest.approx(linear)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="lengths disagree"):
            RichardsonExtrapolator().extrapolate([1.0, 3.0], [1.0])

    def test_rejects_duplicate_scale_factors(self):
        with pytest.raises(ValueError, match="duplicates"):
            RichardsonExtrapolator().extrapolate([1.0, 3.0, 3.0], [0.5, 0.3, 0.3])

    def test_rejects_nan_input(self):
        with pytest.raises(ValueError, match="NaN or Inf"):
            RichardsonExtrapolator().extrapolate([1.0, 3.0], [float("nan"), 1.0])
