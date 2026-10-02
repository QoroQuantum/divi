# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.circuits.qem base protocols."""

import pytest
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from divi.circuits.qem import QEMProtocol, _NoMitigation
from divi.pipeline.abc import ResultFormat


@pytest.fixture
def bell_dag():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return circuit_to_dag(qc)


class _EchoProtocol(QEMProtocol):
    name = "echo"

    def expand(self, dag, observable=None):
        return (dag,), {"observable": observable}

    def reduce(self, quantum_results, context):
        return list(quantum_results)


def test_default_dry_expand_delegates_to_expand(bell_dag):
    observable = ("o1",)
    (dag,), ctx = _EchoProtocol().dry_expand(bell_dag, observable)
    assert dag is bell_dag
    assert ctx["observable"] is observable


class TestQEMProtocol:
    def test_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            QEMProtocol()

    def test_twirl_and_bind_defaults(self):
        proto = _NoMitigation()
        assert proto.n_twirls == 0
        assert proto.requires_bound_params is False


class TestNoMitigation:
    def test_name(self):
        assert _NoMitigation().name == "NoMitigation"

    def test_no_mitigation_vacuously_applies(self):
        proto = _NoMitigation()
        assert proto.applies_to(ResultFormat.EXPVALS) is True
        assert proto.applies_to(ResultFormat.PROBS) is True
        assert proto.applies_to(ResultFormat.COUNTS) is True

    def test_expand_is_identity(self, bell_dag):
        dags, ctx = _NoMitigation().expand(bell_dag)
        assert len(dags) == 1
        assert dags[0] is bell_dag
        assert ctx == {}

    def test_reduce_returns_single_value(self, bell_dag):
        p = _NoMitigation()
        assert p.reduce([1.23], {}) == [1.23]
        assert p.reduce([-0.5], {}) == [-0.5]

    def test_reduce_raises_on_multi_results(self):
        with pytest.raises(
            RuntimeError,
            match=r"^NoMitigation class received multiple partial results\.$",
        ):
            _NoMitigation().reduce([0.1, 0.2], {})

    def test_reduce_raises_on_empty_results(self):
        with pytest.raises(
            RuntimeError, match=r"^NoMitigation received an empty results sequence\.$"
        ):
            _NoMitigation().reduce([], {})


class TestNoMitigationTupleObservable:
    """Tuple-observable expand/reduce on the trivial protocol."""

    def test_expand_with_tuple_returns_single_context(self, bell_dag):
        dags, ctx = _NoMitigation().expand(bell_dag, observable=("o1", "o2"))
        assert ctx == {}
        assert len(dags) == 1

    def test_reduce_with_per_obs_list(self, bell_dag):
        out = _NoMitigation().reduce([[0.7, -0.3]], {})
        assert out == [0.7, -0.3]
