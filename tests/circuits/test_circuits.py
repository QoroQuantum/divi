# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit


@pytest.fixture
def plain_dag():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return circuit_to_dag(qc)


@pytest.fixture
def parametric_dag():
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc = QuantumCircuit(2)
    qc.rx(theta, 0)
    qc.ry(phi, 1)
    qc.cx(0, 1)
    return circuit_to_dag(qc), (theta, phi)


class TestMetaCircuit:
    """Construction, defaults, and setters for the reshaped MetaCircuit."""

    def test_minimal_construction(self, plain_dag):
        meta = MetaCircuit(circuit_bodies=(((), plain_dag),))
        assert meta.circuit_bodies == (((), plain_dag),)
        assert meta.parameters == ()
        assert meta.observable is None
        assert meta.measured_wires is None
        assert meta.measurement_qasms == ()
        assert meta.measurement_groups == ()
        assert meta.precision == 8

    def test_with_parameters(self, parametric_dag):
        dag, params = parametric_dag
        meta = MetaCircuit(circuit_bodies=(((), dag),), parameters=params)
        assert meta.parameters == params

    def test_with_observable(self, plain_dag):
        obs = SparsePauliOp.from_list([("IZ", 1.0), ("ZI", -1.0)])
        meta = MetaCircuit(circuit_bodies=(((), plain_dag),), observable=obs)
        assert meta.observable is obs

    def test_with_measured_wires(self, plain_dag):
        meta = MetaCircuit(
            circuit_bodies=(((), plain_dag),),
            measured_wires=(0, 1),
        )
        assert meta.measured_wires == (0, 1)

    def test_custom_precision(self, plain_dag):
        meta = MetaCircuit(circuit_bodies=(((), plain_dag),), precision=12)
        assert meta.precision == 12

    def test_multi_body_tags(self, plain_dag, parametric_dag):
        dag2, _ = parametric_dag
        bodies = (
            ((("qem", 0),), plain_dag),
            ((("qem", 1),), dag2),
        )
        meta = MetaCircuit(circuit_bodies=bodies)
        assert len(meta.circuit_bodies) == 2
        assert meta.circuit_bodies[1][0] == (("qem", 1),)

    def test_empty_bodies_raises(self):
        with pytest.raises(ValueError, match="at least one circuit body"):
            MetaCircuit(circuit_bodies=())

    def test_set_circuit_bodies_returns_new_instance(self, plain_dag, parametric_dag):
        orig = MetaCircuit(circuit_bodies=(((), plain_dag),))
        new_dag, params = parametric_dag
        out = orig.set_circuit_bodies(
            ((("qem", 0),), new_dag),
        )
        assert out is not orig
        # Original untouched.
        assert orig.circuit_bodies == (((), plain_dag),)
        # set_circuit_bodies doesn't mutate parameters — caller re-wraps if needed.
        assert orig.parameters == ()

    def test_set_measurement_bodies_returns_new_instance(self, plain_dag):
        meta = MetaCircuit(circuit_bodies=(((), plain_dag),))
        meas = (((("meas", 0),), "measure q[0] -> c[0];\n"),)
        out = meta.set_measurement_bodies(meas)
        assert out is not meta
        assert out.measurement_qasms == meas
        assert meta.measurement_qasms == ()

    def test_set_measurement_groups_returns_new_instance(self, plain_dag):
        meta = MetaCircuit(circuit_bodies=(((), plain_dag),))
        groups: tuple[tuple[object, ...], ...] = ((object(),),)
        out = meta.set_measurement_groups(groups)
        assert out is not meta
        assert out.measurement_groups == groups
        assert meta.measurement_groups == ()

    def test_frozen_dataclass_disallows_direct_field_assignment(self, plain_dag):
        meta = MetaCircuit(circuit_bodies=(((), plain_dag),))
        with pytest.raises((AttributeError, Exception)):
            meta.precision = 4  # type: ignore[misc]
