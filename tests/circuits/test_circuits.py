# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit, _core


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

    def test_with_observable(self, plain_dag):
        obs = SparsePauliOp.from_list([("IZ", 1.0), ("ZI", -1.0)])
        meta = MetaCircuit(circuit_bodies=(((), plain_dag),), observable=obs)
        assert meta.observable == (obs,)

    @pytest.mark.parametrize(
        "assign",
        [
            lambda meta, obs: MetaCircuit(
                circuit_bodies=meta.circuit_bodies, observable=obs
            ),
            lambda meta, obs: meta.set_observable(obs),
            lambda meta, obs: meta.set_observable((obs,)),
        ],
        ids=["constructor", "set_observable", "set_observable-tuple"],
    )
    def test_non_hermitian_observable_raises(self, plain_dag, assign):
        meta = MetaCircuit(
            circuit_bodies=(((), plain_dag),),
            observable=SparsePauliOp.from_list([("IZ", 1.0)]),
        )
        with pytest.raises(ValueError, match="Hermitian"):
            assign(meta, SparsePauliOp.from_list([("ZZ", 1.0j)]))

    def test_observable_is_checked_once_per_assignment_not_per_copy(
        self, plain_dag, mocker
    ):
        check = mocker.spy(_core, "_assert_hermitian_spo")
        obs = SparsePauliOp.from_list([("IZ", 1.0), ("ZI", -1.0)])
        meta = MetaCircuit(circuit_bodies=(((), plain_dag),), observable=obs)
        copied = meta.set_measurement_bodies(()).set_qasm_bodies(()).set_group_shots({})
        assert check.call_count == 1
        assert copied.observable is meta.observable
        copied.set_observable(obs)
        assert check.call_count == 2

    def test_set_observable_rechecks_the_observable_it_already_holds(self, plain_dag):
        meta = MetaCircuit(
            circuit_bodies=(((), plain_dag),),
            observable=SparsePauliOp.from_list([("IZ", 1.0)]),
        )
        meta.observable[0].coeffs[0] = 1.0j

        with pytest.raises(ValueError, match="Hermitian"):
            meta.set_observable(meta.observable)

    @pytest.mark.parametrize("as_tuple", [False, True], ids=["bare", "tuple"])
    def test_a_validated_observable_skips_the_check(self, plain_dag, as_tuple):
        non_hermitian = SparsePauliOp.from_list([("ZZ", 1.0j)])
        meta = MetaCircuit(circuit_bodies=(((), plain_dag),))

        updated = meta.set_observable(
            (non_hermitian,) if as_tuple else non_hermitian, validated=True
        )

        assert updated.observable == (non_hermitian,)

    def test_tuple_observable_requires_sparse_pauli_ops(self, plain_dag):
        with pytest.raises(TypeError) as exc_info:
            MetaCircuit(
                circuit_bodies=(((), plain_dag),),
                observable=("not an observable",),  # type: ignore[arg-type]
            )
        assert str(exc_info.value) == (
            "MetaCircuit.observable must be a SparsePauliOp or a tuple of "
            "SparsePauliOp instances."
        )

    def test_empty_bodies_raises(self):
        with pytest.raises(ValueError) as exc_info:
            MetaCircuit(circuit_bodies=())
        assert str(exc_info.value) == "MetaCircuit requires at least one circuit body."

    def test_set_circuit_bodies_returns_new_instance(self, plain_dag, parametric_dag):
        orig = MetaCircuit(circuit_bodies=(((), plain_dag),))
        new_dag, params = parametric_dag
        bodies = (((("qem", 0),), new_dag),)
        out = orig.set_circuit_bodies(bodies)
        assert out is not orig
        assert out.circuit_bodies == bodies
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
