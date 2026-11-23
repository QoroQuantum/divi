# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import U2Gate
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


_THETA = Parameter("theta")
_PHI = Parameter("phi")


def _meta_of(*circuits, parameters):
    """A MetaCircuit with one body per circuit."""
    return MetaCircuit(
        circuit_bodies=tuple(
            ((("body", index),), circuit_to_dag(circuit))
            for index, circuit in enumerate(circuits)
        ),
        parameters=tuple(parameters),
    )


#: Gates applying ``_THETA`` and the ``(omega, order)`` they give it.
_FREQUENCY_CASES = {
    "bare-rotation": (lambda qc: qc.rx(_THETA, 0), (1.0, 1)),
    "half-angle": (lambda qc: qc.rz(_THETA / 2, 0), (0.5, 1)),
    "controlled-rotation": (lambda qc: qc.crz(_THETA, 0, 1), (0.5, 2)),
    "controlled-x-rotation": (lambda qc: qc.crx(_THETA, 0, 1), (0.5, 2)),
    "controlled-y-rotation": (lambda qc: qc.cry(2 * _THETA, 0, 1), (1.0, 2)),
    "xx-rotation": (lambda qc: qc.rxx(_THETA, 0, 1), (1.0, 1)),
    "yy-rotation": (lambda qc: qc.ryy(_THETA, 0, 1), (1.0, 1)),
    "zx-rotation": (lambda qc: qc.rzx(_THETA, 0, 1), (1.0, 1)),
    "controlled-phase": (lambda qc: qc.cp(_THETA, 0, 1), (1.0, 1)),
    "u2-gate-angle": (lambda qc: qc.append(U2Gate(_THETA, 0.4), [0]), (1.0, 1)),
    "shared-symbol": (lambda qc: (qc.rx(_THETA, 0), qc.ry(2 * _THETA, 1)), (1.0, 3)),
    "constant-offset": (lambda qc: qc.rx(_THETA + 0.3, 0), (1.0, 1)),
    "u-gate-angle": (lambda qc: qc.u(0.1, _THETA, 0.2, 0), (1.0, 1)),
    "two-qubit-rotation": (lambda qc: qc.rzz(1.5 * _THETA, 0, 1), (1.5, 1)),
    "phase-gate": (lambda qc: qc.p(_THETA, 0), (1.0, 1)),
    "non-linear-angle": (lambda qc: qc.rx(_THETA * _THETA, 0), None),
    "incommensurate": (
        lambda qc: (qc.rx(_THETA, 0), qc.ry(np.pi * _THETA, 1)),
        None,
    ),
    "unsupported-gate": (lambda qc: qc.cu(_THETA, 0.0, 0.0, 0.0, 0, 1), None),
    "absent": (lambda qc: qc.h(0), (1.0, 1)),
}


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

    @pytest.mark.parametrize(
        ("build", "expected"),
        _FREQUENCY_CASES.values(),
        ids=_FREQUENCY_CASES.keys(),
    )
    def test_parameter_frequencies_follow_the_gates(self, build, expected):
        circuit = QuantumCircuit(2)
        build(circuit)

        assert _meta_of(circuit, parameters=(_THETA,)).parameter_frequencies == (
            expected,
        )

    def test_parameter_frequencies_are_ordered_like_the_parameters(self):
        circuit = QuantumCircuit(1)
        circuit.rx(_PHI / 2, 0)
        circuit.ry(_THETA, 0)

        meta = _meta_of(circuit, parameters=(_THETA, _PHI))

        assert meta.parameter_frequencies == ((1.0, 1), (0.5, 1))

    @pytest.mark.parametrize(
        ("other_angle", "expected"),
        [
            (lambda: (_THETA, -_THETA, _THETA), (1.0, 3)),
            (lambda: (_THETA / 2,), (0.5, 2)),
        ],
        ids=["folded", "different-base-frequency"],
    )
    def test_parameter_frequencies_cover_every_body(self, other_angle, expected):
        plain = QuantumCircuit(1)
        plain.rx(_THETA, 0)
        other = QuantumCircuit(1)
        for angle in other_angle():
            other.rx(angle, 0)

        meta = _meta_of(plain, other, parameters=(_THETA,))

        assert meta.parameter_frequencies == (expected,)

    def test_a_copy_with_new_bodies_derives_its_own_frequencies(self):
        first = QuantumCircuit(1)
        first.rx(_THETA, 0)
        meta = _meta_of(first, parameters=(_THETA,))
        assert meta.parameter_frequencies == ((1.0, 1),)
        second = QuantumCircuit(1)
        second.rx(2 * _THETA, 0)

        body = (("body", 0),), circuit_to_dag(second)
        copied = meta.set_circuit_bodies((body,))

        assert copied.parameter_frequencies == ((2.0, 1),)


@pytest.mark.parametrize(
    ("values", "expected"),
    [([0.74, 0.84, 1.16], 0.02), ([1e-6], 1e-6), ([2e-5, 4e-5], 2e-5)],
    ids=["hundredths", "tiny", "tiny-pair"],
)
def test_common_frequency_is_the_largest_shared_divisor(values, expected):
    assert _core._compute_common_frequency(values) == pytest.approx(expected)


@pytest.mark.parametrize(
    "values",
    [[0.5, np.sqrt(2)], [1.0, np.pi], [1.0, 1 / 10001]],
    ids=["irrational-ratio", "pi-multiple", "denominator-past-the-limit"],
)
def test_incommensurate_frequencies_have_no_common_frequency(values):
    assert _core._compute_common_frequency(values) is None
