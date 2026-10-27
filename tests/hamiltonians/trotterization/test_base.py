# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for TrotterizationResult synthesis and the validation every strategy shares."""

import itertools

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm

from divi.circuits._conversions import _QISKIT_TO_QASM2
from divi.hamiltonians.trotterization import ExactTrotterization, QDrift, TrotterSpec
from tests._helpers import equal_up_to_phase, exact_match, unitary
from tests.hamiltonians.trotterization._helpers import (
    SAMPLING_BUDGET_NOT_SET,
    SIMPLE_HAMILTONIAN,
)


@pytest.mark.parametrize(
    "strategy",
    [
        ExactTrotterization(keep_fraction=0.5),
        ExactTrotterization(keep_fraction=1.0),
        ExactTrotterization(keep_top_n=1),
        QDrift(keep_fraction=0.5, sampling_budget=2, seed=0),
        QDrift(keep_fraction=1.0, sampling_budget=2, seed=0),
        QDrift(keep_top_n=1, sampling_budget=2, seed=0),
    ],
    ids=[
        "exact_partial_fraction",
        "exact_full_fraction",
        "exact_top_n_covers_all",
        "qdrift_partial_fraction",
        "qdrift_full_fraction",
        "qdrift_top_n_covers_all",
    ],
)
def test_truncating_a_constant_only_hamiltonian_raises(strategy):
    """Every truncation setting rejects a constant-only Hamiltonian."""
    with pytest.raises(
        ValueError, match=exact_match("Hamiltonian contains only constant terms.")
    ):
        strategy.process_hamiltonian(TrotterSpec(5.0 * SparsePauliOp("I")))


@pytest.mark.parametrize(
    "build, field, value",
    [
        (lambda v: ExactTrotterization(keep_top_n=v), "keep_top_n", np.int64(3)),
        (
            lambda v: ExactTrotterization(keep_fraction=v),
            "keep_fraction",
            np.float32(0.5),
        ),
        (
            lambda v: QDrift(sampling_budget=v, seed=0),
            "sampling_budget",
            np.int64(2),
        ),
        (lambda v: QDrift(sampling_budget=2, seed=v), "seed", np.uint64(2**63 + 5)),
        (
            lambda v: QDrift(sampling_budget=2, seed=0, n_hamiltonians_per_iteration=v),
            "n_hamiltonians_per_iteration",
            np.int64(4),
        ),
    ],
    ids=[
        "keep_top_n",
        "keep_fraction",
        "sampling_budget",
        "seed",
        "n_hamiltonians_per_iteration",
    ],
)
def test_strategies_accept_numpy_scalars_and_store_plain_python(build, field, value):
    stored = getattr(build(value), field)

    assert stored == value
    assert type(stored) is (float if field == "keep_fraction" else int)


@pytest.mark.parametrize(
    "build, message",
    [
        (
            lambda: ExactTrotterization(keep_top_n=True),
            "keep_top_n must be a positive integer (>= 1), got True",
        ),
        (
            lambda: ExactTrotterization(keep_fraction=float("nan")),
            "keep_fraction must be in (0, 1], got nan",
        ),
        (
            lambda: QDrift(sampling_budget=True),
            "sampling_budget must be a positive integer (>= 1), got True",
        ),
        (
            lambda: QDrift(sampling_budget=2, seed=True),
            "seed must be an integer, got True",
        ),
        (
            lambda: QDrift(sampling_budget=2, seed=-1),
            "seed must be non-negative, got -1",
        ),
        (
            lambda: QDrift(sampling_budget=2, n_hamiltonians_per_iteration=True),
            "n_hamiltonians_per_iteration must be an integer, got True",
        ),
        (
            lambda: QDrift(sampling_budget=2, n_hamiltonians_per_iteration=2.5),
            "n_hamiltonians_per_iteration must be an integer, got 2.5",
        ),
    ],
    ids=[
        "keep_top_n_bool",
        "keep_fraction_nan",
        "sampling_budget_bool",
        "seed_bool",
        "seed_negative",
        "n_hamiltonians_bool",
        "n_hamiltonians_float",
    ],
)
def test_strategies_reject_invalid_scalars(build, message):
    with pytest.raises(ValueError, match=exact_match(message)):
        build()


# Y-carrying terms make the sign of t observable; YY and ZZ commute, so every
# product formula is exact and the circuit must equal exp(-i t H) exactly.
_COMMUTING_YY_ZZ = SparsePauliOp(["YY", "ZZ"], [0.8, 0.3])
_SYNTHESIS_TIME = 0.7


def _synthesize(
    result, *, n_steps: int, order: int, n_qubits: int, time: float = _SYNTHESIS_TIME
) -> QuantumCircuit:
    return result.synthesize_evolution(
        QuantumCircuit(n_qubits),
        time=time,
        n_steps=n_steps,
        order=order,
        qubits=list(range(n_qubits)),
        basis_gates=list(_QISKIT_TO_QASM2),
    )


def _evolution(generator: SparsePauliOp, time: float = _SYNTHESIS_TIME) -> np.ndarray:
    return expm(-1j * time * generator.to_matrix())


@pytest.mark.parametrize(
    "hamiltonian",
    [SparsePauliOp("Y", 0.8), _COMMUTING_YY_ZZ],
    ids=["single-term", "multi-term"],
)
@pytest.mark.parametrize("order", [1, 2])
def test_synthesized_evolution_is_exp_minus_i_t_h(
    default_test_simulator, hamiltonian, order
):
    result = ExactTrotterization().process_hamiltonian(
        TrotterSpec(hamiltonian), n_steps=2
    )
    qc = _synthesize(result, n_steps=2, order=order, n_qubits=hamiltonian.num_qubits)
    assert equal_up_to_phase(
        unitary(default_test_simulator, qc), _evolution(hamiltonian), atol=1e-8
    )


def test_sampled_synthesis_applies_each_step_draw(default_test_simulator):
    """Seed 3 draws different terms per step, so replaying one draw would fail."""
    n_steps = 2
    result = QDrift(sampling_budget=4, seed=3).process_hamiltonian(
        TrotterSpec(_COMMUTING_YY_ZZ), n_steps=n_steps
    )
    assert result.sampled_terms is not None
    assert result.sampled_terms[0] != result.sampled_terms[1]
    qc = _synthesize(result, n_steps=n_steps, order=1, n_qubits=2)
    assert equal_up_to_phase(
        unitary(default_test_simulator, qc),
        _evolution(sum(result.sampled_terms) / n_steps),
        atol=1e-8,
    )


def test_qdrift_without_sampling_budget_synthesizes_deterministically(
    default_test_simulator,
):
    with pytest.warns(UserWarning, match=SAMPLING_BUDGET_NOT_SET):
        strategy = QDrift(keep_top_n=1)
    result = strategy.process_hamiltonian(TrotterSpec(_COMMUTING_YY_ZZ), n_steps=2)
    assert result.sampled_terms is None
    qc = _synthesize(result, n_steps=2, order=2, n_qubits=2)
    assert equal_up_to_phase(
        unitary(default_test_simulator, qc),
        _evolution(SparsePauliOp("YY", 0.8)),
        atol=1e-8,
    )


def test_sampled_synthesis_rejects_higher_order():
    result = QDrift(sampling_budget=2, seed=0).process_hamiltonian(
        TrotterSpec(SIMPLE_HAMILTONIAN), n_steps=3
    )
    with pytest.raises(
        ValueError,
        match=exact_match("Sampled Trotterization supports only order=1, got order=2."),
    ):
        _synthesize(result, n_steps=3, order=2, n_qubits=2)


@pytest.mark.parametrize(
    "pauli",
    ["".join(p) for k in (1, 2, 3, 4) for p in itertools.product("XYZ", repeat=k)],
)
def test_every_pauli_synthesizes_into_the_qasm_basis(default_test_simulator, pauli):
    hamiltonian = SparsePauliOp(pauli, 0.4)
    qc = _synthesize(
        ExactTrotterization().process_hamiltonian(TrotterSpec(hamiltonian)),
        n_steps=1,
        order=1,
        n_qubits=len(pauli),
    )
    assert {inst.operation.name for inst in qc.data} <= set(_QISKIT_TO_QASM2)
    assert equal_up_to_phase(
        unitary(default_test_simulator, qc), _evolution(hamiltonian), atol=1e-10
    )


_NON_COMMUTING = SparsePauliOp(["IX", "ZZ", "XI", "YY"], [0.7, 0.4, -0.5, 0.3])
_NON_COMMUTING_TERMS = list(
    zip(_NON_COMMUTING.paulis.to_labels(), _NON_COMMUTING.coeffs.real)
)
_EVOLUTION_TIME = 0.9


def _synthesized(simulator, order: int, n_steps: int) -> np.ndarray:
    return unitary(
        simulator,
        _synthesize(
            ExactTrotterization().process_hamiltonian(TrotterSpec(_NON_COMMUTING)),
            n_steps=n_steps,
            order=order,
            n_qubits=2,
            time=_EVOLUTION_TIME,
        ),
    )


def _ordered_product(sequence, n_steps: int) -> np.ndarray:
    """``n_steps`` repetitions of ``exp(-i f dt c P)`` applied in ``sequence`` order."""
    dt = _EVOLUTION_TIME / n_steps
    step = np.eye(4, dtype=complex)
    for label, coeff, fraction in sequence:
        generator = coeff * SparsePauliOp(label).to_matrix()
        step = expm(-1j * fraction * dt * generator) @ step
    return np.linalg.matrix_power(step, n_steps)


@pytest.mark.parametrize(
    "order, sequence",
    [
        (1, [(label, coeff, 1.0) for label, coeff in _NON_COMMUTING_TERMS]),
        (
            2,
            [(label, coeff, 0.5) for label, coeff in _NON_COMMUTING_TERMS]
            + [(label, coeff, 0.5) for label, coeff in _NON_COMMUTING_TERMS[::-1]],
        ),
    ],
    ids=["lie_trotter", "suzuki_second_order"],
)
def test_product_formula_applies_terms_in_input_order(
    default_test_simulator, order, sequence
):
    assert equal_up_to_phase(
        _synthesized(default_test_simulator, order, n_steps=3),
        _ordered_product(sequence, n_steps=3),
        atol=1e-8,
    )


def test_fourth_order_formula_is_more_accurate_than_second_order(
    default_test_simulator,
):
    exact = _evolution(_NON_COMMUTING, _EVOLUTION_TIME)
    errors = {
        order: np.linalg.norm(
            _synthesized(default_test_simulator, order, n_steps=3) - exact
        )
        for order in (2, 4)
    }
    assert errors[4] < errors[2] / 10
