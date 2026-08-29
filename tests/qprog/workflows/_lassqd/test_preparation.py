# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for paper-faithful LASSQD fragment-circuit preparation."""

from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Statevector

ffsim = pytest.importorskip("ffsim")

from divi.qprog.workflows._lassqd import _preparation as preparation
from divi.qprog.workflows._lassqd._preparation import (
    LinearMethodFragmentProgram,
    LUCJPreparation,
    _fragment_ccsd,
    _fragment_rohf,
    build_lucj_circuit,
    paper_lucj_interaction_pairs,
    prepare_lucj_fragment,
    rotate_rdms_to_fragment_basis,
)
from divi.qprog.workflows._lassqd._state import FragmentSpec


def test_paper_lucj_interaction_pairs_match_the_reference_topology():
    assert paper_lucj_interaction_pairs(6) == (
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        [(0, 0), (4, 4)],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
    )


def test_lucj_circuit_maps_grouped_spin_gates_to_interleaved_wires():
    norb = 3
    nelec = (2, 0)
    pairs = paper_lucj_interaction_pairs(norb)
    n_params = ffsim.UCJOpSpinUnbalanced.n_params(
        norb,
        1,
        interaction_pairs=pairs,
        with_final_orbital_rotation=True,
    )
    operator = ffsim.UCJOpSpinUnbalanced.from_parameters(
        np.zeros(n_params),
        norb=norb,
        n_reps=1,
        interaction_pairs=pairs,
        with_final_orbital_rotation=True,
    )

    circuit = build_lucj_circuit(operator, norb, nelec)

    unmeasured = circuit.remove_final_measurements(inplace=False)
    probabilities = Statevector.from_instruction(unmeasured).probabilities_dict()
    assert probabilities == {"000101": 1.0}
    assert {instruction.operation.name for instruction in circuit.data} <= {
        "cx",
        "measure",
        "u",
        "x",
    }


def test_prepare_lucj_fragment_uses_ccsd_seed_and_linear_method(mocker):
    h_alpha = np.diag([-1.0, 0.5])
    h_beta = np.diag([-0.8, 0.7])
    two_body = np.zeros((2, 2, 2, 2))
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)

    minimize = mocker.patch(
        "divi.qprog.workflows._lassqd._preparation.minimize_linear_method",
        side_effect=lambda _params_to_vec, _hamiltonian, x0: SimpleNamespace(x=x0),
    )
    from_amplitudes = mocker.spy(ffsim.UCJOpSpinUnbalanced, "from_t_amplitudes")

    result = prepare_lucj_fragment(h_alpha, h_beta, two_body, spec)

    assert from_amplitudes.call_count == 1
    call = from_amplitudes.call_args
    assert isinstance(call.args[0], tuple)
    assert isinstance(call.kwargs["t1"], tuple)
    assert call.kwargs["n_reps"] == 1
    assert call.kwargs["interaction_pairs"] == paper_lucj_interaction_pairs(2)
    assert call.kwargs["optimize"] is True
    assert minimize.call_count == 1
    np.testing.assert_allclose(result.params, minimize.call_args.kwargs["x0"])
    assert result.circuit.num_qubits == 4
    assert result.orbital_rotation.shape == (2, 2)
    assert result.h_alpha.shape == (2, 2)
    assert result.h_beta.shape == (2, 2)
    np.testing.assert_allclose(result.h_beta, np.diag([-0.8, 0.7]))
    assert result.two_body.shape == (2, 2, 2, 2)


def test_prepare_lucj_fragment_refits_nonzero_seed_to_the_paper_topology(mocker):
    rng = np.random.default_rng(17)
    t1 = (
        rng.normal(scale=0.05, size=(2, 2)),
        rng.normal(scale=0.05, size=(2, 2)),
    )
    t2 = (
        rng.normal(scale=0.05, size=(2, 2, 2, 2)),
        rng.normal(scale=0.05, size=(2, 2, 2, 2)),
        rng.normal(scale=0.05, size=(2, 2, 2, 2)),
    )
    spec = FragmentSpec(orbitals=(0, 1, 2, 3), n_alpha=2, n_beta=2)
    pairs = paper_lucj_interaction_pairs(4)
    dense_masked = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
        t2,
        t1=t1,
        n_reps=1,
    ).to_parameters(interaction_pairs=pairs)
    mocker.patch(
        "divi.qprog.workflows._lassqd._preparation._fragment_rohf",
        return_value=SimpleNamespace(mo_coeff=np.eye(4)),
    )
    mocker.patch(
        "divi.qprog.workflows._lassqd._preparation._fragment_ccsd",
        return_value=SimpleNamespace(t1=t1, t2=t2),
    )
    mocker.patch(
        "divi.qprog.workflows._lassqd._preparation.minimize_linear_method",
        side_effect=lambda _params_to_vec, _hamiltonian, x0: SimpleNamespace(x=x0),
    )

    result = prepare_lucj_fragment(
        np.zeros((4, 4)),
        np.zeros((4, 4)),
        np.zeros((4, 4, 4, 4)),
        spec,
    )

    assert np.max(np.abs(result.params - dense_masked)) > 1e-3


def test_prepare_lucj_fragment_rotates_real_beta_integrals_for_sqd(mocker):
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    mocker.patch(
        "divi.qprog.workflows._lassqd._preparation._fragment_rohf",
        return_value=SimpleNamespace(mo_coeff=rotation),
    )
    mocker.patch(
        "divi.qprog.workflows._lassqd._preparation._fragment_ccsd",
        return_value=SimpleNamespace(
            t1=(np.zeros((1, 1)), np.zeros((1, 1))),
            t2=(
                np.zeros((1, 1, 1, 1)),
                np.zeros((1, 1, 1, 1)),
                np.zeros((1, 1, 1, 1)),
            ),
        ),
    )
    mocker.patch(
        "divi.qprog.workflows._lassqd._preparation.minimize_linear_method",
        side_effect=lambda _params_to_vec, _hamiltonian, x0: SimpleNamespace(x=x0),
    )
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)

    result = prepare_lucj_fragment(
        np.diag([-1.0, 0.5]),
        np.diag([-0.8, 0.7]),
        np.zeros((2, 2, 2, 2)),
        spec,
    )

    np.testing.assert_allclose(result.h_beta, np.diag([0.7, -0.8]))


def test_fragment_rohf_retries_unconverged_scf_with_newton(mocker):
    direct_solver = mocker.Mock(converged=False)
    newton_solver = mocker.Mock(converged=True)
    direct_solver.newton.return_value = newton_solver
    rohf = mocker.patch("pyscf.scf.ROHF", return_value=direct_solver)
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)

    result = _fragment_rohf(
        np.diag([-1.0, 0.5]),
        np.zeros((2, 2, 2, 2)),
        spec,
    )

    rohf.assert_called_once()
    direct_solver.kernel.assert_called_once_with()
    direct_solver.newton.assert_called_once_with()
    newton_solver.kernel.assert_called_once_with()
    assert result is newton_solver


def test_fragment_rohf_raises_when_newton_solver_does_not_converge(mocker):
    direct_solver = mocker.Mock(converged=False)
    newton_solver = mocker.Mock(converged=False)
    direct_solver.newton.return_value = newton_solver
    mocker.patch("pyscf.scf.ROHF", return_value=direct_solver)
    spec = FragmentSpec(orbitals=(3, 4), n_alpha=1, n_beta=1)

    with pytest.raises(RuntimeError, match=r"ROHF did not converge.*\(3, 4\)"):
        _fragment_rohf(
            np.diag([-1.0, 0.5]),
            np.zeros((2, 2, 2, 2)),
            spec,
        )


def test_fragment_rohf_uses_positive_local_spin_for_beta_majority(mocker):
    solver = mocker.Mock(converged=True)
    rohf = mocker.patch("pyscf.scf.ROHF", return_value=solver)
    spec = FragmentSpec(orbitals=(0, 1, 2), n_alpha=1, n_beta=2)

    _fragment_rohf(
        np.diag([-1.0, -0.5, 0.5]),
        np.zeros((3, 3, 3, 3)),
        spec,
    )

    assert rohf.call_args.args[0].spin == 1


def test_beta_majority_ccsd_amplitudes_are_relabelled_to_physical_spin_channels():
    t1_majority = np.array([[11.0], [12.0]])
    t1_minority = np.array([[21.0, 22.0]])
    t2_majority = np.arange(4.0).reshape(2, 2, 1, 1) + 100.0
    t2_mixed = np.arange(4.0).reshape(2, 1, 1, 2) + 200.0
    t2_minority = np.arange(4.0).reshape(1, 1, 2, 2) + 300.0
    coupled_cluster = SimpleNamespace(
        t1=(t1_majority, t1_minority),
        t2=(t2_majority, t2_mixed, t2_minority),
    )
    spec = FragmentSpec(orbitals=(0, 1, 2), n_alpha=1, n_beta=2)

    t1, t2 = preparation._physical_spin_amplitudes(coupled_cluster, spec)

    np.testing.assert_array_equal(t1[0], t1_minority)
    np.testing.assert_array_equal(t1[1], t1_majority)
    np.testing.assert_array_equal(t2[0], t2_minority)
    np.testing.assert_array_equal(t2[1], t2_mixed.transpose(1, 0, 3, 2))
    np.testing.assert_array_equal(t2[2], t2_majority)


def test_non_finite_ccsd_amplitudes_fail_before_factorization(mocker):
    spec = FragmentSpec(orbitals=(3, 4), n_alpha=1, n_beta=1)
    mocker.patch(
        "divi.qprog.workflows._lassqd._preparation._fragment_rohf",
        return_value=SimpleNamespace(mo_coeff=np.eye(2)),
    )
    mocker.patch(
        "divi.qprog.workflows._lassqd._preparation._fragment_ccsd",
        return_value=SimpleNamespace(
            t1=(np.zeros((1, 1)), np.zeros((1, 1))),
            t2=(
                np.zeros((1, 1, 1, 1)),
                np.full((1, 1, 1, 1), np.nan),
                np.zeros((1, 1, 1, 1)),
            ),
        ),
    )

    with pytest.raises(RuntimeError, match=r"fragment \(3, 4\).*non-finite CCSD"):
        prepare_lucj_fragment(
            np.diag([-1.0, 0.5]),
            np.diag([-0.8, 0.7]),
            np.zeros((2, 2, 2, 2)),
            spec,
        )


def test_non_finite_linear_method_result_fails_before_circuit_construction(mocker):
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)
    mocker.patch(
        "divi.qprog.workflows._lassqd._preparation.minimize_linear_method",
        side_effect=lambda _params_to_vec, _hamiltonian, x0: SimpleNamespace(
            x=np.full_like(x0, np.nan)
        ),
    )

    with pytest.raises(
        RuntimeError, match=r"fragment \(0, 1\).*non-finite linear-method"
    ):
        prepare_lucj_fragment(
            np.diag([-1.0, 0.5]),
            np.diag([-0.8, 0.7]),
            np.zeros((2, 2, 2, 2)),
            spec,
        )


def test_fragment_ccsd_warns_and_keeps_best_unconverged_amplitudes(mocker):
    coupled_cluster = mocker.Mock(converged=False)
    ccsd = mocker.patch("pyscf.cc.CCSD", return_value=coupled_cluster)
    mean_field = mocker.Mock()
    spec = FragmentSpec(orbitals=(3, 4), n_alpha=1, n_beta=1)

    with pytest.warns(UserWarning, match="best available amplitudes"):
        result = _fragment_ccsd(mean_field, spec)

    ccsd.assert_called_once_with(mean_field)
    assert coupled_cluster.max_cycle == 500
    coupled_cluster.kernel.assert_called_once_with()
    assert result is coupled_cluster


def test_linear_method_program_prepares_classically_then_samples_once(
    dummy_simulator, mocker
):
    circuit = QuantumCircuit(4)
    circuit.x(0)
    classical_bits = ClassicalRegister(4)
    circuit.add_register(classical_bits)
    for index, qubit in enumerate(circuit.qubits):
        circuit.measure(qubit, classical_bits[index])
    preparation = LUCJPreparation(
        circuit=circuit,
        params=np.array([0.1, -0.2]),
        h_alpha=np.eye(2),
        h_beta=2 * np.eye(2),
        two_body=np.zeros((2, 2, 2, 2)),
        orbital_rotation=np.eye(2),
    )
    prepare = mocker.patch(
        "divi.qprog.workflows._lassqd._preparation.prepare_lucj_fragment",
        return_value=preparation,
    )
    submit = mocker.spy(dummy_simulator, "submit_circuits")
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)
    program = LinearMethodFragmentProgram(
        np.eye(2),
        np.eye(2),
        np.zeros((2, 2, 2, 2)),
        spec,
        backend=dummy_simulator,
        program_id="fragment_0",
        seed=7,
    )

    returned = program.run()

    assert returned is program
    prepare.assert_called_once()
    assert submit.call_count == 1
    assert program.total_circuit_count == 1
    assert program.has_results()
    assert set(program.best_probs) == {0}
    np.testing.assert_allclose(program.best_params, preparation.params)
    np.testing.assert_allclose(program.h_alpha, preparation.h_alpha)
    np.testing.assert_allclose(program.h_beta, preparation.h_beta)
    np.testing.assert_allclose(program.two_body, preparation.two_body)
    np.testing.assert_allclose(program.orbital_rotation, preparation.orbital_rotation)


def test_rotates_sqd_rdms_back_to_the_workflow_fragment_basis():
    angle = 0.37
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    rdm1_alpha = np.array([[0.8, 0.1], [0.1, 0.2]])
    rdm1_beta = np.array([[0.3, -0.05], [-0.05, 0.7]])
    rdm1 = rdm1_alpha + rdm1_beta
    rdm2 = np.zeros((2, 2, 2, 2))
    rdm2[0, 1, 0, 1] = 2.5

    actual = rotate_rdms_to_fragment_basis(rdm1, rdm2, rdm1_alpha, rdm1_beta, rotation)

    cosine_squared = cosine**2
    sine_squared = sine**2
    sine_cosine = sine * cosine
    expected_rdm1_alpha = np.array(
        [
            [
                0.8 * cosine_squared - 0.2 * sine_cosine + 0.2 * sine_squared,
                0.6 * sine_cosine + 0.1 * (cosine_squared - sine_squared),
            ],
            [
                0.6 * sine_cosine + 0.1 * (cosine_squared - sine_squared),
                0.8 * sine_squared + 0.2 * sine_cosine + 0.2 * cosine_squared,
            ],
        ]
    )
    expected_rdm1_beta = np.array(
        [
            [
                0.3 * cosine_squared + 0.1 * sine_cosine + 0.7 * sine_squared,
                -0.4 * sine_cosine - 0.05 * (cosine_squared - sine_squared),
            ],
            [
                -0.4 * sine_cosine - 0.05 * (cosine_squared - sine_squared),
                0.3 * sine_squared - 0.1 * sine_cosine + 0.7 * cosine_squared,
            ],
        ]
    )
    expected_rdm1 = expected_rdm1_alpha + expected_rdm1_beta
    transformed_pair = np.array(
        [
            [-sine_cosine, cosine_squared],
            [-sine_squared, sine_cosine],
        ]
    )
    expected_rdm2 = 2.5 * np.einsum("ij,kl->ijkl", transformed_pair, transformed_pair)
    for received, expected in zip(
        actual,
        (expected_rdm1, expected_rdm2, expected_rdm1_alpha, expected_rdm1_beta),
    ):
        np.testing.assert_allclose(received, expected)
