# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for paper-faithful LASSQD fragment-circuit preparation."""

import hashlib
import itertools
import tracemalloc
from operator import attrgetter, methodcaller
from pathlib import Path

import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Statevector

ffsim = pytest.importorskip("ffsim")

from divi.qprog.problems import MolecularProblem
from divi.qprog.workflows._lassqd import _preparation as preparation
from divi.qprog.workflows._lassqd._preparation import (
    LUCJFragmentProgram,
    LUCJPreparation,
    _fragment_ccsd,
    _fragment_rohf,
    build_lucj_circuit,
    paper_lucj_interaction_pairs,
    prepare_lucj_fragment,
    rotate_rdms_to_fragment_basis,
)
from divi.qprog.workflows._lassqd._state import FragmentSpec
from tests._helpers import exact_match

_PREPARATION = "divi.qprog.workflows._lassqd._preparation"


def _patch_linear_method(mocker, optimum):
    """Stub the linear method to return ``optimum(x0)`` as its parameters."""
    return mocker.patch(
        "ffsim.optimize.minimize_linear_method",
        side_effect=lambda _params_to_vec, _hamiltonian, x0, **_options: mocker.Mock(
            x=optimum(x0)
        ),
    )


def _patch_mean_field_and_ccsd(mocker, mo_coeff, t1, t2):
    """Stub the fragment ROHF and CCSD solves with fixed results."""
    mocker.patch(
        f"{_PREPARATION}._fragment_rohf", return_value=mocker.Mock(mo_coeff=mo_coeff)
    )
    mocker.patch(
        f"{_PREPARATION}._fragment_ccsd", return_value=mocker.Mock(t1=t1, t2=t2)
    )


def _one_pair_amplitudes(mixed_double=0.0):
    """CCSD amplitudes for one occupied and one virtual orbital per spin."""
    t1 = (np.zeros((1, 1)), np.zeros((1, 1)))
    t2 = (
        np.zeros((1, 1, 1, 1)),
        np.full((1, 1, 1, 1), mixed_double),
        np.zeros((1, 1, 1, 1)),
    )
    return t1, t2


def test_paper_lucj_interaction_pairs_match_the_reference_topology():
    assert paper_lucj_interaction_pairs(6) == (
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        [(0, 0), (4, 4)],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
    )


def _identity_lucj_operator(norb):
    """The paper-topology LUCJ operator at zero parameters, i.e. the identity."""
    pairs = paper_lucj_interaction_pairs(norb)
    n_params = ffsim.UCJOpSpinUnbalanced.n_params(
        norb,
        1,
        interaction_pairs=pairs,
        with_final_orbital_rotation=True,
    )
    return ffsim.UCJOpSpinUnbalanced.from_parameters(
        np.zeros(n_params),
        norb=norb,
        n_reps=1,
        interaction_pairs=pairs,
        with_final_orbital_rotation=True,
    )


def _reference_probabilities(circuit):
    unmeasured = circuit.remove_final_measurements(inplace=False)
    return Statevector.from_instruction(unmeasured).probabilities_dict()


@pytest.mark.parametrize(
    "nelec, expected",
    [
        pytest.param((2, 0), "000101", id="alpha-only"),
        pytest.param((2, 1), "000111", id="beta-on-odd-wires"),
    ],
)
def test_lucj_circuit_maps_grouped_spin_gates_to_interleaved_wires(nelec, expected):
    """Qubit ``2p`` holds orbital ``p``'s alpha electron and ``2p + 1`` its beta
    one; the key is little-endian, so qubit 0 is the rightmost bit."""
    norb = 3

    circuit = build_lucj_circuit(_identity_lucj_operator(norb), norb, nelec)

    assert _reference_probabilities(circuit) == {expected: 1.0}
    assert circuit.num_clbits == 2 * norb
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

    minimize = _patch_linear_method(mocker, lambda x0: x0)
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


def test_prepare_lucj_fragment_samples_the_ccsd_seed_without_the_linear_method(
    mocker,
):
    minimize = _patch_linear_method(mocker, lambda x0: x0 + 1.0)
    seed = mocker.spy(ffsim.UCJOpSpinUnbalanced, "to_parameters")
    stages = []

    result = prepare_lucj_fragment(
        np.diag([-1.0, 0.5]),
        np.diag([-0.8, 0.7]),
        np.zeros((2, 2, 2, 2)),
        FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
        report=stages.append,
        run_linear_method=False,
    )

    minimize.assert_not_called()
    np.testing.assert_array_equal(result.params, seed.spy_return)
    assert stages == ["Fragment ROHF", "CCSD seed"]


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
    _patch_mean_field_and_ccsd(mocker, np.eye(4), t1, t2)
    _patch_linear_method(mocker, lambda x0: x0)

    result = prepare_lucj_fragment(
        np.zeros((4, 4)),
        np.zeros((4, 4)),
        np.zeros((4, 4, 4, 4)),
        spec,
    )

    assert np.max(np.abs(result.params - dense_masked)) > 1e-3


def test_prepare_lucj_fragment_rotates_real_beta_integrals_for_sqd(mocker):
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    _patch_mean_field_and_ccsd(mocker, rotation, *_one_pair_amplitudes())
    _patch_linear_method(mocker, lambda x0: x0)
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)

    result = prepare_lucj_fragment(
        np.diag([-1.0, 0.5]),
        np.diag([-0.8, 0.7]),
        np.zeros((2, 2, 2, 2)),
        spec,
    )

    np.testing.assert_allclose(result.h_beta, np.diag([0.7, -0.8]))


def test_fragment_rohf_reaches_the_ground_state_of_a_spin_polarised_iron_fragment():
    """An open-shell iron fragment's ROHF reaches its stable ground state."""
    data = np.load(Path(__file__).parent / "data" / "fefe_fragment_round1.npz")
    n_alpha, n_beta = (int(count) for count in data["nelec"])
    spec = FragmentSpec(orbitals=tuple(range(5)), n_alpha=n_alpha, n_beta=n_beta)

    mean_field = _fragment_rohf(data["one_body"], data["two_body"], spec)

    _, stable = preparation.stability.rohf_internal(
        mean_field, with_symmetry=False, return_status=True
    )
    assert stable
    assert mean_field.e_tot == pytest.approx(-18.158417, abs=1e-6)


_ONE_START = np.array([np.diag([1.0, 0.0]), np.diag([1.0, 0.0])])


def _report_stable(mean_field, **_):
    return mean_field.mo_coeff, True


def _patch_rohf_starts(mocker, starts=(_ONE_START,), stability=_report_stable):
    """Run the fragment ROHF from ``starts``, its stability check answered by
    ``stability`` (every solution stable by default)."""
    mocker.patch(f"{_PREPARATION}._rohf_starts", return_value=list(starts))
    return mocker.patch(
        f"{_PREPARATION}.stability.rohf_internal", side_effect=stability
    )


@pytest.mark.parametrize("newton_converges", [True, False])
def test_fragment_rohf_retries_unconverged_scf_with_newton(mocker, newton_converges):
    direct_solver = mocker.Mock(converged=False)
    newton_solver = mocker.Mock(converged=newton_converges)
    direct_solver.newton.return_value = newton_solver
    rohf = mocker.patch("pyscf.scf.ROHF", return_value=direct_solver)
    _patch_rohf_starts(mocker)
    spec = FragmentSpec(orbitals=(3, 4), n_alpha=1, n_beta=1)

    def solve():
        return _fragment_rohf(np.diag([-1.0, 0.5]), np.zeros((2, 2, 2, 2)), spec)

    if newton_converges:
        assert solve() is newton_solver
    else:
        with pytest.raises(RuntimeError, match=r"ROHF did not converge.*\(3, 4\)"):
            solve()

    rohf.assert_called_once()
    direct_solver.kernel.assert_called_once()
    assert direct_solver.kernel.call_args.kwargs["dm0"] is _ONE_START
    direct_solver.newton.assert_called_once_with()
    newton_solver.kernel.assert_called_once_with()


def test_rohf_starts_cover_every_representable_occupation_aufbau_first():
    spec = FragmentSpec(orbitals=(0, 1, 2), n_alpha=1, n_beta=2)

    starts = preparation._rohf_starts(np.diag([-1.0, -0.5, 0.5]), spec)

    # Two singly-or-doubly occupied orbitals of three, one of them doubly.
    assert len(starts) == 3 * 2
    np.testing.assert_allclose(
        np.abs(starts[0]), [np.diag([1.0, 1.0, 0.0]), np.diag([1.0, 0.0, 0.0])]
    )


def test_rohf_starts_are_a_fixed_sample_once_the_cap_binds():
    spec = FragmentSpec(orbitals=tuple(range(10)), n_alpha=6, n_beta=2)
    one_body = np.diag(np.arange(10.0))

    starts = preparation._rohf_starts(one_body, spec)

    assert len(starts) == preparation._ROHF_STARTS
    for again, start in zip(preparation._rohf_starts(one_body, spec), starts):
        np.testing.assert_array_equal(again, start)


def test_capped_rohf_starts_can_select_the_first_non_aufbau_occupation():
    spec = FragmentSpec(orbitals=tuple(range(7)), n_alpha=3, n_beta=1)
    occupations = (
        (singles, doubles)
        for singles in itertools.combinations(range(7), 3)
        for doubles in itertools.combinations(singles, 1)
    )
    next(occupations)
    singles, doubles = next(occupations)
    expected = np.array(
        [
            np.diag(np.isin(range(7), occupied).astype(float))
            for occupied in (singles, doubles)
        ]
    )

    starts = preparation._rohf_starts(np.diag(np.arange(7.0)), spec)

    assert any(np.array_equal(start, expected) for start in starts)


def test_capped_rohf_starts_do_not_store_every_occupation():
    spec = FragmentSpec(orbitals=tuple(range(14)), n_alpha=7, n_beta=3)
    tracemalloc.start()
    try:
        starts = preparation._rohf_starts(np.diag(np.arange(14.0)), spec)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(starts) == preparation._ROHF_STARTS
    assert peak < 8_000_000


def test_a_capped_rohf_sample_is_distinct_and_keeps_aufbau_first():
    """90 occupations exceed the cap, so the sample must still lead with the
    aufbau occupation and never repeat one."""
    spec = FragmentSpec(orbitals=tuple(range(6)), n_alpha=4, n_beta=2)

    starts = preparation._rohf_starts(np.diag(np.arange(6.0)), spec)

    assert len(starts) == preparation._ROHF_STARTS
    np.testing.assert_allclose(
        np.abs(starts[0]),
        [np.diag([1.0] * 4 + [0.0] * 2), np.diag([1.0] * 2 + [0.0] * 4)],
    )
    for first, second in itertools.combinations(starts, 2):
        assert not np.allclose(first, second)


def test_fragment_rohf_keeps_the_lowest_converged_solution(mocker):
    low = mocker.Mock(converged=True, e_tot=-2.0)
    high = mocker.Mock(converged=True, e_tot=-1.0)
    failed = mocker.Mock(converged=False)
    failed.newton.return_value = mocker.Mock(converged=False)
    mocker.patch("pyscf.scf.ROHF", side_effect=[high, failed, low])
    _patch_rohf_starts(mocker, starts=[_ONE_START] * 3)
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)

    assert _fragment_rohf(np.diag([-1.0, 0.5]), np.zeros((2, 2, 2, 2)), spec) is low


def test_fragment_rohf_reoptimises_from_an_unstable_solution(mocker):
    solver = mocker.Mock(converged=True, mo_occ=np.array([2.0, 0.0]))
    mocker.patch("pyscf.scf.ROHF", return_value=solver)
    rotated = np.array([[0.0, 1.0], [1.0, 0.0]])
    _patch_rohf_starts(mocker, stability=[(rotated, False), (rotated, True)])
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)

    _fragment_rohf(np.diag([-1.0, 0.5]), np.zeros((2, 2, 2, 2)), spec)

    assert solver.kernel.call_count == 2
    solver.make_rdm1.assert_called_once_with(rotated, solver.mo_occ)


@pytest.mark.parametrize("restart_converges", [False, True])
def test_fragment_rohf_rejects_an_unstable_or_unconverged_restart(
    mocker, restart_converges
):
    solver = mocker.Mock(converged=True, mo_occ=np.array([2.0, 0.0]))

    def kernel(*args, **kwargs):
        if solver.kernel.call_count > 1:
            solver.converged = restart_converges

    solver.kernel.side_effect = kernel
    mocker.patch("pyscf.scf.ROHF", return_value=solver)
    _patch_rohf_starts(mocker, stability=[(np.eye(2), False)] * 4)
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)

    with pytest.raises(RuntimeError, match="ROHF did not converge"):
        _fragment_rohf(np.diag([-1.0, 0.5]), np.zeros((2, 2, 2, 2)), spec)


def test_fragment_rohf_uses_positive_local_spin_for_beta_majority(mocker):
    solver = mocker.Mock(converged=True)
    rohf = mocker.patch("pyscf.scf.ROHF", return_value=solver)
    _patch_rohf_starts(mocker)
    spec = FragmentSpec(orbitals=(0, 1, 2), n_alpha=1, n_beta=2)

    _fragment_rohf(
        np.diag([-1.0, -0.5, 0.5]),
        np.zeros((3, 3, 3, 3)),
        spec,
    )

    assert rohf.call_args.args[0].spin == 1


def test_beta_majority_ccsd_amplitudes_are_relabelled_to_physical_spin_channels(
    mocker,
):
    t1_majority = np.array([[11.0], [12.0]])
    t1_minority = np.array([[21.0, 22.0]])
    t2_majority = np.arange(4.0).reshape(2, 2, 1, 1) + 100.0
    t2_mixed = np.arange(4.0).reshape(2, 1, 1, 2) + 200.0
    t2_minority = np.arange(4.0).reshape(1, 1, 2, 2) + 300.0
    coupled_cluster = mocker.Mock(
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
    _patch_mean_field_and_ccsd(mocker, np.eye(2), *_one_pair_amplitudes(np.nan))

    with pytest.raises(RuntimeError, match=r"fragment \(3, 4\).*non-finite CCSD"):
        prepare_lucj_fragment(
            np.diag([-1.0, 0.5]),
            np.diag([-0.8, 0.7]),
            np.zeros((2, 2, 2, 2)),
            spec,
        )


def test_non_finite_linear_method_result_fails_before_circuit_construction(mocker):
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)
    _patch_linear_method(mocker, lambda x0: np.full_like(x0, np.nan))

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
    message = (
        "CCSD seed did not converge for fragment (3, 4) after 500 cycles; using "
        "its best available amplitudes, as in the LASSQD reference implementation."
    )

    with pytest.warns(UserWarning, match=exact_match(message)):
        result = _fragment_ccsd(mean_field, spec)

    ccsd.assert_called_once_with(mean_field)
    assert coupled_cluster.max_cycle == 500
    coupled_cluster.kernel.assert_called_once_with()
    assert result is coupled_cluster


def _two_orbital_program(backend, **kwargs):
    problem = MolecularProblem(np.eye(2), np.zeros((2, 2, 2, 2)), n_alpha=1, n_beta=1)
    return LUCJFragmentProgram(
        problem,
        FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
        backend=backend,
        **kwargs,
    )


_PREPARED_FIELDS = ("h_alpha", "h_beta", "two_body", "orbital_rotation")


def _measured_x0_preparation():
    circuit = QuantumCircuit(4)
    circuit.x(0)
    classical_bits = ClassicalRegister(4)
    circuit.add_register(classical_bits)
    for index, qubit in enumerate(circuit.qubits):
        circuit.measure(qubit, classical_bits[index])
    return LUCJPreparation(
        circuit=circuit,
        params=np.array([0.1, -0.2]),
        h_alpha=np.eye(2),
        h_beta=2 * np.eye(2),
        two_body=np.zeros((2, 2, 2, 2)),
        orbital_rotation=np.eye(2),
    )


@pytest.fixture
def prepared_program(dummy_simulator, mocker):
    """A LUCJ fragment program run once against a patched classical preparation.

    Returns ``(program, preparation, prepare, submit)``.
    """
    preparation = _measured_x0_preparation()
    prepare = mocker.patch(
        f"{_PREPARATION}.prepare_lucj_fragment", return_value=preparation
    )
    submit = mocker.spy(dummy_simulator, "submit_circuits")
    program = _two_orbital_program(dummy_simulator, seed=7)
    assert program.run() is program
    return program, preparation, prepare, submit


def test_lucj_fragment_program_prepares_classically_then_samples_once(
    prepared_program,
):
    program, preparation, prepare, submit = prepared_program

    prepare.assert_called_once()
    assert submit.call_count == 1
    assert program.total_circuit_count == 1
    assert program.has_results()
    assert set(program.best_probs) == {0}
    np.testing.assert_allclose(program.best_params, preparation.params)
    for name in _PREPARED_FIELDS:
        np.testing.assert_allclose(getattr(program, name), getattr(preparation, name))


def test_lucj_fragment_program_rejects_an_unclaimed_run_keyword(
    dummy_simulator, mocker
):
    prepare = mocker.patch(f"{_PREPARATION}.prepare_lucj_fragment")
    program = _two_orbital_program(dummy_simulator)

    with pytest.raises(
        TypeError,
        match=exact_match(
            "LUCJFragmentProgram.run() got unexpected keyword argument(s): bogus."
        ),
    ):
        program.run(bogus=1)

    prepare.assert_not_called()


def test_lucj_fragment_checkpoint_restores_without_preparing_or_sampling(
    prepared_program, dummy_simulator, tmp_path
):
    program, _, prepare, submit = prepared_program
    checkpoint = program._make_checkpoint(tmp_path)
    restored = _two_orbital_program(dummy_simulator, seed=7)
    prepare.reset_mock()
    submit.reset_mock()

    assert restored._restore_checkpoint(checkpoint.model_dump_json(), tmp_path) is True

    assert "phase" not in checkpoint.model_dump()
    prepare.assert_not_called()
    submit.assert_not_called()
    assert restored.best_probs == program.best_probs
    np.testing.assert_allclose(restored.best_params, program.best_params)
    for name in _PREPARED_FIELDS:
        np.testing.assert_allclose(getattr(restored, name), getattr(program, name))


def test_lucj_fragment_checkpoint_rejects_a_mismatched_digest(
    prepared_program, dummy_simulator, tmp_path
):
    program, *_ = prepared_program
    checkpoint = program._make_checkpoint(tmp_path).model_copy(
        update={"state_sha256": "0" * 64}
    )
    mismatched = _two_orbital_program(dummy_simulator, seed=7)

    with pytest.raises(
        ValueError,
        match=exact_match("Completed fragment state digest does not match metadata"),
    ):
        mismatched._restore_checkpoint(checkpoint.model_dump_json(), tmp_path)
    assert not mismatched.has_results()


def test_lucj_fragment_checkpoint_writes_its_temporary_state_beside_the_artifact(
    prepared_program, mocker, tmp_path
):
    """The state is staged in the checkpoint directory so the final rename stays
    on one filesystem."""
    program, *_ = prepared_program
    temporary = mocker.spy(preparation.tempfile, "NamedTemporaryFile")

    program._make_checkpoint(tmp_path)

    assert Path(temporary.spy_return.name).parent == tmp_path
    assert [path.name for path in tmp_path.iterdir()] == ["completed_state.npz"]


def test_a_failed_checkpoint_write_leaves_no_temporary_state(
    prepared_program, mocker, tmp_path
):
    program, *_ = prepared_program
    mocker.patch(f"{_PREPARATION}.np.savez", side_effect=OSError("disk full"))

    with pytest.raises(OSError, match=exact_match("disk full")):
        program._make_checkpoint(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_a_checkpoint_into_a_missing_directory_raises_file_not_found(
    prepared_program, tmp_path
):
    program, *_ = prepared_program

    with pytest.raises(FileNotFoundError):
        program._make_checkpoint(tmp_path / "missing")


def _completed_state_arrays():
    return {
        "params": np.array([0.1]),
        "h_alpha": np.eye(2),
        "h_beta": np.eye(2),
        "two_body": np.zeros((2, 2, 2, 2)),
        "orbital_rotation": np.eye(2),
    }


def _restore_completed_state(
    program, directory, arrays, program_type="LUCJFragmentProgram"
):
    """Write ``arrays`` as a completed fragment state and restore it."""
    state_path = directory / "completed_state.npz"
    np.savez(state_path, **arrays)
    with state_path.open("rb") as handle:
        state_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    checkpoint = preparation._LUCJFragmentCheckpoint(
        program_type=program_type,
        total_circuit_count=0,
        total_run_time=0.0,
        state_file="completed_state.npz",
        state_sha256=state_sha256,
        best_probs={0: {"0011": 1.0}},
    )
    return program._restore_checkpoint(checkpoint.model_dump_json(), directory)


def test_completed_fragment_state_is_refused_for_another_program_type(
    dummy_simulator, tmp_path
):
    program = _two_orbital_program(dummy_simulator)

    with pytest.raises(
        ValueError, match=exact_match("Checkpoint is for a different program type.")
    ):
        _restore_completed_state(
            program, tmp_path, _completed_state_arrays(), program_type="VQE"
        )

    assert not program.has_results()


@pytest.mark.parametrize(
    "missing",
    ["params", "h_alpha", "h_beta", "two_body", "orbital_rotation"],
)
def test_completed_fragment_state_requires_every_array(
    missing, dummy_simulator, tmp_path
):
    arrays = _completed_state_arrays()
    arrays.pop(missing)
    program = _two_orbital_program(dummy_simulator)

    with pytest.raises(
        ValueError,
        match=exact_match("Completed fragment state has missing or extra arrays"),
    ):
        _restore_completed_state(program, tmp_path, arrays)

    assert not program.has_results()


_INCOMPATIBLE_SHAPES = "Completed fragment state has incompatible array shapes"
_NON_FINITE = "Completed fragment state arrays must be finite numeric data"


@pytest.mark.parametrize(
    "name, value, message",
    [
        ("h_alpha", np.eye(3), _INCOMPATIBLE_SHAPES),
        ("params", np.zeros((1, 1)), _INCOMPATIBLE_SHAPES),
        ("two_body", np.full((2, 2, 2, 2), np.nan), _NON_FINITE),
        ("params", np.array(["a"]), _NON_FINITE),
    ],
)
def test_completed_fragment_state_rejects_corrupt_arrays(
    name, value, message, dummy_simulator, tmp_path
):
    arrays = _completed_state_arrays()
    arrays[name] = value
    program = _two_orbital_program(dummy_simulator)

    with pytest.raises(ValueError, match=exact_match(message)):
        _restore_completed_state(program, tmp_path, arrays)

    assert not program.has_results()


def test_completed_fragment_state_requires_an_orthonormal_rotation(
    dummy_simulator, tmp_path
):
    arrays = _completed_state_arrays()
    arrays["orbital_rotation"] = 2.0 * np.eye(2)
    program = _two_orbital_program(dummy_simulator)

    with pytest.raises(
        ValueError,
        match=r"^Completed fragment state orbital_rotation is not orthonormal",
    ):
        _restore_completed_state(program, tmp_path, arrays)

    assert not program.has_results()


def test_completed_fragment_state_refuses_an_object_array(dummy_simulator, tmp_path):
    arrays = _completed_state_arrays()
    arrays["params"] = np.array([{"payload": 1}], dtype=object)
    program = _two_orbital_program(dummy_simulator)

    with pytest.raises(ValueError, match="allow_pickle=False"):
        _restore_completed_state(program, tmp_path, arrays)

    assert not program.has_results()


@pytest.mark.parametrize(
    "read",
    [
        *(attrgetter(name) for name in ("best_params", *_PREPARED_FIELDS)),
        methodcaller("_initial_spec"),
    ],
    ids=["best_params", *_PREPARED_FIELDS, "initial_spec"],
)
def test_lucj_fragment_results_are_unavailable_before_run(dummy_simulator, read):
    program = _two_orbital_program(dummy_simulator)

    with pytest.raises(
        RuntimeError,
        match=exact_match("The fragment has not been prepared; call run() first."),
    ):
        read(program)


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
