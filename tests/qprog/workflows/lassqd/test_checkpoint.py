# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for LASSQD checkpoint saving, loading, validation and resume."""

import dataclasses

import numpy as np
import pytest
from pyscf import gto, solvent

from divi.qprog.algorithms import LUCJAnsatz
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.optimizers import MonteCarloOptimizer, ScipyMethod, ScipyOptimizer
from divi.qprog.problems import MolecularProblem
from divi.qprog.workflows.lassqd import (
    CCSDPreparation,
    FixedOrbitals,
    LBFGSOrbitalUpdate,
    SecondOrderOrbitalUpdate,
    VQEPreparation,
    _checkpoint,
)
from divi.qprog.workflows.lassqd._state import (
    FragmentSpec,
    FragmentState,
    LASSQDState,
)
from tests._helpers import exact_match
from tests.qprog.workflows.lassqd._helpers import (  # noqa: F401
    _REPORTED_STATES,
    H4_WHOLE_SPACE,
    _h2o_states_lassqd,
    _lassqd,
    _linearly_dependent_h2,
    _raw_lassqd,
    _round_report,
    _solvated_h2o_lassqd,
    build_exact_sampler_lassqd,
    exact_sampler_lassqd,
    h4_chain,
    vqe_preparation,
)


def test_vqe_checkpoint_configuration_tracks_ansatz_options(dummy_expval_backend):
    preparation = VQEPreparation(ScipyOptimizer(ScipyMethod.COBYLA))
    first = _raw_lassqd(dummy_expval_backend, preparation=preparation, n_layers=1)
    same = _raw_lassqd(dummy_expval_backend, preparation=preparation, n_layers=1)
    changed = _raw_lassqd(dummy_expval_backend, preparation=preparation, n_layers=2)

    assert first._configuration_record() == same._configuration_record()
    assert first._configuration_record() != changed._configuration_record()


@pytest.mark.parametrize(
    "method, n_layers",
    [(ScipyMethod.L_BFGS_B, 1), (ScipyMethod.COBYLA, 2)],
)
def test_vqe_checkpoint_rejects_changed_optimizer_or_ansatz_options(
    dummy_expval_backend, tmp_path, method, n_layers
):
    source = _raw_lassqd(
        dummy_expval_backend,
        preparation=VQEPreparation(ScipyOptimizer(ScipyMethod.COBYLA)),
        n_layers=1,
    )
    payload = source._save_workflow_checkpoint_state(
        source.initial_state(), tmp_path, "output_state"
    )
    target = _raw_lassqd(
        dummy_expval_backend,
        preparation=VQEPreparation(ScipyOptimizer(method)),
        n_layers=n_layers,
    )

    with pytest.raises(ValueError, match="different LASSQD configuration"):
        target._load_workflow_checkpoint_state(payload, tmp_path, "output_state")


def test_a_fixed_orbitals_run_restores_from_its_checkpoint(
    dummy_expval_backend, mocker, tmp_path
):
    """FixedOrbitals rounds report a NaN orbital gradient, which the checkpoint
    stores as null."""
    ensemble, _ = _solvated_h2o_lassqd(
        dummy_expval_backend, mocker, FixedOrbitals(), {1: 2}
    )
    ensemble.run(
        max_rounds=2, checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path)
    )
    restored, _ = _solvated_h2o_lassqd(
        dummy_expval_backend, mocker, FixedOrbitals(), {1: 2}
    )

    restored.restore_state(tmp_path)

    assert restored.electronic_states == ensemble.electronic_states
    assert np.isnan(restored.round_reports[-1].orbital_gradient_norm)


def test_the_solvent_field_survives_a_checkpoint(
    dummy_expval_backend, mocker, tmp_path
):
    ensemble, _ = _solvated_h2o_lassqd(
        dummy_expval_backend, mocker, FixedOrbitals(), {1: 2}
    )
    ensemble.run(max_rounds=2)
    payload = ensemble._save_workflow_checkpoint_state(
        ensemble.workflow_state, tmp_path, "output_state"
    )
    resumed, _ = _solvated_h2o_lassqd(
        dummy_expval_backend, mocker, FixedOrbitals(), {1: 2}
    )

    resumed._load_workflow_checkpoint_state(payload, tmp_path, "output_state")

    np.testing.assert_array_equal(
        resumed._ao_integrals().h_core, ensemble._ao_integrals().h_core
    )


def test_state_wavefunctions_survive_a_checkpoint(
    dummy_expval_backend, mocker, tmp_path
):
    ensemble, _ = _h2o_states_lassqd(dummy_expval_backend, mocker, FixedOrbitals())
    ensemble.run(max_rounds=1)
    payload = ensemble._save_workflow_checkpoint_state(
        ensemble.workflow_state, tmp_path, "output_state"
    )
    resumed, _ = _h2o_states_lassqd(dummy_expval_backend, mocker, FixedOrbitals())

    resumed._load_workflow_checkpoint_state(payload, tmp_path, "output_state")

    for saved, loaded in zip(
        ensemble.state_wavefunctions().values(),
        resumed.state_wavefunctions().values(),
        strict=True,
    ):
        assert loaded.state == saved.state
        np.testing.assert_allclose(loaded.rdm1, saved.rdm1, atol=1e-12)


def _checkpoint_state(state, *, orbitals_converged=False):
    """A post-round state whose every array differs from ``state``'s."""
    fragments = tuple(
        FragmentState(
            spec=fragment.spec,
            rdm1=fragment.rdm1 + 0.1 * (index + 1),
            rdm2=fragment.rdm2 + 0.2 * (index + 1),
            params=np.arange(3, dtype=float) + index,
            rdm1_alpha=fragment.rdm1 / 3 + index,
            rdm1_beta=fragment.rdm1 * 2 / 3 - index,
            carried_alpha=("10",),
            carried_beta=("01", "10")[: index + 1],
            sampled_orbitals=np.full(
                (state.mo_coeff.shape[0], fragment.spec.n_orbitals), 0.1 * (index + 1)
            ),
        )
        for index, fragment in enumerate(state.fragments)
    )
    # A rotation of the first two orbitals, so the orbitals stay orthonormal.
    rotation = np.eye(state.mo_coeff.shape[1])
    rotation[:2, :2] = [[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]]
    return LASSQDState(
        mo_coeff=state.mo_coeff @ rotation,
        fragments=fragments,
        energy=-1.25,
        previous_energy=-1.0,
        orbitals_converged=orbitals_converged,
    )


def _save_checkpoint(ensemble, state, directory):
    """Save ``state`` after two recorded rounds and one cached solver."""
    ensemble._energy_history = [-1.0, -1.25]
    ensemble._round_reports = [
        _round_report(1, None),
        _round_report(
            2,
            -0.25,
            orbital_converged=state.orbitals_converged,
            states=_REPORTED_STATES,
            averaged_energy=-1.125,
            saddle_check="passed",
        ),
    ]
    ensemble._solver_for(0, state.fragments[0].spec)
    return ensemble._save_workflow_checkpoint_state(state, directory, "output_state")


def _rewrite_checkpoint(directory, payload, corrupt):
    """Apply ``corrupt(arrays, payload)`` to a saved checkpoint in place."""
    path = directory / payload["artifact"]
    with np.load(path) as stored:
        arrays = {name: stored[name] for name in stored.files}
    corrupt(arrays, payload)
    np.savez(path, **arrays)


def _set_array(name, value):
    def corrupt(arrays, payload):
        arrays[name] = value

    return corrupt


def _drop_array(name):
    def corrupt(arrays, payload):
        del arrays[name]

    return corrupt


def _set_payload(key, value):
    def corrupt(arrays, payload):
        payload[key] = value

    return corrupt


def _drop_payload(key):
    def corrupt(arrays, payload):
        del payload[key]

    return corrupt


def _duplicate_solver(arrays, payload):
    payload["solvers"] = payload["solvers"] * 2


def _solver_one_past_the_end(arrays, payload):
    payload["solvers"][0]["index"] = len(payload["fragments"])


def _stretch_orbitals(arrays, payload):
    arrays["mo_coeff"] = 1.01 * arrays["mo_coeff"]


def _drop_last_orbital(arrays, payload):
    arrays["mo_coeff"] = arrays["mo_coeff"][:, :-1]


def _set_entry(key, field, value):
    """Set ``field`` of the first entry of the payload list ``key``."""

    def corrupt(arrays, payload):
        payload[key][0][field] = value

    return corrupt


def _drop_entry(key, field):
    """Drop ``field`` from the first entry of the payload list ``key``."""

    def corrupt(arrays, payload):
        del payload[key][0][field]

    return corrupt


_CARRIED_ALPHA_FORMAT = (
    "LASSQD checkpoint fragment 0 carried_alpha must be a list of 2-character "
    "0/1 strings."
)


def _corrupt_case(corrupt, message, case_id):
    """A corrupt-checkpoint case whose load must fail with exactly ``message``."""
    return pytest.param(corrupt, exact_match(message), id=case_id)


def _wrong_shape_case(name, shape, expected):
    return _corrupt_case(
        _set_array(name, np.zeros(shape)),
        f"LASSQD checkpoint {name} has shape {shape}; expected {expected}.",
        f"{name}-shape",
    )


class _PickleTripwire:
    """Records whether an instance was ever unpickled."""

    unpickled = False

    def __init__(self):
        self.marker = 1

    def __setstate__(self, state):
        type(self).unpickled = True
        self.__dict__.update(state)


def _resume_one_round(ensemble, payload, directory, *, restore_solvers):
    """Load a checkpoint into ``ensemble``, run one round, return its history."""
    state = ensemble._load_workflow_checkpoint_state(payload, directory, "output_state")
    if not restore_solvers:
        ensemble._solvers.clear()
    ensemble.create_programs(state)
    ensemble.run_one_round(blocking=True)
    ensemble.update_state(state)
    return ensemble.energy_history


@pytest.mark.parametrize("orbitals_converged", [True, False])
def test_workflow_checkpoint_state_round_trips_npz(
    exact_sampler_lassqd, tmp_path, orbitals_converged
):
    ensemble, initial = exact_sampler_lassqd
    state = _checkpoint_state(initial, orbitals_converged=orbitals_converged)
    payload = _save_checkpoint(ensemble, state, tmp_path)
    expected_reports = ensemble.round_reports
    expected_rng_state = ensemble._rng.bit_generator.state
    expected_solver_state = ensemble._solvers[0]._rng.bit_generator.state
    ensemble._rng.random()
    ensemble._solvers[0]._rng.random()
    ensemble._energy_history.clear()
    ensemble._round_reports.clear()
    ensemble._solvers.clear()

    restored = ensemble._load_workflow_checkpoint_state(
        payload, tmp_path, "output_state"
    )

    np.testing.assert_array_equal(restored.mo_coeff, state.mo_coeff)
    for loaded, saved in zip(restored.fragments, state.fragments, strict=True):
        assert loaded.spec == saved.spec
        for name in (
            "rdm1",
            "rdm2",
            "params",
            "rdm1_alpha",
            "rdm1_beta",
            "sampled_orbitals",
        ):
            np.testing.assert_array_equal(getattr(loaded, name), getattr(saved, name))
        assert (loaded.carried_alpha, loaded.carried_beta) == (
            saved.carried_alpha,
            saved.carried_beta,
        )
    assert (restored.energy, restored.previous_energy) == (-1.25, -1.0)
    assert restored.orbitals_converged is orbitals_converged
    assert ensemble.energy_history == (-1.0, -1.25)
    assert ensemble.round_reports == expected_reports
    assert ensemble.round_reports[0].energy_change is None
    assert ensemble._rng.bit_generator.state == expected_rng_state
    assert set(ensemble._solvers) == {0}
    assert ensemble._solvers[0]._rng.bit_generator.state == expected_solver_state


def test_resuming_from_a_checkpoint_reproduces_an_uninterrupted_run(
    dummy_expval_backend, mocker, tmp_path
):
    """The batch is far smaller than the 36-determinant sector, and no extension
    fills it in, so the second round's subspace depends on where each generator
    was left."""
    settings = dict(
        active_spaces=[H4_WHOLE_SPACE],
        batch_size=3,
        n_recovery_iterations=1,
        extension=None,
    )
    straight, _ = build_exact_sampler_lassqd(dummy_expval_backend, mocker, **settings)
    straight.run(max_rounds=2)

    interrupted, _ = build_exact_sampler_lassqd(
        dummy_expval_backend, mocker, **settings
    )
    interrupted.run(max_rounds=1)
    payload = interrupted._save_workflow_checkpoint_state(
        interrupted.workflow_state, tmp_path, "output_state"
    )

    resumed, control = (
        build_exact_sampler_lassqd(dummy_expval_backend, mocker, **settings)[0]
        for _ in range(2)
    )

    assert len(straight.energy_history) == 2
    assert (
        _resume_one_round(resumed, payload, tmp_path, restore_solvers=True)
        == straight.energy_history
    )
    assert (
        _resume_one_round(control, payload, tmp_path, restore_solvers=False)
        != straight.energy_history
    )


def test_restoring_a_run_checkpoint_reproduces_an_uninterrupted_run(
    dummy_expval_backend, mocker, tmp_path
):
    """The path a user takes: ``run()`` writes the checkpoint, ``restore_state``
    reads it and ``run()`` continues. Restoring a completed round also rebuilds
    that round's programs from its input state, which must not leave the
    workflow's generators, solvers or history at that earlier point."""
    settings = dict(
        active_spaces=[H4_WHOLE_SPACE], batch_size=3, n_recovery_iterations=1
    )
    straight, _ = build_exact_sampler_lassqd(dummy_expval_backend, mocker, **settings)
    straight.run(max_rounds=2)
    interrupted, _ = build_exact_sampler_lassqd(
        dummy_expval_backend, mocker, **settings
    )
    interrupted.run(
        max_rounds=1, checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path)
    )

    resumed, _ = build_exact_sampler_lassqd(dummy_expval_backend, mocker, **settings)
    resumed.restore_state(tmp_path).run(max_rounds=2)

    assert resumed.energy_history == straight.energy_history
    assert [report.number for report in resumed.round_reports] == [1, 2]


@pytest.mark.parametrize(
    "corrupt, match",
    [
        _corrupt_case(
            _set_payload("artifact", 1),
            "LASSQD checkpoint is missing its NPZ artifact.",
            "non-string-artifact",
        ),
        _corrupt_case(
            _set_payload("artifact", "input_state.npz"),
            "LASSQD checkpoint references the wrong state artifact.",
            "wrong-artifact",
        ),
        _corrupt_case(
            _set_payload("fragments", []),
            "LASSQD checkpoint has no fragment metadata.",
            "empty-fragment-list",
        ),
        _corrupt_case(
            _set_payload("rng_state", []),
            "LASSQD checkpoint is missing RNG state.",
            "non-dict-rng-state",
        ),
        _corrupt_case(
            _drop_array("fragment_0_rdm2"),
            "LASSQD checkpoint artifact is missing fragment_0_rdm2.",
            "missing-array",
        ),
        _wrong_shape_case("fragment_0_rdm1", (3, 3), (2, 2)),
        _wrong_shape_case("fragment_0_rdm2", (3, 3, 3, 3), (2, 2, 2, 2)),
        _wrong_shape_case("fragment_1_rdm1_beta", (3, 3), (2, 2)),
        _wrong_shape_case("fragment_0_sampled_orbitals", (3, 2), (4, 2)),
        _wrong_shape_case("energy", (2,), ()),
        _wrong_shape_case("previous_energy", (2,), ()),
        _wrong_shape_case("orbitals_converged", (2,), ()),
        _corrupt_case(
            _drop_last_orbital,
            "LASSQD checkpoint mo_coeff has shape (4, 3); expected (4, 4).",
            "orthonormal-orbital-missing",
        ),
        _corrupt_case(
            _duplicate_solver,
            "LASSQD checkpoint has invalid solver fragment indices.",
            "duplicate-solver",
        ),
        _corrupt_case(
            _solver_one_past_the_end,
            "LASSQD checkpoint has invalid solver fragment indices.",
            "out-of-range-solver",
        ),
        _corrupt_case(
            _drop_entry("solvers", "rng_state"),
            "LASSQD checkpoint is missing solver RNG state.",
            "missing-solver-rng-state",
        ),
        _corrupt_case(
            _set_payload("solvers", [1]),
            "LASSQD checkpoint solver metadata must be objects.",
            "non-object-solver",
        ),
        _corrupt_case(
            _drop_entry("fragments", "orbitals"),
            "LASSQD checkpoint fragment 0 metadata is malformed.",
            "missing-fragment-orbitals",
        ),
        _corrupt_case(
            _set_entry("fragments", "orbitals", 3),
            "LASSQD checkpoint fragment 0 metadata is malformed.",
            "non-list-fragment-orbitals",
        ),
        _corrupt_case(
            _drop_entry("round_reports", "energy"),
            "LASSQD checkpoint round report 0 is malformed.",
            "missing-report-energy",
        ),
        _corrupt_case(
            _set_entry("round_reports", "saddle_check", "skipped"),
            "LASSQD checkpoint round report 0 is malformed.",
            "unknown-saddle-check",
        ),
        _corrupt_case(
            _set_array("fragment_0_params", np.zeros((3, 1))),
            "LASSQD checkpoint fragment 0 parameters are not 1-D.",
            "2d-params",
        ),
        _corrupt_case(
            _set_array("energy_history", np.zeros((2, 1))),
            "LASSQD checkpoint energy history is not 1-D.",
            "2d-energy-history",
        ),
        _corrupt_case(
            _set_array("fragment_0_rdm2", np.full((2, 2, 2, 2), np.nan)),
            "LASSQD checkpoint fragment_0_rdm2 contains non-finite values.",
            "nan",
        ),
        _corrupt_case(
            _set_array("fragment_0_rdm1", np.full((2, 2), np.inf)),
            "LASSQD checkpoint fragment_0_rdm1 contains non-finite values.",
            "inf",
        ),
        _corrupt_case(
            _set_array("mo_coeff", np.full((4, 4), "x")),
            "LASSQD checkpoint mo_coeff has non-numeric dtype <U1.",
            "string-dtype",
        ),
        _corrupt_case(
            _drop_payload("solvers"),
            "LASSQD checkpoint solver metadata must be a list.",
            "missing-solvers",
        ),
        _corrupt_case(
            _drop_payload("round_reports"),
            "LASSQD checkpoint round reports must be a list.",
            "missing-round-reports",
        ),
        pytest.param(
            _stretch_orbitals,
            "^LASSQD checkpoint mo_coeff is not orthonormal: ",
            id="non-orthonormal-orbitals",
        ),
        _corrupt_case(
            _drop_payload("molecule"),
            "LASSQD checkpoint is missing its molecule fingerprint.",
            "missing-molecule",
        ),
        _corrupt_case(
            _drop_payload("configuration"),
            "LASSQD checkpoint is missing its configuration.",
            "missing-configuration",
        ),
        _corrupt_case(
            _set_entry("fragments", "carried_alpha", ["102"]),
            _CARRIED_ALPHA_FORMAT,
            "non-binary-carried-string",
        ),
        _corrupt_case(
            _set_entry("fragments", "carried_alpha", ["100"]),
            _CARRIED_ALPHA_FORMAT,
            "carried-string-width",
        ),
        _corrupt_case(
            _drop_entry("fragments", "carried_alpha"),
            _CARRIED_ALPHA_FORMAT,
            "missing-carried-strings",
        ),
    ],
)
def test_workflow_checkpoint_rejects_a_corrupt_snapshot(
    exact_sampler_lassqd, tmp_path, corrupt, match
):
    ensemble, initial = exact_sampler_lassqd
    payload = _save_checkpoint(ensemble, _checkpoint_state(initial), tmp_path)
    _rewrite_checkpoint(tmp_path, payload, corrupt)

    with pytest.raises(ValueError, match=match):
        ensemble._load_workflow_checkpoint_state(payload, tmp_path, "output_state")


def test_a_checkpoint_resumes_under_an_identically_configured_preparation(
    dummy_expval_backend, mocker, tmp_path
):
    """Two identically configured VQE preparations hold different optimizer
    objects; the record compares their configuration, not their identity."""
    writer, state = build_exact_sampler_lassqd(dummy_expval_backend, mocker)
    payload = _save_checkpoint(writer, _checkpoint_state(state), tmp_path)
    same, _ = build_exact_sampler_lassqd(
        dummy_expval_backend, mocker, preparation=vqe_preparation()
    )

    same._load_workflow_checkpoint_state(payload, tmp_path, "output_state")


def _different_configuration(field):
    """The checkpoint-mismatch message naming one differing ``field``."""
    return (
        "LASSQD checkpoint was written with a different LASSQD configuration: "
        f"its {field} differs from this run's."
    )


_DIFFERENT_MOLECULE = (
    "LASSQD checkpoint was written for a different molecule: its geometry, "
    "basis, electron count or spin differs from this one's."
)


def _pcm(method):
    model = solvent.PCM(h4_chain())
    model.method = method
    return model


@pytest.mark.parametrize(
    "changed, field",
    [
        ({"preparation": CCSDPreparation()}, "preparation"),
        (
            {"preparation": vqe_preparation(optimizer=MonteCarloOptimizer())},
            "preparation",
        ),
        ({"preparation": vqe_preparation(ansatz=LUCJAnsatz())}, "preparation"),
        ({"preparation": vqe_preparation(max_iterations=3)}, "preparation"),
        ({"orbital_update": LBFGSOrbitalUpdate(max_iterations=7)}, "orbital_update"),
        (
            {"orbital_update": SecondOrderOrbitalUpdate(max_iterations=7)},
            "orbital_update",
        ),
        ({"solvent": _pcm("IEF-PCM")}, "solvent"),
    ],
    ids=[
        "preparation",
        "vqe-optimizer",
        "vqe-ansatz",
        "vqe-iteration-budget",
        "full-orbital-solve",
        "second-order-orbital-solve",
        "solvent-model-variant",
    ],
)
def test_a_checkpoint_resumes_only_under_the_same_configuration(
    dummy_expval_backend, mocker, tmp_path, changed, field
):
    writer, state = build_exact_sampler_lassqd(
        dummy_expval_backend,
        mocker,
        **({"solvent": _pcm("C-PCM")} if field == "solvent" else {}),
    )
    payload = _save_checkpoint(writer, _checkpoint_state(state), tmp_path)
    other, _ = build_exact_sampler_lassqd(dummy_expval_backend, mocker, **changed)

    with pytest.raises(ValueError, match=exact_match(_different_configuration(field))):
        other._load_workflow_checkpoint_state(payload, tmp_path, "output_state")


def test_workflow_checkpoint_refuses_to_unpickle_an_object_array(
    exact_sampler_lassqd, tmp_path
):
    ensemble, initial = exact_sampler_lassqd
    payload = _save_checkpoint(ensemble, _checkpoint_state(initial), tmp_path)
    tripwire = np.empty(1, dtype=object)
    tripwire[0] = _PickleTripwire()
    _rewrite_checkpoint(tmp_path, payload, _set_array("fragment_0_rdm1", tripwire))

    with pytest.raises(ValueError, match="allow_pickle=False"):
        ensemble._load_workflow_checkpoint_state(payload, tmp_path, "output_state")
    assert not _PickleTripwire.unpickled


@pytest.mark.parametrize(
    "module, failing",
    [(_checkpoint.np, "savez"), (_checkpoint.tempfile, "NamedTemporaryFile")],
    ids=["write", "create"],
)
def test_a_failed_checkpoint_save_leaves_no_temporary_file(
    exact_sampler_lassqd, tmp_path, mocker, module, failing
):
    ensemble, state = exact_sampler_lassqd
    mocker.patch.object(module, failing, side_effect=OSError("disk full"))

    with pytest.raises(OSError, match=exact_match("disk full")):
        ensemble._save_workflow_checkpoint_state(state, tmp_path, "output_state")

    assert list(tmp_path.iterdir()) == []


def test_a_checkpoint_save_does_not_depend_on_the_system_temporary_directory(
    exact_sampler_lassqd, tmp_path, monkeypatch
):
    """The snapshot is staged beside its destination, so the final rename never
    crosses a filesystem."""
    ensemble, state = exact_sampler_lassqd
    monkeypatch.setattr(_checkpoint.tempfile, "tempdir", str(tmp_path / "missing"))

    ensemble._save_workflow_checkpoint_state(state, tmp_path, "output_state")

    assert [path.name for path in tmp_path.iterdir()] == ["output_state.npz"]


def test_a_checkpoint_save_rejects_a_foreign_state(exact_sampler_lassqd, tmp_path):
    ensemble, _ = exact_sampler_lassqd
    message = "LASSQD checkpoint state must be LASSQDState, got NoneType."

    with pytest.raises(TypeError, match=exact_match(message)):
        ensemble._save_workflow_checkpoint_state(None, tmp_path, "output_state")


def test_a_closed_shell_fragment_without_spin_rdms_round_trips(
    exact_sampler_lassqd, tmp_path
):
    ensemble, initial = exact_sampler_lassqd
    state = dataclasses.replace(
        initial,
        fragments=tuple(
            dataclasses.replace(fragment, rdm1_alpha=None, rdm1_beta=None)
            for fragment in initial.fragments
        ),
    )
    payload = ensemble._save_workflow_checkpoint_state(state, tmp_path, "output_state")

    restored = ensemble._load_workflow_checkpoint_state(
        payload, tmp_path, "output_state"
    )

    assert all(
        fragment.rdm1_alpha is None and fragment.rdm1_beta is None
        for fragment in restored.fragments
    )


_MISMATCHED_STATE = (
    "update_state received a different state than create_programs built this "
    "round's circuits from; the reduction would use different orbitals than the "
    "fragment circuits were prepared in."
)


def _built_round(ensemble, state, directory):
    ensemble.create_programs(state)
    ensemble.run_one_round(blocking=True)
    return state


def _loaded_checkpoint(ensemble, state, directory):
    payload = _save_checkpoint(ensemble, _checkpoint_state(state), directory)
    return ensemble._load_workflow_checkpoint_state(payload, directory, "output_state")


@pytest.mark.parametrize(
    "prepare", [_built_round, _loaded_checkpoint], ids=["built", "loaded"]
)
def test_update_state_rejects_a_state_its_round_was_not_built_from(
    exact_sampler_lassqd, tmp_path, prepare
):
    ensemble, initial = exact_sampler_lassqd
    state = prepare(ensemble, initial, tmp_path)

    with pytest.raises(RuntimeError, match=exact_match(_MISMATCHED_STATE)):
        ensemble.update_state(dataclasses.replace(state))


def test_a_restored_completed_round_reduces_against_its_input_state(
    dummy_expval_backend, mocker, tmp_path
):
    """Restoring a completed round rebuilds its programs from the round's input
    state, so the restored output state is not one they can be reduced against."""
    ensemble, _ = build_exact_sampler_lassqd(dummy_expval_backend, mocker)
    ensemble.run(
        max_rounds=1, checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path)
    )
    restored, _ = build_exact_sampler_lassqd(dummy_expval_backend, mocker)
    restored.restore_state(tmp_path)

    with pytest.raises(RuntimeError, match=exact_match(_MISMATCHED_STATE)):
        restored.update_state(restored.workflow_state)


_STRETCHED_H4 = "H 0 0 0; H 0 0 0.80; H 0 0 2.0; H 0 0 2.80"


@pytest.mark.parametrize(
    "overrides, message",
    [
        pytest.param(
            {
                "problem": MolecularProblem.from_molecule(
                    gto.M(atom=_STRETCHED_H4, basis="sto-3g", verbose=0)
                )
            },
            _DIFFERENT_MOLECULE,
            id="geometry",
        ),
        pytest.param(
            {"active_spaces": [H4_WHOLE_SPACE]},
            _different_configuration("fragmentation"),
            id="fragment-layout",
        ),
        pytest.param(
            {"n_batches": 3}, _different_configuration("sqd"), id="sqd-budget"
        ),
        pytest.param(
            {"solvent": solvent.ddCOSMO(h4_chain())},
            _different_configuration("solvent"),
            id="solvent",
        ),
    ],
)
def test_workflow_checkpoint_rejects_a_different_problem(
    exact_sampler_lassqd, dummy_expval_backend, mocker, tmp_path, overrides, message
):
    """A snapshot from another molecule with the same orbital count, or another
    fragmentation or SQD setup, would otherwise resume silently against the
    wrong orbitals or budget."""
    source, state = exact_sampler_lassqd
    payload = source._save_workflow_checkpoint_state(state, tmp_path, "output_state")
    target, _ = build_exact_sampler_lassqd(dummy_expval_backend, mocker, **overrides)

    with pytest.raises(ValueError, match=exact_match(message)):
        target._load_workflow_checkpoint_state(payload, tmp_path, "output_state")


def test_workflow_checkpoint_round_trips_a_linearly_dependent_basis(
    dummy_expval_backend, tmp_path
):
    """Dropping linearly dependent functions leaves fewer orbitals than basis
    functions, so ``mo_coeff`` is not square."""
    ensemble = _lassqd(
        dummy_expval_backend,
        problem=MolecularProblem.from_molecule(_linearly_dependent_h2()),
        active_spaces=[FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)],
    )
    state = ensemble.initial_state()
    payload = ensemble._save_workflow_checkpoint_state(state, tmp_path, "output_state")

    restored = ensemble._load_workflow_checkpoint_state(
        payload, tmp_path, "output_state"
    )

    np.testing.assert_array_equal(restored.mo_coeff, state.mo_coeff)
