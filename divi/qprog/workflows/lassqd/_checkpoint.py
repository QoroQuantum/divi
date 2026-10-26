# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""The LASSQD checkpoint codec: an NPZ-plus-JSON snapshot of the workflow."""

import hashlib
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, get_args

import numpy as np
from pyscf import gto

from ._sqd import SQDConfig
from ._state import (
    ElectronicState,
    FragmentSpec,
    FragmentState,
    LASSQDRoundReport,
    LASSQDState,
    _SaddleCheck,
    _SampledSubspace,
    require_orthonormal,
    validate_fragment_specs,
)


@dataclass(frozen=True, eq=False)
class _WorkflowSnapshot:
    """What a LASSQD checkpoint restores onto a workflow."""

    state: LASSQDState
    rng_state: Mapping[str, Any]
    solver_rng_states: Mapping[int, Mapping[str, Any]]
    energy_history: tuple[float, ...]
    round_reports: tuple[LASSQDRoundReport, ...]
    final_subspace: _SampledSubspace | None
    reaction_field: tuple[float, np.ndarray] | None


def _molecule_fingerprint(mol) -> str:
    """Digest of what fixes a molecule's integrals: atoms, geometry, basis,
    effective core potentials and electron count."""
    digest = hashlib.sha256()
    for array in (mol._atm, mol._bas, mol._env[gto.PTR_ENV_START :], mol._ecpbas):
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(f"{mol.cart}:{mol.nelectron}".encode())
    if mol.spin:
        digest.update(f":{mol.spin}".encode())
    return digest.hexdigest()


def _stored_array(
    stored: Any,
    name: str,
    shape: Sequence[int] | None = None,
    *,
    allow_infinite: bool = False,
) -> np.ndarray:
    """Copy one array out of a checkpoint archive, rejecting malformed data.

    Raises:
        ValueError: If the array is missing, an object array, non-numeric, of
            the wrong shape, NaN, or infinite where ``allow_infinite`` is not
            set.
    """
    if name not in stored.files:
        raise ValueError(f"LASSQD checkpoint artifact is missing {name}.")
    array = np.asarray(stored[name])
    if not (np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_):
        raise ValueError(
            f"LASSQD checkpoint {name} has non-numeric dtype {array.dtype}."
        )
    if shape is not None and array.shape != shape:
        raise ValueError(
            f"LASSQD checkpoint {name} has shape {array.shape}; expected {shape}."
        )
    invalid = np.isnan(array) if allow_infinite else ~np.isfinite(array)
    if np.any(invalid):
        raise ValueError(f"LASSQD checkpoint {name} contains non-finite values.")
    return array.copy()


def _carried_strings_from_metadata(
    metadata: Mapping[str, Any], name: str, spec: FragmentSpec, index: int
) -> tuple[str, ...]:
    """One sector's carried strings from checkpoint metadata, rejecting any that
    is not a 0/1 string as wide as the fragment.

    Raises:
        ValueError: If the entry is not a list of such strings.
    """
    strings = metadata.get(name)
    if not isinstance(strings, list) or any(
        not isinstance(string, str)
        or len(string) != spec.n_orbitals
        or set(string) - {"0", "1"}
        for string in strings
    ):
        raise ValueError(
            f"LASSQD checkpoint fragment {index} {name} must be a list of "
            f"{spec.n_orbitals}-character 0/1 strings."
        )
    return tuple(strings)


def _labels_from_metadata(entries: Any) -> dict[tuple[int, int], tuple[int, int]]:
    """Tracked state labels from checkpoint metadata.

    Raises:
        ValueError: If an entry is not ``[multiplicity, root, index]`` of
            integers.
    """
    if not isinstance(entries, list) or any(
        not isinstance(entry, list)
        or len(entry) != 3
        or not all(isinstance(value, int) for value in entry)
        for entry in entries
    ):
        raise ValueError(
            "LASSQD checkpoint labels must be a list of [multiplicity, root, "
            "index] integer triples."
        )
    return {
        (multiplicity, root): (multiplicity, label)
        for multiplicity, root, label in entries
    }


def _spec_from_metadata(metadata: Any, index: int) -> FragmentSpec:
    """A fragment's spec from checkpoint metadata.

    Raises:
        ValueError: If the entry lacks integer ``orbitals``, ``n_alpha`` or
            ``n_beta``.
    """
    try:
        return FragmentSpec(
            tuple(metadata["orbitals"]), metadata["n_alpha"], metadata["n_beta"]
        )
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"LASSQD checkpoint fragment {index} metadata is malformed."
        ) from error


def _report_from_metadata(report: Any, index: int) -> LASSQDRoundReport:
    """A round report from checkpoint metadata.

    Raises:
        ValueError: If a field is missing or of the wrong type, or
            ``saddle_check`` is not one of its outcomes.
    """

    def optional(value: Any) -> float | None:
        return None if value is None else float(value)

    try:
        saddle_check = report["saddle_check"]
        if saddle_check not in (*get_args(_SaddleCheck), None):
            raise ValueError(f"unknown saddle_check {saddle_check!r}")
        return LASSQDRoundReport(
            number=int(report["number"]),
            energy=float(report["energy"]),
            energy_change=optional(report["energy_change"]),
            subspace_sizes=tuple(int(size) for size in report["subspace_sizes"]),
            orbital_iterations=int(report["orbital_iterations"]),
            orbital_evaluations=int(report["orbital_evaluations"]),
            # JSON stores the NaN a FixedOrbitals round reports as null.
            orbital_gradient_norm=float(
                "nan"
                if report["orbital_gradient_norm"] is None
                else report["orbital_gradient_norm"]
            ),
            orbital_converged=bool(report["orbital_converged"]),
            rotation_pairs=int(report["rotation_pairs"]),
            recovery_seconds=float(report["recovery_seconds"]),
            orbital_seconds=float(report["orbital_seconds"]),
            states=tuple(
                ElectronicState(
                    multiplicity=int(entry["multiplicity"]),
                    index=int(entry["index"]),
                    energy=float(entry["energy"]),
                    spin_square=float(entry["spin_square"]),
                    charge_transfer=optional(entry.get("charge_transfer")),
                )
                for entry in report["states"]
            ),
            averaged_energy=optional(report["averaged_energy"]),
            saddle_check=saddle_check,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"LASSQD checkpoint round report {index} is malformed."
        ) from error


def save_snapshot(
    snapshot: _WorkflowSnapshot,
    round_dir: Path,
    stem: str,
    *,
    mol,
    configuration: Mapping[str, str],
) -> dict[str, Any]:
    """Write ``snapshot``'s arrays to ``round_dir/<stem>.npz`` and return the
    JSON payload that describes it."""
    state = snapshot.state
    arrays: dict[str, np.ndarray] = {
        "mo_coeff": state.mo_coeff,
        "energy": np.asarray(state.energy),
        "previous_energy": np.asarray(state.previous_energy),
        "orbitals_converged": np.asarray(state.orbitals_converged),
        "energy_history": np.asarray(snapshot.energy_history, dtype=np.float64),
    }
    if snapshot.reaction_field is not None:
        arrays["solvent_energy"] = np.asarray(snapshot.reaction_field[0])
        arrays["solvent_potential"] = snapshot.reaction_field[1]
    fragments = []
    for index, fragment in enumerate(state.fragments):
        prefix = f"fragment_{index}"
        arrays[f"{prefix}_rdm1"] = fragment.rdm1
        arrays[f"{prefix}_rdm2"] = fragment.rdm2
        if fragment.params is not None:
            arrays[f"{prefix}_params"] = fragment.params
        if fragment.rdm1_alpha is not None:
            arrays[f"{prefix}_rdm1_alpha"] = fragment.rdm1_alpha
        if fragment.rdm1_beta is not None:
            arrays[f"{prefix}_rdm1_beta"] = fragment.rdm1_beta
        if fragment.sampled_orbitals is not None:
            arrays[f"{prefix}_sampled_orbitals"] = fragment.sampled_orbitals
        fragments.append(
            {
                "orbitals": list(fragment.spec.orbitals),
                "n_alpha": fragment.spec.n_alpha,
                "n_beta": fragment.spec.n_beta,
                "params": fragment.params is not None,
                "rdm1_alpha": fragment.rdm1_alpha is not None,
                "rdm1_beta": fragment.rdm1_beta is not None,
                "sampled_orbitals": fragment.sampled_orbitals is not None,
                "carried_alpha": list(fragment.carried_alpha),
                "carried_beta": list(fragment.carried_beta),
            }
        )

    solvers = [
        {"index": index, "rng_state": rng_state}
        for index, rng_state in sorted(snapshot.solver_rng_states.items())
    ]
    final_strings = None
    final_subspace = snapshot.final_subspace
    if final_subspace is not None:
        arrays["final_orbitals"] = final_subspace.mo_coeff
        strings_alpha, strings_beta = final_subspace.strings
        final_strings = {
            "strings_alpha": list(strings_alpha),
            "strings_beta": list(strings_beta),
            "labels": [
                [multiplicity, root, label]
                for (multiplicity, root), (_, label) in (
                    final_subspace.labels or {}
                ).items()
            ],
        }

    artifact = f"{stem}.npz"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=round_dir, suffix=".npz", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez(handle, **arrays)  # pyrefly: ignore[bad-argument-type]
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, round_dir / artifact)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return {
        "artifact": artifact,
        "molecule": _molecule_fingerprint(mol),
        "configuration": dict(configuration),
        "fragments": fragments,
        "rng_state": snapshot.rng_state,
        "solvers": solvers,
        "round_reports": [asdict(report) for report in snapshot.round_reports],
        "final_strings": final_strings,
    }


def load_snapshot(
    payload: Mapping[str, Any],
    round_dir: Path,
    stem: str,
    *,
    mol,
    n_orbitals: int,
    n_occupied: int,
    spin: int,
    sqd: SQDConfig,
    configuration: Mapping[str, str],
) -> _WorkflowSnapshot:
    """Load and validate the NPZ snapshot ``payload`` describes, against this
    workflow's molecule, register and ``configuration``, without touching the
    workflow."""
    artifact = payload.get("artifact")
    if not isinstance(artifact, str):
        raise ValueError("LASSQD checkpoint is missing its NPZ artifact.")
    if artifact != f"{stem}.npz":
        raise ValueError("LASSQD checkpoint references the wrong state artifact.")
    molecule = payload.get("molecule")
    if not isinstance(molecule, str):
        raise ValueError("LASSQD checkpoint is missing its molecule fingerprint.")
    if molecule != _molecule_fingerprint(mol):
        raise ValueError(
            "LASSQD checkpoint was written for a different molecule: its "
            "geometry, basis, electron count or spin differs."
        )
    stored_configuration = payload.get("configuration")
    if not isinstance(stored_configuration, dict):
        raise ValueError("LASSQD checkpoint is missing its configuration.")
    differing = [
        name
        for name, record in configuration.items()
        if stored_configuration.get(name) != record
    ]
    if differing:
        raise ValueError(
            "LASSQD checkpoint was written with a different configuration: "
            f"{', '.join(differing)} "
            f"{'differs' if len(differing) == 1 else 'differ'} from this run's."
        )
    fragment_metadata = payload.get("fragments")
    if not isinstance(fragment_metadata, list) or not fragment_metadata:
        raise ValueError("LASSQD checkpoint has no fragment metadata.")
    rng_state = payload.get("rng_state")
    if not isinstance(rng_state, dict):
        raise ValueError("LASSQD checkpoint is missing RNG state.")
    solver_metadata = payload.get("solvers")
    if not isinstance(solver_metadata, list):
        raise ValueError("LASSQD checkpoint solver metadata must be a list.")
    report_payloads = payload.get("round_reports")
    if not isinstance(report_payloads, list):
        raise ValueError("LASSQD checkpoint round reports must be a list.")

    n_ao = mol.nao_nr()
    with np.load(round_dir / artifact, allow_pickle=False) as stored:
        fragments = []
        for index, metadata in enumerate(fragment_metadata):
            prefix = f"fragment_{index}"
            spec = _spec_from_metadata(metadata, index)
            rdm1_shape = (spec.n_orbitals,) * 2
            params = (
                _stored_array(stored, f"{prefix}_params")
                if metadata.get("params")
                else None
            )
            if params is not None and params.ndim != 1:
                raise ValueError(
                    f"LASSQD checkpoint fragment {index} parameters are not 1-D."
                )
            rdm1_alpha, rdm1_beta = (
                (
                    _stored_array(stored, f"{prefix}_{name}", rdm1_shape)
                    if metadata.get(name)
                    else None
                )
                for name in ("rdm1_alpha", "rdm1_beta")
            )
            sampled_orbitals = (
                _stored_array(
                    stored,
                    f"{prefix}_sampled_orbitals",
                    (n_ao, spec.n_orbitals),
                )
                if metadata.get("sampled_orbitals")
                else None
            )
            carried_alpha, carried_beta = (
                _carried_strings_from_metadata(metadata, name, spec, index)
                for name in ("carried_alpha", "carried_beta")
            )
            fragments.append(
                FragmentState(
                    spec=spec,
                    rdm1=_stored_array(stored, f"{prefix}_rdm1", rdm1_shape),
                    rdm2=_stored_array(
                        stored, f"{prefix}_rdm2", (spec.n_orbitals,) * 4
                    ),
                    params=params,
                    rdm1_alpha=rdm1_alpha,
                    rdm1_beta=rdm1_beta,
                    carried_alpha=carried_alpha,
                    carried_beta=carried_beta,
                    sampled_orbitals=sampled_orbitals,
                )
            )
        mo_coeff = _stored_array(stored, "mo_coeff", (n_ao, n_orbitals))
        energy = _stored_array(stored, "energy", (), allow_infinite=True)
        previous_energy = _stored_array(
            stored, "previous_energy", (), allow_infinite=True
        )
        orbitals_converged = _stored_array(stored, "orbitals_converged", ())
        energy_history = _stored_array(stored, "energy_history")
        if energy_history.ndim != 1:
            raise ValueError("LASSQD checkpoint energy history is not 1-D.")
        reaction_field = None
        if "solvent_potential" in stored.files:
            reaction_field = (
                float(_stored_array(stored, "solvent_energy", ())),
                _stored_array(stored, "solvent_potential", (n_ao, n_ao)),
            )
        final_strings = payload.get("final_strings")
        final_subspace = None
        if final_strings is not None:
            if not isinstance(final_strings, dict) or sqd.states is None:
                raise ValueError(
                    "LASSQD checkpoint final_strings must be an object, saved "
                    "only with SQDConfig.states."
                )
            final_subspace = _SampledSubspace(
                _stored_array(stored, "final_orbitals", (n_ao, n_orbitals)),
                (
                    _carried_strings_from_metadata(
                        final_strings, "strings_alpha", fragments[0].spec, 0
                    ),
                    _carried_strings_from_metadata(
                        final_strings, "strings_beta", fragments[0].spec, 0
                    ),
                ),
                sqd.states,
                sqd.lambda_penalty,
                labels=_labels_from_metadata(final_strings.get("labels", [])),
            )

    overlap = mol.intor_symmetric("int1e_ovlp")
    require_orthonormal(mo_coeff, "LASSQD checkpoint mo_coeff", overlap)
    if final_subspace is not None:
        require_orthonormal(
            final_subspace.mo_coeff, "LASSQD checkpoint final_orbitals", overlap
        )
    specs = [fragment.spec for fragment in fragments]
    validate_fragment_specs(specs, n_orbitals, n_occupied, spin=spin)
    state = LASSQDState(
        mo_coeff=mo_coeff,
        fragments=tuple(fragments),
        energy=float(energy),
        previous_energy=float(previous_energy),
        orbitals_converged=bool(orbitals_converged),
    )

    reports = tuple(
        _report_from_metadata(report, index)
        for index, report in enumerate(report_payloads)
    )

    if any(not isinstance(metadata, dict) for metadata in solver_metadata):
        raise ValueError("LASSQD checkpoint solver metadata must be objects.")
    solver_indices = [metadata.get("index") for metadata in solver_metadata]
    if (
        any(not isinstance(index, int) for index in solver_indices)
        or len(set(solver_indices)) != len(solver_indices)
        or any(not 0 <= index < len(state.fragments) for index in solver_indices)
    ):
        raise ValueError("LASSQD checkpoint has invalid solver fragment indices.")
    validation_rng = np.random.default_rng()
    validation_rng.bit_generator.state = rng_state
    for metadata in solver_metadata:
        solver_rng_state = metadata.get("rng_state")
        if not isinstance(solver_rng_state, dict):
            raise ValueError("LASSQD checkpoint is missing solver RNG state.")
        validation_rng.bit_generator.state = solver_rng_state

    return _WorkflowSnapshot(
        state=state,
        rng_state=rng_state,
        solver_rng_states={
            metadata["index"]: metadata["rng_state"] for metadata in solver_metadata
        },
        energy_history=tuple(energy_history.astype(float).tolist()),
        round_reports=reports,
        final_subspace=final_subspace,
        reaction_field=reaction_field,
    )
