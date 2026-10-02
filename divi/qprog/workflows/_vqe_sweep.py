# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import copy
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from itertools import product
from typing import Any, Literal, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from divi.hamiltonians._molecular import (
    is_pennylane_molecule,
    is_pyscf_mean_field,
    is_pyscf_mole,
    pyscf_mole_at,
)
from divi.qprog.algorithms import VQE, Ansatz
from divi.qprog.ensemble import ProgramEnsemble, ReportingLevel
from divi.qprog.optimizers import MonteCarloOptimizer, Optimizer
from divi.qprog.problems import HamiltonianProblem, MolecularProblem


def _normalise_molecule(molecule):
    """Reduce a PySCF mean field to the molecule its geometry belongs to."""
    return molecule.mol if is_pyscf_mean_field(molecule) else molecule


def _geometry_of(molecule) -> npt.NDArray:
    """Atomic coordinates in Bohr — the native unit of both molecule types."""
    if is_pyscf_mole(molecule):
        return np.asarray(molecule.atom_coords())
    return np.asarray(molecule.coordinates)


def _atom_count(molecule) -> int:
    if is_pyscf_mole(molecule):
        return int(molecule.natm)
    return len(molecule.symbols)


def _with_geometry(molecule, coordinates: npt.NDArray):
    """``molecule`` moved to ``coordinates`` (Bohr), leaving the original alone."""
    if is_pyscf_mole(molecule):
        return pyscf_mole_at(molecule, coordinates)
    variant = copy.copy(molecule)
    variant.coordinates = coordinates
    return variant


class _ZMatrixEntry(NamedTuple):
    bond_ref: int | None
    angle_ref: int | None
    dihedral_ref: int | None
    bond_length: float | None
    angle: float | None
    dihedral: float | None


_X_AXIS = np.array([1.0, 0.0, 0.0])
_Y_AXIS = np.array([0.0, 1.0, 0.0])
_Z_AXIS = np.array([0.0, 0.0, 1.0])
_COLLINEAR_TOL = 1e-6


def _safe_normalize(v, fallback=None):
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        if fallback is None:
            fallback = _X_AXIS
        return fallback / np.linalg.norm(fallback)
    return v / norm


def _compute_angle(v1, v2):
    dot = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


def _bfs_parents(
    n_atoms: int, connectivity: Sequence[tuple[int, int]]
) -> dict[int, int | None]:
    """BFS parent of every atom reachable from atom 0, in visiting order."""
    adj = [[] for _ in range(n_atoms)]
    for i, j in connectivity:
        adj[i].append(j)
        adj[j].append(i)

    parents: dict[int, int | None] = {0: None}
    queue = deque([0])
    while queue:
        parent = queue.popleft()
        for child in adj[parent]:
            if child not in parents:
                parents[child] = parent
                queue.append(child)
    return parents


def _is_collinear(a, b, c) -> bool:
    return bool(
        np.linalg.norm(np.cross(_safe_normalize(a - b), _safe_normalize(c - b)))
        < _COLLINEAR_TOL
    )


def _local_frame(bond_pos, angle_pos, dihedral_pos):
    """Orthonormal ``(b, m, n)`` frame for placing an atom bonded to ``bond_pos``.

    ``b`` points from ``angle_pos`` to ``bond_pos`` and ``n`` is normal to the
    plane of the three references. Without a usable dihedral reference, ``n``
    is the direction perpendicular to ``b`` closest to +Z.
    """
    b = _safe_normalize(bond_pos - angle_pos)
    normal = np.zeros(3)
    if dihedral_pos is not None:
        normal = np.cross(_safe_normalize(angle_pos - dihedral_pos), b)
    if np.linalg.norm(normal) < _COLLINEAR_TOL:
        normal = _safe_normalize(_Z_AXIS - np.dot(_Z_AXIS, b) * b, fallback=_Y_AXIS)
    n = _safe_normalize(normal)
    return b, np.cross(n, b), n


def _cartesian_to_zmatrix(
    coords: npt.NDArray[np.float64], connectivity: Sequence[tuple[int, int]]
) -> list[_ZMatrixEntry]:
    """Internal coordinates over the BFS spanning tree of ``connectivity``.

    Each atom is bonded to its BFS parent. The angle reference is the parent's
    own parent, or the first-placed sibling when the parent is the root. The
    dihedral reference is the first placed atom not collinear with the bond
    and angle references, preferring bonded neighbours. Rebuilding with
    :func:`_zmatrix_to_cartesian` reproduces ``coords`` up to a proper rigid
    motion, whatever the atom numbering.
    """
    num_atoms = len(coords)
    if num_atoms == 0:
        raise ValueError(
            "Cannot convert empty coordinate array to Z-matrix: molecule must have at least one atom."
        )

    parents = _bfs_parents(num_atoms, connectivity)
    order = list(parents)
    neighbours = [set() for _ in range(num_atoms)]
    for i, j in connectivity:
        neighbours[i].add(j)
        neighbours[j].add(i)

    entries = {order[0]: _ZMatrixEntry(None, None, None, None, None, None)}
    for k, atom in enumerate(order[1:], start=1):
        parent = parents[atom]
        assert parent is not None
        bond_len = float(np.linalg.norm(coords[atom] - coords[parent]))
        if k == 1:
            entries[atom] = _ZMatrixEntry(parent, None, None, bond_len, None, None)
            continue

        grandparent = parents[parent]
        angle_ref = grandparent if grandparent is not None else order[1]
        angle = float(
            _compute_angle(
                coords[atom] - coords[parent], coords[angle_ref] - coords[parent]
            )
        )
        if k == 2:
            entries[atom] = _ZMatrixEntry(
                parent, angle_ref, None, bond_len, angle, None
            )
            continue

        candidates = sorted(
            (a for a in order[:k] if a not in (parent, angle_ref)),
            key=lambda a: (a not in neighbours[angle_ref], a not in neighbours[parent]),
        )
        dihedral_ref = next(
            (
                a
                for a in candidates
                if not _is_collinear(coords[a], coords[angle_ref], coords[parent])
            ),
            candidates[0],
        )
        _, m, n = _local_frame(coords[parent], coords[angle_ref], coords[dihedral_ref])
        offset = coords[atom] - coords[parent]
        dihedral = float(np.degrees(np.arctan2(np.dot(offset, n), np.dot(offset, m))))
        entries[atom] = _ZMatrixEntry(
            parent, angle_ref, dihedral_ref, bond_len, angle, dihedral
        )

    return [entries[i] for i in range(num_atoms)]


def _place_atom(coords, entry: _ZMatrixEntry) -> npt.NDArray[np.float64]:
    if entry.bond_ref is None:
        return np.zeros(3)
    bond_pos = coords[entry.bond_ref]
    if entry.angle_ref is None:
        return bond_pos + entry.bond_length * _X_AXIS

    dihedral_pos = (
        coords[entry.dihedral_ref] if entry.dihedral_ref is not None else None
    )
    b, m, n = _local_frame(bond_pos, coords[entry.angle_ref], dihedral_pos)
    theta = np.radians(entry.angle or 0.0)
    phi = np.radians(entry.dihedral or 0.0)
    return bond_pos + entry.bond_length * (
        -np.cos(theta) * b + np.sin(theta) * (np.cos(phi) * m + np.sin(phi) * n)
    )


def _zmatrix_to_cartesian(z_matrix: list[_ZMatrixEntry]) -> npt.NDArray[np.float64]:
    """Cartesian coordinates of a Z-matrix, placing each atom after its references.

    The root sits at the origin, the atom with only a bond reference lies along
    +X from its partner, and the atom without a dihedral reference lies in the
    XY plane.
    """
    for i, entry in enumerate(z_matrix):
        if entry.bond_length is not None and entry.bond_length <= 0:
            raise ValueError(
                f"Bond length for atom {i} must be positive, got {entry.bond_length}"
            )

    coords = np.zeros((len(z_matrix), 3))
    placed: set[int] = set()
    pending = list(range(len(z_matrix)))
    while pending:
        ready = [
            i
            for i in pending
            if {z_matrix[i].bond_ref, z_matrix[i].angle_ref, z_matrix[i].dihedral_ref}
            <= placed | {None}
        ]
        if not ready:
            raise ValueError("Z-matrix references form a cycle.")
        for i in ready:
            coords[i] = _place_atom(coords, z_matrix[i])
        placed.update(ready)
        pending = [i for i in pending if i not in placed]

    return coords


def _transform_bonds(
    zmatrix: list[_ZMatrixEntry],
    bonds_to_transform: list[tuple[int, int]],
    value: float,
    transform_type: Literal["scale", "delta"],
) -> list[_ZMatrixEntry]:
    """
    Transform specified bonds in a Z-matrix.

    Args:
        zmatrix: List of _ZMatrixEntry.
        bonds_to_transform: List of (atom1, atom2) tuples specifying bonds.
        value: Multiplier or additive value.
        transform_type: "scale" or "delta".

    Returns:
        New Z-matrix with transformed bond lengths.
    """
    bonds_set = {tuple(sorted(b)) for b in bonds_to_transform}

    new_zmatrix = []
    for i, entry in enumerate(zmatrix):
        if (
            entry.bond_ref is not None
            and entry.bond_length is not None
            and tuple(sorted((i, entry.bond_ref))) in bonds_set
        ):
            old_length = entry.bond_length
            new_length = (
                old_length * value if transform_type == "scale" else old_length + value
            )
            if new_length == 0.0:
                raise RuntimeError(
                    "New bond length can't be zero after transformation."
                )
            new_zmatrix.append(entry._replace(bond_length=new_length))
        else:
            new_zmatrix.append(entry)
    return new_zmatrix


def _kabsch_align(
    P_in: npt.NDArray[np.float64],
    Q_in: npt.NDArray[np.float64],
    reference_atoms_idx=slice(None),
) -> npt.NDArray[np.float64]:
    """
    Align point set P onto Q using the Kabsch algorithm.

    Parameters
    ----------
    P : (N, D) npt.NDArray[np.float64]. Source coordinates.
    Q : (N, D) npt.NDArray[np.float64]. Target coordinates.

    Returns
    -------
    P_aligned : (N, D) npt.NDArray[np.float64]
        P rotated and translated onto Q.
    """

    P = P_in[reference_atoms_idx, :]
    Q = Q_in[reference_atoms_idx, :]

    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    Pc = np.mean(P, axis=0)
    Qc = np.mean(Q, axis=0)

    P_centered = P - Pc
    Q_centered = Q - Qc

    H = P_centered.T @ Q_centered
    U, _, Vt = np.linalg.svd(H)

    # Row vectors: P @ R ≈ Q. Flipping the weakest singular axis keeps det(R) = +1.
    D = np.eye(H.shape[0])
    D[-1, -1] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    t = Qc - Pc @ R

    P_aligned = P_in @ R + t

    P_aligned[np.abs(P_aligned) < 1e-12] = 0.0

    return P_aligned


@dataclass(frozen=True, eq=True)
class MoleculeTransformer:
    """
    A class for transforming molecular structures by modifying bond lengths.

    This class generates variants of a base molecule by adjusting bond lengths
    according to specified modifiers. The modification mode is detected automatically.

    Variants are emitted as the same type as ``base_molecule``, so a PennyLane
    molecule sweeps into PennyLane molecules and a PySCF one into PySCF ones.

    Attributes:
        base_molecule: The reference molecule used as a template for generating
            variants — a PennyLane ``qchem.Molecule`` or a PySCF ``gto.Mole``.
            A PySCF mean field is reduced to the molecule it was built from.
        bond_modifiers: A list of values used to adjust bond lengths. The class will generate
            **one new molecule for each modifier** in this list. The modification
            mode is detected automatically:
            - **Scale mode**: If all values are positive, they are used as scaling
            factors (e.g., 1.1 for a 10% increase).
            - **Delta mode**: If any value is zero or negative, all values are
            treated as additive changes to the bond length, in Bohr — the unit
            both PennyLane and PySCF store coordinates in.
        atom_connectivity: A sequence of atom index pairs specifying the bonds in the molecule.
            If not provided, a chain structure will be assumed
            e.g.: `[(0, 1), (1, 2), (2, 3), ...]`.
        bonds_to_transform: A subset of `atom_connectivity` that specifies the bonds to modify.
            If None, all bonds will be transformed.
        alignment_atoms: Indices of atoms onto which to align the orientation of the resulting
            variants of the molecule. Only useful for visualisation and debugging.
            If None, no alignment is carried out.
    """

    base_molecule: Any
    bond_modifiers: Sequence[float]
    atom_connectivity: Sequence[tuple[int, int]] | None = None
    bonds_to_transform: Sequence[tuple[int, int]] | None = None
    alignment_atoms: Sequence[int] | None = None

    _mode: Literal["scale", "delta"] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(
            self, "base_molecule", _normalise_molecule(self.base_molecule)
        )
        if not (
            is_pyscf_mole(self.base_molecule)
            or is_pennylane_molecule(self.base_molecule)
        ):
            raise ValueError(
                "`base_molecule` is expected to be a PennyLane `qchem.Molecule` "
                "or a PySCF `gto.Mole` instance."
            )

        if not all(isinstance(x, (float, int)) for x in self.bond_modifiers):
            raise ValueError("`bond_modifiers` should be a sequence of floats.")
        if len(set(self.bond_modifiers)) < len(self.bond_modifiers):
            raise ValueError("`bond_modifiers` contains duplicate values.")
        object.__setattr__(
            self,
            "_mode",
            "scale" if all(v > 0 for v in self.bond_modifiers) else "delta",
        )

        n_symbols = _atom_count(self.base_molecule)
        atom_connectivity: Sequence[tuple[int, int]]
        if self.atom_connectivity is None:
            atom_connectivity = tuple(zip(range(n_symbols), range(1, n_symbols)))
            object.__setattr__(self, "atom_connectivity", atom_connectivity)
        else:
            atom_connectivity = self.atom_connectivity
            if len(set(atom_connectivity)) < len(atom_connectivity):
                raise ValueError("`atom_connectivity` contains duplicate values.")

            if not all(
                0 <= a < n_symbols and 0 <= b < n_symbols for a, b in atom_connectivity
            ):
                raise ValueError(
                    "`atom_connectivity` should be a sequence of tuples of"
                    " atom indices in (0, len(molecule.symbols))"
                )
            if len(_bfs_parents(n_symbols, atom_connectivity)) < n_symbols:
                raise ValueError(
                    "`atom_connectivity` must connect every atom of the molecule"
                    " into one bonded structure."
                )

        if self.bonds_to_transform is None:
            object.__setattr__(self, "bonds_to_transform", atom_connectivity)
        else:
            bonds_to_transform = self.bonds_to_transform
            if len(bonds_to_transform) == 0:
                raise ValueError("`bonds_to_transform` cannot be empty.")
            if not set(bonds_to_transform).issubset(atom_connectivity):
                raise ValueError(
                    "`bonds_to_transform` is not a subset of `atom_connectivity`"
                )
            tree_bonds = {
                frozenset((child, parent))
                for child, parent in _bfs_parents(n_symbols, atom_connectivity).items()
                if parent is not None
            }
            ring_bonds = [
                b for b in bonds_to_transform if frozenset(b) not in tree_bonds
            ]
            if ring_bonds:
                raise ValueError(
                    f"Bonds {ring_bonds} close a ring in `atom_connectivity`; a "
                    "ring-closing bond cannot be transformed independently."
                )

        if self.alignment_atoms is not None and not all(
            0 <= idx < n_symbols for idx in self.alignment_atoms
        ):
            raise ValueError(
                "`alignment_atoms` need to be in range (0, len(molecule.symbols))"
            )

    def generate(self) -> dict[float, Any]:
        """Bond-modified variants, each the same molecule type as the base."""
        variants = {}
        original_coords = _geometry_of(self.base_molecule)
        mode = self._mode

        atom_connectivity = list(self.atom_connectivity or ())
        bonds_to_transform = list(self.bonds_to_transform or ())

        z_matrix = _cartesian_to_zmatrix(original_coords, atom_connectivity)

        for value in self.bond_modifiers:
            if (value == 0 and mode == "delta") or (value == 1 and mode == "scale"):
                transformed_coords = original_coords.copy()
            else:
                transformed_z_matrix = _transform_bonds(
                    z_matrix, bonds_to_transform, value, mode
                )

                transformed_coords = _zmatrix_to_cartesian(transformed_z_matrix)

                if self.alignment_atoms is not None:
                    transformed_coords = _kabsch_align(
                        transformed_coords, original_coords, self.alignment_atoms
                    )

            variants[value] = _with_geometry(self.base_molecule, transformed_coords)

        return variants


class VQEHyperparameterSweep(ProgramEnsemble):
    """Allows user to carry out a grid search across different values
    for the ansatz and the bond length used in a VQE program.
    """

    def __init__(
        self,
        ansatze: Sequence[Ansatz],
        molecule_transformer: MoleculeTransformer | None = None,
        problems: (
            Sequence[HamiltonianProblem] | Mapping[Any, HamiltonianProblem] | None
        ) = None,
        optimizer: Optimizer | None = None,
        max_iterations: int = 10,
        **kwargs,
    ):
        """
        Initialise a VQE hyperparameter sweep.

        Parameters
        ----------
        ansatze: Sequence[Ansatz]
            A sequence of ansatz circuits to test.
        problems: Sequence[HamiltonianProblem] | Mapping[Any, HamiltonianProblem], optional
            The problems to use for the VQE runs. A mapping keys each program
            by its own key instead of by position. If ``None`` (the default),
            the problems come from ``molecule_transformer``'s variants.
        molecule_transformer: MoleculeTransformer | None, optional
            A `MoleculeTransformer` object defining the configuration for
            generating the molecule variants, each solved as an
            :class:`~divi.qprog.problems.MolecularProblem`. If
            ``None`` (the default), the provided ``problems`` are used
            directly and no molecular transformation is performed.
        optimizer: Optimizer
            The optimisation algorithm for the VQE runs.
        max_iterations: int
            The maximum number of optimizer iterations for each VQE run.
        **kwargs
            Forwarded to the parent class. ``reporting_level`` accepts a
            :class:`~divi.qprog.ReportingLevel` controlling how much live
            progress is shown. ``sampling_backend`` optionally selects a
            separate backend for final solution sampling.
        """
        super().__init__(
            backend=kwargs.pop("backend"),
            sampling_backend=kwargs.pop("sampling_backend", None),
            reporting_level=kwargs.pop("reporting_level", ReportingLevel.COMPACT),
        )

        self.molecule_transformer = molecule_transformer
        self.ansatze = ansatze
        self.problems = problems
        self.max_iterations = max_iterations

        if molecule_transformer is not None and problems is not None:
            raise ValueError(
                "VQEHyperparameterSweep supports either a molecule sweep "
                "(via molecule_transformer) or a problem sweep (via problems), "
                "but not both."
            )

        if molecule_transformer is None and not problems:
            raise ValueError(
                "At least one of molecule_transformer or problems must be provided."
            )

        self._optimizer_template = (
            optimizer if optimizer is not None else MonteCarloOptimizer()
        )

        self._constructor = partial(
            VQE,
            max_iterations=self.max_iterations,
            backend=self.backend,
            **kwargs,
        )

    def create_programs(self, state=None):
        """
        Create VQE programs for all combinations of ansätze and molecule variants.

        Generates molecule variants using the configured MoleculeTransformer, then
        creates a VQE program for each (ansatz, molecule_variant) pair.

        Note:
            Program IDs are tuples of (ansatz_name, bond_modifier_value).
        """
        super().create_programs()

        if self.molecule_transformer is not None:
            sweep_items = [
                (modifier, MolecularProblem.from_molecule(molecule))
                for modifier, molecule in self.molecule_transformer.generate().items()
            ]
        elif isinstance(self.problems, Mapping):
            sweep_items = list(self.problems.items())
        else:
            sweep_items = list(enumerate(self.problems or ()))

        for ansatz, (item_id, problem) in product(self.ansatze, sweep_items):
            self._programs[(ansatz.name, item_id)] = self._constructor(
                problem,
                ansatz=ansatz,
                optimizer=self._optimizer_template.copy(),
            )

    def aggregate_results(self):
        """
        Find the best ansatz and bond configuration from all VQE runs.

        Compares the final energies across all ansatz/molecule combinations
        and returns the configuration that achieved the lowest ground state energy.

        Returns:
            tuple: A tuple containing:
                - best_config (tuple): (ansatz_name, bond_modifier) of the best result.
                - best_energy (float): The lowest energy achieved.

        Raises:
            RuntimeError: If programs haven't been run or have empty losses.
        """
        super().aggregate_results()

        all_energies = {key: prog.best_loss for key, prog in self.programs.items()}

        smallest_key = min(all_energies, key=lambda k: all_energies[k])
        smallest_value = all_energies[smallest_key]

        return smallest_key, smallest_value

    def visualize_results(self, graph_type: Literal["line", "scatter"] = "line"):
        """
        Visualise the results of the VQE problem.
        """
        if graph_type not in ["line", "scatter"]:
            raise ValueError(
                f"Invalid graph type: {graph_type}. Choose between 'line' and 'scatter'."
            )

        if (transformer := self.molecule_transformer) is None:
            raise RuntimeError(
                "visualize_results currently supports molecule-transformer sweeps only; "
                "visualisation for problem sweeps is not implemented."
            )

        if self._executor is not None:
            self.join()

        # Every configured ansatz, not only those whose programs completed.
        unique_ansatze = self.ansatze

        colors = ["blue", "g", "r", "c", "m", "y", "k"]
        color_map = {
            ansatz: colors[i % len(colors)] for i, ansatz in enumerate(unique_ansatze)
        }

        if graph_type == "scatter":
            for ansatz in unique_ansatze:
                modifiers = []
                energies = []
                for modifier in transformer.bond_modifiers:
                    program_key = (ansatz.name, modifier)
                    if program_key in self._programs:
                        modifiers.append(modifier)
                        energies.append(self._programs[program_key].best_loss)

                plt.scatter(
                    modifiers,
                    energies,
                    color=color_map[ansatz],
                    label=ansatz.name,
                )

        elif graph_type == "line":
            for ansatz in unique_ansatze:
                energies = []
                for modifier in transformer.bond_modifiers:
                    energies.append(self._programs[(ansatz.name, modifier)].best_loss)

                plt.plot(
                    transformer.bond_modifiers,
                    energies,
                    label=ansatz.name,
                    color=color_map[ansatz],
                )

        plt.xlabel("Scale Factor" if transformer._mode == "scale" else "Bond Δ")
        plt.ylabel("Energy level")
        plt.legend()
        plt.show()
