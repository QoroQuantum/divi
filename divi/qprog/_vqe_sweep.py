# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from itertools import product
from typing import Literal, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

from divi.qprog import VQE, ProgramBatch, VQEAnsatz

from .optimizers import Optimizer


def _ctor_attrs(obj):
    sig = inspect.signature(obj.__class__.__init__)
    arg_names = list(sig.parameters.keys())[1:]  # skip 'self'
    return {name: getattr(obj, name) for name in arg_names if hasattr(obj, name)}


class _ZMatrixEntry(NamedTuple):
    bond_ref: int | None
    angle_ref: int | None
    dihedral_ref: int | None
    bond_length: float | None
    angle: float | None
    dihedral: float | None


# --- Helper functions ---
def _safe_normalize(v, fallback=None):
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        if fallback is None:
            fallback = np.array([1.0, 0.0, 0.0])
        return fallback / np.linalg.norm(fallback)
    return v / norm


def _compute_angle(v1, v2):
    dot = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


def _compute_dihedral(b0, b1, b2):
    n1 = np.cross(b0, b1)
    n2 = np.cross(b0, b2)
    if np.linalg.norm(n1) < 1e-6 or np.linalg.norm(n2) < 1e-6:
        return 0.0
    dot = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    if np.dot(b1, n2) < 0:
        angle *= -1
    return angle


def _find_refs(adj, placed, parent, child):
    gp = next((n for n in adj[parent] if n != child and n in placed), -1)
    ggp = -1
    if gp != -1:
        ggp = next((n for n in adj[gp] if n != parent and n in placed), -1)
    return gp if gp != -1 else None, ggp if ggp != -1 else None


# --- Main functions ---
def _cartesian_to_zmatrix(
    coords: np.ndarray, connectivity: list[tuple[int, int]]
) -> list[_ZMatrixEntry]:
    num_atoms = len(coords)
    if num_atoms == 0:
        raise ValueError(
            "Cannot convert empty coordinate array to Z-matrix: molecule must have at least one atom."
        )

    adj = [[] for _ in range(num_atoms)]
    for i, j in connectivity:
        adj[i].append(j)
        adj[j].append(i)

    zmatrix_entries = {0: _ZMatrixEntry(None, None, None, None, None, None)}
    q = deque([0])
    placed_atoms = {0}

    while q:
        parent_idx = q.popleft()
        for child_idx in adj[parent_idx]:
            if child_idx in placed_atoms:
                continue
            placed_atoms.add(child_idx)
            q.append(child_idx)

            bond_len = np.linalg.norm(coords[child_idx] - coords[parent_idx])
            gp, ggp = _find_refs(adj, placed_atoms, parent_idx, child_idx)

            angle = None
            if gp is not None:
                angle = _compute_angle(
                    coords[child_idx] - coords[parent_idx],
                    coords[gp] - coords[parent_idx],
                )

            dihedral = None
            if gp is not None and ggp is not None:
                dihedral = _compute_dihedral(
                    coords[parent_idx] - coords[gp],
                    coords[child_idx] - coords[parent_idx],
                    coords[ggp] - coords[gp],
                )

            zmatrix_entries[child_idx] = _ZMatrixEntry(
                parent_idx, gp, ggp, bond_len, angle, dihedral
            )

    return [zmatrix_entries[i] for i in range(num_atoms)]


def _zmatrix_to_cartesian(z_matrix: list[_ZMatrixEntry]) -> np.ndarray:
    n_atoms = len(z_matrix)
    coords = np.zeros((n_atoms, 3))

    if n_atoms == 0:
        return coords

    # --- First atom at origin ---
    coords[0] = np.array([0.0, 0.0, 0.0])

    # --- Second atom along +X axis ---
    if n_atoms > 1:
        coords[1] = np.array([z_matrix[1].bond_length, 0.0, 0.0])

    # --- Third atom in XY plane ---
    if n_atoms > 2:
        entry = z_matrix[2]
        r = entry.bond_length
        theta = np.radians(entry.angle) if entry.angle is not None else 0.0

        a1 = coords[entry.bond_ref]
        a2 = coords[entry.angle_ref]

        v = a2 - a1
        v /= np.linalg.norm(v)
        # fixed perpendicular in XY plane, fallback handled inline
        perp = np.array([-v[1], v[0], 0.0])
        perp /= np.linalg.norm(perp) if np.linalg.norm(perp) > 1e-6 else 1.0

        coords[2] = a1 + r * (np.cos(theta) * v + np.sin(theta) * perp)

    for i, entry in enumerate(z_matrix[3:], start=3):
        a1 = coords[entry.bond_ref] if entry.bond_ref is not None else np.zeros(3)
        a2 = coords[entry.angle_ref] if entry.angle_ref is not None else np.zeros(3)
        a3 = (
            coords[entry.dihedral_ref]
            if entry.dihedral_ref is not None
            else np.zeros(3)
        )

        r = entry.bond_length

        theta = np.radians(entry.angle) if entry.angle is not None else 0.0
        phi = np.radians(entry.dihedral) if entry.dihedral is not None else 0.0

        b1 = _safe_normalize(a1 - a2, fallback=np.array([1.0, 0.0, 0.0]))
        b2 = a3 - a2
        n = _safe_normalize(np.cross(b1, b2), fallback=np.array([0.0, 0.0, 1.0]))
        nc = np.cross(n, b1)

        coords[i] = a1 + r * (
            -np.cos(theta) * b1 + np.sin(theta) * (np.cos(phi) * nc + np.sin(phi) * n)
        )

    return coords


def _transform_bonds(
    zmatrix: list[_ZMatrixEntry],
    bonds_to_transform: list[tuple[int, int]],
    value: float,
    transform_type: Literal["scale", "delta"] = "scale",
) -> list[_ZMatrixEntry]:
    """
    Transform specified bonds in a Z-matrix.

    Args:
        zmatrix: List of _ZMatrixEntry.
        bonds_to_transform: List of (atom1, atom2) tuples specifying bonds.
        value: Multiplier or additive value.
        transform_type: "scale" or "add".

    Returns:
        New Z-matrix with transformed bond lengths.
    """
    # Convert to set of sorted tuples for quick lookup
    bonds_set = {tuple(sorted(b)) for b in bonds_to_transform}

    new_zmatrix = []
    for i, entry in enumerate(zmatrix):
        if (
            entry.bond_ref is not None
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


def _kabsch_align(P_in: np.ndarray, Q_in: np.ndarray, reference_atoms_idx=slice(None)):
    """
    Align point set P onto Q using the Kabsch algorithm.

    Parameters
    ----------
    P : (N, D) ndarray. Source coordinates.
    Q : (N, D) ndarray. Target coordinates.

    Returns
    -------
    P_aligned : (N, D) ndarray
        P rotated and translated onto Q.
    """

    P = P_in[reference_atoms_idx, :]
    Q = Q_in[reference_atoms_idx, :]

    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    # Centroids
    Pc = np.mean(P, axis=0)
    Qc = np.mean(Q, axis=0)

    # Centered coordinates
    P_centered = P - Pc
    Q_centered = Q - Qc

    # Covariance and SVD
    H = P_centered.T @ Q_centered
    U, _, Vt = np.linalg.svd(H)

    # Reflection check
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1] * (P.shape[1] - 1) + [d])

    # Optimal rotation and translation
    R = Vt.T @ D @ U.T
    t = Qc - Pc @ R

    # Apply transformation
    P_aligned = P_in @ R + t

    P_aligned[np.abs(P_aligned) < 1e-12] = 0.0

    return P_aligned


@dataclass(frozen=True, eq=True)
class MoleculeTransformer:
    """
    base_molecule: qml.qchem.Molecule
        The reference molecule used as a template for generating variants.
    bond_modifiers: Sequence[float]
        A list of values used to adjust bond lengths. The class will generate
        **one new molecule for each modifier** in this list. The modification
        mode is detected automatically:
            - **Scale mode**: If all values are positive, they are used as scaling
            factors (e.g., 1.1 for a 10% increase).
            - **Delta mode**: If any value is zero or negative, all values are
            treated as additive changes to the bond length, in Ångstroms.
    atom_connectivity: Sequence[tuple[int, int]] | None
        A sequence of atom index pairs specifying the bonds in the molecule.
        If not provided, a chain structure will be assumed
        e.g.: `[(0, 1), (1, 2), (2, 3), ...]`.
    bonds_to_transform: Sequence[tuple[int, int]] | None
        A subset of `atom_connectivity` that specifies the bonds to modify.
        If None, all bonds will be transformed.
    alignment_atoms: Sequence[int] | None
        Indices of atoms onto which to align the orientation of the resulting
        variants of the molecule. Only useful for visualization and debuggin.
        If None, no alignment is carried out.
    """

    base_molecule: qml.qchem.Molecule
    bond_modifiers: Sequence[float]
    atom_connectivity: Sequence[tuple[int, int]] | None = None
    bonds_to_transform: Sequence[tuple[int, int]] | None = None
    alignment_atoms: Sequence[int] | None = None

    def __post_init__(self):
        if not isinstance(self.base_molecule, qml.qchem.Molecule):
            raise ValueError(
                "`base_molecule` is expected to be a Pennylane `Molecule` instance."
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

        n_symbols = len(self.base_molecule.symbols)
        if self.atom_connectivity is None:
            object.__setattr__(
                self,
                "atom_connectivity",
                tuple(zip(range(n_symbols), range(1, n_symbols))),
            )
        else:
            if len(set(self.atom_connectivity)) < len(self.atom_connectivity):
                raise ValueError("`atom_connectivity` contains duplicate values.")

            if not all(
                0 <= a < n_symbols and 0 <= b < n_symbols
                for a, b in self.atom_connectivity
            ):
                raise ValueError(
                    "`atom_connectivity` should be a sequence of tuples of"
                    " atom indices in (0, len(molecule.symbols))"
                )

        if self.bonds_to_transform is None:
            object.__setattr__(self, "bonds_to_transform", self.atom_connectivity)
        else:
            if len(self.bonds_to_transform) == 0:
                raise ValueError("`bonds_to_transform` cannot be empty.")
            if not set(self.bonds_to_transform).issubset(self.atom_connectivity):
                raise ValueError(
                    "`bonds_to_transform` is not a subset of `atom_connectivity`"
                )

        if self.alignment_atoms is not None and not all(
            0 <= idx < n_symbols for idx in self.alignment_atoms
        ):
            raise ValueError(
                "`alignment_atoms` need to be in range (0, len(molecule.symbols))"
            )

    def generate(self) -> dict[float, qml.qchem.Molecule]:
        base_attrs = _ctor_attrs(self.base_molecule)

        variants = {}
        original_coords = self.base_molecule.coordinates
        mode = "scale" if all(v > 0 for v in self.bond_modifiers) else "delta"

        # Convert to Z-matrix, with connectivity
        z_matrix = _cartesian_to_zmatrix(original_coords, self.atom_connectivity)

        for value in self.bond_modifiers:
            if (value == 0 and mode == "delta") or (value == 1 and mode == "scale"):
                transformed_coords = original_coords.copy()
            else:
                transformed_z_matrix = _transform_bonds(
                    z_matrix, self.bonds_to_transform, value, mode
                )

                transformed_coords = _zmatrix_to_cartesian(transformed_z_matrix)

                if self.alignment_atoms is not None:
                    transformed_coords = _kabsch_align(
                        transformed_coords, original_coords, self.alignment_atoms
                    )

            # A single molecule is created after all bonds have been modified
            base_attrs["coordinates"] = transformed_coords
            mol = qml.qchem.Molecule(**base_attrs)
            variants[value] = mol

        return variants


class VQEHyperparameterSweep(ProgramBatch):
    """Allows user to carry out a grid search across different values
    for the ansatz and the bond length used in a VQE program.
    """

    def __init__(
        self,
        ansatze: Sequence[VQEAnsatz],
        molecule_transformer: MoleculeTransformer,
        optimizer: Optimizer = Optimizer.MONTE_CARLO,
        max_iterations: int = 10,
        **kwargs,
    ):
        """
        Initialize a VQE hyperparameter sweep.

        Parameters
        ----------
        ansatze: Sequence[VQEAnsatz]
            A sequence of ansatz circuits to test.
        molecule_transformer: MoleculeTransformer
            A `MoleculeTransformer` object defining the configuration for
            generating the molecule variants.
        optimizer: Optimizer
            The optimization algorithm for the VQE runs.
        max_iterations: int
            The maximum number of optimizer iterations for each VQE run.
        **kwargs: Forwarded to parent class.
        """
        super().__init__(backend=kwargs.pop("backend"))

        self.molecule_transformer = molecule_transformer

        self.ansatze = ansatze
        self.max_iterations = max_iterations

        self._constructor = partial(
            VQE,
            optimizer=optimizer,
            max_iterations=self.max_iterations,
            backend=self.backend,
            **kwargs,
        )

    def create_programs(self):
        super().create_programs()

        self.molecule_variants = self.molecule_transformer.generate()

        for ansatz, (modifier, molecule) in product(
            self.ansatze, self.molecule_variants.items()
        ):
            _job_id = (ansatz, modifier)
            self.programs[_job_id] = self._constructor(
                job_id=_job_id,
                molecule=molecule,
                ansatz=ansatz,
                losses=self._manager.list(),
                final_params=self._manager.list(),
                progress_queue=self._queue,
            )

    def aggregate_results(self):
        super().aggregate_results()

        all_energies = {key: prog.losses[-1] for key, prog in self.programs.items()}

        smallest_key = min(all_energies, key=lambda k: min(all_energies[k].values()))
        smallest_value = min(all_energies[smallest_key].values())

        return smallest_key, smallest_value

    def visualize_results(self, graph_type: Literal["line", "scatter"] = "line"):
        """
        Visualize the results of the VQE problem.
        """
        if graph_type not in ["line", "scatter"]:
            raise ValueError(
                f"Invalid graph type: {graph_type}. Choose between 'line' and 'scatter'."
            )

        if self._executor is not None:
            self.join()

        data = []
        colors = ["blue", "g", "r", "c", "m", "y", "k"]

        ansatz_list = list(VQEAnsatz)

        if graph_type == "scatter":
            for ansatz, modifier in self.programs.keys():
                min_energies = []

                curr_energies = self.programs[(ansatz, modifier)].losses[-1]
                min_energies.append(
                    (
                        modifier,
                        min(curr_energies.values()),
                        colors[ansatz_list.index(ansatz)],
                    )
                )
                data.extend(min_energies)

            x, y, z = zip(*data)
            plt.scatter(x, y, color=z, label=ansatz)

        elif graph_type == "line":
            for ansatz in self.ansatze:
                energies = []
                for modifier in self.molecule_transformer.bond_modifiers:
                    energies.append(
                        min(self.programs[(ansatz, modifier)].losses[-1].values())
                    )
                plt.plot(
                    self.molecule_transformer.bond_modifiers, energies, label=ansatz
                )

        plt.xlabel(
            "Scale Factor" if self.molecule_transformer._mode == "scale" else "Bond Δ"
        )
        plt.ylabel("Energy level")
        plt.legend()
        plt.show()
