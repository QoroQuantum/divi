# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from warnings import warn

import numpy as np
import numpy.typing as npt
import pennylane as qml
import sympy as sp

from divi.circuits import MetaCircuit
from divi.qprog.typing import QUBOProblemTypes, qubo_to_matrix

from ._vqe import VQE

# Pre-computed 8-bit popcount table for O(1) lookups
_POPCOUNT_TABLE_8BIT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _fast_popcount_parity(arr_input: npt.NDArray[np.integer]) -> npt.NDArray[np.uint8]:
    """
    Vectorized calculation of (popcount % 2) for an array of integers.
    Uses numpy view casting for extreme speed over large arrays.
    """
    # 1. Ensure array is uint64
    arr_u64 = arr_input.astype(np.uint64)

    # 2. View as bytes to use 8-bit lookup table
    arr_bytes = arr_u64.view(np.uint8).reshape(arr_input.shape + (8,))

    # 3. Lookup and sum bits
    total_bits = _POPCOUNT_TABLE_8BIT[arr_bytes].sum(axis=-1)

    # 4. Return Parity (0 or 1)
    return total_bits % 2


def _aggregate_param_group(
    param_group: list[tuple[str, dict[str, int]]],
    merge_counts_fn,
) -> tuple[list[str], npt.NDArray[np.float64], float]:
    """Aggregate a parameter group into states, counts, and total shots."""
    shots_dict = merge_counts_fn(param_group)
    state_strings = list(shots_dict.keys())
    counts = np.array(list(shots_dict.values()), dtype=float)
    total_shots = counts.sum()
    return state_strings, counts, float(total_shots)


def _decode_parities(
    state_strings: list[str], variable_masks_u64: npt.NDArray[np.uint64]
) -> npt.NDArray[np.uint8]:
    """Decode bitstring parities using the precomputed variable masks."""
    states = np.array([int(s, 2) for s in state_strings], dtype=np.uint64)
    overlaps = variable_masks_u64[:, None] & states[None, :]
    return _fast_popcount_parity(overlaps)


def _compute_soft_energy(
    parities: npt.NDArray[np.uint8],
    probs: npt.NDArray[np.float64],
    alpha: float,
    qubo_matrix: npt.NDArray[np.float64] | np.ndarray,
) -> float:
    """Compute the relaxed (soft) QUBO energy from parity expectations."""
    mean_parities = parities.dot(probs)
    z_expectations = 1.0 - (2.0 * mean_parities)
    x_soft = 0.5 * (1.0 + np.tanh(alpha * z_expectations))
    Qx = qubo_matrix @ x_soft
    return float(np.dot(x_soft, Qx))


def _compute_hard_cvar_energy(
    parities: npt.NDArray[np.uint8],
    counts: npt.NDArray[np.float64],
    total_shots: float,
    qubo_matrix: npt.NDArray[np.float64] | np.ndarray,
    alpha_cvar: float = 0.25,
) -> float:
    """Compute CVaR energy from sampled hard assignments."""
    x_vals = 1.0 - parities.astype(float)
    Qx = qubo_matrix @ x_vals
    energies = np.einsum("ij,ij->j", x_vals, Qx)

    sorted_indices = np.argsort(energies)
    sorted_energies = energies[sorted_indices]
    sorted_counts = counts[sorted_indices]

    cutoff_count = int(np.ceil(alpha_cvar * total_shots))
    accumulated_counts = np.cumsum(sorted_counts)
    limit_idx = np.searchsorted(accumulated_counts, cutoff_count)

    cvar_energy = 0.0
    count_sum = 0
    if limit_idx > 0:
        cvar_energy += np.sum(sorted_energies[:limit_idx] * sorted_counts[:limit_idx])
        count_sum += np.sum(sorted_counts[:limit_idx])

    remaining = cutoff_count - count_sum
    cvar_energy += sorted_energies[limit_idx] * remaining
    return float(cvar_energy / cutoff_count)


class PCE(VQE):
    """
    Generalized Pauli Correlation Encoding (PCE) VQE.

    Encodes an N-variable QUBO into O(log2(N)) qubits by mapping each variable
    to a parity (Pauli-Z correlation) of the measured bitstring. The algorithm
    uses the measurement distribution to estimate these parities, applies a
    smooth relaxation when `alpha` is small, and evaluates the classical QUBO
    objective: E = x.T @ Q @ x. For larger `alpha`, it switches to a discrete
    objective (CVaR over sampled energies) for harder convergence.
    """

    def __init__(
        self,
        qubo_matrix: QUBOProblemTypes,
        n_qubits: int | None = None,
        alpha: float = 2.0,
        **kwargs,
    ):
        """
        Args:
            qubo_matrix (QUBOProblemTypes): The N x N matrix to minimize. Accepts
                a dense array, sparse matrix, list, or BinaryQuadraticModel.
            n_qubits (int | None): Optional override. Must be >= ceil(log2(N)).
                Larger values increase circuit size without adding representational power.
            alpha (float): Scaling factor for the tanh() activation. Higher = harder
                binary constraints, Lower = smoother gradient.
        """

        self.qubo_matrix = qubo_to_matrix(qubo_matrix)
        self.n_vars = self.qubo_matrix.shape[0]
        self.alpha = alpha
        self._final_vector: npt.NDArray[np.integer] | None = None

        if kwargs.get("qem_protocol") is not None:
            raise ValueError("PCE does not currently support qem_protocol.")

        # Calculate required qubits (Logarithmic Scaling)
        min_qubits = int(np.ceil(np.log2(self.n_vars + 1)))
        if n_qubits is not None and n_qubits < min_qubits:
            raise ValueError(
                "n_qubits must be >= ceil(log2(N + 1)) to represent all variables. "
                f"Got n_qubits={n_qubits}, minimum={min_qubits}."
            )
        if n_qubits is not None and n_qubits > min_qubits:
            warn(
                "n_qubits exceeds the minimum required; extra qubits increase circuit "
                "size and can add noise without representing more variables.",
                UserWarning,
            )
        self.n_qubits = n_qubits if n_qubits is not None else min_qubits

        # Pre-compute U64 masks for the fast broadcasting step later
        self._variable_masks_u64 = np.arange(1, self.n_vars + 1, dtype=np.uint64)

        # Placeholder Hamiltonian required by VQE; we care about the measurement
        # probability distribution, and Z-basis measurements provide it.
        placeholder_hamiltonian = qml.Hamiltonian(
            [1.0] * self.n_qubits, [qml.PauliZ(i) for i in range(self.n_qubits)]
        )
        super().__init__(hamiltonian=placeholder_hamiltonian, **kwargs)

    def _create_meta_circuits_dict(self) -> dict[str, MetaCircuit]:
        """Create meta circuits, handling the edge case of zero parameters."""
        n_params = self.ansatz.n_params_per_layer(
            self.n_qubits, n_electrons=self.n_electrons
        )

        weights_syms = sp.symarray("w", (self.n_layers, n_params))

        ops = self.ansatz.build(
            weights_syms,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            n_electrons=self.n_electrons,
        )

        return {
            "cost_circuit": self._meta_circuit_factory(
                qml.tape.QuantumScript(
                    ops=ops, measurements=[qml.expval(self._cost_hamiltonian)]
                ),
                symbols=weights_syms.flatten(),
            ),
            "meas_circuit": self._meta_circuit_factory(
                qml.tape.QuantumScript(ops=ops, measurements=[qml.probs()]),
                symbols=weights_syms.flatten(),
                grouping_strategy="wires",
            ),
        }

    def _post_process_results(
        self, results: dict[str, dict[str, int]]
    ) -> dict[int, float]:
        """
        Calculates loss.
        If self.alpha < 5.0, computes 'Soft Energy' (Relaxed VQE) for smooth gradients.
        If self.alpha >= 5.0, computes 'Hard CVaR Energy' for final convergence.
        """

        # Return raw probabilities if requested (skip processing)
        if getattr(self, "_is_compute_probabilities", False):
            return super()._post_process_results(results)

        losses = {}

        # Hard cutoff for switching strategies
        USE_SOFT_OBJECTIVE = self.alpha < 5.0

        for p_idx, qem_groups in self._group_results(results).items():
            # PCE ignores QEM ids; aggregate all shots for this parameter set.
            param_group = [
                ("0", shots)
                for shots_list in qem_groups.values()
                for shots in shots_list
            ]

            state_strings, counts, total_shots = _aggregate_param_group(
                param_group, self._merge_param_group_counts
            )

            parities = _decode_parities(state_strings, self._variable_masks_u64)
            if USE_SOFT_OBJECTIVE:
                probs = counts / total_shots
                losses[p_idx] = _compute_soft_energy(
                    parities, probs, self.alpha, self.qubo_matrix
                )
            else:
                losses[p_idx] = _compute_hard_cvar_energy(
                    parities, counts, total_shots, self.qubo_matrix
                )

        return losses

    def _perform_final_computation(self, **kwargs) -> None:
        """Compute the final eigenstate and decode it into a PCE vector."""
        super()._perform_final_computation(**kwargs)

        if self._eigenstate is None:
            self._final_vector = None
            return

        best_bitstring = "".join(str(x) for x in self._eigenstate)
        state_int = int(best_bitstring, 2)
        state_u64 = np.array([state_int], dtype=np.uint64)

        overlaps = self._variable_masks_u64[:, None] & state_u64[None, :]
        parities = _fast_popcount_parity(overlaps).flatten()
        self._final_vector = 1 - parities

    @property
    def solution(self) -> npt.NDArray[np.integer]:
        """
        Returns the final optimized vector (hard binary 0/1) based on the best parameters found.
        You must run .run() before calling this.
        """
        if self._final_vector is None:
            raise RuntimeError("Run the VQE optimization first.")

        return self._final_vector
