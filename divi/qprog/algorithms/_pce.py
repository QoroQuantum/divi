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
from divi.qprog.variational_quantum_algorithm import (
    SolutionEntry,
    _merge_param_group_counts,
)

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
        self._use_soft_objective = self.alpha < 5.0
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

        for p_idx, ham_groups in self._group_results(results).items():
            # PCE ignores Hamiltonian and QEM ids; aggregate all shots for this parameter set.
            param_group = [
                ("0", shots)
                for qem_groups in ham_groups.values()
                for shots_list in qem_groups.values()
                for shots in shots_list
            ]

            state_strings, counts, total_shots = _aggregate_param_group(
                param_group, _merge_param_group_counts
            )

            parities = _decode_parities(state_strings, self._variable_masks_u64)
            if self._use_soft_objective:
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

    def get_top_solutions(
        self, n: int = 10, *, min_prob: float = 0.0, include_decoded: bool = False
    ) -> list[SolutionEntry]:
        """Get the top-N solutions sorted by probability, with decoded QUBO variable assignments.

        This method overrides the base implementation to decode encoded qubit states
        into actual QUBO variable assignments. The bitstrings in the probability
        distribution represent encoded qubit states (O(log2(N)) qubits), not the
        decoded QUBO solutions (N variables).

        Args:
            n (int): Maximum number of solutions to return. Must be non-negative.
                If n is 0 or negative, returns an empty list. If n exceeds the
                number of available solutions (after filtering), returns all
                available solutions. Defaults to 10.
            min_prob (float): Minimum probability threshold for including solutions.
                Only solutions with probability >= min_prob will be included.
                Must be in range [0.0, 1.0]. Defaults to 0.0 (no filtering).
            include_decoded (bool): Whether to populate the `decoded` field of
                each SolutionEntry with the numpy array representation of the
                QUBO solution. If False, the decoded field will be None.
                Defaults to False.

        Returns:
            list[SolutionEntry]: List of solution entries sorted by probability
                (descending), then by decoded bitstring (lexicographically ascending)
                for deterministic tie-breaking. The `bitstring` field contains
                the decoded QUBO solution as a binary string (e.g., "01011" for
                5 variables), not the encoded qubit state. Returns an empty list
                if no probability distribution is available or n <= 0.

        Raises:
            RuntimeError: If probability distribution is not available because
                optimization has not been run or final computation was not performed.
            ValueError: If min_prob is not in range [0.0, 1.0] or n is negative.

        Note:
            The probability distribution must be computed by running the algorithm
            with `perform_final_computation=True` (the default):

            >>> program.run(perform_final_computation=True)
            >>> top_10 = program.get_top_solutions(n=10)

        Example:
            >>> # Get top 5 solutions with probability >= 5%
            >>> program.run()
            >>> solutions = program.get_top_solutions(n=5, min_prob=0.05)
            >>> for sol in solutions:
            ...     print(f"{sol.bitstring}: {sol.prob:.2%}")
            01011: 42.50%  # Decoded QUBO solution (5 variables)
            10100: 31.20%
            ...

            >>> # Get solutions with numpy array in decoded field
            >>> solutions = program.get_top_solutions(n=3, include_decoded=True)
            >>> for sol in solutions:
            ...     print(f"{sol.bitstring} -> {sol.decoded}")
            01011 -> [0 1 0 1 1]  # numpy array
            ...
        """
        # Validate inputs
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if not (0.0 <= min_prob <= 1.0):
            raise ValueError(f"min_prob must be in range [0.0, 1.0], got {min_prob}")

        # Handle edge case: n == 0
        if n == 0:
            return []

        # Require probability distribution to exist
        if not self._best_probs:
            raise RuntimeError(
                "No probability distribution available. The final computation step "
                "must be performed to compute the probability distribution. "
                "Call run(perform_final_computation=True) to execute optimization "
                "and compute the distribution."
            )

        # Extract the probability distribution (nested by parameter set)
        # _best_probs structure: {tag: {bitstring: prob}}
        probs_dict = next(iter(self._best_probs.values()))

        # Filter by minimum probability and get top n sorted by probability (descending),
        # then bitstring (ascending) for deterministic tie-breaking
        top_items = sorted(
            filter(
                lambda bitstring_prob: bitstring_prob[1] >= min_prob, probs_dict.items()
            ),
            key=lambda bitstring_prob: (-bitstring_prob[1], bitstring_prob[0]),
        )[:n]

        # Decode each encoded qubit state to QUBO variable assignment
        encoded_bitstrings = [bitstring for bitstring, _ in top_items]
        decoded_parities = _decode_parities(
            encoded_bitstrings, self._variable_masks_u64
        )
        # decoded_parities shape: (n_vars, n_states), transpose to (n_states, n_vars)
        decoded_qubo_solutions = (
            1 - decoded_parities
        ).T  # Convert parities to QUBO assignments

        # Build result list with decoded solutions
        result = []
        for (encoded_bitstring, prob), decoded_solution in zip(
            top_items, decoded_qubo_solutions
        ):
            # Convert decoded solution to binary string representation
            decoded_bitstring = "".join(str(int(x)) for x in decoded_solution)

            result.append(
                SolutionEntry(
                    bitstring=decoded_bitstring,  # Decoded QUBO solution as string
                    prob=prob,
                    decoded=(
                        decoded_solution.astype(np.int32) if include_decoded else None
                    ),
                )
            )

        # Re-sort by decoded bitstring for deterministic tie-breaking
        # (in case multiple encoded states decode to the same QUBO solution)
        result.sort(key=lambda entry: (-entry.prob, entry.bitstring))

        return result

    @property
    def solution(self) -> npt.NDArray[np.integer]:
        """
        Returns the final optimized vector (hard binary 0/1) based on the best parameters found.
        You must run .run() before calling this.
        """
        if self._final_vector is None:
            raise RuntimeError("Run the VQE optimization first.")

        return self._final_vector
