# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Literal
from warnings import warn

import numpy as np
import numpy.typing as npt
import pennylane as qml
import sympy as sp

from divi.circuits import MetaCircuit
from divi.pipeline import CircuitPipeline
from divi.pipeline.stages import CircuitSpecStage, ParameterBindingStage, PCECostStage
from divi.qprog.variational_quantum_algorithm import SolutionEntry
from divi.typing import QUBOProblemTypes, qubo_to_matrix

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
) -> tuple[list[str], npt.NDArray[np.float64], float]:
    """Aggregate a parameter group into states, counts, and total shots."""
    shots_dict: dict[str, int] = {}
    for _, histogram in param_group:
        for bitstring, count in histogram.items():
            shots_dict[bitstring] = shots_dict.get(bitstring, 0) + count
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


def _setup_dense_encoding(
    n_vars: int, n_qubits: int | None
) -> tuple[int, npt.NDArray[np.uint64]]:
    """Compute n_qubits and variable masks for dense (logarithmic) encoding."""
    min_qubits = int(np.ceil(np.log2(n_vars + 1)))
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
    n_q = n_qubits if n_qubits is not None else min_qubits
    masks = np.arange(1, n_vars + 1, dtype=np.uint64)
    return n_q, masks


def _setup_poly_encoding(
    n_vars: int, n_qubits: int | None
) -> tuple[int, npt.NDArray[np.uint64]]:
    """Compute n_qubits and variable masks for poly (weight-1 & 2) encoding."""
    discriminant = 1 + 8 * n_vars
    min_qubits = int(np.ceil((-1 + np.sqrt(discriminant)) / 2))

    if n_qubits is not None and n_qubits < min_qubits:
        raise ValueError(
            f"n_qubits must be >= {min_qubits} for poly encoding with {n_vars} vars. "
            f"Got n_qubits={n_qubits}."
        )
    if n_qubits is not None and n_qubits > min_qubits:
        warn(
            "n_qubits exceeds the minimum required; extra qubits increase circuit "
            "size and can add noise without representing more variables.",
            UserWarning,
        )
    n_q = n_qubits if n_qubits is not None else min_qubits

    max_capacity = n_q + (n_q * (n_q - 1)) // 2
    if n_vars > max_capacity:
        raise ValueError(
            f"Poly encoding with {n_q} qubits supports max {max_capacity} "
            f"vars; problem has {n_vars}. Increase n_qubits."
        )

    masks = []
    for i in range(n_q):
        masks.append(1 << i)
    for i in range(n_q):
        for j in range(i + 1, n_q):
            masks.append((1 << i) | (1 << j))

    return n_q, np.array(masks[:n_vars], dtype=np.uint64)


def _masks_to_ham_ops(variable_masks_u64: npt.NDArray[np.uint64], n_qubits: int) -> str:
    """Convert variable masks to semicolon-separated Pauli strings for expval backend.

    Each variable's parity is <Z on qubits in mask>. For mask with bit i set, include Z
    at wire i. Big Endian: qubit 0 is leftmost.
    """
    terms = []
    for mask in variable_masks_u64:
        m = int(mask)
        paulis = ["I"] * n_qubits
        for i in range(n_qubits):
            if (m >> i) & 1:
                paulis[i] = "Z"
        terms.append("".join(paulis))
    return ";".join(terms)


class PCE(VQE):
    """
    Generalized Pauli Correlation Encoding (PCE) VQE.

    Encodes an N-variable QUBO into qubits by mapping each variable to a parity
    (Pauli-Z correlation) of the measured bitstring. Qubit scaling depends on
    `encoding_type`: O(log2(N)) for dense, O(sqrt(N)) for poly. The algorithm
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
        encoding_type: Literal["dense", "poly"] = "dense",
        decode_parities_fn: (
            Callable[[list[str], npt.NDArray[np.uint64]], npt.NDArray[np.uint8]] | None
        ) = None,
        **kwargs,
    ):
        """
        Args:
            qubo_matrix (QUBOProblemTypes): The N x N matrix to minimize. Accepts
                a dense array, sparse matrix, list, or BinaryQuadraticModel.
            n_qubits (int | None): Optional override. Must be >= minimum for the
                chosen encoding (ceil(log2(N + 1)) for dense; solve n(n+1)/2 >= N
                for poly). Larger values raise a warning for both encodings.
            alpha (float): Scaling factor for the tanh() activation. Higher = harder
                binary constraints, Lower = smoother gradient.
            encoding_type (Literal["dense", "poly"]): "dense" (logarithmic qubits,
                default) or "poly" (each variable maps to parity of 1 or 2 qubits).
            decode_parities_fn (Callable | None): Optional custom decoder for mapping
                encoded bitstrings to parity arrays. Signature:
                (state_strings, variable_masks_u64) -> parities. Defaults to the
                built-in parity decoder.
            **kwargs: Additional arguments passed to VQE (e.g. ansatz, backend).
        """

        self.qubo_matrix = qubo_to_matrix(qubo_matrix)
        self.n_vars = self.qubo_matrix.shape[0]
        self.alpha = alpha
        self.encoding_type = encoding_type
        self._use_soft_objective = self.alpha < 5.0
        self._final_vector: npt.NDArray[np.integer] | None = None
        self._decode_parities_fn = decode_parities_fn or _decode_parities

        if kwargs.get("qem_protocol") is not None:
            raise ValueError("PCE does not currently support qem_protocol.")

        if self.encoding_type == "dense":
            self.n_qubits, self._variable_masks_u64 = _setup_dense_encoding(
                self.n_vars, n_qubits
            )
        elif self.encoding_type == "poly":
            self.n_qubits, self._variable_masks_u64 = _setup_poly_encoding(
                self.n_vars, n_qubits
            )
        else:
            raise ValueError(f"Unknown encoding_type: {self.encoding_type}")

        # Placeholder Hamiltonian required by VQE; we care about the measurement
        # probability distribution, and Z-basis measurements provide it.
        placeholder_hamiltonian = qml.Hamiltonian(
            [1.0] * self.n_qubits, [qml.PauliZ(i) for i in range(self.n_qubits)]
        )
        # PCE uses its own PCECostStage, so disable base-class grouping
        # to suppress the misleading "overriding grouping_strategy" warning.
        kwargs.setdefault("grouping_strategy", None)
        super().__init__(hamiltonian=placeholder_hamiltonian, **kwargs)

    def _build_pipelines(self) -> None:
        # Override VQE's cost pipeline: use PCECostStage instead of
        # MeasurementStage.  PCECostStage has the same expand
        # (Z-basis measurement QASMs) but sets ResultFormat.COUNTS and its
        # reduce applies the nonlinear QUBO energy formula instead of the
        # linear Hamiltonian combination.
        # QEMStage is intentionally excluded — ZNE is not applicable to
        # counts-based measurements.
        self._cost_pipeline = CircuitPipeline(
            stages=[
                CircuitSpecStage(),
                PCECostStage(
                    qubo_matrix=self.qubo_matrix,
                    alpha=self.alpha,
                    use_soft_objective=self._use_soft_objective,
                    decode_parities_fn=self._decode_parities_fn,
                    variable_masks_u64=self._variable_masks_u64,
                ),
                ParameterBindingStage(),
            ]
        )
        self._measurement_pipeline = self._build_measurement_pipeline()

    def _run_optimization_circuits(self, **kwargs) -> dict[int, float]:
        """Run cost evaluation via the pipeline."""
        if not self._use_soft_objective and self.backend.supports_expval:
            raise ValueError(
                "PCE with alpha >= 5.0 (hard CVaR mode) requires shot histograms and "
                "cannot use expectation-value backends. Use a sampling backend or set "
                "force_sampling=True in JobConfig when using QoroService."
            )
        return super()._run_optimization_circuits(**kwargs)

    def _create_meta_circuit_factories(self) -> dict[str, MetaCircuit]:
        """Create meta-circuit factories, handling the edge case of zero parameters."""
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
            "cost_circuit": MetaCircuit(
                source_circuit=qml.tape.QuantumScript(
                    ops=ops, measurements=[qml.expval(self._cost_hamiltonian)]
                ),
                symbols=weights_syms.flatten(),
                precision=self._precision,
            ),
            "meas_circuit": MetaCircuit(
                source_circuit=qml.tape.QuantumScript(
                    ops=ops, measurements=[qml.probs()]
                ),
                symbols=weights_syms.flatten(),
                precision=self._precision,
            ),
        }

    def _perform_final_computation(self, **kwargs) -> None:
        """Compute the final eigenstate and decode it into a PCE vector."""
        super()._perform_final_computation(**kwargs)

        if self._eigenstate is None:
            self._final_vector = None
            return

        best_bitstring = "".join(str(x) for x in self._eigenstate)
        parities = self._decode_parities_fn(
            [best_bitstring], self._variable_masks_u64
        ).flatten()
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
        decoded_parities = self._decode_parities_fn(
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
