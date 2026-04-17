# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Literal
from warnings import warn

import numpy as np
import numpy.typing as npt
import pennylane as qml
from qiskit.circuit import ParameterVector

from divi.circuits import MetaCircuit, qscript_to_meta
from divi.hamiltonians import normalize_binary_polynomial_problem
from divi.pipeline import CircuitPipeline
from divi.pipeline.stages import CircuitSpecStage, ParameterBindingStage, PCECostStage
from divi.pipeline.stages._numba_kernels import compile_problem
from divi.pipeline.stages._pce_cost_stage import _evaluate_binary_polynomial
from divi.qprog.algorithms._numba_kernels import _popcount_parity_jit
from divi.qprog.algorithms._vqe import VQE
from divi.qprog.variational_quantum_algorithm import SolutionEntry
from divi.typing import BinaryPolynomialProblem, HUBOProblemTypes, QUBOProblemTypes


def _fast_popcount_parity(arr_input: npt.NDArray[np.integer]) -> npt.NDArray[np.uint8]:
    """
    Vectorized calculation of (popcount % 2) for an array of integers.
    Uses a Numba JIT kernel with XOR-fold bit manipulation.
    """
    return _popcount_parity_jit(arr_input.astype(np.uint64))


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
        problem: QUBOProblemTypes | HUBOProblemTypes,
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
            problem (QUBOProblemTypes | HUBOProblemTypes): Binary polynomial
                objective to minimize. Supports QUBO and HUBO inputs.
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
        self.problem: BinaryPolynomialProblem = normalize_binary_polynomial_problem(
            problem
        )
        self.n_vars = self.problem.n_vars
        self.alpha = alpha
        self.encoding_type = encoding_type
        self._use_soft_objective = self.alpha < 5.0
        self._final_vector: npt.NDArray[np.integer] | None = None
        self._decode_parities_fn = decode_parities_fn or _decode_parities
        self._compiled_problem = compile_problem(self.problem)

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
        # PCE replaces the cost pipeline with PCECostStage (a standalone
        # BundleStage), so VQE's grouping_strategy is irrelevant for cost
        # evaluation.  Pop it to avoid the "overriding grouping_strategy"
        # warning from VQE.__init__ when the backend supports expval.
        kwargs.pop("grouping_strategy", None)
        super().__init__(hamiltonian=placeholder_hamiltonian, **kwargs)

    def _build_pipelines(self) -> None:
        """Build the PCE-specific cost and measurement pipelines."""
        # PCECostStage is a standalone BundleStage (not a MeasurementStage
        # subclass) that emits one "measure all qubits" QASM per circuit
        # spec and computes the nonlinear binary-polynomial objective from
        # raw shot histograms. QEMStage is intentionally excluded.
        self._cost_pipeline = CircuitPipeline(
            stages=[
                CircuitSpecStage(),
                PCECostStage(
                    problem=self.problem,
                    alpha=self.alpha,
                    use_soft_objective=self._use_soft_objective,
                    decode_parities_fn=self._decode_parities_fn,
                    variable_masks_u64=self._variable_masks_u64,
                ),
                ParameterBindingStage(),
            ]
        )
        self._measurement_pipeline = self._build_measurement_pipeline()

    def _evaluate_cost_param_sets(
        self, param_sets: np.ndarray, **kwargs
    ) -> dict[int, float]:
        """Evaluate the cost pipeline for the provided parameter sets."""
        if not self._use_soft_objective and self.backend.supports_expval:
            raise ValueError(
                "PCE with alpha >= 5.0 (hard CVaR mode) requires shot histograms and "
                "cannot use expectation-value backends. Use a sampling backend or set "
                "force_sampling=True in JobConfig when using QoroService."
            )
        return super()._evaluate_cost_param_sets(param_sets, **kwargs)

    def _create_meta_circuit_factories(self) -> dict[str, MetaCircuit]:
        """Create meta-circuit factories, handling the edge case of zero parameters."""
        n_params = self.ansatz.n_params_per_layer(
            self.n_qubits, n_electrons=self.n_electrons
        )

        weights = np.array(
            [ParameterVector(f"w_{i}", n_params) for i in range(self.n_layers)],
            dtype=object,
        )

        ops = self.ansatz.build(
            weights,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            n_electrons=self.n_electrons,
        )

        flat_params = tuple(weights.flatten())
        return {
            "cost_circuit": qscript_to_meta(
                qml.tape.QuantumScript(
                    ops=ops, measurements=[qml.expval(self._cost_hamiltonian)]
                ),
                precision=self._precision,
                parameter_order=flat_params,
            ),
            "meas_circuit": qscript_to_meta(
                qml.tape.QuantumScript(ops=ops, measurements=[qml.probs()]),
                precision=self._precision,
                parameter_order=flat_params,
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
        self,
        n: int = 10,
        *,
        min_prob: float = 0.0,
        include_decoded: bool = False,
        sort_by: Literal["prob", "energy"] = "prob",
    ) -> list[SolutionEntry]:
        """Get the top-N solutions with decoded QUBO variable assignments.

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
            sort_by: Sort order for the returned solutions.
                ``"prob"`` (default): descending by probability.
                ``"energy"``: ascending by objective energy. When set, the
                ``energy`` field of each ``SolutionEntry`` is populated.

        Returns:
            list[SolutionEntry]: List of solution entries sorted according to
                ``sort_by``, with decoded bitstring for deterministic tie-breaking.
                The `bitstring` field contains the decoded QUBO solution as a
                binary string (e.g., "01011" for 5 variables), not the encoded
                qubit state. Returns an empty list if no probability distribution
                is available or n <= 0.

        Raises:
            RuntimeError: If probability distribution is not available because
                optimization has not been run or final computation was not performed.
            ValueError: If min_prob is not in range [0.0, 1.0], n is negative,
                or sort_by is not one of ``"prob"`` or ``"energy"``.
        """
        # Validate inputs
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if not (0.0 <= min_prob <= 1.0):
            raise ValueError(f"min_prob must be in range [0.0, 1.0], got {min_prob}")
        if sort_by not in ("prob", "energy"):
            raise ValueError(f"sort_by must be 'prob' or 'energy', got '{sort_by}'")

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

        # Filter by minimum probability
        filtered = [(bs, prob) for bs, prob in probs_dict.items() if prob >= min_prob]

        # Decode all filtered encoded qubit states to QUBO variable assignments
        encoded_bitstrings = [bs for bs, _ in filtered]
        decoded_parities = self._decode_parities_fn(
            encoded_bitstrings, self._variable_masks_u64
        )
        # decoded_parities shape: (n_vars, n_states), transpose to (n_states, n_vars)
        decoded_qubo_solutions = (1 - decoded_parities).T

        # Build full result list with decoded solutions and optional energy
        compute_energy = sort_by == "energy"
        result = []
        for (encoded_bitstring, prob), decoded_solution in zip(
            filtered, decoded_qubo_solutions
        ):
            decoded_bitstring = "".join(str(int(x)) for x in decoded_solution)
            energy = (
                float(
                    _evaluate_binary_polynomial(
                        decoded_solution.astype(float),
                        self.problem,
                        _compiled=self._compiled_problem,
                    )
                )
                if compute_energy
                else None
            )
            result.append(
                SolutionEntry(
                    bitstring=decoded_bitstring,
                    prob=prob,
                    decoded=(
                        decoded_solution.astype(np.int32) if include_decoded else None
                    ),
                    energy=energy,
                )
            )

        # Sort and take top n
        if sort_by == "energy":
            result.sort(key=lambda e: (e.energy, e.bitstring))
        else:
            result.sort(key=lambda e: (-e.prob, e.bitstring))

        return result[:n]

    @property
    def solution(self) -> npt.NDArray[np.integer] | dict:
        """Return the most-probable decoded assignment from the final measurement.

        .. note::

            This returns the assignment corresponding to the **highest-probability**
            encoded bitstring, which may not be the lowest-energy solution.
            For energy-ranked solutions, use
            :meth:`get_top_solutions(sort_by="energy") <get_top_solutions>` instead.

        Returns:
            For QUBO problems, a binary 0/1 NumPy array. For HUBO problems
            with non-integer variable names, a dictionary mapping variable
            names to binary values.

        Raises:
            RuntimeError: If ``run()`` has not been called yet.
        """
        if self._final_vector is None:
            raise RuntimeError("Run the VQE optimization first.")

        warn(
            "PCE.solution returns the decoded assignment of the most-probable "
            "encoded bitstring. Because PCE operates in a compressed qubit space "
            "(O(log2(N)) qubits for N variables), the most-probable encoded state "
            "does not necessarily decode to the lowest-energy QUBO solution. "
            "Use get_top_solutions(sort_by='energy') for energy-ranked results.",
            stacklevel=2,
        )

        vo = self.problem.variable_order
        if vo != tuple(range(self.problem.n_vars)):
            return dict(zip(vo, self._final_vector))
        return self._final_vector
