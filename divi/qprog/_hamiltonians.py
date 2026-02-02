# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from functools import reduce
from typing import Literal, Protocol
from warnings import warn

import numpy as np
import numpy.typing as npt
import pennylane as qml
import scipy.sparse as sps


def _clean_hamiltonian(
    hamiltonian: qml.operation.Operator,
) -> tuple[qml.operation.Operator, float]:
    """Separate constant and non-constant terms in a Hamiltonian.

    This function processes a PennyLane Hamiltonian to separate out any terms
    that are constant (i.e. proportional to the identity operator). The sum
    of these constant terms is returned, along with a new Hamiltonian containing
    only the non-constant terms.

    Args:
        hamiltonian: The Hamiltonian operator to process.

    Returns:
        tuple[qml.operation.Operator, float]: A tuple containing:
            - The Hamiltonian without the constant (identity) component.
            - The summed value of all constant terms.
    """

    terms = (
        hamiltonian.operands if isinstance(hamiltonian, qml.ops.Sum) else [hamiltonian]
    )

    constant = 0.0
    non_id_terms = []

    for term in terms:
        coeff = 1.0
        base_op = term
        if isinstance(term, qml.ops.SProd):
            coeff = term.scalar
            base_op = term.base

        # Check for Identity term
        is_constant = False
        if isinstance(base_op, qml.Identity):
            is_constant = True
        elif isinstance(base_op, qml.ops.Prod) and all(
            isinstance(op, qml.Identity) for op in base_op.operands
        ):
            is_constant = True

        if is_constant:
            constant += coeff
        else:
            non_id_terms.append(term)

    if not non_id_terms:
        return qml.Hamiltonian([], []), float(constant)

    # Reconstruct the Hamiltonian from non-constant terms
    if len(non_id_terms) > 1:
        new_hamiltonian = qml.sum(*non_id_terms)
    else:
        new_hamiltonian = non_id_terms[0]

    return new_hamiltonian.simplify(), float(constant)


def _sort_hamiltonian_terms(
    hamiltonian: qml.operation.Operator,
    order: Literal["absolute", "magnitude"] = "absolute",
) -> qml.operation.Operator:
    """Sort the terms of a Hamiltonian by their coefficient magnitude."""
    coeffs, terms = hamiltonian.terms()
    sorted_coeffs, sorted_terms = zip(
        *sorted(
            zip(coeffs, terms), key=lambda x: x[0] if order == "absolute" else abs(x[0])
        )
    )
    return type(hamiltonian)(
        *[cf * trm for cf, trm in zip(sorted_coeffs, sorted_terms)]
    )


class TrotterizationStrategy(Protocol):
    """Trotterization strategy protocol."""

    @property
    def stateful(self) -> bool:
        """True if the strategy retains state across process_hamiltonian calls.
        This should be true for strategies that might re-process the Hamiltonian during execution.
        """
        ...

    def process_hamiltonian(
        self, hamiltonian: qml.operation.Operator
    ) -> qml.operation.Operator:
        """Trotterize the Hamiltonian."""
        ...


@dataclass(frozen=True)
class ExactTrotterization(TrotterizationStrategy):
    """Exact Trotterization strategy."""

    keep_fraction: float | None = None
    keep_top_n: int | None = None

    # Caches processed Hamiltonian to avoid re-sorting and re-slicing when the
    # same Hamiltonian is passed repeatedly (e.g. across optimizer evaluations).
    _cache: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self):
        if self.keep_fraction is not None and self.keep_top_n is not None:
            raise ValueError(
                "At most one of keep_fraction or keep_top_n may be provided."
            )

        if self.keep_fraction is not None and (
            self.keep_fraction <= 0 or self.keep_fraction > 1
        ):
            raise ValueError(
                f"keep_fraction must be in (0, 1], got {self.keep_fraction}"
            )

        if self.keep_top_n is not None and (
            self.keep_top_n <= 0 or not isinstance(self.keep_top_n, int)
        ):
            raise ValueError(
                f"keep_top_n must be a positive integer (>= 1), got {self.keep_top_n}"
            )

    @property
    def stateful(self) -> bool:
        # Despite having a _cache, this strategy is stateless because it only
        # uses the cache as memoization, not as state.
        return False

    def process_hamiltonian(
        self, hamiltonian: qml.operation.Operator
    ) -> qml.operation.Operator:
        """Exact Trotterize the Hamiltonian."""
        if self.keep_fraction is None and self.keep_top_n is None:
            return hamiltonian.simplify()

        if hamiltonian in self._cache:
            return self._cache[hamiltonian]

        if self.keep_fraction is not None and self.keep_fraction == 1.0:
            warn(
                "keep_fraction is 1.0 (no truncation); returning the full Hamiltonian.",
                UserWarning,
            )
            return hamiltonian.simplify()

        if self.keep_top_n is not None and self.keep_top_n >= len(hamiltonian):
            warn(
                "keep_top_n is greater than or equal to the number of terms; "
                "returning the full Hamiltonian.",
                UserWarning,
            )
            return hamiltonian.simplify()

        non_id_terms, constant = _clean_hamiltonian(hamiltonian)
        sorted_non_id_terms = _sort_hamiltonian_terms(non_id_terms, order="magnitude")

        if self.keep_top_n is not None:
            slice_idx = -self.keep_top_n

        if self.keep_fraction is not None:
            absolute_coeffs = np.abs(sorted_non_id_terms.terms()[0])
            target = absolute_coeffs.sum() * self.keep_fraction
            cumsum_from_end = np.cumsum(absolute_coeffs[::-1])
            n_keep = np.searchsorted(cumsum_from_end, target, side="left") + 1
            slice_idx = -min(n_keep, len(absolute_coeffs))

        result = type(hamiltonian)(
            *(*sorted_non_id_terms[slice_idx:], constant * qml.Identity())
        ).simplify()

        self._cache[hamiltonian] = result

        return result


@dataclass(frozen=True)
class QDrift(TrotterizationStrategy):
    """QDrift Trotterization strategy."""

    keep_fraction: float | None = None
    keep_top_n: int | None = None
    sample_budget: int | None = None
    sampling_strategy: Literal["uniform", "weighted"] = "uniform"
    seed: int | None = None
    n_hamiltonians_per_iteration: int = 1
    """Number of Hamiltonian samples per cost evaluation; losses are averaged over them."""

    # Caches the (keep_hamiltonian, to_sample_hamiltonian) split so we avoid
    # recomputing the deterministic part when the same Hamiltonian is passed
    # repeatedly; only the sampling step changes each call.
    _cache: dict = field(default_factory=dict, compare=False, hash=False)
    _rng: np.random.Generator = field(init=False, compare=False, hash=False)

    def __post_init__(self):
        if self.keep_fraction is None and self.sample_budget is None:
            warn(
                "Neither keep_fraction nor sample_budget is set; "
                "the Hamiltonian will be returned unchanged.",
                UserWarning,
            )
        elif self.sample_budget is None:
            warn(
                "sample_budget is not set; only the kept terms will be applied, "
                "equivalent to ExactTrotterization.",
                UserWarning,
            )

        if self.sampling_strategy not in ["uniform", "weighted"]:
            raise ValueError(
                f"Invalid sampling_strategy: {self.sampling_strategy}. Must be 'uniform' or 'weighted'."
            )

        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError(f"seed must be an integer, got {self.seed}")

        if self.n_hamiltonians_per_iteration < 1:
            raise ValueError(
                f"n_hamiltonians_per_iteration must be >= 1, got {self.n_hamiltonians_per_iteration}"
            )

        object.__setattr__(self, "_rng", np.random.default_rng(self.seed))

    @property
    def stateful(self) -> bool:
        return True

    def process_hamiltonian(
        self, hamiltonian: qml.operation.Operator
    ) -> qml.operation.Operator:
        """QDrift Trotterize the Hamiltonian."""
        if (
            self.keep_fraction is None
            and self.keep_top_n is None
            and self.sample_budget is None
        ):
            return hamiltonian.simplify()

        triggered_exact_trotterization = (
            True
            if self.keep_fraction is not None or self.keep_top_n is not None
            else False
        )

        if hamiltonian in self._cache:
            keep_hamiltonian, to_sample_hamiltonian = self._cache[hamiltonian]
        else:
            keep_hamiltonian = ExactTrotterization(
                keep_fraction=self.keep_fraction, keep_top_n=self.keep_top_n
            ).process_hamiltonian(hamiltonian)

            if triggered_exact_trotterization:
                to_sample_hamiltonian = (hamiltonian - keep_hamiltonian).simplify()
            else:
                to_sample_hamiltonian = hamiltonian.simplify()

            self._cache[hamiltonian] = (keep_hamiltonian, to_sample_hamiltonian)

            if triggered_exact_trotterization and qml.equal(
                keep_hamiltonian, hamiltonian
            ):
                warn(
                    "All terms were kept; there are no terms left to sample. "
                    "Returning the full Hamiltonian.",
                    UserWarning,
                )
                return hamiltonian

        if self.sample_budget is None:
            return keep_hamiltonian

        if triggered_exact_trotterization and qml.equal(keep_hamiltonian, hamiltonian):
            return hamiltonian

        # to_sample_hamiltonian already set above (from cache or computation)
        absolute_coeffs = np.abs(to_sample_hamiltonian.terms()[0])
        sampled_hamiltonian = self._rng.choice(
            np.asarray(to_sample_hamiltonian),
            size=self.sample_budget,
            replace=True,
            **(
                {"p": absolute_coeffs / absolute_coeffs.sum()}
                if self.sampling_strategy == "weighted"
                else {}
            ),
        )

        return (
            qml.ops.Sum(*sampled_hamiltonian.tolist()) + keep_hamiltonian
        ).simplify()


def convert_hamiltonian_to_pauli_string(
    hamiltonian: qml.operation.Operator, n_qubits: int
) -> str:
    """
    Convert a PennyLane Operator to a semicolon-separated string of Pauli operators.

    Each term in the Hamiltonian is represented as a string of Pauli letters ('I', 'X', 'Y', 'Z'),
    one per qubit. Multiple terms are separated by semicolons.

    Args:
        hamiltonian (qml.operation.Operator): The PennyLane Operator to convert.
        n_qubits (int): Number of qubits to represent in the string.

    Returns:
        str: The Hamiltonian as a semicolon-separated string of Pauli operators.

    Raises:
        ValueError: If an unknown Pauli operator is encountered or wire index is out of range.
    """
    pauli_letters = {"PauliX": "X", "PauliY": "Y", "PauliZ": "Z"}
    identity_row = np.full(n_qubits, "I", dtype="<U1")

    # Handle both single operators and sums of operators (like Hamiltonians)
    terms_to_process = (
        hamiltonian.operands if isinstance(hamiltonian, qml.ops.Sum) else [hamiltonian]
    )

    terms = []
    for term in terms_to_process:
        op = term
        while isinstance(op, qml.ops.SProd):
            op = op.base
        ops = op.operands if isinstance(op, qml.ops.Prod) else [op]

        paulis = identity_row.copy()
        for p in ops:
            if isinstance(p, qml.Identity):
                continue
            # Better fallback logic with validation
            if p.name in pauli_letters:
                pauli = pauli_letters[p.name]
            else:
                raise ValueError(
                    f"Unknown Pauli operator: {p.name}. "
                    "Expected 'PauliX', 'PauliY', or 'PauliZ'."
                )

            # Bounds checking for wire indices
            if not p.wires:
                raise ValueError(f"Pauli operator {p.name} has no wires")

            wire = int(p.wires[0])
            if wire < 0 or wire >= n_qubits:
                raise ValueError(
                    f"Wire index {wire} out of range for {n_qubits} qubits. "
                    f"Valid range: [0, {n_qubits - 1}]"
                )

            paulis[wire] = pauli
        terms.append("".join(paulis))

    return ";".join(terms)


def _is_sanitized(
    qubo_matrix: npt.NDArray[np.float64] | sps.spmatrix,
) -> npt.NDArray[np.float64] | sps.spmatrix:
    """
    Check if a QUBO matrix is either symmetric or upper triangular.

    This function validates that the input QUBO matrix is in a proper format
    for conversion to an Ising Hamiltonian. The matrix should be either
    symmetric (equal to its transpose) or upper triangular.

    Args:
        qubo_matrix (npt.NDArray[np.float64] | sps.spmatrix): The QUBO matrix to validate.
            Can be a dense NumPy array or a sparse SciPy matrix.

    Returns:
        bool: True if the matrix is symmetric or upper triangular, False otherwise.
    """
    is_sparse = sps.issparse(qubo_matrix)

    return (
        (
            ((qubo_matrix != qubo_matrix.T).nnz == 0)
            or ((qubo_matrix != sps.triu(qubo_matrix)).nnz == 0)
        )
        if is_sparse
        else (
            np.allclose(qubo_matrix, qubo_matrix.T)
            or np.allclose(qubo_matrix, np.triu(qubo_matrix))
        )
    )


def convert_qubo_matrix_to_pennylane_ising(
    qubo_matrix: npt.NDArray[np.float64] | sps.spmatrix,
) -> tuple[qml.operation.Operator, float]:
    """
    Convert a QUBO matrix to an Ising Hamiltonian in PennyLane format.

    The conversion follows the mapping from QUBO variables x_i ∈ {0,1} to
    Ising variables σ_i ∈ {-1,1} via the transformation x_i = (1 - σ_i)/2. This
    transforms a QUBO minimization problem into an equivalent Ising minimization
    problem.

    The function handles both dense NumPy arrays and sparse SciPy matrices efficiently.
    If the input matrix is neither symmetric nor upper triangular, it will be
    symmetrized automatically with a warning.

    Args:
        qubo_matrix (npt.NDArray[np.float64] | sps.spmatrix): The QUBO matrix Q where the
            objective is to minimize x^T Q x. Can be a dense NumPy array or a
            sparse SciPy matrix (any format). Should be square and either
            symmetric or upper triangular.

    Returns:
        tuple[qml.operation.Operator, float]: A tuple containing:
            - Ising Hamiltonian as a PennyLane operator (sum of Pauli Z terms)
            - Constant offset term to be added to energy calculations

    Raises:
        UserWarning: If the QUBO matrix is neither symmetric nor upper triangular.

    Example:
        >>> import numpy as np
        >>> qubo = np.array([[1, 2], [0, 3]])
        >>> hamiltonian, offset = convert_qubo_matrix_to_pennylane_ising(qubo)
        >>> print(f"Offset: {offset}")
    """

    if not _is_sanitized(qubo_matrix):
        warn(
            "The QUBO matrix is neither symmetric nor upper triangular."
            " Symmetrizing it for the Ising Hamiltonian creation."
        )
        qubo_matrix = (qubo_matrix + qubo_matrix.T) / 2

    is_sparse = sps.issparse(qubo_matrix)
    backend = sps if is_sparse else np

    symmetrized_qubo = (qubo_matrix + qubo_matrix.T) / 2

    # Gather non-zero indices in the upper triangle of the matrix
    triu_matrix = backend.triu(
        symmetrized_qubo,
        **(
            {"format": qubo_matrix.format if qubo_matrix.format != "coo" else "csc"}
            if is_sparse
            else {}
        ),
    )

    if is_sparse:
        coo_mat = triu_matrix.tocoo()
        rows, cols, values = coo_mat.row, coo_mat.col, coo_mat.data
    else:
        rows, cols = triu_matrix.nonzero()
        values = triu_matrix[rows, cols]

    n = qubo_matrix.shape[0]
    linear_terms = np.zeros(n)
    constant_term = 0.0
    ising_terms = []
    ising_weights = []

    for i, j, weight in zip(rows, cols, values):
        weight = float(weight)
        i, j = int(i), int(j)

        if i == j:
            # Diagonal elements
            linear_terms[i] -= weight / 2
            constant_term += weight / 2
        else:
            # Off-diagonal elements (i < j since we're using triu)
            ising_terms.append([i, j])
            ising_weights.append(weight / 4)

            # Update linear terms
            linear_terms[i] -= weight / 4
            linear_terms[j] -= weight / 4

            # Update constant term
            constant_term += weight / 4

    # Add the linear terms (Z operators)
    for i, curr_lin_term in filter(lambda x: x[1] != 0, enumerate(linear_terms)):
        ising_terms.append([i])
        ising_weights.append(float(curr_lin_term))

    # Construct the Ising Hamiltonian as a PennyLane operator
    pauli_string = qml.Identity(0) * 0
    for term, weight in zip(ising_terms, ising_weights):
        if len(term) == 1:
            # Single-qubit term (Z operator)
            curr_term = qml.Z(term[0]) * weight
        else:
            # Two-qubit term (ZZ interaction)
            curr_term = (
                reduce(lambda x, y: x @ y, map(lambda x: qml.Z(x), term)) * weight
            )

        pauli_string += curr_term

    return pauli_string.simplify(), constant_term
