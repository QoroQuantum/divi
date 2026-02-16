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


def _is_multi_term_sum(op: qml.operation.Operator) -> bool:
    """True if op is a multi-term Sum or Hamiltonian (has operands and len)."""
    return isinstance(op, (qml.Hamiltonian, qml.ops.Sum))


def _get_terms_iterable(op: qml.operation.Operator) -> list:
    """Return terms as a list for iteration. Works for Sum/Hamiltonian and single-term."""
    return op.operands if _is_multi_term_sum(op) else [op]


def _is_empty_hamiltonian(op: qml.operation.Operator) -> bool:
    """True if op is an empty Sum/Hamiltonian (only constant terms)."""
    return _is_multi_term_sum(op) and len(op) == 0


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

    terms = _get_terms_iterable(hamiltonian)

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


def _hamiltonian_term_count(hamiltonian: qml.operation.Operator) -> int:
    """Return the number of terms in a Hamiltonian.

    Works for qml.Hamiltonian, qml.ops.Sum (multi-term), and single-term operators
    such as SProd or bare Pauli operators, which do not implement __len__.
    """
    return len(hamiltonian) if _is_multi_term_sum(hamiltonian) else 1


def _sort_hamiltonian_terms(
    hamiltonian: qml.operation.Operator,
    order: Literal["absolute", "magnitude"] = "absolute",
) -> qml.operation.Operator:
    """Sort the terms of a Hamiltonian by their coefficient magnitude."""
    if not _is_multi_term_sum(hamiltonian):
        return hamiltonian
    coeffs, terms = hamiltonian.terms()
    sorted_coeffs, sorted_terms = zip(
        *sorted(
            zip(coeffs, terms), key=lambda x: x[0] if order == "absolute" else abs(x[0])
        )
    )
    weighted_terms = [cf * trm for cf, trm in zip(sorted_coeffs, sorted_terms)]
    # Avoid Sum construction for single term; preserves original operator type.
    if len(weighted_terms) == 1:
        return weighted_terms[0]
    return qml.sum(*weighted_terms).simplify()


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
    """Fraction of terms to keep by coefficient magnitude (largest first). Must be in (0, 1]. If None, keep all terms."""
    keep_top_n: int | None = None
    """Number of top terms to keep by coefficient magnitude. Must be >= 1. If None, keep all terms. Mutually exclusive with keep_fraction."""

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

        if self.keep_top_n is not None and self.keep_top_n >= _hamiltonian_term_count(
            hamiltonian
        ):
            warn(
                "keep_top_n is greater than or equal to the number of terms; "
                "returning the full Hamiltonian.",
                UserWarning,
            )
            return hamiltonian.simplify()

        non_id_terms, constant = _clean_hamiltonian(hamiltonian)
        if _is_empty_hamiltonian(non_id_terms):
            raise ValueError("Hamiltonian contains only constant terms.")
        sorted_non_id_terms = _sort_hamiltonian_terms(non_id_terms, order="magnitude")

        if not _is_multi_term_sum(sorted_non_id_terms):
            return (sorted_non_id_terms + constant * qml.Identity()).simplify()

        if self.keep_top_n is not None:
            slice_idx = -self.keep_top_n

        if self.keep_fraction is not None:
            absolute_coeffs = np.abs(sorted_non_id_terms.terms()[0])
            target = absolute_coeffs.sum() * self.keep_fraction
            cumsum_from_end = np.cumsum(absolute_coeffs[::-1])
            n_keep = np.searchsorted(cumsum_from_end, target, side="left") + 1
            slice_idx = -min(n_keep, len(absolute_coeffs))

        coeffs, terms = sorted_non_id_terms.terms()
        sliced_operands = [
            c * t for c, t in zip(list(coeffs)[slice_idx:], list(terms)[slice_idx:])
        ]
        if constant != 0:
            sliced_operands.append(constant * qml.Identity())
        result = qml.sum(*sliced_operands).simplify()

        self._cache[hamiltonian] = result

        return result


@dataclass(frozen=True)
class QDrift(TrotterizationStrategy):
    """QDrift Trotterization strategy."""

    keep_fraction: float | None = None
    """Fraction of terms to keep deterministically by coefficient magnitude (largest first). Must be in (0, 1]. If None, all terms go to the sampling pool. Mutually exclusive with keep_top_n."""
    keep_top_n: int | None = None
    """Number of top terms to keep deterministically by coefficient magnitude. Must be >= 1. If None, all terms go to the sampling pool. Mutually exclusive with keep_fraction."""
    sampling_budget: int | None = None
    """Number of terms to sample from the remaining Hamiltonian per cost evaluation. If None, only kept terms are applied (equivalent to ExactTrotterization)."""
    sampling_strategy: Literal["uniform", "weighted"] = "uniform"
    """How to sample terms: "uniform" (equal probability) or "weighted" (by coefficient magnitude)."""
    seed: int | None = None
    """Random seed for reproducible sampling. If None, sampling is non-deterministic."""
    n_hamiltonians_per_iteration: int = 10
    """Number of Hamiltonian samples per cost evaluation; losses are averaged over them."""

    # Caches the (keep_hamiltonian, to_sample_hamiltonian) split so we avoid
    # recomputing the deterministic part when the same Hamiltonian is passed
    # repeatedly; only the sampling step changes each call.
    _cache: dict = field(default_factory=dict, compare=False, hash=False)
    _rng: np.random.Generator = field(init=False, compare=False, hash=False)

    def __post_init__(self):
        if (
            self.keep_fraction is None
            and self.keep_top_n is None
            and self.sampling_budget is None
        ):
            warn(
                "Neither keep_fraction, keep_top_n, nor sampling_budget is set; "
                "the Hamiltonian will be returned unchanged.",
                UserWarning,
            )
        elif self.sampling_budget is None:
            warn(
                "sampling_budget is not set; only the kept terms will be applied, "
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
        """Apply the QDrift randomized channel to a Hamiltonian.

        Implements the QDrift protocol (Campbell 2019): for H = Σ c_i P_i,
        randomly sample L terms and rescale their coefficients so that
        E[H_sampled] = H.

        Rescaling rules (L = sampling_budget, λ = Σ|c_i|, N = #terms):
          - Weighted: term_i → (λ / (L · |c_i|)) · c_i · P_i
          - Uniform:  term_i → (N / L) · c_i · P_i
        """
        if (
            self.keep_fraction is None
            and self.keep_top_n is None
            and self.sampling_budget is None
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
            if triggered_exact_trotterization:
                keep_hamiltonian = ExactTrotterization(
                    keep_fraction=self.keep_fraction, keep_top_n=self.keep_top_n
                ).process_hamiltonian(hamiltonian)
                to_sample_hamiltonian = (hamiltonian - keep_hamiltonian).simplify()
            else:
                keep_hamiltonian = None
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

        if self.sampling_budget is None:
            if keep_hamiltonian is None:
                return hamiltonian.simplify()
            return keep_hamiltonian

        if triggered_exact_trotterization and qml.equal(keep_hamiltonian, hamiltonian):
            return hamiltonian

        # to_sample_hamiltonian already set above (from cache or computation)
        terms_list = list(_get_terms_iterable(to_sample_hamiltonian))
        if len(terms_list) == 0:
            warn(
                "No terms to sample; returning the kept Hamiltonian.",
                UserWarning,
            )
            if keep_hamiltonian is None:
                return qml.Hamiltonian([], [])
            return keep_hamiltonian

        if not _is_multi_term_sum(to_sample_hamiltonian):
            # Single term: no sampling needed, return as-is.
            sampled_terms = [to_sample_hamiltonian]
        else:
            absolute_coeffs = np.abs(to_sample_hamiltonian.terms()[0])
            coeff_sum = absolute_coeffs.sum()
            if coeff_sum == 0:
                warn(
                    "All term coefficients are zero; returning the kept Hamiltonian.",
                    UserWarning,
                )
                return keep_hamiltonian
            probs = (
                (absolute_coeffs / coeff_sum).tolist()
                if self.sampling_strategy == "weighted"
                else None
            )
            indices = self._rng.choice(
                len(terms_list),
                size=self.sampling_budget,
                replace=True,
                p=probs,
            )

            # --- QDrift coefficient rescaling ---
            # Each sampled term must be rescaled so that E[H_sampled] = H.
            if self.sampling_strategy == "weighted":
                # Weighted (p_i = |c_i|/λ): scale by λ / (L · |c_i|)
                sampled_terms = [
                    (coeff_sum / (self.sampling_budget * absolute_coeffs[i]))
                    * terms_list[i]
                    for i in indices
                ]
            else:
                # Uniform (p_i = 1/N): scale by N/L
                n_terms = len(terms_list)
                sampled_terms = [
                    (n_terms / self.sampling_budget) * terms_list[i] for i in indices
                ]

        sampled_sum = qml.ops.Sum(*sampled_terms)
        if keep_hamiltonian is not None:
            sampled_sum = sampled_sum + keep_hamiltonian
        return sampled_sum.simplify()


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
    terms_to_process = _get_terms_iterable(hamiltonian)

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
