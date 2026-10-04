# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Sample-based quantum diagonalisation post-processing.

Implements self-consistent configuration recovery (arXiv:2405.05068): symmetry
filtering, occupancy-guided bit-flip correction, batched determinant subspace
construction, spin-penalised projected diagonalisation, and reduced
density-matrix reconstruction.
"""

import bisect
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.sparse.linalg
from pyscf.fci import selected_ci

#: Determinant rows per pass in :func:`projected_matrices`. The pair arrays it
#: builds scale with this times the subspace size, so a fixed block keeps peak
#: memory independent of how large the subspace grows; only the returned
#: matrices scale with its square.
_PAIR_BLOCK_ROWS = 512

# Carryover retention threshold, relative to the winning batch's largest
# coefficient (arXiv:2512.14936 converges from 1e-3 down).
_DEFAULT_CARRYOVER_CUTOFF = 1e-5


def deinterleave_spin_bitstring(bitstring: str, n_orb: int) -> str:
    """Convert an interleaved divi bitstring to blocked ``alpha + beta`` order.

    Args:
        bitstring: Measured bitstring where character ``k`` is qubit ``k``, and
            qubit ``2p`` / ``2p + 1`` are the alpha / beta spin-orbitals of
            spatial orbital ``p``.
        n_orb: Number of spatial orbitals.

    Returns:
        A ``2 * n_orb`` string whose first half is alpha occupations by orbital
        and second half is beta occupations by orbital.

    Raises:
        ValueError: If ``bitstring`` is not ``2 * n_orb`` characters wide.
    """
    if len(bitstring) != 2 * n_orb:
        raise ValueError(
            f"bitstring width {len(bitstring)} does not match 2 * n_orb "
            f"({2 * n_orb})."
        )
    alpha = bitstring[0::2]
    beta = bitstring[1::2]
    return alpha + beta


def interleave_spin_bitstring(sqd_bitstring: str, n_orb: int) -> str:
    """Convert a blocked ``alpha + beta`` bitstring back to divi's interleaving.

    Raises:
        ValueError: If ``sqd_bitstring`` is not ``2 * n_orb`` characters wide.
    """
    if len(sqd_bitstring) != 2 * n_orb:
        raise ValueError(
            f"sqd_bitstring width {len(sqd_bitstring)} does not match "
            f"2 * n_orb ({2 * n_orb})."
        )
    alpha = sqd_bitstring[:n_orb]
    beta = sqd_bitstring[n_orb:]
    return "".join(a + b for a, b in zip(alpha, beta))


def probs_to_sqd_bitstrings(probs: dict[str, float], n_orb: int) -> dict[str, float]:
    """Convert a measured distribution into the SQD bitstring convention.

    Raises:
        ValueError: If any key in ``probs`` is not ``2 * n_orb`` characters
            wide, propagated from :func:`deinterleave_spin_bitstring`.
    """
    return {
        deinterleave_spin_bitstring(bitstring, n_orb): prob
        for bitstring, prob in probs.items()
    }


def spin_orbital_integrals(
    one_body: np.ndarray,
    two_body: np.ndarray,
    n_orb: int,
    one_body_beta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert spatial one- and two-body integrals to spin-orbital integrals.

    Spin-orbitals are blocked: index ``p`` is alpha for ``p < n_orb`` and beta
    above. ``one_body_beta`` gives the beta channel a distinct one-body
    potential, as a spin-polarised mean-field embedding produces.
    """
    if one_body_beta is None:
        one_body_beta = one_body
    n_spin_orb = 2 * n_orb
    h_spin = np.zeros((n_spin_orb, n_spin_orb))
    h_spin[:n_orb, :n_orb] = one_body
    h_spin[n_orb:, n_orb:] = one_body_beta

    # The Coulomb term survives where p, r share a spin and q, s do; the
    # exchange term where p, s do and q, r do.
    coulomb = np.einsum("prqs->pqrs", two_body)
    exchange = np.einsum("psqr->pqrs", two_body)

    alpha, beta = slice(None, n_orb), slice(n_orb, None)
    g_spin = np.zeros((n_spin_orb, n_spin_orb, n_spin_orb, n_spin_orb))
    g_spin[alpha, alpha, alpha, alpha] = coulomb - exchange
    g_spin[beta, beta, beta, beta] = coulomb - exchange
    g_spin[alpha, beta, alpha, beta] = coulomb
    g_spin[beta, alpha, beta, alpha] = coulomb
    g_spin[alpha, beta, beta, alpha] = -exchange
    g_spin[beta, alpha, alpha, beta] = -exchange

    return h_spin, g_spin


def _annihilation_sign(occ, p):
    """Return the sign and remaining occupation from annihilating spin-orbital p,
    which must be occupied in ``occ``."""
    occ_tup = tuple(occ)
    idx = occ_tup.index(p)
    sign = (-1) ** idx
    new_occ = occ_tup[:idx] + occ_tup[idx + 1 :]
    return sign, new_occ


def _creation_sign(occ, p):
    """Return the sign and updated occupation from creating spin-orbital p,
    which must be empty in ``occ``."""
    occ_tup = tuple(occ)
    idx = bisect.bisect_left(occ_tup, p)
    sign = (-1) ** idx
    new_occ = occ_tup[:idx] + (p,) + occ_tup[idx:]
    return sign, new_occ


def slater_condon(det_i, det_j, h_spin, g_spin) -> float:
    """Compute the Hamiltonian matrix element between two Slater determinants."""
    set_i = set(det_i)
    set_j = set(det_j)

    diff_i = sorted(list(set_i - set_j))
    diff_j = sorted(list(set_j - set_i))

    if len(diff_i) > 2:
        return 0.0

    if len(diff_i) == 0:  # Identical
        val = 0.0
        for p in det_i:
            val += h_spin[p, p]
            for q in det_i:
                if p < q:
                    val += g_spin[p, q, p, q]
        return val

    if len(diff_i) == 1:  # Differ by 1
        p = diff_i[0]
        q = diff_j[0]
        # We compute <det_i | H | det_j> where det_i = det_j - {q} + {p}
        # Annihilate q in det_j, create p in det_j
        sign_ann, occ_mid = _annihilation_sign(det_j, q)
        sign_cre, _ = _creation_sign(occ_mid, p)
        sign = sign_ann * sign_cre

        val = h_spin[p, q]
        for r in det_j:
            if r != q:
                val += g_spin[p, r, q, r]
        return sign * val

    # Differ by 2
    p, r = diff_i
    q, s = diff_j
    # We compute <det_i | H | det_j> where det_i = det_j - {q, s} + {p, r}
    # Annihilate q in det_j, then s in remaining, then create r, then p
    sign_q, occ_1 = _annihilation_sign(det_j, q)
    sign_s, occ_2 = _annihilation_sign(occ_1, s)
    sign_r, occ_3 = _creation_sign(occ_2, r)
    sign_p, _ = _creation_sign(occ_3, p)
    sign = sign_q * sign_s * sign_r * sign_p

    return sign * g_spin[p, r, q, s]


def spatial_to_spin_occupations(
    alpha_occ: tuple[int, ...], beta_occ: tuple[int, ...], n_orb: int
) -> tuple[int, ...]:
    """Combine alpha/beta spatial occupations into a sorted, blocked spin-orbital
    tuple: alpha keeps its spatial index, beta is offset by ``n_orb``."""
    return tuple(sorted(list(alpha_occ) + [p + n_orb for p in beta_occ]))


def spin_to_spatial_occupations(
    spin_occ, n_orb: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split a spin-orbital occupation tuple back into (alpha_occ, beta_occ)."""
    alpha_occ = tuple(sorted([p for p in spin_occ if p < n_orb]))
    beta_occ = tuple(sorted([p - n_orb for p in spin_occ if p >= n_orb]))
    return alpha_occ, beta_occ


def bitstring_to_spatial_det(
    sqd_bitstring: str, n_orb: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split a blocked ``alpha + beta`` bitstring into occupied-orbital tuples."""
    alpha_part = sqd_bitstring[:n_orb]
    beta_part = sqd_bitstring[n_orb:]
    alpha_occ = tuple(i for i, c in enumerate(alpha_part) if c == "1")
    beta_occ = tuple(i for i, c in enumerate(beta_part) if c == "1")
    return alpha_occ, beta_occ


def _apply_s_plus(det):
    """Apply the S+ ladder operator to a spatial (alpha_occ, beta_occ) determinant."""
    alpha_occ, beta_occ = det
    results = []
    for idx_beta, p in enumerate(beta_occ):
        if p in alpha_occ:
            continue
        idx_alpha = bisect.bisect_left(alpha_occ, p)
        sign = (-1) ** (idx_beta + idx_alpha)
        new_alpha = alpha_occ[:idx_alpha] + (p,) + alpha_occ[idx_alpha:]
        new_beta = beta_occ[:idx_beta] + beta_occ[idx_beta + 1 :]
        results.append((sign, (new_alpha, new_beta)))
    return results


def _apply_s_minus(det):
    """Apply the S- ladder operator to a spatial (alpha_occ, beta_occ) determinant."""
    alpha_occ, beta_occ = det
    results = []
    for idx_alpha, p in enumerate(alpha_occ):
        if p in beta_occ:
            continue
        idx_beta = bisect.bisect_left(beta_occ, p)
        sign = (-1) ** (idx_alpha + idx_beta)
        new_beta = beta_occ[:idx_beta] + (p,) + beta_occ[idx_beta:]
        new_alpha = alpha_occ[:idx_alpha] + alpha_occ[idx_alpha + 1 :]
        results.append((sign, (new_alpha, new_beta)))
    return results


def s2_matrix_element(det_i, det_j) -> float:
    """Compute <det_i | S^2 | det_j> via S^2 = S_z(S_z + 1) + S_- S_+.

    ``det_i`` and ``det_j`` are ``(alpha_occ, beta_occ)`` spatial-orbital pairs.
    """
    alpha_j, beta_j = det_j
    sz_j = 0.5 * (len(alpha_j) - len(beta_j))

    diag = 0.0
    if det_i == det_j:
        diag = sz_j * (sz_j + 1.0)

    s_plus_results = _apply_s_plus(det_j)
    coeff_ij = 0.0
    for sign_p, det_p in s_plus_results:
        s_minus_results = _apply_s_minus(det_p)
        for sign_m, det_m in s_minus_results:
            if det_m == det_i:
                coeff_ij += sign_p * sign_m

    return diag + coeff_ij


def projected_matrices(
    dets: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
    dets_spin: Sequence[tuple[int, ...]],
    h_spin: np.ndarray,
    g_spin: np.ndarray,
    n_orb: int,
) -> tuple[scipy.sparse.csr_array, scipy.sparse.csr_array]:
    """Project the Hamiltonian and ``S^2`` onto the span of ``dets``.

    Returns what filling every entry with :func:`slater_condon` and
    :func:`s2_matrix_element` would -- those remain the reference definitions --
    computed by array operations instead of a double loop. Excitation ranks come
    from one matrix product, then each rank's elements are gathered in one pass.

    The saving is that Slater-Condon vanishes past a double excitation, and the
    fraction of pairs that survive falls as the subspace grows.

    ``S^2``'s off-diagonal is handled differently. It is non-zero only between
    determinants related by exchanging spins across two spatial orbitals, a
    small fraction of pairs, so those are located by array operations and then
    evaluated by the scalar routine -- keeping its ladder-operator signs as the
    single source of truth for a negligible cost.

    Rows are processed in blocks so the pair arrays stay bounded rather than
    scaling with the square of the subspace, and both matrices are returned
    sparse, holding only the connected pairs.

    Args:
        dets: ``(alpha_occ, beta_occ)`` spatial occupations per determinant.
        dets_spin: The same determinants as sorted spin-orbital tuples, all
            holding the same electron count -- the excitation rank is derived
            from that assumption.
        h_spin: One-body integrals over spin-orbitals.
        g_spin: Two-body integrals over spin-orbitals.
        n_orb: Spatial orbitals in the fragment.

    Returns:
        ``(h_proj, s2_proj)``, both ``(len(dets), len(dets))``.
    """
    m = len(dets_spin)
    h_entries: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    s2_entries: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def assemble(entries):
        if not entries:
            return scipy.sparse.csr_array((m, m))
        rows, cols, values = (np.concatenate(part) for part in zip(*entries))
        return scipy.sparse.csr_array((values, (rows, cols)), shape=(m, m))

    if m == 0:
        return assemble(h_entries), assemble(s2_entries)

    n_spin = 2 * n_orb
    occupation = np.zeros((m, n_spin))
    for row, spin_occ in enumerate(dets_spin):
        occupation[row, list(spin_occ)] = 1.0

    n_electrons = int(round(float(occupation[0].sum())))
    # Occupied orbitals strictly below an index: the parity that signs an
    # annihilation or creation there, matching _annihilation_sign's use of the
    # position within the sorted occupation tuple.
    below = np.cumsum(occupation, axis=1) - occupation
    index = np.arange(n_spin)

    # --- Diagonal (identical determinants) ---
    # Strictly upper-triangular so the pair sum runs over p < q, matching
    # slater_condon without assuming g[p, q, p, q] == g[q, p, q, p].
    coulomb = g_spin[index[:, None], index[None, :], index[:, None], index[None, :]]
    coulomb = np.triu(coulomb, k=1)
    diagonal = occupation @ np.diag(h_spin) + np.einsum(
        "ip,pq,iq->i", occupation, coulomb, occupation
    )
    alpha_occupation = occupation[:, :n_orb]
    beta_occupation = occupation[:, n_orb:]
    spin_z = 0.5 * (alpha_occupation.sum(axis=1) - beta_occupation.sum(axis=1))
    s2_diagonal = spin_z * (spin_z + 1.0) + (
        beta_occupation * (1.0 - alpha_occupation)
    ).sum(axis=1)

    # exchange[p, q, r] == g_spin[p, r, q, r], the sum a single excitation takes
    # over the orbitals occupied in the right-hand determinant.
    exchange = g_spin[
        index[:, None, None],
        index[None, None, :],
        index[None, :, None],
        index[None, None, :],
    ]

    for start in range(0, m, _PAIR_BLOCK_ROWS):
        stop = min(start + _PAIR_BLOCK_ROWS, m)
        # Every determinant holds n_electrons, so the count of orbitals in i
        # absent from j is the excitation rank connecting them.
        rank = (n_electrons - np.rint(occupation[start:stop] @ occupation.T)).astype(
            np.int16
        )

        local, cols = np.nonzero(rank == 0)
        rows = local + start
        h_entries.append((rows, cols, diagonal[rows]))
        s2_entries.append((rows, cols, s2_diagonal[rows]))

        # --- Single excitations: q in j replaced by p in i ---
        local, cols = np.nonzero(rank == 1)
        if local.size:
            rows = local + start
            created = (occupation[rows] * (1.0 - occupation[cols])).argmax(axis=1)
            annihilated = (occupation[cols] * (1.0 - occupation[rows])).argmax(axis=1)
            # Annihilating q then creating p, the intermediate occupation losing
            # one orbital below p exactly when q < p.
            exponent = (
                below[cols, annihilated]
                + below[cols, created]
                - (annihilated < created)
            )
            sign = 1.0 - 2.0 * (exponent.astype(np.int64) % 2)
            # The r == q term the reference excludes contributes g[p, q, q, q],
            # which antisymmetry makes identically zero, so no exclusion is
            # needed here.
            summed = np.einsum(
                "kr,kr->k", exchange[created, annihilated], occupation[cols]
            )
            h_entries.append(
                (rows, cols, sign * (h_spin[created, annihilated] + summed))
            )

        # --- Double excitations: q, s in j replaced by p, r in i ---
        local, cols = np.nonzero(rank == 2)
        if local.size:
            rows = local + start
            # np.nonzero walks row-major and each row holds exactly two entries,
            # so every pair contributes its differing orbitals in ascending order.
            _, created_pair = np.nonzero(occupation[rows] * (1.0 - occupation[cols]))
            _, annihilated_pair = np.nonzero(
                occupation[cols] * (1.0 - occupation[rows])
            )
            lower_created, upper_created = created_pair[0::2], created_pair[1::2]
            lower_annihilated = annihilated_pair[0::2]
            upper_annihilated = annihilated_pair[1::2]
            # Annihilate both, then create both, each step's parity read off the
            # original occupation corrected for the orbitals already moved. The
            # -1 is the q < s comparison, always true since the pair is sorted.
            exponent = (
                below[cols, lower_annihilated]
                + below[cols, upper_annihilated]
                - 1
                + below[cols, upper_created]
                - (lower_annihilated < upper_created)
                - (upper_annihilated < upper_created)
                + below[cols, lower_created]
                - (lower_annihilated < lower_created)
                - (upper_annihilated < lower_created)
            )
            sign = 1.0 - 2.0 * (exponent.astype(np.int64) % 2)
            h_entries.append(
                (
                    rows,
                    cols,
                    sign
                    * g_spin[
                        lower_created,
                        upper_created,
                        lower_annihilated,
                        upper_annihilated,
                    ],
                )
            )

            # S^2 connects only spin-exchange pairs: i gains alpha at one spatial
            # orbital and beta at another, losing exactly the opposite pair. The
            # predicate is a superset -- a pair that satisfies it but carries no
            # S^2 weight simply gets assigned zero.
            spin_exchange = (
                (lower_created < n_orb)
                & (upper_created >= n_orb)
                & (lower_annihilated < n_orb)
                & (upper_annihilated >= n_orb)
                & (lower_created == upper_annihilated - n_orb)
                & (lower_annihilated == upper_created - n_orb)
            )
            exchange_rows, exchange_cols = rows[spin_exchange], cols[spin_exchange]
            s2_entries.append(
                (
                    exchange_rows,
                    exchange_cols,
                    np.array(
                        [
                            s2_matrix_element(dets[row], dets[col])
                            for row, col in zip(exchange_rows, exchange_cols)
                        ],
                        dtype=float,
                    ),
                )
            )

    return assemble(h_entries), assemble(s2_entries)


#: Subspace size from which the ground root is found iteratively rather than
#: densely.
_ITERATIVE_SUBSPACE_MIN = 512

#: Ground-root convergence for the iterative solve, well inside the energies
#: LASSQD resolves.
_ITERATIVE_TOLERANCE = 1e-10


def ground_root(
    h_proj: scipy.sparse.csr_array,
    deviation: scipy.sparse.csr_array,
    lambda_penalty: float,
) -> tuple[float, np.ndarray]:
    """Lowest eigenpair of the spin-penalised projected Hamiltonian.

    ``H + lambda * deviation ** 2`` is what gets diagonalised, and only its
    lowest root is ever read. Large subspaces therefore take a Lanczos solve
    with the penalty applied as an operator, which never forms the dense
    ``deviation @ deviation`` product; small ones fall back to the dense form.

    A Lanczos solve can converge on an interior root instead of the lowest one,
    which the dense solve cannot do. Two roots are requested rather than one so
    a missed root has to be missed twice, and the result is checked against the
    smallest diagonal entry -- a variational upper bound on the true lowest
    eigenvalue -- with anything above it, or any solve that does not converge,
    falling back to the dense form.

    Args:
        h_proj: The projected Hamiltonian.
        deviation: ``S^2`` projected onto the subspace, less the target
            eigenvalue on the diagonal.
        lambda_penalty: Weight of the spin-contamination penalty.

    Returns:
        ``(eigenvalue, eigenvector)`` for the lowest root.
    """
    # pyrefly: ignore[unsupported-operation]  # scipy leaves _shape unannotated
    dimension = h_proj.shape[0]
    if dimension >= _ITERATIVE_SUBSPACE_MIN:
        spin_deviation = scipy.sparse.linalg.aslinearoperator(deviation)
        penalized = scipy.sparse.linalg.aslinearoperator(h_proj) + lambda_penalty * (
            spin_deviation @ spin_deviation
        )
        diagonal = (
            h_proj.diagonal()
            + lambda_penalty
            * np.asarray(deviation.multiply(deviation.T).sum(axis=1)).ravel()
        )
        # Started deterministically on the lowest-diagonal determinant, since a
        # random start would let the carryover ranking move between runs.
        start = np.zeros(dimension)
        start[int(np.argmin(diagonal))] = 1.0
        try:
            values, vectors = scipy.sparse.linalg.eigsh(
                penalized, k=2, which="SA", v0=start, tol=_ITERATIVE_TOLERANCE
            )
            lowest = int(np.argmin(values))
            if values[lowest] <= diagonal.min():
                return float(values[lowest]), np.asarray(vectors)[:, lowest]
        except scipy.sparse.linalg.ArpackError:
            pass

    values, vectors = scipy.linalg.eigh(
        (h_proj + lambda_penalty * (deviation @ deviation)).toarray()
    )
    return float(values[0]), np.asarray(vectors)[:, 0]


#: Places the carryover ranking rounds to. Threaded BLAS and set iteration order
#: perturb eigenvector components at the last bit, enough to swap near-equal
#: weights and, once a cap binds, change which determinants survive.
_CARRYOVER_WEIGHT_PLACES = 12


def _heaviest_strings(weights: dict[str, float], limit: int | None) -> tuple[str, ...]:
    """The ``limit`` heaviest strings, near-ties broken on the string so float
    noise cannot decide what is kept."""
    ordered = sorted(
        weights,
        key=lambda string: (
            -round(weights[string], _CARRYOVER_WEIGHT_PLACES),
            string,
        ),
    )
    return tuple(ordered[:limit])


def ci_string_to_int(half: str) -> int:
    """Integer form of one spin sector's occupation string, character ``p`` to bit
    ``p`` -- how PySCF's selected CI addresses determinants."""
    return int(half[::-1], 2)


def _aufbau_string(n_orb: int, n_electrons: int) -> str:
    """One spin sector of the reference determinant, as the ansatz prepares it at
    zero parameters: spatial orbitals ``0 .. n_electrons - 1`` filled."""
    return "1" * n_electrons + "0" * (n_orb - n_electrons)


def _bit_matrix(strings: Sequence[str], width: int) -> np.ndarray:
    """``(len(strings), width)`` uint8 0/1 matrix of equal-width bit strings.

    Raises:
        ValueError: If any string is not ``width`` characters wide.
    """
    offender = next((s for s in strings if len(s) != width), None)
    if offender is not None:
        raise ValueError(
            f"expected {width}-character strings, got one of width {len(offender)}"
        )
    packed = np.frombuffer("".join(strings).encode("ascii"), dtype=np.uint8)
    if np.any((packed < ord("0")) | (packed > ord("1"))):
        raise ValueError("strings contains a non-binary bitstring")
    return (packed - np.uint8(ord("0"))).reshape(len(strings), width)


def _occupations_from_bit_matrix(bits: np.ndarray) -> list[tuple[int, ...]]:
    """Occupied-orbital tuple for each row of a binary indicator matrix."""
    rows, cols = np.nonzero(bits)
    bounds = np.searchsorted(rows, np.arange(len(bits) + 1))
    return [tuple(cols[bounds[i] : bounds[i + 1]].tolist()) for i in range(len(bits))]


@dataclass(frozen=True)
class SQDConfig:
    """Sampling and diagonalisation budget for each fragment's SQD solve.

    Args:
        n_batches: Subspaces diagonalised per recovery iteration; the lowest
            energy wins.
        batch_size: Configurations sampled per batch, so the subspace holds up
            to ``batch_size ** 2`` determinants. The accuracy knob; a
            one-determinant subspace is the mean field.
        n_recovery_iterations: Configuration-recovery passes per fragment solve.
            Each pass reweights the next one's sampling from the orbital
            occupancies the previous pass recovered, so these are
            self-consistent passes over the sampled distribution, not optimizer
            steps.
        lambda_penalty: Weight of the ``S^2`` spin-contamination penalty added
            to the projected Hamiltonian before diagonalisation.
        carryover_cutoff: Carryover SQD's retention threshold
            (arXiv:2512.14936), on by default. Each recovery iteration retains
            the determinants whose coefficient exceeds this fraction of the
            largest coefficient in the winning batch, and extends later
            iterations' subspaces with them. The strings retained from a
            fragment's best result also seed the first recovery iteration of
            its next macro-cycle, mapped into that round's orbital basis by
            ``carryover_mapping``. ``None`` reverts to conventional SQD.
        carryover_mapping: How strings carried between macro-cycles follow the
            orbitals. ``'assignment'`` (default) pairs old and new orbitals
            one-to-one by maximum total overlap, so every string keeps its
            electron count; ``'argmax'`` gives each new orbital the occupation
            of the old orbital it overlaps most, as the reference implementation
            does, and drops strings whose electron count that changes.
        max_carryover: Caps the alpha and beta strings carryover retains, *per
            spin sector*. Carried strings join each batch's own sampled halves
            rather than replacing them, so a cap of ``k`` bounds a batch's
            subspace at ``(k + batch_size) ** 2`` determinants. ``None`` leaves
            it uncapped, and since the cutoff is relative it prunes little: the
            retained set then grows every recovery iteration and the subspace
            with it, quadratically. ``max_dim`` bounds the sector outright rather
            than only the carried part.
        max_dim: Caps each spin sector, as one integer or an ``(alpha, beta)``
            pair (any two-item sequence, stored as a tuple), so the subspace
            never exceeds their product. When it binds,
            strings are kept in priority order: reference, then carried, then
            sampled by descending sample count.
        include_reference: Keep the aufbau reference determinant in every batch,
            bounding the fragment's energy by its reference.
        symmetrize_spin: Pool the alpha and beta halves together for a
            spin-exchange invariant subspace. Inactive unless
            ``n_alpha == n_beta``.
        recovery_energy_tol: Ends a fragment's recovery once the winning energy
            moves less than this between iterations and the occupancies have also
            settled. ``0.0`` (the default) spends every iteration, since a
            settled iteration does not mean carryover had nothing left to add.
            Not ``LASSQD``'s ``energy_tol``, which ends the macro-cycle.
        recovery_occupancies_tol: The occupancy half of that test, on the largest
            change in any orbital's average occupancy.

    Raises:
        ValueError: If ``n_batches``, ``batch_size`` or
            ``n_recovery_iterations`` is below 1; if ``lambda_penalty`` is
            negative; if ``carryover_cutoff`` is outside ``(0, 1)``; if
            ``carryover_mapping`` is neither ``'assignment'`` nor ``'argmax'``; if
            ``max_carryover`` is given without a cutoff or is below 1; if
            ``max_dim`` is not a positive integer or a pair of them; or if
            ``recovery_energy_tol`` or ``recovery_occupancies_tol`` is negative.
    """

    n_batches: int = 15
    batch_size: int = 170
    n_recovery_iterations: int = 6
    lambda_penalty: float = 0.2
    carryover_cutoff: float | None = _DEFAULT_CARRYOVER_CUTOFF
    carryover_mapping: Literal["assignment", "argmax"] = "assignment"
    max_carryover: int | None = None
    max_dim: int | tuple[int, int] | None = None
    include_reference: bool = True
    symmetrize_spin: bool = False
    recovery_energy_tol: float = 0.0
    recovery_occupancies_tol: float = 0.0

    def __post_init__(self):
        if self.n_batches < 1:
            raise ValueError(f"n_batches must be at least 1; got {self.n_batches}.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1; got {self.batch_size}.")
        if self.n_recovery_iterations < 1:
            raise ValueError(
                "n_recovery_iterations must be at least 1; got "
                f"{self.n_recovery_iterations}."
            )
        if self.lambda_penalty < 0:
            raise ValueError(
                f"lambda_penalty must be non-negative; got {self.lambda_penalty}."
            )
        cutoff = self.carryover_cutoff
        if cutoff is not None:
            if not cutoff > 0:
                raise ValueError(f"carryover_cutoff must be positive; got {cutoff}.")
            if not cutoff < 1:
                raise ValueError(
                    "carryover_cutoff must be below 1: it is a fraction of the "
                    "largest coefficient, which none exceeds, so "
                    f"{cutoff} would retain nothing. Use None to turn carryover "
                    "off."
                )
        if self.carryover_mapping not in ("assignment", "argmax"):
            raise ValueError(
                "carryover_mapping must be 'assignment' or 'argmax'; got "
                f"{self.carryover_mapping!r}."
            )
        if self.max_carryover is not None:
            if cutoff is None:
                raise ValueError(
                    "max_carryover caps what carryover retains, so it needs "
                    "carryover_cutoff to be set."
                )
            if self.max_carryover < 1:
                raise ValueError(
                    f"max_carryover must be at least 1; got {self.max_carryover}."
                )
        if self.max_dim is not None:
            if isinstance(self.max_dim, Sequence) and not isinstance(self.max_dim, str):
                object.__setattr__(
                    self, "max_dim", tuple(int(dim) for dim in self.max_dim)
                )
                if len(self.max_dim) != 2:
                    raise ValueError(
                        "max_dim takes one integer or an (alpha, beta) pair; got "
                        f"{len(self.max_dim)} entries."
                    )
                dims = self.max_dim
            else:
                dims = (self.max_dim,)
            for dim in dims:
                if dim < 1:
                    raise ValueError(f"max_dim entries must be at least 1; got {dim}.")
        for name, value in (
            ("recovery_energy_tol", self.recovery_energy_tol),
            ("recovery_occupancies_tol", self.recovery_occupancies_tol),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative; got {value}.")


def carryover_weights(
    strings_alpha: Sequence[str],
    strings_beta: Sequence[str],
    amplitudes: np.ndarray,
    cutoff: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Weigh the alpha and beta strings worth carrying to the next iteration.

    A string is eligible when some determinant containing it clears the cutoff;
    eligible strings are ranked by their marginal weight over the whole subspace.

    The threshold is relative to the largest coefficient because the eigenvector
    is normalised: its typical component falls as ``1 / sqrt(m)``, so a fixed
    threshold would prune almost nothing at large subspace sizes.

    Args:
        strings_alpha: Alpha sector strings indexing ``amplitudes``' rows.
        strings_beta: Beta sector strings indexing its columns.
        amplitudes: Ground-state coefficients, one row per alpha string.
        cutoff: Retain strings appearing in a determinant whose
            ``abs(coefficient)`` exceeds this fraction of the largest.

    Returns:
        ``(alpha_weights, beta_weights)``, each mapping string to weight.
    """
    coefficients = np.asarray(amplitudes, dtype=float)
    if coefficients.size == 0:
        return {}, {}
    magnitude = np.abs(coefficients)
    largest = float(magnitude.max())
    if largest == 0.0:
        return {}, {}

    eligible = magnitude > cutoff * largest
    probability = coefficients**2
    alpha_marginal = probability.sum(axis=1)
    beta_marginal = probability.sum(axis=0)

    alpha_weights = {
        strings_alpha[int(row)]: float(alpha_marginal[row])
        for row in np.flatnonzero(eligible.any(axis=1))
    }
    beta_weights = {
        strings_beta[int(col)]: float(beta_marginal[col])
        for col in np.flatnonzero(eligible.any(axis=0))
    }
    return alpha_weights, beta_weights


def map_carried_strings(
    strings: Sequence[str],
    overlap: np.ndarray,
    mapping: Literal["assignment", "argmax"],
) -> tuple[str, ...]:
    """Carry occupation strings into a new orbital basis, order kept, duplicates
    dropped.

    A determinant in one basis is a superposition in another, so this keeps the
    nearest determinant: each new orbital takes the occupation of one old
    orbital. ``'assignment'`` pairs old and new orbitals one-to-one by maximum
    total ``|overlap|``, so every string keeps its electron count.
    ``'argmax'`` gives each new orbital the old orbital it overlaps most, as
    the reference implementation does; two new orbitals can then share an old
    one and change a string's electron count, and the solver drops such strings.

    Args:
        strings: Occupation strings in the old basis, character ``p`` for
            orbital ``p``.
        overlap: ``overlap[p, q]``, new orbital ``p`` against old orbital ``q``.
        mapping: ``'assignment'`` or ``'argmax'``.
    """
    magnitude = np.abs(overlap)
    if mapping == "assignment":
        _, source = scipy.optimize.linear_sum_assignment(magnitude, maximize=True)
    else:
        source = np.argmax(magnitude, axis=1)
    mapped = ("".join(string[q] for q in source) for string in strings)
    return tuple(dict.fromkeys(mapped))


def filter_symmetry(bitstrings, n_orb: int, n_alpha: int, n_beta: int) -> list[str]:
    """Keep only blocked bitstrings with the target alpha and beta counts."""
    bitstrings = list(bitstrings)
    if not bitstrings:
        return []
    bits = _bit_matrix(bitstrings, 2 * n_orb)
    keep = (bits[:, :n_orb].sum(axis=1, dtype=np.int64) == n_alpha) & (
        bits[:, n_orb:].sum(axis=1, dtype=np.int64) == n_beta
    )
    return [half for half, keeping in zip(bitstrings, keep) if keeping]


def _modified_relu(distance: float, threshold: float, delta: float) -> float:
    """Flip-weight profile from arXiv:2405.05068."""
    if distance <= threshold:
        return delta
    return (distance - threshold) + delta


def _correct_spin_part(
    part: list[int],
    target: int,
    average_occupancy: np.ndarray,
    n_orb: int,
    rng: np.random.Generator,
) -> list[int]:
    """Flip bits in one spin sector until it holds ``target`` electrons.

    Flip candidates are weighted by how far the observed bit is from the
    running average occupancy, so confidently-assigned orbitals are preserved.
    """
    current = sum(part)
    if current == target:
        return part

    delta = 0.01
    threshold = target / n_orb

    # Too many electrons means emptying occupied bits, too few means filling
    # empty ones; the weighting is the same distance either way.
    from_value = 1 if current > target else 0
    indices = [i for i, val in enumerate(part) if val == from_value]
    weights = [
        _modified_relu(abs(from_value - average_occupancy[i]), threshold, delta)
        for i in indices
    ]
    n_flips = abs(current - target)
    new_value = 1 - from_value

    weight_sum = sum(weights)
    probabilities = [w / weight_sum for w in weights]

    chosen = rng.choice(indices, size=n_flips, replace=False, p=probabilities)
    for index in chosen:
        part[index] = new_value
    return part


def bit_flip_correction(
    bitstring: str,
    n_orb: int,
    n_alpha: int,
    n_beta: int,
    occupancy: np.ndarray,
    rng: np.random.Generator,
) -> str:
    """Restore particle-number symmetry using running orbital occupancies.

    Args:
        bitstring: Blocked ``alpha + beta`` bitstring.
        n_orb: Number of spatial orbitals.
        n_alpha: Target alpha electron count.
        n_beta: Target beta electron count.
        occupancy: ``(2, n_orb)`` average occupancy per spin and orbital.
        rng: Generator used to draw which bits to flip.

    Returns:
        A blocked bitstring with exactly ``n_alpha`` / ``n_beta`` electrons.

    Raises:
        ValueError: If ``n_alpha`` or ``n_beta`` is negative or exceeds
            ``n_orb``, since no bitstring of width ``n_orb`` can hold that
            many electrons in one spin sector.
    """
    if not 0 <= n_alpha <= n_orb:
        raise ValueError(
            f"n_alpha must be between 0 and n_orb ({n_orb}), got {n_alpha}."
        )
    if not 0 <= n_beta <= n_orb:
        raise ValueError(f"n_beta must be between 0 and n_orb ({n_orb}), got {n_beta}.")

    alpha_part = [int(c) for c in bitstring[:n_orb]]
    beta_part = [int(c) for c in bitstring[n_orb:]]

    alpha_part = _correct_spin_part(alpha_part, n_alpha, occupancy[0], n_orb, rng)
    beta_part = _correct_spin_part(beta_part, n_beta, occupancy[1], n_orb, rng)

    return "".join(str(c) for c in alpha_part + beta_part)


@dataclass(frozen=True)
class SQDResult:
    """Outcome of one SQD solve.

    The subspace is always the full product of the two sector string lists, which
    is also the layout PySCF's selected-CI routines take.

    Attributes:
        energy: Lowest eigenvalue of the *spin-penalised* projected Hamiltonian,
            ``H + lambda (S^2 - s(s+1))^2``, plus ``constant`` -- not the bare
            expectation value ``<H>``. Batches are ranked on this deliberately,
            so a batch with a lower bare energy in the wrong spin sector loses;
            the cost is that on a spin-incomplete subspace this sits above
            ``<H>`` by the penalty term. Do not treat it as a variational bound.
            LASSQD's reported energy does not come from here: it is recomputed
            from the reassembled RDMs, which carry no penalty.
        amplitudes: Ground-state coefficients, ``amplitudes[i, j]`` belonging to
            the determinant pairing ``strings_alpha[i]`` with
            ``strings_beta[j]``.
        strings_alpha: Alpha sector occupation strings, ascending by
            :func:`ci_string_to_int`.
        strings_beta: Beta sector occupation strings, likewise ascending.
    """

    energy: float
    amplitudes: np.ndarray
    strings_alpha: tuple[str, ...]
    strings_beta: tuple[str, ...]

    @property
    def subspace(self) -> list[str]:
        """Blocked bitstrings spanning the subspace, ordered as
        ``amplitudes.ravel()``."""
        return [
            alpha + beta for alpha in self.strings_alpha for beta in self.strings_beta
        ]

    @property
    def eigenvector(self) -> np.ndarray:
        """``amplitudes`` flattened over :attr:`subspace`'s ordering."""
        return self.amplitudes.ravel()


class SQDSolver:
    """Self-consistent configuration recovery over sampled determinants.

    Implements arXiv:2405.05068. ``occupancy`` holds the running per-spin
    orbital occupancy estimate consumed by the self-consistent recovery step;
    it is refreshed at the end of every iteration from that iteration's batch
    results.

    Its settings come from an :class:`SQDConfig`, which validates them.
    """

    def __init__(
        self,
        n_orb: int,
        n_alpha: int,
        n_beta: int,
        config: SQDConfig,
        *,
        rng: np.random.Generator,
        recovery: bool = True,
    ):
        """Initialise the solver.

        Args:
            n_orb: Spatial orbitals in the fragment.
            n_alpha: Alpha electrons in the target sector.
            n_beta: Beta electrons in the target sector.
            config: Sampling, carryover and stopping settings.
            rng: Subsampling generator.
            recovery: Whether to run configuration recovery.
        """
        self.n_orb = n_orb
        self.n_alpha = n_alpha
        self.n_beta = n_beta
        self.n_batches = config.n_batches
        self.batch_size = config.batch_size
        self.n_iterations = config.n_recovery_iterations
        self.lambda_penalty = config.lambda_penalty
        self.recovery = recovery
        self.carryover_cutoff = config.carryover_cutoff
        self.max_carryover = config.max_carryover
        self.max_dim_alpha, self.max_dim_beta = (
            config.max_dim
            if isinstance(config.max_dim, tuple)
            else (config.max_dim, config.max_dim)
        )
        self.include_reference = config.include_reference
        # Exchanging the sectors is only a symmetry when they hold equal counts.
        self.symmetrize_spin = config.symmetrize_spin and n_alpha == n_beta
        self.energy_tol = config.recovery_energy_tol
        self.occupancies_tol = config.recovery_occupancies_tol
        self._rng = rng
        self.occupancy = np.zeros((2, n_orb))
        self._reference_alpha = _aufbau_string(n_orb, n_alpha)
        self._reference_beta = _aufbau_string(n_orb, n_beta)

    def solve(
        self,
        probs: dict[str, float],
        one_body: np.ndarray,
        two_body: np.ndarray,
        constant: float = 0.0,
        one_body_beta: np.ndarray | None = None,
        carried: tuple[tuple[str, ...], tuple[str, ...]] = ((), ()),
    ) -> SQDResult:
        """Run the SQD solver loop.

        Args:
            probs: Mapping of blocked bitstrings to their sampled probability.
            one_body: Spatial one-body integrals; the alpha channel when
                ``one_body_beta`` is given.
            two_body: Spatial two-body integrals.
            constant: Energy offset (e.g. nuclear repulsion) added to every
                projected eigenvalue.
            one_body_beta: Beta-channel one-body integrals, when the embedding
                potential is spin-dependent.
            carried: Alpha and beta strings carried in from an earlier solve,
                already in this solve's orbital basis. They join the first
                iteration's batches; the carryover rule then decides what stays.

        Returns:
            The lowest-energy :class:`SQDResult` found across all iterations.

        Raises:
            ValueError: If, in some iteration, no sampled bitstring can be
                brought into agreement with the target particle symmetry.
        """
        h_spin, g_spin = spin_orbital_integrals(
            one_body, two_body, self.n_orb, one_body_beta
        )
        target_s = 0.5 * abs(self.n_alpha - self.n_beta)

        best: SQDResult | None = None

        # Re-read each iteration, not accumulated: weights from differently
        # normalised eigenvectors are not comparable, and a carried string is
        # present in the current subspace anyway.
        carried_alpha, carried_beta = carried

        previous_energy: float | None = None
        previous_occupancy: np.ndarray | None = None

        for iteration in range(self.n_iterations):
            candidates = self._recovered_distribution(probs, iteration)

            results = [
                self._diagonalize(sectors, h_spin, g_spin, target_s, constant)
                for sectors in self._draw_batches(
                    candidates, carried_alpha, carried_beta
                )
            ]
            winner, occupancies = min(results, key=lambda pair: pair[0].energy)
            if best is None or winner.energy < best.energy:
                best = winner

            occupancy = np.mean([occ for _, occ in results], axis=0)
            converged = (
                previous_energy is not None
                and abs(previous_energy - winner.energy) < self.energy_tol
                and float(np.abs(occupancy - previous_occupancy).max())
                < self.occupancies_tol
            )
            self.occupancy = occupancy
            if converged:
                break
            previous_energy = winner.energy
            previous_occupancy = occupancy

            carried_alpha, carried_beta = self.carried_strings(winner)

        assert best is not None
        return best

    def carried_strings(
        self, result: SQDResult
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The alpha and beta strings carryover retains from ``result``, heaviest
        first; none when carryover is off."""
        if self.carryover_cutoff is None:
            return (), ()
        alpha_weights, beta_weights = carryover_weights(
            result.strings_alpha,
            result.strings_beta,
            result.amplitudes,
            self.carryover_cutoff,
        )
        return (
            _heaviest_strings(alpha_weights, self.max_carryover),
            _heaviest_strings(beta_weights, self.max_carryover),
        )

    def _recovered_distribution(
        self, probs: dict[str, float], iteration: int
    ) -> dict[str, float]:
        """The distribution this iteration samples its batches from.

        Iteration zero, and every iteration when ``recovery`` is off, postselects
        on particle number. Otherwise each bitstring is bit-flip corrected and its
        probability added to whatever the correction produced, so several samples
        collapsing onto one determinant leave it correspondingly heavier.

        Raises:
            ValueError: If no sampled bitstring survives postselection.
        """
        # Sorted, not insertion-ordered: a dict built in a different order would
        # otherwise draw a different subspace from the same seed.
        ordered = sorted(probs)
        weights: dict[str, float]
        if iteration == 0 or not self.recovery:
            weights = {
                bits: probs[bits]
                for bits in filter_symmetry(
                    ordered, self.n_orb, self.n_alpha, self.n_beta
                )
            }
        else:
            weights = {}
            for bits in ordered:
                corrected = bit_flip_correction(
                    bits,
                    self.n_orb,
                    self.n_alpha,
                    self.n_beta,
                    self.occupancy,
                    self._rng,
                )
                weights[corrected] = weights.get(corrected, 0.0) + probs[bits]

        if not weights:
            raise ValueError(
                "No valid configurations found matching particle symmetry!"
            )

        total = sum(weights.values())
        if total <= 0:
            uniform = 1.0 / len(weights)
            return {bits: uniform for bits in weights}
        return {bits: weight / total for bits, weight in weights.items()}

    def _draw_batches(
        self,
        candidates: dict[str, float],
        carried_alpha: Sequence[str],
        carried_beta: Sequence[str],
    ) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
        """Subsample ``n_batches`` subspaces, each as its two sector string lists.

        Batches draw without replacement (arXiv:2405.05068), capped at the number
        of configurations carrying positive probability.
        """
        strings = sorted(candidates)
        probabilities = np.array([candidates[bits] for bits in strings])
        size = min(self.batch_size, int(np.count_nonzero(probabilities)))
        # Split once per candidate rather than once per draw per batch.
        halves = {bits: (bits[: self.n_orb], bits[self.n_orb :]) for bits in strings}

        batches = []
        for _ in range(self.n_batches):
            sampled = self._rng.choice(
                strings, size=size, replace=False, p=probabilities
            )
            alpha_counts: dict[str, int] = {}
            beta_counts: dict[str, int] = {}
            for bits in sampled:
                alpha, beta = halves[bits]
                alpha_counts[alpha] = alpha_counts.get(alpha, 0) + 1
                beta_counts[beta] = beta_counts.get(beta, 0) + 1

            if self.symmetrize_spin:
                merged: dict[str, int] = dict(alpha_counts)
                for half, count in beta_counts.items():
                    merged[half] = merged.get(half, 0) + count
                alpha_counts = beta_counts = merged
                carried_alpha = carried_beta = sorted({*carried_alpha, *carried_beta})

            batches.append(
                (
                    self._sector_strings(
                        alpha_counts,
                        carried_alpha,
                        self._reference_alpha,
                        self.n_alpha,
                        self.max_dim_alpha,
                    ),
                    self._sector_strings(
                        beta_counts,
                        carried_beta,
                        self._reference_beta,
                        self.n_beta,
                        self.max_dim_beta,
                    ),
                )
            )
        return batches

    def _sector_strings(
        self,
        counts: dict[str, int],
        carried: Sequence[str],
        reference: str,
        target: int,
        max_dim: int | None,
    ) -> tuple[str, ...]:
        """One spin sector's strings, priority-ordered, capped, then sorted.

        Priority decides only what a binding ``max_dim`` keeps. The returned order
        is ascending :func:`ci_string_to_int`, which is what indexes the amplitude
        matrix.
        """
        sampled = sorted(counts, key=lambda half: (-counts[half], half))
        priority = ([reference] if self.include_reference else []) + list(carried)

        kept: dict[str, None] = {}
        for half in priority + sampled:
            if half.count("1") == target:
                kept.setdefault(half, None)
        selected = list(kept)[:max_dim] if max_dim is not None else list(kept)
        return tuple(sorted(selected, key=ci_string_to_int))

    def _diagonalize(
        self,
        sectors: tuple[tuple[str, ...], tuple[str, ...]],
        h_spin: np.ndarray,
        g_spin: np.ndarray,
        target_s: float,
        constant: float,
    ) -> tuple[SQDResult, np.ndarray]:
        """Diagonalize one batch's subspace, returning its result and occupancy."""
        strings_alpha, strings_beta = sectors
        # Occupations decoded once per sector string, not once per determinant.
        alpha_bits = _bit_matrix(strings_alpha, self.n_orb)
        beta_bits = _bit_matrix(strings_beta, self.n_orb)
        alpha_occs = _occupations_from_bit_matrix(alpha_bits)
        beta_occs = _occupations_from_bit_matrix(beta_bits)
        dets = [(alpha, beta) for alpha in alpha_occs for beta in beta_occs]
        dets_spin = [
            spatial_to_spin_occupations(alpha, beta, self.n_orb) for alpha, beta in dets
        ]

        h_proj, s2_proj = projected_matrices(
            dets, dets_spin, h_spin, g_spin, self.n_orb
        )
        deviation = s2_proj - target_s * (target_s + 1.0) * scipy.sparse.eye_array(
            len(dets), format="csr"
        )
        energy, eigenvector = ground_root(h_proj, deviation, self.lambda_penalty)
        amplitudes = eigenvector.reshape(len(strings_alpha), len(strings_beta))

        # A string's orbitals are occupied in every determinant built on it, so
        # the sector marginals suffice.
        probability = amplitudes**2
        occupancy = np.stack(
            [
                probability.sum(axis=1) @ alpha_bits,
                probability.sum(axis=0) @ beta_bits,
            ]
        )
        result = SQDResult(
            energy=float(energy + constant),
            amplitudes=amplitudes,
            strings_alpha=strings_alpha,
            strings_beta=strings_beta,
        )
        return result, occupancy


#: Widest fragment PySCF's selected-CI determinant addressing can take, whose
#: CI strings are 64-bit words. Above it the reconstruction falls back to the
#: in-house kernel, which carries unbounded Python integers.
_MAX_PYSCF_ORBITALS = 63


def compute_spatial_rdms(
    strings_alpha: Sequence[str],
    strings_beta: Sequence[str],
    amplitudes: np.ndarray,
    n_orb: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the spatial 1- and 2-RDM from an SQD state.

    Delegates to PySCF's selected-CI contractions; fragments wider than
    :data:`_MAX_PYSCF_ORBITALS` go through :func:`_spatial_rdms_exact` instead.

    Args:
        strings_alpha: Alpha sector strings indexing ``amplitudes``' rows,
            ascending by :func:`ci_string_to_int`.
        strings_beta: Beta sector strings indexing its columns, likewise.
        amplitudes: SQD ground-state coefficients over their product.
        n_orb: Number of spatial orbitals.

    Returns:
        ``(rdm1, rdm2, rdm1_alpha, rdm1_beta)`` spatial reduced density matrices
        in PySCF's active-space convention, where ``rdm1`` is the spin trace
        ``rdm1_alpha + rdm1_beta``.
    """
    if n_orb > _MAX_PYSCF_ORBITALS:
        return _spatial_rdms_exact(strings_alpha, strings_beta, amplitudes, n_orb)

    ci_strings = (
        np.array([ci_string_to_int(half) for half in strings_alpha], dtype=np.int64),
        np.array([ci_string_to_int(half) for half in strings_beta], dtype=np.int64),
    )
    nelec = (strings_alpha[0].count("1"), strings_beta[0].count("1"))
    civec = selected_ci._as_SCIvector(
        np.ascontiguousarray(amplitudes, dtype=float), ci_strings
    )
    rdm1_alpha, rdm1_beta = selected_ci.make_rdm1s(civec, n_orb, nelec)
    rdm2 = selected_ci.make_rdm2(civec, n_orb, nelec)
    return rdm1_alpha + rdm1_beta, rdm2, rdm1_alpha, rdm1_beta


def _spatial_rdms_exact(
    strings_alpha: Sequence[str],
    strings_beta: Sequence[str],
    amplitudes: np.ndarray,
    n_orb: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct both RDMs from the second-quantised definitions directly.

    Only determinant pairs within a double excitation of each other are
    visited, located by one occupation product per block of rows, and every
    ``a+_p a+_r a_s a_q`` term connecting them is accumulated straight into
    the spatial blocks. Memory therefore scales with ``n_orb ** 4`` rather
    than the spin-orbital ``(2 n_orb) ** 4``. Still far slower than
    :func:`compute_spatial_rdms`'s usual path.
    """
    alpha_bits = _bit_matrix(strings_alpha, n_orb)
    beta_bits = _bit_matrix(strings_beta, n_orb)
    occupation = np.concatenate(
        [
            np.repeat(alpha_bits, len(strings_beta), axis=0),
            np.tile(beta_bits, (len(strings_alpha), 1)),
        ],
        axis=1,
    ).astype(float)
    dets = _occupations_from_bit_matrix(occupation)
    det_sets = [frozenset(det) for det in dets]
    coefficients = np.asarray(amplitudes, dtype=float).ravel()
    n_electrons = len(dets[0])

    rdm1s = np.zeros((2, n_orb, n_orb))
    rdm2 = np.zeros((n_orb,) * 4)
    for start in range(0, len(dets), _PAIR_BLOCK_ROWS):
        stop = min(start + _PAIR_BLOCK_ROWS, len(dets))
        rank = n_electrons - np.rint(occupation[start:stop] @ occupation.T)
        for local, j in zip(*np.nonzero(rank <= 2)):
            i = int(local) + start
            weight = coefficients[i] * coefficients[j]
            det_j = dets[j]
            created = tuple(sorted(det_sets[i] - det_sets[j]))
            annihilated = tuple(sorted(det_sets[j] - det_sets[i]))
            common = sorted(det_sets[i] & det_sets[j])

            if not created:
                for p in det_j:
                    rdm1s[p // n_orb, p % n_orb, p % n_orb] += weight
                spectators = list(itertools.combinations(common, 2))
            elif len(created) == 1:
                (p,), (q,) = created, annihilated
                sign_q, occupied = _annihilation_sign(det_j, q)
                sign_p, _ = _creation_sign(occupied, p)
                rdm1s[p // n_orb, p % n_orb, q % n_orb] += weight * sign_p * sign_q
                spectators = [(t,) for t in common]
            else:
                spectators = [()]

            for spectator in spectators:
                for q, s in itertools.permutations(annihilated + spectator):
                    sign_q, occ_1 = _annihilation_sign(det_j, q)
                    sign_s, occ_2 = _annihilation_sign(occ_1, s)
                    for p, r in itertools.permutations(created + spectator):
                        if p // n_orb != q // n_orb:
                            continue
                        sign_r, occ_3 = _creation_sign(occ_2, r)
                        sign_p, _ = _creation_sign(occ_3, p)
                        rdm2[p % n_orb, q % n_orb, r % n_orb, s % n_orb] += (
                            weight * sign_q * sign_s * sign_r * sign_p
                        )

    rdm1_alpha, rdm1_beta = rdm1s
    return rdm1_alpha + rdm1_beta, rdm2, rdm1_alpha, rdm1_beta
