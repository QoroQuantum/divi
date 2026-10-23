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
import warnings
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import comb, isqrt
from typing import Literal

import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.sparse.linalg
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass as config_dataclass
from pyscf.fci import selected_ci

from ._types import exact_ints

#: Determinant rows per pass in :func:`projected_matrices`. The pair arrays it
#: builds scale with this times the subspace size, so a fixed block keeps peak
#: memory independent of how large the subspace grows; only the returned
#: matrices scale with its square.
_PAIR_BLOCK_ROWS = 512

#: Determinant-pair terms accumulated per pass in :func:`_spatial_rdms_exact`.
_RDM_TERMS_PER_PASS = 1 << 20

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
            f"bitstring width {len(bitstring)} does not match 2 * n_orb ({2 * n_orb})."
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

    diff_i = sorted(set_i - set_j)
    diff_j = sorted(set_j - set_i)

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
    alpha_occ = tuple(sorted(p for p in spin_occ if p < n_orb))
    beta_occ = tuple(sorted(p - n_orb for p in spin_occ if p >= n_orb))
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

    ``S^2``'s off-diagonal is non-zero only between determinants related by
    exchanging spins across two spatial orbitals, where it is +-1 with the sign
    of the ladder-operator parity.

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

    occupied = occupation.astype(bool)
    # 0/1 overlap counts are exact in single precision.
    occupation_f32 = occupation.astype(np.float32)

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
        overlap = occupation_f32[start:stop] @ occupation_f32.T
        connected_local, connected_cols = np.nonzero(overlap >= n_electrons - 2)
        connected_rank = n_electrons - overlap[connected_local, connected_cols]

        keep = connected_rank == 0
        local, cols = connected_local[keep], connected_cols[keep]
        rows = local + start
        h_entries.append((rows, cols, diagonal[rows]))
        s2_entries.append((rows, cols, s2_diagonal[rows]))

        # --- Single excitations: q in j replaced by p in i ---
        keep = connected_rank == 1
        local, cols = connected_local[keep], connected_cols[keep]
        if local.size:
            rows = local + start
            created = (occupied[rows] & ~occupied[cols]).argmax(axis=1)
            annihilated = (occupied[cols] & ~occupied[rows]).argmax(axis=1)
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
        keep = connected_rank == 2
        local, cols = connected_local[keep], connected_cols[keep]
        if local.size:
            rows = local + start
            # np.nonzero walks row-major and each row holds exactly two entries,
            # so every pair contributes its differing orbitals in ascending order.
            _, created_pair = np.nonzero(occupied[rows] & ~occupied[cols])
            _, annihilated_pair = np.nonzero(occupied[cols] & ~occupied[rows])
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
            # orbital and beta at another, losing exactly the opposite pair.
            spin_exchange = (
                (lower_created < n_orb)
                & (upper_created >= n_orb)
                & (lower_annihilated < n_orb)
                & (upper_annihilated >= n_orb)
                & (lower_created == upper_annihilated - n_orb)
                & (lower_annihilated == upper_created - n_orb)
            )
            exchange_cols = cols[spin_exchange]
            gained = lower_created[spin_exchange]
            lost = lower_annihilated[spin_exchange]
            # Every exchange pair has weight +-1: the ladder parity at both orbitals.
            ladder_parity = (
                below[exchange_cols, gained]
                + below[exchange_cols, n_orb + gained]
                + below[exchange_cols, lost]
                + below[exchange_cols, n_orb + lost]
            )
            s2_entries.append(
                (
                    rows[spin_exchange],
                    exchange_cols,
                    1.0 - 2.0 * (ladder_parity.astype(np.int64) % 2),
                )
            )

    return assemble(h_entries), assemble(s2_entries)


#: Subspace size from which the ground root is found iteratively rather than
#: densely.
_ITERATIVE_SUBSPACE_MIN = 512

#: Root convergence for the iterative solve, well inside the energies LASSQD
#: resolves.
_ITERATIVE_TOLERANCE = 1e-10

#: Residual norm a block solve's roots must reach to be accepted.
_BLOCK_RESIDUAL_TOL = 1e-7

#: Block vectors beyond the requested roots, so degenerate roots fit the block.
_BLOCK_MARGIN = 4

#: Iteration cap for the block solve.
_BLOCK_MAX_ITERATIONS = 2_000

#: Largest subspace the dense fallback may diagonalise; one dense float64
#: matrix of this size is 2 GiB, and the solve holds several.
_DENSE_MAX_DIM = 16_384


def lowest_roots(
    h_proj: scipy.sparse.csr_array,
    deviation: scipy.sparse.csr_array,
    lambda_penalty: float,
    n_roots: int = 1,
    *,
    spin: float | None = None,
    block: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Lowest eigenpairs of the spin-penalised projected Hamiltonian.

    ``H + lambda * deviation ** 2`` is diagonalised densely below
    ``_ITERATIVE_SUBSPACE_MIN`` determinants and iteratively above, with the
    penalty applied as an operator. The iterative lowest root is checked against
    the smallest diagonal entry, an upper bound on the true one; a root above it,
    or a solve that does not converge, falls back to the dense form.

    With ``spin`` given, roots whose ``<S^2>`` lies nearer another spin's
    ``S(S+1)`` are skipped, since a penalty smaller than a gap lets them
    outrank the target spin's roots. The iterative solve widens until enough
    remain; when the subspace holds too few, the lowest skipped roots fill in.

    Iterative solves start from a fixed vector, so repeats agree to the last bit.
    The single-root solve is Lanczos from the lowest-diagonal determinant. A
    block solve (LOBPCG from a fixed-seed random block wider than ``n_roots``)
    reaches degenerate roots and spin sectors that determinant never touches.

    Args:
        h_proj: The projected Hamiltonian.
        deviation: ``S^2`` projected onto the subspace, less the target
            eigenvalue on the diagonal.
        lambda_penalty: Weight of the spin-contamination penalty.
        n_roots: How many of the lowest roots to return. More than one implies
            ``block``.
        spin: The target ``S`` that ``deviation`` is taken from; ``None`` keeps
            roots of any spin.
        block: Use the block solve even for one root.

    Returns:
        ``(eigenvalues, eigenvectors)``: ascending eigenvalues, and the
        eigenvectors as columns in the same order.

    Raises:
        ValueError: If the subspace holds fewer than ``n_roots`` determinants,
            or if the iterative solve fails on a subspace too large for the
            dense fallback.
    """
    # pyrefly: ignore[unsupported-operation]  # scipy leaves _shape unannotated
    dimension = h_proj.shape[0]
    if n_roots > dimension:
        raise ValueError(
            f"A {dimension}-determinant subspace is too small for {n_roots} "
            "roots. Raise batch_size, n_recovery_iterations or max_dim, or "
            "request fewer roots in SQDConfig.states."
        )

    def select(values, vectors):
        """The lowest ``n_roots`` of the target spin, and whether there were
        enough of them."""
        values, vectors = np.asarray(values), np.asarray(vectors)
        on_spin = (
            np.ones(len(values), dtype=bool)
            if spin is None
            else _nearest_spin(vectors, deviation, spin) == spin
        )
        order = [*np.flatnonzero(on_spin), *np.flatnonzero(~on_spin)]
        kept = np.sort(order[:n_roots])
        return values[kept], vectors[:, kept], on_spin.sum() >= n_roots

    block = block or n_roots > 1
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
        if not block and n_roots + 1 < dimension:
            solved = _lanczos_root(penalized, diagonal)
            if solved is not None and solved[0][0] <= diagonal.min():
                values, vectors, enough = select(*solved)
                if enough:
                    return values, vectors
                # The lowest root has another spin; look deeper with a block.
                block = True
        n_solved, filled_in = n_roots, None
        # LOBPCG needs the space several blocks wide.
        while block and 5 * (n_solved + _BLOCK_MARGIN) < dimension:
            solved = _block_roots(
                penalized, dimension, n_solved, n_solved + _BLOCK_MARGIN
            )
            if solved is None or solved[0][0] > diagonal.min():
                break
            values, vectors, enough = select(*solved)
            if enough:
                return values, vectors
            filled_in = values, vectors
            n_solved *= 2
        if dimension > _DENSE_MAX_DIM:
            if filled_in is not None:
                return filled_in
            raise ValueError(
                f"The iterative eigensolver failed on a {dimension}-determinant "
                "subspace, and a dense solve would need "
                f"{dimension**2 * 8 / 2**30:.1f} GiB per matrix. Set "
                "SQDConfig.max_dim to bound the subspace."
            )

    values, vectors, _ = select(
        *scipy.linalg.eigh(
            (h_proj + lambda_penalty * (deviation @ deviation)).toarray()
        )
    )
    return values, vectors


def _nearest_spin(
    vectors: np.ndarray, deviation: scipy.sparse.csr_array, spin: float
) -> np.ndarray:
    """For each column of ``vectors``, the ``S`` whose ``S(S+1)`` is nearest its
    ``<S^2>``, in steps of one from ``spin``."""
    spin_square = np.einsum("ij,ij->j", vectors, deviation @ vectors) + spin * (
        spin + 1.0
    )
    return spin + np.round(_spin_of(spin_square) - spin)


def resolves_spin(multiplicity: int, spin_square: float) -> bool:
    """Whether ``<S^2>`` lies nearer ``multiplicity``'s ``S(S+1)`` than the
    spins one step either side, as :func:`lowest_roots` judges a root."""
    return round(float(_spin_of(spin_square)) - 0.5 * (multiplicity - 1)) == 0


def _spin_of(spin_square):
    """The ``S`` with ``S(S+1) = <S^2>``, negative ``<S^2>`` read as 0."""
    return 0.5 * (np.sqrt(1.0 + 4.0 * np.maximum(spin_square, 0.0)) - 1.0)


def _lanczos_root(
    penalized: scipy.sparse.linalg.LinearOperator, diagonal: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Lowest root by Lanczos from the lowest-diagonal determinant, or ``None``
    if it does not converge."""
    start = np.zeros(len(diagonal))
    start[int(np.argmin(diagonal))] = 1.0
    try:
        values, vectors = scipy.sparse.linalg.eigsh(
            penalized, k=2, which="SA", v0=start, tol=_ITERATIVE_TOLERANCE
        )
    except scipy.sparse.linalg.ArpackError:
        return None
    lowest = np.argsort(values, kind="stable")[:1]
    return values[lowest], np.asarray(vectors)[:, lowest]


def _block_roots(
    penalized: scipy.sparse.linalg.LinearOperator,
    dimension: int,
    n_roots: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Lowest ``n_roots`` roots by LOBPCG from a fixed-seed random block, or
    ``None`` unless every one reaches ``_BLOCK_RESIDUAL_TOL``."""
    start = np.random.default_rng(0).standard_normal((dimension, block_size))
    with warnings.catch_warnings():
        # Non-convergence is detected from the residuals below.
        warnings.simplefilter("ignore", UserWarning)
        values, vectors = scipy.sparse.linalg.lobpcg(
            penalized,
            start,
            largest=False,
            tol=_BLOCK_RESIDUAL_TOL,
            maxiter=_BLOCK_MAX_ITERATIONS,
        )
    order = np.argsort(values, kind="stable")[:n_roots]
    values, vectors = values[order], np.asarray(vectors)[:, order]
    residual = penalized.matmat(vectors) - vectors * values
    if np.linalg.norm(residual, axis=0).max() > _BLOCK_RESIDUAL_TOL:
        return None
    return values, vectors


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


def subspace_determinants(
    strings_alpha: Sequence[str], strings_beta: Sequence[str], n_orb: int
) -> tuple[list[tuple[tuple[int, ...], tuple[int, ...]]], list[tuple[int, ...]]]:
    """The alpha x beta product's determinants, alpha-major, as spatial
    ``(alpha_occ, beta_occ)`` pairs and as spin-orbital tuples."""
    # Occupations decoded once per sector string, not once per determinant.
    alpha_occs = _occupations_from_bit_matrix(_bit_matrix(strings_alpha, n_orb))
    beta_occs = _occupations_from_bit_matrix(_bit_matrix(strings_beta, n_orb))
    dets = list(itertools.product(alpha_occs, beta_occs))
    dets_spin = [
        spatial_to_spin_occupations(alpha, beta, n_orb) for alpha, beta in dets
    ]
    return dets, dets_spin


def string_hops(string: str, rank: int) -> list[str]:
    """Strings reached from ``string`` by moving one to ``rank`` electrons to
    empty orbitals, in a fixed order."""
    occupied = [index for index, bit in enumerate(string) if bit == "1"]
    empty = [index for index, bit in enumerate(string) if bit == "0"]
    hops = []
    for moved in range(1, rank + 1):
        for vacated, filled in itertools.product(
            itertools.combinations(occupied, moved),
            itertools.combinations(empty, moved),
        ):
            bits = list(string)
            for index in vacated:
                bits[index] = "0"
            for index in filled:
                bits[index] = "1"
            hops.append("".join(bits))
    return hops


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
    return [
        tuple(cols[start:stop].tolist()) for start, stop in itertools.pairwise(bounds)
    ]


@config_dataclass(frozen=True)
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
            orbitals. ``'assignment'`` (what ``None``, the default, means)
            pairs old and new orbitals one-to-one by maximum total overlap, so
            every string keeps its electron count; ``'argmax'`` gives each new
            orbital the occupation of the old orbital it overlaps most, as the
            reference implementation does, and drops strings whose electron
            count that changes. Rejected without ``carryover_cutoff``.
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
            ``n_alpha == n_beta``. ``None`` (the default) turns it on exactly
            when ``states`` is set, which ``False`` cannot be combined with.
        recovery_energy_tol: Ends a fragment's recovery once the winning energy
            moves less than this between iterations and the occupancies have also
            settled. ``0.0`` (the default) spends every iteration, since a
            settled iteration does not mean carryover had nothing left to add.
            Not ``LASSQD``'s ``energy_tol``, which ends the macro-cycle.
        recovery_occupancies_tol: The occupancy half of that test, on the largest
            change in any orbital's average occupancy.
        states: Eigenstates to solve for, as a mapping from spin multiplicity
            ``2S + 1`` to a number of roots: ``{1: 2, 3: 1}`` asks for the two
            lowest singlets and the lowest triplet. All come from one subspace.
            ``None`` (the default) solves for the lowest state of the
            fragment's own ``Sz``. Each multiplicity is targeted by the
            ``lambda_penalty`` term, and roots whose ``<S^2>`` is nearer
            another spin are skipped. A sampled subspace may hold no pure state
            of a spin; its nearest roots are then reported, off-spin.
        extension: Ext-SQD (arXiv:2411.00468). Recovery then solves for the
            lowest state only; its winning subspace gains every string its
            significant strings reach by moving one electron (``'singles'``)
            or one or two (``'doubles'``), per spin sector, and is diagonalised
            for ``states``. Recovers determinants the circuit under-samples,
            which leave even ground-state energies mHa high without it.
            ``'auto'`` (the default) means ``'singles'``, and an extension on
            a fragment above 16384 determinants needs ``max_dim``, since it can
            grow to the whole space. ``None`` gives plain SQD.
        extension_cutoff: A string is significant when some determinant
            containing it has ``abs(coefficient)`` above this fraction of the
            largest, in any requested state. ``None`` (the default) means
            ``1e-3``. Rejected with ``extension=None``.

    Raises:
        ValueError: If a field has the wrong type or is out of range: a count
            below 1, a negative or non-finite tolerance or penalty, a cutoff
            outside ``(0, 1)``, an unknown ``carryover_mapping`` or
            ``extension``, or a ``max_dim`` that is neither one positive integer
            nor a pair of them. Also if ``carryover_mapping`` or
            ``max_carryover`` is given without a cutoff; if
            ``extension_cutoff`` is given with ``extension=None``; if
            ``symmetrize_spin`` is ``False`` with ``states`` set; or if
            ``states`` is empty.
    """

    n_batches: int = Field(default=15, ge=1)
    batch_size: int = Field(default=170, ge=1)
    n_recovery_iterations: int = Field(default=6, ge=1)
    lambda_penalty: float = Field(default=0.2, ge=0, allow_inf_nan=False)
    carryover_cutoff: float | None = Field(
        default=_DEFAULT_CARRYOVER_CUTOFF, gt=0, lt=1, allow_inf_nan=False
    )
    carryover_mapping: Literal["assignment", "argmax"] | None = None
    max_carryover: int | None = Field(default=None, ge=1)
    max_dim: int | tuple[int, int] | None = None
    include_reference: bool = True
    symmetrize_spin: bool | None = None
    recovery_energy_tol: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    recovery_occupancies_tol: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    states: Mapping[int, int] | None = field(default=None, hash=False)
    extension: Literal["auto", "singles", "doubles"] | None = "auto"
    extension_cutoff: float | None = Field(
        default=None, gt=0, lt=1, allow_inf_nan=False
    )

    _exact_ints = field_validator(
        "n_batches",
        "batch_size",
        "n_recovery_iterations",
        "max_carryover",
        "max_dim",
        "states",
        mode="before",
    )(exact_ints)

    def __post_init__(self):
        if any(dim is not None and dim < 1 for dim in self.resolved_max_dim):
            raise ValueError(f"max_dim must be at least 1; got {self.max_dim}.")
        if self.carryover_mapping is not None and self.carryover_cutoff is None:
            raise ValueError(
                "carryover_mapping maps the strings carryover retains, so it "
                "needs carryover_cutoff to be set."
            )
        if self.max_carryover is not None and self.carryover_cutoff is None:
            raise ValueError(
                "max_carryover caps what carryover retains, so it needs "
                "carryover_cutoff to be set."
            )
        if self.extension_cutoff is not None and self.extension is None:
            raise ValueError(
                "extension_cutoff picks the strings the extension grows from, "
                "so it needs extension to be set."
            )
        if self.states is None:
            return
        if self.symmetrize_spin is False:
            raise ValueError(
                "symmetrize_spin=False contradicts states, which always pools "
                "the alpha and beta halves. Leave symmetrize_spin unset."
            )
        if not self.states:
            raise ValueError("states must request at least one state.")
        if min(*self.states, *self.states.values()) < 1:
            raise ValueError(
                "states multiplicities and root counts must be at least 1; got "
                f"{dict(self.states)}."
            )
        object.__setattr__(self, "states", dict(sorted(self.states.items())))

    @property
    def resolved_extension(self) -> Literal["singles", "doubles"] | None:
        """The extension that runs, with ``'auto'`` resolved."""
        return "singles" if self.extension == "auto" else self.extension

    @property
    def resolved_extension_cutoff(self) -> float:
        """The extension cutoff in force, with ``None`` resolved."""
        return 1e-3 if self.extension_cutoff is None else self.extension_cutoff

    @property
    def resolved_carryover_mapping(self) -> Literal["assignment", "argmax"]:
        """The carryover mapping in force, with ``None`` resolved."""
        return self.carryover_mapping or "assignment"

    @property
    def resolved_max_dim(self) -> tuple[int | None, int | None]:
        """The ``(alpha, beta)`` caps, ``None`` where a sector is uncapped."""
        if isinstance(self.max_dim, tuple):
            return self.max_dim
        return (self.max_dim, self.max_dim)


def spin_square_of(multiplicity: int) -> float:
    """``S(S + 1)`` for spin multiplicity ``2S + 1``."""
    spin = 0.5 * (multiplicity - 1)
    return spin * (spin + 1.0)


def full_space_size(n_orb: int, n_alpha: int, n_beta: int) -> int:
    """Number of determinants with these occupations."""
    return comb(n_orb, n_alpha) * comb(n_orb, n_beta)


def default_states(n_alpha: int, n_beta: int) -> dict[int, int]:
    """The states solved when ``SQDConfig.states`` is ``None``: the lowest
    state of the fragment's own ``Sz``."""
    return {abs(n_alpha - n_beta) + 1: 1}


def validate_sqd_for_fragment(
    config: "SQDConfig", n_orb: int, n_alpha: int, n_beta: int
) -> None:
    """Reject settings a fragment cannot satisfy, before any circuit runs.

    Raises:
        ValueError: If a requested multiplicity has the wrong parity for the
            electron count, lies below ``2 |Sz| + 1``, or asks for more roots
            than the fragment has states of that spin; if states beyond
            :func:`default_states` are requested with ``lambda_penalty`` at 0;
            or if the extension could grow the subspace past
            ``_DENSE_MAX_DIM`` determinants with ``max_dim`` unset.
    """
    full_size = full_space_size(n_orb, n_alpha, n_beta)
    if (
        config.resolved_extension is not None
        and config.max_dim is None
        and full_size > _DENSE_MAX_DIM
    ):
        raise ValueError(
            f"SQDConfig.extension={config.extension!r} can grow this fragment's "
            f"subspace to its full {full_size} determinants, past the "
            f"{_DENSE_MAX_DIM} the solver handles unbounded. Set "
            f"SQDConfig.max_dim to cap each spin sector (max_dim="
            f"{isqrt(_DENSE_MAX_DIM)} keeps it within {_DENSE_MAX_DIM}), "
            "or extension=None for plain SQD, which misses the determinants "
            "the circuit under-samples."
        )
    if config.states is not None:
        _validate_states(config.states, n_orb, n_alpha, n_beta, config.lambda_penalty)


def _validate_states(
    states: Mapping[int, int],
    n_orb: int,
    n_alpha: int,
    n_beta: int,
    lambda_penalty: float,
) -> None:
    """Reject a states request the fragment cannot satisfy; see
    :func:`validate_sqd_for_fragment`."""
    n_electrons = n_alpha + n_beta

    def sector_size(two_ms: int) -> int:
        """Determinants with ``n_alpha - n_beta == two_ms``."""
        if (n_electrons + two_ms) % 2 or two_ms > n_electrons:
            return 0
        return comb(n_orb, (n_electrons + two_ms) // 2) * comb(
            n_orb, (n_electrons - two_ms) // 2
        )

    lowest = abs(n_alpha - n_beta) + 1
    for multiplicity, n_roots in states.items():
        if (multiplicity - lowest) % 2:
            raise ValueError(
                f"SQDConfig.states asks for multiplicity {multiplicity}, which "
                f"{n_electrons} electrons cannot reach: a multiplicity and its "
                "electron count have opposite parity."
            )
        if multiplicity < lowest:
            raise ValueError(
                f"SQDConfig.states asks for multiplicity {multiplicity}, below "
                f"this fragment's lowest, {lowest}: S cannot be below "
                f"|Sz| with n_alpha={n_alpha}, n_beta={n_beta}."
            )
        # States of spin S are those of Ms = S less those of spin S + 1.
        available = sector_size(multiplicity - 1) - sector_size(multiplicity + 1)
        if n_roots > available:
            raise ValueError(
                f"SQDConfig.states asks for {n_roots} of multiplicity "
                f"{multiplicity}, but {n_electrons} electrons in {n_orb} "
                f"orbitals have only {available} of that multiplicity."
            )
    if lambda_penalty == 0 and dict(states) != default_states(n_alpha, n_beta):
        raise ValueError(
            "SQDConfig.states asks for more than the lowest state, which needs "
            "lambda_penalty > 0: the S^2 penalty is what targets each "
            "multiplicity."
        )


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
        amplitudes: Coefficients, one row per alpha string. A leading axis
            stacks several states: a string is then eligible if it is for any
            state, against that state's largest coefficient, and its weight is
            summed over the states.
        cutoff: Retain strings appearing in a determinant whose
            ``abs(coefficient)`` exceeds this fraction of the largest.

    Returns:
        ``(alpha_weights, beta_weights)``, each mapping string to weight.
    """
    coefficients = np.asarray(amplitudes, dtype=float)
    if coefficients.ndim == 2:
        coefficients = coefficients[np.newaxis]
    if coefficients.size == 0:
        return {}, {}
    magnitude = np.abs(coefficients)
    largest = magnitude.max(axis=(1, 2), keepdims=True)
    if not largest.any():
        return {}, {}

    eligible = (magnitude > cutoff * largest).any(axis=0)
    probability = (coefficients**2).sum(axis=0)
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
    return list(itertools.compress(bitstrings, keep))


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
class SQDState:
    """One eigenstate of an SQD solve.

    Attributes:
        multiplicity: Spin multiplicity ``2S + 1`` the penalty targeted.
        index: Position among that multiplicity's roots, ``0`` the lowest.
        penalised_energy: Eigenvalue of the spin-penalised projected
            Hamiltonian, ``H + lambda (S^2 - S(S+1))^2``, plus the solve's
            ``constant``; not ``<H>``, and not a variational bound. LASSQD
            reports energies from the RDMs instead.
        spin_square: ``<S^2>`` of the state, without the penalty.
        amplitudes: Coefficients, ``amplitudes[i, j]`` belonging to the
            determinant pairing the result's ``strings_alpha[i]`` with its
            ``strings_beta[j]``.
    """

    multiplicity: int
    index: int
    penalised_energy: float
    spin_square: float
    amplitudes: np.ndarray


@dataclass(frozen=True)
class SQDResult:
    """Outcome of one SQD solve.

    The subspace is always the full product of the two sector string lists, which
    is also the layout PySCF's selected-CI routines take.

    Attributes:
        states: Every requested eigenstate, ascending in penalised energy.
        strings_alpha: Alpha sector occupation strings, ascending by
            :func:`ci_string_to_int`.
        strings_beta: Beta sector occupation strings, likewise ascending.
        carried: The alpha and beta strings carryover retained, taken before
            any extension; they seed the next solve.
    """

    states: tuple[SQDState, ...]
    strings_alpha: tuple[str, ...]
    strings_beta: tuple[str, ...]
    carried: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())

    @property
    def energy(self) -> float:
        """The lowest state's :attr:`SQDState.penalised_energy`."""
        return self.states[0].penalised_energy

    @property
    def amplitudes(self) -> np.ndarray:
        """Coefficients of the lowest state."""
        return self.states[0].amplitudes

    @property
    def objective(self) -> float:
        """Sum of every state's penalised energy, which batches are ranked on,
        so all states come from one subspace. A ranking, not a bound."""
        return sum(state.penalised_energy for state in self.states)

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

        Raises:
            ValueError: If ``config`` asks for something this fragment cannot
                hold; see :func:`validate_sqd_for_fragment`.
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
        self.max_dim_alpha, self.max_dim_beta = config.resolved_max_dim
        self.include_reference = config.include_reference
        # Exchanging the sectors is only a symmetry when they hold equal counts.
        self.symmetrize_spin = (
            config.symmetrize_spin or config.states is not None
        ) and n_alpha == n_beta
        self.energy_tol = config.recovery_energy_tol
        self.occupancies_tol = config.recovery_occupancies_tol
        validate_sqd_for_fragment(config, n_orb, n_alpha, n_beta)
        lowest = default_states(n_alpha, n_beta)
        self.states = lowest if config.states is None else dict(config.states)
        self.extension = config.resolved_extension
        self.extension_cutoff = config.resolved_extension_cutoff
        # With the extension on, recovery finds the lowest state's subspace and
        # only the extended one is solved for every requested state.
        self._recovery_states = lowest if self.extension else self.states
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
            The :class:`SQDResult` with the lowest
            :attr:`~SQDResult.objective` found across all iterations.

        Raises:
            ValueError: If, in some iteration, no sampled bitstring can be
                brought into agreement with the target particle symmetry, or a
                batch's subspace cannot hold the requested roots.
        """
        h_spin, g_spin = spin_orbital_integrals(
            one_body, two_body, self.n_orb, one_body_beta
        )

        winners: list[SQDResult] = []

        # Re-read each iteration, not accumulated: weights from differently
        # normalised eigenvectors are not comparable, and a carried string is
        # present in the current subspace anyway.
        carried_alpha, carried_beta = carried

        previous_objective: float | None = None
        previous_occupancy: np.ndarray | None = None

        for iteration in range(self.n_iterations):
            candidates = self._recovered_distribution(probs, iteration)

            results = [
                self._diagonalize(
                    sectors, h_spin, g_spin, constant, self._recovery_states
                )
                for sectors in self._draw_batches(
                    candidates, carried_alpha, carried_beta
                )
            ]
            winner, occupancies = min(results, key=lambda pair: pair[0].objective)
            winners.append(winner)

            occupancy = np.mean([occ for _, occ in results], axis=0)
            converged = (
                previous_objective is not None
                and abs(previous_objective - winner.objective) < self.energy_tol
                and float(np.abs(occupancy - previous_occupancy).max())
                < self.occupancies_tol
            )
            self.occupancy = occupancy
            if converged:
                break
            previous_objective = winner.objective
            previous_occupancy = occupancy

            carried_alpha, carried_beta = self.carried_strings(winner)

        best = min(winners, key=lambda result: result.objective)
        # Carried from before the extension, which would otherwise seed every
        # batch of the next solve with the whole extended subspace.
        extended = (
            best
            if self.extension is None
            else self._extend(best, h_spin, g_spin, constant)
        )
        return replace(extended, carried=self.carried_strings(best))

    def _extend(
        self,
        result: SQDResult,
        h_spin: np.ndarray,
        g_spin: np.ndarray,
        constant: float,
    ) -> SQDResult:
        """Ext-SQD (arXiv:2411.00468): enlarge ``result``'s subspace with the
        strings its significant strings reach by moving electrons, and
        diagonalise once more.

        ``result``'s own strings stay in, so the subspace only grows. Strings are
        kept in priority order -- reference, significant by weight, the rest of
        ``result``'s, then hops from the heaviest parents first -- so
        ``max_dim`` stays a hard cap; :func:`validate_sqd_for_fragment` has
        already required it wherever the full space is too large to grow into.
        """
        rank = 1 if self.extension == "singles" else 2
        alpha_weights, beta_weights = carryover_weights(
            result.strings_alpha,
            result.strings_beta,
            np.stack([state.amplitudes for state in result.states]),
            self.extension_cutoff,
        )
        existing_alpha, existing_beta = result.strings_alpha, result.strings_beta
        if self.symmetrize_spin:
            alpha_weights = Counter(alpha_weights)
            alpha_weights.update(beta_weights)
            beta_weights = alpha_weights
            existing_alpha = existing_beta = tuple(
                sorted({*existing_alpha, *existing_beta}, key=ci_string_to_int)
            )

        def sector(weights, existing, reference, target, max_dim):
            parents = _heaviest_strings(weights, None)
            hops = [hop for parent in parents for hop in string_hops(parent, rank)]
            return self._sector_strings(
                {}, [*parents, *existing, *hops], reference, target, max_dim
            )

        strings_alpha = sector(
            alpha_weights,
            existing_alpha,
            self._reference_alpha,
            self.n_alpha,
            self.max_dim_alpha,
        )
        strings_beta = sector(
            beta_weights,
            existing_beta,
            self._reference_beta,
            self.n_beta,
            self.max_dim_beta,
        )
        extended, _ = self._diagonalize(
            (strings_alpha, strings_beta), h_spin, g_spin, constant, self.states
        )
        return extended

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
            np.stack([state.amplitudes for state in result.states]),
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
            weights = defaultdict(float)
            for bits in ordered:
                corrected = bit_flip_correction(
                    bits,
                    self.n_orb,
                    self.n_alpha,
                    self.n_beta,
                    self.occupancy,
                    self._rng,
                )
                weights[corrected] += probs[bits]

        if not weights:
            raise ValueError(
                "No valid configurations found matching particle symmetry! "
                "Increase the backend shot count or n_recovery_iterations."
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
        string_array = np.array(strings)
        probabilities = np.array([candidates[bits] for bits in strings])
        size = min(self.batch_size, int(np.count_nonzero(probabilities)))
        # Split once per candidate rather than once per draw per batch.
        halves = {bits: (bits[: self.n_orb], bits[self.n_orb :]) for bits in strings}

        batches = []
        for _ in range(self.n_batches):
            sampled = self._rng.choice(
                string_array, size=size, replace=False, p=probabilities
            )
            alpha_counts: Counter[str] = Counter()
            beta_counts: Counter[str] = Counter()
            for bits in sampled:
                alpha, beta = halves[bits]
                alpha_counts[alpha] += 1
                beta_counts[beta] += 1

            if self.symmetrize_spin:
                alpha_counts.update(beta_counts)
                beta_counts = alpha_counts
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

        kept = dict.fromkeys(
            half
            for half in itertools.chain(priority, sampled)
            if half.count("1") == target
        )
        selected = list(kept)[:max_dim]
        return tuple(sorted(selected, key=ci_string_to_int))

    def _diagonalize(
        self,
        sectors: tuple[tuple[str, ...], tuple[str, ...]],
        h_spin: np.ndarray,
        g_spin: np.ndarray,
        constant: float,
        states: Mapping[int, int] | None = None,
    ) -> tuple[SQDResult, np.ndarray]:
        """Diagonalise one subspace for ``states`` (default: the requested
        ones), returning its result and occupancy.

        Each multiplicity is a separate solve of the same projected Hamiltonian,
        penalised toward its own ``S(S+1)``. The occupancy is averaged over the
        states.
        """
        states = self.states if states is None else states
        block = states != default_states(self.n_alpha, self.n_beta)
        strings_alpha, strings_beta = sectors
        alpha_bits = _bit_matrix(strings_alpha, self.n_orb)
        beta_bits = _bit_matrix(strings_beta, self.n_orb)
        dets, dets_spin = subspace_determinants(strings_alpha, strings_beta, self.n_orb)

        h_proj, s2_proj = projected_matrices(
            dets, dets_spin, h_spin, g_spin, self.n_orb
        )
        identity = scipy.sparse.eye_array(len(dets), format="csr")
        solved = []
        for multiplicity, n_roots in states.items():
            # A subspace too small for every root reports those it holds.
            n_roots = min(n_roots, len(strings_alpha) * len(strings_beta))
            deviation = s2_proj - spin_square_of(multiplicity) * identity
            values, vectors = lowest_roots(
                h_proj,
                deviation,
                self.lambda_penalty,
                n_roots,
                spin=0.5 * (multiplicity - 1),
                block=block,
            )
            for index in range(n_roots):
                vector = vectors[:, index]
                solved.append(
                    SQDState(
                        multiplicity=multiplicity,
                        index=index,
                        penalised_energy=float(values[index] + constant),
                        spin_square=float(vector @ (s2_proj @ vector)),
                        amplitudes=vector.reshape(
                            len(strings_alpha), len(strings_beta)
                        ),
                    )
                )
        solved.sort(key=lambda state: state.penalised_energy)

        # A string's orbitals are occupied in every determinant built on it, so
        # the sector marginals suffice.
        probability = np.mean([state.amplitudes**2 for state in solved], axis=0)
        occupancy = np.stack(
            [
                probability.sum(axis=1) @ alpha_bits,
                probability.sum(axis=0) @ beta_bits,
            ]
        )
        result = SQDResult(
            states=tuple(solved),
            strings_alpha=strings_alpha,
            strings_beta=strings_beta,
        )
        return result, occupancy


#: Widest fragment PySCF's selected-CI determinant addressing can take, whose
#: CI strings are 64-bit words. Above it the reconstruction falls back to the
#: in-house kernel, which carries unbounded Python integers.
_MAX_PYSCF_ORBITALS = 63


def sci_vector(
    strings_alpha: Sequence[str], strings_beta: Sequence[str], amplitudes: np.ndarray
) -> np.ndarray:
    """``amplitudes`` as a PySCF selected-CI vector over the two string lists."""
    ci_strings = (
        np.array([ci_string_to_int(half) for half in strings_alpha], dtype=np.int64),
        np.array([ci_string_to_int(half) for half in strings_beta], dtype=np.int64),
    )
    return selected_ci._as_SCIvector(
        np.ascontiguousarray(amplitudes, dtype=float), ci_strings
    )


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

    nelec = (strings_alpha[0].count("1"), strings_beta[0].count("1"))
    civec = sci_vector(strings_alpha, strings_beta, amplitudes)
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
    occupied = np.concatenate(
        [
            np.repeat(alpha_bits, len(strings_beta), axis=0),
            np.tile(beta_bits, (len(strings_alpha), 1)),
        ],
        axis=1,
    ).astype(bool)
    occupation = occupied.astype(np.float32)
    n_dets = len(occupied)
    n_electrons = int(occupied[0].sum())
    occupied_orbitals = np.nonzero(occupied)[1].reshape(n_dets, n_electrons)
    below = np.cumsum(occupied, axis=1, dtype=np.int32) - occupied
    coefficients = np.asarray(amplitudes, dtype=float).ravel()

    rdm1s = np.zeros((2, n_orb, n_orb))
    rdm2 = np.zeros((n_orb,) * 4)
    for start in range(0, n_dets, _PAIR_BLOCK_ROWS):
        stop = min(start + _PAIR_BLOCK_ROWS, n_dets)
        overlap = occupation[start:stop] @ occupation.T
        local, cols = np.nonzero(overlap >= n_electrons - 2)
        rank = n_electrons - overlap[local, cols]
        rows = local + start
        weight = coefficients[rows] * coefficients[cols]

        keep = rank == 0
        pair_cols, pair_weight = cols[keep], weight[keep]
        if pair_cols.size:
            orbitals = occupied_orbitals[pair_cols]
            np.add.at(
                rdm1s,
                (orbitals // n_orb, orbitals % n_orb, orbitals % n_orb),
                np.broadcast_to(pair_weight[:, None], orbitals.shape),
            )
            first, second = np.triu_indices(n_electrons, 1)
            for chunk in _spectator_chunks(len(first), pair_cols.size):
                lower = orbitals[:, first[chunk]]
                width = lower.shape[1]
                pair = np.stack(
                    [lower.ravel(), orbitals[:, second[chunk]].ravel()], axis=1
                )
                _accumulate_two_body(
                    rdm2,
                    below,
                    n_orb,
                    np.repeat(pair_cols, width),
                    np.repeat(pair_weight, width),
                    pair,
                    pair,
                )

        keep = rank == 1
        pair_rows, pair_cols, pair_weight = rows[keep], cols[keep], weight[keep]
        if pair_cols.size:
            created = np.nonzero(occupied[pair_rows] & ~occupied[pair_cols])[1]
            annihilated = np.nonzero(occupied[pair_cols] & ~occupied[pair_rows])[1]
            exponent = (
                below[pair_cols, annihilated]
                + below[pair_cols, created]
                - (annihilated < created)
            )
            np.add.at(
                rdm1s,
                (created // n_orb, created % n_orb, annihilated % n_orb),
                pair_weight * (1.0 - 2.0 * (exponent % 2)),
            )
            spectators = np.nonzero(occupied[pair_rows] & occupied[pair_cols])[1]
            spectators = spectators.reshape(pair_cols.size, n_electrons - 1)
            for chunk in _spectator_chunks(n_electrons - 1, pair_cols.size):
                width = spectators[:, chunk].shape[1]
                spectator = spectators[:, chunk].ravel()
                _accumulate_two_body(
                    rdm2,
                    below,
                    n_orb,
                    np.repeat(pair_cols, width),
                    np.repeat(pair_weight, width),
                    np.stack([np.repeat(annihilated, width), spectator], axis=1),
                    np.stack([np.repeat(created, width), spectator], axis=1),
                )

        keep = rank == 2
        pair_rows, pair_cols = rows[keep], cols[keep]
        if pair_cols.size:
            _accumulate_two_body(
                rdm2,
                below,
                n_orb,
                pair_cols,
                weight[keep],
                np.nonzero(occupied[pair_cols] & ~occupied[pair_rows])[1].reshape(
                    -1, 2
                ),
                np.nonzero(occupied[pair_rows] & ~occupied[pair_cols])[1].reshape(
                    -1, 2
                ),
            )

    rdm1_alpha, rdm1_beta = rdm1s
    return rdm1_alpha + rdm1_beta, rdm2, rdm1_alpha, rdm1_beta


def _spectator_chunks(n_spectators: int, n_pairs: int) -> list[slice]:
    """Slices of the spectator axis that keep a pass at ``_RDM_TERMS_PER_PASS``
    determinant-pair terms, at least one spectator each."""
    width = max(1, _RDM_TERMS_PER_PASS // max(n_pairs, 1))
    return [
        slice(lo, min(lo + width, n_spectators)) for lo in range(0, n_spectators, width)
    ]


def _accumulate_two_body(
    rdm2: np.ndarray,
    below: np.ndarray,
    n_orb: int,
    cols: np.ndarray,
    weight: np.ndarray,
    annihilated: np.ndarray,
    created: np.ndarray,
) -> None:
    """Add every ``a+_p a+_r a_s a_q`` term of the right-hand determinants
    ``cols`` into ``rdm2``, with ``(q, s)`` the orderings of each row of
    ``annihilated`` and ``(p, r)`` those of ``created``, keeping those where
    ``p`` and ``q`` share a spin.

    Signs come from the occupation parity below each orbital, corrected for the
    orbitals already annihilated or created.
    """
    for first_q, first_s in ((0, 1), (1, 0)):
        for first_p, first_r in ((0, 1), (1, 0)):
            q, s = annihilated[:, first_q], annihilated[:, first_s]
            p, r = created[:, first_p], created[:, first_r]
            same_spin = p // n_orb == q // n_orb
            c, q, s, p, r = (
                cols[same_spin],
                q[same_spin],
                s[same_spin],
                p[same_spin],
                r[same_spin],
            )
            exponent = (
                below[c, q]
                + below[c, s]
                - (q < s)
                + below[c, r]
                - (q < r)
                - (s < r)
                + below[c, p]
                - (q < p)
                - (s < p)
                + (r < p)
            )
            np.add.at(
                rdm2,
                (p % n_orb, q % n_orb, r % n_orb, s % n_orb),
                weight[same_spin] * (1.0 - 2.0 * (exponent % 2)),
            )
