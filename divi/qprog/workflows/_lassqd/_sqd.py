# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Sample-based quantum diagonalization post-processing.

Implements self-consistent configuration recovery (arXiv:2405.05068): symmetry
filtering, occupancy-guided bit-flip correction, batched determinant subspace
construction, spin-penalized projected diagonalization, and reduced
density-matrix reconstruction.
"""

import bisect
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.linalg

#: Determinant rows per pass in :func:`projected_matrices`. The pair arrays it
#: builds scale with this times the subspace size, so a fixed block keeps peak
#: memory independent of how large the subspace grows; only the returned
#: matrices scale with its square.
_PAIR_BLOCK_ROWS = 512


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

    Probabilities of keys that map onto the same SQD bitstring are summed,
    which cannot happen for well-formed input but keeps the result a valid
    distribution regardless.

    Raises:
        ValueError: If any key in ``probs`` is not ``2 * n_orb`` characters
            wide, propagated from :func:`deinterleave_spin_bitstring`.
    """
    converted: dict[str, float] = {}
    for bitstring, prob in probs.items():
        key = deinterleave_spin_bitstring(bitstring, n_orb)
        converted[key] = converted.get(key, 0.0) + prob
    return converted


def spin_orbital_integrals(
    one_body: np.ndarray,
    two_body: np.ndarray,
    n_orb: int,
    one_body_beta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert spatial one- and two-body integrals to spin-orbital integrals.

    Spin-orbitals are blocked: index ``p`` is alpha for ``p < n_orb`` and beta
    above. ``one_body_beta`` gives the beta channel a distinct one-body
    potential, as a spin-polarized mean-field embedding produces.
    """
    if one_body_beta is None:
        one_body_beta = one_body
    n_spin_orb = 2 * n_orb
    h_spin = np.zeros((n_spin_orb, n_spin_orb))
    h_spin[:n_orb, :n_orb] = one_body
    h_spin[n_orb:, n_orb:] = one_body_beta

    g_spin = np.zeros((n_spin_orb, n_spin_orb, n_spin_orb, n_spin_orb))
    for p in range(n_spin_orb):
        for q in range(n_spin_orb):
            for r in range(n_spin_orb):
                for s in range(n_spin_orb):
                    p_sp, p_spin = p % n_orb, p // n_orb
                    q_sp, q_spin = q % n_orb, q // n_orb
                    r_sp, r_spin = r % n_orb, r // n_orb
                    s_sp, s_spin = s % n_orb, s // n_orb

                    term1 = (
                        two_body[p_sp, r_sp, q_sp, s_sp]
                        if (p_spin == r_spin and q_spin == s_spin)
                        else 0.0
                    )
                    term2 = (
                        two_body[p_sp, s_sp, q_sp, r_sp]
                        if (p_spin == s_spin and q_spin == r_spin)
                        else 0.0
                    )
                    g_spin[p, q, r, s] = term1 - term2

    return h_spin, g_spin


def _annihilation_sign(occ, p):
    """Return the sign and remaining occupation from annihilating spin-orbital p."""
    occ_tup = tuple(occ)
    if p not in occ_tup:
        return 0, None
    idx = occ_tup.index(p)
    sign = (-1) ** idx
    new_occ = occ_tup[:idx] + occ_tup[idx + 1 :]
    return sign, new_occ


def _creation_sign(occ, p):
    """Return the sign and updated occupation from creating spin-orbital p."""
    occ_tup = tuple(occ)
    if p in occ_tup:
        return 0, None
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

    elif len(diff_i) == 1:  # Differ by 1
        p = diff_i[0]
        q = diff_j[0]
        # We compute <det_i | H | det_j> where det_i = det_j - {q} + {p}
        # Annihilate q in det_j, create p in det_j
        sign_ann, occ_mid = _annihilation_sign(det_j, q)
        if sign_ann == 0:
            return 0.0
        sign_cre, _ = _creation_sign(occ_mid, p)
        sign = sign_ann * sign_cre

        val = h_spin[p, q]
        for r in det_j:
            if r != q:
                val += g_spin[p, r, q, r]
        return sign * val

    elif len(diff_i) == 2:  # Differ by 2
        p, r = diff_i
        q, s = diff_j
        # We compute <det_i | H | det_j> where det_i = det_j - {q, s} + {p, r}
        # Annihilate q in det_j, then s in remaining, then create r, then p
        sign_q, occ_1 = _annihilation_sign(det_j, q)
        if sign_q == 0:
            return 0.0
        sign_s, occ_2 = _annihilation_sign(occ_1, s)
        if sign_s == 0:
            return 0.0
        sign_r, occ_3 = _creation_sign(occ_2, r)
        if sign_r == 0:
            return 0.0
        sign_p, _ = _creation_sign(occ_3, p)
        sign = sign_q * sign_s * sign_r * sign_p

        val = g_spin[p, r, q, s]
        return sign * val

    return 0.0


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


def _apply_s_plus(det, n_orb):
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


def _apply_s_minus(det, n_orb):
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


def s2_matrix_element(det_i, det_j, n_orb: int) -> float:
    """Compute <det_i | S^2 | det_j> via S^2 = S_z(S_z + 1) + S_- S_+.

    ``det_i`` and ``det_j`` are ``(alpha_occ, beta_occ)`` spatial-orbital pairs.
    """
    alpha_i, beta_i = det_i
    alpha_j, beta_j = det_j

    sz_j = 0.5 * (len(alpha_j) - len(beta_j))
    sz_i = 0.5 * (len(alpha_i) - len(beta_i))

    if sz_i != sz_j:
        return 0.0

    diag = 0.0
    if det_i == det_j:
        diag = sz_j * (sz_j + 1.0)

    s_plus_results = _apply_s_plus(det_j, n_orb)
    coeff_ij = 0.0
    for sign_p, det_p in s_plus_results:
        s_minus_results = _apply_s_minus(det_p, n_orb)
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
) -> tuple[np.ndarray, np.ndarray]:
    """Project the Hamiltonian and ``S^2`` onto the span of ``dets``.

    Returns what filling every entry with :func:`slater_condon` and
    :func:`s2_matrix_element` would -- those remain the reference definitions --
    computed by array operations instead of a double loop. Excitation ranks come
    from one matrix product, then each rank's elements are gathered in one pass.

    The saving is that Slater-Condon vanishes past a double excitation, and the
    fraction of pairs that survive falls as the subspace grows: 76% at 50
    determinants but 7% at 4900, where the loop form costs 68 s.

    ``S^2``'s off-diagonal is handled differently. It is non-zero only between
    determinants related by exchanging spins across two spatial orbitals, which
    is well under 1% of pairs, so those are located by array operations and then
    evaluated by the scalar routine -- keeping its ladder-operator signs as the
    single source of truth for a negligible cost.

    Rows are processed in blocks so the pair arrays stay bounded rather than
    scaling with the square of the subspace; only the two returned matrices do.

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
    h_proj = np.zeros((m, m))
    s2_proj = np.zeros((m, m))
    if m == 0:
        return h_proj, s2_proj

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
        h_proj[rows, cols] = diagonal[rows]
        s2_proj[rows, cols] = s2_diagonal[rows]

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
            h_proj[rows, cols] = sign * (h_spin[created, annihilated] + summed)

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
            h_proj[rows, cols] = (
                sign
                * g_spin[
                    lower_created, upper_created, lower_annihilated, upper_annihilated
                ]
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
            for row, col in zip(rows[spin_exchange], cols[spin_exchange]):
                s2_proj[row, col] = s2_matrix_element(dets[row], dets[col], n_orb)

    return h_proj, s2_proj


#: Places the carryover ranking rounds to. Threaded BLAS and set iteration order
#: perturb eigenvector components at the last bit, enough to swap near-equal
#: weights and, once a cap binds, change which determinants survive.
_CARRYOVER_WEIGHT_PLACES = 12


def _heaviest_strings(weights: dict[str, float], limit: int | None) -> list[str]:
    """The ``limit`` heaviest strings, near-ties broken on the string so float
    noise cannot decide what is kept."""
    ordered = sorted(
        weights,
        key=lambda string: (
            -round(weights[string], _CARRYOVER_WEIGHT_PLACES),
            string,
        ),
    )
    return ordered if limit is None else ordered[:limit]


def carryover_weights(
    subspace: Sequence[str],
    eigenvector: np.ndarray,
    n_orb: int,
    cutoff: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Weigh the alpha and beta strings worth carrying to the next iteration.

    Splits the determinants carrying real weight into their alpha and beta
    halves, weighting each half by the probability it holds across every
    determinant containing it.

    The threshold is relative to the largest coefficient because the eigenvector
    is normalized: its typical component falls as ``1 / sqrt(m)``, so a fixed
    threshold would prune almost nothing at large subspace sizes.

    Halves are returned separately because the subspace is rebuilt as their
    product, so ``k`` of each reintroduce up to ``k ** 2`` determinants,
    including combinations never sampled together. A half spread thinly over
    many determinants can therefore outrank a concentrated one.

    Args:
        subspace: Blocked bitstrings spanning the diagonalized subspace.
        eigenvector: Coefficients over ``subspace``, in the same order.
        n_orb: Spatial orbitals in the fragment.
        cutoff: Retain determinants whose ``abs(coefficient)`` exceeds this
            fraction of the largest coefficient in ``eigenvector``.

    Returns:
        ``(alpha_weights, beta_weights)``, each mapping string to weight.
    """
    coefficients = np.asarray(eigenvector, dtype=float)
    if coefficients.size == 0:
        return {}, {}
    largest = float(np.max(np.abs(coefficients)))
    if largest == 0.0:
        return {}, {}
    threshold = cutoff * largest

    alpha_weights: dict[str, float] = {}
    beta_weights: dict[str, float] = {}
    for bitstring, coefficient in zip(subspace, coefficients):
        if abs(coefficient) <= threshold:
            continue
        weight = float(coefficient) ** 2
        alpha = bitstring[:n_orb]
        beta = bitstring[n_orb:]
        alpha_weights[alpha] = alpha_weights.get(alpha, 0.0) + weight
        beta_weights[beta] = beta_weights.get(beta, 0.0) + weight

    return alpha_weights, beta_weights


def filter_symmetry(bitstrings, n_orb: int, n_alpha: int, n_beta: int) -> list[str]:
    """Keep only blocked bitstrings with the target alpha and beta counts."""
    kept = []
    for bits in bitstrings:
        if bits[:n_orb].count("1") == n_alpha and bits[n_orb:].count("1") == n_beta:
            kept.append(bits)
    return kept


def _modified_relu(distance: float, threshold: float, delta: float = 0.01) -> float:
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
    if weight_sum > 1e-9:
        probabilities = [w / weight_sum for w in weights]
    else:
        probabilities = [1.0 / len(indices)] * len(indices)

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

    Attributes:
        energy: Lowest eigenvalue of the *spin-penalized* projected Hamiltonian,
            ``H + lambda (S^2 - s(s+1))^2``, plus ``constant`` -- not the bare
            expectation value ``<H>``. Batches are ranked on this deliberately,
            so a batch with a lower bare energy in the wrong spin sector loses;
            the cost is that on a spin-incomplete subspace this sits above
            ``<H>`` by the penalty term. Do not treat it as a variational bound.
            LASSQD's reported energy does not come from here: it is recomputed
            by :func:`~divi.qprog.workflows._lassqd._integrals.total_energy` from
            the reassembled RDMs, which carry no penalty.
        eigenvector: Ground-state coefficients over ``subspace``.
        subspace: Blocked bitstrings spanning the winning batch's determinants,
            in the same order as ``eigenvector``.
    """

    energy: float
    eigenvector: np.ndarray
    subspace: list[str]


class SQDSolver:
    """Self-consistent configuration recovery over sampled determinants.

    Implements arXiv:2405.05068. ``occupancy`` holds the running per-spin
    orbital occupancy estimate consumed by the self-consistent recovery step;
    it is refreshed at the end of every iteration from that iteration's batch
    results, and is seeded from the target electron counts when a solver has
    not yet run.
    """

    def __init__(
        self,
        n_orb: int,
        n_alpha: int,
        n_beta: int,
        *,
        n_batches: int = 15,
        batch_size: int = 170,
        n_iterations: int = 6,
        lambda_penalty: float = 0.2,
        recovery: bool = True,
        carryover_cutoff: float | None = None,
        max_carryover: int | None = None,
        rng: np.random.Generator | None = None,
    ):
        """Initialize the solver.

        Args:
            n_orb: Spatial orbitals in the fragment.
            n_alpha: Alpha electrons in the target sector.
            n_beta: Beta electrons in the target sector.
            n_batches: Subspaces per iteration; the lowest-energy one wins.
            batch_size: Configurations sampled per batch. Alpha and beta halves
                are pooled separately, so the subspace holds up to
                ``batch_size ** 2`` determinants. The accuracy knob: a
                one-determinant subspace is the mean field.
            n_iterations: Configuration-recovery iterations.
            lambda_penalty: Weight of the ``S^2`` penalty.
            recovery: Whether to run configuration recovery.
            carryover_cutoff: Enables carryover when given, as a fraction of the
                winning batch's largest eigenvector coefficient. Determinants
                above it are retained and later batches extended with their
                alpha and beta halves; ``None`` (default) is conventional SQD.
                Re-decided each iteration, and selected from the *penalized*
                ground state, so ``lambda_penalty`` influences what is kept.
            max_carryover: Keeps at most this many alpha and beta strings, the
                heaviest, so retention can shrink between iterations. ``None``
                (default) leaves the subspace bounded only by the fragment's
                determinant space; worth setting on a wide fragment.
            rng: Subsampling generator; fresh default when omitted.

        Raises:
            ValueError: If ``n_batches``, ``batch_size`` or ``n_iterations`` is
                less than 1, if ``carryover_cutoff`` is not positive, or if
                ``max_carryover`` is given without a cutoff or is less than 1.
        """
        if n_batches < 1:
            raise ValueError(f"n_batches must be >= 1, got {n_batches}.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        if n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {n_iterations}.")
        if carryover_cutoff is not None and not carryover_cutoff > 0:
            raise ValueError(
                f"carryover_cutoff must be positive, got {carryover_cutoff}."
            )
        if max_carryover is not None:
            if carryover_cutoff is None:
                raise ValueError(
                    "max_carryover caps what carryover retains, so it needs "
                    "carryover_cutoff to be set."
                )
            if max_carryover < 1:
                raise ValueError(f"max_carryover must be >= 1, got {max_carryover}.")

        self.n_orb = n_orb
        self.n_alpha = n_alpha
        self.n_beta = n_beta
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.lambda_penalty = lambda_penalty
        self.recovery = recovery
        self.carryover_cutoff = carryover_cutoff
        self.max_carryover = max_carryover
        self._rng = np.random.default_rng() if rng is None else rng
        self.occupancy = np.zeros((2, n_orb))

    def solve(
        self,
        probs: dict[str, float],
        one_body: np.ndarray,
        two_body: np.ndarray,
        constant: float = 0.0,
        one_body_beta: np.ndarray | None = None,
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

        Returns:
            The lowest-energy :class:`SQDResult` found across all iterations.

        Raises:
            ValueError: If, in some iteration, no sampled bitstring can be
                brought into agreement with the target particle symmetry, or
                if no batch ever produces a candidate eigenvector.
        """
        # Sorted, not insertion-ordered: the first iteration samples straight
        # from this list, so a dict built in a different order would draw a
        # different subspace from the same distribution and the same seed.
        unique_bs = sorted(probs)

        h_spin, g_spin = spin_orbital_integrals(
            one_body, two_body, self.n_orb, one_body_beta
        )
        target_s = 0.5 * abs(self.n_alpha - self.n_beta)

        best_energy = float("inf")
        best_state = None
        best_subspace = None

        # Re-read each iteration, not accumulated: weights from differently
        # normalized eigenvectors are not comparable, and a carried string is
        # present in the current subspace anyway.
        carried_alpha: list[str] = []
        carried_beta: list[str] = []

        # Only reached when this solver has not produced any batch results yet.
        if np.sum(self.occupancy) == 0:
            self.occupancy[0] = self.n_alpha / self.n_orb
            self.occupancy[1] = self.n_beta / self.n_orb

        for it in range(self.n_iterations):
            if it == 0 or not self.recovery:
                valid_bs = filter_symmetry(
                    unique_bs, self.n_orb, self.n_alpha, self.n_beta
                )
            else:
                valid_bs = [
                    bit_flip_correction(
                        bs,
                        self.n_orb,
                        self.n_alpha,
                        self.n_beta,
                        self.occupancy,
                        self._rng,
                    )
                    for bs in unique_bs
                ]
                valid_bs = sorted(set(valid_bs))

            if not valid_bs:
                raise ValueError(
                    "No valid configurations found matching particle symmetry!"
                )

            bs_probs = []
            for bs in valid_bs:
                if bs in probs:
                    bs_probs.append(probs[bs])
                else:
                    bs_probs.append(1.0 / len(valid_bs))

            prob_sum = sum(bs_probs)
            if prob_sum > 0:
                bs_probs = [p / prob_sum for p in bs_probs]
            else:
                bs_probs = [1.0 / len(valid_bs)] * len(valid_bs)

            batches = []
            for _ in range(self.n_batches):
                sampled = self._rng.choice(
                    valid_bs, size=self.batch_size, replace=True, p=bs_probs
                )

                alpha_pool = set()
                beta_pool = set()
                for bs in sampled:
                    alpha_pool.add(bs[: self.n_orb])
                    beta_pool.add(bs[self.n_orb :])

                # Carried strings join this batch's own pools, so they also pair
                # with freshly sampled halves rather than only with each other.
                alpha_pool.update(carried_alpha)
                beta_pool.update(carried_beta)

                unique_alpha = [s for s in alpha_pool if s.count("1") == self.n_alpha]
                unique_beta = [s for s in beta_pool if s.count("1") == self.n_beta]

                s_k = set()
                for a in unique_alpha:
                    for b in unique_beta:
                        s_k.add(a + b)

                if not s_k:
                    fallback_samples = self._rng.choice(
                        valid_bs,
                        size=min(self.batch_size, len(valid_bs)),
                        replace=False,
                        p=bs_probs,
                    )
                    s_k.update(fallback_samples)

                batches.append(list(s_k))

            batch_energies = []
            batch_states = []
            batch_subspaces = []
            batch_occupancies = []

            for s_k in batches:
                m_dim = len(s_k)

                dets = [bitstring_to_spatial_det(bs, self.n_orb) for bs in s_k]
                dets_spin = [
                    spatial_to_spin_occupations(d[0], d[1], self.n_orb) for d in dets
                ]

                h_proj, s2_proj = projected_matrices(
                    dets, dets_spin, h_spin, g_spin, self.n_orb
                )

                penalty_matrix = s2_proj - target_s * (target_s + 1.0) * np.eye(m_dim)
                penalty_matrix = self.lambda_penalty * np.dot(
                    penalty_matrix, penalty_matrix
                )

                eigenvals, eigenvecs = scipy.linalg.eigh(h_proj + penalty_matrix)

                e_k = eigenvals[0] + constant
                v_k = np.asarray(eigenvecs)[:, 0]

                batch_energies.append(e_k)
                batch_states.append(v_k)
                batch_subspaces.append(s_k)

                occ_k = np.zeros((2, self.n_orb))
                for i, c in enumerate(v_k):
                    coeff_sq = c**2
                    alpha_occ, beta_occ = dets[i]
                    for p in alpha_occ:
                        occ_k[0, p] += coeff_sq
                    for p in beta_occ:
                        occ_k[1, p] += coeff_sq
                batch_occupancies.append(occ_k)

            best_batch_idx = int(np.argmin(batch_energies))
            energy_iter = batch_energies[best_batch_idx]

            if energy_iter < best_energy:
                best_energy = energy_iter
                best_state = batch_states[best_batch_idx]
                best_subspace = batch_subspaces[best_batch_idx]

            if self.carryover_cutoff is not None:
                alpha_weights, beta_weights = carryover_weights(
                    [str(bs) for bs in batch_subspaces[best_batch_idx]],
                    batch_states[best_batch_idx],
                    self.n_orb,
                    self.carryover_cutoff,
                )
                carried_alpha = _heaviest_strings(alpha_weights, self.max_carryover)
                carried_beta = _heaviest_strings(beta_weights, self.max_carryover)

            self.occupancy = np.mean(batch_occupancies, axis=0)

        if best_state is None or best_subspace is None:
            raise ValueError("No batch produced a candidate eigenvector.")

        return SQDResult(
            energy=float(best_energy),
            eigenvector=best_state,
            subspace=[str(bs) for bs in best_subspace],
        )


def compute_spatial_rdms(
    subspace_dets: list[tuple[tuple[int, ...], tuple[int, ...]]],
    eigenvector: np.ndarray,
    n_orb: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the spatial 1- and 2-RDM from an SQD eigenvector.

    Args:
        subspace_dets: ``(alpha_occ, beta_occ)`` spatial-orbital pairs for each
            subspace determinant, in the same order as ``eigenvector``.
        eigenvector: SQD ground-state coefficients over ``subspace_dets``.
        n_orb: Number of spatial orbitals.

    Returns:
        ``(rdm1, rdm2, rdm1_alpha, rdm1_beta)`` spatial reduced density matrices
        in PySCF's active-space convention, where ``rdm1`` is the spin trace
        ``rdm1_alpha + rdm1_beta``.
    """
    m_dim = len(eigenvector)

    dets_spin = [spatial_to_spin_occupations(d[0], d[1], n_orb) for d in subspace_dets]

    n_spin = 2 * n_orb
    rdm1_spin = np.zeros((n_spin, n_spin))
    rdm2_spin = np.zeros((n_spin, n_spin, n_spin, n_spin))
    for i in range(m_dim):
        det_i = dets_spin[i]
        for j in range(m_dim):
            det_j = dets_spin[j]
            val_ij = eigenvector[i] * eigenvector[j]

            for q in det_j:
                sign_q, occ_1 = _annihilation_sign(det_j, q)
                if sign_q == 0 or occ_1 is None:
                    continue

                diff = set(det_i) - set(occ_1)
                if len(diff) == 1:
                    p = next(iter(diff))
                    sign_p, occ_final = _creation_sign(occ_1, p)
                    if sign_p != 0 and occ_final == det_i:
                        rdm1_spin[p, q] += val_ij * sign_p * sign_q

                for s in occ_1:
                    sign_s, occ_2 = _annihilation_sign(occ_1, s)
                    if sign_s == 0 or occ_2 is None:
                        continue
                    diff = set(det_i) - set(occ_2)
                    if len(diff) == 2:
                        p_cand, r_cand = list(diff)
                        for p, r in [(p_cand, r_cand), (r_cand, p_cand)]:
                            sign_r, occ_3 = _creation_sign(occ_2, r)
                            if sign_r == 0:
                                continue
                            sign_p, occ_final = _creation_sign(occ_3, p)
                            if sign_p != 0 and occ_final == det_i:
                                rdm2_spin[p, q, r, s] += (
                                    val_ij * sign_q * sign_s * sign_r * sign_p
                                )

    rdm1_alpha = rdm1_spin[:n_orb, :n_orb].copy()
    rdm1_beta = rdm1_spin[n_orb:, n_orb:].copy()
    rdm1 = rdm1_alpha + rdm1_beta

    rdm2 = np.zeros((n_orb, n_orb, n_orb, n_orb))
    for p in range(n_orb):
        for q in range(n_orb):
            for r in range(n_orb):
                for s in range(n_orb):
                    val = 0.0
                    for s1 in (0, n_orb):
                        for s2 in (0, n_orb):
                            val += rdm2_spin[p + s1, q + s1, r + s2, s + s2]
                    rdm2[p, q, r, s] = val

    return rdm1, rdm2, rdm1_alpha, rdm1_beta
