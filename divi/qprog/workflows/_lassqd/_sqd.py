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


def ci_string_to_int(half: str) -> int:
    """Integer form of one spin sector's occupation string, character ``p`` to bit
    ``p`` -- how PySCF's selected CI addresses determinants."""
    return int(half[::-1], 2)


def _aufbau_string(n_orb: int, n_electrons: int) -> str:
    """One spin sector of the reference determinant, as the ansatz prepares it at
    zero parameters: spatial orbitals ``0 .. n_electrons - 1`` filled."""
    return "1" * n_electrons + "0" * (n_orb - n_electrons)


def _occupation_matrix(strings: Sequence[str], n_orb: int) -> np.ndarray:
    """``(len(strings), n_orb)`` indicator of which orbitals each string fills."""
    return np.array(
        [[1.0 if bit == "1" else 0.0 for bit in half] for half in strings]
    ).reshape(len(strings), n_orb)


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
    is normalized: its typical component falls as ``1 / sqrt(m)``, so a fixed
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

    The subspace is always the full product of the two sector string lists, which
    is also the layout PySCF's selected-CI routines take.

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
        max_dim: int | tuple[int, int] | None = None,
        include_reference: bool = True,
        symmetrize_spin: bool = False,
        energy_tol: float = 0.0,
        occupancies_tol: float = 0.0,
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
            max_dim: Caps each spin sector, as one integer or an
                ``(alpha, beta)`` pair, so the subspace never exceeds their
                product.
            include_reference: Keep the aufbau reference determinant in every
                batch, so the projected energy cannot exceed the reference's.
            symmetrize_spin: Pool alpha and beta halves together for a
                spin-exchange invariant subspace. Ignored unless
                ``n_alpha == n_beta``.
            energy_tol: Stop iterating once the winning energy moves less than
                this between iterations and the occupancies have also settled.
                Zero (the default) never stops early.
            occupancies_tol: The occupancy half of that test, on the largest
                change in any orbital's average occupancy.
            rng: Subsampling generator; fresh default when omitted.

        Raises:
            ValueError: If ``n_batches``, ``batch_size`` or ``n_iterations`` is
                less than 1, if ``carryover_cutoff`` is not positive, if
                ``max_carryover`` is given without a cutoff or is less than 1,
                if any ``max_dim`` entry is less than 1, or if ``energy_tol`` or
                ``occupancies_tol`` is negative.
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
        max_dim_alpha, max_dim_beta = (
            max_dim if isinstance(max_dim, tuple) else (max_dim, max_dim)
        )
        for name, dim in (("alpha", max_dim_alpha), ("beta", max_dim_beta)):
            if dim is not None and dim < 1:
                raise ValueError(f"max_dim ({name}) must be >= 1, got {dim}.")
        for name, tol in (
            ("energy_tol", energy_tol),
            ("occupancies_tol", occupancies_tol),
        ):
            if tol < 0:
                raise ValueError(f"{name} must be non-negative, got {tol}.")

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
        self.max_dim_alpha = max_dim_alpha
        self.max_dim_beta = max_dim_beta
        self.include_reference = include_reference
        # Exchanging the sectors is only a symmetry when they hold equal counts.
        self.symmetrize_spin = symmetrize_spin and n_alpha == n_beta
        self.energy_tol = energy_tol
        self.occupancies_tol = occupancies_tol
        self._rng = np.random.default_rng() if rng is None else rng
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
        h_spin, g_spin = spin_orbital_integrals(
            one_body, two_body, self.n_orb, one_body_beta
        )
        target_s = 0.5 * abs(self.n_alpha - self.n_beta)

        best: SQDResult | None = None

        # Re-read each iteration, not accumulated: weights from differently
        # normalized eigenvectors are not comparable, and a carried string is
        # present in the current subspace anyway.
        carried_alpha: list[str] = []
        carried_beta: list[str] = []

        # Only reached when this solver has not produced any batch results yet.
        if np.sum(self.occupancy) == 0:
            self.occupancy[0] = self.n_alpha / self.n_orb
            self.occupancy[1] = self.n_beta / self.n_orb

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

            if self.carryover_cutoff is not None:
                alpha_weights, beta_weights = carryover_weights(
                    winner.strings_alpha,
                    winner.strings_beta,
                    winner.amplitudes,
                    self.carryover_cutoff,
                )
                carried_alpha = _heaviest_strings(alpha_weights, self.max_carryover)
                carried_beta = _heaviest_strings(beta_weights, self.max_carryover)

        if best is None:
            raise ValueError("No batch produced a candidate eigenvector.")
        return best

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
        weights: dict[str, float] = {}
        if iteration == 0 or not self.recovery:
            for bits in filter_symmetry(ordered, self.n_orb, self.n_alpha, self.n_beta):
                weights[bits] = weights.get(bits, 0.0) + probs[bits]
        else:
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

        batches = []
        for _ in range(self.n_batches):
            sampled = self._rng.choice(
                strings, size=size, replace=False, p=probabilities
            )
            alpha_counts: dict[str, int] = {}
            beta_counts: dict[str, int] = {}
            for bits in sampled:
                alpha = bits[: self.n_orb]
                beta = bits[self.n_orb :]
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
        dets = [
            bitstring_to_spatial_det(alpha + beta, self.n_orb)
            for alpha in strings_alpha
            for beta in strings_beta
        ]
        dets_spin = [
            spatial_to_spin_occupations(alpha, beta, self.n_orb) for alpha, beta in dets
        ]

        h_proj, s2_proj = projected_matrices(
            dets, dets_spin, h_spin, g_spin, self.n_orb
        )
        deviation = s2_proj - target_s * (target_s + 1.0) * np.eye(len(dets))
        eigenvals, eigenvecs = scipy.linalg.eigh(
            h_proj + self.lambda_penalty * (deviation @ deviation)
        )
        amplitudes = np.asarray(eigenvecs)[:, 0].reshape(
            len(strings_alpha), len(strings_beta)
        )

        # A string's orbitals are occupied in every determinant built on it, so
        # the sector marginals suffice.
        probability = amplitudes**2
        occupancy = np.stack(
            [
                probability.sum(axis=1) @ _occupation_matrix(strings_alpha, self.n_orb),
                probability.sum(axis=0) @ _occupation_matrix(strings_beta, self.n_orb),
            ]
        )
        result = SQDResult(
            energy=float(eigenvals[0] + constant),
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

    from pyscf.fci.selected_ci import _as_SCIvector, make_rdm1s, make_rdm2

    ci_strings = (
        np.array([ci_string_to_int(half) for half in strings_alpha], dtype=np.int64),
        np.array([ci_string_to_int(half) for half in strings_beta], dtype=np.int64),
    )
    nelec = (strings_alpha[0].count("1"), strings_beta[0].count("1"))
    civec = _as_SCIvector(np.ascontiguousarray(amplitudes, dtype=float), ci_strings)
    rdm1_alpha, rdm1_beta = make_rdm1s(civec, n_orb, nelec)
    rdm2 = make_rdm2(civec, n_orb, nelec)
    return rdm1_alpha + rdm1_beta, rdm2, rdm1_alpha, rdm1_beta


def _spatial_rdms_exact(
    strings_alpha: Sequence[str],
    strings_beta: Sequence[str],
    amplitudes: np.ndarray,
    n_orb: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct both RDMs from the second-quantized definitions directly.

    One pass over every pair of determinants, so orders of magnitude slower than
    :func:`compute_spatial_rdms`'s usual path.
    """
    subspace_dets = [
        bitstring_to_spatial_det(alpha + beta, n_orb)
        for alpha in strings_alpha
        for beta in strings_beta
    ]
    eigenvector = np.asarray(amplitudes, dtype=float).ravel()
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

    # Spin-trace the 2-RDM: sum the four same-spin-pair blocks (aa, ab, ba, bb).
    alpha, beta = slice(None, n_orb), slice(n_orb, None)
    rdm2 = (
        rdm2_spin[alpha, alpha, alpha, alpha]
        + rdm2_spin[alpha, alpha, beta, beta]
        + rdm2_spin[beta, beta, alpha, alpha]
        + rdm2_spin[beta, beta, beta, beta]
    )

    return rdm1, rdm2, rdm1_alpha, rdm1_beta
