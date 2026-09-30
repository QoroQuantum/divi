# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the QAOAProblem base protocol."""

import pytest
from qiskit.quantum_info import SparsePauliOp

from divi.qprog.problems import QAOAProblem


class MinimalProblem(QAOAProblem):
    @property
    def cost_hamiltonian(self) -> SparsePauliOp:
        return SparsePauliOp.from_list([("ZI", 1.0), ("IZ", -1.0)])

    @property
    def loss_constant(self) -> float:
        return 0.0

    @property
    def decode_fn(self):
        return lambda bitstring: bitstring


class _EvenParityProblem(MinimalProblem):
    """Feasible iff the bitstring has an even number of ones; energy = ones."""

    def is_feasible(self, bitstring: str) -> bool:
        return bitstring.count("1") % 2 == 0

    def compute_energy(self, bitstring: str) -> float:
        return float(bitstring.count("1"))

    def repair_infeasible_bitstring(self, bitstring: str):
        # "111" cannot be repaired; everything else drops its last one.
        if bitstring == "111":
            return bitstring, None, None
        i = bitstring.rindex("1")
        repaired = bitstring[:i] + "0" + bitstring[i + 1 :]
        return repaired, repaired, self.compute_energy(repaired)


class _AuxiliaryBitProblem(_EvenParityProblem):
    """Like ``_EvenParityProblem``, with a trailing auxiliary bit that is ignored."""

    def is_feasible(self, bitstring: str) -> bool:
        return super().is_feasible(bitstring[:-1])

    def _solution_key(self, bitstring: str) -> str:
        return bitstring[:-1]


SAMPLES = [("110", 0.4), ("100", 0.3), ("111", 0.2), ("000", 0.1)]


def test_default_mixer_hamiltonian_is_x_mixer():
    mixer = MinimalProblem().mixer_hamiltonian

    assert mixer.num_qubits == 2
    assert set(mixer.paulis.to_labels()) == {"IX", "XI"}


def test_rank_feasible_filter_drops_infeasible_and_ranks_by_energy():
    result = _EvenParityProblem()._rank_feasible(SAMPLES, "filter", None)
    assert [(s.bitstring, s.prob) for s in result] == [("000", 0.1), ("110", 0.4)]


def test_rank_feasible_repair_merges_duplicates_and_drops_failed_repairs():
    result = _EvenParityProblem()._rank_feasible(SAMPLES, "repair", None)
    # "100" repairs onto "000"; "111" stays infeasible and is dropped.
    assert [s.bitstring for s in result] == ["000", "110"]
    assert result[0].prob == pytest.approx(0.4)
    assert [s.energy for s in result] == [0.0, 2.0]


def test_rank_feasible_merges_samples_differing_only_in_auxiliary_bits():
    samples = [("1100", 0.25), ("1101", 0.5)]
    result = _AuxiliaryBitProblem()._rank_feasible(samples, "filter", None)
    assert [(s.bitstring, s.prob) for s in result] == [("1100", 0.75)]


def test_rank_feasible_decodes_only_when_requested():
    problem = _EvenParityProblem()
    without = problem._rank_feasible(SAMPLES, "repair", None)
    with_decode = problem._rank_feasible(SAMPLES, "repair", str.upper)
    assert all(s.decoded is None for s in without)
    assert {s.bitstring: s.decoded for s in with_decode} == {"000": "000", "110": "110"}
