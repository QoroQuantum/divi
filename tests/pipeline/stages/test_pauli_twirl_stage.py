# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.stages._pauli_twirl_stage."""

import random

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager

from divi.pipeline.stages._pauli_twirl_stage import PauliTwirlPass


class TestPauliTwirlPass:
    """Spec: PauliTwirlPass preserves the ideal unitary up to global phase."""

    @pytest.mark.parametrize("gate_method", ["cx", "cz"])
    def test_twirl_preserves_unitary(self, gate_method):
        qc = QuantumCircuit(2)
        qc.h(0)
        getattr(qc, gate_method)(0, 1)
        qc.h(1)
        getattr(qc, gate_method)(1, 0)

        u_orig = Operator(qc)
        for seed in range(10):
            rng = random.Random(seed)
            twirled = PassManager([PauliTwirlPass(rng=rng)]).run(qc)
            assert Operator(twirled).equiv(
                u_orig
            ), f"Seed {seed}: twirled unitary not equivalent"

    def test_mixed_cx_cz_preserves_unitary(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cz(1, 2)
        qc.cx(2, 0)

        u_orig = Operator(qc)
        for seed in range(10):
            rng = random.Random(seed)
            twirled = PassManager([PauliTwirlPass(rng=rng)]).run(qc)
            assert Operator(twirled).equiv(u_orig)

    def test_non_twirl_gates_untouched(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rz(0.5, 1)
        qc.ry(0.3, 0)

        rng = random.Random(42)
        twirled = PassManager([PauliTwirlPass(rng=rng)]).run(qc)
        assert twirled.size() == qc.size()
        assert Operator(twirled).equiv(Operator(qc))

    def test_parametric_circuit_preserves_bound_unitary(self):
        theta = Parameter("theta")
        qc = QuantumCircuit(2)
        qc.rx(theta, 0)
        qc.cx(0, 1)
        qc.rz(theta, 1)

        rng = random.Random(0)
        twirled = PassManager([PauliTwirlPass(rng=rng)]).run(qc)
        assert {p.name for p in twirled.parameters} == {"theta"}

        for val in [0.0, 0.5, 1.2, -0.7]:
            binding = {theta: val}
            assert Operator(twirled.assign_parameters(binding)).equiv(
                Operator(qc.assign_parameters(binding))
            ), f"theta={val}: bound twirled unitary not equivalent"

    def test_distinct_seeds_produce_distinct_but_equivalent_twirls(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(0, 2)
        qc.cx(1, 0)

        u_orig = Operator(qc)
        variants = set()
        for seed in range(20):
            rng = random.Random(seed)
            twirled = PassManager([PauliTwirlPass(rng=rng)]).run(qc)
            variants.add(tuple(inst.operation.name for inst in twirled.data))
            assert Operator(twirled).equiv(u_orig)

        assert len(variants) > 1
