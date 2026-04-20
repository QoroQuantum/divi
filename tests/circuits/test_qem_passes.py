# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for GlobalFoldPass and PauliTwirlPass."""

import random

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager

from divi.circuits._qem_passes import GlobalFoldPass, PauliTwirlPass

# ---------------------------------------------------------------------------
# GlobalFoldPass
# ---------------------------------------------------------------------------


class TestGlobalFoldPass:
    """Spec: GlobalFoldPass returns U · (U† · U)^k for scale = 1 + 2k."""

    @pytest.fixture
    def two_qubit_qc(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.5, 1)
        return qc

    def test_scale_1_is_identity(self, two_qubit_qc):
        out = PassManager([GlobalFoldPass(1.0)]).run(two_qubit_qc)
        assert Operator(out).equiv(Operator(two_qubit_qc))
        assert out.size() == two_qubit_qc.size()

    def test_scale_3_triples_gate_count(self, two_qubit_qc):
        folded = PassManager([GlobalFoldPass(3.0)]).run(two_qubit_qc)
        assert folded.size() == 3 * two_qubit_qc.size()

    @pytest.mark.parametrize("scale", [1.0, 3.0, 5.0, 7.0])
    def test_folded_unitary_equals_original(self, scale):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rz(0.7, 0)
        qc.cx(0, 1)
        qc.ry(-0.3, 1)
        folded = PassManager([GlobalFoldPass(scale)]).run(qc)
        assert Operator(folded).equiv(Operator(qc))

    def test_parametric_circuit_preserves_unitary(self):
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2)
        qc.rx(theta, 0)
        qc.cx(0, 1)
        qc.rz(2 * phi, 1)

        folded = PassManager([GlobalFoldPass(3.0)]).run(qc)

        assert set(p.name for p in folded.parameters) == {"theta", "phi"}
        binding = {theta: 0.4, phi: 0.2}
        assert Operator(folded.assign_parameters(binding)).equiv(
            Operator(qc.assign_parameters(binding))
        )

    @pytest.mark.parametrize(
        "scale,match",
        [(2.0, "odd integer"), (1.5, "odd integer"), (0.5, ">= 1")],
        ids=["even", "fractional", "below_one"],
    )
    def test_rejects_invalid_scale(self, scale, match):
        with pytest.raises(ValueError, match=match):
            GlobalFoldPass(scale)


# ---------------------------------------------------------------------------
# PauliTwirlPass
# ---------------------------------------------------------------------------


class TestPauliTwirlPass:
    """Spec: PauliTwirlPass preserves the ideal unitary (up to global phase)."""

    @pytest.mark.parametrize(
        "gate_method",
        ["cx", "cz"],
    )
    def test_twirl_preserves_unitary(self, gate_method):
        """Twirling CX or CZ gates preserves the overall unitary."""
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
        """Circuit mixing CX and CZ gates is correctly twirled."""
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
        """Single-qubit gates pass through unmodified."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rz(0.5, 1)
        qc.ry(0.3, 0)

        rng = random.Random(42)
        twirled = PassManager([PauliTwirlPass(rng=rng)]).run(qc)
        assert twirled.size() == qc.size()
        assert Operator(twirled).equiv(Operator(qc))

    def test_parametric_circuit_preserves_bound_unitary(self):
        """Twirled parametric circuit produces correct unitary after binding."""
        theta = Parameter("theta")
        qc = QuantumCircuit(2)
        qc.rx(theta, 0)
        qc.cx(0, 1)
        qc.rz(theta, 1)

        rng = random.Random(0)
        twirled = PassManager([PauliTwirlPass(rng=rng)]).run(qc)
        assert set(p.name for p in twirled.parameters) == {"theta"}

        for val in [0.0, 0.5, 1.2, -0.7]:
            binding = {theta: val}
            assert Operator(twirled.assign_parameters(binding)).equiv(
                Operator(qc.assign_parameters(binding))
            ), f"theta={val}: bound twirled unitary not equivalent"

    def test_distinct_seeds_produce_distinct_but_equivalent_twirls(self):
        """Different seeds produce structurally different but unitarily equivalent circuits."""
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
