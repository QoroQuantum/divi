# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for GlobalFoldPass, LocalFoldPass, and PauliTwirlPass."""

import random

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager

from divi.circuits._qem_passes import GlobalFoldPass, LocalFoldPass, PauliTwirlPass

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

    def test_rejects_below_one_scale(self):
        with pytest.raises(ValueError, match=">= 1"):
            GlobalFoldPass(0.5)

    @pytest.mark.parametrize(
        "scale,d,expected_size",
        [
            (1.5, 4, 6),  # k=0, n=1 → 4 + 2*1
            (2.0, 4, 8),  # k=0, n=2 → 4 + 2*2
            (2.5, 4, 10),  # k=0, n=3 → 4 + 2*3
            (3.5, 4, 14),  # k=1, n=1 → 4*3 + 2*1
            (4.0, 4, 16),  # k=1, n=2 → 4*3 + 2*2
        ],
    )
    def test_fractional_scale_gate_count(self, scale, d, expected_size):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.5, 1)
        qc.ry(-0.3, 0)
        assert qc.size() == d
        folded = PassManager([GlobalFoldPass(scale)]).run(qc)
        assert folded.size() == expected_size

    @pytest.mark.parametrize("scale", [1.25, 1.5, 2.0, 2.5, 3.5, 4.0])
    def test_fractional_scale_preserves_unitary(self, scale):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rx(0.4, 0)
        qc.cx(0, 1)
        qc.ry(-0.2, 1)
        folded = PassManager([GlobalFoldPass(scale)]).run(qc)
        assert Operator(folded).equiv(Operator(qc))

    def test_fractional_scale_folds_the_tail(self):
        """scale=1.5 on a 4-gate linear chain folds only the last gate."""
        qc = QuantumCircuit(1)
        qc.h(0)  # g1
        qc.rz(0.5, 0)  # g2
        qc.rx(0.4, 0)  # g3
        qc.ry(0.3, 0)  # g4 — tail, unambiguous in a linear chain
        folded = PassManager([GlobalFoldPass(1.5)]).run(qc)
        names = [inst.operation.name for inst in folded.data]
        # Original 4 gates, then L† = [ry†], then L = [ry].
        assert names == ["h", "rz", "rx", "ry", "ry", "ry"]
        assert Operator(folded).equiv(Operator(qc))

    def test_fractional_parametric_circuit_preserves_unitary(self):
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2)
        qc.rx(theta, 0)
        qc.cx(0, 1)
        qc.rz(2 * phi, 1)

        folded = PassManager([GlobalFoldPass(2.5)]).run(qc)

        assert set(p.name for p in folded.parameters) == {"theta", "phi"}
        binding = {theta: 0.4, phi: 0.2}
        assert Operator(folded.assign_parameters(binding)).equiv(
            Operator(qc.assign_parameters(binding))
        )


# ---------------------------------------------------------------------------
# LocalFoldPass
# ---------------------------------------------------------------------------


class TestLocalFoldPass:
    """Spec: LocalFoldPass replaces each gate with G·(G†·G)^k, with partial
    folding for fractional scale factors."""

    @pytest.fixture
    def four_gate_qc(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.5, 1)
        qc.ry(-0.3, 0)
        return qc

    def test_scale_1_is_identity(self, four_gate_qc):
        out = PassManager([LocalFoldPass(1.0)]).run(four_gate_qc)
        assert out.size() == four_gate_qc.size()
        assert Operator(out).equiv(Operator(four_gate_qc))

    @pytest.mark.parametrize("scale,expected_mult", [(3.0, 3), (5.0, 5), (7.0, 7)])
    def test_odd_integer_scale_multiplies_gate_count(
        self, four_gate_qc, scale, expected_mult
    ):
        folded = PassManager([LocalFoldPass(scale)]).run(four_gate_qc)
        assert folded.size() == expected_mult * four_gate_qc.size()

    @pytest.mark.parametrize("scale", [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0])
    def test_folded_unitary_equals_original(self, four_gate_qc, scale):
        rng = random.Random(0)
        folded = PassManager([LocalFoldPass(scale, rng=rng)]).run(four_gate_qc)
        assert Operator(folded).equiv(Operator(four_gate_qc))

    @pytest.mark.parametrize(
        "scale,d,expected_size",
        [
            (1.5, 4, 6),  # k=0, n=1 → 4 + 2*1
            (2.0, 4, 8),  # k=0, n=2 → 4 + 2*2
            (2.5, 4, 10),  # k=0, n=3 → 4 + 2*3
            (3.5, 4, 14),  # k=1, n=1 → 4*3 + 2*1
        ],
    )
    def test_fractional_scale_gate_count(self, scale, d, expected_size):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.5, 1)
        qc.ry(-0.3, 0)
        assert qc.size() == d
        folded = PassManager([LocalFoldPass(scale, rng=random.Random(0))]).run(qc)
        assert folded.size() == expected_size

    def test_parametric_circuit_preserves_unitary(self):
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2)
        qc.rx(theta, 0)
        qc.cx(0, 1)
        qc.rz(2 * phi, 1)

        folded = PassManager([LocalFoldPass(2.0, rng=random.Random(0))]).run(qc)

        assert set(p.name for p in folded.parameters) == {"theta", "phi"}
        binding = {theta: 0.4, phi: 0.2}
        assert Operator(folded.assign_parameters(binding)).equiv(
            Operator(qc.assign_parameters(binding))
        )

    def test_non_unitary_ops_are_skipped(self):
        """Barriers/measurements/resets must not be folded."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.barrier()
        qc.measure([0, 1], [0, 1])

        folded = PassManager([LocalFoldPass(3.0)]).run(qc)
        op_counts = folded.count_ops()
        assert op_counts.get("barrier", 0) == 1
        assert op_counts.get("measure", 0) == 2
        # Unitary gates (h, cx) tripled; non-unitary untouched.
        assert op_counts["h"] == 3
        assert op_counts["cx"] == 3

    def test_rng_reproducibility(self, four_gate_qc):
        """Same seed → identical folded structure."""
        a = PassManager([LocalFoldPass(1.5, rng=random.Random(7))]).run(four_gate_qc)
        b = PassManager([LocalFoldPass(1.5, rng=random.Random(7))]).run(four_gate_qc)
        assert [inst.operation.name for inst in a.data] == [
            inst.operation.name for inst in b.data
        ]

    def test_rejects_below_one_scale(self):
        with pytest.raises(ValueError, match=">= 1"):
            LocalFoldPass(0.5)

    def test_rejects_unknown_selection(self):
        with pytest.raises(ValueError, match="selection must be one of"):
            LocalFoldPass(2.0, selection="middle")

    @pytest.mark.parametrize("selection", ["from_left", "from_right", "random"])
    def test_selection_preserves_unitary_and_size(self, four_gate_qc, selection):
        """All selection strategies yield the same gate count and unitary."""
        folded = PassManager(
            [LocalFoldPass(2.0, selection=selection, rng=random.Random(0))]
        ).run(four_gate_qc)
        assert folded.size() == 8  # d=4, s=2 → 4 + 2*2
        assert Operator(folded).equiv(Operator(four_gate_qc))

    def test_from_left_folds_prefix(self):
        """from_left: only the first n gates should have folded neighbors."""
        qc = QuantumCircuit(1)
        for _ in range(4):
            qc.rx(0.3, 0)
        # s=1.5, d=4 → k=0, n=1 → first gate folded (3 gates), rest untouched.
        folded = PassManager([LocalFoldPass(1.5, selection="from_left")]).run(qc)
        # First 3 instructions = the folded prefix (rx, rx†, rx); next 3 = untouched rx gates.
        names = [inst.operation.name for inst in folded.data]
        assert len(names) == 6
        assert names[:3] == ["rx", "rx", "rx"]  # G G† G (rx inverse is still "rx")
        # The remaining 3 are the untouched original rx gates.
        assert names[3:] == ["rx", "rx", "rx"]

    def test_from_right_folds_suffix(self):
        """from_right: only the last n gates should be folded."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(1)
        qc.cx(0, 1)
        qc.rz(0.2, 0)  # this one should be folded
        # s=1.5, d=4 → n=1, last gate (rz) folded.
        folded = PassManager([LocalFoldPass(1.5, selection="from_right")]).run(qc)
        names = [inst.operation.name for inst in folded.data]
        # Expect: h, x, cx, then rz (folded → rz, rz, rz)
        assert names == ["h", "x", "cx", "rz", "rz", "rz"]

    def test_exclude_by_name_leaves_gates_untouched(self):
        """exclude={'cx'} folds everything but cx gates."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.3, 1)
        qc.cx(1, 0)
        # Foldable pool after exclude: [h, rz] → d=2. s=3 → k=1 → each folded to 3 gates.
        folded = PassManager([LocalFoldPass(3.0, exclude={"cx"})]).run(qc)
        counts = folded.count_ops()
        assert counts["cx"] == 2  # untouched
        assert counts["h"] == 3  # folded
        assert counts["rz"] == 3  # folded
        assert Operator(folded).equiv(Operator(qc))

    def test_exclude_by_arity_shorthand(self):
        """exclude={'double'} skips all 2-qubit gates."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.cz(0, 1)
        qc.rz(0.2, 1)
        folded = PassManager([LocalFoldPass(3.0, exclude={"double"})]).run(qc)
        counts = folded.count_ops()
        assert counts["cx"] == 1
        assert counts["cz"] == 1
        assert counts["h"] == 3
        assert counts["rz"] == 3
        assert Operator(folded).equiv(Operator(qc))

    def test_exclude_affects_effective_scale(self):
        """With 2 foldable gates out of 4, scale=2 gives n=1 extra fold in the pool."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)  # excluded
        qc.rz(0.3, 1)
        qc.cx(1, 0)  # excluded
        # Foldable d=2, s=2 → k=0, n=round(2*1/2)=1 → 1 of 2 foldable gates gets 1 fold.
        # Total size: 2 excluded (1 each) + 1 unfolded (1) + 1 folded (3) = 6.
        folded = PassManager(
            [LocalFoldPass(2.0, exclude={"cx"}, rng=random.Random(0))]
        ).run(qc)
        assert folded.size() == 6
        assert Operator(folded).equiv(Operator(qc))

    def test_exclude_unknown_name_is_harmless(self, four_gate_qc):
        """Unknown op names in exclude are silently no-ops."""
        folded = PassManager([LocalFoldPass(3.0, exclude={"nonexistent_gate"})]).run(
            four_gate_qc
        )
        assert folded.size() == 3 * four_gate_qc.size()
        assert Operator(folded).equiv(Operator(four_gate_qc))

    def test_exclude_all_gates_returns_untouched(self, four_gate_qc):
        """Excluding every gate → empty foldable pool → no change."""
        folded = PassManager([LocalFoldPass(3.0, exclude={"single", "double"})]).run(
            four_gate_qc
        )
        assert folded.size() == four_gate_qc.size()
        assert Operator(folded).equiv(Operator(four_gate_qc))

    def test_deterministic_selections_ignore_rng(self, four_gate_qc):
        """from_left / from_right produce identical output regardless of rng."""
        for selection in ("from_left", "from_right"):
            a = PassManager(
                [LocalFoldPass(1.5, selection=selection, rng=random.Random(0))]
            ).run(four_gate_qc)
            b = PassManager(
                [LocalFoldPass(1.5, selection=selection, rng=random.Random(999))]
            ).run(four_gate_qc)
            assert [i.operation.name for i in a.data] == [
                i.operation.name for i in b.data
            ]

    def test_empty_circuit_is_untouched(self):
        qc = QuantumCircuit(2)
        folded = PassManager([LocalFoldPass(3.0)]).run(qc)
        assert folded.size() == 0


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
