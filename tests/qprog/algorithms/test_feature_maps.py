# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

from divi.qprog.algorithms import AngleEmbedding, FeatureMap, ZZFeatureMap


class TestAngleEmbedding:
    @pytest.mark.parametrize("n_qubits", [1, 2, 4])
    def test_n_params(self, n_qubits):
        assert AngleEmbedding().n_params(n_qubits) == n_qubits

    @pytest.mark.parametrize("rotation", ["X", "Y", "Z"])
    def test_rotation_axes(self, rotation):
        n_qubits = 2
        fm = AngleEmbedding(rotation=rotation)
        params = np.array(ParameterVector("x", n_qubits), dtype=object)
        qc = fm.build(params, n_qubits)
        gate_name = {"X": "rx", "Y": "ry", "Z": "rz"}[rotation]
        op_names = [instr.operation.name for instr in qc.data]
        assert op_names == [gate_name, gate_name]

    def test_invalid_rotation_raises(self):
        with pytest.raises(ValueError, match="rotation must be one of"):
            AngleEmbedding(rotation="W")

    def test_ry_amplitudes_match_reference(self):
        # One-qubit RY(theta) on |0> gives |psi> = (cos(theta/2), sin(theta/2)).
        fm = AngleEmbedding("Y")
        params = np.array(ParameterVector("x", 1), dtype=object)
        qc = fm.build(params, n_qubits=1)
        bound = qc.assign_parameters({params[0]: 0.7})
        sv = Statevector.from_instruction(bound)
        assert np.isclose(sv.data[0], np.cos(0.35))
        assert np.isclose(sv.data[1], np.sin(0.35))


class TestZZFeatureMap:
    @pytest.mark.parametrize("n_qubits", [2, 3, 5])
    def test_n_params(self, n_qubits):
        assert ZZFeatureMap().n_params(n_qubits) == n_qubits

    def test_invalid_layout_raises(self):
        with pytest.raises(ValueError, match="entangling_layout must be"):
            ZZFeatureMap(entangling_layout="zigzag")

    @pytest.mark.parametrize(
        "layout,n_qubits,expected_pairs",
        [
            ("linear", 4, [(0, 1), (1, 2), (2, 3)]),
            ("circular", 4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
            ("circular", 2, [(0, 1)]),
            ("all-to-all", 3, [(0, 1), (0, 2), (1, 2)]),
        ],
    )
    def test_pair_iter(self, layout, n_qubits, expected_pairs):
        fm = ZZFeatureMap(entangling_layout=layout)
        assert fm._pair_iter(n_qubits) == expected_pairs

    def test_build_produces_expected_gate_counts(self):
        n_qubits = 3
        n_pairs = n_qubits - 1
        fm = ZZFeatureMap("linear")
        params = np.array(ParameterVector("x", fm.n_params(n_qubits)), dtype=object)
        qc = fm.build(params, n_qubits)
        op_counts: dict[str, int] = {}
        for instr in qc.data:
            op_counts[instr.operation.name] = op_counts.get(instr.operation.name, 0) + 1
        # n_qubits H, n_qubits RZ from feature encoding, plus the CX-RZ-CX
        # decomposition for each pair (2 CX + 1 RZ per pair).
        assert op_counts["h"] == n_qubits
        assert op_counts["rz"] == n_qubits + n_pairs
        assert op_counts["cx"] == 2 * n_pairs

    def test_rejects_single_qubit(self):
        with pytest.raises(ValueError, match="requires at least 2 qubits"):
            ZZFeatureMap().build(
                np.array(ParameterVector("x", 1), dtype=object), n_qubits=1
            )

    def test_n_params_rejects_single_qubit(self):
        # n_params guards independently of build (it's the contract entry point).
        with pytest.raises(ValueError, match="requires at least 2 qubits"):
            ZZFeatureMap().n_params(1)

    def test_encoded_angles_match_reference(self):
        # The bound circuit must apply RZ(2*x_i) and RZZ(2*(pi-x_i)(pi-x_j)),
        # matching a hand-built reference (validates the angle formulas, not
        # just gate counts).
        x = [0.3, 0.7]
        params = np.array(ParameterVector("x", 2), dtype=object)
        bound = (
            ZZFeatureMap("linear")
            .build(params, n_qubits=2)
            .assign_parameters({params[0]: x[0], params[1]: x[1]})
        )

        ref = QuantumCircuit(2)
        ref.h(0)
        ref.h(1)
        ref.rz(2.0 * x[0], 0)
        ref.rz(2.0 * x[1], 1)
        ref.rzz(2.0 * (np.pi - x[0]) * (np.pi - x[1]), 0, 1)

        assert Statevector.from_instruction(bound).equiv(
            Statevector.from_instruction(ref)
        )


def test_feature_map_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        FeatureMap()  # type: ignore[abstract]
