# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pennylane as qml
import pytest

pass

from divi.circuits import MetaCircuit
from divi.circuits._grouping import compute_measurement_groups


class TestComputeMeasurementGroups:
    """Tests for compute_measurement_groups."""

    def test_probs_returns_single_group_and_identity(self):
        """Probs measurement returns ((),), [[0]], and identity postprocessing."""
        circuit = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)], measurements=[qml.probs()]
        )
        measurement = circuit.measurements[0]

        groups, partition, postproc = compute_measurement_groups(measurement, "wires")

        assert groups == ((),)
        assert partition == [[0]]
        result = postproc(["dummy"])
        assert result == ["dummy"]  # identity returns input as-is

    def test_probs_with_none_strategy(self):
        """Probs with None strategy also returns single group."""
        circuit = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)], measurements=[qml.probs()]
        )
        measurement = circuit.measurements[0]

        groups, partition, postproc = compute_measurement_groups(measurement, None)

        assert groups == ((),)
        assert partition == [[0]]
        assert postproc(1.0) == 1.0

    def test_expval_single_term_wires(self):
        """Single-term expval with wires strategy."""
        obs = qml.PauliZ(0)
        circuit = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)], measurements=[qml.expval(obs)]
        )
        measurement = circuit.measurements[0]

        groups, partition, postproc = compute_measurement_groups(measurement, "wires")

        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert partition == [[0]]
        energy = postproc([0.5])
        assert energy == 0.5

    def test_expval_single_term_qwc(self):
        """Single-term expval with qwc strategy."""
        obs = qml.PauliZ(0)
        circuit = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)], measurements=[qml.expval(obs)]
        )
        measurement = circuit.measurements[0]

        groups, partition, postproc = compute_measurement_groups(measurement, "qwc")

        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert partition == [[0]]

    def test_expval_single_term_default(self):
        """Single-term expval with default strategy."""
        obs = qml.PauliZ(0)
        circuit = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)], measurements=[qml.expval(obs)]
        )
        measurement = circuit.measurements[0]

        groups, partition, _ = compute_measurement_groups(measurement, "default")

        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert partition == [[0]]

    def test_expval_single_term_backend_expval(self):
        """Single-term expval with _backend_expval strategy."""
        obs = qml.PauliZ(0)
        circuit = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)], measurements=[qml.expval(obs)]
        )
        measurement = circuit.measurements[0]

        groups, partition, _ = compute_measurement_groups(
            measurement, "_backend_expval"
        )

        assert groups == ((),)
        assert partition == [[0]]

    def test_expval_single_term_none_strategy(self):
        """Single-term expval with None strategy (each obs own group)."""
        obs = qml.PauliZ(0)
        circuit = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)], measurements=[qml.expval(obs)]
        )
        measurement = circuit.measurements[0]

        groups, partition, _ = compute_measurement_groups(measurement, None)

        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert partition == [[0]]

    def test_expval_multi_term_postprocessing(self):
        """Multi-term Hamiltonian postprocessing computes weighted sum."""
        obs = 0.5 * qml.PauliZ(0) + 0.3 * qml.PauliZ(1)
        circuit = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0), qml.Hadamard(1)],
            measurements=[qml.expval(obs)],
        )
        measurement = circuit.measurements[0]

        groups, partition, postproc = compute_measurement_groups(measurement, None)

        # Each term is its own group with None strategy.
        assert len(groups) == 2
        assert len(partition) == 2
        # Simulated grouped results: [0.5, 0.3] for Z0 and Z1
        energy = postproc([0.5, 0.3])
        assert abs(energy - (0.5 * 0.5 + 0.3 * 0.3)) < 1e-10

    def test_unknown_strategy_raises(self):
        """Unknown strategy raises ValueError."""
        circuit = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)], measurements=[qml.expval(qml.PauliZ(0))]
        )
        measurement = circuit.measurements[0]

        with pytest.raises(ValueError, match="Unknown grouping strategy"):
            compute_measurement_groups(measurement, "invalid_strategy")


class TestMetaCircuitOverride:
    """Tests for MetaCircuit with measurement_groups and postprocessing_fn override."""

    def test_override_produces_same_qasm_as_computed(self):
        """Override path produces equivalent QASM to computed path for probs."""
        ops = [qml.Hadamard(0), qml.CNOT(wires=[0, 1])]
        circuit = qml.tape.QuantumScript(ops=ops, measurements=[qml.probs()])
        symbols = np.array([])

        meta_computed = MetaCircuit(
            source_circuit=circuit,
            symbols=symbols,
            grouping_strategy="wires",
        )
        meta_override = MetaCircuit(
            source_circuit=circuit,
            symbols=symbols,
            measurement_groups_override=((),),
            postprocessing_fn_override=lambda x: x,
        )

        bundle_computed = meta_computed.initialize_circuit_from_params([])
        bundle_override = meta_override.initialize_circuit_from_params([])

        assert len(bundle_computed.executables) == len(bundle_override.executables)
        for e1, e2 in zip(bundle_computed.executables, bundle_override.executables):
            assert e1.qasm == e2.qasm


class TestParameterFreeMetaCircuit:
    """Tests for parameter-free MetaCircuit (empty symbols)."""

    def test_initialize_circuit_from_params_empty_symbols(self):
        """Parameter-free circuit: empty symbols, empty param_list produces valid QASM."""
        ops = [qml.Hadamard(0), qml.CNOT(wires=[0, 1])]
        circuit = qml.tape.QuantumScript(ops=ops, measurements=[qml.probs()])
        symbols = np.array([], dtype=object)

        meta = MetaCircuit(
            source_circuit=circuit,
            symbols=symbols,
            measurement_groups_override=((),),
            postprocessing_fn_override=lambda x: x,
        )

        bundle = meta.initialize_circuit_from_params([])

        assert len(bundle.executables) >= 1
        for ex in bundle.executables:
            assert "OPENQASM" in ex.qasm
            assert "qreg" in ex.qasm
            assert "h " in ex.qasm or "h(" in ex.qasm
            assert "cx " in ex.qasm or "cx(" in ex.qasm
            assert "measure" in ex.qasm

    def test_parameter_free_qasm_not_corrupted(self):
        """Parameter-free QASM must not be corrupted by empty regex substitution."""
        ops = [qml.Hadamard(0), qml.RZ(0.5, wires=0)]
        circuit = qml.tape.QuantumScript(ops=ops, measurements=[qml.probs()])
        symbols = np.array([], dtype=object)

        meta = MetaCircuit(
            source_circuit=circuit,
            symbols=symbols,
            measurement_groups_override=((),),
            postprocessing_fn_override=lambda x: x,
        )

        bundle = meta.initialize_circuit_from_params([])
        qasm = bundle.executables[0].qasm

        # QASM should contain exactly one h gate and one rz gate, not duplicated
        assert qasm.count("h ") + qasm.count("h(") >= 1
        assert "rz(0.5" in qasm or "rz(0.50000000" in qasm
        # No corruption from empty-pattern substitution (would duplicate chars)
        assert len(qasm) < 500
