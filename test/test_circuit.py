from divi.circuit import Circuit


class TestCircuit:
    def test_pennylane_circuit_initialization(self):
        import pennylane as qml

        def test_circuit():
            qml.RX(0.5, wires=0)
            qml.RX(0.5, wires=1)
            qml.RY(0.5, wires=2)
            qml.RZ(0.25, wires=3)

            return qml.expval(
                qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliX(2) @ qml.PauliY(3)
            )

        qscript = qml.tape.make_qscript(test_circuit)()

        circuit = Circuit(qscript, tags=["test_circ"])

        assert circuit.main_circuit == qscript
        assert circuit.tags == ["test_circ"]
        assert circuit.circuit_id == 0
        assert circuit.circuit_type == "pennylane"

        assert len(circuit.qasm_circuits) == 1


class TestMetaCircuit:
    pass
