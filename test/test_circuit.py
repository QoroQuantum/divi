from divi.circuit import Circuit


def test_initialization():
    import pennylane as qml

    def test_circuit():
        qml.RX(0.5, wires=0)
        qml.RX(0.5, wires=1)
        qml.RY(0.5, wires=2)
        qml.RZ(0.25, wires=3)

    qscript = qml.tape.make_qscript(test_circuit)()

    circuit = Circuit(qscript, circuit_type="pennylane")

    assert circuit is not None, "Circuit should be initialized"
    assert circuit.circuit_id == 0, "Circuit ID should be 0"
    assert circuit.circuit_type == "pennylane", "Circuit type should be pennylane"
