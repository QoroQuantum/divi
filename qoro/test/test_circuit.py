import pytest
from circuit import Circuit


def test_initialization():
    import pennylane as qml
    device = qml.device("default.qubit", wires=4)
    qml.RX(0.5, wires=0)
    qml.RX(0.5, wires=1)
    qml.RY(0.5, wires=2)
    qml.RZ(0.25, wires=3)
    circuit = Circuit(device, circuit_type="pennylane")
    assert circuit is not None, "Circuit should be initialized"
    assert circuit.circuit_id == 0, "Circuit ID should be 0"
    assert circuit.circuit_type == "pennylane", "Circuit type should be pennylane"
