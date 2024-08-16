import pytest
from qoro_service import QoroService


@pytest.mark.requires_api_token
def test_qoro_service_initialization(setup_module):
    # Test if QoroService is initialized correctly
    api_token = setup_module
    qoro_serivice = QoroService(api_token)
    response = qoro_serivice.test_connection()
    assert response.status_code == 200, "Connection should be successful"


@pytest.mark.requires_api_token
def test_declare_architecture(setup_module):
    # Test if QoroService can declare a QPU architecture
    api_token = setup_module
    service = QoroService(api_token)
    system_name = "test_system"
    qubits = [2, 3]
    classical_bits = [2, 3]
    architectures = ["Test", "Test"]
    system_kinds = ["Test", "Test"]
    system_id = service.declare_architecture(
        system_name, qubits, classical_bits, architectures, system_kinds)

    assert system_id is not None, "Architecture declaration should be successful"
    assert system_id != "", "System ID should not be empty"
    res = service.delete_architecture(system_id)
    assert res.status_code == 204, "Deletion should be successful"


def test_qoro_fail_api_connection(setup_module):
    # Test if QoroService fails to connect to the API
    api_token = "invalid_token"
    qoro_serivce = QoroService(api_token)
    response = qoro_serivce.test_connection()
    assert response.status_code != 200, "Connection should fail with invalid token"
ß