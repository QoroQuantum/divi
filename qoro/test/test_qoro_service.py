import pytest
import requests
from qoro_service import QoroService, JobStatus


@pytest.mark.requires_api_token
def test_service_initialization(setup_module):
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


@pytest.mark.requires_api_token
def test_send_circuits(setup_module):
    # Test if QoroService can send circuits
    api_token = setup_module
    circuit = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[4];\ncreg c[4];\nx q[0];\nx q[1];\nry(0) q[2];\ncx q[2],q[3];\ncx q[2],q[0];\ncx q[3],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\nmeasure q[3] -> c[3];\n"
    circuits = {}
    for i in range(10):
        circuits[f"circuit_{i}"] = circuit

    service = QoroService(api_token)
    job_id = service.send_circuits(circuits)
    assert job_id is not None, "Job ID should not be None"
    res = service.delete_job(job_id)
    res.status_code == 204, "Deletion should be successful"


@pytest.mark.requires_api_token
def test_get_job_status(setup_module):
    # Test if QoroService can get the status of a job
    api_token = setup_module
    circuit = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[4];\ncreg c[4];\nx q[0];\nx q[1];\nry(0) q[2];\ncx q[2],q[3];\ncx q[2],q[0];\ncx q[3],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\nmeasure q[3] -> c[3];\n"
    circuits = {}
    for i in range(10):
        circuits[f"circuit_{i}"] = circuit

    service = QoroService(api_token)
    job_id = service.send_circuits(circuits)
    status = service.job_status(job_id)
    assert status is not None, "Status should not be None"
    assert status != "", "Status should not be empty"
    assert status == JobStatus.PENDING.value, "Status should be PENDING"
    res = service.delete_job(job_id)
    res.status_code == 204, "Deletion should be successful"


def test_fail_declare_architecture(setup_module):
    # Test if QoroService fails to connect declare a QPU architecture
    api_token = "invalid_token"
    service = QoroService(api_token)
    system_name = "test_system"
    qubits = [2, 3]
    classical_bits = [2, 3]
    architectures = ["Test", "Test"]
    system_kinds = ["Test", "Test"]
    pytest.raises(requests.exceptions.HTTPError, service.declare_architecture,
                  system_name, qubits, classical_bits, architectures, system_kinds)


def test_fail_send_circuits(setup_module):
    # Test if QoroService fails to connect send circuits
    api_token = "invalid_token"
    circuit = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[4];\ncreg c[4];\nx q[0];\nx q[1];\nry(0) q[2];\ncx q[2],q[3];\ncx q[2],q[0];\ncx q[3],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\nmeasure q[3] -> c[3];\n"
    service = QoroService(api_token)
    circuits = {}
    for i in range(10):
        circuits[f"circuit_{i}"] = circuit
    pytest.raises(requests.exceptions.HTTPError,
                  service.send_circuits, circuits)


def test_fail_api_connection(setup_module):
    # Test if QoroService fails to connect to the API
    api_token = "invalid_token"
    service = QoroService(api_token)
    response = service.test_connection()
    assert response.status_code != 200, "Connection should fail with invalid token"
