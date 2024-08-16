import pytest
from qoro_service import QoroService

@pytest.mark.requires_api_token
def test_qoro_service_initialization(setup_module):
    # Test if QoroService is initialized correctly
    api_token = setup_module
    qoro_serivice = QoroService(api_token)
    response = qoro_serivice.test_connection()
    assert response.status_code == 200, "Connection should be successful"


def test_qoro_fail_api_connection(setup_module):
    # Test if QoroService fails to connect to the API
    api_token = "invalid_token"
    qoro_serivce = QoroService(api_token)
    response = qoro_serivce.test_connection()
    assert response.status_code != 200, "Connection should fail with invalid token"