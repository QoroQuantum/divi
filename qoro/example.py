from qoro_service import QoroService

if __name__ == "__main__":
    # Test if QoroService is initialized correctly
    api_token = "YOUR_API_TOKEN"
    service = QoroService(api_token)
    service.test_connection()
    id = service.declare_architecture("test_system", [2, 3], [2, 3], ["Test", "Test"], ["Test", "Test"])
    print(id)
    response = service.delete_architecture(id)
    print(response.status_code)