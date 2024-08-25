class QuantumProgram:
    def __init__(self, qoro_service=None) -> None:
        self.qoro_service = qoro_service
        self.circuits = None
        self.job_id = None
