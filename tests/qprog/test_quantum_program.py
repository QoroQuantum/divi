# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from queue import Queue
from threading import Event
from unittest.mock import Mock

import pytest

from divi.qprog.quantum_program import QuantumProgram


class ConcreteQuantumProgram(QuantumProgram):
    """Concrete implementation of QuantumProgram for testing."""

    def __init__(self, backend, seed=None, progress_queue=None, **kwargs):
        super().__init__(backend, seed, progress_queue, **kwargs)
        self._total_circuit_count = 0
        self._total_run_time = 0.0

    def run(self) -> tuple[int, float]:
        """Concrete implementation of run method."""
        return (5, 1.5)

    def _generate_circuits(self, **kwargs):
        """Concrete implementation of _generate_circuits method."""
        # Return empty list as this is a minimal test implementation
        return []

    def _post_process_results(self, results: dict):
        """Concrete implementation of _post_process_results method."""
        return {"processed": "results"}


class TestQuantumProgram:
    """Test suite for QuantumProgram abstract base class."""

    def test_initialization_comprehensive(self):
        """Test QuantumProgram initialization with various parameters."""
        mock_backend = Mock()
        mock_queue = Queue()

        # Test basic initialization
        program1 = ConcreteQuantumProgram(
            backend=mock_backend, seed=42, progress_queue=mock_queue
        )

        assert program1.backend == mock_backend
        assert program1._seed == 42
        assert program1._progress_queue == mock_queue

        # Test initialization with kwargs
        program2 = ConcreteQuantumProgram(
            backend=mock_backend, custom_param="test_value", another_param=123
        )

        assert program2.backend == mock_backend
        assert program2._seed is None
        assert program2._progress_queue is None

    def test_abstract_class_behavior(self):
        """Test abstract class instantiation behavior."""
        mock_backend = Mock()

        # Test that abstract class cannot be instantiated
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            QuantumProgram(backend=mock_backend)

        # Test that concrete implementations can be instantiated
        program = ConcreteQuantumProgram(backend=mock_backend)
        assert isinstance(program, QuantumProgram)
        assert program.backend == mock_backend

    def test_abstract_methods_must_be_implemented(self):
        """Test that all abstract methods must be implemented in subclasses."""
        mock_backend = Mock()

        # Test missing run method
        class IncompleteProgram1(QuantumProgram):
            def _generate_circuits(self, **kwargs):
                return []

            def _post_process_results(self, results):
                return {}

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteProgram1(backend=mock_backend)

        # Test missing _generate_circuits method
        class IncompleteProgram2(QuantumProgram):
            def run(self):
                return (0, 0.0)

            def _post_process_results(self, results):
                return {}

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteProgram2(backend=mock_backend)

        # Test missing _post_process_results method
        class IncompleteProgram3(QuantumProgram):
            def run(self):
                return (0, 0.0)

            def _generate_circuits(self, **kwargs):
                return []

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteProgram3(backend=mock_backend)

    def test_method_implementations(self):
        """Test concrete method implementations."""
        mock_backend = Mock()
        program = ConcreteQuantumProgram(backend=mock_backend)

        # Test run method
        circuits_count, runtime = program.run()
        assert circuits_count == 5
        assert runtime == 1.5

        # Test _generate_circuits method
        circuits = program._generate_circuits(test_param="value")
        assert isinstance(circuits, list)
        assert circuits == []

        # Test _post_process_results method
        results = {"raw": "data"}
        processed = program._post_process_results(results)
        assert processed == {"processed": "results"}

    def test_cancellation_event(self):
        """Test _set_cancellation_event method."""
        mock_backend = Mock()
        program = ConcreteQuantumProgram(backend=mock_backend)

        event = Event()
        program._set_cancellation_event(event)

        assert hasattr(program, "_cancellation_event")
        assert program._cancellation_event == event

    def test_property_methods_comprehensive(self):
        """Test property methods with various attribute states."""
        mock_backend = Mock()

        # Test with both attributes set
        program1 = ConcreteQuantumProgram(backend=mock_backend)
        program1._total_circuit_count = 15
        program1._total_run_time = 3.7

        assert program1.total_circuit_count == 15
        assert program1.total_run_time == 3.7

        # Test with one attribute missing
        class PartialProgram(QuantumProgram):
            def run(self):
                return (0, 0.0)

            def _generate_circuits(self, **kwargs):
                return []

            def _post_process_results(self, results):
                return {}

            def __init__(self, backend):
                super().__init__(backend)
                self._total_circuit_count = 8  # Only set circuit count

        program2 = PartialProgram(mock_backend)
        assert program2.total_circuit_count == 8
        assert program2.total_run_time == 0.0  # Default value

        # Test with both attributes missing
        class NoAttributesProgram(QuantumProgram):
            def run(self):
                return (0, 0.0)

            def _generate_circuits(self, **kwargs):
                return []

            def _post_process_results(self, results):
                return {}

        program3 = NoAttributesProgram(mock_backend)
        assert program3.total_circuit_count == 0  # Default value
        assert program3.total_run_time == 0.0  # Default value
