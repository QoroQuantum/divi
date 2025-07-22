# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import cirq
import pytest
from mitiq.zne.inference import Factory

from divi.qem import ZNE, _NoMitigation


class TestNoMitigation:
    @pytest.fixture
    def dummy_cirq_circuit(self):
        """Returns a simple Cirq circuit for testing."""
        q = cirq.LineQubit(0)
        return cirq.Circuit(cirq.X(q), cirq.Z(q))

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test in TestNoMitigation."""
        self.protocol = _NoMitigation()

    def test_name_property(self):
        """Test the name property."""
        assert self.protocol.name == "NoMitigation"

    def test_modify_circuit_returns_original_circuit_in_list(self, dummy_cirq_circuit):
        modified_circuits = self.protocol.modify_circuit(dummy_cirq_circuit)
        assert len(modified_circuits) == 1
        assert modified_circuits[0] is dummy_cirq_circuit

    def test_postprocess_results_returns_single_result(self):
        assert self.protocol.postprocess_results([1.23]) == 1.23

    def test_postprocess_results_raises_error_for_multiple_results(self):
        with pytest.raises(
            RuntimeError, match="NoMitigation class received multiple partial results."
        ):
            self.protocol.postprocess_results([1.0, 2.0])


class TestZNE:
    """Test suite for the ZNE (Zero Noise Extrapolation) class."""

    @pytest.fixture
    def mock_factory(self, mocker):
        """Create a mock Factory instance."""
        factory = mocker.Mock(spec=Factory)
        factory.extrapolate = mocker.Mock(return_value=0.85)
        return factory

    @pytest.fixture
    def mock_folding_fn(self):
        """Create a mock folding function as a partial."""

        def dummy_folding(circuit, scale_factor, some_param=None):
            return circuit

        return partial(dummy_folding, some_param="test")

    @pytest.fixture
    def valid_scale_factors(self):
        """Provide valid scale factors."""
        return [1.0, 2.0, 3.0]

    @pytest.fixture
    def sample_circuit(self):
        """Create a simple Cirq circuit for testing."""
        circuit = cirq.Circuit()
        qubit = cirq.LineQubit(0)
        circuit.append(cirq.X(qubit))
        return circuit

    @pytest.fixture
    def zne_instance(self, valid_scale_factors, mock_folding_fn, mock_factory):
        """Create a valid ZNE instance for testing."""
        return ZNE(
            scale_factors=valid_scale_factors,
            folding_fn=mock_folding_fn,
            extrapolation_factory=mock_factory,
        )

    def test_zne_initialization_valid_inputs(
        self, zne_instance, valid_scale_factors, mock_folding_fn, mock_factory
    ):
        """Test ZNE initialization with valid inputs."""

        assert zne_instance.scale_factors == valid_scale_factors
        assert zne_instance.folding_fn == mock_folding_fn
        assert zne_instance.extrapolation_factory == mock_factory
        assert zne_instance.name == "zne"

    def test_zne_initialization_invalid_scale_factors(
        self, mock_folding_fn, mock_factory
    ):
        """Test ZNE initialization with invalid scale factors."""
        # Test with non-sequence
        with pytest.raises(
            ValueError,
            match="scale_factors is expected to be a sequence of real numbers",
        ):
            ZNE(
                scale_factors=1.0,
                folding_fn=mock_folding_fn,
                extrapolation_factory=mock_factory,
            )

        # Test with sequence containing non-numeric values
        with pytest.raises(
            ValueError,
            match="scale_factors is expected to be a sequence of real numbers",
        ):
            ZNE(
                scale_factors=[1.0, "invalid", 3.0],
                folding_fn=mock_folding_fn,
                extrapolation_factory=mock_factory,
            )

        with pytest.raises(
            ValueError,
            match="scale_factors is expected to be a sequence of real numbers",
        ):
            ZNE(
                scale_factors=[1.0, -1, 3.0],
                folding_fn=mock_folding_fn,
                extrapolation_factory=mock_factory,
            )

        # Test with empty sequence (should be valid as it's still a sequence)
        zne = ZNE(
            scale_factors=[],
            folding_fn=mock_folding_fn,
            extrapolation_factory=mock_factory,
        )
        assert zne.scale_factors == []

    def test_zne_initialization_invalid_folding_fn(
        self, valid_scale_factors, mock_factory
    ):
        """Test ZNE initialization with invalid folding function."""

        # Test with non-partial function
        def regular_function():
            pass

        with pytest.raises(
            ValueError, match="folding_fn is expected to be of type partial"
        ):
            ZNE(
                scale_factors=valid_scale_factors,
                folding_fn=regular_function,
                extrapolation_factory=mock_factory,
            )

        # Test with None
        with pytest.raises(
            ValueError, match="folding_fn is expected to be of type partial"
        ):
            ZNE(
                scale_factors=valid_scale_factors,
                folding_fn=None,
                extrapolation_factory=mock_factory,
            )

    def test_zne_initialization_invalid_extrapolation_factory(
        self, valid_scale_factors, mock_folding_fn
    ):
        """Test ZNE initialization with invalid extrapolation factory."""
        with pytest.raises(
            ValueError, match="extrapolation_fn is expected to be of Factory"
        ):
            ZNE(
                scale_factors=valid_scale_factors,
                folding_fn=mock_folding_fn,
                extrapolation_factory="not_a_factory",
            )

    def test_modify_circuit_calls_construct_circuits(
        self, mocker, mock_folding_fn, mock_factory, sample_circuit
    ):
        """Test that modify_circuit calls construct_circuits with correct parameters."""
        scale_factors = [[1.0, 2.0, 3.0], [1.0], [1, 2, 3]]
        # Mock the construct_circuits function
        mock_construct = mocker.patch("divi.qem.construct_circuits")
        mock_construct.return_value = [sample_circuit, sample_circuit]

        for scale_factor in scale_factors:
            zne = ZNE(
                scale_factors=scale_factor,
                folding_fn=mock_folding_fn,
                extrapolation_factory=mock_factory,
            )

            result = zne.modify_circuit(sample_circuit)

            mock_construct.assert_called_with(
                sample_circuit,
                scale_factors=scale_factor,
                scale_method=mock_folding_fn,
            )

        # Verify return value
        assert result == [sample_circuit, sample_circuit]

    def test_postprocess_results_calls_combine_results(
        self, mocker, mock_folding_fn, mock_factory
    ):
        """Test that postprocess_results calls combine_results with correct parameters."""
        # Mock the combine_results function
        mock_combine = mocker.patch("divi.qem.combine_results")  # Adjust import path
        mock_combine.return_value = 0.95

        zne = ZNE(
            scale_factors=[1.0, 2.0, 3.0],
            folding_fn=mock_folding_fn,
            extrapolation_factory=mock_factory,
        )

        result = zne.postprocess_results([0.9, 0.8, 0.7])

        # Verify combine_results was called with correct arguments
        mock_combine.assert_called_once_with(
            scale_factors=[1.0, 2.0, 3.0],
            results=[0.9, 0.8, 0.7],
            extrapolation_method=mock_factory.extrapolate,
        )

        # Verify return value
        assert result == 0.95
