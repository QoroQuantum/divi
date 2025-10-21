# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import cirq
import pytest
from mitiq.zne.inference import Factory

from divi.circuits import qem
from divi.circuits.qem import ZNE, QEMProtocol, _NoMitigation


class TestQEMProtocol:
    """Test suite for the abstract QEMProtocol base class."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that the abstract QEMProtocol class cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            QEMProtocol()

    def test_concrete_implementations_can_be_instantiated(self, mocker):
        """Test that concrete implementations of QEMProtocol can be instantiated."""
        # Test _NoMitigation
        no_mitigation = _NoMitigation()
        assert isinstance(no_mitigation, QEMProtocol)

        # Test ZNE (with valid parameters)
        mock_factory = mocker.Mock(spec=Factory)

        def dummy_folding(circuit, scale_factor):
            return circuit

        folding_fn = partial(dummy_folding)

        zne = ZNE(
            scale_factors=[1.0, 2.0],
            folding_fn=folding_fn,
            extrapolation_factory=mock_factory,
        )
        assert isinstance(zne, QEMProtocol)


class TestNoMitigation:
    """Test suite for the _NoMitigation protocol."""

    @pytest.fixture
    def protocol(self):
        """Returns an instance of _NoMitigation."""
        return _NoMitigation()

    @pytest.fixture
    def circuit(self):
        """Returns a simple Cirq circuit."""
        return cirq.Circuit(cirq.X(cirq.LineQubit(0)))

    def test_name_property(self, protocol):
        """Test that the name property is correct."""
        assert protocol.name == "NoMitigation"

    def test_modify_circuit_is_identity(self, protocol, circuit):
        """Test that modify_circuit returns the original circuit in a list."""
        modified_circuits = protocol.modify_circuit(circuit)
        assert modified_circuits == [circuit]
        assert modified_circuits[0] is circuit  # Check it's the same object

    def test_postprocess_results_returns_single_value(self, protocol):
        """Test postprocess_results correctly returns the single result."""
        assert protocol.postprocess_results([1.23]) == 1.23
        assert protocol.postprocess_results([-0.5]) == -0.5

    def test_postprocess_results_raises_error_for_multiple_values(self, protocol):
        """Test postprocess_results raises RuntimeError for more than one result."""
        with pytest.raises(
            RuntimeError, match="NoMitigation class received multiple partial results."
        ):
            protocol.postprocess_results([1.0, 2.0])


class TestZNE:
    """Test suite for the ZNE (Zero Noise Extrapolation) class."""

    @pytest.fixture
    def mock_factory(self, mocker):
        """Create a mock Factory instance with a mock extrapolate method."""
        factory = mocker.Mock(spec=Factory)
        factory.extrapolate = mocker.Mock(return_value=0.85)
        return factory

    @pytest.fixture
    def mock_folding_fn(self):
        """Create a mock folding function as a partial."""

        def dummy_folding(circuit, scale_factor, some_param=None):
            return circuit

        return partial(dummy_folding, some_param="test")

    def test_initialization_valid(self, mock_folding_fn, mock_factory):
        """Test valid ZNE initialization."""
        zne_instance = ZNE(
            scale_factors=[1.0, 2.0, 3.0],
            folding_fn=mock_folding_fn,
            extrapolation_factory=mock_factory,
        )
        assert isinstance(zne_instance, ZNE)
        # Test with empty but valid scale factors
        ZNE(
            scale_factors=[],
            folding_fn=mock_folding_fn,
            extrapolation_factory=mock_factory,
        )

    def test_properties(self, mock_folding_fn, mock_factory):
        """Test that ZNE properties return the correct values."""
        scale_factors = [1.0, 3.0]
        zne = ZNE(
            scale_factors=scale_factors,
            folding_fn=mock_folding_fn,
            extrapolation_factory=mock_factory,
        )
        assert zne.name == "zne"
        assert zne.scale_factors == scale_factors
        assert zne.folding_fn == mock_folding_fn
        assert zne.extrapolation_factory == mock_factory

    @pytest.mark.parametrize(
        "invalid_factors",
        [
            pytest.param(1.0, id="not_a_sequence"),
            pytest.param([1.0, "two", 3.0], id="contains_non_numeric"),
            pytest.param([1.0, -2.0, 3.0], id="contains_negative_value"),
            pytest.param([1.0, 0.5, 2.0], id="contains_value_less_than_1"),
        ],
    )
    def test_initialization_invalid_scale_factors(
        self, invalid_factors, mock_folding_fn, mock_factory
    ):
        """Test ZNE initialization with various invalid scale_factors."""
        with pytest.raises(
            ValueError,
            match="scale_factors is expected to be a sequence of real numbers >=1",
        ):
            ZNE(
                scale_factors=invalid_factors,
                folding_fn=mock_folding_fn,
                extrapolation_factory=mock_factory,
            )

    @pytest.mark.parametrize(
        "invalid_fn",
        [
            pytest.param(lambda: None, id="not_a_partial"),
            pytest.param(None, id="is_None"),
            pytest.param("not_callable", id="is_string"),
        ],
    )
    def test_initialization_invalid_folding_fn(self, invalid_fn, mock_factory):
        """Test ZNE initialization with an invalid folding_fn."""
        with pytest.raises(
            ValueError, match="folding_fn is expected to be of type partial"
        ):
            ZNE(
                scale_factors=[1.0, 2.0],
                folding_fn=invalid_fn,
                extrapolation_factory=mock_factory,
            )

    def test_initialization_invalid_factory(self, mock_folding_fn):
        """Test ZNE initialization with an invalid extrapolation_factory."""
        with pytest.raises(
            ValueError, match="extrapolation_fn is expected to be of Factory"
        ):
            ZNE(
                scale_factors=[1.0, 2.0],
                folding_fn=mock_folding_fn,
                extrapolation_factory="not_a_factory_object",
            )

    def test_modify_circuit_calls_mitiq_construct_circuits(
        self, mocker, mock_folding_fn, mock_factory
    ):
        """Test that modify_circuit correctly calls mitiq.construct_circuits."""
        mock_construct = mocker.patch(f"{qem.__name__}.construct_circuits")
        circuit = cirq.Circuit()
        scale_factors = [1.0, 2.0, 3.0]

        zne = ZNE(
            scale_factors=scale_factors,
            folding_fn=mock_folding_fn,
            extrapolation_factory=mock_factory,
        )
        zne.modify_circuit(circuit)

        mock_construct.assert_called_once_with(
            circuit,
            scale_factors=scale_factors,
            scale_method=mock_folding_fn,
        )

    def test_postprocess_results_calls_mitiq_combine_results(
        self, mocker, mock_folding_fn, mock_factory
    ):
        """Test that postprocess_results correctly calls mitiq.combine_results."""
        mock_combine = mocker.patch(f"{qem.__name__}.combine_results")
        mock_combine.return_value = 0.95  # Set a return value to check

        scale_factors = [1.0, 2.0, 3.0]
        results = [0.9, 0.8, 0.7]

        zne = ZNE(
            scale_factors=scale_factors,
            folding_fn=mock_folding_fn,
            extrapolation_factory=mock_factory,
        )
        final_result = zne.postprocess_results(results)

        mock_combine.assert_called_once_with(
            scale_factors=scale_factors,
            results=results,
            extrapolation_method=mock_factory.extrapolate,
        )
        assert final_result == 0.95
