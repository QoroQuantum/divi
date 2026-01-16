# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import datetime

from qiskit_ibm_runtime.models.backend_properties import BackendProperties

from divi.backends._backend_properties_conversion import (
    _normalize_nduv,
    _normalize_properties,
    create_backend_from_properties,
)


class TestNormalizeNduv:
    """Tests for _normalize_nduv helper function."""

    def test_adds_missing_date(self):
        """Test that missing date field is added."""
        default_date = datetime.datetime(2025, 1, 1, 12, 0, 0)
        nduv = {"name": "T1", "value": 100.0}

        result = _normalize_nduv(nduv, default_date)

        assert result["date"] == default_date
        assert result["name"] == "T1"
        assert result["value"] == 100.0

    def test_preserves_existing_date(self):
        """Test that existing date field is preserved."""
        existing_date = datetime.datetime(2024, 12, 25, 10, 0, 0)
        default_date = datetime.datetime(2025, 1, 1, 12, 0, 0)
        nduv = {"name": "T1", "date": existing_date, "value": 100.0}

        result = _normalize_nduv(nduv, default_date)

        assert result["date"] == existing_date
        assert result["date"] != default_date

    def test_adds_unit_for_gate_error(self):
        """Test that gate_error gets empty string unit."""
        default_date = datetime.datetime(2025, 1, 1)
        nduv = {"name": "gate_error", "value": 0.01}

        result = _normalize_nduv(nduv, default_date)

        assert result["unit"] == ""

    def test_adds_unit_for_readout_error(self):
        """Test that readout_error gets empty string unit."""
        default_date = datetime.datetime(2025, 1, 1)
        nduv = {"name": "readout_error", "value": 0.02}

        result = _normalize_nduv(nduv, default_date)

        assert result["unit"] == ""

    def test_adds_unit_for_t1(self):
        """Test that T1 gets microseconds unit."""
        default_date = datetime.datetime(2025, 1, 1)
        nduv = {"name": "T1", "value": 100.0}

        result = _normalize_nduv(nduv, default_date)

        assert result["unit"] == "us"

    def test_adds_unit_for_t2(self):
        """Test that T2 gets microseconds unit."""
        default_date = datetime.datetime(2025, 1, 1)
        nduv = {"name": "T2", "value": 80.0}

        result = _normalize_nduv(nduv, default_date)

        assert result["unit"] == "us"

    def test_adds_unit_for_gate_length(self):
        """Test that gate_length gets nanoseconds unit."""
        default_date = datetime.datetime(2025, 1, 1)
        nduv = {"name": "gate_length", "value": 35.0}

        result = _normalize_nduv(nduv, default_date)

        assert result["unit"] == "ns"

    def test_adds_unit_for_readout_length(self):
        """Test that readout_length gets nanoseconds unit."""
        default_date = datetime.datetime(2025, 1, 1)
        nduv = {"name": "readout_length", "value": 1000.0}

        result = _normalize_nduv(nduv, default_date)

        assert result["unit"] == "ns"

    def test_adds_unit_for_frequency(self):
        """Test that frequency gets GHz unit."""
        default_date = datetime.datetime(2025, 1, 1)
        nduv = {"name": "frequency", "value": 5.0}

        result = _normalize_nduv(nduv, default_date)

        assert result["unit"] == "GHz"

    def test_adds_unit_for_freq(self):
        """Test that freq gets GHz unit."""
        default_date = datetime.datetime(2025, 1, 1)
        nduv = {"name": "freq", "value": 5.0}

        result = _normalize_nduv(nduv, default_date)

        assert result["unit"] == "GHz"

    def test_adds_empty_unit_for_unknown_parameter(self):
        """Test that unknown parameters get empty string unit."""
        default_date = datetime.datetime(2025, 1, 1)
        nduv = {"name": "unknown_param", "value": 42.0}

        result = _normalize_nduv(nduv, default_date)

        assert result["unit"] == ""

    def test_preserves_existing_unit(self):
        """Test that existing unit field is preserved."""
        default_date = datetime.datetime(2025, 1, 1)
        nduv = {"name": "T1", "unit": "ms", "value": 100.0}

        result = _normalize_nduv(nduv, default_date)

        assert result["unit"] == "ms"  # Preserved, not overridden to "us"

    def test_case_insensitive_name_matching(self):
        """Test that parameter name matching is case-insensitive."""
        default_date = datetime.datetime(2025, 1, 1)
        nduv_upper = {"name": "GATE_ERROR", "value": 0.01}
        nduv_lower = {"name": "gate_error", "value": 0.01}
        nduv_mixed = {"name": "Gate_Error", "value": 0.01}

        result_upper = _normalize_nduv(nduv_upper, default_date)
        result_lower = _normalize_nduv(nduv_lower, default_date)
        result_mixed = _normalize_nduv(nduv_mixed, default_date)

        assert result_upper["unit"] == ""
        assert result_lower["unit"] == ""
        assert result_mixed["unit"] == ""


class TestNormalizeProperties:
    """Tests for _normalize_properties function."""

    def test_adds_missing_top_level_fields(self):
        """Test that missing top-level fields are added."""
        props = {}
        default_date = datetime.datetime(2025, 1, 1, 12, 0, 0)

        result = _normalize_properties(props, default_date)

        assert result["backend_name"] == "custom_backend"
        assert result["backend_version"] == "1.0.0"
        assert result["last_update_date"] == default_date
        assert result["general"] == []
        assert result["gates"] == []
        assert result["qubits"] == []

    def test_uses_current_time_when_no_default_date(self):
        """Test that current time is used when default_date is None."""
        props = {}
        before = datetime.datetime.now()

        result = _normalize_properties(props, None)

        after = datetime.datetime.now()
        assert before <= result["last_update_date"] <= after

    def test_preserves_existing_top_level_fields(self):
        """Test that existing top-level fields are preserved."""
        props = {
            "backend_name": "my_backend",
            "backend_version": "2.0.0",
            "last_update_date": datetime.datetime(2024, 1, 1),
        }
        default_date = datetime.datetime(2025, 1, 1)

        result = _normalize_properties(props, default_date)

        assert result["backend_name"] == "my_backend"
        assert result["backend_version"] == "2.0.0"
        assert result["last_update_date"] == datetime.datetime(2024, 1, 1)

    def test_normalizes_qubits(self):
        """Test that qubit parameters are normalized."""
        default_date = datetime.datetime(2025, 1, 1)
        props = {
            "qubits": [
                [{"name": "T1", "value": 100.0}],
                [{"name": "T2", "value": 80.0}],
            ]
        }

        result = _normalize_properties(props, default_date)

        assert len(result["qubits"]) == 2
        assert result["qubits"][0][0]["unit"] == "us"
        assert result["qubits"][0][0]["date"] == default_date
        assert result["qubits"][1][0]["unit"] == "us"
        assert result["qubits"][1][0]["date"] == default_date

    def test_normalizes_gate_parameters(self):
        """Test that gate parameters are normalized."""
        default_date = datetime.datetime(2025, 1, 1)
        props = {
            "gates": [
                {
                    "gate": "sx",
                    "qubits": [0],
                    "parameters": [
                        {"name": "gate_error", "value": 0.01},
                        {"name": "gate_length", "value": 35.0},
                    ],
                }
            ]
        }

        result = _normalize_properties(props, default_date)

        assert len(result["gates"]) == 1
        assert len(result["gates"][0]["parameters"]) == 2
        assert result["gates"][0]["parameters"][0]["unit"] == ""
        assert result["gates"][0]["parameters"][0]["date"] == default_date
        assert result["gates"][0]["parameters"][1]["unit"] == "ns"
        assert result["gates"][0]["parameters"][1]["date"] == default_date

    def test_normalizes_general_parameters(self):
        """Test that general parameters are normalized."""
        default_date = datetime.datetime(2025, 1, 1)
        props = {
            "general": [
                {"name": "frequency", "value": 5.0},
                {"name": "unknown", "value": 42.0},
            ]
        }

        result = _normalize_properties(props, default_date)

        assert len(result["general"]) == 2
        assert result["general"][0]["unit"] == "GHz"
        assert result["general"][0]["date"] == default_date
        assert result["general"][1]["unit"] == ""
        assert result["general"][1]["date"] == default_date

    def test_does_not_mutate_input(self):
        """Test that input dictionary is not mutated."""
        props = {
            "qubits": [[{"name": "T1", "value": 100.0}]],
        }
        props_copy = {
            "qubits": [[{"name": "T1", "value": 100.0}]],
        }
        default_date = datetime.datetime(2025, 1, 1)

        _normalize_properties(props, default_date)

        # Original should be unchanged (no unit/date added)
        assert "unit" not in props["qubits"][0][0]
        assert "date" not in props["qubits"][0][0]
        assert props == props_copy

    def test_handles_empty_qubits_list(self):
        """Test that empty qubits list is handled."""
        props = {"qubits": []}
        default_date = datetime.datetime(2025, 1, 1)

        result = _normalize_properties(props, default_date)

        assert result["qubits"] == []

    def test_handles_empty_gates_list(self):
        """Test that empty gates list is handled."""
        props = {"gates": []}
        default_date = datetime.datetime(2025, 1, 1)

        result = _normalize_properties(props, default_date)

        assert result["gates"] == []

    def test_handles_gates_without_parameters(self):
        """Test that gates without parameters are handled."""
        default_date = datetime.datetime(2025, 1, 1)
        props = {
            "gates": [
                {
                    "gate": "id",
                    "qubits": [0],
                    # No parameters field
                }
            ]
        }

        result = _normalize_properties(props, default_date)

        assert len(result["gates"]) == 1
        assert result["gates"][0]["parameters"] == []

    def test_handles_missing_qubits_field(self):
        """Test that missing qubits field is added as empty list."""
        props = {
            "backend_name": "test",
            "gates": [],
        }
        default_date = datetime.datetime(2025, 1, 1)

        result = _normalize_properties(props, default_date)

        assert result["qubits"] == []
        # Verify it can be converted to BackendProperties
        backend_props = BackendProperties.from_dict(result)
        assert backend_props.qubits == []


class TestCreateBackendFromProperties:
    """Tests for create_backend_from_properties function."""

    def test_infers_qubits_from_qubits_list(self):
        """Test that qubit count is inferred from qubits list length."""
        props = {
            "qubits": [
                [{"name": "T1", "value": 100.0}],
                [{"name": "T1", "value": 120.0}],
                [{"name": "T1", "value": 110.0}],
            ]
        }

        backend = create_backend_from_properties(props)

        assert backend.num_qubits == 3

    def test_overrides_qubit_count(self):
        """Test that qubit count can be overridden."""
        props = {
            "qubits": [[{"name": "T1", "value": 100.0}]],  # Would infer 1
        }

        backend = create_backend_from_properties(props, n_qubits=120)

        assert backend.num_qubits == 120

    def test_creates_valid_backend(self):
        """Test that created backend is valid and can be used."""
        props = {
            "qubits": [
                [{"name": "T1", "value": 100.0, "unit": "us"}],
                [{"name": "T2", "value": 80.0, "unit": "us"}],
            ],
            "gates": [
                {
                    "gate": "sx",
                    "qubits": [0],
                    "parameters": [
                        {"name": "gate_error", "value": 0.01, "unit": ""},
                        {"name": "gate_length", "value": 35.0, "unit": "ns"},
                    ],
                }
            ],
        }

        backend = create_backend_from_properties(props)

        assert backend.num_qubits == 2
        assert backend._properties is not None
        assert isinstance(backend._properties, BackendProperties)

    def test_backend_properties_are_populated(self):
        """Test that backend properties are correctly populated."""
        default_date = datetime.datetime(2025, 1, 1, 12, 0, 0)
        props = {
            "backend_name": "test_backend",
            "backend_version": "1.5.0",
            "qubits": [
                [{"name": "T1", "value": 100.0, "unit": "us", "date": default_date}],
                [{"name": "T1", "value": 120.0, "unit": "us", "date": default_date}],
            ],
            "gates": [
                {
                    "gate": "sx",
                    "qubits": [0],
                    "parameters": [
                        {
                            "name": "gate_error",
                            "value": 0.01,
                            "unit": "",
                            "date": default_date,
                        }
                    ],
                }
            ],
        }

        backend = create_backend_from_properties(props, default_date=default_date)

        assert backend._properties is not None
        # Verify properties were converted successfully
        assert backend._properties.backend_name == "test_backend"
        assert backend._properties.backend_version == "1.5.0"

    def test_uses_provided_default_date(self):
        """Test that provided default_date is used for missing dates."""
        default_date = datetime.datetime(2025, 6, 15, 10, 30, 0)
        props = {
            "qubits": [
                [{"name": "T1", "value": 100.0}],  # No date
                [{"name": "T1", "value": 120.0}],  # Need at least 2 qubits
            ],
            "gates": [
                {
                    "gate": "sx",
                    "qubits": [0],
                    "parameters": [{"name": "gate_error", "value": 0.01}],  # No date
                }
            ],
        }

        backend = create_backend_from_properties(props, default_date=default_date)

        # Verify dates were added
        normalized = _normalize_properties(props, default_date)
        assert normalized["qubits"][0][0]["date"] == default_date
        assert normalized["gates"][0]["parameters"][0]["date"] == default_date

    def test_complex_properties_dictionary(self):
        """Test with a complex, realistic properties dictionary."""
        default_date = datetime.datetime(2025, 1, 1)
        props = {
            "backend_name": "complex_backend",
            "backend_version": "2.0.0",
            "qubits": [
                [{"name": "T1", "value": 100.0}, {"name": "T2", "value": 80.0}],
                [{"name": "T1", "value": 120.0}, {"name": "T2", "value": 90.0}],
            ],
            "gates": [
                {
                    "gate": "sx",
                    "qubits": [0],
                    "parameters": [
                        {"name": "gate_error", "value": 0.01},
                        {"name": "gate_length", "value": 35.0},
                    ],
                },
                {
                    "gate": "cx",
                    "qubits": [0, 1],
                    "parameters": [
                        {"name": "gate_error", "value": 0.05},
                        {"name": "gate_length", "value": 250.0},
                    ],
                },
            ],
            "general": [{"name": "frequency", "value": 5.0}],
        }

        backend = create_backend_from_properties(props, default_date=default_date)

        assert backend.num_qubits == 2
        assert backend._properties is not None
        # Verify it can be converted successfully
        assert isinstance(backend._properties, BackendProperties)

    def test_raises_error_when_no_qubits_and_no_n_qubits(self):
        """Test that ValueError is raised when qubits is empty and n_qubits is None."""
        import pytest

        props = {"backend_name": "test", "gates": []}

        with pytest.raises(ValueError, match="n_qubits must be provided"):
            create_backend_from_properties(props)

    def test_raises_error_when_n_qubits_is_zero(self):
        """Test that ValueError is raised when n_qubits is 0."""
        import pytest

        props = {"backend_name": "test"}

        with pytest.raises(ValueError, match="n_qubits must be at least 1"):
            create_backend_from_properties(props, n_qubits=0)
