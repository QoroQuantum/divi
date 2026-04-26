# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import math

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

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

    def test_raises_error_when_no_qubits_and_no_n_qubits(self):
        """Test that ValueError is raised when qubits is empty and n_qubits is None."""
        props = {"backend_name": "test", "gates": []}

        with pytest.raises(ValueError, match="n_qubits must be provided"):
            create_backend_from_properties(props)

    def test_raises_error_when_n_qubits_is_zero(self):
        """Test that ValueError is raised when n_qubits is 0."""
        props = {"backend_name": "test"}

        with pytest.raises(ValueError, match="n_qubits must be at least 1"):
            create_backend_from_properties(props, n_qubits=0)


class TestTargetReflectsCalibration:
    """User-supplied calibration must reach ``backend.target`` so the
    transpiler and AerSimulator/NoiseModel actually consume it."""

    def test_t1_t2_reach_target_qubit_properties(self):
        """T1/T2 from the input dict end up on ``target.qubit_properties[i]``."""
        props = {
            "qubits": [
                [
                    {"name": "T1", "value": 100.0, "unit": "us"},
                    {"name": "T2", "value": 80.0, "unit": "us"},
                ],
                [
                    {"name": "T1", "value": 200.0, "unit": "us"},
                    {"name": "T2", "value": 150.0, "unit": "us"},
                ],
            ],
        }

        backend = create_backend_from_properties(props)

        assert backend.target.qubit_properties[0].t1 == pytest.approx(100e-6)
        assert backend.target.qubit_properties[0].t2 == pytest.approx(80e-6)
        assert backend.target.qubit_properties[1].t1 == pytest.approx(200e-6)
        assert backend.target.qubit_properties[1].t2 == pytest.approx(150e-6)

    def test_frequency_reaches_target_qubit_properties(self):
        """Qubit frequency in GHz reaches the target in Hz."""
        props = {
            "qubits": [
                [{"name": "frequency", "value": 5.0, "unit": "GHz"}],
                [{"name": "frequency", "value": 5.2, "unit": "GHz"}],
            ],
        }

        backend = create_backend_from_properties(props)

        assert backend.target.qubit_properties[0].frequency == pytest.approx(5e9)
        assert backend.target.qubit_properties[1].frequency == pytest.approx(5.2e9)

    def test_gate_calibration_reaches_target(self):
        """gate_error and gate_length reach the corresponding InstructionProperties."""
        props = {
            "qubits": [
                [{"name": "T1", "value": 100.0, "unit": "us"}],
                [{"name": "T1", "value": 100.0, "unit": "us"}],
            ],
            "gates": [
                {
                    "gate": "sx",
                    "qubits": [0],
                    "parameters": [
                        {"name": "gate_error", "value": 0.01},
                        {"name": "gate_length", "value": 35.0, "unit": "ns"},
                    ],
                },
                {
                    "gate": "cx",
                    "qubits": [0, 1],
                    "parameters": [
                        {"name": "gate_error", "value": 0.05},
                        {"name": "gate_length", "value": 250.0, "unit": "ns"},
                    ],
                },
            ],
        }

        backend = create_backend_from_properties(props)

        assert backend.target["sx"][(0,)].error == pytest.approx(0.01)
        assert backend.target["sx"][(0,)].duration == pytest.approx(35e-9)
        assert backend.target["cx"][(0, 1)].error == pytest.approx(0.05)
        assert backend.target["cx"][(0, 1)].duration == pytest.approx(250e-9)

    def test_readout_calibration_reaches_target_measure(self):
        """readout_error and readout_length end up on the ``measure`` instruction."""
        props = {
            "qubits": [
                [
                    {"name": "T1", "value": 100.0, "unit": "us"},
                    {"name": "readout_error", "value": 0.02},
                    {"name": "readout_length", "value": 1000.0, "unit": "ns"},
                ],
            ],
        }

        backend = create_backend_from_properties(props)

        assert backend.target["measure"][(0,)].error == pytest.approx(0.02)
        assert backend.target["measure"][(0,)].duration == pytest.approx(1000e-9)

    def test_unspecified_gates_are_absent_from_target(self):
        """Only gates listed in ``properties["gates"]`` plus the universal
        infrastructure ops (``measure``, ``reset``, ``delay``) appear on
        the target — no leftover random calibration from
        ``GenericBackendV2``'s default basis."""
        props = {
            "qubits": [[{"name": "T1", "value": 100.0, "unit": "us"}]],
            "gates": [
                {
                    "gate": "sx",
                    "qubits": [0],
                    "parameters": [
                        {"name": "gate_error", "value": 0.01},
                        {"name": "gate_length", "value": 35.0, "unit": "ns"},
                    ],
                }
            ],
        }

        backend = create_backend_from_properties(props)

        op_names = set(backend.target.operation_names)
        assert op_names == {"sx", "measure", "reset", "delay"}

    def test_qubits_beyond_qubits_list_get_default_properties(self):
        """When n_qubits exceeds len(qubits), extra qubits get blank QubitProperties."""
        props = {
            "qubits": [[{"name": "T1", "value": 100.0, "unit": "us"}]],
        }

        backend = create_backend_from_properties(props, n_qubits=3)

        assert backend.target.qubit_properties[0].t1 == pytest.approx(100e-6)
        # Extra qubits — no T1/T2 specified, should fall back to defaults (None)
        assert backend.target.qubit_properties[1].t1 is None
        assert backend.target.qubit_properties[2].t1 is None

    def test_unknown_gate_names_are_skipped(self):
        """A gate whose name isn't in qiskit's standard gate registry is dropped."""
        props = {
            "qubits": [[{"name": "T1", "value": 100.0, "unit": "us"}]],
            "gates": [
                {
                    "gate": "definitely_not_a_real_gate",
                    "qubits": [0],
                    "parameters": [{"name": "gate_error", "value": 0.5}],
                }
            ],
        }

        backend = create_backend_from_properties(props)

        assert "definitely_not_a_real_gate" not in backend.target.operation_names


class TestNoiseModelRoundtrip:
    """The Target produced by the rewrite must be readable by Aer's
    :func:`NoiseModel.from_backend` so the user's calibration can drive a
    simulation. This is the test that would have caught the original bug."""

    def test_noise_model_picks_up_user_gate_errors(self):
        """User-supplied gate errors propagate into ``NoiseModel.from_backend``."""
        props = {
            "qubits": [
                [{"name": "T1", "value": 100.0, "unit": "us"}],
                [{"name": "T1", "value": 100.0, "unit": "us"}],
            ],
            "gates": [
                {
                    "gate": "sx",
                    "qubits": [0],
                    "parameters": [
                        {"name": "gate_error", "value": 0.01},
                        {"name": "gate_length", "value": 35.0, "unit": "ns"},
                    ],
                },
                {
                    "gate": "cx",
                    "qubits": [0, 1],
                    "parameters": [
                        {"name": "gate_error", "value": 0.05},
                        {"name": "gate_length", "value": 250.0, "unit": "ns"},
                    ],
                },
            ],
        }

        backend = create_backend_from_properties(props)
        nm = NoiseModel.from_backend(backend)

        # Aer derives noise channels for every gate it found a non-zero
        # error on. Our user-specified gates must all show up.
        noisy = set(nm.noise_instructions)
        assert {"sx", "cx"}.issubset(
            noisy
        ), f"Noise model missed user-specified gates: got {sorted(noisy)}"

    def test_user_calibration_drives_simulation_outcomes(self):
        """End-to-end: a backend built from a high-error properties dict
        produces a *visibly noisier* Bell-state simulation than one built
        from a low-error dict — proving the calibration actually flows
        through to the AerSimulator."""

        def make_props(gate_err: float) -> dict:
            return {
                "qubits": [
                    [
                        {"name": "T1", "value": 100.0, "unit": "us"},
                        {"name": "T2", "value": 80.0, "unit": "us"},
                    ],
                    [
                        {"name": "T1", "value": 100.0, "unit": "us"},
                        {"name": "T2", "value": 80.0, "unit": "us"},
                    ],
                ],
                "gates": [
                    {
                        "gate": "sx",
                        "qubits": [0],
                        "parameters": [
                            {"name": "gate_error", "value": gate_err},
                            {"name": "gate_length", "value": 35.0, "unit": "ns"},
                        ],
                    },
                    {
                        "gate": "sx",
                        "qubits": [1],
                        "parameters": [
                            {"name": "gate_error", "value": gate_err},
                            {"name": "gate_length", "value": 35.0, "unit": "ns"},
                        ],
                    },
                    {
                        "gate": "rz",
                        "qubits": [0],
                        "parameters": [
                            {"name": "gate_error", "value": 0.0},
                            {"name": "gate_length", "value": 0.0, "unit": "ns"},
                        ],
                    },
                    {
                        "gate": "rz",
                        "qubits": [1],
                        "parameters": [
                            {"name": "gate_error", "value": 0.0},
                            {"name": "gate_length", "value": 0.0, "unit": "ns"},
                        ],
                    },
                    {
                        "gate": "cx",
                        "qubits": [0, 1],
                        "parameters": [
                            {"name": "gate_error", "value": gate_err * 5},
                            {"name": "gate_length", "value": 250.0, "unit": "ns"},
                        ],
                    },
                ],
            }

        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)
        bell.measure_all()

        shots = 8000

        def error_rate(gate_err: float, seed: int) -> float:
            backend = create_backend_from_properties(make_props(gate_err))
            sim = AerSimulator.from_backend(backend, seed_simulator=seed)
            transpiled = transpile(bell, backend=backend, seed_transpiler=seed)
            counts = sim.run(transpiled, shots=shots).result().get_counts()
            wrong = counts.get("01", 0) + counts.get("10", 0)
            return wrong / shots

        clean_rate = error_rate(0.001, seed=42)
        noisy_rate = error_rate(0.05, seed=42)

        # The noisy backend should produce many more |01⟩/|10⟩ outcomes
        # than the clean one. Demand at least a 5× ratio so the test is
        # robust to shot noise on either side.
        assert noisy_rate > 5 * max(clean_rate, 1e-4), (
            f"Noisy backend not noisier than clean: clean={clean_rate:.4f}, "
            f"noisy={noisy_rate:.4f}"
        )

    def test_t1_drives_simulation_decoherence(self):
        """A backend with very short T1 produces excited-state decay during a
        long delay; one with very long T1 does not. This exercises the
        QubitProperties path of the target."""

        def make_props_with_t1(t1_us: float) -> dict:
            return {
                "qubits": [
                    [
                        {"name": "T1", "value": t1_us, "unit": "us"},
                        {"name": "T2", "value": t1_us, "unit": "us"},
                    ],
                ],
                "gates": [
                    {
                        "gate": "x",
                        "qubits": [0],
                        "parameters": [
                            {"name": "gate_error", "value": 0.0},
                            {"name": "gate_length", "value": 35.0, "unit": "ns"},
                        ],
                    },
                ],
            }

        # Prepare |1⟩, wait, measure. With a 1μs delay and T1=0.5μs the
        # excited-state population should decay substantially; with
        # T1=10ms the decay is negligible.
        circ = QuantumCircuit(1, 1)
        circ.x(0)
        circ.delay(1000, 0, unit="ns")  # 1 μs hold
        circ.measure(0, 0)

        shots = 8000

        def excited_pop(t1_us: float, seed: int) -> float:
            backend = create_backend_from_properties(make_props_with_t1(t1_us))
            sim = AerSimulator.from_backend(backend, seed_simulator=seed)
            transpiled = transpile(circ, backend=backend, seed_transpiler=seed)
            counts = sim.run(transpiled, shots=shots).result().get_counts()
            return counts.get("1", 0) / shots

        short_t1_pop = excited_pop(0.5, seed=42)
        long_t1_pop = excited_pop(10000.0, seed=42)

        # With T1 = 0.5 μs and a 1 μs delay, e^{-1/0.5} ≈ 0.135 of the
        # population should remain excited. With T1 = 10 ms it stays ≈ 1.
        # We just need a clear gap.
        assert long_t1_pop - short_t1_pop > 0.3, (
            f"T1 calibration did not affect simulation: "
            f"short_T1={short_t1_pop:.3f}, long_T1={long_t1_pop:.3f}"
        )
        # Sanity check on the short-T1 side — must be far from full
        # excitation, demonstrating the user's T1 actually drove the noise.
        assert (
            short_t1_pop < 0.4
        ), f"Short-T1 backend showed no decoherence: pop={short_t1_pop:.3f}"
        # And it must be near the analytic expectation. exp(-2) ≈ 0.135;
        # accept anything in [0.05, 0.30] to give a wide margin for shot
        # noise and aer's depolarizing-channel approximation.
        assert 0.05 < short_t1_pop < 0.30, (
            f"Short-T1 excited-state fraction outside expected range: "
            f"got {short_t1_pop:.3f}, expected ≈ {math.exp(-2):.3f}"
        )
