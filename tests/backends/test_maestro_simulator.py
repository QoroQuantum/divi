# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from divi.backends._maestro_simulator import MaestroSimulator, _strip_measurements
from tests.backends import circuit_runner_contracts as contracts
from tests.backends.circuit_runner_contracts import QASM_DEPTH_2, QASM_DEPTH_3

# ---------------------------------------------------------------------------
# Helpers — build a fake ``maestro`` module that MaestroSimulator can import
# ---------------------------------------------------------------------------


def _make_fake_maestro(mocker, counts=None, expvals=None):
    """Return a mock ``maestro`` module with ``simple_execute`` and circuit API."""
    maestro = mocker.MagicMock()

    # Enum-like objects for config resolution
    maestro.SimulatorType = {"QCSim": "QCSim", "Gpu": "Gpu"}
    maestro.SimulationType = {
        "Statevector": "Statevector",
        "MPS": "MPS",
        "MatrixProductState": "MatrixProductState",
    }

    # Sampling
    if counts is None:
        counts = {"00": 2500, "11": 2500}
    maestro.simple_execute.return_value = {"counts": counts}

    # Expval — maestro returns {"expectation_values": [...], ...}
    if expvals is None:
        expvals = [0.5, -0.3]
    maestro.simple_estimate.return_value = {"expectation_values": expvals}

    return maestro


def _make_simulator(mocker, fake_maestro, **kwargs):
    """Instantiate MaestroSimulator with a pre-injected fake maestro module."""
    mocker.patch("divi.backends._maestro_simulator.maestro", fake_maestro)
    return MaestroSimulator(**kwargs)


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


class TestImportGuard:
    def test_import_error_without_maestro(self, mocker):
        """MaestroSimulator raises a helpful ImportError when maestro is missing."""
        mocker.patch("divi.backends._maestro_simulator.maestro", None)
        with pytest.raises(ImportError, match="qoro-maestro is required"):
            MaestroSimulator()


# ---------------------------------------------------------------------------
# Properties & defaults
# ---------------------------------------------------------------------------


class TestProperties:
    def test_supports_expval(self, mocker):
        sim = _make_simulator(mocker, _make_fake_maestro(mocker))
        assert sim.supports_expval is True

    def test_is_async(self, mocker):
        sim = _make_simulator(mocker, _make_fake_maestro(mocker))
        assert sim.is_async is False

    def test_default_shots(self, mocker):
        sim = _make_simulator(mocker, _make_fake_maestro(mocker))
        assert sim.shots == 5000

    def test_custom_shots(self, mocker):
        sim = _make_simulator(mocker, _make_fake_maestro(mocker), shots=1024)
        assert sim.shots == 1024

    def test_config_storage(self, mocker):
        sim = _make_simulator(
            mocker,
            _make_fake_maestro(mocker),
            simulator_type="QCSim",
            simulation_type="MPS",
            max_bond_dimension=64,
            singular_value_threshold=1e-8,
            use_double_precision=True,
        )
        assert sim.simulator_type == "QCSim"
        assert sim.simulation_type == "MPS"
        assert sim.max_bond_dimension == 64
        assert sim.singular_value_threshold == 1e-8
        assert sim.use_double_precision is True


# ---------------------------------------------------------------------------
# Automatic MPS threshold
# ---------------------------------------------------------------------------

# Minimal QASM templates for qubit-count tests (no real gates needed).
_QASM_SMALL = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[10];\ncreg c[10];\n'
    "h q[0];\nmeasure q[0] -> c[0];\n"
)
_QASM_LARGE = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[25];\ncreg c[25];\n'
    "h q[0];\nmeasure q[0] -> c[0];\n"
)


class TestMpsThreshold:
    """Automatic simulation type selection based on qubit count."""

    def test_default_threshold(self, mocker):
        sim = _make_simulator(mocker, _make_fake_maestro(mocker))
        assert sim.mps_qubit_threshold == 22

    def test_custom_threshold(self, mocker):
        sim = _make_simulator(
            mocker, _make_fake_maestro(mocker), mps_qubit_threshold=10
        )
        assert sim.mps_qubit_threshold == 10

    def test_below_threshold_no_simulation_type(self, mocker):
        """Circuits below the threshold should not set simulation_type."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake)

        sim.submit_circuits({"c0": _QASM_SMALL})

        call_kwargs = fake.simple_execute.call_args[1]
        assert "simulation_type" not in call_kwargs

    def test_above_threshold_selects_mps(self, mocker):
        """Circuits above the threshold should auto-select MPS."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake)

        sim.submit_circuits({"c0": _QASM_LARGE})

        call_kwargs = fake.simple_execute.call_args[1]
        assert call_kwargs["simulation_type"] == "MatrixProductState"

    def test_explicit_simulation_type_overrides_threshold(self, mocker):
        """An explicit simulation_type should not be overridden by the threshold."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake, simulation_type="Statevector")

        sim.submit_circuits({"c0": _QASM_LARGE})

        call_kwargs = fake.simple_execute.call_args[1]
        assert call_kwargs["simulation_type"] == "Statevector"

    def test_threshold_applies_to_expval_mode(self, mocker):
        """MPS threshold also applies in expectation value mode."""
        fake = _make_fake_maestro(mocker, expvals=[0.5])
        sim = _make_simulator(mocker, fake)

        sim.submit_circuits({"c0": _QASM_LARGE}, ham_ops="Z" + "I" * 24)

        call_kwargs = fake.simple_estimate.call_args[1]
        assert call_kwargs["simulation_type"] == "MatrixProductState"

    def test_custom_threshold_respected(self, mocker):
        """A custom threshold of 5 should trigger MPS for a 10-qubit circuit."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake, mps_qubit_threshold=5)

        sim.submit_circuits({"c0": _QASM_SMALL})

        call_kwargs = fake.simple_execute.call_args[1]
        assert call_kwargs["simulation_type"] == "MatrixProductState"

    def test_at_threshold_no_mps(self, mocker):
        """Circuits exactly at the threshold should NOT trigger MPS (> not >=)."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake, mps_qubit_threshold=10)

        sim.submit_circuits({"c0": _QASM_SMALL})

        call_kwargs = fake.simple_execute.call_args[1]
        assert "simulation_type" not in call_kwargs

    def test_auto_mps_sets_default_bond_dimension(self, mocker):
        """Auto-MPS should set bond dimension to 64 when not explicitly configured."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake)

        sim.submit_circuits({"c0": _QASM_LARGE})

        call_kwargs = fake.simple_execute.call_args[1]
        assert call_kwargs["max_bond_dimension"] == 64

    def test_explicit_bond_dimension_not_overridden(self, mocker):
        """User-specified bond dimension should not be overridden by auto-MPS."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake, max_bond_dimension=128)

        sim.submit_circuits({"c0": _QASM_LARGE})

        call_kwargs = fake.simple_execute.call_args[1]
        assert call_kwargs["max_bond_dimension"] == 128

    def test_no_auto_bond_dimension_below_threshold(self, mocker):
        """Below threshold, bond dimension should not be set unless explicit."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake)

        sim.submit_circuits({"c0": _QASM_SMALL})

        call_kwargs = fake.simple_execute.call_args[1]
        assert "max_bond_dimension" not in call_kwargs


# ---------------------------------------------------------------------------
# Sampling mode
# ---------------------------------------------------------------------------


class TestSamplingSubmission:
    def test_basic_sampling(self, mocker):
        """submit_circuits in sampling mode returns correct ExecutionResult."""
        fake = _make_fake_maestro(mocker, counts={"00": 3000, "11": 2000})
        sim = _make_simulator(mocker, fake)

        result = sim.submit_circuits({"c0": QASM_DEPTH_2, "c1": QASM_DEPTH_3})

        assert result.results is not None
        assert len(result.results) == 2
        assert result.results[0]["label"] == "c0"
        # Palindromic bitstrings are unchanged by reversal
        assert result.results[0]["results"] == {"00": 3000, "11": 2000}
        assert result.results[1]["label"] == "c1"

    def test_bitstring_reversal(self, mocker):
        """Bitstrings are reversed from maestro big-endian to Qiskit little-endian."""
        fake = _make_fake_maestro(mocker, counts={"100": 70, "001": 30})
        sim = _make_simulator(mocker, fake)

        result = sim.submit_circuits({"c0": QASM_DEPTH_2})

        # "100" (maestro) -> "001" (qiskit), "001" -> "100"
        assert result.results[0]["results"] == {"001": 70, "100": 30}

    def test_config_passthrough(self, mocker):
        """Non-None config options are passed through to simple_execute."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(
            mocker,
            fake,
            simulator_type="QCSim",
            simulation_type="MPS",
            max_bond_dimension=32,
            singular_value_threshold=1e-6,
            use_double_precision=True,
        )

        sim.submit_circuits({"c0": QASM_DEPTH_2})

        call_kwargs = fake.simple_execute.call_args[1]
        assert call_kwargs["shots"] == 5000
        assert call_kwargs["simulator_type"] == "QCSim"
        assert call_kwargs["simulation_type"] == "MPS"
        assert call_kwargs["max_bond_dimension"] == 32
        assert call_kwargs["singular_value_threshold"] == 1e-6
        assert call_kwargs["use_double_precision"] is True

    def test_none_config_not_passed(self, mocker):
        """None config options are not passed to simple_execute."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake)

        sim.submit_circuits({"c0": QASM_DEPTH_2})

        call_kwargs = fake.simple_execute.call_args[1]
        assert "simulator_type" not in call_kwargs
        assert "simulation_type" not in call_kwargs
        assert "max_bond_dimension" not in call_kwargs
        assert "singular_value_threshold" not in call_kwargs
        assert "use_double_precision" not in call_kwargs


# ---------------------------------------------------------------------------
# Expectation value mode
# ---------------------------------------------------------------------------


class TestExpvalSubmission:
    def test_basic_expval(self, mocker):
        """submit_circuits with ham_ops returns expectation values as {op: val} dict."""
        fake = _make_fake_maestro(mocker, expvals=[0.5, -0.3])
        sim = _make_simulator(mocker, fake)

        result = sim.submit_circuits({"c0": QASM_DEPTH_2}, ham_ops="ZI;IZ")

        assert result.results is not None
        assert len(result.results) == 1
        assert result.results[0]["label"] == "c0"
        assert result.results[0]["results"] == {"ZI": 0.5, "IZ": -0.3}

    def test_expval_calls_simple_estimate_not_execute(self, mocker):
        """Expval mode must call simple_estimate, never simple_execute."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake)

        sim.submit_circuits({"c0": QASM_DEPTH_2}, ham_ops="ZI;IZ")

        fake.simple_estimate.assert_called_once()
        fake.simple_execute.assert_not_called()

    def test_expval_passes_observables(self, mocker):
        """Observables string is forwarded to simple_estimate."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake)

        sim.submit_circuits({"c0": QASM_DEPTH_2}, ham_ops="ZI;IZ")

        assert fake.simple_estimate.call_args[1]["observables"] == "ZI;IZ"

    def test_expval_strips_measurements(self, mocker):
        """Measurement instructions are stripped so they don't collapse
        the statevector before expectation values are computed."""
        fake = _make_fake_maestro(mocker)
        fake.simple_estimate.side_effect = [
            {"expectation_values": [0.5]},
            {"expectation_values": [0.8]},
        ]
        sim = _make_simulator(mocker, fake)

        sim.submit_circuits({"c0": QASM_DEPTH_2, "c1": QASM_DEPTH_3}, ham_ops="ZI")

        for call in fake.simple_estimate.call_args_list:
            assert "measure" not in call[0][0]

    def test_expval_preserves_circuit_body(self, mocker):
        """Stripping measurements must leave the circuit gates intact."""
        fake = _make_fake_maestro(mocker, expvals=[0.5])
        sim = _make_simulator(mocker, fake)

        sim.submit_circuits({"c0": QASM_DEPTH_2}, ham_ops="ZI")

        called_qasm = fake.simple_estimate.call_args[0][0]
        # The H gate from QASM_DEPTH_2 must survive stripping
        assert "h q[0]" in called_qasm
        # Header must survive
        assert "OPENQASM 2.0" in called_qasm
        assert "qreg q[2]" in called_qasm

    def test_expval_zips_ops_to_values(self, mocker):
        """Each Pauli operator maps to the corresponding expectation value."""
        fake = _make_fake_maestro(mocker, expvals=[0.1, 0.2, 0.3])
        sim = _make_simulator(mocker, fake)

        result = sim.submit_circuits({"c0": QASM_DEPTH_2}, ham_ops="ZI;IX;YY")

        assert result.results[0]["results"] == {"ZI": 0.1, "IX": 0.2, "YY": 0.3}

    def test_expval_config_passthrough(self, mocker):
        """Non-None config options are passed through to simple_estimate."""
        fake = _make_fake_maestro(mocker, expvals=[0.5])
        sim = _make_simulator(
            mocker,
            fake,
            simulator_type="QCSim",
            simulation_type="MPS",
            max_bond_dimension=32,
        )

        sim.submit_circuits({"c0": QASM_DEPTH_2}, ham_ops="ZI")

        call_kwargs = fake.simple_estimate.call_args[1]
        assert call_kwargs["simulator_type"] == "QCSim"
        assert call_kwargs["simulation_type"] == "MPS"
        assert call_kwargs["max_bond_dimension"] == 32

    def test_sampling_retains_measurements(self, mocker):
        """Sampling mode must NOT strip measurements — they are needed."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake)

        sim.submit_circuits({"c0": QASM_DEPTH_2})

        called_qasm = fake.simple_execute.call_args[0][0]
        assert "measure" in called_qasm

    def test_circuit_ham_map_routing(self, mocker):
        """circuit_ham_map routes correct observables to each circuit."""
        fake = _make_fake_maestro(mocker)
        fake.simple_estimate.side_effect = [
            {"expectation_values": [0.5]},
            {"expectation_values": [0.8]},
        ]
        sim = _make_simulator(mocker, fake)

        result = sim.submit_circuits(
            {"c0": QASM_DEPTH_2, "c1": QASM_DEPTH_3},
            ham_ops="ZI|XX",
            circuit_ham_map=[[0, 1], [1, 2]],
        )

        assert result.results[0]["results"] == {"ZI": 0.5}
        assert result.results[1]["results"] == {"XX": 0.8}

        calls = fake.simple_estimate.call_args_list
        assert calls[0][1]["observables"] == "ZI"
        assert calls[1][1]["observables"] == "XX"

    def test_circuit_ham_map_fallback(self, mocker):
        """Circuits not in any group fall back to full ham_ops string."""
        fake = _make_fake_maestro(mocker)
        fake.simple_estimate.side_effect = [
            {"expectation_values": [0.5]},
            {"expectation_values": [0.5, -0.3]},
        ]
        sim = _make_simulator(mocker, fake)

        sim.submit_circuits(
            {"c0": QASM_DEPTH_2, "c1": QASM_DEPTH_3},
            ham_ops="ZI|XX",
            circuit_ham_map=[[0, 1]],
        )

        calls = fake.simple_estimate.call_args_list
        assert calls[0][1]["observables"] == "ZI"
        # Circuit 1 not in any group — falls back to full ham_ops
        assert calls[1][1]["observables"] == "ZI|XX"


# ---------------------------------------------------------------------------
# _strip_measurements
# ---------------------------------------------------------------------------


class TestStripMeasurements:
    """Verify _strip_measurements preserves gates and removes only measurements."""

    def test_removes_all_measure_lines(self):
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\n'
            "h q[0];\ncx q[0],q[1];\nrz(0.5) q[2];\n"
            "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\n"
        )
        result = _strip_measurements(qasm)
        assert "measure" not in result
        assert "h q[0]" in result
        assert "cx q[0],q[1]" in result
        assert "rz(0.5) q[2]" in result

    def test_no_measurements_unchanged(self):
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
            "h q[0];\ncx q[0],q[1];\n"
        )
        assert _strip_measurements(qasm) == qasm

    def test_preserves_creg(self):
        """creg declarations must survive even though measurements are stripped."""
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
            "h q[0];\nmeasure q[0] -> c[0];\n"
        )
        result = _strip_measurements(qasm)
        assert "creg c[2]" in result


# ---------------------------------------------------------------------------
# Depth tracking contracts
# ---------------------------------------------------------------------------


class TestDepthContracts:
    """Run all depth-tracking contracts from circuit_runner_contracts."""

    @pytest.fixture()
    def _fake_maestro(self, mocker):
        return _make_fake_maestro(mocker)

    @pytest.fixture()
    def _mocker(self, mocker):
        return mocker

    def _sim(self, mocker, fake, **kwargs):
        return _make_simulator(mocker, fake, **kwargs)

    def test_disabled(self, _mocker, _fake_maestro):
        runner = self._sim(_mocker, _fake_maestro, track_depth=False)
        contracts.verify_depth_tracking_disabled(
            runner, {"c0": QASM_DEPTH_2, "c1": QASM_DEPTH_3}
        )

    def test_records(self, _mocker, _fake_maestro):
        runner = self._sim(_mocker, _fake_maestro, track_depth=True)
        contracts.verify_depth_tracking_records(
            runner, {"c0": QASM_DEPTH_2, "c1": QASM_DEPTH_3}, [2, 3]
        )

    def test_accumulates(self, _mocker, _fake_maestro):
        runner = self._sim(_mocker, _fake_maestro, track_depth=True)
        contracts.verify_depth_history_accumulates(
            runner,
            {"c0": QASM_DEPTH_2},
            {"c1": QASM_DEPTH_3},
        )

    def test_clear(self, _mocker, _fake_maestro):
        runner = self._sim(_mocker, _fake_maestro, track_depth=True)
        contracts.verify_clear_depth_history(runner, {"c0": QASM_DEPTH_2})

    def test_returns_copy(self, _mocker, _fake_maestro):
        runner = self._sim(_mocker, _fake_maestro, track_depth=True)
        contracts.verify_depth_history_returns_copy(runner, {"c0": QASM_DEPTH_2})

    def test_std_zero_single(self, _mocker, _fake_maestro):
        runner = self._sim(_mocker, _fake_maestro, track_depth=True)
        contracts.verify_std_depth_zero_for_single_value(runner, {"c0": QASM_DEPTH_2})
