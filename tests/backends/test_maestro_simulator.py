# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import fields
from threading import Event, Thread

import pytest

import divi.backends._maestro_simulator as maestro_module
from divi.backends._maestro_simulator import (
    MaestroConfig,
    MaestroSimulator,
    _strip_measurements,
)
from tests.backends import circuit_runner_contracts as contracts
from tests.backends.circuit_runner_contracts import QASM_DEPTH_2, QASM_DEPTH_3

try:
    import maestro as _real_maestro
except ImportError:
    _real_maestro = None

requires_real_maestro = pytest.mark.skipif(
    _real_maestro is None, reason="qoro-maestro is not installed"
)

_BELL_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
    "h q[0];\ncx q[0], q[1];\n"
    "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
)

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

    # ``SimulatorConfig(**kwargs)`` — echo kwargs back as a ``spec=dict`` mock so
    # tests can inspect what was passed.
    def _make_sim_config(**kwargs):
        cfg = mocker.MagicMock(name="SimulatorConfig")
        cfg.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg

    maestro.SimulatorConfig.side_effect = _make_sim_config

    # Sampling
    if counts is None:
        counts = {"00": 2500, "11": 2500}
    maestro.simple_execute.return_value = {"counts": counts}
    maestro.noisy_execute.return_value = {"counts": counts}

    # Expval — maestro returns {"expectation_values": [...], ...}
    if expvals is None:
        expvals = [0.5, -0.3]
    maestro.simple_estimate.return_value = {"expectation_values": expvals}
    maestro.noisy_estimate.return_value = {"expectation_values": expvals}
    maestro.noisy_estimate_montecarlo.return_value = {"expectation_values": expvals}

    # ``QasmToCirc().parse_and_translate(qasm) -> "MaestroCircuit"`` —
    # tag the parser so we can assert it was used (instead of the raw qasm).
    parser = mocker.MagicMock(name="QasmToCirc")
    parser.parse_and_translate.side_effect = lambda qasm: ("maestro_circuit", qasm)
    maestro.QasmToCirc.return_value = parser

    return maestro


def _make_simulator(mocker, fake_maestro, *, config=None, **kwargs):
    """Instantiate MaestroSimulator with a pre-injected fake maestro module."""
    mocker.patch("divi.backends._maestro_simulator.maestro", fake_maestro)
    return MaestroSimulator(config=config, **kwargs)


def _make_noisy_sim(mocker, fake_maestro, *, shots=None, **noise_kwargs):
    """Build a simulator with a fresh ``NoiseModel`` mock baked into its config.

    ``noise_kwargs`` are forwarded to ``MaestroConfig`` (so callers can
    pass ``noise_seed=...``, ``noise_realizations=...``).  Returns
    ``(sim, noise_model)`` so tests can assert the model identity.
    """
    nm = mocker.MagicMock(name="NoiseModel")
    cfg = MaestroConfig(noise_model=nm, **noise_kwargs)
    sim_kwargs: dict = {"config": cfg}
    if shots is not None:
        sim_kwargs["shots"] = shots
    return _make_simulator(mocker, fake_maestro, **sim_kwargs), nm


def _sim_config_call(fake_maestro):
    """Return kwargs passed to the most recent ``SimulatorConfig(**kwargs)`` call."""
    return fake_maestro.SimulatorConfig.call_args.kwargs


def _submit_config_arg(call):
    """Pull the ``config=`` kwarg from a ``simple_execute``/``simple_estimate`` call."""
    return call[1]["config"]


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
# MaestroConfig behaviors
# ---------------------------------------------------------------------------


class TestMaestroConfig:
    def test_defaults(self):
        """Default MaestroConfig matches maestro's SimulatorConfig defaults."""
        cfg = MaestroConfig()
        assert cfg.simulator_type is None
        assert cfg.simulation_type is None
        assert cfg.max_bond_dimension is None
        assert cfg.singular_value_threshold is None
        assert cfg.use_double_precision is False
        assert cfg.disable_optimized_swapping is False
        assert cfg.lookahead_depth == -1
        assert cfg.mps_measure_no_collapse is True
        assert cfg.mps_qubit_threshold == 22

    def test_frozen(self):
        """MaestroConfig is a frozen dataclass — attributes are immutable."""
        cfg = MaestroConfig(simulation_type="Statevector")
        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.simulation_type = "MatrixProductState"

    def test_override_replaces_non_default_fields(self):
        """override() copies only non-default fields from ``other``."""
        base = MaestroConfig(
            simulation_type="Statevector",
            max_bond_dimension=128,
        )
        override = MaestroConfig(max_bond_dimension=256)
        merged = base.override(override)

        assert merged.simulation_type == "Statevector"  # preserved from base
        assert merged.max_bond_dimension == 256  # taken from override

    def test_override_returns_new_instance(self):
        """override() never mutates the original."""
        base = MaestroConfig(simulation_type="Statevector")
        override = MaestroConfig(simulation_type="MatrixProductState")
        merged = base.override(override)

        assert base.simulation_type == "Statevector"
        assert merged is not base
        assert merged.simulation_type == "MatrixProductState"

    def test_rejects_unknown_field(self):
        """Unknown fields raise TypeError — no silent kwarg-dropping."""
        with pytest.raises(TypeError):
            MaestroConfig(bogus_field=1)


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

    def test_default_config(self, mocker):
        """No ``config=`` argument → backend uses a default MaestroConfig."""
        sim = _make_simulator(mocker, _make_fake_maestro(mocker))
        assert sim.config == MaestroConfig()

    def test_custom_config_stored(self, mocker):
        """The ``config`` attribute carries the user-provided MaestroConfig."""
        cfg = MaestroConfig(
            simulator_type="QCSim",
            simulation_type="MatrixProductState",
            max_bond_dimension=64,
            singular_value_threshold=1e-8,
            use_double_precision=True,
        )
        sim = _make_simulator(mocker, _make_fake_maestro(mocker), config=cfg)
        assert sim.config is cfg


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
        assert sim.config.mps_qubit_threshold == 22

    def test_custom_threshold(self, mocker):
        cfg = MaestroConfig(mps_qubit_threshold=10)
        sim = _make_simulator(mocker, _make_fake_maestro(mocker), config=cfg)
        assert sim.config.mps_qubit_threshold == 10

    def test_below_threshold_no_simulation_type(self, mocker):
        """Circuits below the threshold should not set simulation_type."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake)

        sim.submit_circuits({"c0": _QASM_SMALL})

        kwargs = _sim_config_call(fake)
        assert "simulation_type" not in kwargs

    def test_above_threshold_selects_mps(self, mocker):
        """Circuits above the threshold should auto-select MPS."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake)

        sim.submit_circuits({"c0": _QASM_LARGE})

        kwargs = _sim_config_call(fake)
        assert kwargs["simulation_type"] == "MatrixProductState"

    def test_explicit_simulation_type_overrides_threshold(self, mocker):
        """An explicit simulation_type should not be overridden by the threshold."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(
            mocker, fake, config=MaestroConfig(simulation_type="Statevector")
        )

        sim.submit_circuits({"c0": _QASM_LARGE})

        kwargs = _sim_config_call(fake)
        assert kwargs["simulation_type"] == "Statevector"

    def test_threshold_applies_to_expval_mode(self, mocker):
        """MPS threshold also applies in expectation value mode."""
        fake = _make_fake_maestro(mocker, expvals=[0.5])
        sim = _make_simulator(mocker, fake)

        sim.submit_circuits({"c0": _QASM_LARGE}, ham_ops="Z" + "I" * 24)

        kwargs = _sim_config_call(fake)
        assert kwargs["simulation_type"] == "MatrixProductState"

    def test_custom_threshold_respected(self, mocker):
        """A custom threshold of 5 should trigger MPS for a 10-qubit circuit."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake, config=MaestroConfig(mps_qubit_threshold=5))

        sim.submit_circuits({"c0": _QASM_SMALL})

        kwargs = _sim_config_call(fake)
        assert kwargs["simulation_type"] == "MatrixProductState"

    def test_at_threshold_no_mps(self, mocker):
        """Circuits exactly at the threshold should NOT trigger MPS (> not >=)."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(
            mocker, fake, config=MaestroConfig(mps_qubit_threshold=10)
        )

        sim.submit_circuits({"c0": _QASM_SMALL})

        kwargs = _sim_config_call(fake)
        assert "simulation_type" not in kwargs

    def test_auto_mps_sets_default_bond_dimension(self, mocker):
        """Auto-MPS should set bond dimension to 64 when not explicitly configured."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake)

        sim.submit_circuits({"c0": _QASM_LARGE})

        kwargs = _sim_config_call(fake)
        assert kwargs["max_bond_dimension"] == 64

    def test_explicit_bond_dimension_not_overridden(self, mocker):
        """User-specified bond dimension should not be overridden by auto-MPS."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(
            mocker, fake, config=MaestroConfig(max_bond_dimension=128)
        )

        sim.submit_circuits({"c0": _QASM_LARGE})

        kwargs = _sim_config_call(fake)
        assert kwargs["max_bond_dimension"] == 128

    def test_no_auto_bond_dimension_below_threshold(self, mocker):
        """Below threshold, bond dimension should not be set unless explicit."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake)

        sim.submit_circuits({"c0": _QASM_SMALL})

        kwargs = _sim_config_call(fake)
        assert "max_bond_dimension" not in kwargs


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

    def test_shots_and_config_passed_to_simple_execute(self, mocker):
        """shots is forwarded, and the SimulatorConfig built from MaestroConfig
        is passed via config=."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake, shots=1234)

        sim.submit_circuits({"c0": QASM_DEPTH_2})

        call = fake.simple_execute.call_args
        assert call[1]["shots"] == 1234
        passed_config = _submit_config_arg(call)
        assert passed_config is fake.SimulatorConfig.return_value or hasattr(
            passed_config, "kwargs"
        )

    def test_config_passthrough(self, mocker):
        """MaestroConfig fields are forwarded to maestro.SimulatorConfig."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(
            mocker,
            fake,
            config=MaestroConfig(
                simulator_type="QCSim",
                simulation_type="MatrixProductState",
                max_bond_dimension=32,
                singular_value_threshold=1e-6,
                use_double_precision=True,
            ),
        )

        sim.submit_circuits({"c0": QASM_DEPTH_2})

        kwargs = _sim_config_call(fake)
        assert kwargs["simulator_type"] == "QCSim"
        assert kwargs["simulation_type"] == "MatrixProductState"
        assert kwargs["max_bond_dimension"] == 32
        assert kwargs["singular_value_threshold"] == 1e-6
        assert kwargs["use_double_precision"] is True

    def test_extra_knobs_passed_when_non_default(self, mocker):
        """New maestro knobs are forwarded when set to non-default values."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(
            mocker,
            fake,
            config=MaestroConfig(
                disable_optimized_swapping=True,
                lookahead_depth=4,
                mps_measure_no_collapse=False,
            ),
        )

        sim.submit_circuits({"c0": QASM_DEPTH_2})

        kwargs = _sim_config_call(fake)
        assert kwargs["disable_optimized_swapping"] is True
        assert kwargs["lookahead_depth"] == 4
        assert kwargs["mps_measure_no_collapse"] is False

    def test_none_config_not_passed(self, mocker):
        """None-valued config options are not passed to SimulatorConfig."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake)

        sim.submit_circuits({"c0": QASM_DEPTH_2})

        kwargs = _sim_config_call(fake)
        assert "simulator_type" not in kwargs
        assert "simulation_type" not in kwargs
        assert "max_bond_dimension" not in kwargs
        assert "singular_value_threshold" not in kwargs
        assert "use_double_precision" not in kwargs
        assert "disable_optimized_swapping" not in kwargs
        assert "lookahead_depth" not in kwargs
        assert "mps_measure_no_collapse" not in kwargs


class TestParallelExecution:
    """Multi-circuit submissions fan out across a per-instance thread pool
    sized at ``max(1, cpu_count // 2)`` so maestro's internal OpenMP
    threads aren't oversubscribed.  The pool is constructed lazily on the
    first :meth:`MaestroSimulator.submit_circuits` call and reused across
    subsequent submissions; these tests pin that lifecycle plus the
    result-label alignment that the parallel path must preserve."""

    @staticmethod
    def _spy_pool(mocker):
        return mocker.spy(maestro_module, "ThreadPoolExecutor")

    def test_pool_sized_at_half_cpu(self, mocker):
        """The persistent pool is sized at ``max(1, cpu_count // 2)`` regardless
        of how many circuits the first submission carries."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake)
        spy = self._spy_pool(mocker)

        sim.submit_circuits({f"c{i}": QASM_DEPTH_2 for i in range(3)})

        spy.assert_called_once()
        expected_cap = max(1, (os.cpu_count() or 2) // 2)
        assert spy.call_args.kwargs["max_workers"] == expected_cap

    def test_pool_persists_across_submissions(self, mocker):
        """A second ``submit_circuits`` reuses the pool created on the first
        call instead of constructing a fresh one (the load-bearing perf win
        of moving from per-call pools to a persistent per-instance pool)."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake)
        spy = self._spy_pool(mocker)

        sim.submit_circuits({"c0": QASM_DEPTH_2})
        first_pool = sim._executor
        sim.submit_circuits({"c1": QASM_DEPTH_3})

        spy.assert_called_once()
        assert sim._executor is first_pool

    def test_pool_floors_at_one_with_low_cpu_count(self, mocker):
        """If ``os.cpu_count()`` returns 0 or 1, the pool still has at least
        one worker — never zero."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake)
        spy = self._spy_pool(mocker)
        mocker.patch.object(maestro_module.os, "cpu_count", return_value=1)

        sim.submit_circuits({"c0": QASM_DEPTH_2})

        spy.assert_called_once()
        assert spy.call_args.kwargs["max_workers"] == 1

    def test_close_shuts_down_pool(self, mocker):
        """``close()`` releases the pool so its threads don't outlive the
        simulator instance.  Calling ``close()`` a second time is a no-op."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake)
        sim.submit_circuits({"c0": QASM_DEPTH_2})
        assert sim._executor is not None

        sim.close()
        assert sim._executor is None
        # Idempotent.
        sim.close()
        assert sim._executor is None

    def test_submit_after_close_recreates_pool(self, mocker):
        """A ``submit_circuits`` after ``close()`` lazily re-initializes a
        fresh pool and registers a new finalizer."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake)
        sim.submit_circuits({"c0": QASM_DEPTH_2})
        first_pool = sim._executor
        first_finalizer = sim._executor_finalizer

        sim.close()
        assert sim._executor is None
        assert sim._executor_finalizer is None

        sim.submit_circuits({"c1": QASM_DEPTH_3})
        assert sim._executor is not None
        assert sim._executor is not first_pool
        assert sim._executor_finalizer is not None
        assert sim._executor_finalizer is not first_finalizer

    def test_close_does_not_block_concurrent_submit(self, mocker):
        """``close()`` releases ``_executor_lock`` before draining the old
        pool, so a concurrent ``submit_circuits`` on another thread can
        acquire the lock and lazily create a fresh pool while the old one
        finishes shutting down."""
        fake = _make_fake_maestro(mocker)

        # The first submission's worker signals ``worker_started`` once
        # the task is actually executing inside the pool, then blocks on
        # ``drain_release`` until the test lets it finish.  This avoids a
        # ``time.sleep`` race for "is the worker mid-task yet?"
        worker_started = Event()
        drain_release = Event()
        fast_response = {"counts": {"00": 1}}

        def _slow_run(*_args, **_kwargs):
            worker_started.set()
            drain_release.wait(timeout=5)
            return fast_response

        fake.simple_execute.side_effect = _slow_run
        sim = _make_simulator(mocker, fake)

        first_thread = Thread(target=sim.submit_circuits, args=({"c0": QASM_DEPTH_2},))
        first_thread.start()
        assert worker_started.wait(
            timeout=5
        ), "first submission did not reach the worker"
        first_pool = sim._executor

        # close() will block in shutdown(wait=True) waiting for the slow
        # task.  After the fix, _executor_lock is released *before*
        # draining, so a concurrent submit can grab it and re-init.
        close_thread = Thread(target=sim.close)
        close_thread.start()

        # New submit uses a non-blocking fake so it completes promptly.
        fake.simple_execute.side_effect = lambda *a, **kw: fast_response
        submit_done = Event()

        def _fast_submit():
            sim.submit_circuits({"c1": QASM_DEPTH_3})
            submit_done.set()

        new_submit_thread = Thread(target=_fast_submit)
        new_submit_thread.start()
        assert submit_done.wait(
            timeout=2
        ), "concurrent submit was blocked by close()'s drain"

        drain_release.set()
        first_thread.join(timeout=5)
        close_thread.join(timeout=5)
        new_submit_thread.join(timeout=5)

        # A fresh pool was created during the concurrent submit; the old
        # one was discarded.
        assert sim._executor is not None
        assert sim._executor is not first_pool

    def test_label_alignment_under_parallel_dispatch_sampling(self, mocker):
        """Label binding must use the input index, not the completion order.

        ``pool.map`` returns results in submission order, but each worker's
        side-effect must be looked up by *input identity* (the QASM passed
        in) rather than call order — otherwise thread interleaving (which
        differs between Python versions) silently scrambles which result
        attaches to which label.
        """
        fake = _make_fake_maestro(mocker)
        # Three distinct QASMs → three distinct counts.  A side_effect
        # function keyed on the QASM avoids any dependence on call order.
        qasms = [
            QASM_DEPTH_2,
            QASM_DEPTH_3,
            QASM_DEPTH_2.replace("h q[0]", "x q[0]"),
        ]
        per_qasm_counts = {
            qasms[0]: {"00": 100, "11": 0},
            qasms[1]: {"00": 0, "11": 100},
            qasms[2]: {"01": 50, "10": 50},
        }
        fake.simple_execute.side_effect = lambda qasm, **_: {
            "counts": per_qasm_counts[qasm]
        }
        sim = _make_simulator(mocker, fake)

        result = sim.submit_circuits({"c0": qasms[0], "c1": qasms[1], "c2": qasms[2]})

        labels = [r["label"] for r in result.results]
        assert labels == ["c0", "c1", "c2"]
        # Each label retains the counts produced for *its* QASM.
        # (Counts are reversed big-endian → little-endian, so palindromic
        # strings are unchanged; "01"/"10" swap.)
        assert result.results[0]["results"] == {"00": 100, "11": 0}
        assert result.results[1]["results"] == {"00": 0, "11": 100}
        assert result.results[2]["results"] == {"10": 50, "01": 50}

    def test_label_alignment_under_parallel_dispatch_expval(self, mocker):
        fake = _make_fake_maestro(mocker)
        qasms = [
            QASM_DEPTH_2,
            QASM_DEPTH_3,
            QASM_DEPTH_2.replace("h q[0]", "x q[0]"),
        ]
        per_qasm_expval = {
            _strip_measurements(qasms[0]): 0.1,
            _strip_measurements(qasms[1]): 0.2,
            _strip_measurements(qasms[2]): 0.3,
        }
        fake.simple_estimate.side_effect = lambda qasm, **_: {
            "expectation_values": [per_qasm_expval[qasm]]
        }
        sim = _make_simulator(mocker, fake)

        result = sim.submit_circuits(
            {"a": qasms[0], "b": qasms[1], "c": qasms[2]},
            ham_ops="ZI",
        )

        labels = [r["label"] for r in result.results]
        assert labels == ["a", "b", "c"]
        assert result.results[0]["results"] == {"ZI": 0.1}
        assert result.results[1]["results"] == {"ZI": 0.2}
        assert result.results[2]["results"] == {"ZI": 0.3}


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
        """MaestroConfig fields are passed through to simple_estimate via config=."""
        fake = _make_fake_maestro(mocker, expvals=[0.5])
        sim = _make_simulator(
            mocker,
            fake,
            config=MaestroConfig(
                simulator_type="QCSim",
                simulation_type="MatrixProductState",
                max_bond_dimension=32,
            ),
        )

        sim.submit_circuits({"c0": QASM_DEPTH_2}, ham_ops="ZI")

        kwargs = _sim_config_call(fake)
        assert kwargs["simulator_type"] == "QCSim"
        assert kwargs["simulation_type"] == "MatrixProductState"
        assert kwargs["max_bond_dimension"] == 32
        # config is forwarded to simple_estimate
        assert "config" in fake.simple_estimate.call_args[1]

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
# Noisy simulation: sampling (noisy_execute) and expval (noisy_estimate /
# noisy_estimate_montecarlo) dispatch.
# ---------------------------------------------------------------------------


class TestMaestroConfigNoiseDefaults:
    """``MaestroConfig`` carries noise knobs; defaults disable noise."""

    def test_default_noise_fields(self, mocker):
        """No noise overrides → noise_model is None, default seed=42, no realizations."""
        sim = _make_simulator(mocker, _make_fake_maestro(mocker))
        assert sim.config.noise_model is None
        assert sim.config.noise_seed == 42
        assert sim.config.noise_realizations is None

    def test_noise_fields_stored(self, mocker):
        """noise_model / noise_seed / noise_realizations land on MaestroConfig."""
        sim, nm = _make_noisy_sim(
            mocker, _make_fake_maestro(mocker), noise_seed=7, noise_realizations=4
        )
        assert sim.config.noise_model is nm
        assert sim.config.noise_seed == 7
        assert sim.config.noise_realizations == 4

    def test_override_carries_noise_model(self, mocker):
        """``MaestroConfig.override`` propagates a noise_model from ``other``."""
        nm = mocker.MagicMock(name="NoiseModel")
        base = MaestroConfig(simulation_type="Statevector")
        merged = base.override(MaestroConfig(noise_model=nm, noise_realizations=3))
        assert merged.noise_model is nm
        assert merged.noise_realizations == 3
        # Base's other fields survive.
        assert merged.simulation_type == "Statevector"

    def test_override_preserves_base_realizations_when_other_uses_default(self, mocker):
        """override() must not clobber base's noise_realizations when other's is
        still at the default ``None``."""
        nm = mocker.MagicMock(name="NoiseModel")
        base = MaestroConfig(noise_realizations=5)
        merged = base.override(MaestroConfig(noise_model=nm))
        assert merged.noise_model is nm
        assert merged.noise_realizations == 5  # base value, not other's None default


class TestNoisySamplingSubmission:
    """Sampling-mode dispatch: ``simple_execute`` vs ``noisy_execute``."""

    def test_no_noise_uses_simple_execute(self, mocker):
        """``noise_model=None`` keeps the sampling path on ``simple_execute``."""
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake)

        sim.submit_circuits({"c0": _BELL_QASM})

        fake.simple_execute.assert_called_once()
        fake.noisy_execute.assert_not_called()
        fake.QasmToCirc.assert_not_called()

    def test_noise_routes_to_noisy_execute(self, mocker):
        """A non-None ``noise_model`` swaps ``simple_execute`` for ``noisy_execute``."""
        fake = _make_fake_maestro(mocker)
        sim, _ = _make_noisy_sim(mocker, fake)

        sim.submit_circuits({"c0": _BELL_QASM})

        fake.noisy_execute.assert_called_once()
        fake.simple_execute.assert_not_called()
        # noisy_execute consumes a parsed maestro Circuit, not raw QASM.
        fake.QasmToCirc.assert_called_once()
        fake.QasmToCirc.return_value.parse_and_translate.assert_called_once()

    def test_noisy_execute_passes_noise_model_seed_and_default_realizations(
        self, mocker
    ):
        """``noisy_execute`` receives the noise model, default seed=42, and
        ``noise_realizations`` falls back to 1 when unset."""
        fake = _make_fake_maestro(mocker)
        sim, nm = _make_noisy_sim(mocker, fake, shots=321)

        sim.submit_circuits({"c0": _BELL_QASM})

        call = fake.noisy_execute.call_args
        # First positional: parsed maestro circuit; second positional: noise_model
        assert call.args[0] == ("maestro_circuit", _BELL_QASM)
        assert call.args[1] is nm
        assert call.kwargs["shots"] == 321
        assert call.kwargs["seed"] == 42
        assert call.kwargs["noise_realizations"] == 1
        assert "config" in call.kwargs

    def test_noisy_execute_forwards_explicit_seed_and_realizations(self, mocker):
        fake = _make_fake_maestro(mocker)
        sim, _ = _make_noisy_sim(mocker, fake, noise_seed=11, noise_realizations=8)

        sim.submit_circuits({"c0": _BELL_QASM})

        call = fake.noisy_execute.call_args
        assert call.kwargs["seed"] == 11
        assert call.kwargs["noise_realizations"] == 8

    def test_noisy_sampling_reverses_bitstrings(self, mocker):
        """Big-endian → little-endian reversal applies to noisy results too."""
        fake = _make_fake_maestro(mocker, counts={"100": 70, "001": 30})
        sim, _ = _make_noisy_sim(mocker, fake)

        result = sim.submit_circuits({"c0": _BELL_QASM})

        assert result.results[0]["results"] == {"001": 70, "100": 30}

    def test_noisy_sampling_honors_shot_groups(self, mocker):
        """Per-circuit shot allocation is forwarded to ``noisy_execute``."""
        fake = _make_fake_maestro(mocker)
        sim, _ = _make_noisy_sim(mocker, fake, shots=999)

        sim.submit_circuits(
            {"c0": _QASM_SMALL, "c1": _QASM_SMALL, "c2": _QASM_SMALL},
            shot_groups=[[0, 1, 50], [1, 3, 200]],
        )

        calls = fake.noisy_execute.call_args_list
        assert len(calls) == 3
        # ``ThreadPoolExecutor.map`` may reorder per-call kwargs; verify the
        # shot multiset rather than positional ordering.
        seen_shots = sorted(call.kwargs["shots"] for call in calls)
        assert seen_shots == [50, 200, 200]

    @pytest.mark.parametrize("realizations", [0, -1])
    def test_invalid_realizations_raises_in_sampling(self, mocker, realizations):
        """``noise_realizations`` must be ``None`` or a positive integer."""
        fake = _make_fake_maestro(mocker)
        sim, _ = _make_noisy_sim(mocker, fake, noise_realizations=realizations)
        with pytest.raises(ValueError, match="noise_realizations"):
            sim.submit_circuits({"c0": _BELL_QASM})


class TestNoisyExpvalSubmission:
    """Expval-mode dispatch: ``simple_estimate`` vs ``noisy_estimate``
    vs ``noisy_estimate_montecarlo``."""

    def test_no_noise_uses_simple_estimate(self, mocker):
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake)

        sim.submit_circuits({"c0": _BELL_QASM}, ham_ops="ZI;IZ")

        fake.simple_estimate.assert_called_once()
        fake.noisy_estimate.assert_not_called()
        fake.noisy_estimate_montecarlo.assert_not_called()
        fake.QasmToCirc.assert_not_called()

    def test_noise_no_realizations_uses_noisy_estimate(self, mocker):
        """``noise_realizations`` unset (None) takes the analytical noisy path."""
        fake = _make_fake_maestro(mocker)
        sim, _ = _make_noisy_sim(mocker, fake)

        sim.submit_circuits({"c0": _BELL_QASM}, ham_ops="ZI;IZ")

        fake.noisy_estimate.assert_called_once()
        fake.noisy_estimate_montecarlo.assert_not_called()
        fake.simple_estimate.assert_not_called()

    def test_analytical_noisy_estimate_does_not_receive_seed(self, mocker):
        """``noisy_estimate`` (analytical) must not receive ``seed=`` — it is only
        relevant for Monte-Carlo sampling.  A non-default seed is used to confirm
        it does not leak onto the analytical path."""
        fake = _make_fake_maestro(mocker)
        sim, _ = _make_noisy_sim(mocker, fake, noise_seed=99)

        sim.submit_circuits({"c0": _BELL_QASM}, ham_ops="ZI")

        assert "seed" not in fake.noisy_estimate.call_args.kwargs

    def test_realizations_1_uses_montecarlo_not_analytical(self, mocker):
        """``noise_realizations=1`` must route to Monte Carlo, not the analytical
        backend — one random Pauli sampling is not equivalent to the analytical mean.
        This pins the documented non-equivalence: None → analytical, 1 → MC."""
        fake = _make_fake_maestro(mocker)
        sim, _ = _make_noisy_sim(mocker, fake, noise_realizations=1)

        sim.submit_circuits({"c0": _BELL_QASM}, ham_ops="ZI")

        fake.noisy_estimate_montecarlo.assert_called_once()
        fake.noisy_estimate.assert_not_called()

    @pytest.mark.parametrize("realizations", [0, -1])
    def test_invalid_realizations_raises_in_expval(self, mocker, realizations):
        """``noise_realizations`` must be ``None`` or a positive integer."""
        fake = _make_fake_maestro(mocker)
        sim, _ = _make_noisy_sim(mocker, fake, noise_realizations=realizations)
        with pytest.raises(ValueError, match="noise_realizations"):
            sim.submit_circuits({"c0": _BELL_QASM}, ham_ops="ZI")

    def test_noise_with_realizations_uses_montecarlo(self, mocker):
        """``noise_realizations >= 1`` switches to ``noisy_estimate_montecarlo``."""
        fake = _make_fake_maestro(mocker)
        sim, nm = _make_noisy_sim(mocker, fake, noise_realizations=5)

        sim.submit_circuits({"c0": _BELL_QASM}, ham_ops="ZI;IZ")

        fake.noisy_estimate_montecarlo.assert_called_once()
        fake.noisy_estimate.assert_not_called()
        call = fake.noisy_estimate_montecarlo.call_args
        assert call.kwargs["noise_model"] is nm
        assert call.kwargs["noise_realizations"] == 5
        assert call.kwargs["observables"] == "ZI;IZ"
        assert "config" in call.kwargs

    def test_noisy_estimate_strips_measurements(self, mocker):
        """``noisy_estimate``/``montecarlo`` receive a parsed circuit built
        from QASM that has had its ``measure`` lines stripped — measurements
        would corrupt the analytical statevector."""
        fake = _make_fake_maestro(mocker)
        sim, _ = _make_noisy_sim(mocker, fake)

        sim.submit_circuits({"c0": _BELL_QASM}, ham_ops="ZI")

        parsed_input = fake.QasmToCirc.return_value.parse_and_translate.call_args.args[
            0
        ]
        assert "measure" not in parsed_input
        assert "h q[0]" in parsed_input  # body preserved

    def test_noisy_estimate_passes_observables_string(self, mocker):
        fake = _make_fake_maestro(mocker)
        sim, _ = _make_noisy_sim(mocker, fake)

        sim.submit_circuits({"c0": _BELL_QASM}, ham_ops="ZI;IZ")

        assert fake.noisy_estimate.call_args.kwargs["observables"] == "ZI;IZ"

    def test_noisy_estimate_zips_results(self, mocker):
        """Pauli operators map to expectation values regardless of which
        noisy backend was used."""
        fake = _make_fake_maestro(mocker, expvals=[0.1, 0.2, 0.3])
        sim, _ = _make_noisy_sim(mocker, fake, noise_realizations=2)

        result = sim.submit_circuits({"c0": _BELL_QASM}, ham_ops="ZI;IX;YY")

        assert result.results[0]["results"] == {"ZI": 0.1, "IX": 0.2, "YY": 0.3}

    def test_montecarlo_multi_circuit_label_alignment_and_seed_offset(self, mocker):
        """Submit two circuits through the Monte Carlo path and verify:
        labels keep their input order in the result; per-circuit seed is
        ``noise_seed + i`` so circuit 0 sees seed N, circuit 1 sees N+1."""
        fake = _make_fake_maestro(mocker, expvals=[0.9])
        seeds_seen: dict[int, int] = {}

        def _capture(
            circuit, observables, noise_model, noise_realizations, seed, config
        ):
            # Each call gets a unique parsed-circuit tuple keyed on the qasm
            # string, recovered from the parser side_effect.
            seeds_seen[id(circuit)] = seed
            return {"expectation_values": [0.9]}

        fake.noisy_estimate_montecarlo.side_effect = _capture
        sim, _ = _make_noisy_sim(mocker, fake, noise_seed=10, noise_realizations=3)

        result = sim.submit_circuits(
            {"a": _BELL_QASM, "b": _BELL_QASM.replace("h q[0]", "x q[0]")},
            ham_ops="ZI",
        )

        labels = [r["label"] for r in result.results]
        assert labels == ["a", "b"]
        assert sorted(seeds_seen.values()) == [10, 11]


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

    def test_strips_non_default_creg_name(self):
        """Measurements targeting a non-``c`` classical register must be stripped."""
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg meas[2];\n'
            "h q[0];\nmeasure q[0] -> meas[0];\nmeasure q[1] -> meas[1];\n"
        )
        result = _strip_measurements(qasm)
        assert "measure" not in result
        assert "h q[0]" in result
        assert "creg meas[2]" in result


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


# ---------------------------------------------------------------------------
# shot_groups (per-circuit shot allocation)
# ---------------------------------------------------------------------------


class TestShotGroupsSampling:
    """Spec: in sampling mode, MaestroSimulator runs each circuit with the
    shot count assigned by ``shot_groups``."""

    def test_per_circuit_shots_passed_to_simple_execute(self, mocker):
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake, shots=999)
        circuits = {"c0": _QASM_SMALL, "c1": _QASM_SMALL, "c2": _QASM_SMALL}
        sim.submit_circuits(circuits, shot_groups=[[0, 1, 50], [1, 3, 200]])

        # One call per circuit; each call's shots= matches the range.
        call_args_list = fake.simple_execute.call_args_list
        assert len(call_args_list) == 3
        assert call_args_list[0].kwargs["shots"] == 50
        assert call_args_list[1].kwargs["shots"] == 200
        assert call_args_list[2].kwargs["shots"] == 200

    def test_no_shot_groups_uses_instance_shots(self, mocker):
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake, shots=999)
        sim.submit_circuits({"c0": _QASM_SMALL, "c1": _QASM_SMALL})

        for call in fake.simple_execute.call_args_list:
            assert call.kwargs["shots"] == 999

    def test_shot_groups_with_ham_ops_raises(self, mocker):
        """maestro.simple_estimate is analytical -> shots are irrelevant.
        Passing both ham_ops and shot_groups is a programming error and
        must be caught at the backend boundary."""
        fake = _make_fake_maestro(mocker, expvals=[0.5])
        sim = _make_simulator(mocker, fake)
        with pytest.raises(ValueError, match="incompatible with ham_ops"):
            sim.submit_circuits(
                {"c0": _QASM_SMALL, "c1": _QASM_SMALL},
                ham_ops="Z" + "I" * 9,
                shot_groups=[[0, 2, 7]],
            )

    def test_partial_coverage_raises_at_submit(self, mocker):
        fake = _make_fake_maestro(mocker)
        sim = _make_simulator(mocker, fake)
        with pytest.raises(ValueError, match="do not cover every circuit"):
            sim.submit_circuits(
                {"c0": _QASM_SMALL, "c1": _QASM_SMALL, "c2": _QASM_SMALL},
                shot_groups=[[0, 2, 100]],
            )


# ---------------------------------------------------------------------------
# Real-maestro integration — guards against upstream API drift
# ---------------------------------------------------------------------------
#
# The mocked tests above rubber-stamp any kwargs MaestroSimulator hands off,
# so a breaking change in ``maestro.SimulatorConfig`` slips through them.
# These tests exercise the *real* module: any renamed, removed, or newly
# added knob will fail one of the assertions below and force a review.


@requires_real_maestro
class TestRealMaestroIntegration:
    def test_knob_parity_with_maestro_simulator_config(self):
        """MaestroConfig must cover every knob on maestro.SimulatorConfig.

        Compares field names by introspecting ``maestro.SimulatorConfig``'s
        data descriptors against ``MaestroConfig``'s dataclass fields.  Any
        drift — maestro adds, removes, or renames a knob — fails here and
        forces a deliberate decision about whether to expose it.
        """
        maestro_fields = {
            name
            for name in dir(_real_maestro.SimulatorConfig)
            if not name.startswith("_")
        }
        _divi_only = {
            # mps_qubit_threshold is divi-side auto-MPS logic, not a maestro knob.
            "mps_qubit_threshold",
            # Noise lives on MaestroConfig but is consumed by the noisy entry
            # points (noisy_execute / noisy_estimate / *_montecarlo) — Maestro
            # keeps noise out of SimulatorConfig itself.
            "noise_model",
            "noise_seed",
            "noise_realizations",
        }
        # Guard: if any divi-only field is removed from MaestroConfig, the
        # exclusion set would silently over-exclude and the parity check would
        # pass even though a field went missing.
        assert _divi_only <= {f.name for f in fields(MaestroConfig)}, (
            f"Exclusion set names fields no longer in MaestroConfig: "
            f"{_divi_only - {f.name for f in fields(MaestroConfig)}}"
        )
        divi_fields = {f.name for f in fields(MaestroConfig)} - _divi_only
        assert maestro_fields == divi_fields, (
            "MaestroConfig is out of sync with maestro.SimulatorConfig.\n"
            f"  Missing in MaestroConfig: {sorted(maestro_fields - divi_fields)}\n"
            f"  Extra in MaestroConfig:   {sorted(divi_fields - maestro_fields)}"
        )

    def test_every_knob_round_trips_to_real_maestro(self):
        """Every MaestroConfig field survives the hand-off to real maestro.

        Sets every knob to a non-default value and runs a small circuit.  If
        maestro renames or removes one of the kwargs we forward, the nanobind
        dispatcher raises ``TypeError`` here — catching the class of break
        that slipped past us on the previous maestro release.
        """
        cfg = MaestroConfig(
            simulator_type="QCSim",
            simulation_type="MatrixProductState",
            max_bond_dimension=16,
            singular_value_threshold=1e-7,
            use_double_precision=True,
            disable_optimized_swapping=True,
            lookahead_depth=2,
            mps_measure_no_collapse=False,
        )
        sim = MaestroSimulator(shots=100, config=cfg)

        result = sim.submit_circuits({"c0": _BELL_QASM})

        assert sum(result.results[0]["results"].values()) == 100

    def test_expval_path_on_real_maestro(self):
        """The simple_estimate call path is covered separately from simple_execute."""
        sim = MaestroSimulator(shots=100)

        result = sim.submit_circuits({"c0": _BELL_QASM}, ham_ops="ZI;IZ")

        assert set(result.results[0]["results"]) == {"ZI", "IZ"}
