# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.stages._pce_cost_stage (PCECostStage).

Focus: expand behaviour (single Z-basis circuit regardless of backend),
path routing (soft vs hard CVaR), and multi-param-set independence.
Energy *values* for the helpers are already covered in
tests/qprog/algorithms/test_pce.py, so here we use hand-computed expected
values or comparative assertions.
"""

import numpy as np
import pennylane as qml
import pytest

from divi.circuits._core import MetaCircuit
from divi.hamiltonians import normalize_binary_polynomial_problem
from divi.pipeline.abc import PipelineEnv, ResultFormat
from divi.pipeline.stages._pce_cost_stage import PCE_MEAS_AXIS, PCECostStage
from divi.qprog.algorithms._pce import _decode_parities

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stage(
    qubo: np.ndarray,
    *,
    alpha: float = 1.0,
    soft: bool = True,
    alpha_cvar: float = 0.25,
):
    """Build a PCECostStage from a simple QUBO matrix."""
    problem = normalize_binary_polynomial_problem(qubo)
    n_vars = problem.n_vars
    masks = np.arange(1, n_vars + 1, dtype=np.uint64)
    return PCECostStage(
        problem=problem,
        alpha=alpha,
        use_soft_objective=soft,
        decode_parities_fn=_decode_parities,
        variable_masks_u64=masks,
        alpha_cvar=alpha_cvar,
    )


def _make_env(result_format: ResultFormat):
    """Build a minimal PipelineEnv with the given result_format."""
    env = PipelineEnv(backend=None)
    env.result_format = result_format
    return env


def _meas_key(param_idx: int):
    """Build a child-result label for (param_set, pce_meas)."""
    return (("param_set", param_idx), (PCE_MEAS_AXIS, 0))


def _make_z_hamiltonian_batch(n_qubits: int) -> dict[int, MetaCircuit]:
    """Build a single-entry MetaCircuit batch with expval(Z0+Z1+...+Zn)."""
    H = qml.Hamiltonian([1.0] * n_qubits, [qml.PauliZ(i) for i in range(n_qubits)])
    tape = qml.tape.QuantumScript(
        [qml.RY(0.5, wires=i) for i in range(n_qubits)],
        [qml.expval(H)],
    )
    return {0: MetaCircuit(source_circuit=tape, symbols=np.array([]))}


def _make_expval_backend():
    """Create a mock backend that supports expval."""
    return type("MockBackend", (), {"supports_expval": True})()


def _make_sampling_backend():
    """Create a mock backend that does NOT support expval."""
    return type("MockBackend", (), {"supports_expval": False})()


# ---------------------------------------------------------------------------
# Tests: expand produces a single Z-basis circuit (no obs_group blowup)
# ---------------------------------------------------------------------------


class TestExpandSingleCircuit:
    """PCECostStage.expand should produce one measurement circuit per param_set."""

    def test_expand_one_circuit_per_param_set_expval_backend(self):
        """With an expval-capable backend, still produces exactly 1 circuit."""
        n_qubits = 16
        batch = _make_z_hamiltonian_batch(n_qubits)
        stage = _make_stage(np.eye(n_qubits), alpha=1.0, soft=True)

        env = PipelineEnv(backend=_make_expval_backend())
        result, _ = stage.expand(batch, env)

        expanded = list(result.batch.values())[0]
        assert len(expanded.measurement_qasms) == 1

    def test_expand_one_circuit_per_param_set_sampling_backend(self):
        """With a sampling-only backend, produces exactly 1 circuit."""
        n_qubits = 16
        batch = _make_z_hamiltonian_batch(n_qubits)
        stage = _make_stage(np.eye(n_qubits), alpha=1.0, soft=True)

        env = PipelineEnv(backend=_make_sampling_backend())
        result, _ = stage.expand(batch, env)

        expanded = list(result.batch.values())[0]
        assert len(expanded.measurement_qasms) == 1

    def test_expand_result_format_is_counts(self):
        """Result format must be COUNTS after expand, regardless of backend."""
        n_qubits = 4
        batch = _make_z_hamiltonian_batch(n_qubits)
        stage = _make_stage(np.eye(n_qubits), alpha=1.0, soft=True)

        env = PipelineEnv(backend=_make_expval_backend())
        stage.expand(batch, env)

        assert env.result_format == ResultFormat.COUNTS

    def test_expand_no_ham_ops_artifact(self):
        """ham_ops must NOT be set — PCE never uses the backend expval path."""
        n_qubits = 4
        batch = _make_z_hamiltonian_batch(n_qubits)
        stage = _make_stage(np.eye(n_qubits), alpha=1.0, soft=True)

        env = PipelineEnv(backend=_make_expval_backend())
        stage.expand(batch, env)

        assert "ham_ops" not in env.artifacts


# ---------------------------------------------------------------------------
# Tests: reduce — histogram processing and path routing
# ---------------------------------------------------------------------------


class TestReduceHistogram:
    """Verify that reduce correctly processes single-histogram results."""

    def test_single_histogram_produces_energy(self):
        """A single histogram per param_set produces a valid energy."""
        qubo = np.diag([1.0, 2.0])
        stage = _make_stage(qubo, alpha=1.0, soft=True)
        env = _make_env(ResultFormat.COUNTS)

        result = stage.reduce(
            {_meas_key(0): {"00": 30, "01": 10, "10": 20, "11": 40}},
            env,
            token=None,
        )

        assert len(result) == 1
        assert isinstance(list(result.values())[0], float)

    def test_different_histograms_produce_different_energies(self):
        """Different shot distributions yield different energies."""
        qubo = np.diag([1.0, 2.0])
        stage = _make_stage(qubo, alpha=1.0, soft=True)
        env = _make_env(ResultFormat.COUNTS)

        result_a = stage.reduce(
            {_meas_key(0): {"00": 100}},
            env,
            token=None,
        )
        result_b = stage.reduce(
            {_meas_key(0): {"11": 100}},
            env,
            token=None,
        )

        assert list(result_a.values())[0] != pytest.approx(list(result_b.values())[0])


class TestReducePathRouting:
    """Verify reduce dispatches to the correct energy computation path."""

    def test_soft_and_hard_produce_different_energies(self):
        """Soft energy and hard CVaR energy differ for the same histogram."""
        qubo = np.diag([1.0, 2.0])
        histogram = {"11": 2, "10": 3, "01": 10, "00": 25}
        env = _make_env(ResultFormat.COUNTS)

        soft_stage = _make_stage(qubo, alpha=1.0, soft=True)
        hard_stage = _make_stage(qubo, alpha=6.0, soft=False, alpha_cvar=0.25)

        soft_energy = list(
            soft_stage.reduce({_meas_key(0): histogram}, env, token=None).values()
        )[0]
        hard_energy = list(
            hard_stage.reduce({_meas_key(0): histogram}, env, token=None).values()
        )[0]

        assert soft_energy != pytest.approx(hard_energy)

    def test_deterministic_histogram_soft_energy(self):
        """All shots in one bitstring → known energy.

        qubo = diag([1, 2]), all shots "00" → parities [0, 0] for masks [1, 2].
        mean_parities = [0, 0], z = 1 - 2*0 = [1, 1].
        x_soft = 0.5*(1 + tanh(1*1)) = 0.5*(1 + tanh(1)) for both vars.
        energy = 1*x0² + 2*x1² (degree-1 terms use x²).
        """
        qubo = np.diag([1.0, 2.0])
        stage = _make_stage(qubo, alpha=1.0, soft=True)
        env = _make_env(ResultFormat.COUNTS)

        result = stage.reduce({_meas_key(0): {"00": 100}}, env, token=None)

        x = 0.5 * (1.0 + np.tanh(1.0))  # ≈ 0.8808
        expected = 1.0 * x**2 + 2.0 * x**2  # 3 * x²
        assert list(result.values())[0] == pytest.approx(expected)

    def test_deterministic_histogram_hard_cvar_energy(self):
        """All shots in one bitstring → known CVaR energy.

        qubo = diag([1, 2]), all shots "11" → parities [1, 1] for masks [1, 2].
        x_vals = 1 - parities = [0, 0].  Energy = 0 for every shot.
        CVaR of a single-valued distribution is that value: 0.
        """
        qubo = np.diag([1.0, 2.0])
        stage = _make_stage(qubo, alpha=6.0, soft=False, alpha_cvar=0.25)
        env = _make_env(ResultFormat.COUNTS)

        result = stage.reduce({_meas_key(0): {"11": 100}}, env, token=None)

        assert list(result.values())[0] == pytest.approx(0.0)

    def test_hard_cvar_selects_low_energy_tail(self):
        """CVaR with alpha_cvar=0.5 selects the lower-energy half of shots.

        qubo = diag([1, 2]), masks = [1, 2].
        Bitstring "11" → parities [1, 1] → x = [0, 0] → energy = 0.
        Bitstring "00" → parities [0, 0] → x = [1, 1] → energy = 1+2 = 3.

        50 shots of "11" (energy 0) + 50 shots of "00" (energy 3).
        Mean energy = 1.5.
        CVaR(0.5) takes the lowest 50 shots → all "11" → energy = 0.
        """
        qubo = np.diag([1.0, 2.0])
        stage = _make_stage(qubo, alpha=6.0, soft=False, alpha_cvar=0.5)
        env = _make_env(ResultFormat.COUNTS)

        result = stage.reduce({_meas_key(0): {"11": 50, "00": 50}}, env, token=None)

        assert list(result.values())[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: multi-param-set independence
# ---------------------------------------------------------------------------


class TestReduceMultiParamSet:
    """Each param_set is reduced independently."""

    def test_two_param_sets_independent(self):
        """Two param_sets with different histograms produce different energies."""
        qubo = np.diag([1.0, 2.0])
        stage = _make_stage(qubo, alpha=1.0, soft=True)
        env = _make_env(ResultFormat.COUNTS)

        results = {
            _meas_key(0): {"00": 100},  # all parities 0
            _meas_key(1): {"11": 100},  # all parities 1
        }

        reduced = stage.reduce(results, env, token=None)

        assert len(reduced) == 2
        energies = list(reduced.values())
        # Different histograms must yield different energies
        assert energies[0] != pytest.approx(energies[1])
