# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.stages._pce_cost_stage (PCECostStage.reduce).

Focus: histogram merging, path routing (soft/hard/expval), observable ordering,
and multi-param-set independence.  Energy *values* for the helpers are already
covered in tests/qprog/algorithms/test_pce.py, so here we use hand-computed
expected values or comparative assertions.
"""

import numpy as np
import pytest

from divi.hamiltonians import normalize_binary_polynomial_problem
from divi.pipeline.abc import PipelineEnv, ResultFormat
from divi.pipeline.stages._pce_cost_stage import PCECostStage
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


def _obs_key(param_idx: int, obs_idx: int):
    """Build a child-result label for (param_set, obs_group)."""
    return (("param_set", param_idx), ("obs_group", obs_idx))


# ---------------------------------------------------------------------------
# Tests: histogram merging
# ---------------------------------------------------------------------------


class TestReduceHistogramMerging:
    """Verify that reduce correctly merges multiple observable-group histograms."""

    def test_split_histograms_equal_single_merged(self):
        """Splitting a histogram across obs groups gives the same energy as one group."""
        qubo = np.diag([1.0, 2.0])
        stage = _make_stage(qubo, alpha=1.0, soft=True)
        env = _make_env(ResultFormat.COUNTS)

        merged_result = stage.reduce(
            {_obs_key(0, 0): {"00": 30, "01": 10, "10": 20, "11": 40}},
            env,
            token=None,
        )

        split_result = stage.reduce(
            {
                _obs_key(0, 0): {"00": 30, "01": 10},
                _obs_key(0, 1): {"10": 20, "11": 40},
            },
            env,
            token=None,
        )

        assert list(split_result.values())[0] == pytest.approx(
            list(merged_result.values())[0]
        )

    def test_overlapping_bitstrings_are_summed(self):
        """Overlapping keys across obs groups are summed, not overwritten."""
        qubo = np.diag([1.0, 2.0])
        stage = _make_stage(qubo, alpha=1.0, soft=True)
        env = _make_env(ResultFormat.COUNTS)

        # "01" appears in both groups: 5 + 3 = 8
        overlap_result = stage.reduce(
            {
                _obs_key(0, 0): {"00": 10, "01": 5},
                _obs_key(0, 1): {"01": 3, "10": 7},
            },
            env,
            token=None,
        )

        # Equivalent single group with the merged histogram
        single_result = stage.reduce(
            {_obs_key(0, 0): {"00": 10, "01": 8, "10": 7}},
            env,
            token=None,
        )

        assert list(overlap_result.values())[0] == pytest.approx(
            list(single_result.values())[0]
        )


# ---------------------------------------------------------------------------
# Tests: path routing (soft vs hard CVaR vs expval)
# ---------------------------------------------------------------------------


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
            soft_stage.reduce({_obs_key(0, 0): histogram}, env, token=None).values()
        )[0]
        hard_energy = list(
            hard_stage.reduce({_obs_key(0, 0): histogram}, env, token=None).values()
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

        result = stage.reduce({_obs_key(0, 0): {"00": 100}}, env, token=None)

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

        result = stage.reduce({_obs_key(0, 0): {"11": 100}}, env, token=None)

        assert list(result.values())[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: expval path
# ---------------------------------------------------------------------------


class TestReduceExpvalPath:
    """PCECostStage.reduce with ResultFormat.EXPVALS."""

    def test_expval_zero_expectations(self):
        """Z-expectations all zero → x_soft = [0.5, 0.5] → hand-computed energy.

        qubo = diag([1, 2]), z = [0, 0].
        x_soft = 0.5*(1 + tanh(0)) = [0.5, 0.5].
        energy = 1*0.5² + 2*0.5² = 0.25 + 0.5 = 0.75.
        """
        qubo = np.diag([1.0, 2.0])
        stage = _make_stage(qubo, alpha=1.0, soft=True)
        env = _make_env(ResultFormat.EXPVALS)

        result = stage.reduce(
            {_obs_key(0, 0): 0.0, _obs_key(0, 1): 0.0},
            env,
            token=None,
        )

        assert list(result.values())[0] == pytest.approx(0.75)

    def test_expval_saturated_positive(self):
        """Large positive Z → x_soft ≈ 1 → energy ≈ sum of diagonal.

        qubo = diag([1, 2]), z = [100, 100].
        tanh(100) ≈ 1, x_soft ≈ [1, 1].
        energy ≈ 1*1² + 2*1² = 3.0.
        """
        qubo = np.diag([1.0, 2.0])
        stage = _make_stage(qubo, alpha=1.0, soft=True)
        env = _make_env(ResultFormat.EXPVALS)

        result = stage.reduce(
            {_obs_key(0, 0): 100.0, _obs_key(0, 1): 100.0},
            env,
            token=None,
        )

        assert list(result.values())[0] == pytest.approx(3.0, abs=1e-6)

    def test_expval_preserves_observable_order(self):
        """Out-of-order obs_group indices are sorted, mapping Z values correctly.

        qubo = diag([1, 2]), alpha=1. Provide z0=0, z1=100 but obs_group 1 first.
        Correct ordering: x_soft ≈ [0.5, 1.0].
        energy = 1*0.5² + 2*1² = 0.25 + 2.0 = 2.25.
        Wrong ordering (swapped): x_soft ≈ [1.0, 0.5].
        energy = 1*1² + 2*0.5² = 1.0 + 0.5 = 1.5.
        """
        qubo = np.diag([1.0, 2.0])
        stage = _make_stage(qubo, alpha=1.0, soft=True)
        env = _make_env(ResultFormat.EXPVALS)

        # Deliberately provide obs_group 1 before obs_group 0
        result = stage.reduce(
            {_obs_key(0, 1): 100.0, _obs_key(0, 0): 0.0},
            env,
            token=None,
        )

        # If ordering is correct: z=[0, 100] → x≈[0.5, 1.0] → ≈2.25
        assert list(result.values())[0] == pytest.approx(2.25, abs=1e-2)


# ---------------------------------------------------------------------------
# Tests: multi-param-set independence
# ---------------------------------------------------------------------------


class TestReduceMultiParamSet:
    """Each param_set is reduced independently."""

    def test_counts_two_param_sets_independent(self):
        """Two param_sets with different histograms produce different energies."""
        qubo = np.diag([1.0, 2.0])
        stage = _make_stage(qubo, alpha=1.0, soft=True)
        env = _make_env(ResultFormat.COUNTS)

        results = {
            _obs_key(0, 0): {"00": 100},  # all parities 0
            _obs_key(1, 0): {"11": 100},  # all parities 1
        }

        reduced = stage.reduce(results, env, token=None)

        assert len(reduced) == 2
        energies = list(reduced.values())
        # Different histograms must yield different energies
        assert energies[0] != pytest.approx(energies[1])

    def test_expval_two_param_sets_independent(self):
        """Two param_sets with different Z-expectations produce different energies."""
        qubo = np.diag([1.0, 2.0])
        stage = _make_stage(qubo, alpha=1.0, soft=True)
        env = _make_env(ResultFormat.EXPVALS)

        results = {
            _obs_key(0, 0): 0.0,
            _obs_key(0, 1): 0.0,
            _obs_key(1, 0): 100.0,
            _obs_key(1, 1): 100.0,
        }

        reduced = stage.reduce(results, env, token=None)

        assert len(reduced) == 2
        key_0 = (("param_set", 0),)
        key_1 = (("param_set", 1),)
        # z=[0,0] → energy=0.75; z=[100,100] → energy≈3.0
        assert reduced[key_0] == pytest.approx(0.75)
        assert reduced[key_1] == pytest.approx(3.0, abs=1e-6)
