# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest

from divi.qprog.early_stopping import EarlyStopping, StopReason

# ======================================================================
#  Construction & validation
# ======================================================================


class TestEarlyStoppingInit:
    """Tests for EarlyStopping parameter validation and defaults."""

    def test_default_values(self):
        es = EarlyStopping()
        sig = inspect.signature(EarlyStopping.__init__)
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            assert getattr(es, name) == param.default

    def test_patience_must_be_at_least_one(self):
        with pytest.raises(ValueError, match="patience must be >= 1"):
            EarlyStopping(patience=0)

    def test_min_delta_must_be_non_negative(self):
        with pytest.raises(ValueError, match="min_delta must be >= 0"):
            EarlyStopping(min_delta=-0.1)

    def test_variance_window_must_be_at_least_two(self):
        with pytest.raises(ValueError, match="variance_window must be >= 2"):
            EarlyStopping(variance_window=1)

    def test_custom_values_accepted(self):
        es = EarlyStopping(
            patience=3,
            min_delta=0.01,
            grad_norm_threshold=1e-5,
            variance_window=10,
            variance_threshold=1e-6,
        )
        assert es.patience == 3
        assert es.min_delta == 0.01
        assert es.grad_norm_threshold == 1e-5
        assert es.variance_window == 10
        assert es.variance_threshold == 1e-6


# ======================================================================
#  Patience criterion
# ======================================================================


class TestPatienceCriterion:
    """Tests for the patience-based stopping criterion."""

    def test_triggers_after_n_stale_iterations(self):
        es = EarlyStopping(patience=3, min_delta=0.0)
        assert es.check(1.0) is None
        assert es._stale_count == 0
        assert es._tracked_best == 1.0

        assert es.check(1.0) is None
        assert es._stale_count == 1

        assert es.check(1.0) is None
        assert es._stale_count == 2

        assert es.check(1.0) == StopReason.PATIENCE_EXCEEDED
        assert es._stale_count == 3

    def test_resets_on_improvement(self):
        es = EarlyStopping(patience=3, min_delta=0.0)
        assert es.check(1.0) is None
        assert es._stale_count == 0
        assert es._tracked_best == 1.0

        assert es.check(1.0) is None
        assert es._stale_count == 1

        # Improvement resets counter
        assert es.check(0.5) is None
        assert es._stale_count == 0
        assert es._tracked_best == 0.5

        assert es.check(0.5) is None
        assert es._stale_count == 1

        assert es.check(0.5) is None
        assert es._stale_count == 2

        assert es.check(0.5) == StopReason.PATIENCE_EXCEEDED
        assert es._stale_count == 3

    def test_min_delta_respected(self):
        """Tiny improvements below min_delta should NOT reset patience."""
        es = EarlyStopping(patience=3, min_delta=0.1)
        assert es.check(1.0) is None
        assert es._stale_count == 0
        assert es._tracked_best == 1.0

        # delta=0.05 < min_delta=0.1, so still stale
        assert es.check(0.95) is None
        assert es._stale_count == 1
        assert es._tracked_best == 1.0  # unchanged

        assert es.check(0.95) is None
        assert es._stale_count == 2

        assert es.check(0.95) == StopReason.PATIENCE_EXCEEDED
        assert es._stale_count == 3

    def test_min_delta_large_improvement_resets(self):
        """A sufficiently large improvement should reset patience."""
        es = EarlyStopping(patience=3, min_delta=0.1)
        assert es.check(1.0) is None
        assert es._stale_count == 0

        assert es.check(1.0) is None
        assert es._stale_count == 1

        # improvement=0.2 > min_delta=0.1, resets
        assert es.check(0.8) is None
        assert es._stale_count == 0
        assert es._tracked_best == 0.8

        assert es.check(0.8) is None
        assert es._stale_count == 1

        assert es.check(0.8) is None
        assert es._stale_count == 2

        assert es.check(0.8) == StopReason.PATIENCE_EXCEEDED

    def test_monotonically_decreasing_loss_never_triggers(self):
        es = EarlyStopping(patience=3, min_delta=0.0)
        for i in range(100):
            assert es.check(100.0 - i) is None
            assert es._stale_count == 0


# ======================================================================
#  Gradient norm criterion
# ======================================================================


class TestGradientNormCriterion:
    """Tests for the gradient-norm stopping criterion."""

    def test_triggers_when_below_threshold(self):
        es = EarlyStopping(patience=100, grad_norm_threshold=1e-5)
        result = es.check(0.5, grad_norm=1e-6)
        assert result == StopReason.GRADIENT_BELOW_THRESHOLD
        assert es._stale_count == 0
        assert es._tracked_best == 0.5

    def test_does_not_trigger_when_above_threshold(self):
        es = EarlyStopping(patience=100, grad_norm_threshold=1e-5)
        result = es.check(0.5, grad_norm=1e-3)
        assert result is None
        assert es._stale_count == 0

    def test_skipped_when_grad_norm_is_none(self):
        es = EarlyStopping(patience=100, grad_norm_threshold=1e-5)
        result = es.check(0.5, grad_norm=None)
        assert result is None
        assert es._stale_count == 0

    def test_skipped_when_threshold_is_none(self):
        es = EarlyStopping(patience=100, grad_norm_threshold=None)
        result = es.check(0.5, grad_norm=1e-10)
        assert result is None
        assert es._stale_count == 0


# ======================================================================
#  Cost variance criterion
# ======================================================================


class TestCostVarianceCriterion:
    """Tests for the cost-variance stopping criterion."""

    def test_triggers_when_variance_below_threshold(self):
        es = EarlyStopping(
            patience=100,
            variance_window=5,
            variance_threshold=1e-6,
        )
        # Fill the window with identical values (variance = 0)
        for i in range(4):
            assert es.check(1.0) is None
            assert len(es._loss_history) == i + 1

        assert es.check(1.0) == StopReason.COST_VARIANCE_SETTLED
        assert len(es._loss_history) == 5

    def test_does_not_trigger_when_variance_above_threshold(self):
        es = EarlyStopping(
            patience=100,
            variance_window=5,
            variance_threshold=1e-6,
        )
        losses = [1.0, 0.9, 1.1, 0.8, 1.2]  # high variance
        for loss in losses:
            result = es.check(loss)
        assert result is None
        assert len(es._loss_history) == 5

    def test_does_not_trigger_before_window_fills(self):
        es = EarlyStopping(
            patience=100,
            variance_window=10,
            variance_threshold=1e-6,
        )
        # Only 5 values (window needs 10)
        for i in range(5):
            assert es.check(1.0) is None
            assert len(es._loss_history) == i + 1

    def test_skipped_when_threshold_is_none(self):
        es = EarlyStopping(
            patience=100,
            variance_window=5,
            variance_threshold=None,
        )
        # Even constant losses shouldn't trigger
        for _ in range(20):
            assert es.check(1.0) is None
        # loss_history should not be populated when threshold is None
        assert len(es._loss_history) == 0

    def test_rolling_window_discards_old_values(self):
        es = EarlyStopping(
            patience=100,
            variance_window=3,
            variance_threshold=1e-6,
        )
        # First fill with varied values
        es.check(1.0)
        es.check(2.0)
        es.check(3.0)
        assert list(es._loss_history) == [1.0, 2.0, 3.0]

        # Now push 3 identical values that should push out the old ones
        es.check(5.0)
        es.check(5.0)
        assert list(es._loss_history) == [3.0, 5.0, 5.0]

        result = es.check(5.0)
        assert result == StopReason.COST_VARIANCE_SETTLED
        assert list(es._loss_history) == [5.0, 5.0, 5.0]


# ======================================================================
#  Priority / interaction
# ======================================================================


class TestCriteriaPriority:
    """Test that criteria are checked in the documented order."""

    def test_patience_checked_before_gradient(self):
        """When both patience and gradient would trigger, patience wins."""
        es = EarlyStopping(patience=1, grad_norm_threshold=1e-5)
        # First call: improvement from inf, stale=0, no trigger
        es.check(1.0)
        assert es._stale_count == 0

        # Second call: stale=1 == patience, triggers patience before gradient
        result = es.check(1.0, grad_norm=1e-10)
        assert result == StopReason.PATIENCE_EXCEEDED
        assert es._stale_count == 1

    def test_gradient_checked_before_variance(self):
        """When gradient and variance would both trigger, gradient wins."""
        es = EarlyStopping(
            patience=100,
            grad_norm_threshold=1e-5,
            variance_window=3,
            variance_threshold=1e-6,
        )
        # Fill the variance window
        es.check(1.0)
        es.check(1.0)
        assert len(es._loss_history) == 2

        result = es.check(1.0, grad_norm=1e-10)
        assert result == StopReason.GRADIENT_BELOW_THRESHOLD
        assert es._stale_count == 2


# ======================================================================
#  Reset
# ======================================================================


class TestReset:
    """Tests for reset()."""

    def test_reset_clears_state(self):
        es = EarlyStopping(patience=3)
        # Build up some state
        es.check(1.0)
        assert es._stale_count == 0
        es.check(1.0)
        assert es._stale_count == 1

        es.reset()
        assert es._stale_count == 0
        assert es._tracked_best == float("inf")

        # After reset, patience counter should restart from zero
        assert es.check(1.0) is None
        assert es._stale_count == 0

        assert es.check(1.0) is None
        assert es._stale_count == 1

        assert es.check(1.0) is None
        assert es._stale_count == 2

        assert es.check(1.0) == StopReason.PATIENCE_EXCEEDED
        assert es._stale_count == 3

    def test_reset_clears_loss_history(self):
        es = EarlyStopping(patience=100, variance_window=3, variance_threshold=1e-6)
        # Fill window
        es.check(1.0)
        es.check(1.0)
        es.check(1.0)
        assert len(es._loss_history) == 3

        es.reset()
        assert len(es._loss_history) == 0

        # After reset, window is empty, so variance check should not fire
        assert es.check(1.0) is None
        assert len(es._loss_history) == 1

        assert es.check(1.0) is None
        assert len(es._loss_history) == 2

        # Window is now full
        result = es.check(1.0)
        assert result == StopReason.COST_VARIANCE_SETTLED
        assert len(es._loss_history) == 3
