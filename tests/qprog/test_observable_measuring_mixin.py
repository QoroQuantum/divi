# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Direct tests for ``ObservableMeasuringMixin``."""

import warnings

import pytest

from divi.qprog import ObservableMeasuringMixin
from divi.qprog.quantum_program import QuantumProgram


class ConcreteObservableMeasuringProgram(ObservableMeasuringMixin, QuantumProgram):
    """Minimal concrete program for exercising the mixin in isolation."""

    def _build_pipelines(self) -> None:
        pass

    def has_results(self) -> bool:
        return False

    def run(self):
        return self


class TestMixinDefaults:
    """Defaults applied when no kwargs are passed."""

    def test_grouping_strategy_defaults_to_qwc(self, dummy_simulator):
        program = ConcreteObservableMeasuringProgram(backend=dummy_simulator)
        assert program._grouping_strategy == "qwc"

    def test_shot_distribution_defaults_to_none(self, dummy_simulator):
        program = ConcreteObservableMeasuringProgram(backend=dummy_simulator)
        assert program._shot_distribution is None


class TestVerbatimStorage:
    """Explicit kwargs are stored as passed; no construction-time auto-flip."""

    @pytest.mark.parametrize("grouping_strategy", ["qwc", "default", "wires", None])
    def test_explicit_grouping_stored_verbatim(
        self, dummy_simulator, grouping_strategy
    ):
        program = ConcreteObservableMeasuringProgram(
            backend=dummy_simulator, grouping_strategy=grouping_strategy
        )
        assert program._grouping_strategy == grouping_strategy

    @pytest.mark.parametrize(
        "shot_distribution", ["uniform", "weighted", "weighted_random"]
    )
    def test_explicit_shot_distribution_stored_verbatim(
        self, dummy_simulator, shot_distribution
    ):
        program = ConcreteObservableMeasuringProgram(
            backend=dummy_simulator, shot_distribution=shot_distribution
        )
        assert program._shot_distribution == shot_distribution

    def test_callable_shot_distribution_stored_verbatim(self, dummy_simulator):
        def custom(norms, total):
            return [total] + [0] * (len(norms) - 1)

        program = ConcreteObservableMeasuringProgram(
            backend=dummy_simulator, shot_distribution=custom
        )
        assert program._shot_distribution is custom


class TestValidation:
    """Construction-time validation of user inputs."""

    @pytest.mark.parametrize(
        "grouping_strategy",
        ["_backend_expval", "graph-coloring", "QWC", "", 42],
    )
    def test_invalid_grouping_strategy_rejected(
        self, dummy_simulator, grouping_strategy
    ):
        """Any value outside the user-facing set ({qwc, default, wires, None})
        is rejected. ``"_backend_expval"`` is internal-only; the rest are
        typos / wrong type."""
        with pytest.raises(ValueError, match="Invalid grouping_strategy"):
            ConcreteObservableMeasuringProgram(
                backend=dummy_simulator, grouping_strategy=grouping_strategy
            )

    @pytest.mark.parametrize(
        "grouping_strategy,shot_distribution",
        [
            ("qwc", "weighted"),
            ("wires", "uniform"),
            (None, "weighted_random"),
            ("qwc", None),
            (None, None),
        ],
    )
    def test_compatible_combinations_do_not_raise(
        self, dummy_simulator, grouping_strategy, shot_distribution
    ):
        ConcreteObservableMeasuringProgram(
            backend=dummy_simulator,
            grouping_strategy=grouping_strategy,
            shot_distribution=shot_distribution,
        )


class TestOverrideWarning:
    """Construction-time UserWarning when an explicit grouping_strategy
    will likely be overridden by MeasurementStage at runtime."""

    @pytest.mark.parametrize("grouping_strategy", ["qwc", "wires"])
    def test_warns_when_explicit_grouping_with_expval_backend(
        self, dummy_expval_backend, grouping_strategy
    ):
        with pytest.warns(UserWarning, match="may be auto-overridden"):
            ConcreteObservableMeasuringProgram(
                backend=dummy_expval_backend, grouping_strategy=grouping_strategy
            )

    def test_does_not_warn_on_default_grouping_with_expval_backend(
        self, dummy_expval_backend
    ):
        """The warning targets *explicit* preferences; accepting the
        default must not warn or every program-on-expval-backend would
        emit noise."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ConcreteObservableMeasuringProgram(backend=dummy_expval_backend)

    def test_does_not_warn_on_explicit_none(self, dummy_expval_backend):
        """``None`` (one circuit per term) is never overridden, so it must
        not warn even on an expval-capable backend."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ConcreteObservableMeasuringProgram(
                backend=dummy_expval_backend, grouping_strategy=None
            )

    def test_does_not_warn_when_shot_distribution_set(self, dummy_expval_backend):
        """``shot_distribution`` declares sampling intent — the runtime
        will not flip to ``"_backend_expval"`` regardless of grouping
        choice, so the warning is suppressed."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ConcreteObservableMeasuringProgram(
                backend=dummy_expval_backend,
                grouping_strategy="qwc",
                shot_distribution="weighted",
            )

    def test_does_not_warn_on_non_expval_backend(self, dummy_simulator):
        """If the backend does not support analytic expval, no auto-flip
        is possible, so no warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ConcreteObservableMeasuringProgram(
                backend=dummy_simulator, grouping_strategy="qwc"
            )


class TestKwargForwarding:
    """The mixin's ``super().__init__()`` must reach ``QuantumProgram``
    so base-class kwargs (backend, seed, precision, …) still work."""

    def test_backend_threaded_to_quantum_program(self, dummy_simulator):
        program = ConcreteObservableMeasuringProgram(backend=dummy_simulator)
        assert program.backend is dummy_simulator

    def test_seed_threaded_to_quantum_program(self, dummy_simulator):
        program = ConcreteObservableMeasuringProgram(backend=dummy_simulator, seed=42)
        assert program._seed == 42

    def test_precision_threaded_to_quantum_program(self, dummy_simulator):
        program = ConcreteObservableMeasuringProgram(
            backend=dummy_simulator, precision=4
        )
        assert program._precision == 4
        assert program.precision == 4

    def test_unknown_kwarg_still_raises(self, dummy_simulator):
        """The mixin does not absorb unknown kwargs — they must reach
        ``QuantumProgram``'s strict check and raise."""
        with pytest.raises(TypeError, match="Unexpected keyword argument"):
            ConcreteObservableMeasuringProgram(
                backend=dummy_simulator, mystery_kwarg=True
            )


class TestMROGuard:
    """``__init_subclass__`` rejects subclasses where the mixin sits
    after :class:`QuantumProgram` in the inheritance order."""

    def test_wrong_order_raises_at_class_definition(self):
        with pytest.raises(TypeError, match="must precede QuantumProgram"):

            class _WrongOrder(QuantumProgram, ObservableMeasuringMixin):
                def _build_pipelines(self):
                    pass

                def has_results(self):
                    return False

                def run(self):
                    return self

    def test_correct_order_succeeds(self):
        class _RightOrder(ObservableMeasuringMixin, QuantumProgram):
            def _build_pipelines(self):
                pass

            def has_results(self):
                return False

            def run(self):
                return self

        mro = _RightOrder.__mro__
        assert mro.index(ObservableMeasuringMixin) < mro.index(QuantumProgram)
