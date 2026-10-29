# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Direct tests for ``ObservableMeasuringMixin``."""

import numpy as np
import pytest

from divi.qprog import ObservableMeasuringMixin
from divi.qprog.quantum_program import QuantumProgram


class ConcreteObservableMeasuringProgram(ObservableMeasuringMixin, QuantumProgram):
    """Minimal concrete program for exercising the mixin in isolation."""

    def has_results(self) -> bool:
        return False

    def run(self):
        return self


def test_mixin_only_subclass_defines_without_quantum_program():
    partial = type("_PartialObservableMeasuring", (ObservableMeasuringMixin,), {})

    assert issubclass(partial, ObservableMeasuringMixin)


@pytest.mark.parametrize(
    "attribute, expected",
    [
        pytest.param("_grouping_strategy", "qwc", id="grouping_strategy_qwc"),
        pytest.param("_shot_distribution", None, id="shot_distribution_none"),
        pytest.param("_measure_all_qubits", False, id="measure_all_qubits_false"),
    ],
)
def test_mixin_defaults(dummy_simulator, attribute, expected):
    """Defaults applied when no kwargs are passed."""
    program = ConcreteObservableMeasuringProgram(backend=dummy_simulator)
    assert getattr(program, attribute) == expected
    assert type(getattr(program, attribute)) is type(expected)


class TestVerbatimStorage:
    """Explicit kwargs are stored as passed; no construction-time auto-flip."""

    @pytest.mark.parametrize(
        "grouping_strategy",
        [
            "qwc",
            "default",
            "wires",
            "shadows",
            "biased_shadows",
            "derandomised_shadows",
            None,
        ],
    )
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

    @pytest.mark.parametrize("measure_all_qubits", [True, False])
    def test_measure_all_qubits_stored_verbatim(
        self, dummy_simulator, measure_all_qubits
    ):
        program = ConcreteObservableMeasuringProgram(
            backend=dummy_simulator, measure_all_qubits=measure_all_qubits
        )
        assert program._measure_all_qubits is measure_all_qubits

    @pytest.mark.parametrize("measure_all_qubits", [True, False])
    def test_measure_all_qubits_threaded_to_measurement_stage(
        self, dummy_simulator, measure_all_qubits
    ):
        program = ConcreteObservableMeasuringProgram(
            backend=dummy_simulator, measure_all_qubits=measure_all_qubits
        )
        stage = program._make_measurement_stage()
        assert stage._measure_all is measure_all_qubits


class TestValidation:
    """Construction-time validation of user inputs."""

    @pytest.mark.parametrize(
        "grouping_strategy",
        ["_backend_expval", "graph-coloring", "QWC", "", 42],
    )
    def test_invalid_grouping_strategy_rejected(
        self, dummy_simulator, grouping_strategy
    ):
        """Any value outside the user-facing set ({qwc, default, wires, shadows,
        biased_shadows, derandomised_shadows, None}) is rejected.
        ``"_backend_expval"`` is internal-only; the rest are typos / wrong type."""
        with pytest.raises(ValueError, match="Invalid grouping_strategy"):
            ConcreteObservableMeasuringProgram(
                backend=dummy_simulator, grouping_strategy=grouping_strategy
            )

    @pytest.mark.parametrize(
        "bad_value", ["true", 1, None, 0, np.int64(1), np.float64(1.0)]
    )
    def test_non_bool_measure_all_qubits_rejected(self, dummy_simulator, bad_value):
        """A non-bool measure_all_qubits (e.g. a typo'd string) would silently
        act as truthy; reject it instead of quietly forcing full measurement."""
        with pytest.raises(TypeError, match="measure_all_qubits must be a bool"):
            ConcreteObservableMeasuringProgram(
                backend=dummy_simulator, measure_all_qubits=bad_value
            )

    @pytest.mark.parametrize("value", [np.True_, np.False_])
    def test_numpy_bool_measure_all_qubits_is_stored_as_a_plain_bool(
        self, dummy_simulator, value
    ):
        program = ConcreteObservableMeasuringProgram(
            backend=dummy_simulator, measure_all_qubits=value
        )

        assert program._measure_all_qubits is bool(value)

    @pytest.mark.parametrize(
        "grouping_strategy", ["shadows", "biased_shadows", "derandomised_shadows"]
    )
    def test_shadows_with_shot_distribution_rejected_at_construction(
        self, dummy_simulator, grouping_strategy
    ):
        with pytest.raises(ValueError, match="cannot be combined with"):
            ConcreteObservableMeasuringProgram(
                backend=dummy_simulator,
                grouping_strategy=grouping_strategy,
                shot_distribution="weighted",
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


def test_mixin_after_quantum_program_raises_at_class_definition():
    """``__init_subclass__`` rejects subclasses where the mixin sits
    after :class:`QuantumProgram` in the inheritance order."""
    with pytest.raises(TypeError, match="must precede QuantumProgram"):

        class _WrongOrder(QuantumProgram, ObservableMeasuringMixin):
            def has_results(self):
                return False

            def run(self):
                return self
