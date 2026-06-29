# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Behavioural contracts for ``QuantumProgram``, ``VariationalQuantumAlgorithm``,
and ``ProgramEnsemble``.

``verify_correct_circuit_count`` recomputes expected counts using the same
measurement-grouping logic as production code. It catches accidental changes
to pipeline fan-out (regression guard), but does not prove numerical or
algorithmic correctness.
"""

import pytest

from divi.circuits import MetaCircuit
from divi.pipeline._grouping import _compute_measurement_groups
from divi.qprog import (
    MonteCarloOptimizer,
    ProgramEnsemble,
    QuantumProgram,
    ScipyMethod,
    ScipyOptimizer,
    VariationalQuantumAlgorithm,
)
from divi.qprog._solution_sampling_mixin import SolutionSamplingMixin


def verify_cost_circuit(obj: QuantumProgram) -> None:
    assert isinstance(
        obj.cost_circuit, MetaCircuit
    ), "cost_circuit must be a MetaCircuit"


def _get_n_obs_groups(meta: MetaCircuit, supports_expval: bool) -> int:
    """Compute the number of observable groups the MeasurementStage would produce.

    Note: this mirrors production grouping logic, so it acts as a regression
    guard (catches accidental changes to circuit counts) rather than a
    correctness test. If the production logic has a bug, this helper has
    the same bug.
    """
    if meta.observable is None:
        return 1

    strategy = "_backend_expval" if supports_expval else "qwc"
    groups, _, _, _ = _compute_measurement_groups(
        meta.observable, strategy, meta.n_qubits
    )
    return max(len(groups), 1)


def verify_correct_circuit_count(obj: QuantumProgram):
    """Verify the total circuit count matches the expected value for one iteration.

    Note: this recomputes the expected count using the same grouping logic as
    production code, so it serves as a regression guard — not a correctness
    proof. See ``_get_n_obs_groups`` for details.
    """
    assert obj.current_iteration == 1
    assert len(obj.losses_history) == 1

    supports_expval = obj.backend.supports_expval

    extra_computation_offset = 0
    if isinstance(obj, SolutionSamplingMixin):
        extra_computation_offset = 1  # sample_solution() runs one all-wires measurement

    adjusted_total_circuit_count = obj.total_circuit_count - extra_computation_offset

    if isinstance(obj.optimizer, MonteCarloOptimizer):
        circuits_per_param_set = _get_n_obs_groups(obj.cost_circuit, supports_expval)
        assert (
            adjusted_total_circuit_count
            == obj.optimizer.n_param_sets * circuits_per_param_set
        )
    elif isinstance(obj.optimizer, ScipyOptimizer):
        circuits_per_param_set = _get_n_obs_groups(obj.cost_circuit, supports_expval)
        if obj.optimizer.method in (ScipyMethod.NELDER_MEAD, ScipyMethod.COBYLA):
            assert (
                adjusted_total_circuit_count
                == obj.optimize_result.nfev * circuits_per_param_set
            )
        elif obj.optimizer.method == ScipyMethod.L_BFGS_B:
            evaluation_circuits_count = (
                obj.optimize_result.nfev * circuits_per_param_set
            )
            gradient_circuits_count = (
                obj.optimize_result.njev
                * circuits_per_param_set
                * obj.n_layers
                * obj.n_params_per_layer
                * 2
            )

            assert (
                adjusted_total_circuit_count
                == evaluation_circuits_count + gradient_circuits_count
            )


def verify_precision_kwarg_threads_through(make_program) -> None:
    """``precision=`` reaches ``QuantumProgram._precision`` regardless of
    the subclass's own ``__init__`` shape."""
    program = make_program(precision=4)
    assert program.precision == 4
    assert program._precision == 4


def verify_grouping_strategy_kwarg_threads_through(make_program) -> None:
    """``grouping_strategy=`` reaches the ``ObservableMeasuringMixin``
    field on the constructed program."""
    program = make_program(grouping_strategy="wires")
    assert program._grouping_strategy == "wires"


def verify_shot_distribution_kwarg_threads_through(make_program) -> None:
    """``shot_distribution=`` reaches the ``ObservableMeasuringMixin``
    field on the constructed program."""
    program = make_program(shot_distribution="weighted")
    assert program._shot_distribution == "weighted"


OBSERVABLE_MEASURING_CONTRACTS = [
    (verify_precision_kwarg_threads_through, "precision"),
    (verify_grouping_strategy_kwarg_threads_through, "grouping_strategy"),
    (verify_shot_distribution_kwarg_threads_through, "shot_distribution"),
]


class ObservableMeasuringContractsBase:
    """Base class wiring ``OBSERVABLE_MEASURING_CONTRACTS`` to a program
    test class. Subclasses must provide a ``make_program`` fixture returning
    a callable ``make_program(**kwargs) -> QuantumProgram`` that builds the
    program type under test with arbitrary mixin kwargs.
    """

    @pytest.mark.parametrize(
        "contract", OBSERVABLE_MEASURING_CONTRACTS, ids=lambda c: c[1]
    )
    def test_contract(self, contract, make_program):
        verify, _ = contract
        verify(make_program)


def verify_basic_program_ensemble_behaviour(obj: ProgramEnsemble, mocker) -> None:
    with pytest.raises(RuntimeError, match="No programs to run"):
        obj.run()

    with pytest.raises(RuntimeError, match="No programs to aggregate"):
        obj.aggregate_results()

    obj.programs = {"dummy": "program"}

    with pytest.raises(RuntimeError, match="Some programs already exist"):
        obj.create_programs()

    mock_program = mocker.MagicMock(spec=VariationalQuantumAlgorithm)
    mock_program.has_results.return_value = False

    obj.programs = {"dummy": mock_program}

    with pytest.raises(RuntimeError, match="Some/All programs have no results"):
        obj.aggregate_results()
