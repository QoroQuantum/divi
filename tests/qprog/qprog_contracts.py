# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from divi.circuits import MetaCircuit
from divi.qprog import (
    MonteCarloOptimizer,
    ProgramBatch,
    QuantumProgram,
    ScipyMethod,
    ScipyOptimizer,
    VariationalQuantumAlgorithm,
)
from divi.qprog.optimizers import PymooMethod, PymooOptimizer

OPTIMIZERS_TO_TEST = {
    "argvalues": [
        lambda: MonteCarloOptimizer(population_size=5, n_best_sets=2),
        lambda: ScipyOptimizer(method=ScipyMethod.L_BFGS_B),
        lambda: ScipyOptimizer(method=ScipyMethod.COBYLA),
        lambda: ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        lambda: PymooOptimizer(method=PymooMethod.CMAES, population_size=10),
        lambda: PymooOptimizer(method=PymooMethod.DE, population_size=5),
    ],
    "ids": ["MonteCarlo", "L_BFGS_B", "COBYLA", "NELDER_MEAD", "CMAES", "DE"],
}

# Optimizers that support checkpointing - derived from OPTIMIZERS_TO_TEST
# Filters out ScipyOptimizer which doesn't support checkpointing
CHECKPOINTING_OPTIMIZERS = {
    "argvalues": [
        factory
        for factory, opt_id in zip(
            OPTIMIZERS_TO_TEST["argvalues"], OPTIMIZERS_TO_TEST["ids"]
        )
        if opt_id
        not in ["L_BFGS_B", "COBYLA", "NELDER_MEAD"]  # ScipyOptimizer variants
    ],
    "ids": [
        opt_id
        for opt_id in OPTIMIZERS_TO_TEST["ids"]
        if opt_id not in ["L_BFGS_B", "COBYLA", "NELDER_MEAD"]
    ],
}


def verify_metacircuit_dict(obj: QuantumProgram, expected_keys: list[str]):
    assert isinstance(obj.meta_circuit_factories, dict)
    assert hasattr(
        obj, "meta_circuit_factories"
    ), "Meta circuits attribute does not exist"
    assert isinstance(
        obj.meta_circuit_factories, dict
    ), "Meta circuits object not a dict"
    assert all(
        isinstance(val, MetaCircuit) for val in obj.meta_circuit_factories.values()
    ), "All values on meta circuit must be of type MetaCircuit"
    assert all(
        key == expected
        for key, expected in zip(obj.meta_circuit_factories.keys(), expected_keys)
    )


def verify_correct_circuit_count(obj: QuantumProgram):
    assert obj.current_iteration == 1
    assert len(obj.losses_history) == 1

    extra_computation_offset = 0
    if isinstance(obj, VariationalQuantumAlgorithm):
        # VQA subclasses with a meas_circuit will run an extra computation
        if "meas_circuit" in obj.meta_circuit_factories:
            meas_groups = obj.meta_circuit_factories["meas_circuit"].measurement_groups
            # ProbsMeasurementStage produces 1 circuit without measurement
            # groups; ObservableGroupingStage produces len(groups) circuits.
            extra_computation_offset = max(len(meas_groups), 1)

    adjusted_total_circuit_count = obj.total_circuit_count - extra_computation_offset

    if isinstance(obj.optimizer, MonteCarloOptimizer):
        # Calculate expected circuits per parameter set based on measurement groups.
        # max(len, 1) handles cases where groups are empty (e.g. TrotterSpecStage
        # creates MetaCircuits internally, so the reference cost_circuit may not
        # have measurement_groups set).
        circuits_per_param_set = max(
            len(obj.meta_circuit_factories["cost_circuit"].measurement_groups), 1
        )
        assert (
            adjusted_total_circuit_count
            == obj.optimizer.n_param_sets * circuits_per_param_set
        )
    elif isinstance(obj.optimizer, ScipyOptimizer):
        circuits_per_param_set = max(
            len(obj.meta_circuit_factories["cost_circuit"].measurement_groups), 1
        )
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


def verify_basic_program_batch_behaviour(mocker, obj: ProgramBatch):

    with pytest.raises(RuntimeError, match="No programs to run"):
        obj.run()

    with pytest.raises(RuntimeError, match="No programs to aggregate"):
        obj.aggregate_results()

    obj.programs = {"dummy": "program"}

    with pytest.raises(RuntimeError, match="Some programs already exist"):
        obj.create_programs()

    mock_program = mocker.MagicMock()

    obj.programs = {"dummy": mock_program}

    with pytest.raises(RuntimeError, match="Some/All programs have empty losses"):
        obj.aggregate_results()
