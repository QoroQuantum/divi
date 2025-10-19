# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
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
        MonteCarloOptimizer(n_param_sets=10, n_best_sets=3),
        ScipyOptimizer(method=ScipyMethod.L_BFGS_B),
        ScipyOptimizer(method=ScipyMethod.COBYLA),
        ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        PymooOptimizer(method=PymooMethod.CMAES, population_size=10),
        PymooOptimizer(method=PymooMethod.DE, population_size=10),
    ],
    "ids": ["MonteCarlo", "L_BFGS_B", "COBYLA", "NELDER_MEAD", "CMAES", "DE"],
}


def verify_metacircuit_dict(obj: QuantumProgram, expected_keys: list[str]):
    assert isinstance(obj.meta_circuits, dict)
    assert hasattr(obj, "_meta_circuits"), "Meta circuits attribute does not exist"
    assert isinstance(obj._meta_circuits, dict), "Meta circuits object not a dict"
    assert all(
        isinstance(val, MetaCircuit) for val in obj._meta_circuits.values()
    ), "All values on meta circuit must be of type MetaCircuit"
    assert all(
        key == expected
        for key, expected in zip(obj._meta_circuits.keys(), expected_keys)
    )


def verify_correct_circuit_count(obj: QuantumProgram):
    assert obj.current_iteration == 1
    assert len(obj.losses_history) == 1

    extra_computation_offset = 0
    if isinstance(obj, VariationalQuantumAlgorithm):
        # VQA subclasses with a meas_circuit will run an extra computation
        if "meas_circuit" in obj.meta_circuits:
            extra_computation_offset = len(
                obj.meta_circuits["meas_circuit"].measurement_groups
            )

    adjusted_total_circuit_count = obj.total_circuit_count - extra_computation_offset

    if isinstance(obj.optimizer, MonteCarloOptimizer):
        # Calculate expected circuits per parameter set based on measurement groups
        circuits_per_param_set = len(
            obj._meta_circuits["cost_circuit"].measurement_groups
        )
        assert (
            adjusted_total_circuit_count
            == obj.optimizer.n_param_sets * circuits_per_param_set
        )
    elif isinstance(obj.optimizer, ScipyOptimizer):
        circuits_per_param_set = len(
            obj._meta_circuits["cost_circuit"].measurement_groups
        )
        if obj.optimizer.method in (ScipyMethod.NELDER_MEAD, ScipyMethod.COBYLA):
            assert (
                adjusted_total_circuit_count
                == obj._minimize_res.nfev * circuits_per_param_set
            )
        elif obj.optimizer.method == ScipyMethod.L_BFGS_B:
            evaluation_circuits_count = obj._minimize_res.nfev * circuits_per_param_set
            gradient_circuits_count = (
                obj._minimize_res.njev * circuits_per_param_set * obj.n_params * 2
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
