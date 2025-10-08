# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from divi.qprog import (
    MonteCarloOptimizer,
    ProgramBatch,
    QuantumProgram,
    ScipyMethod,
    ScipyOptimizer,
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


def verify_metacircuit_dict(obj, expected_keys):
    from divi.circuits import MetaCircuit

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

    assert len(obj.losses) == 1

    if isinstance(obj.optimizer, MonteCarloOptimizer):
        assert obj.total_circuit_count == obj.optimizer.n_param_sets * len(
            obj.cost_hamiltonian
        )
    elif isinstance(obj.optimizer, ScipyOptimizer):
        if obj.optimizer.method in (ScipyMethod.NELDER_MEAD, ScipyMethod.COBYLA):
            assert obj.total_circuit_count == obj._minimize_res.nfev * len(
                obj.cost_hamiltonian
            )
        elif obj.optimizer.method == ScipyMethod.L_BFGS_B:
            evaluation_circuits_count = obj._minimize_res.nfev * len(
                obj.cost_hamiltonian
            )

            gradient_circuits_count = (
                obj._minimize_res.njev * len(obj.cost_hamiltonian) * obj.n_params * 2
            )

            assert (
                obj.total_circuit_count
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
