# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import Future

import numpy as np
import pennylane as qml
import pytest
import sympy as sp

from divi.circuits import MetaCircuit
from divi.qprog import ProgramBatch, QuantumProgram


class SampleProgram(QuantumProgram):
    def __init__(self, circ_count, run_time, **kwargs):
        self.circ_count = circ_count
        self.run_time = run_time

        self.n_layers = 1
        self.n_params = 4

        super().__init__(backend=None, **kwargs)

        self._meta_circuits = self._create_meta_circuits_dict()

    def _create_meta_circuits_dict(self):

        def simple_circuit(params):
            qml.RX(params[0], wires=0)
            qml.U3(*params[1], wires=1)
            qml.CNOT(wires=[0, 1])

            return qml.expval(qml.PauliX(0) + qml.PauliZ(1) + qml.PauliX(0) @ qml.Z(1))

        # Create symbolic parameters
        symbols = [sp.Symbol("beta"), sp.symarray("theta", 3)]

        # Create a MetaCircuit using the grouping_strategy
        meta_circuit = MetaCircuit(
            main_circuit=qml.tape.make_qscript(simple_circuit)(symbols),
            symbols=symbols,
            grouping_strategy=self._grouping_strategy,
        )

        return {"cost_circuit": meta_circuit}

    def _generate_circuits(self, params=None, **kwargs):
        pass

    def run(self, store_data=False, data_file=None):
        return self.circ_count, self.run_time


class SampleProgramBatch(ProgramBatch):
    def __init__(self, backend):
        super().__init__(backend)
        self.max_iterations = 5

    def create_programs(self):
        super().create_programs()

        self.programs = {
            "prog1": SampleProgram(10, 5.5),
            "prog2": SampleProgram(5, 10),
        }

    def aggregate_results(self):
        pass


@pytest.fixture
def program_batch(default_test_simulator):
    batch = SampleProgramBatch(backend=default_test_simulator)
    yield batch
    try:
        batch.reset()
    except Exception:
        pass  # Don't break test teardown due to a race condition


@pytest.fixture(autouse=True)
def stop_live_display(program_batch):
    yield
    if (
        hasattr(program_batch, "_progress_bar")
        and program_batch._progress_bar is not None
    ):
        program_batch._progress_bar.stop()


class TestProgram:
    def test_correct_random_behavior(self, mocker):
        program = SampleProgram(10, 5.5, seed=1997)

        program.optimizer = mocker.MagicMock()
        program.optimizer.n_param_sets = 1

        assert (
            program._rng.bit_generator.state
            == np.random.default_rng(seed=1997).bit_generator.state
        )

        program._initialize_params()
        first_init = program._curr_params[0]
        assert first_init.shape == (program.n_layers * program.n_params,)

        program._initialize_params()
        second_init = program._curr_params[0]

        # Ensure we got different values
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, first_init, second_init
        )

    @pytest.fixture(scope="session")
    def expvals_collector(self):
        return []

    @pytest.mark.parametrize(
        "strategy,expected_n_groups,expected_n_diag",
        [[None, 3, 2], ["wires", 2, 2], ["qwc", 1, 1]],
    )
    def test_grouping_produces_expected_number_of_groups(
        self, strategy, expected_n_groups, expected_n_diag, expvals_collector, mocker
    ):
        program = SampleProgram(10, 5.5, seed=1997, grouping_strategy=strategy)
        program.loss_constant = 0.5
        program.optimizer = mocker.MagicMock()
        program.optimizer.n_param_sets = 1

        meta_circuit = program._meta_circuits["cost_circuit"]
        assert len(meta_circuit.measurement_groups) == expected_n_groups
        assert len(meta_circuit.measurements) == expected_n_groups
        assert (
            len(tuple(filter(lambda x: "h" in x, meta_circuit.measurements)))
            == expected_n_diag
        )

        program._initialize_params()
        fake_shot_histogram = {"00": 23, "01": 27, "10": 15, "11": 35}
        fake_results = {f"0_{i}": fake_shot_histogram for i in range(expected_n_groups)}
        expvals_collector.append(program._post_process_results(fake_results)[0])

    def test_assert_all_groupings_return_same_expval(self, expvals_collector):
        assert len(expvals_collector) == 3
        assert all(value == expvals_collector[0] for value in expvals_collector[1:])


class TestProgramBatch:
    def test_correct_initialization(self, program_batch):
        assert program_batch._executor is None
        assert len(program_batch.programs) == 0

    def test_programs_dict_is_correct(self, program_batch):
        program_batch.create_programs()
        assert len(program_batch.programs) == 2
        assert "prog1" in program_batch.programs
        assert "prog2" in program_batch.programs

    def test_reset(self, program_batch, mocker):
        program_batch.create_programs()
        program_batch._executor = mocker.MagicMock()

        program_batch.reset()

        assert program_batch.programs == {}
        assert program_batch._executor is None

    def test_total_circuit_count_setter(self, program_batch):
        with pytest.raises(
            AttributeError,
            match="property 'total_circuit_count' of 'SampleProgramBatch'",
        ):
            program_batch.total_circuit_count = 100

    def test_total_run_time_setter(self, program_batch):
        with pytest.raises(
            AttributeError,
            match="property 'total_run_time' of 'SampleProgramBatch'",
        ):
            program_batch.total_run_time = 100

    def test_run_sets_executor_and_returns_expected_number_of_futures(
        self, program_batch
    ):
        program_batch.create_programs()
        program_batch.run()
        assert program_batch._executor is not None
        assert len(program_batch.futures) == 2

    def test_run_fails_if_no_programs(self, program_batch):
        with pytest.raises(RuntimeError, match="No programs to run."):
            program_batch.run()

    def test_run_fails_if_already_running(self, mocker, program_batch):
        program_batch.create_programs()

        # Mock ProcessPoolExecutor to simulate a long-running batch
        mock_executor = mocker.patch("divi.qprog.batch.ProcessPoolExecutor")
        mock_instance = mock_executor.return_value

        # Create futures that simulate a long-running process
        future1 = Future()
        future2 = Future()
        mock_instance.submit.side_effect = [future1, future2]

        # First run should work
        program_batch.run()

        # Subsequent run should raise an exception
        with pytest.raises(RuntimeError, match="A batch is already being run."):
            program_batch.run()

    def test_run_executes_successfully(self, mocker, program_batch):
        """Test successful run of batch with multiple programs."""
        program_batch.create_programs()

        mock_executor_class = mocker.patch("divi.qprog.batch.ProcessPoolExecutor")

        # Create a mock executor instance.
        mock_executor = mocker.MagicMock()
        mock_executor_class.return_value = mock_executor

        # Run the batch
        program_batch.run()

        # Assert submit was called correct number of times
        assert mock_executor.submit.call_count == len(program_batch.programs)

        for program in program_batch.programs.values():
            mock_executor.submit.assert_any_call(program.run)

    def test_check_all_done_true_when_all_futures_ready(self, program_batch):
        future_1 = Future()
        future_2 = Future()
        program_batch.futures = [future_1, future_2]

        # Test when no futures are done
        assert not program_batch.check_all_done()

        # Complete one future
        future_1.set_result(None)
        assert not program_batch.check_all_done()

        # Complete second future
        future_2.set_result(None)
        assert program_batch.check_all_done()

    def test_wait_for_all_does_not_hold_if_all_ready(self, program_batch):
        program_batch.create_programs()
        program_batch.run()

        program_batch.wait_for_all()

        assert program_batch.total_circuit_count == 15
        assert program_batch.total_run_time == 15.5

        # Ensure instance is properly reset
        assert program_batch._executor is None
        assert len(program_batch.futures) == 0

    def test_wait_for_all_exception(self, mocker, program_batch):
        # Create an instance of your class and simulate an executor.
        program_batch._done_event = mocker.MagicMock()
        program_batch._listener_thread = mocker.MagicMock()
        program_batch._progress_bar = mocker.MagicMock()
        program_batch._executor = mocker.MagicMock()
        program_batch._executor.shutdown = mocker.MagicMock()

        # Create a mock future that fails when its result() is first called,
        # then returns a tuple (e.g., (1, 2)) when called again.
        failing_future = mocker.MagicMock()
        failing_future.result.side_effect = RuntimeError("Test error")

        # Set the futures list to contain our failing future.
        program_batch.futures = [failing_future]

        mocker.patch("divi.qprog.batch.as_completed", return_value=[failing_future])

        # The wait_for_all method should detect the exception from the as_completed loop,
        # perform the shutdown and then raise a RuntimeError.
        with pytest.raises(
            RuntimeError, match="One or more tasks failed. Check logs for details."
        ):
            program_batch.wait_for_all()
