# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import time
from concurrent.futures import Future, ProcessPoolExecutor
from functools import partial
from multiprocessing import Event, Manager
from queue import Queue
from threading import Lock, Thread

import numpy as np
import pennylane as qml
import pytest
import sympy as sp
from mitiq.zne.inference import LinearFactory
from mitiq.zne.scaling import fold_global
from rich.progress import Progress

from divi.circuits import MetaCircuit
from divi.circuits.qem import ZNE
from divi.qprog.batch import ProgramBatch, QuantumProgram, queue_listener
from tests.qprog.qprog_contracts import verify_basic_program_batch_behaviour


class SampleProgram(QuantumProgram):
    def __init__(self, circ_count, run_time, **kwargs):
        self.circ_count = circ_count
        self.run_time = run_time

        self.n_layers = 1
        self._n_params = 4

        super().__init__(backend=None, **kwargs)

        self._meta_circuits = self._create_meta_circuits_dict()

    def _create_meta_circuits_dict(self):

        def simple_circuit(params):
            qml.RX(params[0], wires=0)
            qml.U3(*params[1], wires=1)
            qml.CNOT(wires=[0, 1])

            return qml.expval(
                qml.PauliX(0) + qml.PauliZ(1) + qml.PauliX(0) @ qml.PauliZ(1)
            )

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
    """A mock ProgramBatch for testing."""

    def __init__(self, backend):
        super().__init__(backend)
        self.max_iterations = 5

    def create_programs(self):
        """Creates a set of mock programs."""
        super().create_programs()
        self.programs = {
            "prog1": SampleProgram(10, 5.5, job_id="prog1"),
            "prog2": SampleProgram(5, 10.0, job_id="prog2"),
        }

    def aggregate_results(self):
        """A mock aggregation function."""
        # The super() call is important to trigger checks and the join()
        super().aggregate_results()
        return sum(p.circ_count for p in self.programs.values())


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
    """Fixture to automatically stop any active Rich progress bars after a test."""
    yield
    if (
        hasattr(program_batch, "_progress_bar")
        and program_batch._progress_bar is not None
        and not program_batch._progress_bar.finished
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
        fake_results = {
            f"0_mock-qem:0_{i}": fake_shot_histogram for i in range(expected_n_groups)
        }
        expvals_collector.append(program._post_process_results(fake_results)[0])

    def test_assert_all_groupings_return_same_expval(self, expvals_collector):
        assert len(expvals_collector) == 3
        assert all(value == expvals_collector[0] for value in expvals_collector[1:])

    def test_post_process_with_zne(self):
        """
        Tests that _post_process_results correctly applies ZNE extrapolation.
        """
        # 1. Setup Program with ZNE Protocol
        scale_factors = [1.0, 2.0, 3.0]
        zne_protocol = ZNE(
            folding_fn=partial(fold_global),
            scale_factors=scale_factors,
            # The traceback shows you used LinearFactory, so we'll stick with that.
            extrapolation_factory=LinearFactory(scale_factors=scale_factors),
        )
        program = SampleProgram(10, 5.5, seed=1997, qem_protocol=zne_protocol)
        program.loss_constant = 0.0

        # 2. Create mock shot histograms.
        # The shots below are crafted to produce expectation values of 0.9, 0.8, and 0.7
        # for the scale factors 1.0, 2.0, and 3.0, respectively.
        mock_shots_per_sf = [
            {"00": 95, "11": 5},  # Produces expval of 0.9
            {"00": 90, "11": 10},  # Produces expval of 0.8
            {"00": 85, "11": 15},  # Produces expval of 0.7
        ]

        mock_results = {}
        n_measurement_groups = 3
        # Create the full results dictionary for ALL 3 measurement groups.
        for qem_run_id, shots in enumerate(mock_shots_per_sf):
            for meas_group_id in range(n_measurement_groups):
                key = f"0_zne:{qem_run_id}_{meas_group_id}"
                mock_results[key] = shots

        # 3. Call the function and assert the outcome
        final_losses = program._post_process_results(mock_results)

        # With 3 measurement groups, the final loss is a sum of the 3 mitigated expectation values.
        # Assuming each observable term behaves identically under noise for this test,
        # the mitigated value for each is 1.0. The final loss (sum) should be 3.0.
        # Note: The actual postprocessing_fn for a Sum observable will sum the results.
        assert np.isclose(final_losses[0], 3.0)


class TestProgramBatch:
    def test_correct_initialization(self, program_batch):
        assert program_batch._executor is None
        assert len(program_batch.programs) == 0
        assert program_batch.total_circuit_count == 0
        assert program_batch.total_run_time == 0.0

    def test_basic_program_batch_behaviour(self, program_batch, mocker):
        """Uses the contract to verify basic error handling and state checks."""
        verify_basic_program_batch_behaviour(mocker, program_batch)

    def test_programs_dict_is_correct(self, program_batch):
        program_batch.create_programs()
        assert len(program_batch.programs) == 2
        assert "prog1" in program_batch.programs
        assert "prog2" in program_batch.programs
        assert isinstance(program_batch._manager, type(Manager()))
        assert hasattr(program_batch._queue, "get")  # Check if it's queue-like
        assert isinstance(program_batch._done_event, type(Event()))

    def test_reset_cleans_up_all_resources(self, program_batch, mocker):
        """Tests that reset() correctly shuts down and cleans up all resources."""
        # First, create programs to initialize the manager, queue, etc.
        program_batch.create_programs()
        # Now, simulate a running state by creating an executor and futures
        program_batch._executor = mocker.MagicMock(spec=ProcessPoolExecutor)
        program_batch._listener_thread = mocker.MagicMock(spec=Thread)
        program_batch._progress_bar = mocker.MagicMock()
        program_batch.futures = [mocker.MagicMock()]
        program_batch._pb_task_map = {}  # This was the missing attribute

        # Configure the mock to behave like a stopped thread to prevent warnings
        program_batch._listener_thread.is_alive.return_value = False

        # Spy on the original objects before they are cleared by reset()
        mock_executor = program_batch._executor
        manager_shutdown_spy = mocker.spy(program_batch._manager, "shutdown")
        done_event_set_spy = mocker.spy(program_batch._done_event, "set")
        mock_listener_thread = program_batch._listener_thread
        mock_progress_bar = program_batch._progress_bar

        # Call the method under test
        program_batch.reset()

        # Assert that cleanup methods were called on the spied objects
        mock_executor.shutdown.assert_called_once_with(wait=False)
        done_event_set_spy.assert_called_once()
        mock_listener_thread.join.assert_called_once()
        manager_shutdown_spy.assert_called_once()
        mock_progress_bar.stop.assert_called_once()

        # Assert that state attributes are cleared
        assert program_batch._executor is None
        assert program_batch.futures is None
        assert program_batch._manager is None

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

    def test_run_submits_correct_tasks(self, program_batch, mocker):
        """Tests that run() submits the correct function and arguments to the executor."""
        program_batch.create_programs()
        mock_executor_class = mocker.patch("divi.qprog.batch.ProcessPoolExecutor")
        mock_executor = mock_executor_class.return_value

        # The executor's submit method returns Future objects. When we mock the executor,
        # it returns MagicMocks by default. The `join` method will hang waiting for
        # these mock futures to complete.
        mock_future_1 = mocker.MagicMock(spec=Future)
        mock_future_2 = mocker.MagicMock(spec=Future)
        mock_executor.submit.side_effect = [mock_future_1, mock_future_2]

        # To prevent `join` from hanging, we must also mock `as_completed`. This function
        # normally yields futures as they complete. We make it yield our mocks immediately.
        mocker.patch(
            "divi.qprog.batch.as_completed", return_value=[mock_future_1, mock_future_2]
        )

        # Run non-blocking to inspect the state before it's cleaned up
        program_batch.run(blocking=False)

        assert mock_executor.submit.call_count == len(program_batch.programs)

        # The function submitted should be the internal _task_fn
        task_fn = program_batch._task_fn
        for program in program_batch.programs.values():
            mock_executor.submit.assert_any_call(task_fn, program)

        # Clean up the non-blocking run. This should now terminate correctly.
        program_batch.join()

    def test_blocking_run_executes_and_joins_correctly(self, program_batch):
        """Integration test for a standard blocking run."""
        program_batch.create_programs()
        # run(blocking=True) is the default, which should execute and then join.
        result = program_batch.run(blocking=True)

        assert result is program_batch
        assert program_batch.total_circuit_count == 15
        assert program_batch.total_run_time == 15.5
        # After a blocking run, the executor should be gone.
        assert program_batch._executor is None
        assert len(program_batch.futures) == 0

    def test_non_blocking_run_registers_atexit_hook(self, program_batch, mocker):
        """Tests that a non-blocking run correctly registers a cleanup hook."""
        mock_atexit_register = mocker.patch("atexit.register")
        program_batch.create_programs()
        program_batch.run(blocking=False)

        # Check that the executor is active and hook is registered
        assert program_batch._executor is not None
        mock_atexit_register.assert_called_once_with(program_batch._atexit_cleanup_hook)

        # Manually clean up to avoid side effects
        program_batch.join()

    def test_join_unregisters_atexit_hook(self, program_batch, mocker):
        """Tests that join() unregisters the cleanup hook after a non-blocking run."""
        mock_atexit_register = mocker.patch("atexit.register")
        mock_atexit_unregister = mocker.patch("atexit.unregister")

        program_batch.create_programs()
        program_batch.run(blocking=False)  # This registers the hook

        mock_atexit_register.assert_called_once()

        program_batch.join()  # This should unregister it

        mock_atexit_unregister.assert_called_once_with(
            program_batch._atexit_cleanup_hook
        )

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

    def test_join_does_not_hold_if_all_ready(self, program_batch):
        program_batch.create_programs()
        program_batch.run()

        program_batch.join()

        assert program_batch.total_circuit_count == 15
        assert program_batch.total_run_time == 15.5

        # Ensure instance is properly reset
        assert program_batch._executor is None
        assert len(program_batch.futures) == 0

    def test_join_handles_task_exceptions(self, program_batch, mocker):
        """Ensures join() catches exceptions from futures and still cleans up."""
        program_batch.create_programs()
        program_batch.run(blocking=False)  # Start a non-blocking run

        # Mock the futures to simulate one failing task
        failing_future = Future()
        failing_future.set_exception(ValueError("Task failed"))
        successful_future = Future()
        successful_future.set_result((5, 5.0))
        program_batch.futures = [failing_future, successful_future]

        # Mock executor shutdown to confirm it's called during cleanup
        mock_shutdown = mocker.spy(program_batch._executor, "shutdown")

        with pytest.raises(RuntimeError, match="One or more tasks failed"):
            program_batch.join()

        # Verify that cleanup still happened
        mock_shutdown.assert_called_once_with(wait=True, cancel_futures=False)
        assert program_batch._executor is None  # Should be cleared after join

    def test_aggregate_results_calls_join_and_aggregates(self, program_batch):
        """Tests that aggregate_results works correctly after a successful run."""
        program_batch.create_programs()
        # Run to completion. This executes the programs in separate processes.
        program_batch.run(blocking=True)

        # The `losses` list is updated in the child process, but not in the parent.
        # To pass the check in `aggregate_results`, we manually populate the list
        # here, simulating a successful run where state was propagated.
        for p in program_batch.programs.values():
            p.losses.append(0.1)

        # Now, aggregate_results should pass its internal checks and work.
        # It should not call join() again as the executor is already shut down.
        result = program_batch.aggregate_results()

        assert result == 15  # 10 + 5 from SampleProgram circ_counts


def test_queue_listener(mocker):
    """Unit test for the queue_listener function."""
    # Setup mock objects
    mock_queue = Queue()
    mock_progress_bar = mocker.MagicMock(spec=Progress)
    mock_done_event = Event()
    lock = Lock()
    pb_task_map = {"job1": 1, "job2": 2}

    # Put some messages in the queue
    mock_queue.put(
        {
            "job_id": "job1",
            "progress": 1,
            "message": "step 1",
            "final_status": "running",
        }
    )
    mock_queue.put({"job_id": "job2", "progress": 1, "poll_attempt": 3})

    # Run the listener in a separate thread
    listener_thread = Thread(
        target=queue_listener,
        args=(mock_queue, mock_progress_bar, pb_task_map, mock_done_event, False, lock),
    )
    listener_thread.start()

    # Give the thread a moment to process messages
    time.sleep(0.1)

    # Signal the thread to stop
    mock_done_event.set()
    listener_thread.join(timeout=1)
    assert not listener_thread.is_alive()

    # Check that the progress bar was updated correctly
    common_kwargs = {
        "advance": 1,
        "refresh": False,
    }
    expected_calls = [
        mocker.call(
            1,
            message="step 1",
            final_status="running",
            **common_kwargs,
        ),
        mocker.call(
            2,
            poll_attempt=3,
            **common_kwargs,
        ),
    ]

    mock_progress_bar.update.assert_has_calls(expected_calls, any_order=True)
    assert mock_queue.empty()
