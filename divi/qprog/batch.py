import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed


class ProgramBatch(ABC):
    """This abstract class provides the basic scaffolding for higher-order
    computations that require more than one quantum program to achieve its goal.

    Each implementation of this class has to have an implementation of two functions:
        1. `create_programs`: This function generates the independent programs that
            are needed to achieve the objective of the job. The creation of those
            programs can utilize the instance variables of the class to initialize
            their parameters. The programs should be stored in a key-value store
            where the keys represent the identifier of the program, whether random
            or identificatory.

        2. `aggregate_results`: This function aggregates the results of the programs
            after they are done executing. This function should be aware of the different
            formats the programs might have (counts dictionary, expectation value, etc) and
            handle such cases accordingly.
    """

    def __init__(self):
        super().__init__()

        self._executor = None
        self.programs = {}

        self._total_circuit_count = 0
        self._total_run_time = 0

    @property
    def total_circuit_count(self):
        return self._total_circuit_count

    @total_circuit_count.setter
    def total_circuit_count(self, _):
        raise RuntimeError("Can not set total circuit count value.")

    @property
    def total_run_time(self):
        return self._total_run_time

    @total_run_time.setter
    def total_run_time(self, _):
        raise RuntimeError("Can not set total run time value.")

    @abstractmethod
    def create_programs(self):
        pass

    def reset(self):
        self.programs.clear()
        self._executor = None

    def run(self):
        if self._executor is not None:
            raise RuntimeError("A batch is already being run.")

        if len(self.programs) == 0:
            raise RuntimeError("No programs to run.")

        self._executor = ProcessPoolExecutor()

        self.futures = [
            self._executor.submit(program.run) for program in self.programs.values()
        ]

    def check_all_done(self):
        return all(future.done() for future in self.futures)

    def wait_for_all(self):
        if self._executor is None:
            return

        exceptions = []
        try:
            # Ensure all futures are completed and handle exceptions.
            for future in as_completed(self.futures):
                try:
                    future.result()  # Raises an exception if the task failed.
                except Exception as e:
                    exceptions.append(e)

            if exceptions:
                for i, exc in enumerate(exceptions, 1):
                    print(f"Task {i} failed with exception:")
                    traceback.print_exception(type(exc), exc, exc.__traceback__)
                raise RuntimeError("One or more tasks failed. Check logs for details.")
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None

        self._total_circuit_count = sum(future.result()[0] for future in self.futures)
        self._total_run_time = sum(future.result()[1] for future in self.futures)
        self.futures = []

    @abstractmethod
    def aggregate_results(self):
        pass
