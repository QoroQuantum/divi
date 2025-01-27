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

        self.executor = None
        self.manager = None
        self.programs = {}

    @abstractmethod
    def create_programs(self):
        pass

    def reset(self):
        self.programs.clear()
        self.executor = None
        self.manager = None

    def run(self):
        if self.executor is not None:
            raise RuntimeError("A batch is already being run.")

        if len(self.programs) == 0:
            raise RuntimeError("No programs to run.")

        self.executor = ProcessPoolExecutor()

        self.futures = [
            self.executor.submit(program.run) for program in self.programs.values()
        ]

    def check_all_done(self):
        return all(future.done() for future in self.futures)

    def wait_for_all(self):
        if self.executor is None:
            return

        exceptions = []
        try:
            # Ensure all futures are completed and handle exceptions.
            for future in as_completed(self.futures):
                try:
                    future.result()  # Raises an exception if the task failed.
                except Exception as e:
                    exceptions.append(e)
        finally:
            if self.executor:
                self.executor.shutdown(wait=True, cancel_futures=False)
                self.executor = None
                self.futures = []

        if exceptions:
            for i, exc in enumerate(exceptions, 1):
                print(f"Task {i} failed with exception: {exc}")
            raise RuntimeError(
                "One or more tasks failed. Check logs for details.")

    @property
    def total_circuit_count(self):
        return sum(prog.total_circuit_count for prog in self.programs.values())

    @total_circuit_count.setter
    def _(self, value):
        raise RuntimeError("Can not set total circuit count value.")

    @abstractmethod
    def aggregate_results(self):
        pass
