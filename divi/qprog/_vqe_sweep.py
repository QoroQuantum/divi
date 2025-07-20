from functools import partial
from itertools import product
from typing import Literal

import matplotlib.pyplot as plt

from divi.interfaces import CircuitRunner
from divi.qprog import VQE, ProgramBatch, VQEAnsatz

from .optimizers import Optimizers


class VQEHyperparameterSweep(ProgramBatch):
    """Allows user to carry out a grid search across different values
    for the ansatz and the bond length used in a VQE program.
    """

    def __init__(
        self,
        bond_lengths: list[float],
        ansatze: list[VQEAnsatz],
        symbols: list[str],
        coordinate_structure: list[tuple[float, float, float]],
        charge: float = 0,
        optimizer: Optimizers = Optimizers.MONTE_CARLO,
        max_iterations: int = 10,
        **kwargs,
    ):
        """Initiates the class.

        Args:
            bond_lengths (list): The bond lengths to consider.
            ansatze (list): The ansatze to use for the VQE problem.
            symbols (list): The symbols of the atoms in the molecule.
            coordinate_structure (list): The coordinate structure of the molecule.
            optimizer (Optimizers): The optimizer to use.
            max_iterations (int): Maximum number of iteration optimizers.
        """
        super().__init__(backend=kwargs.pop("backend"))

        self.ansatze = ansatze
        self.bond_lengths = [round(bnd, 9) for bnd in bond_lengths]
        self.max_iterations = max_iterations

        self._constructor = partial(
            VQE,
            symbols=symbols,
            coordinate_structure=coordinate_structure,
            charge=charge,
            optimizer=optimizer,
            max_iterations=self.max_iterations,
            backend=self.backend,
            **kwargs,
        )

    def create_programs(self):
        if len(self.programs) > 0:
            raise RuntimeError(
                "Some programs already exist. "
                "Clear the program dictionary before creating new ones by using batch.reset()."
            )

        super().create_programs()

        for ansatz, bond_length in product(self.ansatze, self.bond_lengths):
            _job_id = (ansatz, bond_length)
            self.programs[_job_id] = self._constructor(
                job_id=_job_id,
                bond_length=bond_length,
                ansatz=ansatz,
                losses=self._manager.list(),
                final_params=self._manager.list(),
                progress_queue=self._queue,
            )

    def aggregate_results(self):
        if len(self.programs) == 0:
            raise RuntimeError("No programs to aggregate. Run create_programs() first.")

        if self._executor is not None:
            self.wait_for_all()

        if any(len(program.losses) == 0 for program in self.programs.values()):
            raise RuntimeError(
                "Some/All programs have empty losses. Did you call run()?"
            )

        all_energies = {key: prog.losses[-1] for key, prog in self.programs.items()}

        smallest_key = min(all_energies, key=lambda k: min(all_energies[k].values()))
        smallest_value = min(all_energies[smallest_key].values())

        return smallest_key, smallest_value

    def visualize_results(self, graph_type: Literal["line", "scatter"] = "line"):
        """
        Visualize the results of the VQE problem.
        """
        if graph_type not in ["line", "scatter"]:
            raise ValueError(
                f"Invalid graph type: {graph_type}. Choose between 'line' and 'scatter'."
            )

        if self._executor is not None:
            self.wait_for_all()

        data = []
        colors = ["blue", "g", "r", "c", "m", "y", "k"]

        ansatz_list = list(VQEAnsatz)

        if graph_type == "scatter":
            for ansatz, bond_length in self.programs.keys():
                min_energies = []

                curr_energies = self.programs[(ansatz, bond_length)].losses[-1]
                min_energies.append(
                    (
                        bond_length,
                        min(curr_energies.values()),
                        colors[ansatz_list.index(ansatz)],
                    )
                )
                data.extend(min_energies)

            x, y, z = zip(*data)
            plt.scatter(x, y, color=z, label=ansatz)

        elif graph_type == "line":
            for ansatz in self.ansatze:
                energies = []
                for bond_length in self.bond_lengths:
                    energies.append(
                        min(self.programs[(ansatz, bond_length)].losses[-1].values())
                    )
                plt.plot(self.bond_lengths, energies, label=ansatz)

        plt.xlabel("Bond length")
        plt.ylabel("Energy level")
        plt.legend()
        plt.show()
