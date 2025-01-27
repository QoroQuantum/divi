from functools import partial
from itertools import product
from multiprocessing import Manager

import matplotlib.pyplot as plt

from divi.qprog import VQE
from divi.qprog import ProgramBatch
from divi.qprog import VQEAnsatze

from .optimizers import Optimizers


class VQEHyperparameterSweep(ProgramBatch):
    """Allows user to carry out a grid search across different values
    for the ansatz and the bond length used in a VQE program.
    """

    def __init__(
        self,
        bond_lengths,
        ansatze,
        symbols,
        coordinate_structure,
        optimizer=Optimizers.MONTE_CARLO,
        max_iterations=10,
        shots=5000,
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
            shots (int): Number of shots for each circuit execution.
        """
        super().__init__()

        self.ansatze = ansatze
        self.bond_lengths = bond_lengths

        self._constructor = partial(
            VQE,
            symbols=symbols,
            coordinate_structure=coordinate_structure,
            optimizer=optimizer,
            max_iterations=max_iterations,
            shots=shots,
            **kwargs,
        )

    def create_programs(self):
        if len(self.programs) > 0:
            raise RuntimeError(
                "Some programs already exist. "
                "Clear the program dictionary before creating new ones by using batch.reset()."
            )
        self.manager = Manager()

        for ansatz, bond_length in product(self.ansatze, self.bond_lengths):
            self.programs[(ansatz, bond_length)] = self._constructor(
                bond_length=bond_length,
                ansatz=ansatz,
                energies=self.manager.list(),
            )
        return

    def aggregate_results(self):
        if self.executor is not None:
            self.wait_for_all()

        all_energies = {key: prog.energies[-1]
                        for key, prog in self.programs.items()}

        smallest_key = min(all_energies, key=lambda k: min(
            all_energies[k].values()))
        smallest_value = min(all_energies[smallest_key].values())

        return smallest_key, smallest_value

    def visualize_results(self, graph_type="line"):
        """Visualize the results of the VQE problem.
        """
        if self.executor is not None:
            self.wait_for_all()

        data = []
        colors = ["blue", "g", "r", "c", "m", "y", "k"]

        ansatz_list = list(VQEAnsatze)

        if graph_type == "scatter":
            for ansatz, bond_length in product(self.ansatze, self.bond_lengths):
                min_energies = []

                curr_energies = self.programs[(
                    ansatz, bond_length)].energies[-1]
                min_energies.append(
                    (
                        bond_length,
                        min(curr_energies.values()),
                        colors[ansatz_list.index(ansatz)],
                    )
                )
                data.extend(min_energies)

            x, y, z = zip(*data)
            plt.scatter(x, y, color=z)
            plt.xlabel("Bond length")
            plt.ylabel("Energy level")
            plt.show()

        elif graph_type == 'line':
            for ansatz in self.ansatze:
                energies = []
                for bond_length in self.bond_lengths:
                    energies.append(
                        min(self.programs[(ansatz, bond_length)
                                          ].energies[-1].values())
                    )
                plt.plot(self.bond_lengths, energies, label=ansatz)
            plt.xlabel("Bond length")
            plt.ylabel("Energy level")
            plt.legend()
            plt.show()
