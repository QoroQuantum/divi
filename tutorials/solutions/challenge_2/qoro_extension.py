from itertools import product
from divi.qprog.workflows import VQEHyperparameterSweep
"""

Task: Implement an extension to VQEHyperparameterSweep, 
a more sophisticated VQE class, or a whole new distributed quantum chemistry 
class using existing base classes


"""

class VQEHyperparameterSweepExtension(VQEHyperparameterSweep):
    """
    Extends VQEHyperparameterSweep to allow sweeping over different number of layers.
    n_layers: is a list we iterate over
    """
    def __init__(
        self,
        ansatze,
        molecule_transformer,
        optimizer=None,
        max_iterations=10,
        n_layers_list=None,  
        **kwargs
    ):
        self.n_layers_list = n_layers_list or [None]  # default: no layer sweep
        super().__init__(
            ansatze,
            molecule_transformer,
            optimizer=optimizer,
            max_iterations=max_iterations,
            **kwargs
        )

    def create_programs(self):
        """
        Create VQE programs for all combinations of ansätze, molecule variants,
        and number of layers (if provided).
        """
        super().create_programs()  # this sets up self.programs

        # Clear programs created by parent; we will rebuild with n_layers
        self._programs = {}

        # Generate molecule variants
        molecule_variants = self.molecule_transformer.generate()

        # Loop over all combinations of ansatz, molecule, n_layers
        for ansatz, (modifier, molecule), n_layers in product(
            self.ansatze, molecule_variants.items(), self.n_layers_list
        ):
            # Only set n_layers if ansatz supports it
            if hasattr(ansatz, "n_layers") and n_layers is not None:
                ansatz.n_layers = n_layers

            job_id = (ansatz.name, modifier, n_layers)
            self._programs[job_id] = self._constructor(
                job_id=job_id,
                molecule=molecule,
                ansatz=ansatz,
                optimizer=self._optimizer_template.copy() if hasattr(self._optimizer_template, "copy") else self._optimizer_template,
                progress_queue=self._queue,
            )

    def aggregate_results(self):
        """
        Aggregates results considering n_layers as part of the configuration.
        """
        super().aggregate_results()

        all_energies = {key: prog.best_loss for key, prog in self.programs.items()}
        best_key = min(all_energies, key=lambda k: all_energies[k])
        best_energy = all_energies[best_key]

        return best_key, best_energy
