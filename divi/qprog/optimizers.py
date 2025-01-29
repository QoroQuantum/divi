from enum import Enum

import numpy as np


class Optimizers(Enum):
    NELDER_MEAD = "Nelder-Mead"
    MONTE_CARLO = "Monte Carlo"

    def describe(self):
        return self.name, self.value

    def num_param_sets(self):
        if self == Optimizers.NELDER_MEAD:
            return 1
        elif self == Optimizers.MONTE_CARLO:
            return 10

    def samples(self):
        if self == Optimizers.MONTE_CARLO:
            return 10
        return 1

    def compute_new_parameters(self, params, iteration, **kwargs):
        if self == Optimizers.MONTE_CARLO:
            losses = kwargs.pop("losses")
            smallest_energy_keys = sorted(losses, key=lambda k: losses[k])[:self.samples()]
            new_params = []
            for key in smallest_energy_keys:
                new_param_set = [
                    np.random.normal( params[int(key)], 1 / (2 * iteration), size=params[int(key)].shape)
                    for _ in range(int(self.num_param_sets()))
                ]
                for new_param in new_param_set:
                    new_param = np.clip(new_param, 0, 2 * np.pi)
                new_params.extend(new_param_set)
            return new_params
        else:
            raise NotImplementedError
