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
            return 3

    def samples(self):
        if self == Optimizers.MONTE_CARLO:
            return 2
        return 1

    def update_params(self, params, iteration):
        if self == Optimizers.MONTE_CARLO:
            return [
                np.random.normal(params, 1 / iteration, size=params.shape)
                for _ in range(self.num_param_sets())
            ]
        else:
            raise NotImplementedError
