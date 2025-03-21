from enum import Enum

import numpy as np


class Optimizers(Enum):
    NELDER_MEAD = "Nelder-Mead"
    MONTE_CARLO = "Monte Carlo"
    L_BFGS_B = "L-BFGS-B"

    def describe(self):
        return self.name, self.value

    @property
    def n_param_sets(self):
        if self in (Optimizers.NELDER_MEAD, Optimizers.L_BFGS_B):
            return 1
        elif self == Optimizers.MONTE_CARLO:
            return 10

    @property
    def n_samples(self):
        if self == Optimizers.MONTE_CARLO:
            return 10
        return 1

    def compute_new_parameters(self, params, iteration, **kwargs):
        if self != Optimizers.MONTE_CARLO:
            raise NotImplementedError

        losses = kwargs.pop("losses")
        smallest_energy_keys = sorted(losses, key=lambda k: losses[k])[: self.n_samples]

        new_params = []

        for key in smallest_energy_keys:
            new_param_set = [
                np.random.normal(
                    params[int(key)],
                    1 / (2 * iteration),
                    size=params[int(key)].shape,
                )
                for _ in range(self.n_param_sets)
            ]

            for new_param in new_param_set:
                new_param = np.clip(new_param, 0, 2 * np.pi)

            new_params.extend(new_param_set)

        return new_params

    def compute_parameter_shift_mask(self, n_params):
        if self != Optimizers.L_BFGS_B:
            raise NotImplementedError

        mask_arr = np.arange(0, 2 * n_params, 2)
        mask_arr[0] = 1

        binary_matrix = (
            (mask_arr[:, np.newaxis] & (1 << np.arange(n_params))) > 0
        ).astype(np.float64)

        binary_matrix = binary_matrix.repeat(2, axis=0)
        binary_matrix[1::2] *= -1
        binary_matrix *= 0.5 * np.pi

        return binary_matrix
