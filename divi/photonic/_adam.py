# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal numpy implementation of Adam.

We roll our own rather than dragging PyTorch into divi's dependency set
just for an optimiser — the photonic variational loops only need the
scalar parameter update, not autodiff. Matches the Kingma & Ba (2014)
formulation.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class AdamState:
    learning_rate: float = 5e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    m: np.ndarray = None  # type: ignore[assignment]
    v: np.ndarray = None  # type: ignore[assignment]
    t: int = 0


def adam_step(state: AdamState, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
    """Return the updated ``params`` after one Adam step.

    Mutates ``state.m``, ``state.v``, ``state.t`` in place.
    """
    if state.m is None:
        state.m = np.zeros_like(params)
        state.v = np.zeros_like(params)

    state.t += 1
    state.m = state.beta1 * state.m + (1.0 - state.beta1) * grads
    state.v = state.beta2 * state.v + (1.0 - state.beta2) * (grads * grads)

    m_hat = state.m / (1.0 - state.beta1**state.t)
    v_hat = state.v / (1.0 - state.beta2**state.t)

    return params - state.learning_rate * m_hat / (np.sqrt(v_hat) + state.eps)
