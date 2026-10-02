# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared builders, constants, and cost functions for optimizer tests."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RYGate, RZGate
from qiskit.quantum_info import SparsePauliOp

from divi.qprog import VQE, CustomVQA
from divi.qprog.algorithms import GenericLayerAnsatz
from divi.qprog.problems import HamiltonianProblem
from divi.qprog.variational_quantum_algorithm import _compute_parameter_shift_rule

#: Eigenvalues of the bowl the noisy contracts descend; condition number 10. Its
#: length fixes the parameter count for every ``noisy_bowl`` run.
BOWL_CURVATURE = np.logspace(0.0, 1.0, 4)


def sphere_cost_fn_population(params: np.ndarray) -> np.ndarray:
    """Sphere cost function for a population of parameter sets."""
    if params.ndim != 2:
        raise ValueError(
            "Input params for population cost function must be a 2D array."
        )
    return np.sum(params**2, axis=1)


def sphere_cost_fn_single(params: np.ndarray) -> float:
    """Sphere cost function for a single parameter set."""
    if params.ndim != 1:
        if params.ndim == 2 and params.shape[0] == 1:
            params = params.squeeze(0)
        else:
            raise ValueError(
                "Input params for single cost function must be a 1D array."
            )
    return float(np.sum(params**2))


def sphere_cost_fn_batch_aware(params: np.ndarray) -> float | np.ndarray:
    """Sphere cost supporting both one parameter set and a 2-D batch."""
    params = np.atleast_2d(params)
    values = np.sum(params**2, axis=1)
    return values if params.shape[0] > 1 else float(values[0])


def noisy_bowl(rng, sigma=0.0, dropout=0.0):
    """Return a noisy quadratic cost and its noise-free scoring function."""

    def true_cost(params):
        rows = np.atleast_2d(np.asarray(params, dtype=np.float64))
        values = 0.5 * np.einsum("ij,j,ij->i", rows, BOWL_CURVATURE, rows)
        return values if rows.shape[0] > 1 else float(values[0])

    def cost_fn(params):
        rows = np.atleast_2d(np.asarray(params, dtype=np.float64))
        values = 0.5 * np.einsum("ij,j,ij->i", rows, BOWL_CURVATURE, rows)
        if sigma:
            values = values + rng.normal(0.0, sigma, size=values.shape)
        if dropout:
            values = np.where(rng.random(values.shape) < dropout, np.nan, values)
        return values if rows.shape[0] > 1 else float(values[0])

    return cost_fn, true_cost


#: QUIVER settings that pin one direction per step with no ``V``/``M`` adaptation.
FIXED_SINGLE_DIRECTION = {"V_init": 1, "V_min": 1, "adapt_V": False, "adapt_M": False}


def callback_trace(optimizer, cost_fn, initial_params, max_iterations, **kwargs):
    """Run ``optimizer`` (seeded with ``default_rng(0)`` unless ``rng`` is given)
    and return its result with every callback ``OptimizeResult``."""
    trace = []
    kwargs.setdefault("rng", np.random.default_rng(0))
    result = optimizer.optimize(
        cost_fn,
        initial_params=np.asarray(initial_params),
        max_iterations=max_iterations,
        callback_fn=trace.append,
        **kwargs,
    )
    return result, trace


def counting_cost_fn(cost_fn=sphere_cost_fn_batch_aware):
    """Wrap ``cost_fn``; returns ``(calls, wrapped)`` with ``calls["n"]`` the call count."""
    calls = {"n": 0}

    def wrapped(params):
        calls["n"] += 1
        return cost_fn(params)

    return calls, wrapped


def count_divergence_warnings(record) -> int:
    """Number of SPSA-family divergence warnings in a recorded warning list."""
    return sum("appears to be diverging" in str(w.message) for w in record)


def bowl_jac(params: np.ndarray) -> np.ndarray:
    """Exact gradient of :func:`noisy_bowl`'s noise-free quadratic."""
    return BOWL_CURVATURE * np.asarray(params, dtype=np.float64)


def bowl_metric(_params: np.ndarray) -> np.ndarray:
    """Positive-definite metric matching the bowl curvature."""
    return np.diag(BOWL_CURVATURE)


def small_vqe(backend, optimizer, **kwargs) -> VQE:
    """Two-qubit, three-term VQE with one RY-RZ layer (seed 1997 unless given)."""
    return VQE(
        HamiltonianProblem(
            SparsePauliOp.from_list([("ZI", 0.5), ("IZ", -0.3), ("XX", 0.2)])
        ),
        ansatz=GenericLayerAnsatz([RYGate, RZGate]),
        n_layers=1,
        backend=backend,
        optimizer=optimizer,
        **{"seed": 1997, **kwargs},
    )


def set_unit_shift_rule(vqe) -> int:
    """Give every parameter of ``vqe`` a unit two-term shift rule; return the
    parameter count."""
    n_params = vqe.n_layers * vqe.n_params_per_layer
    vqe._grad_shift_rule = _compute_parameter_shift_rule([(1.0, 1)] * n_params)
    return n_params


def inject_term_expectations(monkeypatch, term_values) -> list[np.ndarray]:
    """Replace the pullback measurement seam with one branch whose per-term
    expectations and coefficients are ``term_values(param_sets)``; return the list
    of parameter batches it was called with."""
    calls = []

    def fake(_program, param_sets):
        calls.append(np.array(param_sets))
        return {(("circuit", 0),): term_values(param_sets)}

    monkeypatch.setattr("divi.qprog._metrics._term_expectations", fake)
    return calls


def data_bound_custom_vqa(backend, optimizer, labels=None) -> CustomVQA:
    """One-qubit CustomVQA with a data angle and a weight angle; supervised when
    ``labels`` are given."""
    x = Parameter("x")
    w = Parameter("w")
    qc = QuantumCircuit(1, 1)
    qc.ry(x, 0)
    qc.rz(w, 0)
    qc.measure(0, 0)
    return CustomVQA(
        qscript=qc,
        data_param_indices=[list(qc.parameters).index(x)],
        feature_batch=np.array([[0.1], [0.3]]),
        labels=labels,
        backend=backend,
        optimizer=optimizer,
    )
