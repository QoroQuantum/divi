# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Minimum Birkhoff Decomposition built on Divi's standalone CircuitPipeline.

Each cost evaluation samples a parameterised circuit on the configured
backend, decodes measured bitstrings into permutation combinations, and
solves two CPLEX models (an integer least-squares fit, then a binary-indicator
model that minimises the number of nonzero weights subject to that squared
error) for convex weights over the decoded permutations that approximate a
target doubly-stochastic matrix.

The pipeline (``QiskitSpecStage → MeasurementStage(COUNTS) →
ParameterBindingStage``) maps parameter sets to raw shot histograms.  A
Divi optimiser (``ScipyOptimizer`` / ``MonteCarloOptimizer``) drives the
loop through its ``optimize(cost_fn, initial_params, ...)`` API.
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache
from itertools import repeat

import numpy as np
from docplex.mp.model import Model
from qiskit.circuit import ClassicalRegister, ParameterVector

from divi.backends import CircuitRunner
from divi.pipeline import (
    CircuitPipeline,
    PipelineEnv,
    ResultFormat,
    extract_param_set_idx,
)
from divi.pipeline.stages import (
    MeasurementStage,
    ParameterBindingStage,
    QiskitSpecStage,
)
from divi.qprog.algorithms import Ansatz
from divi.qprog.optimizers import Optimizer, ScipyMethod, ScipyOptimizer


def decode_combination(target_id: int, k: int) -> list[int]:
    """Decode a single integer back into a combination of k unique integer IDs."""
    combination = []
    for i in range(k, 0, -1):
        x = i - 1
        while math.comb(x, i) <= target_id:
            x += 1
        x -= 1
        combination.append(x)
        target_id -= math.comb(x, i)
    return sorted(combination)


def encode_combination(combination: list[int], k: int) -> int:
    """Encode a combination of k unique integer IDs into a single integer.

    Inverse of :func:`decode_combination`: with sorted ``combination =
    [a_1 < a_2 < ... < a_k]`` returns ``Σ_{i=1..k} C(a_i, i)``.
    """
    combination = sorted(combination)
    target_id = 0
    for i in range(1, k + 1):
        target_id += math.comb(combination[i - 1], i)
    return target_id


def solve_black_box(
    combination_ids: tuple[int, ...],
    target_matrix: np.ndarray,
    all_permutation_matrices_flat: np.ndarray,
    scale: int,
) -> tuple[int, float, list[float] | None]:
    """Solve the two-step approximation for a fixed permutation combination.

    Returns ``(n_nonzero, error, weights)``.  ``error`` is ``inf`` and
    ``weights`` is ``None`` when no feasible solution exists.
    """
    n = target_matrix.shape[0]
    k = len(combination_ids)

    if not combination_ids or max(combination_ids) >= len(
        all_permutation_matrices_flat
    ):
        return k + 1, float("inf"), None

    selected_perms_flat = all_permutation_matrices_flat[list(combination_ids)]

    try:
        # Each model solve uses one thread; solves run in a ThreadPoolExecutor.
        integer_model = Model(name="integer_approximation")
        integer_model.parameters.threads = 1

        u = integer_model.integer_var_list(k, name="u")
        integer_model.add_constraint(integer_model.sum(u) == scale)
        for i in range(k):
            integer_model.add_constraint(u[i] >= 0)
            integer_model.add_constraint(u[i] <= scale)

        target_flat = target_matrix.flatten()
        reconstructed_flat = u @ selected_perms_flat
        integer_model.minimize(
            integer_model.sum_squares(
                target_flat[i] - reconstructed_flat[i] for i in range(n * n)
            )
        )
        integer_solution = integer_model.solve()
        if not integer_solution:
            return k + 1, float("inf"), None
        min_error_found = integer_solution.get_objective_value()

        continuous_model = Model(name="sparsification")
        continuous_model.parameters.threads = 1

        c = continuous_model.continuous_var_list(k, name="c")
        y = continuous_model.binary_var_list(k, name="y")
        continuous_model.add_constraint(continuous_model.sum(c) == 1)
        for i in range(k):
            continuous_model.add_constraint(c[i] >= 0)
            continuous_model.add_constraint(c[i] <= y[i])

        target_unscaled_flat = target_flat / scale
        reconstructed_flat_c = c @ selected_perms_flat
        continuous_model.add_constraint(
            continuous_model.sum_squares(
                target_unscaled_flat[i] - reconstructed_flat_c[i] for i in range(n * n)
            )
            <= min_error_found / (scale**2) + 1e-9
        )
        continuous_model.minimize(continuous_model.sum(y))
        continuous_solution = continuous_model.solve()
        if not continuous_solution:
            return k + 1, float("inf"), None
    except Exception as exc:
        if "CPLEX" in str(exc) or "runtime" in str(exc).lower():
            raise RuntimeError(
                "CPLEX runtime not found. Install it with: pip install cplex"
            ) from exc
        raise

    continuous_weights = continuous_solution.get_value_list(c)
    final_error = np.linalg.norm(
        target_unscaled_flat - np.array(continuous_weights) @ selected_perms_flat,
        ord=2,
    )
    return (
        int(round(continuous_solution.get_objective_value())),
        final_error,
        continuous_weights,
    )


def build_parameterized_circuit(ansatz: Ansatz, n_qubits: int, n_layers: int):
    """Build a parameterised Qiskit circuit with a counts measurement."""
    n_params_per_layer = ansatz.n_params_per_layer(n_qubits)
    weights = np.array(
        [ParameterVector(f"w_{i}", n_params_per_layer) for i in range(n_layers)],
        dtype=object,
    )

    circuit = ansatz.build(weights, n_qubits=n_qubits, n_layers=n_layers)
    circuit.add_register(ClassicalRegister(n_qubits, "meas"))
    circuit.measure(range(n_qubits), range(n_qubits))
    return circuit


def build_pipeline() -> CircuitPipeline:
    """Build a standalone pipeline that emits raw shot histograms per parameter set."""
    return CircuitPipeline(
        stages=[
            QiskitSpecStage(),
            MeasurementStage(result_format_override=ResultFormat.COUNTS),
            ParameterBindingStage(),
        ]
    )


def _process_one_combination(
    solve_cached,
    comb: list[int],
    count: int,
    total_shots: int,
    penalty_value: float,
) -> tuple[float, float, list[int]]:
    """Score one decoded combination for the per-parameter-set thread pool."""
    _, raw_error, _ = solve_cached(tuple(comb))
    loss_value = raw_error if not np.isinf(raw_error) else penalty_value
    weighted_loss = (count / total_shots) * loss_value
    return weighted_loss, raw_error, comb


def _compute_histogram_losses(
    histogram: dict[str, int],
    k: int,
    solve_cached,
    penalty_value: float,
    executor: ThreadPoolExecutor,
) -> tuple[float, list[tuple[float, float, list[int]]]]:
    """Reduce a single shot histogram to a scalar loss + per-combination rows."""
    total_shots = sum(histogram.values())
    if total_shots == 0:
        return float("inf"), []

    valid_combs: list[list[int]] = []
    valid_shots: list[int] = []
    for bitstring, shots in histogram.items():
        try:
            valid_combs.append(decode_combination(int(bitstring, 2), k))
            valid_shots.append(shots)
        except (ValueError, IndexError):
            continue

    if not valid_combs:
        return penalty_value, []

    rows = list(
        executor.map(
            _process_one_combination,
            repeat(solve_cached),
            valid_combs,
            valid_shots,
            repeat(total_shots),
            repeat(penalty_value),
            chunksize=1,
        )
    )

    weighted_losses, _, _ = zip(*rows)
    return float(sum(weighted_losses)), rows


@dataclass
class BirkhoffResult:
    """Output of :func:`run_birkhoff`.

    Attributes:
        combination: Indices into ``all_perms_matrix`` of the chosen
            permutations, or ``None`` when no valid decomposition was found.
        weights: Convex weights aligned with ``combination``, or ``None``.
        final_error: 2-norm of the flattened ``target_unscaled - reconstructed``
            for the chosen decomposition, or ``inf`` when no decomposition was
            found.
        final_histogram: Shot histogram from the final measurement at the
            best parameters (``{bitstring: counts}``).
        top_outcomes: The most frequent measured bitstrings, each as
            ``{"bitstring", "shots", "combination", "error"}``.
        total_circuit_count: Total number of circuits submitted to the backend.
    """

    combination: list[int] | None
    weights: list[float] | None
    final_error: float
    final_histogram: dict[str, int]
    top_outcomes: list[dict]
    total_circuit_count: int = 0


def run_birkhoff(
    matrix: np.ndarray,
    scale: int,
    k: int,
    all_perms_matrix: np.ndarray,
    backend: CircuitRunner,
    optimizer: Optimizer,
    ansatz: Ansatz,
    max_iterations: int = 10,
    n_layers: int = 3,
    n_top_bitstrings: int = 10,
    seed: int | None = None,
) -> BirkhoffResult:
    """Run the standalone-pipeline Birkhoff decomposition.

    Args:
        matrix: ``(n, n)`` doubly stochastic matrix scaled by ``scale``.
        scale: Integer factor that ``matrix`` was multiplied by.
        k: Number of permutations to combine in the decomposition.
        all_perms_matrix: ``(num_perms, n, n)`` stack of all candidate
            permutation matrices to draw from.
        backend: A Divi ``CircuitRunner`` (e.g. ``MaestroSimulator`` or
            ``QoroService``).
        optimizer: Any Divi ``Optimizer`` exposing ``optimize(cost_fn, ...)``.
        ansatz: Divi ``Ansatz`` that builds the parameterised circuit.
        max_iterations: Optimiser iteration cap.
        n_layers: Ansatz depth.
        n_top_bitstrings: Top-N most-frequent measurements to score during
            final decomposition selection.
        seed: RNG seed for the initial parameters and the optimiser.

    Returns:
        A :class:`BirkhoffResult` with the lowest-error decomposition among the
        scored outcomes, and diagnostics.
    """
    n = matrix.shape[0]
    all_perms_flat = all_perms_matrix.reshape(all_perms_matrix.shape[0], -1)
    target_unscaled_flat = matrix.flatten() / scale
    n_qubits = int(np.ceil(np.log2(math.comb(math.factorial(n), k))))
    penalty_value = float(np.linalg.norm(target_unscaled_flat, ord=2))

    circuit = build_parameterized_circuit(ansatz, n_qubits, n_layers)
    pipeline = build_pipeline()

    @cache
    def solve_cached(combination_ids: tuple[int, ...]):
        return solve_black_box(combination_ids, matrix, all_perms_flat, scale)

    state = {
        "circuit_count": 0,
        "best_loss": float("inf"),
        "best_params": None,
    }

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:

        def evaluate_cost(params: np.ndarray) -> float | np.ndarray:
            param_sets = np.atleast_2d(params)
            env = PipelineEnv(backend=backend, param_sets=param_sets)
            result = pipeline.run(initial_spec=circuit, env=env)
            state["circuit_count"] += env.artifacts.get("circuit_count", 0)

            ordered = sorted(
                result.items(), key=lambda kv: extract_param_set_idx(kv[0])
            )
            losses = np.empty(len(ordered), dtype=np.float64)
            best_in_call: float = float("inf")
            for i, (_, histogram) in enumerate(ordered):
                losses[i], _ = _compute_histogram_losses(
                    histogram, k, solve_cached, penalty_value, executor
                )
                if losses[i] < best_in_call:
                    best_in_call = losses[i]
                    best_params_in_call = param_sets[i]

            if best_in_call < state["best_loss"]:
                state["best_loss"] = best_in_call
                state["best_params"] = best_params_in_call.copy()

            return losses if params.ndim > 1 else losses.item()

        def record_iteration(intermediate_result):
            # Stop COBYLA when the iteration count reaches `max_iterations`.
            if (
                isinstance(optimizer, ScipyOptimizer)
                and optimizer.method == ScipyMethod.COBYLA
                and intermediate_result.nit + 1 == max_iterations
            ):
                raise StopIteration

        rng = np.random.default_rng(seed)
        n_params = n_layers * ansatz.n_params_per_layer(n_qubits)

        initial_params = rng.uniform(0, 2 * np.pi, (optimizer.n_param_sets, n_params))

        try:
            optimizer.optimize(
                cost_fn=evaluate_cost,
                initial_params=initial_params,
                callback_fn=record_iteration,
                max_iterations=max_iterations,
                rng=rng,
            )
        except StopIteration:
            pass

        best_params = (
            np.atleast_2d(state["best_params"])
            if state["best_params"] is not None
            else np.atleast_2d(initial_params[0])
        )

        env = PipelineEnv(backend=backend, param_sets=best_params)
        final_result = pipeline.run(initial_spec=circuit, env=env)
    state["circuit_count"] += env.artifacts.get("circuit_count", 0)
    final_histogram: dict[str, int] = next(iter(final_result.values()))

    sorted_outcomes = sorted(
        final_histogram.items(), key=lambda item: item[1], reverse=True
    )
    best = {"combination": None, "weights": None, "error": float("inf")}
    top_outcomes = []
    for bitstring, shots in sorted_outcomes[:n_top_bitstrings]:
        try:
            comb = decode_combination(int(bitstring, 2), k)
        except (ValueError, IndexError):
            continue
        _, error, weights = solve_cached(tuple(comb))
        top_outcomes.append(
            {
                "bitstring": bitstring,
                "shots": shots,
                "combination": comb,
                "error": error,
            }
        )
        if weights is not None and error < best["error"]:
            best = {"combination": comb, "weights": weights, "error": error}

    if best["combination"] is not None:
        reconstructed = (
            np.asarray(best["weights"]) @ all_perms_flat[best["combination"]]
        )
        final_error = float(np.linalg.norm(target_unscaled_flat - reconstructed, ord=2))
    else:
        final_error = float("inf")

    return BirkhoffResult(
        combination=best["combination"],
        weights=best["weights"],
        final_error=final_error,
        final_histogram=final_histogram,
        top_outcomes=top_outcomes,
        total_circuit_count=state["circuit_count"],
    )
