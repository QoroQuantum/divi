# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import pickle

import dimod
import numpy as np
import pytest

from divi.qprog.problems import BinaryOptimizationProblem, LinearConstraint
from tests.qprog.problems._helpers import (
    assert_penalty_zero_iff_feasible,
    iter_assignments,
    n_slack,
    polynomial_value,
)

OBJECTIVE = np.diag([-1.0, -2.0, -3.0])


def _constrained(*constraints, **kwargs) -> BinaryOptimizationProblem:
    return BinaryOptimizationProblem(OBJECTIVE, constraints=list(constraints), **kwargs)


def _fuzzed_constraints(
    n_cases: int = 200, seed: int = 0, magnitudes: bool = False
) -> list[tuple]:
    """Integer, decimal and fractional constraints of every sense, float noise included.

    With ``magnitudes``, the data is also scaled by ``10**-6`` to ``10**9`` and
    some constraints get one coefficient ``10**8`` steps wide.
    """
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n_cases):
        scale = 10.0 ** rng.integers(-6, 10) if magnitudes else 1.0
        step = rng.choice([1, 0.1, 0.01, 1 / 3, 1 / 7]) * scale
        coefficients = rng.integers(-9, 10, rng.integers(1, 5)) * step
        coefficients[0] = coefficients[0] or step
        if magnitudes and rng.random() < 0.25:
            coefficients = np.append(coefficients, rng.integers(1, 10) * 1e8 * step)
        bound = coefficients @ rng.integers(0, 2, len(coefficients))
        bound += rng.choice([0.0, step / 2, -step / 2])
        # Float noise below is_satisfied's tolerance must not change the encoding.
        magnitude = max(abs(bound), np.abs(coefficients).sum())
        bound += rng.choice([0.0, 0.3, -0.3]) * 1e-12 * magnitude
        sense = str(rng.choice(["==", "<=", ">="]))
        cases.append((coefficients.tolist(), sense, bound, magnitudes))
    return cases


class TestLinearConstraint:
    def test_sequence_coefficients_are_keyed_by_position(self):
        c = LinearConstraint([2, 0, -1], "<=", 1)
        assert dict(c.coefficients) == {0: 2.0, 2: -1.0}

    @pytest.mark.parametrize(
        "coefficients, sense, bound, match",
        [
            ([1], "<", 1, "sense"),
            ([0, 0], "==", 1, "non-zero"),
            ([np.inf], "==", 1, "finite"),
            ([1], "==", np.nan, "finite"),
        ],
    )
    def test_invalid_values_raise(self, coefficients, sense, bound, match):
        with pytest.raises(ValueError, match=match):
            LinearConstraint(coefficients, sense, bound)

    @pytest.mark.parametrize(
        "coefficients, sense, bound, assignment, expected",
        [
            ([1, 1], "==", 1, (0, 0), False),
            ([1, 1], "==", 1, (1, 0), True),
            ([1, 1], "==", 1, (1, 1), False),
            ([1, 1], ">=", 1, (1, 1), True),
            ([0.1e9, 0.2e9], "==", 0.3e9, (1, 1), True),  # float round-off
            ([1, 1], "==", 1.001, (1, 0), False),
            ([1e9, 1, 1], "<=", 1e9 + 1, (1, 1, 1), False),  # tolerance is relative
            ([1e-12] * 3, "==", 1e-12, (0, 0, 0), False),  # no absolute floor
            ([1e-12] * 3, "==", 1e-12, (1, 0, 0), True),
        ],
    )
    def test_is_satisfied(self, coefficients, sense, bound, assignment, expected):
        constraint = LinearConstraint(coefficients, sense, bound)
        assert constraint.is_satisfied(dict(enumerate(assignment))) is expected


@pytest.mark.parametrize(
    "coefficients, sense, bound, n_slack",
    [
        ([1, 1, 1], "==", 2, 0),
        ([1, 1, 1], "<=", 1, 1),  # slack range 0..1
        ([1, 1, 1], ">=", 1, 2),  # slack range 0..2
        ([1, 1, 1], ">=", 3, 0),  # only the full set reaches 3
        ([1, 1, 1], ">=", 1.5, 1),  # the bound tightens to 2
        ([1, 1, 1], "<=", 1000, 0),  # always holds: no penalty
        ([1, 1, 1], ">=", -1000, 0),
        ([2, -1, 3], ">=", 2, 2),
        ([100, 100, 100], "<=", 250, 2),
        ([1000, 1000, 1000], ">=", 1500, 1),
        ([0.3, 0.5, 0.2], ">=", 0.5, 3),
        ([1 / 3] * 3, "==", 1, 0),
        ([1 / 3] * 3, "==", 2 / 3, 0),
        ([1 / 7, 2 / 7, 3 / 7], "<=", 3 / 7, 2),
        ([88, 17, 54, 79, 96, 18], "<=", 168, 8),
        ([0.66, 0.31, 0.09, 0.07, 0.82, 0.92], "<=", 2.5, 8),
        ([1, 1, 1], ">=", 2.0000000000000004, 1),  # float noise on the bound
        ([1, 1, 1], "<=", 0.9999999999999999, 1),
        ([0.7, 0.1], "==", 0.8, 0),  # float sum 0.7999999999999999
        ([0.1, 0.2], "<=", 0.3, 0),  # always holds; float sum overshoots 0.3
        ([-1, 0], ">=", 0, 0),  # a single negative coefficient
        ([-3, 0, 0], "<=", -1, 0),
    ],
)
def test_exact_data_is_encoded_exactly(coefficients, sense, bound, n_slack, recwarn):
    n = len(coefficients)
    problem = BinaryOptimizationProblem(
        np.diag(-np.ones(n)), constraints=[LinearConstraint(coefficients, sense, bound)]
    )
    assert problem.cost_hamiltonian.num_qubits == n + n_slack
    assert_penalty_zero_iff_feasible(problem, n_decision=n)
    assert not recwarn.list


@pytest.mark.parametrize(
    "coefficients, sense, bound, may_round",
    _fuzzed_constraints() + _fuzzed_constraints(seed=1, magnitudes=True),
)
def test_penalty_is_zero_exactly_where_the_constraint_holds(
    coefficients, sense, bound, may_round, recwarn
):
    n = len(coefficients)
    constraint = LinearConstraint(coefficients, sense, bound)
    objective = np.diag(np.sqrt(np.arange(2, n + 2)))
    try:
        problem = BinaryOptimizationProblem(objective, constraints=[constraint])
    except ValueError as error:
        if not (may_round and "slack bits" in str(error)):
            assignments = itertools.product((0, 1), repeat=n)
            assert not any(
                constraint.is_satisfied(dict(enumerate(x))) for x in assignments
            )
        return
    if recwarn.list:
        assert may_round
        return
    assert_penalty_zero_iff_feasible(problem, n_decision=n, require_feasible=False)


@pytest.mark.parametrize(
    "coefficients, bound, violation",
    [
        ([1, 1, 1], 2, (1, 0, 0)),
        ([0.25, 0.5, 0.75], 1, (0, 1, 0)),
        ([3e6, 1e6, 2e6], 3e6, (0, 1, 0)),
    ],
)
def test_penalty_is_the_squared_violation_in_the_constraints_units(
    coefficients, bound, violation
):
    problem = _constrained(LinearConstraint(coefficients, "==", bound))
    assignment = dict(enumerate(violation))
    missing = bound - float(np.dot(coefficients, violation))
    assert polynomial_value(
        problem.penalty_canonical_problem.terms, assignment
    ) == pytest.approx(missing**2)


_FLOATS = [0.15763401497257323, 0.08892135493055972, 0.06583280005864131]


@pytest.mark.parametrize(
    "coefficients, sense, bound, match",
    [
        ([np.pi, np.e, np.sqrt(2)], "<=", np.pi + np.e, "rounded to steps"),
        # Exact at steps of 2.5, but a slack range of 800 there forces steps of 10.
        (2.5 * np.arange(1, 41), "<=", 2000, "rounded to steps of 10"),
        (_FLOATS, "==", _FLOATS[1] + _FLOATS[2], "rounded to steps"),
        ([3.4796921, 4.9014428], "==", 8.3811349, "rounded to steps"),
        ([1 + 2.7e-12] * 3, "==", 3, "rounded to steps"),
        ({0: 1e5, 1: 1e-7, 2: 1e-7}, "==", 1e5, "rounded to steps"),
        ([1e8, 1e8, 2e8 + 1, 1], "==", 2e8 + 1, "rounded to steps"),
    ],
    ids=[
        "irrational",
        "slack-too-wide",
        "float-equality",
        "seven-places",
        "off-by-noise",
        "tiny-coefficients",
        "beyond-float-precision",
    ],
)
def test_inexact_encodings_warn(coefficients, sense, bound, match):
    with pytest.warns(UserWarning, match=match):
        BinaryOptimizationProblem(
            np.diag(-np.ones(len(coefficients))),
            constraints=[LinearConstraint(coefficients, sense, bound)],
        )


_NAMED = dimod.BinaryQuadraticModel(
    {"a": -1.0, "b": -1.0, ("slack", 0, 0): -1.0}, {}, 0.0, "BINARY"
)


@pytest.mark.parametrize(
    "problem, constraint, kwargs, match",
    [
        (OBJECTIVE, LinearConstraint([1, 1, 1], ">=", 4), {}, "only reaches"),
        (OBJECTIVE, LinearConstraint([1, 1, 1], "==", 4), {}, "only reaches"),
        (OBJECTIVE, LinearConstraint([2, 4, 6], "==", 5), {}, "multiple of 2"),
        (
            OBJECTIVE,
            LinearConstraint([1, 1, 1], "==", 1),
            {"decomposer": object()},
            "decomposer",
        ),
        (
            np.diag(-np.ones(300)),
            LinearConstraint([1] * 300, ">=", 1),
            {},
            "more than 8 slack bits",
        ),
        # Rounded to fit 8 slack bits, it would always hold.
        (-np.eye(4), LinearConstraint([100, 100, 56, 1], "<=", 256), {}, "always hold"),
        (_NAMED, LinearConstraint({"a": 1, "b": 1}, "<=", 1), {}, "reserved"),
    ],
    ids=[
        "unreachable-geq",
        "unreachable-eq",
        "no-combination",
        "decomposer",
        "slack-bits",
        "rounded-trivial",
        "reserved-label",
    ],
)
def test_invalid_problems_raise(problem, constraint, kwargs, match):
    with pytest.raises(ValueError, match=match):
        BinaryOptimizationProblem(problem, constraints=[constraint], **kwargs)


def test_decode_and_energy_ignore_slack():
    problem = _constrained(LinearConstraint([1, 1, 1], "<=", 1), penalty_weight=10.0)
    slack = "0" * n_slack(problem, 3)
    np.testing.assert_array_equal(problem.decode_fn("101" + slack), [1, 0, 1])
    assert problem.compute_energy("111" + slack) == -6.0


def test_constrained_variable_keeps_its_qubit_when_terms_cancel():
    problem = BinaryOptimizationProblem(
        np.diag([1.0, -1.0]), constraints=[LinearConstraint([1, 0], "==", 1)]
    )
    assert problem.cost_hamiltonian.num_qubits == 2
    assert problem.is_feasible("10")
    assert not problem.is_feasible("01")


def test_user_labels_do_not_collide_with_dimod_slack_names():
    names = ["a", "slack_slack_0", "slack_divi_0"]
    bqm = dimod.BinaryQuadraticModel({v: -1.0 for v in names}, {}, 0.0, "BINARY")
    problem = BinaryOptimizationProblem(
        bqm, constraints=[LinearConstraint(dict.fromkeys(names, 1), "<=", 2)]
    )
    assert set(names) < set(problem.canonical_problem.variable_order)
    assert_penalty_zero_iff_feasible(problem, n_decision=3)


def test_quadratized_samples_of_one_solution_are_merged():
    hubo = {(0, 1, 2): -3.0, (0,): 1.0, (1,): 1.0, (2,): 1.0, (3,): -1.0}
    problem = BinaryOptimizationProblem(
        hubo,
        constraints=[LinearConstraint([1, 1, 1, 1], "<=", 3)],
        hamiltonian_builder="quadratized",
    )
    n_qubits = problem.cost_hamiltonian.num_qubits
    samples = [
        ("".join(bits), 1.0) for bits in itertools.product("01", repeat=n_qubits)
    ]
    keys = [
        problem._solution_key(s.bitstring)
        for s in problem._rank_feasible(samples, "filter", None)
    ]
    assert keys and len(keys) == len(set(keys))


def test_numpy_integer_labels_keep_slack_after_the_decision_variables():
    bqm = dimod.BinaryQuadraticModel(
        {np.int64(i): -1.0 for i in range(3)}, {}, 0.0, "BINARY"
    )
    problem = BinaryOptimizationProblem(
        bqm, constraints=[LinearConstraint([1, 1, 1], "<=", 1)]
    )
    assert problem.canonical_problem.variable_order[:3] == (0, 1, 2)


def test_constrained_problem_survives_pickling():
    problem = _constrained(LinearConstraint([1, 1, 1], "<=", 1))
    restored = pickle.loads(pickle.dumps(problem))
    slack = "0" * n_slack(problem, 3)
    assert restored.is_feasible("100" + slack)
    assert not restored.is_feasible("110" + slack)


def test_constraints_add_to_user_penalty():
    user_penalty = np.diag([5.0, 0.0, 0.0])
    constraint = LinearConstraint([1, 1, 1], "==", 1)
    combined = _constrained(constraint, penalty=user_penalty)
    alone = _constrained(constraint)
    for _, assignment in iter_assignments(combined):
        assert polynomial_value(
            combined.penalty_canonical_problem.terms, assignment
        ) == pytest.approx(
            polynomial_value(alone.penalty_canonical_problem.terms, assignment)
            + 5.0 * assignment[0]
        )
