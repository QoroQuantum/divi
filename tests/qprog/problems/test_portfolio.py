# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import math

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from divi.qprog import PCE, QAOA
from divi.qprog.algorithms import DickeState, SuperpositionState
from divi.qprog.optimizers import MonteCarloOptimizer
from divi.qprog.problems import (
    BinaryOptimizationProblem,
    LinearConstraint,
    PortfolioAllocationProblem,
    PortfolioSelectionProblem,
)
from tests.qprog.problems._helpers import (
    assert_penalty_zero_iff_feasible,
    n_slack,
    penalty_at,
)

MU = np.array([0.10, 0.12, 0.08, 0.15])
ESG = np.array([70.0, 55.0, 80.0, 60.0])
TAU = 0.5


def _covariance(n: int, seed: int = 7) -> np.ndarray:
    a = np.random.default_rng(seed).normal(size=(n, n))
    return a @ a.T / (10 * n) + np.diag(np.full(n, 0.01))


SIGMA = _covariance(4)
ESG_FLOOR = LinearConstraint(ESG, ">=", 65)  # weighted-mean ESG of at least 65


def _bits(n: int):
    for bits in itertools.product((0, 1), repeat=n):
        yield "".join(map(str, bits))


def _make_selection(**kwargs) -> PortfolioSelectionProblem:
    return PortfolioSelectionProblem(
        MU, SIGMA, **{"n_holdings": 2, "risk_tolerance": TAU, **kwargs}
    )


def _make_allocation(**kwargs) -> PortfolioAllocationProblem:
    return PortfolioAllocationProblem(
        MU[:3], SIGMA[:3, :3], **{"n_steps": 3, "risk_tolerance": TAU, **kwargs}
    )


ENCODINGS = [("log", 2), ("domain_wall", 3)]
ESG_72 = LinearConstraint(ESG[:3], ">=", 72)
SELECTION_CASES = [
    pytest.param(_make_selection, 4, id="selection"),
    pytest.param(
        lambda: _make_selection(constraints=[ESG_FLOOR]), 4, id="selection-esg"
    ),
]


@pytest.mark.parametrize(
    "make, n_decision",
    [
        *SELECTION_CASES,
        pytest.param(_make_allocation, 6, id="allocation-log"),
        pytest.param(
            lambda: _make_allocation(encoding="domain_wall"),
            9,
            id="allocation-domain-wall",
        ),
        pytest.param(
            lambda: PortfolioAllocationProblem(
                MU[:2],
                SIGMA[:2, :2],
                n_steps=3,
                constraints=[LinearConstraint(ESG[:2], ">=", 65)],
            ),
            4,
            id="allocation-esg",
        ),
    ],
)
def test_penalty_is_zero_exactly_on_feasible_portfolios(make, n_decision):
    assert_penalty_zero_iff_feasible(make(), n_decision=n_decision)


@pytest.mark.parametrize(
    "make, n_decision",
    [
        *SELECTION_CASES,
        pytest.param(
            lambda: _make_allocation(constraints=[ESG_72]), 6, id="allocation-log-esg"
        ),
        pytest.param(
            lambda: _make_allocation(encoding="domain_wall", constraints=[ESG_72]),
            9,
            id="allocation-domain-wall-esg",
        ),
    ],
)
def test_repair_reaches_feasibility_with_matching_slack(make, n_decision):
    problem = make()
    slack = "0" * n_slack(problem, n_decision)
    for head in _bits(n_decision):
        repaired, decoded, energy = problem.repair_infeasible_bitstring(head + slack)
        assert problem.is_feasible(repaired), head
        np.testing.assert_allclose(decoded, problem.decode_fn(repaired))
        assert energy == pytest.approx(problem.compute_energy(repaired))
        assert penalty_at(problem, repaired) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    "make, n_decision",
    [
        # Over every bit assignment each slack range exceeds 8 bits and would be
        # rounded; over fully invested portfolios it fits in 5.
        pytest.param(
            lambda: PortfolioSelectionProblem(
                MU[:3].repeat(2),
                np.eye(6),
                2,
                constraints=[LinearConstraint([70, 56, 81, 60, 75, 53], ">=", 65)],
            ),
            6,
            id="selection",
        ),
        pytest.param(
            lambda: _make_allocation(
                constraints=[LinearConstraint([71, 55, 80], ">=", 72)]
            ),
            6,
            id="allocation",
        ),
    ],
)
def test_slack_covers_only_fully_invested_portfolios(make, n_decision, recwarn):
    problem = make()
    assert n_slack(problem, n_decision) <= 5
    assert not recwarn.list
    assert_penalty_zero_iff_feasible(problem, n_decision=n_decision)


class TestPortfolioSelection:
    def test_objective_is_mean_variance_at_equal_weight(self):
        problem = _make_selection()
        for bitstring in _bits(4):
            x = np.array([int(b) for b in bitstring], dtype=float)
            expected = x @ SIGMA @ x / 4 - TAU * MU @ x / 2
            assert problem.compute_energy(bitstring) == pytest.approx(expected)

    def test_constraints_returns_the_weight_constraints_passed(self):
        assert _make_selection(constraints=[ESG_FLOOR]).constraints == (ESG_FLOOR,)

    def test_weight_constraints_apply_to_the_equal_weight_portfolio(self):
        problem = _make_selection(constraints=[ESG_FLOOR])
        slack = "0" * n_slack(problem, 4)
        for head in _bits(4):
            x = np.array([int(b) for b in head])
            expected = x.sum() == 2 and ESG @ x / 2 >= 65
            assert problem.is_feasible(head + slack) == expected

    def test_decode_returns_held_indices(self):
        problem = _make_selection(constraints=[ESG_FLOOR])
        assert problem.decode_fn("1010" + "0" * n_slack(problem, 4)) == [0, 2]

    def test_metrics_use_equal_weights(self):
        problem = _make_selection()
        w = np.array([0.5, 0.0, 0.0, 0.5])
        vol = math.sqrt(w @ SIGMA @ w)
        metrics = problem.metrics("1001", risk_free=0.02)
        assert metrics["expected_return"] == pytest.approx(MU @ w)
        assert metrics["volatility"] == pytest.approx(vol)
        assert metrics["sharpe"] == pytest.approx((MU @ w - 0.02) / vol)

    def test_repair_exchanges_two_holdings_when_single_swaps_stall(self):
        # Only {2, 3} is feasible, and every single swap from {0, 1} raises the
        # total violation.
        problem = PortfolioSelectionProblem(
            np.zeros(5),
            np.eye(5),
            n_holdings=2,
            constraints=[
                LinearConstraint({2: 1, 3: -1}, "==", 0),
                LinearConstraint({2: 1, 3: 1, 4: -3}, ">=", 0.5),
            ],
        )
        slack = "0" * n_slack(problem, 5)
        repaired, held, energy = problem.repair_infeasible_bitstring("11000" + slack)
        assert held == [2, 3]
        assert problem.is_feasible(repaired)
        assert energy is not None

    def test_default_mixer_and_initial_state(self):
        problem = _make_selection()
        assert isinstance(problem.recommended_initial_state, SuperpositionState)
        assert problem.mixer_hamiltonian.equiv(
            BinaryOptimizationProblem(problem.raw_problem).mixer_hamiltonian
        )

    def test_constrained_mixer_conserves_asset_weight_and_mixes_slack(self):
        problem = _make_selection(constraints=[ESG_FLOOR], use_constrained_mixer=True)
        n_qubits = problem.cost_hamiltonian.num_qubits
        mixer = problem.mixer_hamiltonian

        weight = SparsePauliOp.from_sparse_list(
            [("Z", [q], -0.5) for q in range(4)], num_qubits=n_qubits
        )
        commutator = (mixer @ weight - weight @ mixer).simplify()
        assert np.allclose(commutator.coeffs, 0.0)

        single_x = [
            label
            for label in mixer.paulis.to_labels()
            if label.count("X") == 1 and "Y" not in label
        ]
        assert len(single_x) == n_slack(problem, 4)

    def test_constrained_initial_state_is_dicke_on_assets(self):
        state = _make_selection(use_constrained_mixer=True).recommended_initial_state
        assert isinstance(state, DickeState)
        assert (state.hamming_weight, state.n_qubits) == (2, 4)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"n_holdings": 0}, "n_holdings"),
            ({"n_holdings": 5}, "n_holdings"),
            ({"risk_tolerance": -1.0}, "risk_tolerance"),
            ({"constraints": [LinearConstraint({7: 1.0}, "<=", 1)]}, "asset index"),
            ({"constraints": [LinearConstraint(ESG, ">=", 90)]}, "infeasible"),
            ({"constraints": [LinearConstraint(ESG[:3], ">=", 65)]}, "3 coefficients"),
        ],
    )
    def test_invalid_arguments_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            _make_selection(**kwargs)

    def test_wrong_length_bitstring_raises(self):
        with pytest.raises(ValueError, match="Expected a bitstring"):
            _make_selection().decode_fn("10")

    @pytest.mark.parametrize(
        "mu, sigma, match",
        [
            (MU, SIGMA[:3, :3], "shape"),
            (MU, SIGMA + np.triu(np.ones((4, 4)), 1), "symmetric"),
            (np.array([np.nan, 0, 0, 0]), SIGMA, "finite"),
        ],
    )
    def test_invalid_market_raises(self, mu, sigma, match):
        with pytest.raises(ValueError, match=match):
            PortfolioSelectionProblem(mu, sigma, 2)

    def test_qaoa_with_constrained_mixer_samples_only_k_holdings(
        self, default_test_simulator, default_optimizer
    ):
        qaoa = QAOA(
            _make_selection(use_constrained_mixer=True),
            n_layers=1,
            max_iterations=2,
            optimizer=default_optimizer,
            backend=default_test_simulator,
        )
        qaoa.run()
        solutions = qaoa.get_top_solutions(n=0)
        assert solutions
        assert all(sol.bitstring.count("1") == 2 for sol in solutions)


class TestPortfolioAllocation:
    @pytest.mark.parametrize("encoding, width", ENCODINGS)
    def test_objective_is_mean_variance_of_encoded_weights(self, encoding, width):
        problem = _make_allocation(encoding=encoding)
        for bitstring in _bits(3 * width):
            w = problem.decode_fn(bitstring)
            expected = w @ SIGMA[:3, :3] @ w - TAU * MU[:3] @ w
            assert problem.compute_energy(bitstring) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "encoding, bitstring, weights",
        [
            ("log", "10" "01" "00", [1 / 3, 2 / 3, 0]),
            ("domain_wall", "100" "110" "000", [1 / 3, 2 / 3, 0]),
        ],
    )
    def test_decode_returns_weights(self, encoding, bitstring, weights):
        problem = _make_allocation(encoding=encoding)
        np.testing.assert_allclose(problem.decode_fn(bitstring), weights)

    @pytest.mark.parametrize("encoding, width", ENCODINGS)
    def test_is_feasible_matches_budget_and_walls(self, encoding, width):
        problem = _make_allocation(encoding=encoding)
        for bitstring in _bits(3 * width):
            blocks = [bitstring[i * width : (i + 1) * width] for i in range(3)]
            walls = encoding == "log" or all(
                b == "1" * b.count("1") + "0" * b.count("0") for b in blocks
            )
            budget = math.isclose(problem.decode_fn(bitstring).sum(), 1.0)
            assert problem.is_feasible(bitstring) == (walls and budget), bitstring

    def test_metrics(self):
        problem = _make_allocation()
        w = np.array([1 / 3, 2 / 3, 0.0])
        metrics = problem.metrics("100100")
        assert metrics["expected_return"] == pytest.approx(MU[:3] @ w)
        assert metrics["volatility"] == pytest.approx(math.sqrt(w @ SIGMA[:3, :3] @ w))

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"n_steps": 4}, "2\\*\\*b - 1"),
            ({"n_steps": 0}, "n_steps"),
            ({"encoding": "unary"}, "encoding"),
            # 0.5 is not a multiple of 1/7.
            (
                {"n_steps": 7, "constraints": [LinearConstraint({0: 1}, "==", 0.5)]},
                "infeasible",
            ),
        ],
    )
    def test_invalid_arguments_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            _make_allocation(**kwargs)

    def test_pce_decodes_and_repairs_through_the_problem(self, default_test_simulator):
        problem = _make_allocation()
        pce = PCE(
            problem,
            encoding_type="poly",
            n_layers=1,
            optimizer=MonteCarloOptimizer(population_size=4),
            max_iterations=1,
            backend=default_test_simulator,
            seed=3,
        )
        pce.run()
        repaired = pce.get_top_solutions(
            n=100, include_decoded=True, feasibility="repair"
        )
        assert repaired
        for sol in repaired:
            assert problem.is_feasible(sol.bitstring)
            np.testing.assert_allclose(sol.decoded, problem.decode_fn(sol.bitstring))
            assert sol.energy == pytest.approx(problem.compute_energy(sol.bitstring))
        assert [s.energy for s in repaired] == sorted(s.energy for s in repaired)
        raw = pce.get_top_solutions(n=1, include_decoded=True)[0]
        np.testing.assert_allclose(raw.decoded, problem.decode_fn(raw.bitstring))
