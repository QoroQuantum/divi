# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
from functools import partial

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import block_diag

from divi.qprog import PCE, QAOA
from divi.qprog.algorithms import DickeState, SuperpositionState
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.optimizers import MonteCarloOptimizer, ScipyMethod, ScipyOptimizer
from divi.qprog.problems import (
    BinaryOptimizationProblem,
    GraphPartitioningConfig,
    LinearConstraint,
    PortfolioAllocationProblem,
    PortfolioSelectionProblem,
    QUBOPartitioningConfig,
)
from divi.qprog.workflows import PartitioningProgramEnsemble
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


# ---------------------------------------------------------------------------
# Partitioned solving
# ---------------------------------------------------------------------------


def _partitioned(cls, sizes, **kwargs):
    """Block-diagonal covariance, so each block of ``sizes`` is one cluster."""
    return cls(
        np.linspace(0.05, 0.15, sum(sizes)),
        block_diag(*(_covariance(s, 11 + i) for i, s in enumerate(sizes))),
        risk_tolerance=TAU,
        config=QUBOPartitioningConfig(max_n_variables_per_cluster=max(sizes)),
        **kwargs,
    )


_partitioned_selection = partial(_partitioned, PortfolioSelectionProblem)
_partitioned_allocation = partial(
    _partitioned, PortfolioAllocationProblem, encoding="domain_wall"
)


def _selection_objectives(problem):
    """Objective of every portfolio that holds ``n_holdings`` of the assets."""
    n = problem.n_assets
    return {
        held: float(problem._objective(np.isin(np.arange(n), held).astype(int)))
        for held in itertools.combinations(range(n), problem.n_holdings)
    }


def _partitioned_ensemble(problem, backend, max_iterations=5):
    return PartitioningProgramEnsemble(
        problem=problem,
        n_layers=1,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=max_iterations,
        backend=backend,
    )


@pytest.fixture
def pin_shares(mocker):
    """Pin each cluster's share of the budget by the cluster's size."""

    def pin(problem, shares_by_size):
        mocker.patch.object(
            problem,
            "_cluster_shares",
            side_effect=lambda clusters, relaxed: np.array(
                [shares_by_size[len(c)] for c in clusters]
            ),
        )
        return problem

    return pin


class TestPortfolioPartitioning:
    @pytest.mark.parametrize(
        "make, shares",
        [
            pytest.param(
                lambda: _partitioned_selection([4, 3], n_holdings=3),
                {4: 2, 3: 1},
                id="selection",
            ),
            pytest.param(
                lambda: _partitioned_allocation([2, 3], n_steps=5),
                {2: 2, 3: 3},
                id="allocation",
            ),
        ],
    )
    def test_cluster_objectives_add_up_to_the_global_objective(
        self, make, shares, pin_shares
    ):
        problem = make()
        subs = pin_shares(problem, shares).decompose()
        assert len(subs) == 2

        rng = np.random.default_rng(0)
        units = np.zeros(problem.n_assets, dtype=int)
        sub_total = 0.0
        for prog_id, sub in subs.items():
            assets, share = problem._clusters[prog_id]
            # A random holding of the cluster's share, respecting each asset's cap.
            cluster_units = np.zeros(len(assets), dtype=int)
            for _ in range(share):
                cluster_units[rng.choice(np.flatnonzero(cluster_units < sub._cap))] += 1
            units[assets] = cluster_units
            sub_total += float(sub._objective(cluster_units))

        assert sub_total == pytest.approx(float(problem._objective(units)))

    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(
                lambda: _partitioned_selection([4, 3, 2], n_holdings=4), id="selection"
            ),
            pytest.param(
                lambda: _partitioned_allocation([2, 3], n_steps=5), id="allocation"
            ),
            pytest.param(
                lambda: _partitioned_selection(
                    [4, 4, 4],
                    n_holdings=6,
                    constraints=[LinearConstraint(np.arange(12.0), ">=", 6)],
                ),
                id="constrained",
            ),
        ],
    )
    def test_shares_add_up_to_the_budget_and_fit_their_clusters(self, make):
        problem = make()
        subs = problem.decompose()
        shares = [s for _, s in problem._clusters.values()]
        assert sum(shares) + sum(problem._fixed_units.values()) == problem._total
        for pid, sub in subs.items():
            assets, share = problem._clusters[pid]
            assert 0 < share < len(assets) * problem._cap
            assert sub._total == share

    def test_shares_follow_the_relaxation(self):
        # Two identical blocks; the second has twice the returns, so the
        # relaxation puts both names there.
        mu = np.r_[np.full(4, 0.05), np.full(4, 0.10)]
        block = np.full((4, 4), 0.01) + np.eye(4) * 0.03
        problem = PortfolioSelectionProblem(
            mu,
            block_diag(block, block),
            2,
            risk_tolerance=TAU,
            config=QUBOPartitioningConfig(max_n_variables_per_cluster=4),
        )
        problem.decompose()
        assert [(a.tolist(), s) for a, s in problem._clusters.values()] == [
            ([4, 5, 6, 7], 2)
        ]
        assert problem._fixed_units == dict.fromkeys(range(4), 0)

    @pytest.mark.parametrize(
        "constraint, shares",
        [
            # Unconstrained, all four names go to the high-return block.
            pytest.param(None, {0: 0, 5: 4}, id="none"),
            pytest.param(
                LinearConstraint({i: 1.0 for i in range(5)}, ">=", 0.5),
                {0: 2, 5: 2},
                id=">=",
            ),
            pytest.param(
                LinearConstraint({i: 1.0 for i in range(5, 10)}, "<=", 0.5),
                {0: 2, 5: 2},
                id="<=",
            ),
            pytest.param(
                LinearConstraint({i: 1.0 for i in range(5)}, "==", 0.25),
                {0: 1, 5: 3},
                id="==",
            ),
        ],
    )
    @pytest.mark.parametrize("scale", [1.0, 1 / 252])  # annual and daily data
    def test_shares_follow_the_constrained_relaxation(self, constraint, shares, scale):
        mu = np.r_[np.full(5, 0.05), np.full(5, 0.10)] * scale
        block = (np.full((5, 5), 0.01) + np.eye(5) * 0.03) * scale
        problem = PortfolioSelectionProblem(
            mu,
            block_diag(block, block),
            4,
            risk_tolerance=TAU,
            constraints=[constraint] if constraint else [],
            config=QUBOPartitioningConfig(max_n_variables_per_cluster=5),
        )
        problem.decompose()
        by_block = {int(a[0]): s for a, s in problem._clusters.values()}
        by_block.update({a: u for a, u in problem._fixed_units.items() if a in (0, 5)})
        assert by_block == shares

    def test_shares_fall_back_to_cluster_sizes_without_a_relaxation(self, mocker):
        mocker.patch(
            "divi.qprog.problems._portfolio.minimize",
            return_value=mocker.Mock(success=False),
        )
        # Quotas 1.78, 1.33, 0.89 by size: floors 1, 1, 0 and the leftover
        # names go to the largest remainders.
        problem = _partitioned_selection([4, 3, 2], n_holdings=4)
        problem.decompose()
        assert {len(a): s for a, s in problem._clusters.values()} == {4: 2, 3: 1, 2: 1}

    def test_decompose_is_reproducible(self):
        problem = _partitioned_selection([4, 3, 2], n_holdings=4)
        problem.decompose()
        first = {pid: (a.tolist(), s) for pid, (a, s) in problem._clusters.items()}
        problem.decompose()
        assert {
            pid: (a.tolist(), s) for pid, (a, s) in problem._clusters.items()
        } == first

    def test_sub_problems_inherit_mixer_and_penalty_weight(self, pin_shares):
        problem = _partitioned_selection(
            [4, 3], n_holdings=3, use_constrained_mixer=True, penalty_weight=2.5
        )
        for sub in pin_shares(problem, {4: 2, 3: 1}).decompose().values():
            assert sub.use_constrained_mixer
            assert sub._penalty_weight == 2.5
            assert not sub.constraints

    @pytest.mark.parametrize(
        "make, shares, fixed",
        [
            pytest.param(
                lambda: _partitioned_selection([4, 1], n_holdings=2),
                {4: 2, 1: 0},
                {4: 0},
                id="no-share",
            ),
            pytest.param(
                lambda: _partitioned_selection([4, 2], n_holdings=4),
                {4: 2, 2: 2},
                {4: 1, 5: 1},
                id="every-asset-held",
            ),
            pytest.param(
                lambda: _partitioned_allocation([3, 1], n_steps=4),
                {3: 3, 1: 1},
                {3: 1},
                id="single-asset",
            ),
        ],
    )
    def test_clusters_with_one_possible_holding_get_no_program(
        self, make, shares, fixed, pin_shares
    ):
        problem = make()
        assert len(pin_shares(problem, shares).decompose()) == 1
        assert problem._fixed_units == fixed

    def test_shares_fall_back_to_cluster_sizes_when_the_relaxation_leaves_nothing(
        self, mocker
    ):
        problem = _partitioned_selection([4, 3], n_holdings=3)
        spy = mocker.spy(problem, "_cluster_shares")
        subs = problem.decompose()
        assert [c.kwargs["relaxed"] for c in spy.call_args_list] == [True, False]
        # The relaxation fills the 3-asset block and empties the 4-asset one.
        assert sorted(spy.spy_return_list[0].tolist()) == [0, 3]
        assert {len(a): s for a, s in problem._clusters.values()} == {4: 2, 3: 1}
        assert len(subs) == 2

    def test_decompose_raises_when_no_cluster_is_left_to_solve(self, mocker):
        problem = _partitioned_selection([2, 2], n_holdings=4)
        spy = mocker.spy(problem, "_cluster_shares")
        with pytest.raises(ValueError, match="without partitioning"):
            problem.decompose()
        assert [c.kwargs["relaxed"] for c in spy.call_args_list] == [True, False]

    @pytest.mark.parametrize(
        "limits, program_sizes, n_fixed",
        [
            ({"max_n_variables_per_cluster": 4}, [4], 0),
            ({"max_n_variables_per_cluster": 2}, [2, 2], 0),
            # The floor of 3 clusters splits the four assets 2 + 1 + 1.
            ({"max_n_variables_per_cluster": 4, "minimum_n_clusters": 3}, [2], 2),
        ],
    )
    def test_uncorrelated_assets_are_grouped_within_the_limits(
        self, limits, program_sizes, n_fixed
    ):
        problem = PortfolioSelectionProblem(
            MU, np.diag(np.diag(SIGMA)), 2, config=QUBOPartitioningConfig(**limits)
        )
        problem.decompose()
        assert sorted(len(a) for a, _ in problem._clusters.values()) == program_sizes
        assert len(problem._fixed_units) == n_fixed

    @pytest.mark.parametrize(
        "make, shares, decoded, expected_units",
        [
            pytest.param(
                lambda: _partitioned_selection([2, 3, 1], n_holdings=3),
                {2: 1, 3: 2, 1: 0},
                [1],
                [0, 1],
                id="selection",
            ),
            pytest.param(
                lambda: _partitioned_allocation([2, 3, 1], n_steps=6),
                {2: 2, 3: 3, 1: 1},
                [0.0, 1.0],
                [0, 2],
                id="allocation",
            ),
        ],
    )
    def test_extend_solution_writes_only_its_cluster(
        self, make, shares, decoded, expected_units, pin_shares
    ):
        problem = make()
        pin_shares(problem, shares).decompose()
        # The 2-asset block is assets 0 and 1.
        (prog_id,) = [pid for pid, (a, _) in problem._clusters.items() if len(a) == 2]
        extended = problem.extend_solution([7] * problem.n_assets, prog_id, decoded)
        assert extended == [*expected_units, 7, 7, 7, 7]

    @pytest.mark.parametrize(
        "constraint, satisfied, violated",
        [
            pytest.param(
                LinearConstraint([90, 90, 40, 40, 90, 40, 40], ">=", 70),
                [1, 1, 0, 0, 1, 0, 0],
                [0, 0, 1, 1, 0, 1, 0],
                id=">=",
            ),
            pytest.param(
                LinearConstraint({0: 1, 1: 1}, "<=", 1 / 3),
                [1, 0, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0],
                id="<=",
            ),
            pytest.param(
                LinearConstraint({0: 1, 4: 1}, "==", 2 / 3),
                [1, 0, 1, 0, 1, 0, 0],
                [1, 0, 1, 1, 0, 0, 0],
                id="==",
            ),
        ],
    )
    def test_evaluate_penalises_only_violated_constraints(
        self, constraint, satisfied, violated, pin_shares
    ):
        problem = _partitioned_selection([4, 3], n_holdings=3, constraints=[constraint])
        pin_shares(problem, {4: 2, 3: 1}).decompose()
        objective = float(problem._objective(np.array(satisfied)))
        assert problem.evaluate_global_solution(satisfied) == pytest.approx(
            objective, abs=1e-9
        )
        objective = float(problem._objective(np.array(violated)))
        assert problem.evaluate_global_solution(violated) >= objective + 1 - 1e-9

    def test_evaluate_penalises_a_wrong_budget(self, pin_shares):
        problem = _partitioned_selection([4, 3], n_holdings=3)
        pin_shares(problem, {4: 2, 3: 1}).decompose()
        x = [1, 1, 1, 1, 0, 0, 0]
        assert problem.evaluate_global_solution(x) == pytest.approx(
            float(problem._objective(np.array(x))) + 1.0
        )

    def test_evaluate_and_postprocess_use_the_fixed_assets(self, pin_shares):
        problem = _partitioned_selection([4, 3], n_holdings=2)
        pin_shares(problem, {4: 2, 3: 0}).decompose()
        clean, junk = [1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 1, 1, 1]
        assert problem.evaluate_global_solution(junk) == pytest.approx(
            problem.evaluate_global_solution(clean)
        )
        assert problem.postprocess_candidates(
            [(0.0, junk)]
        ) == problem.postprocess_candidates([(0.0, clean)])

    def test_postprocess_never_worsens_a_candidate(self, pin_shares):
        problem = _partitioned_selection([4, 3], n_holdings=3)
        pin_shares(problem, {4: 2, 3: 1}).decompose()
        for held, objective in _selection_objectives(problem).items():
            x = np.isin(np.arange(7), held).astype(int).tolist()
            ((_, energy),) = problem.postprocess_candidates([(0.0, x)])
            assert energy <= objective + 1e-12

    def test_postprocess_merges_candidates_that_improve_to_the_same_portfolio(
        self, pin_shares
    ):
        problem = _partitioned_selection([4, 3], n_holdings=3)
        pin_shares(problem, {4: 2, 3: 1}).decompose()
        optimum = min(_selection_objectives(problem).values())
        candidates = [
            (0.0, [1, 1, 0, 0, 1, 0, 0]),
            (0.0, [0, 0, 1, 1, 0, 0, 1]),
        ]
        ((_, energy),) = problem.postprocess_candidates(candidates)
        assert energy == pytest.approx(optimum)

    def test_postprocess_repairs_and_improves_within_the_constraints(self, pin_shares):
        esg = np.array([90.0, 90.0, 40.0, 40.0, 90.0, 40.0, 40.0])
        problem = _partitioned_selection(
            [4, 3], n_holdings=3, constraints=[LinearConstraint(esg, ">=", 70)]
        )
        pin_shares(problem, {4: 2, 3: 1}).decompose()
        ((decoded, energy),) = problem.postprocess_candidates(
            [(0.0, [0, 0, 1, 1, 0, 1, 0])]
        )
        objective = _selection_objectives(problem)
        feasible = {h: o for h, o in objective.items() if esg[list(h)].mean() >= 70}
        # Without the constraint the polish would end somewhere infeasible.
        assert min(objective, key=objective.get) not in feasible
        assert tuple(decoded) in feasible
        assert energy == pytest.approx(min(feasible.values()))

    def test_postprocess_strict_drops_infeasible_candidates(self, pin_shares):
        esg = np.array([90.0, 90.0, 40.0, 40.0, 90.0, 40.0, 40.0])
        problem = _partitioned_selection(
            [4, 3], n_holdings=3, constraints=[LinearConstraint(esg, ">=", 70)]
        )
        pin_shares(problem, {4: 2, 3: 1}).decompose()
        assert (
            problem.postprocess_candidates([(0.0, [0, 0, 1, 1, 0, 1, 0])], strict=True)
            == []
        )

    def test_decompose_without_config_raises(self):
        with pytest.raises(ValueError, match="no partitioning config"):
            _make_selection().decompose()

    def test_rejects_a_graph_config(self):
        with pytest.raises(TypeError, match="QUBOPartitioningConfig"):
            _make_selection(config=GraphPartitioningConfig(max_n_nodes_per_cluster=2))

    def test_log_encoded_allocation_rejects_partitioning(self):
        with pytest.raises(ValueError, match="domain_wall"):
            _make_allocation(
                config=QUBOPartitioningConfig(max_n_variables_per_cluster=2)
            )

    def test_partitioned_selection_beats_its_cluster_budgets(
        self, default_test_simulator
    ):
        default_test_simulator.set_seed(1997)
        problem = _partitioned_selection(
            [4, 4], n_holdings=4, use_constrained_mixer=True
        )
        ensemble = _partitioned_ensemble(problem, default_test_simulator)
        ensemble.run()

        holdings, energy = ensemble.aggregate_results()

        best_for_budgets = min(
            objective
            for held, objective in _selection_objectives(problem).items()
            if all(np.isin(a, held).sum() == s for a, s in problem._clusters.values())
        )
        assert len(holdings) == 4
        assert energy <= best_for_budgets + 1e-12

    def test_partitioned_allocation_returns_a_fully_invested_portfolio(
        self, default_test_simulator
    ):
        default_test_simulator.set_seed(1997)
        problem = _partitioned_allocation([2, 2], n_steps=3)
        ensemble = _partitioned_ensemble(problem, default_test_simulator)
        ensemble.run()

        weights, energy = ensemble.aggregate_results()

        units = weights * 3
        np.testing.assert_allclose(units, np.rint(units))
        assert units.sum() == pytest.approx(3)
        assert energy == pytest.approx(float(problem._objective(np.rint(units))))

    def test_partitioned_run_supports_checkpointing(
        self, dummy_simulator, tmp_path, pin_shares
    ):
        problem = pin_shares(_partitioned_selection([4, 3], n_holdings=3), {4: 2, 3: 1})
        ensemble = _partitioned_ensemble(problem, dummy_simulator, max_iterations=1)

        ensemble.run(checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path))

        assert (tmp_path / "round_001" / "round_completion.json").is_file()
