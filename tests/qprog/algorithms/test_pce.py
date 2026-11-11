# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import warnings

import dimod
import numpy as np
import pytest
from qiskit.circuit.library import CXGate, CYGate, RYGate, RZGate
from qiskit.quantum_info import SparsePauliOp

from divi.backends import MaestroConfig
from divi.circuits._conversions import _QISKIT_TO_QASM2
from divi.hamiltonians._polynomial import _evaluate_binary_polynomial
from divi.pipeline import ResultFormat
from divi.pipeline.stages import MeasurementStage
from divi.qprog.algorithms import (
    PCE,
    CVaRObjective,
    DenseEncoding,
    ExpectationObjective,
    GenericLayerAnsatz,
    MultiBasisEncoding,
    PolyEncoding,
    SoftObjective,
)
from divi.qprog.algorithms._pce import _decode_parities, _pack_masks
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.optimizers import (
    MonteCarloOptimizer,
    QNGOptimizer,
    ScipyMethod,
    ScipyOptimizer,
)
from divi.qprog.problems import BinaryOptimizationProblem
from tests._helpers import exact_match
from tests.qprog._helpers import central_difference_of_loss
from tests.qprog._program_contracts import (
    ObservableMeasuringContractsBase,
    verify_cost_circuit,
)
from tests.qprog.algorithms._helpers import seed_best_probs
from tests.qprog.problems._helpers import (
    HUBO_CUBIC,
    PCE_QUBO_MATRIX,
    PCE_QUBO_SOLUTION,
    exact_hubo_minima,
)

_QUBO_3 = np.array([[1.0, -2.0, 0.5], [0.0, 3.0, 1.5], [0.0, 0.0, -1.0]])

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_ansatz() -> GenericLayerAnsatz:
    return GenericLayerAnsatz([RYGate, RZGate])


@pytest.fixture
def qubo_identity() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, 1.0]])


@pytest.fixture
def make_pce(basic_ansatz, dummy_simulator, default_optimizer):
    """Build a PCE from standard test defaults.

    ``problem``, ``ansatz``, ``optimizer``, and ``backend`` all have sensible
    defaults; any can be overridden per call, e.g.
    ``make_pce(backend=default_test_simulator, optimizer=ScipyOptimizer(...))``.
    """
    _default_problem = np.array([[1.0, 0.2], [0.2, 2.0]])

    def _make(**kwargs):
        problem = kwargs.pop("problem", _default_problem)
        if not isinstance(problem, BinaryOptimizationProblem):
            problem = BinaryOptimizationProblem(problem)
        kwargs["problem"] = problem
        kwargs.setdefault("ansatz", basic_ansatz)
        kwargs.setdefault("optimizer", default_optimizer)
        kwargs.setdefault("backend", dummy_simulator)
        return PCE(**kwargs)

    return _make


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _histogram_to_probs(histogram):
    """Convert a {state: count} dict to (state_strings, probs)."""
    state_strings = list(histogram.keys())
    counts = np.array(list(histogram.values()), dtype=float)
    probs = counts / counts.sum()
    return state_strings, probs


def _read_solution(pce):
    """Read pce.solution while asserting the expected deprecation warning."""
    with pytest.warns(UserWarning, match="PCE.solution returns the decoded"):
        return pce.solution


def _set_probs(pce, probs_dict, *, key="0_NoMitigation:0_ham:0_0"):
    """Seed ``pce`` under the result key its own pipeline produces."""
    seed_best_probs(pce, probs_dict, key)


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


def test_pce_basic_initialization(make_pce):
    pce = make_pce(
        problem=np.array([[1.0, 0.2], [0.2, 2.0]]),
        n_layers=1,
    )

    assert pce.n_vars == 2
    assert pce.n_qubits == 2  # ceil(log2(2 + 1)) = 2
    verify_cost_circuit(pce)


def test_pce_hubo_basic_initialization(make_pce):
    hubo = {
        ("x0",): -1.0,
        ("x0", "x1"): 0.5,
        ("x0", "x1", "x2"): 2.0,
        (): 0.25,
    }

    pce = make_pce(problem=hubo, n_layers=1)

    assert pce.n_vars == 3
    assert pce.problem.canonical_problem.constant == pytest.approx(0.25)
    verify_cost_circuit(pce)


@pytest.mark.parametrize(
    "objective, result_format",
    [(SoftObjective(), ResultFormat.EXPVALS), (CVaRObjective(), ResultFormat.COUNTS)],
    ids=["soft", "cvar"],
)
def test_pce_cost_pipeline_measures_through_measurement_stage(
    make_pce, objective, result_format
):
    """Expectation objectives read the variables' expectations, sample
    objectives the raw counts, both through the ordinary MeasurementStage."""
    pce = make_pce(n_layers=1, objective=objective)

    preprocessor = pce.cost_preprocessor()
    pipeline = pce._build_preprocessor_pipeline(preprocessor)
    assert preprocessor.result_format is result_format
    assert any(isinstance(stage, MeasurementStage) for stage in pipeline.stages)


@pytest.mark.parametrize(
    "encoding, strings",
    [
        (DenseEncoding(), ["ZI", "IZ", "ZZ"]),
        (PolyEncoding(), ["ZI", "IZ", "ZZ"]),
        (MultiBasisEncoding(order=1), ["X", "Y", "Z"]),
    ],
    ids=["dense", "poly", "multi-basis"],
)
def test_pce_cost_circuit_measures_one_pauli_string_per_variable(
    make_pce, encoding, strings
):
    """Mask bit j is bitstring position n - 1 - j, so dense variable 0 (mask 0b01)
    is Z on qubit 1, which Qiskit labels ZI."""
    pce = make_pce(problem=np.eye(3), encoding=encoding)
    assert [str(op.paulis[0]) for op in pce.cost_circuit.observable] == strings


def test_pce_cost_hamiltonian_is_read_only(make_pce):
    pce = make_pce(problem=np.eye(3))

    with pytest.raises(AttributeError):
        pce.cost_hamiltonian = SparsePauliOp(["ZZ"])


_EXTRA_QUBITS_WARNING = exact_match(
    "n_qubits exceeds the minimum required; extra qubits increase circuit "
    "size and can add noise without representing more variables."
)


@pytest.mark.parametrize(
    "n_vars, too_few, too_many, minimum",
    [
        (3, DenseEncoding(n_qubits=1), DenseEncoding(n_qubits=3), 2),
        (4, PolyEncoding(n_qubits=2), PolyEncoding(n_qubits=4), 3),
        (
            10,
            MultiBasisEncoding(order=2, n_qubits=3),
            MultiBasisEncoding(order=2, n_qubits=5),
            4,
        ),
    ],
    ids=["dense", "poly", "multi-basis"],
)
def test_pce_n_qubits_validation_and_warning(
    make_pce, n_vars, too_few, too_many, minimum
):
    qubo = np.zeros((n_vars, n_vars))

    with pytest.raises(
        ValueError,
        match=exact_match(
            f"{type(too_few).__name__} needs n_qubits >= {minimum} for {n_vars} "
            f"variables, got n_qubits={too_few.n_qubits}."
        ),
    ):
        make_pce(problem=qubo, encoding=too_few)

    with pytest.warns(UserWarning, match=_EXTRA_QUBITS_WARNING):
        pce = make_pce(problem=qubo, encoding=too_many)
    assert pce.n_qubits == too_many.n_qubits


@pytest.mark.parametrize(
    "n_vars, encoding",
    [
        (3, DenseEncoding(n_qubits=2)),
        (6, PolyEncoding(n_qubits=3)),
        (9, MultiBasisEncoding(order=2, n_qubits=3)),
    ],
    ids=["dense", "poly", "multi-basis"],
)
def test_pce_accepts_the_minimum_n_qubits_silently(make_pce, n_vars, encoding):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        pce = make_pce(problem=np.zeros((n_vars, n_vars)), encoding=encoding)

    assert pce.n_qubits == encoding.n_qubits


@pytest.mark.parametrize(
    "n_vars, expected_masks",
    [(5, [1, 2, 4, 3, 5]), (6, [1, 2, 4, 3, 5, 6])],
    ids=["stops-at-n-vars", "reaches-the-last-pair"],
)
def test_poly_encoding_masks(make_pce, n_vars, expected_masks):
    pce = make_pce(problem=np.zeros((n_vars, n_vars)), encoding=PolyEncoding())

    assert pce.n_qubits == 3
    assert pce.encoding._masks(n_vars, pce.n_qubits).tolist() == expected_masks


def test_pce_default_ansatz_is_hardware_efficient_and_entangling(
    dummy_simulator, default_optimizer
):
    """PCE defaults to an entangling GenericLayerAnsatz, not VQE's chemistry ansatz.

    Regression: VQE's HartreeFockAnsatz default queries n_electrons (None for
    PCE) and raises; PCE must supply its own applicable default.
    """
    pce = PCE(
        problem=BinaryOptimizationProblem(
            np.array([[1.0, 0.2, 0.0], [0.2, 2.0, 0.3], [0.0, 0.3, 1.5]])
        ),
        backend=dummy_simulator,
        optimizer=default_optimizer,
    )

    assert isinstance(pce.ansatz, GenericLayerAnsatz)
    assert pce.ansatz.entangler is not None
    # n_params_per_layer must resolve without an n_electrons-dependent ansatz.
    assert pce.n_params_per_layer > 0

    cost_dag = pce._create_cost_circuit().circuit_bodies[0][1]
    two_qubit_ops = [n.op.name for n in cost_dag.op_nodes() if n.op.num_qubits == 2]
    assert two_qubit_ops, "default PCE ansatz must be entangling"


def test_pce_lowers_ansatz_gates_outside_the_qasm2_basis(
    default_test_simulator, default_optimizer
):
    """Regression test: an ansatz emitting non-QASM2 gates must complete a run.

    PCE's cost-circuit builder used to hand the DAG to the emitter unlowered,
    raising ``ValueError`` at circuit submission.
    """
    assert "cy" not in _QISKIT_TO_QASM2
    pce = PCE(
        problem=BinaryOptimizationProblem(np.array([[1.0, 0.2], [0.2, 2.0]])),
        ansatz=GenericLayerAnsatz([RYGate], entangler=CYGate),
        optimizer=default_optimizer,
        max_iterations=2,
        backend=default_test_simulator,
    )

    cost_dag = pce._create_cost_circuit().circuit_bodies[0][1]
    unsupported = {
        node.op.name
        for node in cost_dag.op_nodes()
        if node.op.name not in _QISKIT_TO_QASM2
    }
    assert not unsupported, f"gates outside the QASM2 basis: {sorted(unsupported)}"

    pce.run()

    assert len(pce.losses_history) == 2


def test_pce_qem_protocol_raises(make_pce):
    """PCE raises when qem_protocol is passed (not supported)."""
    with pytest.raises(ValueError, match="PCE does not currently support qem_protocol"):
        make_pce(problem=np.eye(2), qem_protocol=object())


def test_pce_rejects_a_non_encoding(make_pce):
    with pytest.raises(
        TypeError, match=exact_match("encoding must be a PCEEncoding, got str.")
    ):
        make_pce(encoding="dense")


def test_pce_rejects_a_sample_objective_on_multi_basis(make_pce):
    with pytest.raises(
        TypeError,
        match=exact_match(
            "CVaRObjective scores sampled bitstrings, which MultiBasisEncoding "
            "cannot decode. Use an ExpectationObjective such as SoftObjective."
        ),
    ):
        make_pce(encoding=MultiBasisEncoding(), objective=CVaRObjective())


def test_multi_basis_encoding_rejects_order_zero():
    with pytest.raises(
        ValueError, match=exact_match("order must be at least 1, got 0")
    ):
        MultiBasisEncoding(order=0)


@pytest.mark.parametrize(
    "order, n_vars, n_qubits, first, last",
    [
        (1, 3, 1, ("X", (0,)), ("Z", (0,))),
        (2, 9, 3, ("XX", (0, 1)), ("ZZ", (1, 2))),
        (2, 10, 4, ("XX", (0, 1)), ("YY", (1, 2))),
        (3, 12, 4, ("XXX", (0, 1, 2)), ("ZZZ", (1, 2, 3))),
    ],
    ids=["order-1", "order-2-full", "order-2-partial", "order-3-full"],
)
def test_multi_basis_layout_fills_letters_then_subsets(
    order, n_vars, n_qubits, first, last
):
    """q qubits hold 3*C(q, k) variables: X strings first, then Y, then Z."""
    q, strings = MultiBasisEncoding(order=order).layout(n_vars)
    assert q == n_qubits
    assert len(strings) == n_vars
    assert strings[0] == first
    assert strings[-1] == last


def test_pce_rejects_non_binary_problem():
    """PCE only accepts a BinaryOptimizationProblem; a raw QUBO raises clearly."""
    with pytest.raises(
        TypeError,
        match=exact_match(
            "PCE requires a BinaryOptimizationProblem, got ndarray. "
            "Wrap your QUBO/HUBO: PCE(BinaryOptimizationProblem(qubo))."
        ),
    ):
        PCE(np.array([[1.0, 0.0], [0.0, 1.0]]))


def _two_variable_pce(basic_ansatz, backend, optimizer, **kwargs):
    return PCE(
        problem=BinaryOptimizationProblem(np.array([[1.0, 0.2], [0.2, 2.0]])),
        ansatz=basic_ansatz,
        optimizer=optimizer,
        backend=backend,
        max_iterations=1,
        **kwargs,
    )


def test_pce_cvar_expval_backend_raises(
    basic_ansatz, default_test_simulator, default_optimizer
):
    pce = _two_variable_pce(
        basic_ansatz,
        default_test_simulator,
        default_optimizer,
        objective=CVaRObjective(),
    )
    with pytest.raises(
        ValueError,
        match=exact_match(
            "PCE with CVaRObjective requires shot histograms and "
            "cannot use expectation-value backends. Use a sampling backend: "
            "MaestroSimulator(force_sampling=True), "
            "QiskitSimulator(force_sampling=True), or QoroService with "
            "JobConfig(force_sampling=True)."
        ),
    ):
        pce.run()


@pytest.mark.parametrize(
    "objective,force_sampling",
    [(SoftObjective(), False), (CVaRObjective(), True)],
    ids=["soft-objective", "cvar-forced-sampling"],
)
def test_pce_runs_when_the_backend_suits_the_objective(
    objective, force_sampling, basic_ansatz, make_maestro_simulator, default_optimizer
):
    backend = make_maestro_simulator(force_sampling=force_sampling)
    pce = _two_variable_pce(
        basic_ansatz, backend, default_optimizer, objective=objective
    )
    pce.run()
    assert len(pce.losses_history) == 1
    assert np.isfinite(pce.best_loss)


class _ConstantObjective(ExpectationObjective):
    def score(self, z, problem):
        return 7.0

    def gradient(self, z, problem):
        return np.zeros_like(z)


def test_pce_minimises_a_custom_objective(make_pce):
    pce = make_pce(objective=_ConstantObjective(), max_iterations=1)
    pce.run(perform_final_computation=False)
    assert pce.best_loss == pytest.approx(7.0)


def test_pce_defaults_to_the_soft_objective(make_pce):
    assert make_pce().objective == SoftObjective(alpha=2.0)


@pytest.mark.parametrize(
    "order, n_vars, n_qubits, alpha",
    [(1, 3, 1, 1.5), (2, 9, 3, 4.5), (3, 12, 4, 6.0), (4, 15, 5, 37.5)],
    ids=["k1", "k2", "k3", "k4"],
)
def test_multi_basis_defaults_alpha_to_its_qubit_scaling(
    make_pce, order, n_vars, n_qubits, alpha
):
    """alpha = 1.5 * n ** (k // 2), after Sciorilli et al. (2025)."""
    pce = make_pce(
        problem=np.zeros((n_vars, n_vars)), encoding=MultiBasisEncoding(order=order)
    )

    assert pce.n_qubits == n_qubits
    assert pce.objective == SoftObjective(alpha=alpha)


def test_pce_rejects_a_non_objective(make_pce):
    with pytest.raises(
        TypeError,
        match=exact_match(
            "objective must be an ExpectationObjective or SampleObjective, got str."
        ),
    ):
        make_pce(objective="cvar")


@pytest.mark.parametrize(
    "build, message",
    [
        (
            lambda: SoftObjective(alpha=0.0),
            "alpha must be a positive finite number, got 0.0",
        ),
        (
            lambda: SoftObjective(alpha=float("inf")),
            "alpha must be a positive finite number, got inf",
        ),
        (lambda: CVaRObjective(fraction=0.0), "fraction must be in (0, 1], got 0.0"),
        (lambda: CVaRObjective(fraction=1.5), "fraction must be in (0, 1], got 1.5"),
    ],
    ids=["zero-alpha", "infinite-alpha", "zero-fraction", "fraction-above-one"],
)
def test_objectives_reject_out_of_range_settings(build, message):
    with pytest.raises(ValueError, match=exact_match(message)):
        build()


# ---------------------------------------------------------------------------
# Energy computation helpers
# ---------------------------------------------------------------------------


def _parity_expectations(pce, histogram):
    """Each variable's Z-parity expectation over a ``{state: count}`` histogram."""
    states, probs = _histogram_to_probs(histogram)
    masks = pce.encoding._masks(pce.n_vars, pce.n_qubits)
    return 1.0 - 2.0 * _decode_parities(states, masks).dot(probs)


def _score(objective, pce, *args):
    return objective.score(*args, pce.problem)


def test_soft_objective_scores_a_cubic_hubo(make_pce):
    """hubo = -x0 + 0.25*x1 + 1.5*x0*x1*x2, alpha = 1, histogram (100 shots):
      "000" count=30, "001"=10, "010"=20, "011"=40.
    Parities (x0, x1, x2): 000→[0,0,0], 001→[1,0,1], 010→[0,1,1], 011→[1,1,0].
    mean_parity = [0.5, 0.6, 0.3] → z = [0, -0.2, 0.4].
    x = 0.5*(1 + tanh(z)), and the energy is the hubo's multilinear extension.
    """
    hubo = {
        ("x0",): -1.0,
        ("x1",): 0.25,
        ("x0", "x1", "x2"): 1.5,
    }
    pce = make_pce(problem=hubo)
    z = _parity_expectations(pce, {"000": 30, "001": 10, "010": 20, "011": 40})

    energy = _score(SoftObjective(alpha=1.0), pce, z)

    x0, x1, x2 = (0.5 * (1.0 + np.tanh(z)) for z in (0.0, -0.2, 0.4))
    expected = -1.0 * x0 + 0.25 * x1 + 1.5 * x0 * x1 * x2
    assert energy == pytest.approx(expected)


def test_soft_objective_scores_a_mixed_histogram(make_pce):
    """qubo = diag([1, 2]), alpha = 1, histogram: {"00": 3, "11": 1} (4 shots)
    masks = [1, 2] (one qubit per variable).

    mean_parity = [0.25, 0.25], so z = [0.5, 0.5].
    x = 0.5*(1 + tanh(0.5)) = e/(e+1)   [algebraic identity].
    The diagonal entries are linear terms: energy = 1*x + 2*x = 3*e/(e+1).
    """
    pce = make_pce(problem=np.diag([1.0, 2.0]))
    z = _parity_expectations(pce, {"00": 3, "11": 1})

    result = _score(SoftObjective(alpha=1.0), pce, z)

    assert result == pytest.approx(3.0 * np.e / (np.e + 1))


def test_soft_objective_gradient_matches_central_differences(make_pce):
    pce = make_pce(problem=_QUBO_3)
    objective = SoftObjective(alpha=1.7)
    z = np.array([0.2, -0.5, 0.7])
    step = 1e-6
    numeric = [
        (_score(objective, pce, z + step * e) - _score(objective, pce, z - step * e))
        / (2 * step)
        for e in np.eye(3)
    ]

    np.testing.assert_allclose(objective.gradient(z, pce.problem), numeric, rtol=1e-6)


@pytest.mark.parametrize(
    "histogram, expected",
    [
        pytest.param({"11": 4, "10": 2, "01": 3, "00": 1}, 0.2, id="zero-energy-head"),
        pytest.param({"10": 2, "01": 3, "00": 5}, 1.6, id="nonzero-tail"),
    ],
)
def test_cvar_objective_averages_the_low_energy_tail(make_pce, histogram, expected):
    """qubo = diag([1, 2]), fraction = 0.5, 10 shots; "11" → x=[0,0] (energy 0),
    "10" → x=[1,0] (1), "01" → x=[0,1] (2), "00" → x=[1,1] (3).

    CVaR(0.5) averages the best ceil(0.5*10)=5 shots:
      zero-energy-head: 4 shots at 0 + 1 shot at 1 → 1/5 = 0.2
      nonzero-tail: 2 shots at 1 + 3 shots at 2 → 8/5 = 1.6
    """
    pce = make_pce(problem=np.diag([1.0, 2.0]), objective=CVaRObjective(fraction=0.5))

    assert pce._score_histogram(pce.objective, histogram) == pytest.approx(expected)


@pytest.mark.parametrize("histogram", [{}, {"00": 0}], ids=["empty", "zero-counts"])
def test_cvar_scoring_rejects_a_histogram_without_shots(make_pce, histogram):
    pce = make_pce(objective=CVaRObjective())
    with pytest.raises(
        ValueError,
        match=exact_match(
            "Backend returned an empty PCE histogram; the energy is undefined "
            "without shots. This indicates a backend fault, so check the job's "
            "results."
        ),
    ):
        pce._score_histogram(pce.objective, histogram)


@pytest.mark.parametrize("state", ["0", "000"], ids=["narrower", "wider"])
def test_cvar_scoring_rejects_bitstrings_not_spanning_the_register(make_pce, state):
    pce = make_pce(objective=CVaRObjective())
    with pytest.raises(
        ValueError,
        match=exact_match(
            "Backend returned PCE bitstrings narrower or wider than the 2-qubit "
            "register; every bitstring must report all qubits for the variables' "
            "parities to decode."
        ),
    ):
        pce._score_histogram(pce.objective, {state: 10})


def test_cvar_cutoff_does_not_round_a_float_product_up(make_pce):
    """0.07 * 100 is 7.000000000000001 in floating point; the tail is still 7
    shots, all at energy 0, not 8 with one at energy 3."""
    pce = make_pce(problem=np.diag([1.0, 2.0]), objective=CVaRObjective(fraction=0.07))
    assert pce._score_histogram(pce.objective, {"11": 7, "00": 93}) == 0.0


@pytest.mark.parametrize("z", [17.0, -18.5], ids=["positive", "negative"])
def test_soft_objective_gradient_keeps_precision_near_saturation(make_pce, z):
    """sech^2 is computed without 1 - tanh^2, and x without 1 + tanh, both of
    which cancel catastrophically once tanh rounds towards +-1."""
    pce = make_pce(problem=np.array([[1.0]]))
    gradient = SoftObjective(alpha=1.0).gradient(np.array([z]), pce.problem)
    np.testing.assert_allclose(gradient, [0.5 / np.cosh(z) ** 2], rtol=1e-12)


def test_soft_objective_keeps_the_precision_of_a_vanishing_variable(make_pce):
    """At z = -25, (1 + tanh(z)) / 2 rounds to 0; x is exp(-50) / (1 + exp(-50))."""
    pce = make_pce(problem=np.array([[1.0]]))
    x = np.exp(-50.0) / (1.0 + np.exp(-50.0))
    assert _score(SoftObjective(alpha=1.0), pce, np.array([-25.0])) == pytest.approx(
        x, rel=1e-12
    )


# ---------------------------------------------------------------------------
# sample_solution / solution property
# ---------------------------------------------------------------------------


def test_pce_perform_final_computation_sets_solution(mocker, make_pce, qubo_identity):
    pce = make_pce(problem=qubo_identity)

    pce._best_params = np.zeros(pce.n_layers * pce.n_params_per_layer)
    pce._results["best_probs"] = {"0_NoMitigation:0_0": {"01": 1.0}}
    mocker.patch.object(pce, "_run_solution_measurement_for")

    pce.sample_solution()

    np.testing.assert_array_equal(_read_solution(pce), np.array([0, 1]))


def test_pce_solution_requires_run(make_pce, qubo_identity):
    pce = make_pce(problem=qubo_identity)

    with pytest.raises(RuntimeError, match=r"not available yet.*sample_solution\(\)"):
        _ = pce.solution


def test_pce_solution_needs_a_measured_distribution(mocker, make_pce, qubo_identity):
    """A sample_solution that measures nothing leaves no solution to read."""
    pce = make_pce(problem=qubo_identity)
    pce._best_params = np.zeros(pce.n_layers * pce.n_params_per_layer)
    mocker.patch.object(pce, "_run_solution_measurement_for")

    pce.sample_solution()

    with pytest.raises(RuntimeError, match="not available yet"):
        pce.solution


# ---------------------------------------------------------------------------
# get_top_solutions
# ---------------------------------------------------------------------------


def test_pce_get_top_solutions_decodes_bitstrings(make_pce):
    """get_top_solutions decodes encoded qubit states to QUBO variable assignments."""
    # Create a 4-variable QUBO (requires 3 qubits: ceil(log2(4+1)) = 3)
    pce = make_pce(problem=np.eye(4))

    # For 4 variables, we need 3 qubits
    _set_probs(
        pce,
        {
            "000": 0.5,  # Encoded qubit state
            "001": 0.3,
            "010": 0.15,
            "011": 0.05,
        },
    )

    solutions = pce.get_top_solutions(n=4)

    assert len(solutions) == 4
    for sol in solutions:
        assert len(sol.bitstring) == pce.n_vars  # Decoded QUBO solution length
        assert all(c in "01" for c in sol.bitstring)  # Binary string
        assert 0.0 <= sol.prob <= 1.0
        assert sol.decoded is None  # include_decoded=False by default

    # Verify sorting by probability (descending)
    assert solutions[0].prob == 0.5
    assert solutions[1].prob == 0.3
    assert solutions[2].prob == 0.15
    assert solutions[3].prob == 0.05


def test_pce_get_top_solutions_include_decoded(make_pce):
    """get_top_solutions with include_decoded=True returns numpy arrays."""
    pce = make_pce(problem=np.eye(3))

    # 2 qubits for 3 variables
    _set_probs(pce, {"00": 0.6, "01": 0.4})

    solutions = pce.get_top_solutions(n=2, include_decoded=True)

    assert len(solutions) == 2
    for sol in solutions:
        assert len(sol.bitstring) == pce.n_vars  # Decoded length
        assert sol.decoded is not None
        assert isinstance(sol.decoded, np.ndarray)
        assert sol.decoded.shape == (pce.n_vars,)
        assert sol.decoded.dtype == np.int32
        # Verify decoded array matches bitstring
        expected_array = np.array([int(c) for c in sol.bitstring], dtype=np.int32)
        np.testing.assert_array_equal(sol.decoded, expected_array)


def test_pce_get_top_solutions_decoded_keys_by_variable_name(make_pce):
    """For a string-keyed BQM, decoded is a name-keyed dict, positionally
    consistent with the decoded bitstring (matching .solution)."""
    bqm = dimod.BinaryQuadraticModel(
        {"w": 1.0, "x": 1.0, "y": 1.0}, {}, 0.0, dimod.Vartype.BINARY
    )
    pce = make_pce(problem=bqm)  # 3 variables -> 2 qubits (dense)
    _set_probs(pce, {"00": 0.6, "01": 0.4})

    sol = pce.get_top_solutions(n=1, include_decoded=True)[0]
    assert isinstance(sol.decoded, dict)
    assert set(sol.decoded) == {"w", "x", "y"}
    assert sol.decoded == {
        name: int(bit)
        for name, bit in zip(
            pce.problem.canonical_problem.variable_order, sol.bitstring
        )
    }


def test_pce_get_top_solutions_min_prob_filtering(make_pce):
    """get_top_solutions filters by min_prob correctly."""
    pce = make_pce(problem=np.eye(2))

    _set_probs(pce, {"00": 0.5, "01": 0.3, "10": 0.15, "11": 0.05})

    solutions = pce.get_top_solutions(n=10, min_prob=0.2)

    # Only solutions with prob >= 0.2 should be included
    assert len(solutions) == 2
    assert solutions[0].prob == 0.5
    assert solutions[1].prob == 0.3
    assert all(sol.prob >= 0.2 for sol in solutions)


def test_pce_get_top_solutions_validation_errors(make_pce):
    """get_top_solutions raises for invalid n, min_prob, and missing probs."""
    pce = make_pce(problem=np.eye(2))
    _set_probs(pce, {"00": 0.5, "01": 0.5})

    with pytest.raises(ValueError, match="n must be non-negative"):
        pce.get_top_solutions(n=-1)

    with pytest.raises(ValueError, match="min_prob must be in range"):
        pce.get_top_solutions(n=2, min_prob=1.5)

    with pytest.raises(ValueError, match="min_prob must be in range"):
        pce.get_top_solutions(n=2, min_prob=-0.1)


@pytest.mark.parametrize("sort_by", ["prob", "energy"])
def test_pce_get_top_solutions_n_zero_returns_all(make_pce, sort_by):
    """n=0 returns every solution that passes min_prob, like QAOA."""
    pce = make_pce(problem=np.eye(2))
    _set_probs(pce, {"00": 0.5, "01": 0.3, "10": 0.15, "11": 0.05})

    assert len(pce.get_top_solutions(n=0, sort_by=sort_by)) == 4
    assert len(pce.get_top_solutions(n=0, min_prob=0.2, sort_by=sort_by)) == 2


def test_pce_get_top_solutions_defaults_to_ten(make_pce):
    pce = make_pce(problem=np.eye(9))  # 4 qubits, 16 encoded states
    _set_probs(pce, {format(i, "04b"): (i + 1) / 136 for i in range(16)})

    assert len(pce.get_top_solutions()) == 10


def test_pce_get_top_solutions_min_prob_one_can_select_nothing(make_pce):
    pce = make_pce(problem=np.eye(2))
    _set_probs(pce, {"00": 0.5, "01": 0.3, "10": 0.2})

    assert pce.get_top_solutions(min_prob=1.0) == []


def test_pce_filter_ranks_by_energy_without_decoding(make_pce):
    """Encoded 00/01/10 decode to x = 11/01/10 with energies 2/1/1."""
    pce = make_pce(problem=np.eye(2))
    _set_probs(pce, {"00": 0.5, "01": 0.3, "10": 0.2})

    solutions = pce.get_top_solutions(feasibility="filter")

    assert [(s.bitstring, s.decoded) for s in solutions] == [
        ("01", None),
        ("10", None),
        ("11", None),
    ]


class _OneHotProblem(BinaryOptimizationProblem):
    """Feasible only with exactly one variable set; repairs to ``100``."""

    def is_feasible(self, bitstring):
        return bitstring.count("1") == 1

    def repair_infeasible_bitstring(self, bitstring):
        return "100", "repaired", 0.0


def test_pce_repair_folds_infeasible_samples_into_the_repair(make_pce):
    """Encoded 00/01/10 decode to x = 111/010/101; the two infeasible ones
    repair to 100 and merge."""
    pce = make_pce(problem=_OneHotProblem(np.eye(3)))
    _set_probs(pce, {"00": 0.5, "01": 0.3, "10": 0.2})

    solutions = pce.get_top_solutions(feasibility="repair")

    assert [(s.bitstring, pytest.approx(s.prob)) for s in solutions] == [
        ("100", 0.7),
        ("010", 0.3),
    ]


@pytest.fixture
def quadratized_pce(make_pce):
    """A cubic HUBO on named variables, quadratized so it gains auxiliary qubits."""
    problem = BinaryOptimizationProblem(
        {("a",): -1.0, ("a", "b", "c"): 1.5, ("b",): 0.5},
        hamiltonian_builder="quadratized",
    )
    pce = make_pce(problem=problem)
    _set_probs(pce, {"00": 0.6, "01": 0.4})
    return pce


def test_pce_quadratized_problem_decodes_by_variable_name(quadratized_pce):
    solutions = quadratized_pce.get_top_solutions(include_decoded=True)

    assert [(s.bitstring, s.decoded) for s in solutions] == [
        ("111", {"a": 1, "b": 1, "c": 1}),
        ("010", {"a": 0, "b": 1, "c": 0}),
    ]


def test_pce_quadratized_problem_rejects_feasibility_ranking(quadratized_pce):
    with pytest.raises(
        ValueError,
        match=exact_match(
            "feasibility='filter' and 'repair' need a problem built with "
            "hamiltonian_builder='native'."
        ),
    ):
        quadratized_pce.get_top_solutions(feasibility="filter")


def test_pce_quadratized_integer_problem_decodes_to_an_array(make_pce):
    pce = make_pce(
        problem=BinaryOptimizationProblem(
            {(0,): -1.0, (0, 1, 2): 1.5}, hamiltonian_builder="quadratized"
        )
    )
    _set_probs(pce, {"00": 0.6, "01": 0.4})

    solutions = pce.get_top_solutions(include_decoded=True)

    assert all(isinstance(s.decoded, np.ndarray) for s in solutions)


def test_pce_cost_evaluation_forwards_its_budget_keywords(make_pce):
    pce = make_pce()

    with pytest.raises(
        ValueError,
        match=exact_match("shots and estimator_samples are mutually exclusive."),
    ):
        pce._evaluate_cost_param_sets(
            np.zeros((1, pce.n_params)), shots=5, estimator_samples=3
        )


def test_pce_get_top_solutions_no_probs_raises(make_pce):
    """get_top_solutions raises when no probability distribution available."""
    pce = make_pce(problem=np.eye(2))
    pce._results["best_probs"] = {}
    pce._losses_history = []

    with pytest.raises(RuntimeError, match="No sampled distribution yet"):
        pce.get_top_solutions(n=2)


@pytest.mark.parametrize(
    "encoding,qubo,encoded_probs,expected_decoded_map,n_vars",
    [
        # Dense: 2 vars, masks [1,2]. "00"->"11", "01"->"01", "10"->"10", "11"->"00"
        (
            DenseEncoding(),
            np.array([[1.0, 0.2], [0.2, 2.0]]),
            {"00": 0.4, "01": 0.3, "10": 0.2, "11": 0.1},
            {"11": 0.4, "01": 0.3, "10": 0.2, "00": 0.1},
            2,
        ),
        # Poly: 3 vars, masks [1,2,3]. "00"->"111", "01"->"010", "10"->"100", "11"->"001"
        (
            PolyEncoding(),
            np.zeros((3, 3)),
            {"00": 0.4, "01": 0.3, "10": 0.2, "11": 0.1},
            {"111": 0.4, "010": 0.3, "100": 0.2, "001": 0.1},
            3,
        ),
    ],
)
def test_pce_get_top_solutions_decoding_correctness(
    make_pce,
    encoding,
    qubo,
    encoded_probs,
    expected_decoded_map,
    n_vars,
):
    """Decoding produces correct QUBO variable assignments for both encodings."""
    pce = make_pce(problem=qubo, encoding=encoding)

    _set_probs(pce, encoded_probs)

    solutions = pce.get_top_solutions(n=len(encoded_probs))

    decoded_map = {sol.bitstring: sol.prob for sol in solutions}
    for decoded_bitstring, expected_prob in expected_decoded_map.items():
        assert decoded_map[decoded_bitstring] == expected_prob
    assert all(len(sol.bitstring) == n_vars for sol in solutions)


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_pce_qubo_e2e_solution(default_test_simulator, basic_ansatz):
    default_test_simulator.set_seed(1997)
    qubo = PCE_QUBO_MATRIX.copy()

    pce = PCE(
        problem=BinaryOptimizationProblem(qubo),
        ansatz=basic_ansatz,
        n_layers=1,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=5,
        backend=default_test_simulator,
        seed=1997,
    )

    pce.run()

    assert len(pce.losses_history) == 5
    assert isinstance(pce.best_loss, float)
    assert isinstance(pce.best_params, np.ndarray)
    solution = _read_solution(pce)
    assert solution.shape == (qubo.shape[0],)
    assert set(np.unique(solution)).issubset({0, 1})
    [best] = pce.get_top_solutions(n=1, sort_by="energy")
    assert best.bitstring == "".join(map(str, PCE_QUBO_SOLUTION))
    assert all(
        len(bitstring) == pce.n_qubits
        for probs_dict in pce.best_probs.values()
        for bitstring in probs_dict.keys()
    )


@pytest.mark.e2e
def test_pce_qubo_e2e_checkpointing_resume(
    default_test_simulator, basic_ansatz, tmp_path
):
    default_test_simulator.set_seed(1997)
    qubo = PCE_QUBO_MATRIX.copy()
    checkpoint_dir = tmp_path / "checkpoint_test"

    pce1 = PCE(
        problem=BinaryOptimizationProblem(qubo),
        ansatz=basic_ansatz,
        n_layers=1,
        optimizer=MonteCarloOptimizer(population_size=5, n_best_sets=2),
        max_iterations=2,
        backend=default_test_simulator,
        seed=1997,
    )
    pce1.run(checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir))
    assert pce1.current_iteration == 2

    checkpoint_path = checkpoint_dir / "checkpoint_002"
    assert checkpoint_path.exists()
    assert (checkpoint_path / "program_state.json").exists()

    first_run_iteration = pce1.current_iteration
    first_run_losses_count = len(pce1.losses_history)

    pce2 = PCE.load_state(
        checkpoint_dir,
        backend=default_test_simulator,
        problem=BinaryOptimizationProblem(qubo),
        ansatz=basic_ansatz,
        n_layers=1,
    )
    assert pce2.current_iteration == first_run_iteration
    assert len(pce2.losses_history) == first_run_losses_count
    np.testing.assert_array_equal(_read_solution(pce2), _read_solution(pce1))

    pce2.max_iterations = 4
    pce2.run(checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir))
    assert pce2.current_iteration == 4
    assert (checkpoint_dir / "checkpoint_004").exists()

    pce3 = PCE.load_state(
        checkpoint_dir,
        backend=default_test_simulator,
        problem=BinaryOptimizationProblem(qubo),
        ansatz=basic_ansatz,
        n_layers=1,
    )
    assert pce3.current_iteration == 4
    pce3.max_iterations = 5
    pce3.run()
    assert pce3.current_iteration == 5

    assert isinstance(pce3.best_loss, float)
    assert _read_solution(pce3).shape == (qubo.shape[0],)


@pytest.mark.parametrize(
    "changed",
    [
        {"problem": BinaryOptimizationProblem(PCE_QUBO_MATRIX * 2.0)},
        {"encoding": PolyEncoding()},
        {"objective": SoftObjective(alpha=5.0)},
    ],
    ids=["problem", "encoding", "objective"],
)
def test_pce_refuses_a_checkpoint_from_another_pce(
    make_maestro_simulator, basic_ansatz, tmp_path, changed
):
    """The fingerprint covers the problem's coefficients, the encoding and the
    objective, not only the qubit layout."""
    build = dict(
        problem=BinaryOptimizationProblem(PCE_QUBO_MATRIX),
        ansatz=basic_ansatz,
        n_layers=1,
    )
    PCE(
        **build,
        optimizer=MonteCarloOptimizer(population_size=3, n_best_sets=1),
        max_iterations=1,
        backend=make_maestro_simulator(),
        seed=1997,
    ).run(checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path))

    with pytest.raises(
        ValueError, match="The checkpoint was written for a different cost Hamiltonian"
    ):
        PCE.load_state(tmp_path, backend=make_maestro_simulator(), **(build | changed))


@pytest.mark.e2e
def test_pce_poly_e2e_solution(default_test_simulator, basic_ansatz):
    """E2E test for PCE with poly encoding (3 vars, 2 qubits)."""
    default_test_simulator.set_seed(1997)
    qubo = PCE_QUBO_MATRIX.copy()

    pce = PCE(
        problem=BinaryOptimizationProblem(qubo),
        ansatz=basic_ansatz,
        n_layers=1,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=5,
        backend=default_test_simulator,
        encoding=PolyEncoding(),
        seed=1997,
    )

    pce.run()

    assert len(pce.losses_history) == 5
    assert isinstance(pce.best_loss, float)
    assert isinstance(pce.best_params, np.ndarray)
    assert pce.n_qubits == 2  # N=3: min 2 qubits, capacity 3
    solution = _read_solution(pce)
    assert solution.shape == (qubo.shape[0],)
    assert set(np.unique(solution)).issubset({0, 1})
    assert all(
        len(bitstring) == pce.n_qubits
        for probs_dict in pce.best_probs.values()
        for bitstring in probs_dict.keys()
    )


@pytest.mark.e2e
def test_pce_hubo_e2e_solution(default_test_simulator, basic_ansatz):
    """PCE solves a small cubic HUBO and reaches exact minimum assignment."""
    default_test_simulator.set_seed(1997)
    _, exact_minima = exact_hubo_minima(HUBO_CUBIC, n_vars=3)

    pce = PCE(
        problem=BinaryOptimizationProblem(HUBO_CUBIC),
        ansatz=basic_ansatz,
        n_layers=1,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=10,
        backend=default_test_simulator,
        seed=1997,
    )

    pce.run()
    assert any(np.array_equal(_read_solution(pce), x) for x in exact_minima)


@pytest.mark.e2e
def test_pce_named_variables_decode_to_names(default_test_simulator, basic_ansatz):
    """A string-indexed BQM decodes to name-keyed solutions on both surfaces."""
    default_test_simulator.set_seed(1997)
    bqm = dimod.BinaryQuadraticModel(
        {"w": 10, "x": -3, "y": 2},
        {("w", "x"): -1, ("x", "y"): 1},
        0.0,
        dimod.Vartype.BINARY,
    )
    pce = PCE(
        problem=BinaryOptimizationProblem(bqm),
        ansatz=basic_ansatz,
        n_layers=1,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=3,
        backend=default_test_simulator,
        seed=1997,
    )
    pce.run()

    names = {"w", "x", "y"}
    solution = _read_solution(pce)
    assert isinstance(solution, dict) and set(solution) == names

    top = pce.get_top_solutions(n=1, include_decoded=True)[0]
    assert isinstance(top.decoded, dict) and set(top.decoded) == names


# ---------------------------------------------------------------------------
# Wide registers (> 64 qubits): limb-based masks and decoding
# ---------------------------------------------------------------------------


def _big_endian_bitstring(set_bits, n_qubits):
    """Build an ``n_qubits``-wide big-endian bitstring with the given bits set
    (bit i is the LSB, i.e. char position ``n_qubits - 1 - i``)."""
    chars = ["0"] * n_qubits
    for b in set_bits:
        chars[n_qubits - 1 - b] = "1"
    return "".join(chars)


def _bits_to_int(bits):
    val = 0
    for b in bits:
        val |= 1 << b
    return val


def _limbs_to_int(limbs):
    """The integer a row of little-endian 64-bit limbs packs."""
    return sum(int(limb) << (64 * j) for j, limb in enumerate(limbs))


class TestWideRegisterMasks:
    """PCE mask/decode machinery must work past the 64-qubit uint64 boundary."""

    def test_pack_masks_1d_below_boundary(self):
        packed = _pack_masks([1, 2, 3], n_qubits=4)
        assert packed.ndim == 1
        assert packed.dtype == np.uint64
        assert list(packed) == [1, 2, 3]

    def test_pack_masks_2d_above_boundary(self):
        # bit 70 lives in limb 1 (bits 64-127).
        packed = _pack_masks([1 << 3, 1 << 70, (1 << 5) | (1 << 70)], n_qubits=72)
        assert packed.shape == (3, 2)
        assert [_limbs_to_int(row) for row in packed] == [
            1 << 3,
            1 << 70,
            (1 << 5) | (1 << 70),
        ]

    def test_decode_parities_across_limb_boundary(self):
        nq = 72
        mask_ints = [1 << 3, 1 << 70, (1 << 5) | (1 << 70)]
        packed = _pack_masks(mask_ints, nq)
        sbits = [[], [3], [70], [3, 70], [5, 70]]
        states = [_big_endian_bitstring(b, nq) for b in sbits]

        parities = _decode_parities(states, packed)

        assert parities.dtype == np.uint8
        expected = np.array(
            [
                [bin((m & _bits_to_int(sb))).count("1") % 2 for sb in sbits]
                for m in mask_ints
            ],
            dtype=np.uint8,
        )
        assert np.array_equal(parities, expected)

    def test_limb_decode_matches_1d_below_boundary(self):
        """Forcing the limb representation on a <=64-bit mask set gives the
        same parities as the native 1-D path."""
        mask_ints = [1 << 3, (1 << 5) | (1 << 7)]
        nq = 8
        states = [_big_endian_bitstring(b, nq) for b in ([3], [5, 7], [3, 5])]

        native = _decode_parities(states, np.array(mask_ints, dtype=np.uint64))
        forced_limb = _decode_parities(states, _pack_masks(mask_ints, n_qubits=65))
        assert np.array_equal(native, forced_limb)

    @pytest.mark.parametrize("n_qubits", [8, 72])
    def test_decode_parities_rejects_ragged_states(self, n_qubits):
        """Mixed-width states must raise, not be regrouped by the reshape.

        Widths summing to the expected total would otherwise pack rows that
        straddle two states.
        """
        packed = _pack_masks([1 << 3, 1 << 5], n_qubits)
        good = _big_endian_bitstring([3], n_qubits)
        states = [good, good[:-1], good + "0"]
        with pytest.raises(ValueError, match="got one of width"):
            _decode_parities(states, packed)

    def test_decode_parities_fills_a_single_limb(self):
        """A 64-bit state exactly fills one limb; its top bit is qubit 63."""
        packed = _pack_masks([1 << 63], n_qubits=64)

        parities = _decode_parities(["1" + "0" * 63], packed)

        assert parities.tolist() == [[1]]

    def test_decode_parities_rejects_states_wider_than_the_masks(self):
        """States wider than the masks' limbs cannot be decoded against them."""
        packed = _pack_masks([1 << 3], n_qubits=64)
        with pytest.raises(ValueError, match="do not fit"):
            _decode_parities([_big_endian_bitstring([3], 65)], packed)

    def test_decode_parities_rejects_non_binary_states(self):
        """Packed decoding must not reinterpret non-binary characters as bits."""
        packed = _pack_masks([1 << 3], n_qubits=72)
        with pytest.raises(ValueError, match="non-binary"):
            _decode_parities(["2" + "0" * 71], packed)

    def test_decode_parities_empty_wide_states_preserve_variable_axis(self):
        """No states still returns one empty row per wide variable mask."""
        packed = _pack_masks([1 << 3, 1 << 65], n_qubits=72)

        result = _decode_parities([], packed)

        assert result.shape == (2, 0)
        assert result.dtype == np.uint8


def test_pce_poly_encoding_supports_more_than_64_qubits(make_pce):
    """A poly-encoded problem forced past 64 qubits builds 2-D limb masks and
    measures a variable on qubit 64 — no silent uint64 overflow."""
    qubo = np.diag(np.ones(65))
    with pytest.warns(UserWarning, match=_EXTRA_QUBITS_WARNING):
        pce = make_pce(problem=qubo, encoding=PolyEncoding(n_qubits=65))

    masks = pce.encoding._masks(pce.n_vars, pce.n_qubits)
    assert pce.n_qubits == 65
    assert masks.ndim == 2
    assert masks.shape[1] == 2  # two 64-bit limbs
    # Variable 64 reads qubit 64 (bit 64 → limb 1).
    assert _limbs_to_int(masks[64]) == 1 << 64
    assert pce.cost_circuit.observable[64].paulis[0].z[0]


def _seventy_qubit_pce(make_pce, **kwargs):
    """70 poly-encoded variables, one per qubit: variable i reads qubit 69 - i."""
    with pytest.warns(UserWarning, match=_EXTRA_QUBITS_WARNING):
        return make_pce(
            problem=np.diag(np.arange(1.0, 71.0)),
            encoding=PolyEncoding(n_qubits=70),
            **kwargs,
        )


def test_pce_decodes_a_state_wider_than_64_qubits(make_pce):
    """Qubit 0 set flips only variable 69; every other parity is even."""
    pce = _seventy_qubit_pce(make_pce)
    _set_probs(pce, {"1" + "0" * 69: 1.0})

    [entry] = pce.get_top_solutions(n=1, sort_by="energy")

    assert entry.bitstring == "1" * 69 + "0"
    assert entry.energy == pytest.approx(sum(range(1, 70)))


def test_cvar_scores_a_histogram_wider_than_64_qubits(make_pce):
    """Half the shots flip variable 69 (energy 2415), half flip variable 0
    (energy 2484); CVaR(1.0) averages both."""
    pce = _seventy_qubit_pce(make_pce, objective=CVaRObjective(fraction=1.0))
    histogram = {"1" + "0" * 69: 5, "0" * 69 + "1": 5}

    cost = pce._score_histogram(pce.objective, histogram)

    total = sum(range(1, 71))
    assert cost == pytest.approx(((total - 70) + (total - 1)) / 2)


# ---------------------------------------------------------------------------
# get_top_solutions sort_by
# ---------------------------------------------------------------------------


class TestGetTopSolutionsSortBy:
    """Tests for the sort_by parameter of PCE.get_top_solutions."""

    @pytest.fixture
    def pce_with_probs(self, make_pce):
        """PCE instance with a 2-var diagonal QUBO and pre-set probability distribution.

        qubo = diag([1, 2]), masks = [1, 2], 2 qubits.
        Encoded states and their decoded solutions + energies:
          "00" (int 0) → parities [0, 0] → x = [1, 1] → energy = 1*1 + 2*1 = 3
          "01" (int 1) → parities [1, 0] → x = [0, 1] → energy = 0 + 2*1 = 2
          "10" (int 2) → parities [0, 1] → x = [1, 0] → energy = 1*1 + 0 = 1
          "11" (int 3) → parities [1, 1] → x = [0, 0] → energy = 0
        """
        pce = make_pce(problem=np.diag([1.0, 2.0]))
        _set_probs(
            pce,
            {
                "00": 0.4,  # decoded "11", energy 3
                "01": 0.1,  # decoded "01", energy 2
                "10": 0.2,  # decoded "10", energy 1
                "11": 0.3,  # decoded "00", energy 0
            },
        )
        return pce

    def test_default_sort_by_prob(self, pce_with_probs):
        """Default sort_by='prob' sorts descending by probability."""
        solutions = pce_with_probs.get_top_solutions(n=4)
        probs = [s.prob for s in solutions]
        assert probs == [0.4, 0.3, 0.2, 0.1]

    def test_sort_by_energy_ascending(self, pce_with_probs):
        """sort_by='energy' sorts ascending by energy."""
        solutions = pce_with_probs.get_top_solutions(n=4, sort_by="energy")

        # Verify energy ordering: 0, 1, 2, 3
        energies = []
        for sol in solutions:
            x = np.array([int(c) for c in sol.bitstring], dtype=float)
            energy = _evaluate_binary_polynomial(
                x, pce_with_probs.problem.canonical_problem
            )
            energies.append(energy)
        assert energies == pytest.approx([0.0, 1.0, 2.0, 3.0])

    def test_sort_by_energy_includes_energy_field(self, pce_with_probs):
        """sort_by='energy' populates the energy field in SolutionEntry."""
        solutions = pce_with_probs.get_top_solutions(n=4, sort_by="energy")

        assert solutions[0].energy == pytest.approx(0.0)
        assert solutions[1].energy == pytest.approx(1.0)
        assert solutions[2].energy == pytest.approx(2.0)
        assert solutions[3].energy == pytest.approx(3.0)

    def test_sort_by_prob_energy_is_none(self, pce_with_probs):
        """sort_by='prob' leaves the energy field as None."""
        solutions = pce_with_probs.get_top_solutions(n=4)
        for sol in solutions:
            assert sol.energy is None

    def test_sort_by_energy_with_min_prob(self, pce_with_probs):
        """sort_by='energy' respects min_prob filtering."""
        solutions = pce_with_probs.get_top_solutions(
            n=4, sort_by="energy", min_prob=0.15
        )
        # Only probs >= 0.15: 0.4, 0.3, 0.2 → sorted by energy
        assert len(solutions) == 3
        assert all(sol.prob >= 0.15 for sol in solutions)
        # Energy should still be ascending
        assert solutions[0].energy < solutions[1].energy < solutions[2].energy

    def test_sort_by_energy_with_n_limit(self, pce_with_probs):
        """sort_by='energy' with n < total returns only the n lowest-energy solutions."""
        solutions = pce_with_probs.get_top_solutions(n=2, sort_by="energy")
        assert len(solutions) == 2
        assert solutions[0].energy == pytest.approx(0.0)
        assert solutions[1].energy == pytest.approx(1.0)

    def test_sort_by_invalid_raises(self, pce_with_probs):
        """Invalid sort_by value raises ValueError."""
        with pytest.raises(ValueError, match="sort_by must be"):
            pce_with_probs.get_top_solutions(n=4, sort_by="invalid")


# ---------------------------------------------------------------------------
# ObservableMeasuring contracts
# ---------------------------------------------------------------------------


class TestObservableMeasuringContracts(ObservableMeasuringContractsBase):
    @pytest.fixture
    def make_program(self, make_pce):
        def _make(**kwargs):
            kwargs.setdefault("n_layers", 1)
            return make_pce(**kwargs)

        return _make


# ---------------------------------------------------------------------------
# Multi-basis encoding and the chain-rule gradient (exact Maestro expectations)
# ---------------------------------------------------------------------------


def _entangling_pce(backend, optimizer, **kwargs):
    """``_QUBO_3`` on a two-layer RY/RZ ansatz with CX entanglers."""
    kwargs.setdefault("problem", BinaryOptimizationProblem(_QUBO_3))
    return PCE(
        ansatz=GenericLayerAnsatz([RYGate, RZGate], entangler=CXGate),
        n_layers=2,
        optimizer=optimizer,
        backend=backend,
        **kwargs,
    )


def _bloch_pce(backend, optimizer, objective=None):
    """Three variables on one qubit, read as <X>, <Y>, <Z> of RY(theta)|0>."""
    return PCE(
        problem=BinaryOptimizationProblem(_QUBO_3),
        encoding=MultiBasisEncoding(order=1),
        objective=objective,
        ansatz=GenericLayerAnsatz([RYGate]),
        n_layers=1,
        optimizer=optimizer,
        backend=backend,
    )


def test_multi_basis_cost_is_the_soft_objective_of_the_bloch_vector(
    make_maestro_simulator, default_optimizer
):
    """RY(theta)|0> has <X> = sin(theta), <Y> = 0, <Z> = cos(theta)."""
    objective = SoftObjective(alpha=1.3)
    pce = _bloch_pce(make_maestro_simulator(), default_optimizer, objective)
    theta = 0.7

    cost = pce._evaluate_cost_param_sets(np.array([[theta]]))[0]

    z = np.array([np.sin(theta), 0.0, np.cos(theta)])
    assert cost == pytest.approx(_score(objective, pce, z))


def test_multi_basis_solution_is_the_sign_of_each_expectation(
    make_maestro_simulator, default_optimizer
):
    """At theta = 2: <X> = 0.91 > 0, <Y> = 0, <Z> = -0.42, so x = 100."""
    pce = _bloch_pce(make_maestro_simulator(), default_optimizer)

    pce.sample_solution(np.array([2.0]))

    assert pce.best_probs == {0: {"100": 1.0}}
    np.testing.assert_array_equal(pce.solution, np.array([1, 0, 0]))
    [entry] = pce.get_top_solutions(sort_by="energy")
    assert (entry.bitstring, entry.prob) == ("100", 1.0)
    assert entry.energy == pytest.approx(1.0)


@pytest.mark.parametrize(
    "encoding", [DenseEncoding(), MultiBasisEncoding()], ids=["dense", "multi-basis"]
)
def test_pce_gradient_matches_central_differences(
    make_maestro_simulator, default_optimizer, encoding
):
    """The Jacobian chained through the objective is the exact gradient of the
    non-linear loss."""
    pce = _entangling_pce(
        make_maestro_simulator(),
        default_optimizer,
        problem=BinaryOptimizationProblem(
            np.array(
                [
                    [1.0, -2.0, 0.5, 0.0],
                    [0, 3.0, 1.5, -1.0],
                    [0, 0, -1.0, 2.0],
                    [0, 0, 0, 0.5],
                ]
            )
        ),
        encoding=encoding,
    )
    params = np.random.default_rng(5).uniform(0, 2 * np.pi, pce.n_params)

    np.testing.assert_allclose(
        pce._evaluate_gradient_at(params),
        central_difference_of_loss(pce, params, step=1e-5),
        atol=1e-6,
    )


def test_pce_slopes_reuse_the_latest_cost_evaluation(
    make_maestro_simulator, default_optimizer, mocker
):
    """QNG evaluates the cost at theta just before the gradient, so the slopes
    need no new measurement; a cold call measures them and agrees."""
    pce = _entangling_pce(make_maestro_simulator(), default_optimizer)
    theta = np.random.default_rng(4).uniform(0, 2 * np.pi, pce.n_params)
    cold = pce._loss_gradient_by_observable(theta)

    pce._evaluate_cost_param_sets(np.atleast_2d(theta))
    evaluate = mocker.spy(pce, "evaluate")
    warm = pce._loss_gradient_by_observable(theta)

    assert evaluate.call_count == 0
    np.testing.assert_allclose(warm, cold)


def test_pce_slopes_at_other_parameters_are_measured(
    make_maestro_simulator, default_optimizer, mocker
):
    pce = _entangling_pce(make_maestro_simulator(), default_optimizer)
    rng = np.random.default_rng(4)
    evaluated, other = rng.uniform(0, 2 * np.pi, (2, pce.n_params))
    expected = _entangling_pce(
        make_maestro_simulator(), default_optimizer
    )._loss_gradient_by_observable(other)

    pce._evaluate_cost_param_sets(np.atleast_2d(evaluated))
    evaluate = mocker.spy(pce, "evaluate")
    slopes = pce._loss_gradient_by_observable(other)

    assert evaluate.call_count == 1
    np.testing.assert_allclose(slopes, expected)


def test_pce_slopes_reuse_every_parameter_set_of_a_batched_evaluation(
    make_maestro_simulator, default_optimizer, mocker
):
    """Every row of a batched cost evaluation is reused, whatever the sign of
    zero in the parameters."""
    pce = _entangling_pce(make_maestro_simulator(), default_optimizer)
    batch = np.random.default_rng(4).uniform(0, 2 * np.pi, (2, pce.n_params))
    batch[1, 0] = 0.0

    negative_zero = batch[1].copy()
    negative_zero[0] = -0.0

    pce._evaluate_cost_param_sets(batch)
    evaluate = mocker.spy(pce, "evaluate")
    pce._loss_gradient_by_observable(batch[0])
    pce._loss_gradient_by_observable(negative_zero)

    assert evaluate.call_count == 0


@pytest.mark.parametrize(
    "encoding", [DenseEncoding(), PolyEncoding()], ids=["dense", "poly"]
)
@pytest.mark.parametrize("flipped", [(0,), (2,), (0, 1)], ids=["q0", "q2", "q0-q1"])
def test_parity_expectations_agree_with_decoded_samples(
    make_maestro_simulator, default_optimizer, encoding, flipped
):
    """A basis state measures each variable's Z-parity as exactly +-1, and the
    cost path must read the same signs the sampled bitstring decodes to."""
    pce = PCE(
        problem=BinaryOptimizationProblem(np.eye(6)),
        encoding=encoding,
        ansatz=GenericLayerAnsatz([RYGate]),
        n_layers=1,
        optimizer=default_optimizer,
        backend=make_maestro_simulator(),
    )
    params = np.array([np.pi if q in flipped else 0.0 for q in range(pce.n_qubits)])

    z = pce._variable_expectations(params)[0]
    pce.sample_solution(params)
    [state] = pce.best_probs[0]

    assignment = pce._decode_states([state])[0].astype(int)
    np.testing.assert_allclose(z, 2 * assignment - 1, atol=1e-9)


@pytest.mark.parametrize(
    "encoding", [DenseEncoding(), PolyEncoding()], ids=["dense", "poly"]
)
@pytest.mark.parametrize("flipped", [0, 1, 2], ids=["q0", "q1", "q2"])
def test_cvar_cost_scores_the_assignment_the_solution_reports(
    make_maestro_simulator, default_optimizer, encoding, flipped
):
    """Raw counts and sampled probabilities both put qubit 0 leftmost, so on a
    basis state the CVaR cost is the energy of the reported solution."""
    qubo = np.diag([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    pce = PCE(
        problem=BinaryOptimizationProblem(qubo),
        encoding=encoding,
        objective=CVaRObjective(fraction=1.0),
        ansatz=GenericLayerAnsatz([RYGate]),
        n_layers=1,
        optimizer=default_optimizer,
        backend=make_maestro_simulator(force_sampling=True),
    )
    params = np.array([np.pi if q == flipped else 0.0 for q in range(pce.n_qubits)])

    [counts_key] = pce.evaluate(np.atleast_2d(params), pce.cost_preprocessor())[0]
    cost = pce._evaluate_cost_param_sets(np.atleast_2d(params))[0]
    pce.sample_solution(params)
    [probs_key] = pce.best_probs[0]
    x = np.array([int(bit) for bit in pce.get_top_solutions(n=1)[0].bitstring])

    assert counts_key == probs_key
    assert counts_key[flipped] == "1"
    assert cost == pytest.approx(float(x @ qubo @ x))


def test_multi_basis_cost_on_a_sampling_backend_matches_the_exact_cost(
    make_maestro_simulator, default_optimizer
):
    """Basis-rotated sampled estimates of the X, Y and Z correlators give the
    same soft loss as exact expectation values, within shot noise."""

    def pce_on(backend):
        return PCE(
            problem=BinaryOptimizationProblem(_QUBO_3),
            encoding=MultiBasisEncoding(order=1),
            ansatz=GenericLayerAnsatz([RYGate, RZGate]),
            n_layers=2,
            optimizer=default_optimizer,
            backend=backend,
        )

    exact = pce_on(make_maestro_simulator())
    sampled = pce_on(
        make_maestro_simulator(
            shots=200_000, force_sampling=True, maestro_config=MaestroConfig(seed=7)
        )
    )
    params = np.random.default_rng(7).uniform(0, 2 * np.pi, (1, exact.n_params))

    assert sampled._evaluate_cost_param_sets(params)[0] == pytest.approx(
        exact._evaluate_cost_param_sets(params)[0], abs=0.03
    )


@pytest.mark.parametrize(
    "encoding", [DenseEncoding(), MultiBasisEncoding()], ids=["dense", "multi-basis"]
)
@pytest.mark.parametrize(
    "optimizer",
    [
        ScipyOptimizer(method=ScipyMethod.L_BFGS_B),
        QNGOptimizer(step_size=0.2, max_step_norm=0.5),
    ],
    ids=["l-bfgs-b", "qng"],
)
def test_gradient_optimizers_lower_the_pce_loss(
    make_maestro_simulator, encoding, optimizer
):
    pce = _entangling_pce(
        make_maestro_simulator(),
        optimizer,
        encoding=encoding,
        max_iterations=8,
        seed=11,
    )
    pce.run(perform_final_computation=False)

    first_loss = min(pce.losses_history[0].values())
    assert pce.best_loss < first_loss


@pytest.mark.e2e
def test_multi_basis_pce_resumes_from_a_checkpoint(
    make_maestro_simulator, basic_ansatz, tmp_path
):
    """The encoding is a constructor argument, so a resumed program rebuilt
    with the same one restores the same sign-decoded solution."""
    qubo = PCE_QUBO_MATRIX.copy()

    def build(**kwargs):
        return dict(
            problem=BinaryOptimizationProblem(qubo),
            encoding=MultiBasisEncoding(),
            ansatz=basic_ansatz,
            n_layers=1,
            **kwargs,
        )

    first = PCE(
        **build(),
        optimizer=MonteCarloOptimizer(population_size=5, n_best_sets=2),
        max_iterations=2,
        backend=make_maestro_simulator(),
        seed=1997,
    )
    first.run(checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path))

    resumed = PCE.load_state(tmp_path, backend=make_maestro_simulator(), **build())

    assert resumed.current_iteration == 2
    assert resumed.best_probs == first.best_probs
    np.testing.assert_array_equal(resumed.solution, first.solution)


def test_pce_sample_objective_has_no_gradient(make_pce):
    pce = make_pce(objective=CVaRObjective())
    with pytest.raises(
        ValueError,
        match=exact_match(
            "CVaRObjective scores sampled bitstrings and has no gradient; use a "
            "gradient-free optimizer."
        ),
    ):
        pce._evaluate_gradient_at(np.zeros(pce.n_params))


def test_pce_rejects_shot_variance_requests(make_pce):
    pce = make_pce()
    with pytest.raises(
        ValueError,
        match=exact_match(
            "PCE does not provide shot-noise variances of its loss, so "
            "shot-adaptive optimizers (RosalinOptimizer, QUIVEROptimizer with "
            "adaptive shots) cannot drive it."
        ),
    ):
        pce._evaluate_cost_param_sets(
            np.zeros((1, pce.n_params)), collect_variance=True
        )


@pytest.mark.parametrize(
    "encoding, circuits_per_set",
    [(DenseEncoding(), 1), (MultiBasisEncoding(order=1), 3)],
    ids=["dense", "multi-basis"],
)
def test_cost_spends_one_circuit_per_measured_basis_per_param_set(
    make_pce, encoding, circuits_per_set
):
    """Dense Z-parities share one basis, so each parameter set costs one
    circuit; order-1 X, Y and Z correlators on one qubit need one each. The
    run spends what the dry run predicts, plus the final readout."""
    pce = make_pce(problem=np.eye(3), encoding=encoding, max_iterations=3)
    per_evaluation = pce.dry_run()["cost"].total_circuits

    pce.run()

    n_param_sets = pce.optimizer.n_param_sets
    assert per_evaluation == circuits_per_set * n_param_sets
    final_readout = 1 if isinstance(encoding, DenseEncoding) else circuits_per_set
    assert pce.total_circuit_count == 3 * per_evaluation + final_readout
