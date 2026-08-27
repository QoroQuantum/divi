# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""QuantumProgram.dry_run: what a program previews, and that it matches what a run submits."""

import warnings

import networkx as nx
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RYGate, RZGate
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit
from divi.circuits.quepp import QuEPP
from divi.pipeline import (
    CircuitPipeline,
    CircuitPreprocessor,
    PipelineCadence,
)
from divi.pipeline.abc import (
    ContractViolation,
    PipelineEnv,
)
from divi.pipeline.stages import (
    MeasurementStage,
    ParameterBindingStage,
    QEMStage,
)
from divi.qprog import QAOA, VQE, QuantumProgram
from divi.qprog._metrics import (
    METRIC_ROUTINE,
    FubiniStudyMetricEstimator,
    MetricEstimator,
    PullbackMetricEstimator,
    StochasticFidelityMetricEstimator,
)
from divi.qprog.algorithms import PCE, GenericLayerAnsatz, TimeEvolution
from divi.qprog.optimizers import (
    MonteCarloOptimizer,
    QNGOptimizer,
    QNSPSAOptimizer,
    ScipyMethod,
    ScipyOptimizer,
    SPSAOptimizer,
)
from divi.qprog.problems import BinaryOptimizationProblem, MaxCutProblem
from tests.pipeline._helpers import (
    DummySpecStage,
    FanoutAndSumStage,
    assert_report_dicts_match,
    dry_run_stages,
    h2_vqe,
    metric_compatible_vqe,
    parametric_twirlable_meta,
    two_group_meta,
)


def _h2_vqe_for_totals(backend, optimizer) -> VQE:
    """A circuit-seeded program for the submitted-circuit oracle."""
    return h2_vqe(backend, optimizer, max_iterations=1)


def _maxcut_qaoa_for_totals(backend, optimizer) -> QAOA:
    """A Hamiltonian-seeded program for the submitted-circuit oracle: its seed is a
    ``SparsePauliOp``, which declares no parameters of its own."""
    return QAOA(
        MaxCutProblem(nx.random_regular_graph(3, 6, seed=1)),
        n_layers=1,
        backend=backend,
        optimizer=optimizer,
        max_iterations=1,
    )


class _ParametricProgram(QuantumProgram):
    """A direct ``QuantumProgram`` subclass with a parametric seed — the shape the
    custom-program docs describe, and the one a VQA-only test never exercises
    (``VariationalQuantumAlgorithm`` overrides the env construction involved)."""

    def __init__(self, backend):
        super().__init__(backend=backend)
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        self._meta = MetaCircuit(
            circuit_bodies=(((), circuit_to_dag(qc)),),
            observable=SparsePauliOp("Z"),
            parameters=(theta,),
        )

    def has_results(self) -> bool:
        return False

    def _initial_spec(self):
        return self._meta

    def _preprocessors(self):
        return (CircuitPreprocessor("cost", terminal_stage=MeasurementStage()),)

    def _assemble_pipeline(
        self, spec_stage, terminal_stage, *, result_format, extra_stages=()
    ):
        return CircuitPipeline(
            stages=[
                spec_stage,
                *extra_stages,
                *self._mitigation_stages(result_format),
                terminal_stage,
                ParameterBindingStage(),
            ]
        )

    def run(self):
        pass


@pytest.mark.parametrize("force", [False, True])
def test_direct_subclass_previews_its_real_parameter_width(dummy_simulator, force):
    """A direct subclass must get a parameter vector of its seed's width in both
    modes: the analytic path otherwise reports ``n_params: 0`` for a parametric
    circuit, and the forced path fails outright."""
    program = _ParametricProgram(dummy_simulator)
    report = program.dry_run(force_circuit_generation=force)["cost"]
    binding = next(s for s in report.stages if s.name == "ParameterBindingStage")
    assert binding.metadata["n_bound_params"] == 1


def test_routines_sharing_a_name_are_all_reported(dummy_simulator, mocker):
    """The routine name is the report key, so a second routine claiming it would
    replace the first — reporting fewer pipelines than the program runs."""
    program = _ParametricProgram(dummy_simulator)
    duplicate = CircuitPreprocessor("cost", terminal_stage=MeasurementStage())
    mocker.patch.object(
        program, "_preprocessors", return_value=(duplicate, duplicate, duplicate)
    )

    with pytest.warns(UserWarning, match="are both named 'cost'"):
        reports = program.dry_run()

    assert set(reports) == {"cost", "cost#2", "cost#3"}


@pytest.mark.usefixtures("suppress_quepp_warnings")
@pytest.mark.filterwarnings("ignore::UserWarning:divi.qprog.algorithms._vqe")
@pytest.mark.filterwarnings("ignore:shot_distribution is set but backend")
class TestQuantumProgramDryRun:
    """``QuantumProgram.dry_run`` analytic default + ``force_circuit_generation`` escape hatch."""

    @pytest.fixture
    def time_evolution_program(self, default_test_simulator):
        return TimeEvolution(
            hamiltonian=SparsePauliOp.from_list([("X", 1.0), ("Z", 1.0)]),
            observable=SparsePauliOp("Z"),
            backend=default_test_simulator,
            qem_protocol=QuEPP(truncation_order=1, n_twirls=3),
        )

    def test_default_and_forced_match(self, time_evolution_program):
        """``dry_run()`` (analytic) and ``dry_run(force_circuit_generation=True)``
        must produce the same total circuit count and per-stage fan-out."""
        assert_report_dicts_match(
            time_evolution_program.dry_run(),
            time_evolution_program.dry_run(force_circuit_generation=True),
        )

    def test_env_artifacts_exposed_via_dry_run(
        self, sampling_test_simulator, default_optimizer
    ):
        """Callers can read ``per_group_shots`` (and any other stage artifact)
        straight off the :class:`DryRunReport` — no private hooks needed."""
        vqe = h2_vqe(
            sampling_test_simulator,
            default_optimizer,
            grouping_strategy="qwc",
            shot_distribution="weighted",
        )
        cost_report = vqe.dry_run()["cost"]
        assert "per_group_shots" in cost_report.env_artifacts
        # The allocation is a dict {spec_key: {group_idx: shots}}; at least
        # one spec, at least one group with a non-zero shot budget.
        spec_alloc = next(iter(cost_report.env_artifacts["per_group_shots"].values()))
        assert any(shots > 0 for shots in spec_alloc.values())

    def test_qng_surfaces_metric_terms_pipeline(
        self, default_test_simulator, default_optimizer
    ):
        vqe = h2_vqe(default_test_simulator, QNGOptimizer())
        assert METRIC_ROUTINE in vqe.dry_run()

    def test_plain_optimizer_has_no_metric_pipeline(
        self, default_test_simulator, default_optimizer
    ):
        reports = h2_vqe(default_test_simulator, default_optimizer).dry_run()
        assert set(reports) == {"cost", "sample"}

    def test_an_incompatible_optimizer_refuses_the_preview(
        self, default_test_simulator, default_optimizer, mocker
    ):
        """A preview of a pairing ``run()`` will reject is worth nothing: the report
        would omit exactly the routines that make it invalid, so it costs a program
        that cannot run. The refusal comes from the same check ``run()`` makes."""
        vqe = h2_vqe(default_test_simulator, QNSPSAOptimizer())
        mocker.patch.object(
            vqe.optimizer,
            "preprocessors",
            side_effect=ContractViolation("nope"),
        )
        with pytest.raises(ContractViolation, match="nope"):
            vqe.dry_run()

    def test_a_real_incompatible_pairing_refuses_with_the_estimators_reason(
        self, dummy_simulator, default_optimizer
    ):
        """PCE's cost is a classical (COUNTS) objective, which the pullback metric
        rejects. The estimator's own message names the alternative, so previewing
        surfaces it before any circuit is built."""
        pce = PCE(
            problem=BinaryOptimizationProblem(np.array([[1.0, 0.2], [0.2, 2.0]])),
            ansatz=GenericLayerAnsatz([RYGate, RZGate]),
            optimizer=QNGOptimizer(),  # default pullback metric
            backend=dummy_simulator,
        )
        with pytest.raises(ContractViolation, match="Fubini–Study estimator"):
            pce.dry_run()

    def test_fubini_study_surfaces_one_pipeline_per_block(
        self, default_test_simulator, default_optimizer
    ):
        """A block-diagonal Fubini-Study metric drives one prefix pipeline per
        commuting-gate block; each must appear with a unique report key (no
        same-named pipeline silently overwriting another)."""
        vqe = metric_compatible_vqe(
            default_test_simulator,
            QNGOptimizer(metric_estimator=FubiniStudyMetricEstimator()),
            n_layers=2,
        )
        reports = vqe.dry_run()
        prefix_keys = [k for k in reports if k.startswith(f"{METRIC_ROUTINE}[block")]
        n_blocks = len(vqe.optimizer.preprocessors(vqe))
        assert len(prefix_keys) == n_blocks > 1  # one per block, none dropped
        assert len(set(prefix_keys)) == len(prefix_keys)  # keys are unique

    def test_qnspsa_surfaces_overlap_pipeline(
        self, default_test_simulator, default_optimizer
    ):
        """A natural-gradient optimizer's metric/overlap pipeline is no longer
        invisible: QN-SPSA's overlap circuits appear alongside cost/sample."""
        vqe = h2_vqe(default_test_simulator, QNSPSAOptimizer())
        reports = vqe.dry_run()
        assert METRIC_ROUTINE in reports
        assert reports[METRIC_ROUTINE].cadence is PipelineCadence.PER_EVALUATION
        assert reports[METRIC_ROUTINE].total_circuits > 0

    def test_stage_report_exposes_class_label_and_axis_separately(
        self, default_test_simulator, default_optimizer
    ):
        """Three identifiers, each needed: the class (composition assertions), the
        constructor ``name=`` (telling two instances of one class apart), and the
        axis (what the rendered tree shows in brackets)."""
        stages = (
            h2_vqe(default_test_simulator, default_optimizer).dry_run()["cost"].stages
        )
        measurement = next(s for s in stages if s.name == "MeasurementStage")
        assert measurement.axis == "obs_group"
        assert measurement.label == "MeasurementStage"

    def test_label_and_name_differ_for_a_renamed_stage(self, dummy_pipeline_env):
        """The two coincide for built-in stages, so a stage whose constructor was
        given a different name is what proves they are separate identifiers."""
        _, report = dry_run_stages(
            [
                DummySpecStage(meta=two_group_meta()),
                FanoutAndSumStage(branch_prefix="b", n_children=2),
                MeasurementStage(),
            ],
            dummy_pipeline_env,
            dry=False,
        )
        info = next(s for s in report.stages if s.name == "FanoutAndSumStage")
        assert info.label == "FanoutAndSumStage:b"

    def test_sample_pipeline_is_once_cadence(
        self, default_test_simulator, default_optimizer
    ):
        reports = h2_vqe(default_test_simulator, default_optimizer).dry_run()
        assert reports["sample"].cadence is PipelineCadence.ONCE
        assert reports["cost"].cadence is PipelineCadence.PER_EVALUATION

    @pytest.mark.parametrize(
        "optimizer_factory, qem",
        [
            (MonteCarloOptimizer, None),
            (lambda: ScipyOptimizer(method=ScipyMethod.COBYLA), None),
            (SPSAOptimizer, None),
            (
                lambda: ScipyOptimizer(method=ScipyMethod.COBYLA),
                lambda: QuEPP(truncation_order=1, n_twirls=3),
            ),
        ],
        ids=["population", "gradient-free", "spsa", "mitigated"],
    )
    def test_two_qubit_total_matches_the_submitted_circuit_count(
        self, default_test_simulator, optimizer_factory, qem
    ):
        """Asserted as the invariant ``total × per-circuit == total_2q_gates`` rather
        than against fixed numbers. The stats walk DAG bodies and never sees
        ``qasm_bodies``, where parameter binding fans out — so a population
        optimizer's multiplicity went missing while mitigation's was counted, and a
        spot-check on one optimizer could not tell the difference."""
        kwargs = {"qem_protocol": qem()} if qem else {}
        vqe = h2_vqe(default_test_simulator, optimizer_factory(), **kwargs)
        for forced in (False, True):
            report = vqe.dry_run(force_circuit_generation=forced)["cost"]
            per_circuit = next(
                s.metadata["n_2q_gates"]
                for s in report.stages
                if "n_2q_gates" in s.metadata
            )
            assert report.circuit_stats["total_2q_gates"] == (
                report.total_circuits * per_circuit
            )

    def test_two_previews_of_one_program_compare_equal(self, sampling_test_simulator):
        """Nothing about a preview depends on when it ran, so a report is a value:
        two previews of an unchanged program are the same report."""
        vqe = h2_vqe(
            sampling_test_simulator,
            SPSAOptimizer(),
            shot_distribution="weighted",
        )
        report, twin = vqe.dry_run()["cost"], vqe.dry_run()["cost"]
        assert report == twin
        # The nested allocation documented in the pipeline guide is readable either way.
        allocations = report.env_artifacts["per_group_shots"]
        assert next(iter(next(iter(allocations.values())).values())) is not None

    @pytest.mark.parametrize(
        "estimator",
        [
            PullbackMetricEstimator(),
            FubiniStudyMetricEstimator(),
            StochasticFidelityMetricEstimator(),
        ],
    )
    def test_every_estimator_validates_before_enumerating(
        self, default_test_simulator, default_optimizer, estimator, mocker
    ):
        """Every estimator's ``preprocessors`` must call ``check_compatible`` first,
        so enumerating routines cannot describe a pairing ``run()`` would reject —
        the contract can't hold for one estimator and silently lapse in another. It
        must also validate exactly once per call, not per routine returned."""
        vqe = metric_compatible_vqe(default_test_simulator, default_optimizer)
        spy = mocker.spy(estimator, "check_compatible")
        estimator.preprocessors(vqe)
        spy.assert_called_once_with(vqe)

    def test_qng_rejects_a_fidelity_sampling_estimator(self):
        """QNG needs a closed-form metric; the stochastic-fidelity estimator only
        supplies a fidelity function, so the pairing fails at construction rather
        than with a bare ValueError deep inside optimize()."""
        with pytest.raises(ValueError, match="supplies no closed-form metric"):
            QNGOptimizer(metric_estimator=StochasticFidelityMetricEstimator())

    def test_a_metric_the_ansatz_rejects_refuses_the_preview(
        self, default_test_simulator, default_optimizer
    ):
        """The ansatz's angles are expressions rather than bare parameters, which
        Fubini–Study cannot differentiate. Refusing names the gate and the angle;
        reporting the cost routine alone would price a run that cannot happen."""
        vqe = h2_vqe(
            default_test_simulator,
            QNSPSAOptimizer(metric_estimator=FubiniStudyMetricEstimator()),
        )
        with pytest.raises(ContractViolation, match="not a bare trainable parameter"):
            vqe.dry_run()

    # The forced path runs QuEPP's real expand, which warns that Monte Carlo
    # sampling needs concrete angles — expected when comparing against it.
    @pytest.mark.filterwarnings("ignore:QuEPP")
    @pytest.mark.parametrize("stat", ["mean_depth", "max_depth"])
    def test_analytic_path_matches_counts_but_not_mitigated_depth(
        self, default_test_simulator, default_optimizer, stat
    ):
        """The documented split: circuit counts are exact on the analytic path,
        while the shape figures predate a rewriting stage (QuEPP + twirls here),
        because it is previewed as placeholders rather than real circuits."""
        vqe = h2_vqe(
            default_test_simulator,
            default_optimizer,
            qem_protocol=QuEPP(truncation_order=1, n_twirls=4),
        )
        analytic = vqe.dry_run()["cost"]
        forced = vqe.dry_run(force_circuit_generation=True)["cost"]

        assert analytic.total_circuits == forced.total_circuits
        assert analytic.circuit_stats[stat] < forced.circuit_stats[stat]

    def test_once_pipeline_previews_a_single_parameter_set(
        self, default_test_simulator, default_optimizer
    ):
        """A one-time readout runs after optimization at the trained parameters —
        one set — not over the optimizer's working set. Previewing it with the
        population would report circuits the run never submits."""
        population = 6
        vqe = metric_compatible_vqe(
            default_test_simulator,
            MonteCarloOptimizer(population_size=population, n_best_sets=1),
        )
        reports = vqe.dry_run()
        assert reports["cost"].total_circuits == population
        assert reports["sample"].total_circuits == 1

    def test_sampling_readout_reports_no_observable_instead_of_zeros(
        self, default_test_simulator, default_optimizer
    ):
        """A computational-basis readout has no observable to partition, so the
        group/term counts are omitted rather than printed as a misleading ``0``."""
        vqe = metric_compatible_vqe(default_test_simulator, default_optimizer)
        sample_meta = next(
            s.metadata
            for s in vqe.dry_run()["sample"].stages
            if s.name == "MeasurementStage"
        )
        assert "readout" in sample_meta
        assert "n_groups" not in sample_meta
        assert "n_pauli_terms" not in sample_meta

    def test_each_metric_block_gets_its_own_report_entry(
        self, default_test_simulator, dummy_expval_backend
    ):
        """The pipeline name is the report key, so blocks sharing one would collapse
        into a single entry showing only the last block's circuits."""
        vqe = metric_compatible_vqe(
            default_test_simulator,
            QNGOptimizer(metric_estimator=FubiniStudyMetricEstimator()),
        )
        prefix_keys = [
            k for k in vqe.dry_run() if k.startswith(f"{METRIC_ROUTINE}[block")
        ]
        assert len(prefix_keys) > 1
        assert len(set(prefix_keys)) == len(prefix_keys)
        assert all("block" in k for k in prefix_keys)

    @pytest.mark.parametrize(
        "optimizer_factory",
        [
            QNSPSAOptimizer,
            QNGOptimizer,
            lambda: QNGOptimizer(metric_estimator=FubiniStudyMetricEstimator()),
            MonteCarloOptimizer,
            lambda: ScipyOptimizer(method=ScipyMethod.L_BFGS_B),
        ],
        ids=["qnspsa", "qng", "qng-fs", "montecarlo", "lbfgsb"],
    )
    def test_forced_generation_works_for_every_auxiliary_pipeline(
        self, default_test_simulator, optimizer_factory
    ):
        """The report recommends ``force_circuit_generation=True`` for real shapes,
        so it must work on the pipelines the report shows. An auxiliary routine may
        bind a different parameter width than the cost one — an overlap circuit
        carries two concatenated vectors — which the env has to supply per routine.
        """
        vqe = metric_compatible_vqe(default_test_simulator, optimizer_factory())
        lazy = vqe.dry_run()
        forced = vqe.dry_run(force_circuit_generation=True)
        assert set(lazy) == set(forced)
        for name, report in lazy.items():
            assert report.total_circuits == forced[name].total_circuits

    def test_forced_shape_is_reproducible_under_twirling(
        self, default_test_simulator, default_optimizer
    ):
        """Twirl labels are sampled, so an unseeded stage makes the one figure a
        hardware budget quotes — max depth — drift between calls on the same
        program, silently and with nothing marking it as a sample."""
        vqe = h2_vqe(
            default_test_simulator,
            default_optimizer,
            qem_protocol=QuEPP(truncation_order=1, n_twirls=4),
            seed=1234,
        )
        first, second = (
            vqe.dry_run(force_circuit_generation=True)["cost"].circuit_stats
            for _ in range(2)
        )
        assert first == second

        twin = h2_vqe(
            default_test_simulator,
            default_optimizer,
            qem_protocol=QuEPP(truncation_order=1, n_twirls=4),
            seed=1234,
        )
        assert (
            twin.dry_run(force_circuit_generation=True)["cost"].circuit_stats == first
        )

    def test_widest_basis_change_spans_all_groups(
        self, default_test_simulator, default_optimizer
    ):
        """With grouping off every group holds one term, so reading the width off
        the largest-by-size group degenerates to 1 — it must be the widest basis
        change any group needs (the pair-hopping term spans all four qubits)."""
        vqe = h2_vqe(default_test_simulator, default_optimizer, grouping_strategy=None)
        meta = next(
            s.metadata
            for s in vqe.dry_run()["cost"].stages
            if s.name == "MeasurementStage"
        )
        assert meta["largest_group_size"] == 1
        assert meta["largest_group_width"] == 4

    def test_stochastic_bind_and_dry_run_share_overlap_preprocessor(
        self, default_test_simulator, default_optimizer
    ):
        """``bind`` and ``preprocessors`` build the same overlap routine (single
        source), so what is reported can't drift from what runs."""
        vqe = metric_compatible_vqe(default_test_simulator, default_optimizer)
        est = StochasticFidelityMetricEstimator()
        (preview_pp,) = est.preprocessors(vqe)
        run_pp = est._build_overlap_preprocessor(vqe)
        assert preview_pp.name == run_pp.name == METRIC_ROUTINE
        assert preview_pp.cache_key == run_pp.cache_key == METRIC_ROUTINE
        assert preview_pp.result_format == run_pp.result_format

    def test_dry_run_does_not_advance_program_rng(
        self, sampling_test_simulator, default_optimizer
    ):
        """A weighted-random preview draws from a throwaway RNG, so it neither
        advances the program's live generator (a later run() is unaffected) nor
        varies between repeated previews."""
        vqe = h2_vqe(
            sampling_test_simulator,
            default_optimizer,
            grouping_strategy="qwc",
            shot_distribution="weighted_random",
            seed=1234,
        )
        before = vqe._serialized_rng_state
        reports_a = vqe.dry_run()
        assert vqe._serialized_rng_state == before  # live RNG untouched

        reports_b = vqe.dry_run()
        assert (
            reports_a["cost"].env_artifacts["per_group_shots"]
            == reports_b["cost"].env_artifacts["per_group_shots"]
        )

    def test_dry_run_builds_env_without_reporter(
        self, default_test_simulator, default_optimizer, mocker
    ):
        """The preview must stay silent — no progress reporter is wired into the
        forward-pass env, so nothing bleeds into stdout."""
        vqe = h2_vqe(default_test_simulator, default_optimizer)
        spy = mocker.spy(vqe, "_build_pipeline_env")
        vqe.dry_run()
        assert spy.call_count > 0
        assert all(call.kwargs.get("reporter") is None for call in spy.call_args_list)

    @pytest.mark.filterwarnings(
        "ignore:Backend supports analytic expectation values:UserWarning"
    )
    def test_default_and_forced_match_non_qem(
        self, default_test_simulator, default_optimizer
    ):
        """Parity between analytic and ``force_circuit_generation=True`` must
        also hold for programs without any QEM stage — the ``TimeEvolution``
        fixture above brings QuEPP, so the non-QEM path needs its own lock."""
        vqe = h2_vqe(default_test_simulator, default_optimizer, grouping_strategy="qwc")
        assert_report_dicts_match(
            vqe.dry_run(), vqe.dry_run(force_circuit_generation=True)
        )

    def test_dry_run_skips_initial_spec_without_preprocessors(
        self, default_test_simulator, default_optimizer, mocker
    ):
        """A program exposing no named pipelines returns ``{}`` from
        ``dry_run()`` without invoking ``_initial_spec()`` — honoring the
        documented opt-out for programs that never call ``evaluate()``."""
        vqe = h2_vqe(default_test_simulator, default_optimizer)
        mocker.patch.object(vqe, "_preprocessors", return_value=())
        spec = mocker.patch.object(
            vqe, "_initial_spec", side_effect=NotImplementedError
        )

        assert vqe.dry_run() == {}
        spec.assert_not_called()


def test_measurement_reports_which_qubits_it_reads(dummy_simulator):
    """``measure_all`` narrows the outcome space below the register width, which no
    other figure in the report reflects. A sampling backend, since a backend that
    evaluates the observable itself builds no basis-change circuits to narrow."""
    env = PipelineEnv(backend=dummy_simulator)

    def measured(flag):
        _, report = dry_run_stages(
            [DummySpecStage(meta=two_group_meta()), MeasurementStage(measure_all=flag)],
            env,
        )
        return report.stages[-1].metadata["measured_qubits"]

    assert measured(False) == "per group (observable support only)"
    assert measured(True) == "all"


@pytest.mark.parametrize(
    "make_program",
    [
        pytest.param(_h2_vqe_for_totals, id="vqe-circuit-seed"),
        pytest.param(_maxcut_qaoa_for_totals, id="qaoa-hamiltonian-seed"),
    ],
)
def test_cost_evaluation_submits_what_the_report_says(
    make_program, default_test_simulator, mocker
):
    """Oracle for the one claim the report makes: what a single evaluation submits.

    Measured against real submissions rather than restated arithmetic, and across
    both seed kinds — a Hamiltonian-seeded program declares no parameters on its
    own seed.
    """
    program = make_program(
        default_test_simulator, MonteCarloOptimizer(population_size=4, n_best_sets=1)
    )
    expected = program.dry_run()["cost"].total_circuits

    spy = mocker.spy(program.backend, "submit_circuits")
    params = np.zeros((program.optimizer.n_param_sets, program.n_params))
    program.evaluate(params, program.cost_preprocessor())

    submitted = sum(
        len(payload.parameter_sets)
        for call in spy.call_args_list
        for payload in call.args[0]
    )
    assert submitted == expected


@pytest.mark.parametrize(
    "make_program",
    [
        pytest.param(_h2_vqe_for_totals, id="vqe-circuit-seed"),
        pytest.param(_maxcut_qaoa_for_totals, id="qaoa-hamiltonian-seed"),
    ],
)
def test_one_time_readout_binds_one_parameter_set(make_program, default_test_simulator):
    """The final readout samples the trained parameters — one set, whatever the
    optimizer's working set was during the loop."""
    program = make_program(
        default_test_simulator, MonteCarloOptimizer(population_size=4)
    )
    once = [r for r in program.dry_run().values() if r.cadence is PipelineCadence.ONCE]
    assert once, "expected a one-time readout routine"
    for report in once:
        binding = next(s for s in report.stages if "Binding" in s.name)
        assert binding.metadata["n_param_sets"] == 1


@pytest.mark.parametrize(
    "make_program",
    [
        pytest.param(_h2_vqe_for_totals, id="vqe-circuit-seed"),
        pytest.param(_maxcut_qaoa_for_totals, id="qaoa-hamiltonian-seed"),
    ],
)
def test_recurring_and_one_time_routines_are_labeled_apart(
    make_program, default_test_simulator
):
    """The only run-level fact the report carries is which routines recur, so a
    reader can multiply the recurring ones by their own iteration count."""
    reports = make_program(
        default_test_simulator, MonteCarloOptimizer(population_size=4)
    ).dry_run()
    cadences = {name: r.cadence for name, r in reports.items()}
    assert cadences["cost"] is PipelineCadence.PER_EVALUATION
    assert cadences["sample"] is PipelineCadence.ONCE


def test_quepp_montecarlo_dry_run_suppresses_fallback_warning(dummy_pipeline_env):
    """A dry preview always sees symbolic angles, so montecarlo QuEPP would warn
    it is falling back to exhaustive enumeration. That is execution-path noise
    for a nothing-executes preview and must stay silent."""
    meta = parametric_twirlable_meta()
    dummy_pipeline_env.param_sets = np.asarray([[0.1, 0.2]])
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        dry_run_stages(
            [
                DummySpecStage(meta=meta),
                QEMStage(protocol=QuEPP(truncation_order=1, n_twirls=0, n_samples=4)),
                MeasurementStage(),
            ],
            dummy_pipeline_env,
            suppress_performance_warnings=True,
        )
    assert not any("Monte Carlo sampling" in str(w.message) for w in record)


class _IncompleteMetricEstimator(MetricEstimator):
    """A metric estimator missing ``preprocessors`` — must stay abstract."""

    def check_compatible(self, program):
        pass

    def bind(self, program):
        return {}

    # deliberately omits preprocessors


def test_metric_estimator_must_declare_its_routines():
    """The abstract contract that prevents silent omission: a new metric estimator
    cannot be instantiated without declaring the routines it measures through, so
    they cannot be left out of what a program reports."""
    with pytest.raises(TypeError):
        _IncompleteMetricEstimator()


def test_pce_cost_pipeline_has_no_phantom_reduction(dummy_simulator, default_optimizer):
    """PCE reads raw bitstrings — one circuit per spec, no per-Pauli-term
    grouping — so its cost pipeline must not show a per-term baseline or a
    phantom reduction from the placeholder Hamiltonian it never measures."""
    pce = PCE(
        problem=BinaryOptimizationProblem(np.array([[1.0, 0.2], [0.2, 2.0]])),
        ansatz=GenericLayerAnsatz([RYGate, RZGate]),
        optimizer=default_optimizer,
        backend=dummy_simulator,
    )
    cost = pce.dry_run()["cost"]
    assert all(
        stage.factor in (1.0, float(cost.total_circuits)) for stage in cost.stages
    )
    assert not any(stage.factor < 1.0 for stage in cost.stages)  # no phantom ÷N
