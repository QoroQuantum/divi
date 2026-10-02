# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""format_dry_run: single-program trees, ensemble styles, grouping, and render safety."""

import io
import warnings
from dataclasses import replace

import numpy as np
import pytest
from qiskit.circuit.library import RYGate, RZGate

from divi.pipeline import (
    DryRunReport,
    PipelineCadence,
    StageInfo,
    format_dry_run,
)
from divi.pipeline._dry_run import (
    _cost_headline,
)
from divi.pipeline._dry_run_format import (
    _AUTO_COMPACT_MAX,
    _AUTO_VERBOSE_MAX,
    _auto_style,
    _cadence_totals,
    _distinguishing_trait,
    _merge_group_report,
    _program_signature,
    _shared_metadata,
)
from divi.pipeline.stages import (
    MeasurementStage,
    PauliTwirlStage,
)
from divi.qprog.algorithms import PCE, GenericLayerAnsatz
from divi.qprog.optimizers import (
    MonteCarloOptimizer,
)
from divi.qprog.problems import BinaryOptimizationProblem
from tests._helpers import exact_match
from tests.pipeline._helpers import (
    DummySpecStage,
    dry_run_stages,
    h2_vqe,
    metric_compatible_vqe,
    parametric_twirlable_meta,
    two_group_meta,
)

_IGNORED_STYLE_WARNING = (
    "style='grouped' was ignored: it selects between multi-program layouts, and "
    "this is a single program's report. Pass the nested result from "
    "ProgramEnsemble.dry_run() to use it."
)


def _mixed_flat_message(type_name, key):
    return (
        f"format_dry_run expects DryRunReport values throughout, got {type_name} "
        f"for {key!r}. A dict may not mix a single program's reports with an "
        "ensemble's."
    )


_STATS = {
    "mean_depth": 4,
    "std_depth": 0.0,
    "min_depth": 4,
    "max_depth": 4,
    "mean_2q_depth": 2.0,
    "total_2q_gates": 6,
    "mean_width": 2.0,
    "std_width": 0.0,
    "min_width": 2,
    "max_width": 2,
}


def _stage(name, factor, axis=None, **metadata):
    return StageInfo(name=name, axis=axis, factor=factor, metadata=metadata)


def _hand_report(stages, total_circuits, total_shots=0, **fields):
    fields.setdefault("pipeline_name", "cost")
    return DryRunReport(
        stages=tuple(stages),
        total_circuits=total_circuits,
        total_shots=total_shots,
        **fields,
    )


def _render_lines(payload, style=None, width=200):
    buffer = io.StringIO()
    format_dry_run(payload, style=style, file=buffer, width=width)
    return buffer.getvalue().splitlines()


def _cost_report():
    """Spec ×2, binding ×3, grouping ÷2: three circuits that multiply out."""
    return _hand_report(
        [
            _stage("CircuitSpecStage", 2.0, "circuit"),
            _stage("ParameterBindingStage", 3.0, "param_set"),
            _stage("MeasurementStage", 0.5, "obs_group"),
        ],
        3,
        300,
        circuit_stats=_STATS,
    )


def _sampled_once_report():
    return _hand_report(
        [
            _stage("CircuitSpecStage", 1.0, "circuit"),
            _stage(
                "QEMStage",
                3.0,
                "qem",
                path_count="sampled (an estimate)",
                n_samples=8,
            ),
        ],
        3,
        30,
        pipeline_name="sample",
        cadence=PipelineCadence.ONCE,
    )


@pytest.fixture(autouse=True)
def plain_rich_output(monkeypatch):
    """Render without color, whatever the ambient terminal settings are.

    Rich honors ``FORCE_COLOR`` even when writing to a ``StringIO``, and the
    escape sequences then count toward ``len(line)`` and interleave with the
    text these tests assert on -- a 20-column line measures 29 characters.
    """
    for variable in ("FORCE_COLOR", "CLICOLOR_FORCE", "COLORTERM"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("TERM", "dumb")


class TestEnsembleDryRunFormatting:
    """``format_dry_run`` dispatches on report shape (flat single-program vs
    nested ensemble) and honors the ``style`` argument for the nested form."""

    def _simple_report(self, env):
        _, report = dry_run_stages(
            [DummySpecStage(meta=two_group_meta()), MeasurementStage()], env, dry=False
        )
        return report

    def _twirl_report(self, env):
        """A structurally distinct report (extra PauliTwirl stage)."""
        _, report = dry_run_stages(
            [
                DummySpecStage(meta=parametric_twirlable_meta()),
                PauliTwirlStage(n_twirls=3, seed=0),
                MeasurementStage(),
            ],
            env,
            dry=False,
        )
        return report

    def test_signature_buckets_identical_and_separates_distinct(
        self, dummy_pipeline_env
    ):
        r1 = self._simple_report(dummy_pipeline_env)
        r2 = self._simple_report(dummy_pipeline_env)
        r3 = self._twirl_report(dummy_pipeline_env)
        assert _program_signature({"cost": r1}) == _program_signature({"cost": r2})
        assert _program_signature({"cost": r1}) != _program_signature({"cost": r3})

    def test_signature_ignores_routine_insertion_order(self, dummy_pipeline_env):
        cost = self._simple_report(dummy_pipeline_env)
        final = self._twirl_report(dummy_pipeline_env)
        assert _program_signature({"cost": cost, "final": final}) == (
            _program_signature({"final": final, "cost": cost})
        )

    def test_signature_ignores_compensating_stage_factors(self, dummy_pipeline_env):
        """A spec stage emitting one circuit per observable term, then a
        measurement stage grouping them back down, differs numerically between
        programs while leaving the outcome identical — grouping must not split on
        it, or a uniform sweep fragments into unreadably many groups."""
        few_terms = self._simple_report(dummy_pipeline_env)
        spec, *rest = few_terms.stages
        many_terms = replace(
            few_terms,
            stages=(
                replace(spec, factor=spec.factor * 3),
                *(replace(s, factor=s.factor / 3) for s in rest),
            ),
        )
        # Different per-stage fan-out, identical outcome.
        assert few_terms.stages[0].factor != many_terms.stages[0].factor
        assert few_terms.total_circuits == many_terms.total_circuits
        assert _program_signature({"cost": few_terms}) == _program_signature(
            {"cost": many_terms}
        )

    def test_signature_separates_programs_on_different_registers(
        self, dummy_pipeline_env
    ):
        """A program built on the wrong number of qubits is a different program
        and must not hide inside a group — that outlier is the whole reason to
        read the report."""
        healthy = self._simple_report(dummy_pipeline_env)
        narrow = replace(
            healthy, circuit_stats={**healthy.circuit_stats, "max_width": 3}
        )
        assert _program_signature({"cost": healthy}) != _program_signature(
            {"cost": narrow}
        )

    def test_signature_separates_programs_with_different_parameter_counts(
        self, dummy_pipeline_env
    ):
        """A layer/ansatz sweep varies the parameter count — the swept axis must
        stay visible rather than collapsing into one group."""
        base = self._simple_report(dummy_pipeline_env)
        spec, *rest = base.stages
        few = replace(base, stages=(replace(spec, metadata={"n_params": 4}), *rest))
        many = replace(base, stages=(replace(spec, metadata={"n_params": 32}), *rest))
        assert _program_signature({"cost": few}) != _program_signature({"cost": many})

    def test_signature_separates_programs_optimizing_different_objectives(
        self, dummy_pipeline_env
    ):
        """Same register, same ansatz, same parameter count, different coefficients
        — every count agrees, so nothing else in the key can tell these apart, and
        collapsing them reports one program's objective as the group's."""
        base = self._simple_report(dummy_pipeline_env)
        ising = replace(
            base, objective_fingerprint=(("ZZ", "XI"), (1.0, 0.5), (0.0, 0.0))
        )
        reweighted = replace(
            base, objective_fingerprint=(("ZZ", "XI"), (1.0, -0.5), (0.0, 0.0))
        )
        assert _program_signature({"cost": ising}) != _program_signature(
            {"cost": reweighted}
        )
        assert _program_signature({"cost": ising}) == _program_signature(
            {"cost": replace(ising)}
        )

    def test_signature_separates_supervised_from_unsupervised(self, dummy_pipeline_env):
        """A supervised reduction against labels and a bare expectation value over
        the *same* observable are different objectives. The observable alone cannot
        tell them apart, so the key has to carry the loss shaping too."""
        base = self._simple_report(dummy_pipeline_env)
        spec, *rest = base.stages

        def with_supervision(flag):
            return replace(
                base,
                stages=(replace(spec, metadata={"supervised": flag}), *rest),
                objective_fingerprint=(
                    (("ZZ",), (1.0,), (0.0,)),
                    (("supervised", flag),),
                ),
            )

        assert _program_signature(
            {"cost": with_supervision(True)}
        ) != _program_signature({"cost": with_supervision(False)})

    def test_objective_fingerprint_tracks_the_terminal_stage_not_the_last_stage(
        self, dummy_simulator
    ):
        """Authors are required to put ParameterBindingStage last, so reading
        ``stages[-1]`` fingerprints the binding config — ``fast_path``,
        ``template_path`` — and every program measuring the same observable looks
        identical however differently its terminal stage reduces the results. PCE's
        CVaR tail is exactly that: same Pauli terms, different objective."""
        problem = BinaryOptimizationProblem(np.array([[1.0, 0.2], [0.2, 2.0]]))

        def pce(alpha):
            return PCE(
                problem=problem,
                ansatz=GenericLayerAnsatz([RYGate, RZGate]),
                alpha=alpha,
                backend=dummy_simulator,
                optimizer=MonteCarloOptimizer(),
            ).dry_run()

        soft, hard = pce(0.1), pce(0.9)
        assert soft["cost"].objective_fingerprint != hard["cost"].objective_fingerprint
        assert _program_signature(soft) != _program_signature(hard)
        # ...and an unchanged objective still groups.
        assert _program_signature(pce(0.1)) == _program_signature(soft)

    def test_merged_group_marks_metadata_the_members_disagree_on(
        self, dummy_pipeline_env
    ):
        """A dropped row is invisible without a uniform run to diff against, so a
        divergent field is marked rather than omitted."""
        base = self._simple_report(dummy_pipeline_env)
        spec, *rest = base.stages
        soft = replace(
            base, stages=(replace(spec, metadata={"objective": "soft"}), *rest)
        )
        hard = replace(
            base, stages=(replace(spec, metadata={"objective": "hard"}), *rest)
        )
        merged = _merge_group_report([soft, hard])
        assert merged.stages[0].metadata["objective"] == "mixed (soft | hard)"

    def test_merged_group_surfaces_differing_factors_as_a_range(
        self, dummy_pipeline_env
    ):
        """Members of a group may differ in a per-stage factor; the merged tree
        must disclose the spread rather than pass one member's number off as the
        group's."""
        base = self._simple_report(dummy_pipeline_env)
        wider = replace(
            base,
            stages=(replace(base.stages[0], factor=base.stages[0].factor * 2),)
            + base.stages[1:],
        )
        merged = _merge_group_report([base, wider])
        assert "factor_range" in merged.stages[0].metadata

    @pytest.mark.parametrize("style", ["compact", "grouped", "verbose"])
    def test_ensemble_styles_render_with_total(self, dummy_pipeline_env, capsys, style):
        report = self._simple_report(dummy_pipeline_env)
        nested = {"p1": {"cost": report}, "p2": {"cost": report}}
        format_dry_run(nested, style=style)
        assert "Ensemble total" in capsys.readouterr().out

    def test_invalid_style_raises(self):
        with pytest.raises(
            ValueError,
            match=exact_match(
                "Unknown style 'bogus'; expected one of compact, grouped, verbose."
            ),
        ):
            format_dry_run({}, style="bogus")

    def test_grouped_aggregates_stats_across_members(self):
        """Grouped members share a pipeline shape and a register width but not
        circuit content, so the depth summary AND content-dependent metadata must
        describe the whole group rather than echoing one member."""

        def _report(depth, group_width):
            return _hand_report(
                [
                    _stage(
                        "CircuitSpecStage",
                        14.0,
                        "circuit",
                        strategy="qwc",
                        largest_group_width=group_width,
                    )
                ],
                14,
                circuit_stats={
                    **_STATS,
                    "mean_depth": depth,
                    "min_depth": depth,
                    "max_depth": depth,
                },
            )

        # Same register so they bucket together, but different circuit content —
        # the case where a group must describe its members honestly.
        nested = {"p1": {"cost": _report(10, 4)}, "p2": {"cost": _report(20, 8)}}
        out = "\n".join(_render_lines(nested, "grouped"))
        assert "2 programs" in out  # same signature -> one bucket
        assert "depth 10-20" in out  # span, not a mean describing neither member
        assert "±" not in out  # no misleading combined std for a group
        # Divergent metadata is marked, never silently dropped; metadata
        # identical across members passes through unchanged.
        assert "largest_group_width: mixed (4 | 8)" in out
        assert "strategy: qwc" in out

    def test_flat_input_warns_and_still_renders(self, dummy_pipeline_env, capsys):
        """A style has nothing to select between for one program, so it warns —
        but the single-program tree still renders rather than being suppressed."""
        report = self._simple_report(dummy_pipeline_env)
        with pytest.warns(UserWarning, match=exact_match(_IGNORED_STYLE_WARNING)):
            format_dry_run({"cost": report}, style="grouped")
        out = capsys.readouterr().out
        assert "Total (per evaluation):" in out
        assert "Ensemble total" not in out

    def test_once_pipeline_total_label(self, dummy_pipeline_env, capsys):
        report = replace(
            self._simple_report(dummy_pipeline_env),
            pipeline_name="sample",
            cadence=PipelineCadence.ONCE,
        )
        format_dry_run({"sample": report})  # flat single-program render
        out = capsys.readouterr().out
        assert "Total (once):" in out
        assert "Total (per evaluation):" not in out

    def test_cadence_totals_split_by_cadence(self, dummy_pipeline_env):
        cost = self._simple_report(dummy_pipeline_env)  # PER_EVALUATION
        sample = replace(cost, cadence=PipelineCadence.ONCE)
        nested = {
            "p1": {"cost": cost, "sample": sample},
            "p2": {"cost": cost, "sample": sample},
        }
        assert _cadence_totals(nested.values(), PipelineCadence.PER_EVALUATION) == (
            2 * cost.total_circuits,
            2 * cost.total_shots,
        )
        assert _cadence_totals(nested.values(), PipelineCadence.ONCE) == (
            2 * sample.total_circuits,
            2 * sample.total_shots,
        )
        # A cadence with no matching pipeline sums to zero, never raises.
        assert _cadence_totals([{"cost": cost}], PipelineCadence.ONCE) == (0, 0)

    def test_grouped_subtotal_splits_cadence(self, dummy_pipeline_env, capsys):
        cost = self._simple_report(dummy_pipeline_env)
        sample = replace(cost, cadence=PipelineCadence.ONCE)
        nested = {f"p{i}": {"cost": cost, "sample": sample} for i in range(20)}
        format_dry_run(nested, style="grouped")
        out = capsys.readouterr().out
        # The subtotal must split cadences like the grand total, not sum them.
        assert "Subtotal" in out and "per evaluation:" in out and "once:" in out

    def test_signature_ignores_depth_but_not_register_width(self):
        """Depth varies across a legitimately-uniform sweep (different graphs of
        the same size), so keying on it would fragment one group per program.
        Register width does not vary that way — a differing width means a
        differently-built program."""

        def report(mean_depth: int, width: int) -> DryRunReport:
            return _hand_report(
                [_stage("CircuitSpecStage", 14.0, "circuit")],
                14,
                circuit_stats={
                    **_STATS,
                    "mean_depth": mean_depth,
                    "min_depth": mean_depth,
                    "max_depth": mean_depth,
                    "mean_width": float(width),
                    "min_width": width,
                    "max_width": width,
                },
            )

        shallow, deep = report(5, 4), report(99, 4)
        assert _program_signature({"cost": shallow}) == _program_signature(
            {"cost": deep}
        )
        assert _program_signature({"cost": shallow}) != _program_signature(
            {"cost": report(5, 20)}
        )

    def test_compact_renders_tuple_program_ids(self, dummy_pipeline_env, capsys):
        report = self._simple_report(dummy_pipeline_env)
        nested = {("A", 0): {"cost": report}, ("A", 1): {"cost": report}}
        format_dry_run(nested, style="compact")
        assert "('A', 0)" in capsys.readouterr().out

    def test_multi_pipeline_program_totals(self, dummy_pipeline_env, capsys):
        r1 = self._simple_report(dummy_pipeline_env)
        r2 = self._twirl_report(dummy_pipeline_env)
        nested = {
            "p1": {"cost": r1, "extra": r2},
            "p2": {"cost": r1, "extra": r2},
        }
        per = r1.total_circuits + r2.total_circuits
        per_shots = r1.total_shots + r2.total_shots
        format_dry_run(nested, style="grouped")
        out = capsys.readouterr().out
        # Both pipelines are per-evaluation, so the subtotal is single-cadence:
        # the ×2 group total carries circuits and shots.
        assert (
            f"Subtotal (× 2) (per evaluation): {_cost_headline(2 * per, 2 * per_shots)}"
            in out
        )

    def test_compact_single_program_is_grammatical(self, dummy_pipeline_env, capsys):
        report = self._simple_report(dummy_pipeline_env)
        format_dry_run({"p1": {"cost": report}}, style="compact")
        out = capsys.readouterr().out
        assert "(1 program)" in out
        assert "(1 programs)" not in out

    def test_styles_produce_distinct_shapes(self, dummy_pipeline_env, capsys):
        report = self._simple_report(dummy_pipeline_env)
        nested = {"p1": {"cost": report}, "p2": {"cost": report}}
        stage_name = report.stages[0].name

        format_dry_run(nested, style="compact")
        compact = capsys.readouterr().out
        format_dry_run(nested, style="verbose")
        verbose = capsys.readouterr().out

        assert "p1" in compact and "p2" in compact
        # verbose expands the per-stage tree; compact stays at one line per pipeline
        assert stage_name in verbose
        assert stage_name not in compact

    def test_auto_style_thresholds(self):
        assert _auto_style(1) == "verbose"
        assert _auto_style(_AUTO_VERBOSE_MAX) == "verbose"
        assert _auto_style(_AUTO_VERBOSE_MAX + 1) == "compact"
        assert _auto_style(_AUTO_COMPACT_MAX) == "compact"
        assert _auto_style(_AUTO_COMPACT_MAX + 1) == "grouped"

    def test_default_style_small_ensemble_is_verbose(self, dummy_pipeline_env, capsys):
        report = self._simple_report(dummy_pipeline_env)
        format_dry_run({"p1": {"cost": report}, "p2": {"cost": report}})
        out = capsys.readouterr().out
        # Verbose is the per-stage-tree layout; compact would show a summary row
        # per pipeline and no stage names. Every style states the program count.
        assert report.stages[0].name in out
        assert "2 programs" in out

    def test_default_style_compact_band(self, dummy_pipeline_env, capsys):
        report = self._simple_report(dummy_pipeline_env)
        n = _AUTO_VERBOSE_MAX + 1
        format_dry_run({f"p{i}": {"cost": report} for i in range(n)})
        out = capsys.readouterr().out
        assert "Ensemble Dry Run" in out  # compact header
        assert report.stages[0].name not in out  # compact omits stage trees

    def test_default_style_large_ensemble_is_grouped(self, dummy_pipeline_env, capsys):
        report = self._simple_report(dummy_pipeline_env)
        n = _AUTO_COMPACT_MAX + 1
        format_dry_run({f"p{i}": {"cost": report} for i in range(n)})
        assert f"{n} programs" in capsys.readouterr().out  # grouped count header

    def test_flat_empty_prints_note(self, capsys):
        format_dry_run({})
        assert "No dry-run reports" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "name",
        ["metric-prefix[block 7]", "cost[/]", "x[red]y"],
        ids=["block-id", "closing-tag", "style-tag"],
    )
    def test_pipeline_names_with_brackets_render_literally(
        self, default_test_simulator, default_optimizer, capsys, name
    ):
        """Names reach a markup renderer, and a bracketed one is a legal name — the
        built-in Fubini-Study routines carry ``[block N]``. Unescaped, the
        identifier is silently swallowed as a style tag and a closing tag raises."""
        report = replace(
            metric_compatible_vqe(default_test_simulator, default_optimizer).dry_run()[
                "cost"
            ],
            pipeline_name=name,
        )
        format_dry_run({name: report})
        assert name in capsys.readouterr().out

    def test_malformed_nested_input_raises_before_any_output(
        self, default_test_simulator, default_optimizer, capsys
    ):
        """``EnsembleReports`` is a plain dict, so hand-assembling one is expected.
        Validating only the outer level lets a bad inner value print a header and
        then raise from inside the renderer — the failure the outer check exists to
        prevent."""
        good = metric_compatible_vqe(
            default_test_simulator, default_optimizer
        ).dry_run()
        for payload, message in (
            (
                {"p0": {"cost": "nope"}},
                "format_dry_run expects DryRunReport values, got str for 'p0'/'cost'.",
            ),
            (
                {"x": {"y": good}},
                "format_dry_run expects DryRunReport values, got dict for 'x'/'y'.",
            ),
            (
                {"p0": 7},
                "format_dry_run expects DryRunReport values (or, for an ensemble, "
                "dicts of them), got int.",
            ),
            (
                {"p0": good, "p1": "nope"},
                "format_dry_run expects each program's value to be a dict of "
                "DryRunReport, got str for program 'p1'.",
            ),
        ):
            with pytest.raises(TypeError, match=exact_match(message)):
                format_dry_run(payload)
            assert capsys.readouterr().out == ""

    def test_grouping_tolerates_metadata_that_is_not_truth_testable(self):
        """A custom stage's ``introspect`` may return an array, whose elementwise
        ``==`` cannot be used in a boolean context — grouping must not assume it."""
        report = _hand_report([_stage("S", 1.0, "a", weights=np.zeros(3), n=2)], 1, 10)
        out = "\n".join(
            _render_lines({"p0": {"cost": report}, "p1": {"cost": report}}, "grouped")
        )
        assert "weights" in out
        assert "mixed" not in out

    def test_style_on_single_program_input_warns(
        self, default_test_simulator, default_optimizer
    ):
        """Styles only arrange multiple programs, so passing one for a single
        program has nothing to select between — say so instead of ignoring it."""
        reports = metric_compatible_vqe(
            default_test_simulator, default_optimizer
        ).dry_run()
        with pytest.warns(UserWarning, match=exact_match(_IGNORED_STYLE_WARNING)):
            format_dry_run(reports, style="grouped")

    def test_format_dry_run_rejects_a_bare_report(
        self, default_test_simulator, default_optimizer
    ):
        """``dry_run()`` returns a dict of reports, so passing one is a natural
        slip — it must name the fix instead of failing on ``.values()``."""
        report = h2_vqe(default_test_simulator, default_optimizer).dry_run()["cost"]
        with pytest.raises(
            TypeError,
            match=exact_match(
                "format_dry_run expects the dict returned by dry_run(), not a single "
                'DryRunReport. Pass the whole result, or wrap it: {"cost": report}.'
            ),
        ):
            format_dry_run(report)

    def test_format_dry_run_rejects_a_value_that_is_not_a_dict(self):
        with pytest.raises(
            TypeError,
            match=exact_match(
                "format_dry_run expects the dict returned by dry_run(), got list. "
                "Call it first: format_dry_run(program.dry_run())."
            ),
        ):
            format_dry_run([])

    def test_format_dry_run_validates_every_flat_value_before_rendering(
        self, default_test_simulator, default_optimizer
    ):
        """Reaching a bad value mid-tree leaves partial output above the traceback,
        so the flat path checks all of its values, not just the first."""
        reports = h2_vqe(default_test_simulator, default_optimizer).dry_run()
        buffer = io.StringIO()
        with pytest.raises(
            TypeError, match=exact_match(_mixed_flat_message("int", "junk"))
        ):
            format_dry_run({"cost": reports["cost"], "junk": 42}, file=buffer)
        assert buffer.getvalue() == ""

        # A dict mixing one program's reports with an ensemble's nested dicts is the
        # same mistake, and must be caught in the same place.
        with pytest.raises(
            TypeError, match=exact_match(_mixed_flat_message("dict", "prog"))
        ):
            format_dry_run(
                {"cost": reports["cost"], "prog": reports}, file=io.StringIO()
            )

    @pytest.mark.parametrize(
        "width", [0, -5, float("nan")], ids=["zero", "negative", "nan"]
    )
    def test_format_dry_run_rejects_a_non_positive_width(
        self, default_test_simulator, default_optimizer, width
    ):
        """A gateway computing width from an unset ``$COLUMNS`` lands on 0, which
        rich renders as nothing at all — an empty pre-flight log and no error. The
        check is stated positively so NaN fails it too; every NaN comparison is
        False."""
        reports = h2_vqe(default_test_simulator, default_optimizer).dry_run()
        with pytest.raises(
            ValueError,
            match=exact_match(
                f"width must be a positive number of columns, got {width}."
            ),
        ):
            format_dry_run(reports, file=io.StringIO(), width=width)

    def test_a_report_reloaded_from_json_still_totals(self):
        """``cadence`` is bucketed by identity, so a plain string from ``json.loads``
        would be dropped from the roll-up — reporting zero circuits for a program
        the row above it says costs seven."""
        report = _hand_report([], 7, 700, cadence="per_evaluation")
        assert report.cadence is PipelineCadence.PER_EVALUATION
        assert "7 circuits" in _render_lines({"p1": {"cost": report}}, "compact")[-1]

    def test_pipeline_and_stage_names_cannot_inject_control_characters(
        self, default_test_simulator, default_optimizer
    ):
        """Pipeline names come from ``CircuitPreprocessor(name=...)`` and stage names
        from custom stages, so both are caller-controlled — the same threat as a
        program id, and they render through different code paths."""
        forged = "x\n└── FORGED: 999 circuits\x1b[31mRED"
        report = _hand_report([_stage(forged, 1.0)], 1, 1, pipeline_name=forged)
        # Compact labels each row with the dict key rather than the report's own
        # name, so the forged text has to arrive by that route to be rendered there.
        for style, payload in (
            (None, {"cost": report}),
            ("compact", {"p": {forged: report}}),
            ("verbose", {"p": {"cost": report}}),
            ("grouped", {"p": {"cost": report}, "q": {"cost": report}}),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                output = "\n".join(_render_lines(payload, style))
            assert "\x1b[31m" not in output, style
            assert "\\x1b" in output and "\\n" in output, style

    def test_stage_metadata_cannot_inject_control_characters(self):
        """Metadata comes from a stage's own ``introspect``, so a custom stage is as
        caller-controlled as a program id — and both its keys and its values are
        rendered."""
        forged = "x\n└── FORGED: 999 circuits\x1b[31mRED"
        report = _hand_report(
            [_stage("S", 1.0, **{forged: "value", "key": forged})], 1, 1
        )
        output = "\n".join(_render_lines({"cost": report}))
        assert "\x1b[31m" not in output
        # Both routes escaped: twice over, once for the key and once for the value.
        assert output.count("\\x1b") == 2
        assert output.count("\\n") == 2

    def test_width_is_threaded_into_the_console(self):
        """``width`` only matters if it reaches the console — a narrow one has to
        actually wrap."""
        report = _hand_report([_stage("AVeryLongStageNameIndeed", 1.0)], 1, 1)
        widths = {
            width: max(
                len(line) for line in _render_lines({"cost": report}, width=width)
            )
            for width in (20, 200)
        }
        assert widths[20] <= 20 < widths[200]

    def test_program_ids_cannot_forge_tree_rows_or_emit_ansi(
        self, default_test_simulator, default_optimizer
    ):
        """Program ids are sweep keys, so they are caller-controlled: a raw newline
        would forge what reads as another pipeline row in a captured log, and a raw
        escape byte would put live ANSI into it."""
        reports = h2_vqe(default_test_simulator, default_optimizer).dry_run()
        forged = "arm1\nfake ├── Total (once): 999999 circuits"
        buffer = io.StringIO()
        format_dry_run(
            {forged: reports, "\x1b[31mred": reports},
            style="compact",
            file=buffer,
            width=200,
        )
        output = buffer.getvalue()
        assert "\\n" in output and "\\x1b" in output
        assert "\x1b[31mred" not in output
        # The forged row is inert: it renders on the id's own line, not as a row.
        assert not any(line.lstrip().startswith("fake") for line in output.splitlines())


@pytest.mark.parametrize(
    "report, expected",
    [
        (
            _hand_report(
                [
                    _stage(
                        "CircuitSpecStage",
                        2.0,
                        "circuit",
                        n_qubits=2,
                        span=(1, 3),
                        ratio=0.123456,
                    ),
                    _stage("ParameterBindingStage", 3.0, "param_set", n_param_sets=3),
                    _stage("MeasurementStage", 0.5, "obs_group"),
                    _stage("Custom", 1.0),
                ],
                3,
                300,
                circuit_stats=_STATS,
            ),
            [
                "cost",
                "├── CircuitSpecStage [circuit] → 2",
                "│   ├── n_qubits: 2",
                "│   ├── span: 1 .. 3",
                "│   └── ratio: 0.1235",
                "├── ParameterBindingStage [param_set] → ×3",
                "│   └── n_param_sets: 3",
                "├── MeasurementStage [obs_group] → ÷2",
                "├── Custom → 1",
                "├── Total (per evaluation, all 3 parameter sets): "
                "2 × 3 ÷ 2 = 3 circuits · 300 shots",
                "└── Summary: avg depth 4, width 2, 6 2q-gates total",
                "",
            ],
        ),
        (
            _hand_report(
                [
                    _stage("S", 14.0, "circuit"),
                    _stage("M", 1 / 2.8, "obs_group"),
                    _stage("T", 1.5, "twirl"),
                ],
                7,
                circuit_stats={
                    **_STATS,
                    "mean_depth": 5.5,
                    "max_depth": 7,
                    "max_width": 4,
                    "total_2q_gates": 1,
                },
            ),
            [
                "cost",
                "├── S [circuit] → 14",
                "├── M [obs_group] → ÷2.8",
                "├── T [twirl] → ×1.5",
                "├── Total (per evaluation): 7 circuits",
                "└── Summary: avg depth 5.5 (range 4-7), width 2-4, 1 2q-gate total",
                "",
            ],
        ),
        (
            _hand_report(
                [
                    _stage("S", 6.0, "circuit"),
                    _stage("ParameterBindingStage", 1.0, "param_set", n_param_sets=1),
                ],
                6,
            ),
            [
                "cost",
                "├── S [circuit] → 6",
                "├── ParameterBindingStage [param_set] → 1",
                "│   └── n_param_sets: 1",
                "└── Total (per evaluation): 6 circuits",
                "",
            ],
        ),
    ],
    ids=["factors-multiply-out", "factors-do-not-reconcile", "single-factor"],
)
def test_single_program_tree_renders_rows_total_and_summary(report, expected):
    assert _render_lines({"cost": report}) == expected


def test_compact_rows_carry_headline_scope_expression_and_tags():
    lines = _render_lines(
        {"p0": {"cost": _cost_report(), "sample": _sampled_once_report()}, "p1": {}},
        style="compact",
    )
    assert lines == [
        "Ensemble Dry Run  (2 programs)",
        "├── p0  (2q, 4 deep, 8 samples)",
        "│   ├── cost: 3 circuits · 300 shots  per evaluation, 2 × 3 ÷ 2",
        "│   └── sample: 3 circuits · 30 shots  once  sampled count",
        "├── p1",
        "│   └── (no preprocessors — 0 circuits)",
        "└── Ensemble total — per evaluation: 3 circuits · 300 shots; "
        "once: 3 circuits · 30 shots · widest 2q · deepest 4",
    ]


@pytest.mark.parametrize(
    "nested, expected",
    [
        (
            {"p0": {"sample": _sampled_once_report()}},
            "└── Ensemble total (once): 3 circuits · 30 shots",
        ),
        ({"p0": {}, "p1": {}}, "└── Ensemble total (per evaluation): 0 circuits"),
    ],
    ids=["one-time-only", "empty-programs"],
)
def test_ensemble_total_line_names_the_cadence_it_sums(nested, expected):
    assert _render_lines(nested, style="compact")[-1] == expected


def test_verbose_renders_a_tree_per_program():
    assert _render_lines({"p0": {"cost": _cost_report()}, "p1": {}}, "verbose") == [
        "Ensemble Dry Run  (2 programs)",
        "p0",
        "└── cost",
        "    ├── CircuitSpecStage [circuit] → 2",
        "    ├── ParameterBindingStage [param_set] → ×3",
        "    ├── MeasurementStage [obs_group] → ÷2",
        "    ├── Total (per evaluation): 2 × 3 ÷ 2 = 3 circuits · 300 shots",
        "    └── Summary: avg depth 4, width 2, 6 2q-gates total",
        "",
        "p1",
        "└── (no preprocessors — 0 circuits)",
        "",
        "Ensemble total (per evaluation): 3 circuits · 300 shots · widest 2q · deepest 4",
    ]


def test_grouped_merges_members_and_keeps_singletons_whole():
    cost = _cost_report()
    heavier = replace(
        cost,
        stages=(replace(cost.stages[0], factor=4.0), *cost.stages[1:]),
        circuit_stats={**_STATS, "max_depth": 9, "total_2q_gates": 40},
    )
    nested = {
        "p0": {"cost": cost},
        "p1": {"cost": cost},
        "p2": {"cost": heavier},
        "p3": {"cost": replace(cost, total_circuits=5)},
        "p4": {},
    }
    assert _render_lines(nested, "grouped") == [
        "Ensemble Dry Run  (5 programs)",
        "3 programs  (p0, p1, p2)",
        "├── cost",
        "│   ├── CircuitSpecStage [circuit] → 2",
        "│   │   └── factor_range: ×2 .. ×4",
        "│   ├── ParameterBindingStage [param_set] → ×3",
        "│   ├── MeasurementStage [obs_group] → ÷2",
        "│   ├── Total (per evaluation): 2 × 3 ÷ 2 = 3 circuits · 300 shots",
        "│   └── Summary: avg depth 4 (range 4-9), width 2, "
        "6-40 2q-gates total per program",
        "└── Subtotal (× 3) (per evaluation): 9 circuits · 900 shots",
        "",
        "1 program  (p3)",
        "├── cost",
        "│   ├── CircuitSpecStage [circuit] → 2",
        "│   ├── ParameterBindingStage [param_set] → ×3",
        "│   ├── MeasurementStage [obs_group] → ÷2",
        "│   ├── Total (per evaluation): 5 circuits · 300 shots",
        "│   └── Summary: avg depth 4, width 2, 6 2q-gates total",
        "└── Subtotal (× 1) (per evaluation): 5 circuits · 300 shots",
        "",
        "1 program  (p4)",
        "├── (no preprocessors — 0 circuits)",
        "└── Subtotal (× 1) (per evaluation): 0 circuits",
        "",
        "Ensemble total (per evaluation): 14 circuits · 1,200 shots · widest 2q · "
        "deepest 9",
    ]


@pytest.mark.parametrize(
    "n_programs, header, label, subtotal, total",
    [
        (
            1,
            "Ensemble Dry Run  (1 program)",
            "1 program  (p0)",
            "└── Subtotal (× 1) (per evaluation): 3 circuits · 300 shots",
            "Ensemble total (per evaluation): 3 circuits · 300 shots · widest 2q · "
            "deepest 4",
        ),
        (
            4,
            "Ensemble Dry Run  (4 programs)",
            "4 programs  (p0 … p3)",
            "└── Subtotal (× 4) (per evaluation): 12 circuits · 1,200 shots",
            "Ensemble total (per evaluation): 12 circuits · 1,200 shots · widest 2q · "
            "deepest 4",
        ),
    ],
    ids=["single-program", "middle-ids-elided"],
)
def test_grouped_identical_programs_share_one_per_program_tree(
    n_programs, header, label, subtotal, total
):
    """Every figure in the group node is per program; the member count is applied
    once, on the subtotal."""
    nested = {f"p{i}": {"cost": _cost_report()} for i in range(n_programs)}
    assert _render_lines(nested, "grouped") == [
        header,
        label,
        "├── cost",
        "│   ├── CircuitSpecStage [circuit] → 2",
        "│   ├── ParameterBindingStage [param_set] → ×3",
        "│   ├── MeasurementStage [obs_group] → ÷2",
        "│   ├── Total (per evaluation): 2 × 3 ÷ 2 = 3 circuits · 300 shots",
        "│   └── Summary: avg depth 4, width 2, 6 2q-gates total",
        subtotal,
        "",
        total,
    ]


def _differing_in(n_programs, **fields):
    return {
        f"p{i}": {
            "cost": replace(
                _cost_report(), **{key: value(i) for key, value in fields.items()}
            )
        }
        for i in range(n_programs)
    }


@pytest.mark.parametrize(
    "nested, banner",
    [
        (
            _differing_in(2, total_circuits=lambda i: 3 + i),
            "These 2 programs are all distinct (they differ in circuit count), so "
            "grouping would print one full tree each — showing the compact view "
            "instead.",
        ),
        (
            _differing_in(
                4, objective_fingerprint=lambda i: (("Z",), (float(i),), (0.0,))
            ),
            "These 4 programs are all distinct (they differ in the objective they "
            "optimise), so grouping would print one full tree each — showing the "
            "compact view instead.",
        ),
    ],
    ids=["two-programs-circuit-count", "four-programs-objective"],
)
def test_grouped_falls_back_when_nothing_groups(nested, banner):
    """Programs that share no signature would render one full tree each — more
    output than the compact rows grouping exists to replace. The reason given has
    to be the real one: a "different pipeline shape" claim would be contradicted
    by the rows underneath it."""
    lines = _render_lines(nested, "grouped")
    assert lines[0] == banner
    assert lines[1] == f"Ensemble Dry Run  ({len(nested)} programs)"
    assert not any("Subtotal" in line for line in lines)


_TRAIT_BASE = _hand_report([_stage("S", 2.0, "circuit")], 2)


@pytest.mark.parametrize(
    "other, expected",
    [
        (
            {"cost": _TRAIT_BASE, "sample": _TRAIT_BASE},
            "they differ in the routines they run",
        ),
        (
            {
                "cost": replace(_TRAIT_BASE, objective_fingerprint=1),
                "sample": _TRAIT_BASE,
            },
            "they differ in the routines they run and the objective they optimise",
        ),
        (
            {
                "cost": replace(
                    _TRAIT_BASE,
                    objective_fingerprint=1,
                    cadence=PipelineCadence.ONCE,
                    stages=(_stage("S", 2.0, "circuit", n_params=4),),
                )
            },
            "they differ in the objective they optimise, parameter count and cadence",
        ),
        (
            {
                "cost": replace(
                    _TRAIT_BASE,
                    total_circuits=3,
                    stages=(_stage("S", 2.0, "circuit", n_samples=9),),
                )
            },
            "they differ in batch size",
        ),
        (
            {"cost": replace(_TRAIT_BASE, total_circuits=3)},
            "they differ in circuit count",
        ),
    ],
    ids=[
        "routines",
        "routines-and-cause",
        "three-causes",
        "cause-not-its-consequence",
        "consequence-when-no-cause",
    ],
)
def test_distinguishing_trait_names_every_difference(other, expected):
    """A differing upstream trait also changes the counts it drives; naming the
    count would send the reader hunting in the wrong place, so a cause wins and
    the consequence is named only when there is no cause."""
    assert _distinguishing_trait({"p0": {"cost": _TRAIT_BASE}, "p1": other}) == expected


@pytest.mark.parametrize(
    "n_values, expected",
    [
        (3, "mixed (0 | 1 | 2)"),
        (4, "mixed (0 | 1 | 2 | … (+1 more))"),
        (5, "mixed (0 | 1 | 2 | … (+2 more))"),
    ],
)
def test_mixed_metadata_counts_only_values_beyond_three(n_values, expected):
    """Truncating the value list defeats the disclosure when a group is big enough
    to need it, so the count of hidden values is part of the row."""
    assert _shared_metadata([{"depth": d} for d in range(n_values)]) == {
        "depth": expected
    }


def test_a_field_missing_from_one_member_does_not_hide_later_fields():
    assert _shared_metadata([{"a": 1, "b": 2}, {"b": 2}]) == {"b": 2}


def test_format_dry_run_accepts_a_one_column_width():
    lines = _render_lines({"cost": _hand_report([_stage("S", 1.0)], 1)}, width=1)
    assert "".join(lines) == "cost"
    assert all(len(line) <= 1 for line in lines)
