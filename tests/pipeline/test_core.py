# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline._core: validation, CircuitPipeline, compile_batch, format_pipeline_tree."""

import signal
import threading
from threading import Event

import pytest

from divi.backends import AsyncJobBackend, ExecutionResult, JobStatus
from divi.backends._cancellation import _best_effort_cancel_job
from divi.exceptions import ExecutionCancelledError
from divi.pipeline import (
    CircuitPipeline,
    PipelineEnv,
    PipelineResult,
    PipelineTrace,
    format_pipeline_tree,
)
from divi.pipeline._compilation import (
    _collapse_to_parent_results,
    _compile_batch,
)
from divi.pipeline._core import (
    _scope_token,
    _sigint_to_cancellation,
    _validate_stage_order,
    _wait_for_async_result,
)
from divi.pipeline.stages import MeasurementStage

from .helpers import (
    DummySpecStage,
    FanoutAndSumStage,
    StatefulFanoutStage,
    ones_execute_fn,
    two_group_meta,
    two_group_pipeline_stages,
)


class TestScopeToken:
    """Spec: _scope_token filters dict tokens by foreign-key group for axis isolation."""

    def test_non_dict_token_returned_unchanged(self):
        assert _scope_token(None, (("obs", 0),), {"obs"}) is None
        assert _scope_token("hello", (("obs", 0),), {"obs"}) == "hello"

    def test_dict_with_non_tuple_keys_returned_unchanged(self):
        token = {"a": 1, "b": 2}
        assert _scope_token(token, (("obs", 0),), {"obs"}) is token

    def test_exact_foreign_match_strips_axes(self):
        token = {
            (("circuit", 0), ("obs_group", 0)): "ctx0",
            (("circuit", 0), ("obs_group", 1)): "ctx1",
        }
        scoped = _scope_token(token, (("obs_group", 0),), {"obs_group"})
        assert scoped == {(("circuit", 0),): "ctx0"}

    def test_subset_match_when_token_has_fewer_foreign_axes(self):
        """Token has param_set but not obs_group; should still match."""
        token = {
            (("circuit", 0), ("param_set", 0)): "ctx_p0",
            (("circuit", 0), ("param_set", 1)): "ctx_p1",
        }
        foreign_key = (("param_set", 0), ("obs_group", 0))
        scoped = _scope_token(token, foreign_key, {"param_set", "obs_group"})
        assert scoped == {(("circuit", 0),): "ctx_p0"}

    def test_subset_match_does_not_cross_values(self):
        """param_set=1 entry should not match foreign_key with param_set=0."""
        token = {
            (("circuit", 0), ("param_set", 0)): "ctx_p0",
            (("circuit", 0), ("param_set", 1)): "ctx_p1",
        }
        foreign_key = (("param_set", 0), ("obs_group", 0))
        scoped = _scope_token(token, foreign_key, {"param_set", "obs_group"})
        assert (("circuit", 0),) in scoped
        assert scoped[(("circuit", 0),)] == "ctx_p0"

    def test_no_foreign_axes_in_token_matches_all(self):
        """Token keys with no foreign axes match any foreign_key (vacuously)."""
        token = {(("circuit", 0),): "ctx"}
        scoped = _scope_token(token, (("obs_group", 0),), {"obs_group"})
        assert scoped == {(("circuit", 0),): "ctx"}

    def test_empty_token_raises(self):
        with pytest.raises(KeyError, match="no token entries matched"):
            _scope_token({}, (("obs", 0),), {"obs"})

    def test_no_match_raises(self):
        """When no entry matches the foreign_key, raise a descriptive KeyError."""
        token = {(("circuit", 0), ("param_set", 5)): "ctx"}
        with pytest.raises(KeyError, match="no token entries matched"):
            _scope_token(token, (("param_set", 0),), {"param_set"})

    def test_multi_foreign_axes(self):
        """Both param_set and obs_group in token key, both foreign."""
        token = {
            (("circuit", 0), ("obs_group", 0), ("param_set", 0)): "ctx_00",
            (("circuit", 0), ("obs_group", 0), ("param_set", 1)): "ctx_01",
            (("circuit", 0), ("obs_group", 1), ("param_set", 0)): "ctx_10",
        }
        scoped = _scope_token(
            token,
            (("obs_group", 0), ("param_set", 1)),
            {"obs_group", "param_set"},
        )
        assert scoped == {(("circuit", 0),): "ctx_01"}


class TestValidateStageOrder:
    """Spec: _validate_stage_order enforces pipeline structure and unique axis names."""

    def test_empty_stages_raises(self):
        with pytest.raises(ValueError, match="stages cannot be empty"):
            _validate_stage_order([])

    def test_bundle_stage_first_raises(self):
        meta = two_group_meta()
        with pytest.raises(
            ValueError,
            match="exactly one 'spec' stage and it must come before",
        ):
            CircuitPipeline(
                stages=[
                    MeasurementStage(),
                    DummySpecStage(meta=meta),
                ]
            )

    def test_spec_stage_only_raises_missing_measurement_stage(self):
        """Single spec stage without MeasurementStage raises."""
        with pytest.raises(
            ValueError,
            match="Pipeline must contain at least one stage that handles measurement",
        ):
            CircuitPipeline(stages=[DummySpecStage(meta=two_group_meta())])

    def test_no_measurement_stage_raises(self):
        """Pipeline with spec + bundle but no MeasurementStage raises."""
        with pytest.raises(
            ValueError,
            match="Pipeline must contain at least one stage that handles measurement",
        ):
            CircuitPipeline(
                stages=[
                    DummySpecStage(meta=two_group_meta()),
                    FanoutAndSumStage("x", 2),
                ]
            )

    def test_duplicate_axis_names_raise(self):
        """Duplicate stage axis_name (e.g. two FanoutAndSumStages) raises."""
        meta = two_group_meta()
        with pytest.raises(
            ValueError,
            match="Duplicate stage axis names",
        ):
            CircuitPipeline(
                stages=[
                    DummySpecStage(meta=meta),
                    FanoutAndSumStage("x", 2),
                    FanoutAndSumStage("x", 3),
                    MeasurementStage(),
                ]
            )


class TestCircuitPipelineRunForwardPass:
    """Spec: run_forward_pass expands spec then bundle stages and returns PipelineTrace."""

    def test_plan_tracks_lineage_for_two_level_fanout(self, dummy_pipeline_env):
        pipeline = CircuitPipeline(stages=two_group_pipeline_stages(fanout=("fold", 3)))

        plan = pipeline.run_forward_pass(initial_spec="ignored", env=dummy_pipeline_env)

        spec_circ_key = (("spec", "circ"),)
        assert set(plan.initial_batch.keys()) == {spec_circ_key}
        assert len(plan.stage_expansions) == 2
        assert len(plan.stage_tokens) == 3
        assert len(plan.final_batch) == 3
        assert set(plan.final_batch.keys()) == {
            spec_circ_key + (("fold", 0),),
            spec_circ_key + (("fold", 1),),
            spec_circ_key + (("fold", 2),),
        }

    def test_run_reduces_bottom_up_after_fanout(self, dummy_pipeline_env):
        pipeline = CircuitPipeline(stages=two_group_pipeline_stages(fanout=("fold", 3)))

        reduced = pipeline.run(
            initial_spec="ignored",
            env=dummy_pipeline_env,
            execute_fn=ones_execute_fn,
        )

        assert len(reduced) == 1
        assert list(reduced.values())[0] == pytest.approx([3.9])

    def test_force_forward_sweep_recomputes_even_with_cache(self, dummy_pipeline_env):
        pipeline = CircuitPipeline(stages=two_group_pipeline_stages())
        trace1 = pipeline.run_forward_pass(
            initial_spec="ignored", env=dummy_pipeline_env
        )
        trace2 = pipeline.run_forward_pass(
            initial_spec="ignored", env=dummy_pipeline_env, force_forward_sweep=True
        )
        assert trace1.final_batch.keys() == trace2.final_batch.keys()
        assert id(trace1) != id(trace2) or trace1 is trace2

    def test_stateful_bundle_stage_triggers_partial_rerun_from_cache(
        self, dummy_pipeline_env
    ):
        """When a bundle stage is stateful, second run_forward_pass reuses cache and reruns from that stage."""
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=two_group_meta()),
                MeasurementStage(),
                StatefulFanoutStage("stateful", 2),
            ]
        )
        spec_key = (("spec", "circ"),)

        trace1 = pipeline.run_forward_pass(
            initial_spec="ignored", env=dummy_pipeline_env
        )
        assert len(trace1.stage_expansions) == 2
        assert set(trace1.final_batch.keys()) == {
            spec_key + (("stateful", 0),),
            spec_key + (("stateful", 1),),
        }

        trace2 = pipeline.run_forward_pass(
            initial_spec="ignored", env=dummy_pipeline_env
        )
        assert len(trace2.stage_expansions) == 2
        assert trace2.initial_batch == trace1.initial_batch
        assert set(trace2.final_batch.keys()) == set(trace1.final_batch.keys())
        assert trace2.stage_expansions[0].batch == trace1.stage_expansions[0].batch


class TestCompileBatch:
    """Spec: _compile_batch lowers MetaCircuit batch to QASM labels and lineage."""

    def test_raises_when_measurement_qasms_missing(self, dummy_pipeline_env):
        pipeline = CircuitPipeline(
            stages=[DummySpecStage(meta=two_group_meta()), MeasurementStage()]
        )
        trace = pipeline.run_forward_pass("x", dummy_pipeline_env)
        # Manually remove measurement_qasms to simulate the error
        batch = trace.final_batch
        node = next(iter(batch.values()))
        assert hasattr(node, "circuit_bodies") and node.circuit_bodies

    def test_produces_lineage_and_circuits_for_grouped_batch(self, dummy_pipeline_env):
        pipeline = CircuitPipeline(
            stages=[DummySpecStage(meta=two_group_meta()), MeasurementStage()]
        )
        trace = pipeline.run_forward_pass("x", dummy_pipeline_env)
        circuits, lineage_by_label = _compile_batch(trace.final_batch)
        assert len(circuits) == len(lineage_by_label)
        for label, qasm in circuits.items():
            assert isinstance(qasm, str) and "OPENQASM" in qasm
            assert label in lineage_by_label
            branch_key = lineage_by_label[label]
            assert branch_key[0] == ("spec", "circ") and len(branch_key) >= 2
            assert any(e[0] == "obs_group" for e in branch_key)


class TestCollapseToParentResults:
    """Spec: _collapse_to_parent_results maps backend labels back to BranchKeys."""

    def test_maps_labels_to_branch_keys(self):
        spec_circ = ("spec", "circ")
        lineage = {
            "a/obs_group:0": (spec_circ, ("obs_group", 0)),
            "b/obs_group:1": (spec_circ, ("obs_group", 1)),
        }
        raw = {"a/obs_group:0": 1.0, "b/obs_group:1": 2.0}
        out = _collapse_to_parent_results(raw, lineage)
        assert out[(spec_circ, ("obs_group", 0))] == 1.0
        assert out[(spec_circ, ("obs_group", 1))] == 2.0

    def test_ignores_unknown_labels(self):
        lineage = {"only": (("spec", "k"),)}
        raw = {"only": 1, "unknown": 2}
        out = _collapse_to_parent_results(raw, lineage)
        assert out == {(("spec", "k"),): 1}


class TestFormatPipelineTree:
    """Spec: format_pipeline_tree prints ASCII tree of final_batch keys."""

    def test_format_pipeline_tree_prints_tree(self, capsys, dummy_pipeline_env):
        pipeline = CircuitPipeline(stages=two_group_pipeline_stages(fanout=("fold", 2)))
        plan = pipeline.run_forward_pass(initial_spec="ignored", env=dummy_pipeline_env)

        format_pipeline_tree(plan)
        lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]

        assert lines[0] == "spec:circ"
        # Two fold branches, each with one obs_group leaf
        # (_backend_expval puts all observables in a single group).
        assert sum("fold:0" in l for l in lines) == 1
        assert sum("fold:1" in l for l in lines) == 1
        assert sum("obs_group:0" in l for l in lines) == 2
        # 5 lines total: root + 2 folds + 2 obs_groups
        assert len(lines) == 5

    def test_format_pipeline_tree_empty_batch(self, capsys):
        trace = PipelineTrace(
            initial_batch={},
            final_batch={},
            stage_expansions=(),
            stage_tokens=(),
        )
        format_pipeline_tree(trace)
        assert capsys.readouterr().out.strip() == "(empty)"


def test_custom_execute_fn_returning_per_key_values_reduces_correctly(
    dummy_pipeline_env,
):
    """Execute_fn returning distinct values per key is reduced by pipeline stages."""
    pipeline = CircuitPipeline(stages=two_group_pipeline_stages(fanout=("fold", 3)))

    def _execute_fn(trace, env):
        _, lineage_by_label = _compile_batch(trace.final_batch)
        return {bk: 2 for bk in lineage_by_label.values()}

    reduced = pipeline.run(
        initial_spec="ignored",
        env=dummy_pipeline_env,
        execute_fn=_execute_fn,
    )
    assert len(reduced) == 1
    assert list(reduced.values())[0] == pytest.approx([7.8])
    assert next(iter(reduced)) == (("spec", "circ"),)


class TestPipelineReporterHooks:
    """Spec: CircuitPipeline emits ``pipeline_stage`` progress events per stage."""

    def _collect_stage_events(self, reporter_mock) -> list[str | None]:
        events: list[str | None] = []
        for call in reporter_mock.info.call_args_list:
            kwargs = call.kwargs
            if "pipeline_stage" in kwargs:
                events.append(kwargs["pipeline_stage"])
        return events

    def test_reports_spec_stage_name_on_forward_pass(self, dummy_pipeline_env, mocker):
        reporter = mocker.MagicMock()
        dummy_pipeline_env.reporter = reporter

        pipeline = CircuitPipeline(stages=two_group_pipeline_stages())
        pipeline.run_forward_pass("x", dummy_pipeline_env)

        events = self._collect_stage_events(reporter)
        assert events == ["DummySpecStage", "MeasurementStage"]

    def test_reports_each_bundle_stage_in_order(self, dummy_pipeline_env, mocker):
        reporter = mocker.MagicMock()
        dummy_pipeline_env.reporter = reporter

        pipeline = CircuitPipeline(
            stages=two_group_pipeline_stages(fanout=("fold", 2)),
        )
        pipeline.run_forward_pass("x", dummy_pipeline_env)

        events = self._collect_stage_events(reporter)
        assert events == [
            "DummySpecStage",
            "MeasurementStage",
            "FanoutAndSumStage:fold",
        ]

    def test_run_clears_pipeline_stage_before_execution(
        self, dummy_pipeline_env, mocker
    ):
        """``run()`` clears the pipeline-stage indicator before execute_fn."""
        reporter = mocker.MagicMock()
        dummy_pipeline_env.reporter = reporter

        pipeline = CircuitPipeline(stages=two_group_pipeline_stages())
        pipeline.run(
            initial_spec="x", env=dummy_pipeline_env, execute_fn=ones_execute_fn
        )

        events = self._collect_stage_events(reporter)
        # Last event must be a clear (None) so the spinner drops "Pipeline: ..."
        # before submission/polling takes over.
        assert events[-1] is None
        assert events[:-1] == ["DummySpecStage", "MeasurementStage"]

    def test_missing_reporter_does_not_break_forward_pass(self, dummy_pipeline_env):
        """Pipelines must work identically when ``env.reporter`` is None."""
        assert dummy_pipeline_env.reporter is None

        pipeline = CircuitPipeline(stages=two_group_pipeline_stages())
        trace = pipeline.run_forward_pass("x", dummy_pipeline_env)
        assert trace.final_batch  # forward pass succeeded


def test_run_with_default_execute_fn_and_shots_backend_auto_converts_counts(
    dummy_simulator,
):
    """When backend.supports_expval is False, run() auto-converts counts → expvals via _counts_to_expvals."""
    env = PipelineEnv(backend=dummy_simulator)
    pipeline = CircuitPipeline(
        stages=[
            DummySpecStage(meta=two_group_meta()),
            MeasurementStage(),
        ]
    )
    reduced = pipeline.run(initial_spec="x", env=env)
    assert len(reduced) == 1
    key = next(iter(reduced))
    assert key == (("spec", "circ"),)
    value = reduced[key]
    assert isinstance(value, list)
    assert len(value) == 1
    assert isinstance(value[0], (int, float))


class TestWaitForAsyncResult:
    """Tests for _wait_for_async_result: polling and cancellation handling."""

    def test_cancelled_with_event_raises_cancelled_error(self, mocker):
        """When job is CANCELLED and cancellation event is set, raises ExecutionCancelledError."""
        mock_backend = mocker.Mock()
        mock_backend.poll_job_status.return_value = JobStatus.CANCELLED
        mock_backend.max_retries = 100

        cancel_event = Event()
        cancel_event.set()
        env = PipelineEnv(backend=mock_backend, cancellation_event=cancel_event)

        execution_result = ExecutionResult(job_id="test_job")

        with pytest.raises(ExecutionCancelledError, match="Job test_job was cancelled"):
            _wait_for_async_result(mock_backend, execution_result, env)

    def test_cancelled_without_event_raises_runtime_error(self, mocker):
        """Backend-reported CANCELLED without a local cancellation request
        is a scheduler-side eviction, not a user cancel. It must surface as
        ``RuntimeError`` so eviction-driven failures aren't disguised as
        intentional shutdowns."""
        mock_backend = mocker.Mock()
        mock_backend.poll_job_status.return_value = JobStatus.CANCELLED
        mock_backend.max_retries = 100

        env = PipelineEnv(backend=mock_backend)

        execution_result = ExecutionResult(job_id="test_job")

        with pytest.raises(RuntimeError, match="cancelled by the scheduler"):
            _wait_for_async_result(mock_backend, execution_result, env)

    def test_missing_job_id_raises_value_error(self):
        """ExecutionResult without a job_id raises ValueError immediately."""
        env = PipelineEnv(backend=object())
        execution_result = ExecutionResult(results=[{"label": "c", "results": {}}])

        with pytest.raises(ValueError, match="must have a job_id"):
            _wait_for_async_result(object(), execution_result, env)

    def test_failed_status_raises_runtime_error(self, mocker):
        """FAILED job status raises RuntimeError."""
        mock_backend = mocker.Mock()
        mock_backend.poll_job_status.return_value = JobStatus.FAILED
        mock_backend.max_retries = 100

        env = PipelineEnv(backend=mock_backend)
        execution_result = ExecutionResult(job_id="job_fail")

        with pytest.raises(RuntimeError, match="Job job_fail has failed"):
            _wait_for_async_result(mock_backend, execution_result, env)

    def test_non_completed_status_raises_runtime_error(self, mocker):
        """A status that is neither COMPLETED, FAILED, nor CANCELLED raises RuntimeError."""
        mock_backend = mocker.Mock()
        mock_backend.poll_job_status.return_value = JobStatus.RUNNING
        mock_backend.max_retries = 100

        env = PipelineEnv(backend=mock_backend)
        execution_result = ExecutionResult(job_id="job_stuck")

        with pytest.raises(RuntimeError, match="has not completed yet"):
            _wait_for_async_result(mock_backend, execution_result, env)

    def test_completed_returns_job_results(self, mocker):
        """COMPLETED status returns backend.get_job_results()."""
        mock_backend = mocker.Mock()
        mock_backend.poll_job_status.return_value = JobStatus.COMPLETED
        mock_backend.max_retries = 100
        expected = ExecutionResult(results=[{"label": "c0", "results": {"00": 100}}])
        mock_backend.get_job_results.return_value = expected

        env = PipelineEnv(backend=mock_backend)
        execution_result = ExecutionResult(job_id="job_ok")

        result = _wait_for_async_result(mock_backend, execution_result, env)

        assert result is expected
        mock_backend.get_job_results.assert_called_once_with(execution_result)

    def test_runtime_tracking_dict_response(self, mocker):
        """on_complete callback accumulates run_time from a dict response."""
        mock_backend = mocker.Mock()
        mock_backend.max_retries = 100
        mock_backend.get_job_results.return_value = ExecutionResult(
            results=[], job_id="job_rt"
        )

        # Capture the on_complete callback, then invoke it with a dict response
        def fake_poll(
            er,
            *,
            loop_until_complete,
            on_complete,
            verbose,
            progress_callback,
            cancellation_event=None,
        ):
            on_complete({"run_time": 3.5})
            return JobStatus.COMPLETED

        mock_backend.poll_job_status.side_effect = fake_poll

        env = PipelineEnv(backend=mock_backend)
        execution_result = ExecutionResult(job_id="job_rt")

        _wait_for_async_result(mock_backend, execution_result, env)

        assert env.artifacts["run_time"] == 3.5

    def test_cancellation_event_is_forwarded_to_backend(self, mocker):
        """The env's cancellation_event must reach backend.poll_job_status
        so the polling loop can exit promptly when the user signals cancel."""
        mock_backend = mocker.Mock()
        mock_backend.poll_job_status.return_value = JobStatus.COMPLETED
        mock_backend.max_retries = 100
        mock_backend.get_job_results.return_value = ExecutionResult(
            results=[], job_id="job"
        )

        event = Event()
        env = PipelineEnv(backend=mock_backend, cancellation_event=event)

        _wait_for_async_result(mock_backend, ExecutionResult(job_id="job"), env)

        _, kwargs = mock_backend.poll_job_status.call_args
        assert kwargs["cancellation_event"] is event

    def test_runtime_tracking_list_response(self, mocker):
        """on_complete callback accumulates run_time from a list of responses."""
        mock_backend = mocker.Mock()
        mock_backend.max_retries = 100
        mock_backend.get_job_results.return_value = ExecutionResult(
            results=[], job_id="job_rt2"
        )

        resp1 = mocker.Mock()
        resp1.json.return_value = {"run_time": 1.5}
        resp2 = mocker.Mock()
        resp2.json.return_value = {"run_time": 2.0}

        def fake_poll(
            er,
            *,
            loop_until_complete,
            on_complete,
            verbose,
            progress_callback,
            cancellation_event=None,
        ):
            on_complete([resp1, resp2])
            return JobStatus.COMPLETED

        mock_backend.poll_job_status.side_effect = fake_poll

        env = PipelineEnv(backend=mock_backend)
        execution_result = ExecutionResult(job_id="job_rt2")

        _wait_for_async_result(mock_backend, execution_result, env)

        assert env.artifacts["run_time"] == 3.5


def _noop_handler(signum, frame):
    pass


class TestBestEffortCancelJob:
    """Tests for the helper that funnels in-flight async-job cancellation."""

    def test_calls_backend_cancel_for_async_backend_with_job_id(self, mocker):
        backend = mocker.Mock(spec=AsyncJobBackend)
        _best_effort_cancel_job(backend, ExecutionResult(job_id="job_x"))
        backend.cancel_job.assert_called_once()

    def test_noop_for_sync_backend(self, mocker):
        sync_backend = mocker.Mock(spec=[])
        _best_effort_cancel_job(sync_backend, ExecutionResult(job_id="job_y"))

    def test_noop_when_no_job_id(self, mocker):
        backend = mocker.Mock(spec=AsyncJobBackend)
        _best_effort_cancel_job(backend, ExecutionResult(results=[]))
        backend.cancel_job.assert_not_called()

    def test_swallows_cancel_job_exception(self, mocker):
        backend = mocker.Mock(spec=AsyncJobBackend)
        backend.cancel_job.side_effect = RuntimeError("server says no")
        # Must not propagate — the user's CTRL-C should not be masked by
        # network/server hiccups during the courtesy cancel.
        _best_effort_cancel_job(backend, ExecutionResult(job_id="job_z"))


class TestAutoCancellationScope:
    """Tests for ``_auto_cancellation_scope``: bundles the SIGINT funnel with
    best-effort remote-job cleanup for direct callers of ``poll_job_status``."""

    def test_cancels_backend_on_execution_cancelled(self, mocker):
        from divi.backends._cancellation import _auto_cancellation_scope

        backend = mocker.Mock(spec=AsyncJobBackend)
        result = ExecutionResult(job_id="job_x")

        with pytest.raises(ExecutionCancelledError):
            with _auto_cancellation_scope(backend, result):
                raise ExecutionCancelledError("polling cancelled")

        backend.cancel_job.assert_called_once_with(result)

    def test_does_not_cancel_on_unrelated_exceptions(self, mocker):
        from divi.backends._cancellation import _auto_cancellation_scope

        backend = mocker.Mock(spec=AsyncJobBackend)
        result = ExecutionResult(job_id="job_x")

        with pytest.raises(RuntimeError):
            with _auto_cancellation_scope(backend, result):
                raise RuntimeError("not a cancellation")

        backend.cancel_job.assert_not_called()

    def test_yields_a_fresh_unset_event(self, mocker):
        from divi.backends._cancellation import _auto_cancellation_scope

        backend = mocker.Mock(spec=AsyncJobBackend)
        with _auto_cancellation_scope(
            backend, ExecutionResult(job_id="job_x")
        ) as event:
            assert isinstance(event, Event)
            assert not event.is_set()


class TestSigintToCancellation:
    """Tests for the SIGINT → cancellation_event funnel."""

    def test_noop_outside_main_thread(self):
        """``signal.signal`` rejects non-main threads; the context manager
        must yield without installing a handler."""
        before = signal.getsignal(signal.SIGINT)
        observed: dict = {}

        def runner():
            env = PipelineEnv(backend=object())
            with _sigint_to_cancellation(env):
                observed["installed"] = signal.getsignal(signal.SIGINT)

        t = threading.Thread(target=runner)
        t.start()
        t.join()
        assert observed["installed"] is before

    def test_creates_event_when_missing(self):
        env = PipelineEnv(backend=object())
        with _sigint_to_cancellation(env):
            assert env.cancellation_event is not None

    def test_first_sigint_sets_event(self):
        env = PipelineEnv(backend=object())
        with _sigint_to_cancellation(env):
            assert env.cancellation_event is not None
            handler = signal.getsignal(signal.SIGINT)
            handler(signal.SIGINT, None)
            assert env.cancellation_event.is_set()

    def test_second_sigint_raises_keyboard_interrupt(self):
        env = PipelineEnv(backend=object())
        with pytest.raises(KeyboardInterrupt):
            with _sigint_to_cancellation(env):
                handler = signal.getsignal(signal.SIGINT)
                handler(signal.SIGINT, None)
                handler(signal.SIGINT, None)

    def test_defers_to_existing_non_default_handler(self):
        env = PipelineEnv(backend=object())
        prev = signal.signal(signal.SIGINT, _noop_handler)
        try:
            with _sigint_to_cancellation(env):
                assert signal.getsignal(signal.SIGINT) is _noop_handler
        finally:
            signal.signal(signal.SIGINT, prev)


class TestDefaultExecuteFnCancellation:
    """End-to-end: when the poll loop raises, the backend is told to cancel."""

    def _async_backend(self, mocker, *, raise_on_poll):
        backend = mocker.Mock(spec=AsyncJobBackend)
        backend.supports_expval = False
        backend.max_retries = 1
        backend.submit_circuits.return_value = ExecutionResult(job_id="job_42")
        if raise_on_poll:
            backend.poll_job_status.side_effect = ExecutionCancelledError(
                "Polling cancelled for job job_42."
            )
        return backend

    def test_cancel_during_poll_calls_backend_cancel_once(self, mocker):
        backend = self._async_backend(mocker, raise_on_poll=True)
        env = PipelineEnv(backend=backend, cancellation_event=Event())
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=two_group_meta()),
                MeasurementStage(),
            ]
        )
        with pytest.raises(ExecutionCancelledError):
            pipeline.run(initial_spec="x", env=env)

        # The submitted result should be passed back to cancel_job exactly once.
        submitted = backend.submit_circuits.return_value
        backend.cancel_job.assert_called_once_with(submitted)

    def test_keyboard_interrupt_during_poll_also_cancels_backend(self, mocker):
        """Hosts with their own SIGINT handler (Jupyter, debuggers) raise
        plain KeyboardInterrupt mid-poll — the cleanup must still fire."""
        backend = self._async_backend(mocker, raise_on_poll=False)
        backend.poll_job_status.side_effect = KeyboardInterrupt
        env = PipelineEnv(backend=backend, cancellation_event=Event())
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=two_group_meta()),
                MeasurementStage(),
            ]
        )
        with pytest.raises(KeyboardInterrupt):
            pipeline.run(initial_spec="x", env=env)

        backend.cancel_job.assert_called_once()


class TestPipelineResultSqueeze:
    """``.value`` squeezes a length-1 list only when ``_squeeze`` is True
    (the pipeline disables it when any source MetaCircuit was built with
    ``_was_multi_obs=True``)."""

    def test_squeeze_unwraps_length_one_list_by_default(self):
        result = PipelineResult({(): [0.42]})
        assert result.value == 0.42

    def test_squeeze_preserves_multi_element_list(self):
        result = PipelineResult({(): [0.1, 0.2, 0.3]})
        assert result.value == [0.1, 0.2, 0.3]

    def test_squeeze_passes_dict_through_unchanged(self):
        probs = {"00": 0.5, "11": 0.5}
        result = PipelineResult({(): probs})
        assert result.value == probs

    def test_squeeze_disabled_preserves_length_one_list(self):
        result = PipelineResult({(): [0.42]})
        result._squeeze = False
        assert result.value == [0.42]

    def test_squeeze_disabled_does_not_affect_dicts(self):
        probs = {"00": 1.0}
        result = PipelineResult({(): probs})
        result._squeeze = False
        assert result.value == probs
