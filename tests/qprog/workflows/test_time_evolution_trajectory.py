# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import warnings

import matplotlib.pyplot as plt
import pytest
from qiskit.circuit import ParameterExpression
from qiskit.converters import dag_to_circuit
from qiskit.quantum_info import SparsePauliOp

from divi.circuits.quepp import QuEPP
from divi.hamiltonians import ExactTrotterization, QDrift
from divi.pipeline import DiviPerformanceWarning
from divi.qprog import ReportingLevel, TimeEvolutionTrajectory
from divi.qprog.algorithms import TimeEvolution
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.ensemble import BatchConfig, BatchMode
from divi.qprog.initial_states import SuperpositionState
from divi.qprog.workflows import _time_evolution_trajectory as workflow

_PROB_TOL = 0.05

_H_Z0_Z1 = SparsePauliOp.from_sparse_list(
    [("Z", [0], 0.5), ("Z", [1], 0.3)], num_qubits=2
)
_Z0_2Q = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=2)


@pytest.fixture
def two_qubit_hamiltonian():
    return _H_Z0_Z1


class TestTimeEvolutionTrajectoryInit:
    def test_valid_initialization(self, two_qubit_hamiltonian, dummy_simulator):
        traj = TimeEvolutionTrajectory(
            hamiltonian=two_qubit_hamiltonian,
            time_points=[0.0, 0.5, 1.0],
            backend=dummy_simulator,
        )
        assert traj._time_points == [0.0, 0.5, 1.0]

    def test_empty_time_points_raises(self, two_qubit_hamiltonian, dummy_simulator):
        with pytest.raises(ValueError, match="must not be empty"):
            TimeEvolutionTrajectory(
                hamiltonian=two_qubit_hamiltonian,
                time_points=[],
                backend=dummy_simulator,
            )

    def test_duplicate_time_points_raises(self, two_qubit_hamiltonian, dummy_simulator):
        with pytest.raises(ValueError, match="must not contain duplicates"):
            TimeEvolutionTrajectory(
                hamiltonian=two_qubit_hamiltonian,
                time_points=[0.5, 1.0, 0.5],
                backend=dummy_simulator,
            )


class TestTimeEvolutionTrajectoryKwargForwarding:
    """``**kwargs`` reach every per-time-point ``TimeEvolution``."""

    def test_precision_forwarded_to_programs(
        self, two_qubit_hamiltonian, dummy_simulator
    ):
        traj = TimeEvolutionTrajectory(
            hamiltonian=two_qubit_hamiltonian,
            time_points=[0.1, 0.5, 1.0],
            backend=dummy_simulator,
            precision=4,
        )
        traj.create_programs()
        assert all(prog.precision == 4 for prog in traj.programs.values())

    def test_grouping_strategy_forwarded_to_programs(
        self, two_qubit_hamiltonian, dummy_simulator
    ):
        traj = TimeEvolutionTrajectory(
            hamiltonian=two_qubit_hamiltonian,
            time_points=[0.1, 0.5, 1.0],
            backend=dummy_simulator,
            observable=two_qubit_hamiltonian,
            grouping_strategy="wires",
        )
        traj.create_programs()
        assert all(
            prog._grouping_strategy == "wires" for prog in traj.programs.values()
        )

    def test_shot_distribution_forwarded_to_programs(
        self, two_qubit_hamiltonian, dummy_simulator
    ):
        traj = TimeEvolutionTrajectory(
            hamiltonian=two_qubit_hamiltonian,
            time_points=[0.1, 0.5, 1.0],
            backend=dummy_simulator,
            observable=two_qubit_hamiltonian,
            shot_distribution="weighted",
        )
        traj.create_programs()
        assert all(
            prog._shot_distribution == "weighted" for prog in traj.programs.values()
        )

    def test_trajectory_settings_reach_every_program(
        self, two_qubit_hamiltonian, dummy_simulator
    ):
        """Constructor settings reach each child; ``reporting_level`` stays here."""
        strategy = ExactTrotterization(keep_top_n=1)
        initial_state = SuperpositionState()
        traj = TimeEvolutionTrajectory(
            hamiltonian=two_qubit_hamiltonian,
            time_points=[0.1, 0.5],
            backend=dummy_simulator,
            trotterization_strategy=strategy,
            n_steps=3,
            order=2,
            initial_state=initial_state,
            seed=9,
            reporting_level="off",
        )

        traj.create_programs()

        assert traj.reporting_level == ReportingLevel.OFF
        for prog in traj.programs.values():
            assert prog._seed == 9
            assert prog.n_steps == 3
            assert prog.order == 2
            assert prog.initial_state is initial_state
            assert prog.trotterization_strategy == strategy
            assert prog.trotterization_strategy is not strategy

    def test_invalid_kwarg_rejected_at_program_construction(
        self, two_qubit_hamiltonian, dummy_simulator
    ):
        """``**kwargs`` are forwarded verbatim; invalid values surface when
        the first per-time-point program is constructed."""
        traj = TimeEvolutionTrajectory(
            hamiltonian=two_qubit_hamiltonian,
            time_points=[0.1, 0.5],
            backend=dummy_simulator,
            grouping_strategy="_backend_expval",
        )
        with pytest.raises(ValueError, match="Invalid grouping_strategy"):
            traj.create_programs()


class TestTimeEvolutionTrajectoryCreatePrograms:
    def test_creates_correct_number_of_programs(
        self, two_qubit_hamiltonian, dummy_simulator
    ):
        traj = TimeEvolutionTrajectory(
            hamiltonian=two_qubit_hamiltonian,
            time_points=[0.0, 0.5, 1.0],
            backend=dummy_simulator,
        )
        traj.create_programs()
        assert len(traj.programs) == 3

    def test_program_map_keys_contain_time_values(
        self, two_qubit_hamiltonian, dummy_simulator
    ):
        traj = TimeEvolutionTrajectory(
            hamiltonian=two_qubit_hamiltonian,
            time_points=[0.1, 0.5],
            backend=dummy_simulator,
        )
        traj.create_programs()
        assert "t=0.1" in traj.programs
        assert "t=0.5" in traj.programs
        assert all(
            not hasattr(program, "program_id") for program in traj.programs.values()
        )

    def test_programs_have_correct_time(self, two_qubit_hamiltonian, dummy_simulator):
        traj = TimeEvolutionTrajectory(
            hamiltonian=two_qubit_hamiltonian,
            time_points=[0.3, 0.7],
            backend=dummy_simulator,
        )
        traj.create_programs()
        assert traj.programs["t=0.3"].time == 0.3
        assert traj.programs["t=0.7"].time == 0.7

    def test_create_programs_twice_raises(self, two_qubit_hamiltonian, dummy_simulator):
        traj = TimeEvolutionTrajectory(
            hamiltonian=two_qubit_hamiltonian,
            time_points=[0.5],
            backend=dummy_simulator,
        )
        traj.create_programs()
        with pytest.raises(RuntimeError, match="Some programs already exist"):
            traj.create_programs()


class TestTimeEvolutionTrajectoryRun:
    def test_checkpointing_never_passes_vqa_arguments(
        self, two_qubit_hamiltonian, dummy_simulator, tmp_path, mocker
    ):
        traj = TimeEvolutionTrajectory(
            hamiltonian=two_qubit_hamiltonian,
            time_points=[0.5, 1.0],
            backend=dummy_simulator,
        )
        run_spy = mocker.spy(TimeEvolution, "run")

        traj.run(checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path))

        assert run_spy.call_count == 2
        assert all(
            "checkpoint_config" not in call.kwargs for call in run_spy.call_args_list
        )
        restored = TimeEvolutionTrajectory(
            hamiltonian=two_qubit_hamiltonian,
            time_points=[0.5, 1.0],
            backend=dummy_simulator,
        )
        restored.restore_state(tmp_path).run()
        assert restored.stop_reason.value == "complete"

    def test_run_probs_mode(self, two_qubit_hamiltonian, default_test_simulator):
        traj = TimeEvolutionTrajectory(
            hamiltonian=two_qubit_hamiltonian,
            time_points=[0.5, 1.0],
            backend=default_test_simulator,
        )
        traj.create_programs()
        traj.run()
        results = traj.aggregate_results()

        assert len(results) == 2
        assert 0.5 in results
        assert 1.0 in results
        for t, probs in results.items():
            assert isinstance(probs, dict)
            assert sum(probs.values()) == pytest.approx(1.0, abs=1e-12)
            # A diagonal Hamiltonian leaves |00> fixed.
            assert probs["00"] == pytest.approx(1.0, abs=1e-12)

    def test_run_expval_mode(self, two_qubit_hamiltonian, default_test_simulator):
        traj = TimeEvolutionTrajectory(
            hamiltonian=two_qubit_hamiltonian,
            time_points=[0.5, 1.0],
            observable=_Z0_2Q,
            backend=default_test_simulator,
        )
        traj.create_programs()
        traj.run()
        results = traj.aggregate_results()

        assert len(results) == 2
        for t, expval in results.items():
            assert isinstance(expval, float)
            assert -1.1 <= expval <= 1.1

    def test_aggregate_before_run_raises(self, two_qubit_hamiltonian, dummy_simulator):
        traj = TimeEvolutionTrajectory(
            hamiltonian=two_qubit_hamiltonian,
            time_points=[0.5],
            backend=dummy_simulator,
        )
        traj.create_programs()
        with pytest.raises(RuntimeError, match="no results"):
            traj.aggregate_results()

    def test_aggregate_without_create_raises(
        self, two_qubit_hamiltonian, dummy_simulator
    ):
        traj = TimeEvolutionTrajectory(
            hamiltonian=two_qubit_hamiltonian,
            time_points=[0.5],
            backend=dummy_simulator,
        )
        with pytest.raises(RuntimeError, match="No programs to aggregate"):
            traj.aggregate_results()

    def test_total_circuit_count(self, two_qubit_hamiltonian, default_test_simulator):
        traj = TimeEvolutionTrajectory(
            hamiltonian=two_qubit_hamiltonian,
            time_points=[0.5, 1.0, 1.5],
            backend=default_test_simulator,
        )
        traj.create_programs()
        traj.run()
        assert traj.total_circuit_count >= 3

    def test_results_ordered_by_time_points(
        self, two_qubit_hamiltonian, default_test_simulator
    ):
        time_points = [1.5, 0.5, 1.0]
        traj = TimeEvolutionTrajectory(
            hamiltonian=two_qubit_hamiltonian,
            time_points=time_points,
            backend=default_test_simulator,
        )
        traj.create_programs()
        traj.run()
        results = traj.aggregate_results()
        assert list(results.keys()) == time_points

    def test_reset_and_rerun(self, two_qubit_hamiltonian, default_test_simulator):
        traj = TimeEvolutionTrajectory(
            hamiltonian=two_qubit_hamiltonian,
            time_points=[0.5],
            backend=default_test_simulator,
        )
        traj.create_programs()
        traj.run()
        results1 = traj.aggregate_results()

        traj.reset()
        traj.create_programs()
        traj.run()
        results2 = traj.aggregate_results()

        assert 0.5 in results1
        assert 0.5 in results2

    def test_batch_submissions(self, two_qubit_hamiltonian, default_test_simulator):
        traj = TimeEvolutionTrajectory(
            hamiltonian=two_qubit_hamiltonian,
            time_points=[0.5, 1.0],
            backend=default_test_simulator,
        )
        traj.create_programs()
        traj.run(batch_config=BatchConfig())
        results = traj.aggregate_results()

        assert len(results) == 2
        for probs in results.values():
            assert isinstance(probs, dict)

    def test_batch_off_mode(self, two_qubit_hamiltonian, default_test_simulator):
        traj = TimeEvolutionTrajectory(
            hamiltonian=two_qubit_hamiltonian,
            time_points=[0.5, 1.0],
            backend=default_test_simulator,
        )
        traj.create_programs()
        traj.run(batch_config=BatchConfig(mode=BatchMode.OFF))
        results = traj.aggregate_results()

        assert len(results) == 2


@pytest.mark.e2e
class TestTimeEvolutionTrajectoryE2E:
    def test_x_rotation_trajectory(self, default_test_simulator):
        """H=X, |0⟩: at t=0 P(0)=1, at t=π/4 P(0)≈0.5, at t=π/2 P(1)=1."""
        traj = TimeEvolutionTrajectory(
            hamiltonian=SparsePauliOp("X"),
            time_points=[0.01, math.pi / 4, math.pi / 2],
            backend=default_test_simulator,
        )
        traj.create_programs()
        traj.run()
        results = traj.aggregate_results()

        # Near t=0: P(0) ≈ 1
        probs_t0 = results[0.01]
        assert probs_t0.get("0", 0.0) >= 1.0 - _PROB_TOL

        # t=π/4: roughly equal superposition
        probs_mid = results[math.pi / 4]
        assert abs(probs_mid.get("0", 0.0) - 0.5) < _PROB_TOL + 0.05

        # t=π/2: P(1) ≈ 1
        probs_end = results[math.pi / 2]
        assert probs_end.get("1", 0.0) >= 1.0 - _PROB_TOL

    def test_eigenstate_stays_constant(self, default_test_simulator):
        """H=Z₀+Z₁, |00⟩ is eigenstate: P(00)=1 at all times."""
        traj = TimeEvolutionTrajectory(
            hamiltonian=_H_Z0_Z1,
            time_points=[0.5, 1.0, 2.0],
            backend=default_test_simulator,
        )
        traj.create_programs()
        traj.run()
        results = traj.aggregate_results()

        for t, probs in results.items():
            assert probs.get("00", 0.0) >= 1.0 - _PROB_TOL


@pytest.fixture(scope="module")
def cache_test_hamiltonian():
    """Multi-term H exercising both Z and XX terms — enough structural
    variety to hit non-trivial decomposition."""
    return SparsePauliOp.from_sparse_list(
        [("Z", [0], 0.5), ("Z", [1], 0.3), ("XX", [0, 1], 0.2)], num_qubits=2
    )


def _build_meta_at(H, observable, time, backend, **program_kwargs):
    """Helper: instantiate a single-shot ``TimeEvolution`` and run its
    factory to obtain the un-cached ``MetaCircuit`` at ``time``."""
    prog = TimeEvolution(
        hamiltonian=H,
        time=time,
        observable=observable,
        backend=backend,
        **program_kwargs,
    )
    result = prog.trotterization_strategy.process_hamiltonian(H)
    return prog._meta_circuit_factory(result, ham_id=0)


def _assert_template_binds_to(template, t_param, fresh, t_value, atol):
    """The template bound at ``t_value`` has ``fresh``'s gates, wires and angles."""
    for (tag_t, dag_t), (tag_f, dag_f) in zip(
        template.circuit_bodies, fresh.circuit_bodies, strict=True
    ):
        assert tag_t == tag_f
        bound = dag_to_circuit(dag_t).assign_parameters({t_param: t_value})
        unbound = dag_to_circuit(dag_f)
        assert len(bound.data) == len(unbound.data)
        for inst_b, inst_f in zip(bound.data, unbound.data, strict=True):
            assert inst_b.operation.name == inst_f.operation.name
            assert tuple(bound.find_bit(q).index for q in inst_b.qubits) == tuple(
                unbound.find_bit(q).index for q in inst_f.qubits
            )
            for pb, pf in zip(
                inst_b.operation.params, inst_f.operation.params, strict=True
            ):
                assert abs(float(pb) - float(pf)) < atol


def _dag_signature(meta):
    """Per-gate signature: (op-name, wire-tuple, n-params)."""
    return [
        (
            node.op.name,
            tuple(dag.find_bit(q).index for q in node.qargs),
            len(node.op.params),
        )
        for _, dag in meta.circuit_bodies
        for node in dag.topological_op_nodes()
    ]


class TestCacheStructuralInvariant:
    """Phase 0 verification: the cache assumes that decomposition produces
    structurally identical DAGs across t, with rotation angles scaling
    linearly. These tests prove that invariant on the multi-term
    Hamiltonian we'd cache for."""

    @pytest.mark.parametrize(
        "times, build_kwargs",
        [
            pytest.param((1.0, 2.0, 0.5), {}, id="across_t"),
            pytest.param((1.0, 0.0), {}, id="t_zero_keeps_zero_angle_rotations"),
            pytest.param((1.0, 2.0), {"n_steps": 2}, id="n_steps_gt_1"),
        ],
    )
    def test_dag_topology_identical(
        self, cache_test_hamiltonian, dummy_simulator, times, build_kwargs
    ):
        """Decomposition yields the same DAG at every ``t``. Zero-angle rotations
        at ``t=0`` must not be pruned, or the cache would inflate gate counts
        there; with multiple Trotter steps the same symbolic ``t`` flows through
        more rotations and the topology must still hold."""
        first, *rest = [
            _dag_signature(
                _build_meta_at(
                    cache_test_hamiltonian, _Z0_2Q, t, dummy_simulator, **build_kwargs
                )
            )
            for t in times
        ]
        assert all(signature == first for signature in rest)

    def test_rotation_angles_scale_linearly(
        self, cache_test_hamiltonian, dummy_simulator
    ):
        H = cache_test_hamiltonian
        obs = _Z0_2Q
        m1 = _build_meta_at(H, obs, 1.0, dummy_simulator)
        m2 = _build_meta_at(H, obs, 2.0, dummy_simulator)

        params_t1 = [
            float(p)
            for _, dag in m1.circuit_bodies
            for node in dag.topological_op_nodes()
            for p in node.op.params
            if isinstance(p, (int, float))
        ]
        params_t2 = [
            float(p)
            for _, dag in m2.circuit_bodies
            for node in dag.topological_op_nodes()
            for p in node.op.params
            if isinstance(p, (int, float))
        ]
        assert len(params_t1) == len(params_t2)

        # Each numeric param must be either constant (ratio 1) or t-scaled
        # (ratio 2). No third class.
        for x1, x2 in zip(params_t1, params_t2, strict=True):
            if abs(x1) < 1e-12:
                # A param that's zero at t=1 must also be zero at t=2; a
                # non-zero t=2 value would imply a constant offset, which
                # the cache cannot represent as ``coef * t``.
                assert (
                    abs(x2) < 1e-12
                ), f"Param zero at t=1 but {x2} at t=2 — non-linear scaling"
                continue
            ratio = x2 / x1
            assert (
                abs(ratio - 1.0) < 1e-9 or abs(ratio - 2.0) < 1e-9
            ), f"Non-linear param scaling: t1={x1}, t2={x2}, ratio={ratio}"


class TestParametricTemplate:
    """Phase 0 verification: the trajectory's parametric template carries
    a single ``t`` Parameter and, when bound at ``t=t_test``, matches a
    fresh un-cached build at the same ``t_test``."""

    def test_template_carries_one_parameter(
        self, cache_test_hamiltonian, dummy_simulator
    ):
        traj = TimeEvolutionTrajectory(
            hamiltonian=cache_test_hamiltonian,
            time_points=[0.0, 0.25, 0.5, 0.75, 1.0],
            observable=_Z0_2Q,
            backend=dummy_simulator,
        )
        template, t_param = traj._maybe_build_template()

        assert template is not None
        assert t_param is not None
        assert template.parameters == (t_param,)
        n_pe = sum(
            1
            for _, dag in template.circuit_bodies
            for node in dag.topological_op_nodes()
            for p in node.op.params
            if isinstance(p, ParameterExpression) and t_param in p.parameters
        )
        assert n_pe > 0, "expected at least one gate with a t-parametric angle"

    def test_bound_template_matches_fresh_build(
        self, cache_test_hamiltonian, dummy_simulator
    ):
        H = cache_test_hamiltonian
        obs = _Z0_2Q
        traj = TimeEvolutionTrajectory(
            hamiltonian=H,
            time_points=[0.0, 0.25, 0.5, 0.75, 1.0],
            observable=obs,
            backend=dummy_simulator,
        )
        template, t_param = traj._maybe_build_template()
        assert template is not None and t_param is not None

        # Bind at a third time point and compare to a fresh build there.
        t_test = 0.7
        m_fresh = _build_meta_at(H, obs, t_test, dummy_simulator)

        _assert_template_binds_to(template, t_param, m_fresh, t_test, atol=1e-9)

    def test_shared_template_honours_program_settings(
        self, cache_test_hamiltonian, dummy_simulator
    ):
        """The shared template is built with the trajectory's initial state,
        term truncation and precision, not the defaults."""
        settings = dict(
            initial_state=SuperpositionState(),
            trotterization_strategy=ExactTrotterization(keep_top_n=1),
            precision=4,
        )
        traj = TimeEvolutionTrajectory(
            hamiltonian=cache_test_hamiltonian,
            time_points=[0.1 * (i + 1) for i in range(workflow._CACHE_MIN_TIME_POINTS)],
            observable=_Z0_2Q,
            backend=dummy_simulator,
            **settings,
        )
        traj.create_programs()
        programs = list(traj.programs.values())
        template, t_param = programs[0]._template_meta, programs[0]._template_param
        assert template is not None

        fresh = _build_meta_at(
            cache_test_hamiltonian, _Z0_2Q, 0.7, dummy_simulator, **settings
        )

        assert template.precision == 4
        _assert_template_binds_to(template, t_param, fresh, 0.7, atol=1e-9)


class TestCacheGating:
    """Cache opt-in/out logic: must engage when the trajectory benefits
    from it, must skip cleanly otherwise."""

    def test_cache_skipped_below_min_time_points(
        self, cache_test_hamiltonian, dummy_simulator
    ):
        traj = TimeEvolutionTrajectory(
            hamiltonian=cache_test_hamiltonian,
            time_points=[0.1, 0.2],
            backend=dummy_simulator,
        )
        traj.create_programs()
        for prog in traj.programs.values():
            assert prog._template_meta is None

    @pytest.mark.parametrize(
        "time_points, strategy",
        [
            (
                [0.1 * (i + 1) for i in range(workflow._CACHE_MIN_TIME_POINTS)],
                None,
            ),
            ([0.0, 0.25, 0.5, 0.75, 1.0], None),
            ([0.1, 0.2, 0.3, 0.4, 0.5], ExactTrotterization()),
        ],
        ids=["exact_threshold", "above_threshold", "explicit_exact_trotterization"],
    )
    def test_cache_engaged_at_or_above_threshold(
        self, cache_test_hamiltonian, dummy_simulator, time_points, strategy
    ):
        """Every program shares one template from ``_CACHE_MIN_TIME_POINTS``
        points on, for the implicit and the explicit ``ExactTrotterization``."""
        traj = TimeEvolutionTrajectory(
            hamiltonian=cache_test_hamiltonian,
            time_points=time_points,
            trotterization_strategy=strategy,
            backend=dummy_simulator,
        )
        traj.create_programs()
        assert len(traj.programs) == len(time_points)
        for prog in traj.programs.values():
            assert prog._template_meta is not None
            assert prog._template_param is not None
        templates = {id(p._template_meta) for p in traj.programs.values()}
        assert len(templates) == 1

    def test_cache_skipped_for_qdrift(self, cache_test_hamiltonian, dummy_simulator):
        """QDrift's per-program random sampling invalidates the structural
        invariant; the trajectory must opt out."""
        traj = TimeEvolutionTrajectory(
            hamiltonian=cache_test_hamiltonian,
            time_points=[0.1, 0.2, 0.3, 0.4, 0.5],
            trotterization_strategy=QDrift(sampling_budget=4),
            backend=dummy_simulator,
        )
        traj.create_programs()
        for prog in traj.programs.values():
            assert prog._template_meta is None

    def test_falls_back_when_probe_raises(
        self, cache_test_hamiltonian, dummy_simulator, monkeypatch, caplog
    ):
        """If the symbolic-time probe raises, the trajectory must log a
        warning and degrade to per-program construction (every program's
        ``_template_meta`` is ``None``)."""
        original = TimeEvolution._meta_circuit_factory

        # Raise only during the trajectory's symbolic probe (when
        # ``self.time`` is the qnp tensor); let the per-program numeric
        # path through so create_programs / construction can still complete.
        def maybe_boom(self, hamiltonian, ham_id):
            if not isinstance(self.time, (int, float)):
                raise RuntimeError("simulated probe failure")
            return original(self, hamiltonian, ham_id)

        monkeypatch.setattr(TimeEvolution, "_meta_circuit_factory", maybe_boom)
        traj = TimeEvolutionTrajectory(
            hamiltonian=cache_test_hamiltonian,
            time_points=[0.0, 0.25, 0.5, 0.75, 1.0],
            backend=dummy_simulator,
        )
        with caplog.at_level(logging.WARNING, logger=workflow.logger.name):
            traj.create_programs()

        assert [record.getMessage() for record in caplog.records] == [
            "TimeEvolutionTrajectory: parametric template build failed; "
            "falling back to per-program circuit construction."
        ]
        assert len(traj.programs) == 5
        for prog in traj.programs.values():
            assert prog._template_meta is None
            assert prog._template_param is None


@pytest.mark.parametrize(
    "n_steps, order",
    [(1, 1), (2, 1), (1, 2), (2, 2), (3, 2)],
)
def test_bound_angles_match_fresh_at_third_t(
    cache_test_hamiltonian, dummy_simulator, n_steps, order
):
    """Deterministic numeric parity: bound template angles must equal
    un-cached numeric angles bit-for-bit across Trotter configurations.

    This complements the stochastic regression test below: the
    shots-based equivalence check has shot noise above any sub-1e-8 angle
    drift, so we verify circuit-level numeric agreement here directly."""
    H = cache_test_hamiltonian
    obs = _Z0_2Q
    # Build the trajectory's parametric template at this (n_steps, order).
    traj = TimeEvolutionTrajectory(
        hamiltonian=H,
        time_points=[0.1 * (i + 1) for i in range(workflow._CACHE_MIN_TIME_POINTS)],
        observable=obs,
        backend=dummy_simulator,
        n_steps=n_steps,
        order=order,
    )
    template, t_param = traj._maybe_build_template()
    assert template is not None and t_param is not None

    t_test = 0.7
    m_fresh = _build_meta_at(
        H, obs, t_test, dummy_simulator, n_steps=n_steps, order=order
    )
    _assert_template_binds_to(template, t_param, m_fresh, t_test, atol=1e-12)


def test_cached_and_uncached_results_agree(
    cache_test_hamiltonian, default_test_simulator, monkeypatch
):
    """Phase 3 regression test: the trajectory's *results* must be
    identical whether the cache is on or off."""
    time_points = [0.0, 0.2, 0.4, 0.6, 0.8]

    # NOTE: ``traj_cached`` is built and run BEFORE the monkeypatch so
    # the cache engages for it. Do not hoist the patch above this block.
    traj_cached = TimeEvolutionTrajectory(
        hamiltonian=cache_test_hamiltonian,
        time_points=time_points,
        observable=_Z0_2Q,
        backend=default_test_simulator,
    )
    traj_cached.create_programs()
    assert all(p._template_meta is not None for p in traj_cached.programs.values())
    traj_cached.run()
    cached = traj_cached.aggregate_results()

    # Force the un-cached path by raising the threshold.
    monkeypatch.setattr(workflow, "_CACHE_MIN_TIME_POINTS", 999)
    traj_un = TimeEvolutionTrajectory(
        hamiltonian=cache_test_hamiltonian,
        time_points=time_points,
        observable=_Z0_2Q,
        backend=default_test_simulator,
    )
    traj_un.create_programs()
    assert all(p._template_meta is None for p in traj_un.programs.values())
    traj_un.run()
    uncached = traj_un.aggregate_results()

    for t in time_points:
        assert abs(cached[t] - uncached[t]) < 1e-9

    assert traj_cached.total_circuit_count == traj_un.total_circuit_count


def _templated_trajectory(backend, **kwargs):
    """A trajectory long enough to share one parametric template across times."""
    trajectory = TimeEvolutionTrajectory(
        SparsePauliOp.from_sparse_list(
            [("ZZ", [0, 1], 1.0), ("X", [1], 0.5)], num_qubits=2
        ),
        time_points=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
        backend=backend,
        observable=SparsePauliOp.from_sparse_list([("ZZ", [0, 1], 1.0)], num_qubits=2),
        **kwargs,
    )
    trajectory.create_programs()
    assert all(p._template_meta is not None for p in trajectory.programs.values())
    return trajectory


def test_templated_programs_bind_their_time_in_a_forced_dry_run(
    default_test_simulator,
):
    trajectory = _templated_trajectory(default_test_simulator)

    reports = trajectory.dry_run(force_circuit_generation=True)

    assert {r["evolution"].total_circuits for r in reports.values()} == {1}


@pytest.mark.usefixtures("suppress_quepp_warnings")
def test_templated_programs_run_exhaustive_quepp(default_test_simulator):
    """Exhaustive QuEPP binds parameters before it runs, and its performance
    warnings honour the program's opt-out."""
    trajectory = _templated_trajectory(
        default_test_simulator,
        qem_protocol=QuEPP(sampling="exhaustive"),
        suppress_performance_warnings=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", DiviPerformanceWarning)
        trajectory.run()

    assert all(math.isfinite(p.results) for p in trajectory.programs.values())


def _plotted_lines(dummy_simulator, mocker, observable, results):
    """Draw a trajectory whose programs hold ``results`` and return its lines."""
    show = mocker.patch("matplotlib.pyplot.show")
    traj = TimeEvolutionTrajectory(
        hamiltonian=_H_Z0_Z1,
        time_points=list(results),
        observable=observable,
        backend=dummy_simulator,
    )
    traj.create_programs()
    for t, value in results.items():
        traj.programs[f"t={t}"]._results["evolved_state_measurement"] = value

    plt.figure()
    try:
        traj.visualize_results()
        lines = [
            (list(line.get_xdata()), list(line.get_ydata()), line.get_label())
            for line in plt.gca().get_lines()
        ]
        has_legend = plt.gca().get_legend() is not None
    finally:
        plt.close("all")
    show.assert_called_once()
    return lines, has_legend


@pytest.mark.parametrize(
    "observable, results, expected_lines, expected_legend",
    [
        pytest.param(
            _Z0_2Q,
            {0.5: 0.25, 0.1: -0.75, 1.0: 0.5},
            [([0.5, 0.1, 1.0], [0.25, -0.75, 0.5])],
            None,
            id="single-observable",
        ),
        pytest.param(
            [_Z0_2Q, _H_Z0_Z1],
            {0.5: [0.25, 1.0], 0.1: [-0.75, 0.5], 1.0: [0.5, -1.0]},
            [
                ([0.5, 0.1, 1.0], [0.25, -0.75, 0.5]),
                ([0.5, 0.1, 1.0], [1.0, 0.5, -1.0]),
            ],
            ["Observable 0", "Observable 1"],
            id="one-line-per-observable",
        ),
    ],
)
def test_visualize_results_plots_each_time_against_its_expectation(
    dummy_simulator, mocker, observable, results, expected_lines, expected_legend
):
    lines, has_legend = _plotted_lines(dummy_simulator, mocker, observable, results)

    assert [(x, y) for x, y, _ in lines] == expected_lines
    assert has_legend is (expected_legend is not None)
    if expected_legend is not None:
        assert [label for *_, label in lines] == expected_legend
