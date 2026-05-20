# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import math

import pennylane as qp
import pytest
from qiskit.circuit import ParameterExpression
from qiskit.converters import dag_to_circuit

from divi.hamiltonians import ExactTrotterization, QDrift, to_spo
from divi.qprog import TimeEvolutionTrajectory
from divi.qprog.algorithms import TimeEvolution
from divi.qprog.ensemble import BatchConfig, BatchMode
from divi.qprog.workflows import _time_evolution_trajectory as workflow

_PROB_TOL = 0.05


@pytest.fixture
def two_qubit_hamiltonian():
    return 0.5 * qp.PauliZ(0) + 0.3 * qp.PauliZ(1)


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

    def test_program_ids_contain_time_values(
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
    def test_run_probs_mode(self, two_qubit_hamiltonian, default_test_simulator):
        traj = TimeEvolutionTrajectory(
            hamiltonian=two_qubit_hamiltonian,
            time_points=[0.5, 1.0],
            backend=default_test_simulator,
        )
        traj.create_programs()
        traj.run(blocking=True)
        results = traj.aggregate_results()

        assert len(results) == 2
        assert 0.5 in results
        assert 1.0 in results
        for t, probs in results.items():
            assert isinstance(probs, dict)
            assert abs(sum(probs.values()) - 1.0) < 0.1

    def test_run_expval_mode(self, two_qubit_hamiltonian, default_test_simulator):
        traj = TimeEvolutionTrajectory(
            hamiltonian=two_qubit_hamiltonian,
            time_points=[0.5, 1.0],
            observable=qp.PauliZ(0),
            backend=default_test_simulator,
        )
        traj.create_programs()
        traj.run(blocking=True)
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
        traj.run(blocking=True)
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
        traj.run(blocking=True)
        results = traj.aggregate_results()
        assert list(results.keys()) == time_points

    def test_reset_and_rerun(self, two_qubit_hamiltonian, default_test_simulator):
        traj = TimeEvolutionTrajectory(
            hamiltonian=two_qubit_hamiltonian,
            time_points=[0.5],
            backend=default_test_simulator,
        )
        traj.create_programs()
        traj.run(blocking=True)
        results1 = traj.aggregate_results()

        traj.reset()
        traj.create_programs()
        traj.run(blocking=True)
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
        traj.run(blocking=True, batch_config=BatchConfig())
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
        traj.run(blocking=True, batch_config=BatchConfig(mode=BatchMode.OFF))
        results = traj.aggregate_results()

        assert len(results) == 2


@pytest.mark.e2e
class TestTimeEvolutionTrajectoryE2E:
    def test_x_rotation_trajectory(self, default_test_simulator):
        """H=X, |0⟩: at t=0 P(0)=1, at t=π/4 P(0)≈0.5, at t=π/2 P(1)=1."""
        traj = TimeEvolutionTrajectory(
            hamiltonian=qp.PauliX(0),
            time_points=[0.01, math.pi / 4, math.pi / 2],
            backend=default_test_simulator,
        )
        traj.create_programs()
        traj.run(blocking=True)
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
            hamiltonian=0.5 * qp.PauliZ(0) + 0.3 * qp.PauliZ(1),
            time_points=[0.5, 1.0, 2.0],
            backend=default_test_simulator,
        )
        traj.create_programs()
        traj.run(blocking=True)
        results = traj.aggregate_results()

        for t, probs in results.items():
            assert probs.get("00", 0.0) >= 1.0 - _PROB_TOL


@pytest.fixture(scope="module")
def cache_test_hamiltonian():
    """Multi-term H exercising both Z and XX terms — enough structural
    variety to hit non-trivial decomposition."""
    return 0.5 * qp.PauliZ(0) + 0.3 * qp.PauliZ(1) + 0.2 * (qp.PauliX(0) @ qp.PauliX(1))


def _build_meta_at(H, observable, time, backend, *, n_steps=1, order=1):
    """Helper: instantiate a single-shot ``TimeEvolution`` and run its
    factory to obtain the un-cached ``MetaCircuit`` at ``time``."""
    prog = TimeEvolution(
        hamiltonian=H,
        time=time,
        n_steps=n_steps,
        order=order,
        observable=observable,
        backend=backend,
    )
    return prog._meta_circuit_factory(to_spo(H), ham_id=0)


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

    def test_dag_topology_identical_across_t(
        self, cache_test_hamiltonian, dummy_simulator
    ):
        H = cache_test_hamiltonian
        obs = qp.PauliZ(0)
        m1 = _build_meta_at(H, obs, 1.0, dummy_simulator)
        m2 = _build_meta_at(H, obs, 2.0, dummy_simulator)
        m05 = _build_meta_at(H, obs, 0.5, dummy_simulator)

        assert _dag_signature(m1) == _dag_signature(m2) == _dag_signature(m05)

    def test_rotation_angles_scale_linearly(
        self, cache_test_hamiltonian, dummy_simulator
    ):
        H = cache_test_hamiltonian
        obs = qp.PauliZ(0)
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

    def test_t_zero_preserves_structure(self, cache_test_hamiltonian, dummy_simulator):
        """t=0 must keep the same DAG length — zero-angle rotations don't
        get pruned. If they did, the cache would inflate gate counts at t=0."""
        H = cache_test_hamiltonian
        obs = qp.PauliZ(0)
        m1 = _build_meta_at(H, obs, 1.0, dummy_simulator)
        m0 = _build_meta_at(H, obs, 0.0, dummy_simulator)
        assert _dag_signature(m1) == _dag_signature(m0)

    def test_invariant_holds_with_n_steps_gt_1(
        self, cache_test_hamiltonian, dummy_simulator
    ):
        """With multiple Trotter steps the same symbolic ``t`` flows through
        more rotations; the topology and linear-scaling invariants must
        still hold."""
        H = cache_test_hamiltonian
        obs = qp.PauliZ(0)
        m1 = _build_meta_at(H, obs, 1.0, dummy_simulator, n_steps=2)
        m2 = _build_meta_at(H, obs, 2.0, dummy_simulator, n_steps=2)
        assert _dag_signature(m1) == _dag_signature(m2)


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
            observable=qp.PauliZ(0),
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
        obs = qp.PauliZ(0)
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

        for (tag_template, dag_template), (tag_fresh, dag_fresh) in zip(
            template.circuit_bodies, m_fresh.circuit_bodies, strict=True
        ):
            assert tag_template == tag_fresh
            qc = dag_to_circuit(dag_template)
            bound = qc.assign_parameters({t_param: t_test}, inplace=False)
            fresh = dag_to_circuit(dag_fresh)

            assert len(bound.data) == len(fresh.data)
            for inst_b, inst_f in zip(bound.data, fresh.data, strict=True):
                assert inst_b.operation.name == inst_f.operation.name
                wb = tuple(bound.find_bit(q).index for q in inst_b.qubits)
                wf = tuple(fresh.find_bit(q).index for q in inst_f.qubits)
                assert wb == wf
                for pb, pf in zip(
                    inst_b.operation.params, inst_f.operation.params, strict=True
                ):
                    assert abs(float(pb) - float(pf)) < 1e-9


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

    def test_cache_engaged_at_or_above_threshold(
        self, cache_test_hamiltonian, dummy_simulator
    ):
        traj = TimeEvolutionTrajectory(
            hamiltonian=cache_test_hamiltonian,
            time_points=[0.0, 0.25, 0.5, 0.75, 1.0],
            backend=dummy_simulator,
        )
        traj.create_programs()
        for prog in traj.programs.values():
            assert prog._template_meta is not None
            assert prog._template_param is not None
        # All programs share the same template object — that's the point.
        templates = {id(p._template_meta) for p in traj.programs.values()}
        assert len(templates) == 1

    def test_cache_engaged_at_exact_threshold(
        self, cache_test_hamiltonian, dummy_simulator
    ):
        """Boundary check: ``len(time_points) == _CACHE_MIN_TIME_POINTS``
        must engage the cache (the gate is ``< MIN``, not ``<= MIN``)."""
        n = workflow._CACHE_MIN_TIME_POINTS
        traj = TimeEvolutionTrajectory(
            hamiltonian=cache_test_hamiltonian,
            time_points=[0.1 * (i + 1) for i in range(n)],
            backend=dummy_simulator,
        )
        traj.create_programs()
        assert all(p._template_meta is not None for p in traj.programs.values())

    def test_cache_engaged_with_explicit_exact_trotterization(
        self, cache_test_hamiltonian, dummy_simulator
    ):
        """Explicit ``ExactTrotterization()`` must engage the cache, not
        only the implicit-default-via-None path."""
        traj = TimeEvolutionTrajectory(
            hamiltonian=cache_test_hamiltonian,
            time_points=[0.1, 0.2, 0.3, 0.4, 0.5],
            trotterization_strategy=ExactTrotterization(),
            backend=dummy_simulator,
        )
        traj.create_programs()
        assert all(p._template_meta is not None for p in traj.programs.values())

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
        self, cache_test_hamiltonian, dummy_simulator, monkeypatch
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
        traj.create_programs()
        for prog in traj.programs.values():
            assert prog._template_meta is None
            assert prog._template_param is None


class TestCachedBoundAnglesMatchUncached:
    """Deterministic numeric parity: bound template angles must equal
    un-cached numeric angles bit-for-bit across Trotter configurations.

    This complements the stochastic regression test below: the
    shots-based equivalence check has shot noise above any sub-1e-8 angle
    drift, so we verify circuit-level numeric agreement here directly.
    """

    @pytest.mark.parametrize(
        "n_steps, order",
        [(1, 1), (2, 1), (1, 2), (2, 2), (3, 2)],
    )
    def test_bound_angles_match_fresh_at_third_t(
        self, cache_test_hamiltonian, dummy_simulator, n_steps, order
    ):
        H = cache_test_hamiltonian
        obs = qp.PauliZ(0)
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
        for (_, dag_t), (_, dag_f) in zip(
            template.circuit_bodies, m_fresh.circuit_bodies, strict=True
        ):
            qc = dag_to_circuit(dag_t)
            bound = qc.assign_parameters({t_param: t_test}, inplace=False)
            fresh = dag_to_circuit(dag_f)
            assert len(bound.data) == len(fresh.data)
            for inst_b, inst_f in zip(bound.data, fresh.data, strict=True):
                assert inst_b.operation.name == inst_f.operation.name
                for pb, pf in zip(
                    inst_b.operation.params, inst_f.operation.params, strict=True
                ):
                    assert abs(float(pb) - float(pf)) < 1e-12


class TestCachedRunMatchesUncached:
    """Phase 3 regression test: the trajectory's *results* must be
    identical whether the cache is on or off."""

    def test_cached_and_uncached_results_agree(
        self, cache_test_hamiltonian, default_test_simulator, monkeypatch
    ):
        time_points = [0.0, 0.2, 0.4, 0.6, 0.8]

        # NOTE: ``traj_cached`` is built and run BEFORE the monkeypatch so
        # the cache engages for it. Do not hoist the patch above this block.
        traj_cached = TimeEvolutionTrajectory(
            hamiltonian=cache_test_hamiltonian,
            time_points=time_points,
            observable=qp.PauliZ(0),
            backend=default_test_simulator,
        )
        traj_cached.create_programs()
        assert all(p._template_meta is not None for p in traj_cached.programs.values())
        traj_cached.run(blocking=True)
        cached = traj_cached.aggregate_results()

        # Force the un-cached path by raising the threshold.
        monkeypatch.setattr(workflow, "_CACHE_MIN_TIME_POINTS", 999)
        traj_un = TimeEvolutionTrajectory(
            hamiltonian=cache_test_hamiltonian,
            time_points=time_points,
            observable=qp.PauliZ(0),
            backend=default_test_simulator,
        )
        traj_un.create_programs()
        assert all(p._template_meta is None for p in traj_un.programs.values())
        traj_un.run(blocking=True)
        uncached = traj_un.aggregate_results()

        for t in time_points:
            assert abs(cached[t] - uncached[t]) < 1e-9

        assert traj_cached.total_circuit_count == traj_un.total_circuit_count
