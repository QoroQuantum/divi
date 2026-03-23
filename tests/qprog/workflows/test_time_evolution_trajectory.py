# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import math

import pennylane as qml
import pytest

from divi.qprog import TimeEvolutionTrajectory
from divi.qprog.ensemble import BatchConfig, BatchMode

_PROB_TOL = 0.05


@pytest.fixture
def two_qubit_hamiltonian():
    return 0.5 * qml.PauliZ(0) + 0.3 * qml.PauliZ(1)


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
            observable=qml.PauliZ(0),
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
            hamiltonian=qml.PauliX(0),
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
            hamiltonian=0.5 * qml.PauliZ(0) + 0.3 * qml.PauliZ(1),
            time_points=[0.5, 1.0, 2.0],
            backend=default_test_simulator,
        )
        traj.create_programs()
        traj.run(blocking=True)
        results = traj.aggregate_results()

        for t, probs in results.items():
            assert probs.get("00", 0.0) >= 1.0 - _PROB_TOL
