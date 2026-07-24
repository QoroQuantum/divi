# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.stages._pce_cost_stage (PCECostStage).

Focus: expand behaviour (single Z-basis circuit regardless of backend),
path routing (soft vs hard CVaR), and multi-param-set independence.
Energy *values* for the helpers are already covered in
tests/qprog/algorithms/test_pce.py, so here we use hand-computed expected
values or comparative assertions.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit
from divi.hamiltonians import normalize_binary_polynomial_problem
from divi.pipeline.abc import PipelineEnv, ResultFormat
from divi.pipeline.stages import PCECostStage
from divi.pipeline.stages._pce_cost_stage import PCE_MEAS_AXIS, _PCEMeasToken
from divi.qprog.algorithms._pce import _decode_parities, _pack_masks
from tests.pipeline._helpers import measured_qubits


def _make_stage(
    qubo: np.ndarray,
    *,
    alpha: float = 1.0,
    soft: bool = True,
    alpha_cvar: float = 0.25,
    masks: np.ndarray | None = None,
    measure_all: bool = False,
):
    """Build a PCECostStage from a simple QUBO matrix.

    ``masks`` defaults to the dense ``arange`` encoding; pass an explicit array
    (1-D uint64 or 2-D limbs) to touch a specific subset of qubits.
    """
    problem = normalize_binary_polynomial_problem(qubo)
    if masks is None:
        masks = np.arange(1, problem.n_vars + 1, dtype=np.uint64)
    return PCECostStage(
        problem=problem,
        alpha=alpha,
        use_soft_objective=soft,
        decode_parities_fn=_decode_parities,
        variable_masks_u64=masks,
        alpha_cvar=alpha_cvar,
        measure_all=measure_all,
    )


def _make_env(result_format: ResultFormat):
    """Build a minimal PipelineEnv with the given result_format."""
    env = PipelineEnv(backend=None)
    env.result_format = result_format
    return env


def _meas_key(param_idx: int):
    """Build a child-result label for (param_set, pce_meas)."""
    return (("param_set", param_idx), (PCE_MEAS_AXIS, 0))


def _make_z_hamiltonian_batch(n_qubits: int) -> dict[int, MetaCircuit]:
    """Build a single-entry MetaCircuit batch with expval(Z0+Z1+...+Zn)."""
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(0.5, i)
    obs = SparsePauliOp.from_list(
        [("I" * (n_qubits - 1 - i) + "Z" + "I" * i, 1.0) for i in range(n_qubits)]
    )
    return {
        0: MetaCircuit(
            circuit_bodies=(((), circuit_to_dag(qc)),),
            observable=obs,
        )
    }


def _make_expval_backend():
    """Create a mock backend that supports expval."""
    return type("MockBackend", (), {"supports_expval": True})()


def _make_sampling_backend():
    """Create a mock backend that does NOT support expval."""
    return type("MockBackend", (), {"supports_expval": False})()


class TestExpandSingleCircuit:
    """PCECostStage.expand should produce one measurement circuit per param_set."""

    def test_expand_one_circuit_per_param_set_expval_backend(self):
        """With an expval-capable backend, still produces exactly 1 circuit."""
        n_qubits = 16
        batch = _make_z_hamiltonian_batch(n_qubits)
        stage = _make_stage(np.eye(n_qubits), alpha=1.0, soft=True)

        env = PipelineEnv(backend=_make_expval_backend())
        output = stage.expand(batch, env)

        expanded = list(output.batch.values())[0]
        assert len(expanded.measurement_qasms) == 1

    def test_expand_one_circuit_per_param_set_sampling_backend(self):
        """With a sampling-only backend, produces exactly 1 circuit."""
        n_qubits = 16
        batch = _make_z_hamiltonian_batch(n_qubits)
        stage = _make_stage(np.eye(n_qubits), alpha=1.0, soft=True)

        env = PipelineEnv(backend=_make_sampling_backend())
        output = stage.expand(batch, env)

        expanded = list(output.batch.values())[0]
        assert len(expanded.measurement_qasms) == 1

    def test_expand_result_format_is_counts(self):
        """Result format must be COUNTS after expand, regardless of backend."""
        n_qubits = 4
        batch = _make_z_hamiltonian_batch(n_qubits)
        stage = _make_stage(np.eye(n_qubits), alpha=1.0, soft=True)

        env = PipelineEnv(backend=_make_expval_backend())
        output = stage.expand(batch, env)

        assert all(
            meta.result_format is ResultFormat.COUNTS for meta in output.batch.values()
        )

    def test_expand_no_ham_ops_artifact(self):
        """ham_ops must NOT be set — PCE never uses the backend expval path."""
        n_qubits = 4
        batch = _make_z_hamiltonian_batch(n_qubits)
        stage = _make_stage(np.eye(n_qubits), alpha=1.0, soft=True)

        env = PipelineEnv(backend=_make_expval_backend())
        output = stage.expand(batch, env)

        assert all(meta.backend_ham_ops is None for meta in output.batch.values())


class TestMeasureAll:
    """``measure_all`` restricts (default) or forces full-register measurement."""

    _MASKS_Q0_Q1 = np.array([0b0001, 0b0010], dtype=np.uint64)

    def test_default_restricts_to_mask_union(self):
        # masks touch only qubits 0 and 1; the 4-qubit register leaves 2, 3 idle.
        stage = _make_stage(np.diag([1.0, 2.0]), masks=self._MASKS_Q0_Q1)
        output = stage.expand(
            _make_z_hamiltonian_batch(4), PipelineEnv(backend=_make_sampling_backend())
        )
        meta = list(output.batch.values())[0]
        assert measured_qubits(meta.measurement_qasms[0][1]) == {0, 1}

    def test_measure_all_true_measures_every_qubit(self):
        stage = _make_stage(
            np.diag([1.0, 2.0]), masks=self._MASKS_Q0_Q1, measure_all=True
        )
        output = stage.expand(
            _make_z_hamiltonian_batch(4), PipelineEnv(backend=_make_sampling_backend())
        )
        meta = list(output.batch.values())[0]
        assert measured_qubits(meta.measurement_qasms[0][1]) == {0, 1, 2, 3}

    def test_idle_qubit_outcome_does_not_change_energy(self):
        """Bits outside every mask never enter a parity, so histograms that
        differ only on an idle qubit reduce to the same energy."""
        stage = _make_stage(np.diag([1.0, 2.0]), masks=self._MASKS_Q0_Q1)
        env = _make_env(ResultFormat.COUNTS)
        # "0011" and "0111" agree on qubits 0,1 (masked) and differ on qubit 2.
        e_restricted = stage.reduce({_meas_key(0): {"0011": 100}}, env, token=None)
        e_full = stage.reduce({_meas_key(0): {"0111": 100}}, env, token=None)
        assert list(e_restricted.values()) == list(e_full.values())

    def test_reduce_rejects_narrowed_histogram_keys(self):
        """If a backend narrows keys to only measured clbits, positional parity
        decoding would silently corrupt — the reduce guard must fail loudly."""
        stage = _make_stage(np.diag([1.0, 2.0]), masks=self._MASKS_Q0_Q1)
        env = _make_env(ResultFormat.COUNTS)
        # expand emitted an n_qubits=3 circuit, but the backend returned 2-bit keys.
        token = _PCEMeasToken(n_qubits=3)
        with pytest.raises(ValueError, match="full-width keys"):
            stage.reduce({_meas_key(0): {"01": 100}}, env, token=token)

    def test_wide_register_limb_masks_through_stage(self):
        """A >64-qubit problem uses 2-D limb masks; expand must measure the
        mask-relevant wires (incl. one past the 64-bit boundary) and reduce must
        decode a full-width key without tripping the width guard."""
        nq = 70
        # Two variables read qubit 3 (limb 0) and qubit 68 (limb 1).
        masks = _pack_masks([1 << 3, 1 << 68], nq)
        assert masks.shape == (2, 2)
        stage = _make_stage(np.diag([1.0, 2.0]), masks=masks)

        output = stage.expand(
            _make_z_hamiltonian_batch(nq), PipelineEnv(backend=_make_sampling_backend())
        )
        meta = list(output.batch.values())[0]
        assert measured_qubits(meta.measurement_qasms[0][1]) == {3, 68}

        # Full-width key with qubit 68 set (big-endian: char at nq-1-68).
        key = ["0"] * nq
        key[nq - 1 - 68] = "1"
        env = _make_env(ResultFormat.COUNTS)
        reduced = stage.reduce(
            {_meas_key(0): {"".join(key): 100}},
            env,
            token=_PCEMeasToken(n_qubits=nq),
        )
        energy = list(reduced.values())[0]
        assert isinstance(energy[0], float)


class TestReduceHistogram:
    """Verify that reduce correctly processes single-histogram results."""

    def test_single_histogram_produces_energy(self):
        """A single histogram per param_set produces a valid energy."""
        qubo = np.diag([1.0, 2.0])
        stage = _make_stage(qubo, alpha=1.0, soft=True)
        env = _make_env(ResultFormat.COUNTS)

        result = stage.reduce(
            {_meas_key(0): {"00": 30, "01": 10, "10": 20, "11": 40}},
            env,
            token=None,
        )

        assert len(result) == 1
        value = list(result.values())[0]
        assert isinstance(value, list)
        assert len(value) == 1
        assert isinstance(value[0], float)

    def test_different_histograms_produce_different_energies(self):
        """Different shot distributions yield different energies."""
        qubo = np.diag([1.0, 2.0])
        stage = _make_stage(qubo, alpha=1.0, soft=True)
        env = _make_env(ResultFormat.COUNTS)

        result_a = stage.reduce(
            {_meas_key(0): {"00": 100}},
            env,
            token=None,
        )
        result_b = stage.reduce(
            {_meas_key(0): {"11": 100}},
            env,
            token=None,
        )

        assert list(result_a.values())[0][0] != pytest.approx(
            list(result_b.values())[0][0]
        )


class TestReducePathRouting:
    """Verify reduce dispatches to the correct energy computation path."""

    def test_soft_and_hard_produce_different_energies(self):
        """Soft energy and hard CVaR energy differ for the same histogram."""
        qubo = np.diag([1.0, 2.0])
        histogram = {"11": 2, "10": 3, "01": 10, "00": 25}
        env = _make_env(ResultFormat.COUNTS)

        soft_stage = _make_stage(qubo, alpha=1.0, soft=True)
        hard_stage = _make_stage(qubo, alpha=6.0, soft=False, alpha_cvar=0.25)

        soft_energy = list(
            soft_stage.reduce({_meas_key(0): histogram}, env, token=None).values()
        )[0][0]
        hard_energy = list(
            hard_stage.reduce({_meas_key(0): histogram}, env, token=None).values()
        )[0][0]

        assert soft_energy != pytest.approx(hard_energy)

    def test_deterministic_histogram_soft_energy(self):
        """All shots in one bitstring → known energy.

        qubo = diag([1, 2]), all shots "00" → parities [0, 0] for masks [1, 2].
        mean_parities = [0, 0], z = 1 - 2*0 = [1, 1].
        x_soft = 0.5*(1 + tanh(1*1)) = 0.5*(1 + tanh(1)) for both vars.
        energy = 1*x0² + 2*x1² (degree-1 terms use x²).
        """
        qubo = np.diag([1.0, 2.0])
        stage = _make_stage(qubo, alpha=1.0, soft=True)
        env = _make_env(ResultFormat.COUNTS)

        result = stage.reduce({_meas_key(0): {"00": 100}}, env, token=None)

        x = 0.5 * (1.0 + np.tanh(1.0))  # ≈ 0.8808
        expected = 1.0 * x**2 + 2.0 * x**2  # 3 * x²
        assert list(result.values())[0][0] == pytest.approx(expected)

    def test_deterministic_histogram_hard_cvar_energy(self):
        """All shots in one bitstring → known CVaR energy.

        qubo = diag([1, 2]), all shots "11" → parities [1, 1] for masks [1, 2].
        x_vals = 1 - parities = [0, 0].  Energy = 0 for every shot.
        CVaR of a single-valued distribution is that value: 0.
        """
        qubo = np.diag([1.0, 2.0])
        stage = _make_stage(qubo, alpha=6.0, soft=False, alpha_cvar=0.25)
        env = _make_env(ResultFormat.COUNTS)

        result = stage.reduce({_meas_key(0): {"11": 100}}, env, token=None)

        assert list(result.values())[0][0] == pytest.approx(0.0)

    def test_hard_cvar_selects_low_energy_tail(self):
        """CVaR with alpha_cvar=0.5 selects the lower-energy half of shots.

        qubo = diag([1, 2]), masks = [1, 2].
        Bitstring "11" → parities [1, 1] → x = [0, 0] → energy = 0.
        Bitstring "00" → parities [0, 0] → x = [1, 1] → energy = 1+2 = 3.

        50 shots of "11" (energy 0) + 50 shots of "00" (energy 3).
        Mean energy = 1.5.
        CVaR(0.5) takes the lowest 50 shots → all "11" → energy = 0.
        """
        qubo = np.diag([1.0, 2.0])
        stage = _make_stage(qubo, alpha=6.0, soft=False, alpha_cvar=0.5)
        env = _make_env(ResultFormat.COUNTS)

        result = stage.reduce({_meas_key(0): {"11": 50, "00": 50}}, env, token=None)

        assert list(result.values())[0][0] == pytest.approx(0.0)


def test_two_param_sets_independent():
    """Two param_sets with different histograms produce different energies."""
    qubo = np.diag([1.0, 2.0])
    stage = _make_stage(qubo, alpha=1.0, soft=True)
    env = _make_env(ResultFormat.COUNTS)

    results = {
        _meas_key(0): {"00": 100},  # all parities 0
        _meas_key(1): {"11": 100},  # all parities 1
    }

    reduced = stage.reduce(results, env, token=None)

    assert len(reduced) == 2
    energies = [v[0] for v in reduced.values()]
    # Different histograms must yield different energies
    assert energies[0] != pytest.approx(energies[1])
