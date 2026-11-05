# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""QRMIBackend against an IBM Quantum System stand-in.

Needs qrmi with its ``ibm`` extra, a running Simulated Quantum Resource
(https://github.com/qrmi-community/simulated_quantum_resource) serving
``fake_lagos``, and the settings qrmi's own end-to-end workflow exports.
``fake_lagos`` is noisy, so only robust properties are checked.
"""

import os

import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler.exceptions import TranspilerError

from divi.backends import DeviceConfig
from divi.qprog.algorithms import VQE, HartreeFockAnsatz
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import MolecularProblem
from tests.backends._circuit_runner_contracts import QASM_X_ON_FIRST_QUBIT

qrmi = pytest.importorskip("qrmi")

_SETTINGS = [
    f"fake_lagos_QRMI_IBM_QS_{name}"
    for name in (
        "ENDPOINT",
        "IAM_ENDPOINT",
        "IAM_APIKEY",
        "SERVICE_CRN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "S3_ENDPOINT",
        "S3_BUCKET",
        "S3_REGION",
    )
] + ["fake_lagos_QRMI_JOB_TIMEOUT_SECONDS"]

pytestmark = pytest.mark.skipif(
    not all(name in os.environ for name in _SETTINGS),
    reason="needs a Simulated Quantum Resource serving fake_lagos",
)


def _most_frequent(counts: dict[str, int]) -> str:
    return max(counts, key=counts.get)


@pytest.fixture
def backend(make_qrmi_backend):
    resource = qrmi.QuantumResource("fake_lagos", qrmi.ResourceType.IBMQuantumSystem)
    return make_qrmi_backend(resource, shots=1000)


class TestSimulatedIBMQuantumSystem:
    def test_counts_are_keyed_with_qubit_zero_rightmost(self, backend):
        (entry,) = backend.submit_circuits({"x0": QASM_X_ON_FIRST_QUBIT}).results

        assert entry["label"] == "x0"
        assert sum(entry["results"].values()) == 1000
        assert _most_frequent(entry["results"]) == "01"

    def test_each_circuit_gets_its_shots(self, backend):
        result = backend.submit_circuits(
            {"a": QASM_X_ON_FIRST_QUBIT, "b": QASM_X_ON_FIRST_QUBIT},
            shot_groups=[[0, 1, 200], [1, 2, 600]],
        )

        assert [sum(entry["results"].values()) for entry in result.results] == [
            200,
            600,
        ]
        assert result.backend_jobs == 2

    def test_runs_with_error_suppression_and_heavier_compilation(self, backend):
        """The stand-in ignores decoupling and twirling, so this checks only
        that a job carrying them runs."""
        backend.device_config = DeviceConfig(
            transpile_optimization_level=3,
            dynamical_decoupling_enable=True,
            twirling_enable_gates=True,
            twirling_enable_measure=True,
        )

        (entry,) = backend.submit_circuits({"x0": QASM_X_ON_FIRST_QUBIT}).results

        assert sum(entry["results"].values()) == 1000
        assert _most_frequent(entry["results"]) == "01"

    def test_a_circuit_wider_than_the_device_fails_before_submission(self, backend):
        wide = QuantumCircuit(8, 8)
        wide.h(range(8))
        wide.measure(range(8), range(8))

        with pytest.raises(TranspilerError, match="More virtual qubits"):
            backend.submit_circuits({"wide": wide})

    def test_a_vqe_reaches_a_physical_energy(self, backend, pyscf_h2):
        """fake_lagos' noise holds three COBYLA steps between -0.54 and -0.60
        Ha; the noiseless ground state is -1.137 Ha."""
        vqe = VQE(
            MolecularProblem.from_molecule(pyscf_h2),
            ansatz=HartreeFockAnsatz(),
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=3,
            backend=backend,
        )

        vqe.run()

        assert len(vqe.losses_history) == 3
        assert -1.2 < vqe.best_loss < -0.3
