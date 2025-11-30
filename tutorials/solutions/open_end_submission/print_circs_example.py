from divi.qprog import HartreeFockAnsatz
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.backends import ParallelSimulator

# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from warnings import warn

import numpy as np
import numpy.typing as npt
import pennylane as qml
import sympy as sp

from divi.circuits import CircuitBundle, MetaCircuit
from divi.qprog._hamiltonians import _clean_hamiltonian
from divi.qprog.algorithms._ansatze import Ansatz, HartreeFockAnsatz
from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm
from divi.qprog import HartreeFockAnsatz, GenericLayerAnsatz
from pennylane import qchem

from divi.qprog.algorithms._vqe_extension import VQEExtension


class HFLayerAnsatz(GenericLayerAnsatz):
    """
    GenericLayerAnsatz on top of a Hartree-Fock reference state.

    Usage:
        ansatz = HFLayerAnsatz(
            gate_sequence=[qml.RY, qml.RZ],
            entangler=qml.CNOT,
            entangling_layout="linear",
        )

        # later, when building:
        ops = ansatz.build(
            params,
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_electrons=n_electrons,  # or hf_state=...
        )
    """

    def build(
        self,
        params: Any,
        n_qubits: int,
        n_layers: int,
        **kwargs: Any,
    ) -> list[qml.operation.Operator]:
        # Option A: user directly passes a bitstring/array as hf_state
        hf_state = kwargs.pop("hf_state", None)

        # Option B: derive HF state from number of electrons
        if hf_state is None:
            n_electrons = kwargs.get("n_electrons", None)
            if n_electrons is None:
                raise ValueError(
                    "HFLayerAnsatz.build requires either `hf_state` or `n_electrons` "
                    "in kwargs."
                )
            hf_state = qchem.hf_state(n_electrons, n_qubits)

        wires = list(range(n_qubits))

        # 1) HF preparation as the very first operation
        operations: list[qml.operation.Operator] = [
            qml.BasisState(hf_state, wires=wires)
        ]

        # 2) All the usual layers from GenericLayerAnsatz
        layer_ops = super().build(params, n_qubits=n_qubits, n_layers=n_layers, **kwargs)

        return operations + layer_ops
    

if __name__ == "__main__":
   ## Example usage of VQEPlotCircuitExtension

   ## H2 example 

   # Create your VQE instance
   h2_molecule = qml.qchem.Molecule(
      symbols=["H", "H"], coordinates=np.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]])
   )

   optimizer = ScipyOptimizer(method=ScipyMethod.COBYLA)

   vqe_h2 = VQEExtension(
      molecule=h2_molecule,
      ansatz=HartreeFockAnsatz(),
      n_layers=2,  # Circuit depth
      optimizer=optimizer,
      max_iterations=10,  # Optimization steps
      backend=ParallelSimulator(shots=1000),  # Local simulator
   )

   # Plot your circuits
   vqe_h2.plot_circuits(backend='text')