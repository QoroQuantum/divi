from itertools import product
from divi.qprog.workflows import VQEHyperparameterSweep
from divi.qprog.algorithms._vqe import VQE

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
from itertools import product

from divi.circuits import CircuitBundle, MetaCircuit
from divi.qprog._hamiltonians import _clean_hamiltonian
from divi.qprog.algorithms._ansatze import Ansatz, HartreeFockAnsatz
from divi.qprog.algorithms._vqe import VariationalQuantumAlgorithm


class VQEExtension(VQE):
    """Extension of the Variational Quantum Eigensolver (VQE) implementation with the added
    functionality of plotting the circuits.

    Attributes:
        ansatz (Ansatz): The parameterized quantum circuit ansatz.
        n_layers (int): Number of ansatz layers.
        n_qubits (int): Number of qubits in the system.
        n_electrons (int): Number of electrons (for molecular systems).
        cost_hamiltonian (qml.operation.Operator): The Hamiltonian to minimize.
        loss_constant (float): Constant term extracted from the Hamiltonian.
        molecule (qml.qchem.Molecule): The molecule object (if applicable).
        optimizer (Optimizer): Classical optimizer for parameter updates.
        max_iterations (int): Maximum number of optimization iterations.
        current_iteration (int): Current optimization iteration.
    """

    def __init__(
            self,
        **kwargs,
    ) -> None:
        """Initialize the VQE problem.

        Args:
          **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._circuits : dict[str, qml.tape.QuantumScript] = None  # Will be initialized lazily

    def plot_circuits(self, backend: str = 'text') -> None:
        """Plot the circuits used in the VQE program.

        Args:
            backend (str): The plotting backend to use. Options are 'mpl' for Matplotlib 
                and 'text' for text-based representation. Defaults to 'text'.
        """
        circuits = self.circuits

        for name, circuit in circuits.items():
            print(f"\nCircuit: {name}")

            if backend == 'mpl':
                pass  # Matplotlib plotting not implemented in this snippet
            elif backend == 'text':
                # Text-based representation
                print(circuit.draw())     
            else:
                warn(f"Unsupported backend '{backend}'. Supported backends are 'mpl' and 'text'.")


    @property    
    def circuits(self) -> dict[str, qml.tape.QuantumScript]:
        """Get the circuit used by this program.

        Returns:
            dict[str, qml.tape.QuantumScript]: Dictionary mapping circuit names to their
                Circuits.
        """
        # Lazy initialization: each instance has its own _meta_circuits.
        # Note: When used with ProgramBatch, meta_circuits is initialized sequentially
        # in the main thread before parallel execution to avoid thread-safety issues.
        if self._circuits is None:
            self._circuits = self._create_circuits_dict()
        return self._circuits

    def _create_circuits_dict(self) -> dict[qml.tape.QuantumScript]:
        """Create the circuit dictionary for VQE.

        Returns:
            dict[str, qml.tape.QuantumScript]: Dictionary containing the cost circuit template.
        """
        weights_syms = sp.symarray(
            "w",
            (
                self.n_layers,
                self.ansatz.n_params_per_layer(
                    self.n_qubits, n_electrons=self.n_electrons
                ),
            ),
        )

        ops = self.ansatz.build(
            weights_syms,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            n_electrons=self.n_electrons,
        )

        return {
            "cost_circuit": 
                qml.tape.QuantumScript(
                    ops=ops, measurements=[qml.expval(self._cost_hamiltonian)]
                )
            ,
            "meas_circuit":
                qml.tape.QuantumScript(ops=ops, measurements=[qml.probs()])
                ,
        }

    def _create_meta_circuits_dict(self) -> dict[str, MetaCircuit]:
        """Create the meta-circuit dictionary for VQE. Overrides VQE class method.

        Returns:
            dict[str, MetaCircuit]: Dictionary containing the cost circuit template.
        """
        
        weights_syms = sp.symarray(
            "w",
            (
                self.n_layers,
                self.ansatz.n_params_per_layer(
                    self.n_qubits, n_electrons=self.n_electrons
                ),
            ),
        )

        
        circuits = self._create_circuits_dict()

        return {
            "cost_circuit": self._meta_circuit_factory(
                circuits["cost_circuit"],
                symbols=weights_syms.flatten(),
            ),
            "meas_circuit": self._meta_circuit_factory(
                circuits["meas_circuit"],
                symbols=weights_syms.flatten(),
                grouping_strategy="wires",
            ),
        }

