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

from qoro_extension import VQEPlotCircuitExtension

# Step 1: Define your molecule
h2_molecule = qml.qchem.Molecule(
   symbols=["H", "H"], coordinates=np.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]])
)

# Step 2: Choose your optimizer
optimizer = ScipyOptimizer(method=ScipyMethod.COBYLA)

# Step 3: Set up your quantum program
vqe = VQEPlotCircuitExtension(
   molecule=h2_molecule,
   ansatz=HartreeFockAnsatz(),
   n_layers=2,  # Circuit depth
   optimizer=optimizer,
   max_iterations=10,  # Optimization steps
   backend=ParallelSimulator(shots=1000),  # Local simulator
)

vqe.plot_circuits(backend='text')