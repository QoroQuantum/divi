# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from warnings import warn

import numpy as np
import pennylane as qp
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit, qscript_to_meta
from divi.hamiltonians._mixers import single_pauli_label
from divi.hamiltonians._term_ops import _clean_hamiltonian_spo, to_spo
from divi.pipeline.stages import CircuitSpecStage
from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm


def _measured_wires_of(qc: QuantumCircuit) -> list[int]:
    """Indices of qubits targeted by ``measure`` instructions, ascending."""
    return sorted(
        {
            qc.qubits.index(qubit)
            for instruction in qc.data
            if instruction.operation.name == "measure"
            for qubit in instruction.qubits
        }
    )


def _strip_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of ``qc`` with all ``measure`` instructions dropped."""
    stripped = QuantumCircuit(*qc.qregs, *qc.cregs)
    for instruction in qc.data:
        if instruction.operation.name == "measure":
            continue
        stripped.append(instruction.operation, instruction.qubits, instruction.clbits)
    return stripped


def _z_sum_observable(n_qubits: int, measured_wires: list[int]) -> SparsePauliOp:
    """Sum of ``Z_i`` over the given wires, lifted to an ``n_qubits``-wire SPO."""
    return SparsePauliOp.from_list(
        [(single_pauli_label(n_qubits, wire, "Z"), 1.0) for wire in measured_wires]
    )


class CustomVQA(VariationalQuantumAlgorithm):
    """Custom variational algorithm for a parameterized circuit.

    Wraps either a PennyLane ``QuantumScript`` or a Qiskit ``QuantumCircuit``
    and optimizes its trainable parameters to minimize a single
    expectation-value measurement. Qiskit measurements on selected qubits
    convert to a sum-of-Z observable on those wires. Qiskit
    ``ParameterExpression`` objects (e.g. ``2 * theta``) are preserved
    natively through the Qiskit DAG.

    Attributes:
        qscript: The input ``QuantumScript`` or ``QuantumCircuit``.
        param_shape: Shape of a single parameter set.
        n_qubits (int): Number of qubits in the circuit.
        n_layers (int): Layer count (fixed to 1 for this wrapper).
        cost_hamiltonian: Observable being minimized, as a Qiskit
            ``SparsePauliOp``.
        loss_constant (float): Constant term extracted from the observable.
        measured_wires: For Qiskit input, the qubit indices targeted by
            ``measure`` instructions.
        optimizer: Classical optimizer for parameter updates.
        max_iterations (int): Maximum number of optimization iterations.
        current_iteration (int): Current optimization iteration.
    """

    def __init__(
        self,
        qscript: qp.tape.QuantumScript | QuantumCircuit,
        *,
        param_shape: tuple[int, ...] | int | None = None,
        max_iterations: int = 10,
        **kwargs,
    ) -> None:
        """Initialize a CustomVQA instance.

        Args:
            qscript: A parameterized ``QuantumScript`` with a single
                expectation-value measurement, or a Qiskit ``QuantumCircuit``
                with computational-basis measurements (mapped to a sum-of-Z
                observable on the measured wires).
            param_shape: Shape of a single parameter set. If None, uses a
                flat shape inferred from trainable parameters.
            max_iterations: Maximum number of optimization iterations.
            **kwargs: Additional keyword arguments passed to the parent
                class, including backend and optimizer.

        Raises:
            TypeError: If ``qscript`` is not a supported type.
            ValueError: If the input has an invalid measurement or no
                trainable parameters.
        """
        super().__init__(**kwargs)

        self.qscript = qscript
        self.n_layers = 1
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.measured_wires: tuple[int, ...] = ()
        self._qiskit_circuit: QuantumCircuit | None = None

        if isinstance(qscript, QuantumCircuit):
            base_params = self._prepare_qiskit_input(qscript)
        elif isinstance(qscript, qp.tape.QuantumScript):
            base_params = self._prepare_pennylane_input(qscript)
        else:
            raise TypeError(
                "qscript must be a PennyLane QuantumScript or a Qiskit "
                "QuantumCircuit."
            )

        if not len(base_params):
            raise ValueError("QuantumScript does not contain any trainable parameters.")

        self._param_shape = self._resolve_param_shape(param_shape, len(base_params))
        self._param_symbols = base_params.reshape(self._param_shape)

        self._pipelines = self._build_pipelines()

    @property
    def n_params_per_layer(self) -> int:
        return int(np.prod(self._param_shape))

    @property
    def param_shape(self) -> tuple[int, ...]:
        """Shape of a single parameter set."""
        return self._param_shape

    def _prepare_qiskit_input(self, qc: QuantumCircuit) -> np.ndarray:
        """Set state for a Qiskit ``QuantumCircuit`` input; return base parameters."""
        measured_wires = _measured_wires_of(qc)
        if not measured_wires:
            warn(
                "Provided QuantumCircuit has no measurement operations. "
                "Defaulting to all wires.",
                UserWarning,
                stacklevel=3,
            )
            measured_wires = list(range(qc.num_qubits))

        self.n_qubits = qc.num_qubits
        self.measured_wires = tuple(measured_wires)
        self._qiskit_circuit = _strip_measurements(qc)

        observable = _z_sum_observable(qc.num_qubits, measured_wires)
        self.cost_hamiltonian, self.loss_constant = _clean_hamiltonian_spo(observable)
        if self.cost_hamiltonian.size == 0:
            raise ValueError("Hamiltonian contains only constant terms.")

        return np.array(list(qc.parameters), dtype=object)

    def _prepare_pennylane_input(self, qs: qp.tape.QuantumScript) -> np.ndarray:
        """Set state for a PennyLane ``QuantumScript`` input; return base parameters."""
        if len(qs.measurements) != 1:
            raise ValueError(
                "QuantumScript must contain exactly one measurement for optimization."
            )
        measurement = qs.measurements[0]
        if not hasattr(measurement, "obs") or measurement.obs is None:
            raise ValueError(
                "QuantumScript must contain a single expectation-value measurement."
            )

        self.cost_hamiltonian, self.loss_constant = _clean_hamiltonian_spo(
            to_spo(measurement.obs)
        )
        if self.cost_hamiltonian.size == 0:
            raise ValueError("Hamiltonian contains only constant terms.")

        self.n_qubits = qs.num_wires
        trainable_indices = (
            list(qs.trainable_params)
            if qs.trainable_params
            else list(range(len(qs.get_parameters())))
        )
        base_params = np.array(
            ParameterVector("p", len(trainable_indices)), dtype=object
        )
        self._bound_qscript = qs.bind_new_parameters(
            base_params.tolist(), trainable_indices
        )
        return base_params

    def _build_pipelines(self) -> dict:
        return {"cost": self._build_cost_pipeline(CircuitSpecStage())}

    def _resolve_param_shape(
        self, param_shape: tuple[int, ...] | int | None, n_params: int
    ) -> tuple[int, ...]:
        """Validate and normalize the parameter shape."""
        if param_shape is None:
            return (n_params,)

        param_shape = (param_shape,) if isinstance(param_shape, int) else param_shape

        if any(dim <= 0 for dim in param_shape):
            raise ValueError(
                f"param_shape entries must be positive, got {param_shape}."
            )

        if int(np.prod(param_shape)) != n_params:
            raise ValueError(
                f"param_shape does not match the number of trainable parameters. "
                f"Expected product {n_params}, got {int(np.prod(param_shape))}."
            )

        return tuple(param_shape)

    def _create_meta_circuit_factories(self) -> dict[str, MetaCircuit]:
        """Create the cost meta-circuit factory for CustomVQA."""
        if self._qiskit_circuit is not None:
            meta = MetaCircuit(
                circuit_bodies=(((), circuit_to_dag(self._qiskit_circuit)),),
                parameters=tuple(self._param_symbols.flatten()),
                observable=self.cost_hamiltonian,
                precision=self._precision,
            )
        else:
            meta = qscript_to_meta(
                self._bound_qscript,
                precision=self._precision,
            )
        return {"cost_circuit": meta}
