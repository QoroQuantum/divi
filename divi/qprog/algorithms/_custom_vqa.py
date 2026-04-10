# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pennylane as qml
import sympy as sp
from qiskit import QuantumCircuit

from divi.circuits import MetaCircuit
from divi.hamiltonians import _clean_hamiltonian, _is_empty_hamiltonian
from divi.pipeline.stages import CircuitSpecStage
from divi.pipeline.stages._qiskit_spec_stage import (
    _bind_qiskit_expressions,
    qiskit_to_pennylane,
)
from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm


class CustomVQA(VariationalQuantumAlgorithm):
    """Custom variational algorithm for a parameterized QuantumScript.

    This implementation wraps a PennyLane QuantumScript (or converts a Qiskit
    QuantumCircuit into one) and optimizes its trainable parameters to minimize
    a single expectation-value measurement. Qiskit measurements are converted
    into a PauliZ expectation on the measured wires. Parameters are bound to sympy
    symbols to enable QASM substitution and reuse of MetaCircuit templates
    during optimization.  Qiskit ``ParameterExpression`` objects
    (e.g. ``2 * theta``) are preserved as sympy expressions.

    Attributes:
        qscript (``qml.tape.QuantumScript``): The parameterized ``QuantumScript``.
        param_shape (tuple[int, ...]): Shape of a single parameter set.
        n_qubits (int): Number of qubits in the script.
        n_layers (int): Layer count (fixed to 1 for this wrapper).
        cost_hamiltonian (qml.operation.Operator): Observable being minimized.
        loss_constant (float): Constant term extracted from the observable.
        optimizer (Optimizer): Classical optimizer for parameter updates.
        max_iterations (int): Maximum number of optimization iterations.
        current_iteration (int): Current optimization iteration.
    """

    def __init__(
        self,
        qscript: qml.tape.QuantumScript | QuantumCircuit,
        *,
        param_shape: tuple[int, ...] | int | None = None,
        max_iterations: int = 10,
        **kwargs,
    ) -> None:
        """Initialize a CustomVQA instance.

        Args:
            qscript (qml.tape.QuantumScript | QuantumCircuit): A parameterized QuantumScript with a
                single expectation-value measurement, or a Qiskit QuantumCircuit with
                computational basis measurements.
            param_shape (tuple[int, ...] | int | None): Shape of a single parameter
                set. If None, uses a flat shape inferred from trainable parameters.
            max_iterations (int): Maximum number of optimization iterations.
            **kwargs: Additional keyword arguments passed to the parent class, including
                backend and optimizer.

        Raises:
            TypeError: If ``qscript`` is not a supported PennyLane ``QuantumScript`` or Qiskit ``QuantumCircuit``.
            ValueError: If the script has an invalid measurement or no trainable parameters.
        """
        super().__init__(**kwargs)

        self._qiskit_param_names = (
            [param.name for param in qscript.parameters]
            if isinstance(qscript, QuantumCircuit)
            else None
        )
        self.qscript = self._coerce_to_quantum_script(qscript)

        if len(self.qscript.measurements) != 1:
            raise ValueError(
                "QuantumScript must contain exactly one measurement for optimization."
            )

        measurement = self.qscript.measurements[0]
        if not hasattr(measurement, "obs") or measurement.obs is None:
            raise ValueError(
                "QuantumScript must contain a single expectation-value measurement."
            )

        self._cost_hamiltonian, self.loss_constant = _clean_hamiltonian(measurement.obs)
        if _is_empty_hamiltonian(self._cost_hamiltonian):
            raise ValueError("Hamiltonian contains only constant terms.")

        self.n_qubits = self.qscript.num_wires
        self.n_layers = 1
        self.max_iterations = max_iterations
        self.current_iteration = 0

        if self._qiskit_param_names is not None:
            # Qiskit path: symbols already bound by _bind_qiskit_expressions
            # in _coerce_to_quantum_script — no rebinding needed.
            n_trainable = len(self._qiskit_param_names)
            if n_trainable == 0:
                raise ValueError(
                    "QuantumScript does not contain any trainable parameters."
                )
            self._param_shape = self._resolve_param_shape(param_shape, n_trainable)
            self._param_symbols = self._qiskit_base_symbols.reshape(self._param_shape)
            self._trainable_param_indices = list(range(n_trainable))
            self._qscript = self.qscript
        else:
            # PennyLane path: create symbols and bind them.
            trainable_param_indices = (
                list(self.qscript.trainable_params)
                if self.qscript.trainable_params
                else list(range(len(self.qscript.get_parameters())))
            )
            if not trainable_param_indices:
                raise ValueError(
                    "QuantumScript does not contain any trainable parameters."
                )
            self._trainable_param_indices = trainable_param_indices
            self._param_shape = self._resolve_param_shape(
                param_shape, len(trainable_param_indices)
            )
            self._param_symbols = sp.symarray("p", self._param_shape)
            flat_symbols = self._param_symbols.flatten().tolist()
            self._qscript = self.qscript.bind_new_parameters(
                flat_symbols, trainable_param_indices
            )

        self._n_params_per_layer = int(np.prod(self._param_shape))

        # Build cost pipeline once (structure is fixed; only env changes per call).
        # No measurement pipeline needed — _perform_final_computation is a no-op.
        self._build_pipelines()

    def _build_pipelines(self) -> None:
        self._cost_pipeline = self._build_cost_pipeline(CircuitSpecStage())

    @property
    def param_shape(self) -> tuple[int, ...]:
        """Shape of a single parameter set."""
        return self._param_shape

    def _resolve_param_shape(
        self, param_shape: tuple[int, ...] | int | None, n_params: int
    ) -> tuple[int, ...]:
        """Validate and normalize the parameter shape.

        Args:
            param_shape (tuple[int, ...] | int | None): User-provided parameter shape.
            n_params (int): Number of trainable parameters in the script.

        Returns:
            tuple[int, ...]: Normalized parameter shape.

        Raises:
            ValueError: If the shape is invalid or does not match n_params.
        """
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

    def _coerce_to_quantum_script(
        self,
        qscript: qml.tape.QuantumScript | QuantumCircuit,
    ) -> qml.tape.QuantumScript:
        """Convert supported inputs into a PennyLane QuantumScript.

        Args:
            qscript (qml.tape.QuantumScript): Input QuantumScript or Qiskit QuantumCircuit.

        Returns:
            qml.tape.QuantumScript: The converted QuantumScript.

        Raises:
            TypeError: If the input type is unsupported.
        """
        if isinstance(qscript, qml.tape.QuantumScript):
            return qscript

        if isinstance(qscript, QuantumCircuit):

            def _expval_measurement(measured_wires):
                obs = (
                    qml.Z(measured_wires[0])
                    if len(measured_wires) == 1
                    else qml.sum(*(qml.Z(wire) for wire in measured_wires))
                )
                return qml.expval(obs)

            qs = qiskit_to_pennylane(qscript, _expval_measurement)
            bound, self._qiskit_base_symbols = _bind_qiskit_expressions(qs, qscript)
            return bound

        raise TypeError(
            "qscript must be a PennyLane QuantumScript or a Qiskit QuantumCircuit."
        )

    def _create_meta_circuit_factories(self) -> dict[str, MetaCircuit]:
        """Create the meta-circuit factories for CustomVQA.

        Returns:
            dict[str, MetaCircuit]: Dictionary containing the cost circuit factory.
        """
        return {
            "cost_circuit": MetaCircuit(
                source_circuit=self._qscript,
                symbols=self._param_symbols.flatten(),
                precision=self._precision,
            )
        }

    def _perform_final_computation(self, **kwargs) -> None:
        """No-op by default for custom QuantumScript optimization."""
