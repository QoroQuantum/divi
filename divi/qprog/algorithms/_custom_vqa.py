# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from warnings import warn

import numpy as np
import pennylane as qp
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector

from divi.circuits import MetaCircuit, qscript_to_meta
from divi.hamiltonians import _clean_hamiltonian, _is_empty_hamiltonian
from divi.pipeline.stages import CircuitSpecStage
from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm

# ---------------------------------------------------------------------------
# Qiskit → PennyLane bridges used by the Qiskit-QC input path below.
#
# CustomVQA is the only remaining consumer of these converters — the pipeline
# spec stages now ingest Qiskit circuits directly into DAGs via
# :func:`divi.circuits._conversions._qscript_to_dag`.  Once CustomVQA's
# ansatz construction is rewritten to operate on ``QuantumCircuit`` objects
# natively, these helpers can go away.
# ---------------------------------------------------------------------------


def _qiskit_to_pennylane(
    qc: QuantumCircuit,
    measurement_fn: Callable[[list[int]], qp.measurements.MeasurementProcess],
) -> qp.tape.QuantumScript:
    """Convert a Qiskit QuantumCircuit to a PennyLane QuantumScript."""
    measured_wires = sorted(
        {
            qc.qubits.index(qubit)
            for instruction in qc.data
            if instruction.operation.name == "measure"
            for qubit in instruction.qubits
        }
    )
    if not measured_wires:
        warn(
            "Provided QuantumCircuit has no measurement operations. "
            "Defaulting to all wires.",
            UserWarning,
            stacklevel=2,
        )
        measured_wires = list(range(len(qc.qubits)))

    qc_no_measure = QuantumCircuit(*qc.qregs, *qc.cregs)
    for instruction in qc.data:
        if instruction.operation.name != "measure":
            qc_no_measure.append(
                instruction.operation, instruction.qubits, instruction.clbits
            )

    qfunc = qp.from_qiskit(qc_no_measure)
    params = [qp.numpy.array(0.0, requires_grad=True) for _ in qc.parameters]

    def qfunc_with_measurement(*p):
        qfunc(*p)
        return measurement_fn(measured_wires)

    return qp.tape.make_qscript(qfunc_with_measurement)(*params)


def _bind_qiskit_expressions(
    qscript: qp.tape.QuantumScript,
    qc: QuantumCircuit,
) -> tuple[qp.tape.QuantumScript, np.ndarray]:
    """Bind Qiskit parameter expressions from gate params into a QuantumScript."""
    base_params = np.array(list(qc.parameters), dtype=object)
    if len(base_params) == 0:
        return qscript, base_params

    gate_exprs: list[ParameterExpression] = []
    expr_indices: list[int] = []
    total_params = 0
    for instruction in qc.data:
        if instruction.operation.name == "measure":
            continue
        for param in instruction.operation.params:
            if isinstance(param, ParameterExpression):
                gate_exprs.append(param)
                expr_indices.append(total_params)
            total_params += 1

    qs_param_count = len(qscript.get_parameters())
    if total_params != qs_param_count:
        raise RuntimeError(
            f"Gate parameter count mismatch: Qiskit circuit has "
            f"{total_params} gate parameters but the converted "
            f"QuantumScript has {qs_param_count}."
        )

    bound = qscript.bind_new_parameters(gate_exprs, expr_indices)
    return bound, base_params


class CustomVQA(VariationalQuantumAlgorithm):
    """Custom variational algorithm for a parameterized QuantumScript.

    This implementation wraps a PennyLane QuantumScript (or converts a Qiskit
    QuantumCircuit into one) and optimizes its trainable parameters to minimize
    a single expectation-value measurement. Qiskit measurements are converted
    into a PauliZ expectation on the measured wires. Parameters are represented
    as Qiskit ``Parameter`` objects for QASM substitution and MetaCircuit
    template reuse during optimization. Qiskit ``ParameterExpression`` objects
    (e.g. ``2 * theta``) are preserved natively.

    Attributes:
        qscript (``qp.tape.QuantumScript``): The parameterized ``QuantumScript``.
        param_shape: Shape of a single parameter set.
        n_qubits (int): Number of qubits in the script.
        n_layers (int): Layer count (fixed to 1 for this wrapper).
        cost_hamiltonian: Observable being minimized.
        loss_constant (float): Constant term extracted from the observable.
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
            qscript (qp.tape.QuantumScript | QuantumCircuit): A parameterized QuantumScript with a
                single expectation-value measurement, or a Qiskit QuantumCircuit with
                computational basis measurements.
            param_shape: Shape of a single parameter
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

        self.cost_hamiltonian, self.loss_constant = _clean_hamiltonian(measurement.obs)
        if _is_empty_hamiltonian(self.cost_hamiltonian):
            raise ValueError("Hamiltonian contains only constant terms.")

        self.n_qubits = self.qscript.num_wires
        self.n_layers = 1
        self.max_iterations = max_iterations
        self.current_iteration = 0

        if self._qiskit_param_names is not None:
            # Qiskit path: ParameterExpressions already bound by
            # _bind_qiskit_expressions in _coerce_to_quantum_script.
            base_params = self._qiskit_base_params
            trainable_indices = list(range(len(base_params)))
            bound_qscript = self.qscript
        else:
            # PennyLane path: create Parameters and bind them.
            trainable_indices = (
                list(self.qscript.trainable_params)
                if self.qscript.trainable_params
                else list(range(len(self.qscript.get_parameters())))
            )
            base_params = np.array(
                ParameterVector("p", len(trainable_indices)), dtype=object
            )
            bound_qscript = self.qscript.bind_new_parameters(
                base_params.tolist(), trainable_indices
            )

        if not len(trainable_indices):
            raise ValueError("QuantumScript does not contain any trainable parameters.")

        self._param_shape = self._resolve_param_shape(
            param_shape, len(trainable_indices)
        )
        self._param_symbols = base_params.reshape(self._param_shape)
        self._trainable_param_indices = trainable_indices
        self._qscript = bound_qscript

        self.n_params_per_layer = int(np.prod(self._param_shape))

        # Build cost pipeline once (structure is fixed; only env changes per call).
        # No measurement pipeline needed — _perform_final_computation is a no-op.
        self._pipelines = self._build_pipelines()

    def _build_pipelines(self) -> dict:
        return {"cost": self._build_cost_pipeline(CircuitSpecStage())}

    @property
    def param_shape(self) -> tuple[int, ...]:
        """Shape of a single parameter set."""
        return self._param_shape

    def _resolve_param_shape(
        self, param_shape: tuple[int, ...] | int | None, n_params: int
    ) -> tuple[int, ...]:
        """Validate and normalize the parameter shape.

        Args:
            param_shape: User-provided parameter shape.
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
        qscript: qp.tape.QuantumScript | QuantumCircuit,
    ) -> qp.tape.QuantumScript:
        """Convert supported inputs into a PennyLane QuantumScript.

        Args:
            qscript (qp.tape.QuantumScript): Input QuantumScript or Qiskit QuantumCircuit.

        Returns:
            qp.tape.QuantumScript: The converted QuantumScript.

        Raises:
            TypeError: If the input type is unsupported.
        """
        if isinstance(qscript, qp.tape.QuantumScript):
            return qscript

        if isinstance(qscript, QuantumCircuit):

            def _expval_measurement(measured_wires):
                obs = (
                    qp.Z(measured_wires[0])
                    if len(measured_wires) == 1
                    else qp.sum(*(qp.Z(wire) for wire in measured_wires))
                )
                return qp.expval(obs)

            qs = _qiskit_to_pennylane(qscript, _expval_measurement)
            bound, self._qiskit_base_params = _bind_qiskit_expressions(qs, qscript)
            return bound

        raise TypeError(
            "qscript must be a PennyLane QuantumScript or a Qiskit QuantumCircuit."
        )

    def _create_meta_circuit_factories(self) -> dict[str, MetaCircuit]:
        """Create the meta-circuit factories for CustomVQA."""
        return {
            "cost_circuit": qscript_to_meta(
                self._qscript,
                precision=self._precision,
            )
        }
