# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections.abc import Mapping, Sequence
from typing import cast
from warnings import warn

import numpy as np
import numpy.typing as npt
import pennylane as qp
import sympy as sp
from pennylane.measurements import ExpectationMP
from pennylane.workflow.qnode import QNode
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import (
    MetaCircuit,
    qscript_to_meta,
)
from divi.circuits._pennylane_utils import (
    _detect_batch_input_argnames,
    _qnode_to_symbolic_qscript,
    _symbol_arg_name,
    _validate_single_measurement,
)
from divi.hamiltonians._mixers import single_pauli_label
from divi.hamiltonians._term_ops import _clean_hamiltonian_spo
from divi.pipeline.stages import (
    CircuitSpecStage,
    LossReductionFn,
    SampleLossFn,
    resolve_loss_reduction,
)
from divi.qprog.algorithms._data_binding import DataBindingMixin
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


class CustomVQA(DataBindingMixin, VariationalQuantumAlgorithm):
    """Custom variational algorithm for a parameterized circuit.

    Wraps a PennyLane ``QuantumScript``, a PennyLane ``QNode``, or a Qiskit
    ``QuantumCircuit`` and optimizes its trainable parameters to minimize a
    single expectation-value measurement. A ``QNode`` is converted to a
    ``QuantumScript`` upfront, with its required (no-default) arguments taken
    as the trainable parameters. Qiskit measurements on selected qubits
    convert to a sum-of-Z observable on those wires. Qiskit
    ``ParameterExpression`` objects (e.g. ``2 * theta``) are preserved
    natively through the Qiskit DAG.

    Optionally accepts a classical ``feature_batch`` plus
    ``data_param_indices`` (indices into the circuit's parameter ordering)
    to train in QNN style: the selected parameters are bound from each row
    of the batch via :class:`~divi.pipeline.stages.DataBindingStage`,
    per-sample expectation values are aggregated by ``loss_reduction``,
    and only the remaining parameters are exposed to the optimizer.

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
        feature_batch (numpy.ndarray or None): Classical feature batch when
            data binding is active; otherwise ``None``.
        labels (numpy.ndarray or None): Supervised targets of shape
            ``(n_samples,)`` when training a supervised loss; otherwise
            ``None``.
        loss_reduction: User-facing aggregation across samples when data
            binding is active; ignored otherwise.
        optimizer: Classical optimizer for parameter updates.
        max_iterations (int): Maximum number of optimization iterations.
        current_iteration (int): Current optimization iteration.
    """

    def __init__(
        self,
        qscript: qp.tape.QuantumScript | QNode | QuantumCircuit,
        *,
        param_shape: tuple[int, ...] | int | None = None,
        data_param_indices: Sequence[int] | None = None,
        feature_batch: npt.ArrayLike | None = None,
        arg_shapes: Mapping[str, tuple[int, ...]] | None = None,
        data_arg: str | None = None,
        labels: npt.ArrayLike | None = None,
        loss_fn: SampleLossFn = "squared_error",
        loss_reduction: LossReductionFn = "mean",
        max_iterations: int = 10,
        **kwargs,
    ) -> None:
        """Initialize a CustomVQA instance.

        Args:
            qscript: A parameterized ``QuantumScript`` or ``QNode`` with a
                single expectation-value measurement, or a Qiskit
                ``QuantumCircuit`` with computational-basis measurements
                (mapped to a sum-of-Z observable on the measured wires).
                ``QNode`` inputs are converted to ``QuantumScript`` upfront;
                the stored ``qscript`` attribute then holds the converted
                ``QuantumScript``, not the original ``QNode``. A ``QNode``
                argument with a Python default is treated as a fixed constant
                (not trained), matching PennyLane's trainability semantics.
                Structural values (qubit/layer counts used only for control
                flow, never as gate angles) are neither data nor weights:
                close over them in the enclosing scope or give them a default —
                a no-default structural argument is symbolized like a weight and
                then breaks (e.g. ``range(<symbol>)``). The QNode is traced one
                sample at a time, so index by the structural size
                (``range(n_qubits)``), not the batch dimension
                (``len(inputs[0])``).
            param_shape: Shape of a single parameter set. If None, uses a
                flat shape inferred from trainable parameters. Must be
                ``None`` when ``data_param_indices`` is set — the optimizer
                view is automatically flat over the remaining weight
                parameters.
            data_param_indices: Integer indices (into the circuit's flat
                parameter ordering) marking which parameters are bound
                from ``feature_batch`` rather than optimized. For Qiskit
                input, indices reference ``list(qc.parameters)``; for
                PennyLane input, indices reference the trainable-parameter
                ordering used to synthesize the internal parameter vector.
                Mutually required with ``feature_batch``.
            feature_batch: Classical feature batch of shape
                ``(n_samples, len(data_param_indices))``. Mutually
                required with ``data_param_indices`` (or ``data_arg``).
            arg_shapes: For multi-argument or structured-shape **QNode**
                inputs (e.g. ``circuit(inputs, weights)`` with
                ``StronglyEntanglingLayers``), maps each array argument's name
                to its shape so the QNode can be traced symbolically. The
                ``data_arg``'s shape is filled in automatically from
                ``feature_batch``. Only valid for QNode inputs.
            data_arg: Name of the QNode argument fed from ``feature_batch``
                (the data axis), as an alternative to ``data_param_indices``
                for multi-argument QNodes. The remaining arguments' parameters
                are the trainable weights. Requires ``feature_batch`` and a
                QNode input; mutually exclusive with ``data_param_indices``.
            labels: Optional supervised targets of shape ``(n_samples,)``,
                aligned with ``feature_batch``'s rows. When given (data binding
                must be active), each sample's prediction (the observable's
                expectation value) is compared to its label via ``loss_fn``
                before ``loss_reduction`` aggregates — turning the unsupervised
                objective into a supervised training loss. ``None`` (default)
                keeps the unsupervised behavior.
            loss_fn: Per-sample supervised loss ``(prediction, label) ->
                float``, used only when ``labels`` is given. ``"squared_error"``
                (default) with the default ``"mean"`` reduction yields
                mean-squared error; pass a callable for a custom loss. A custom
                callable must return a finite value — a NaN/Inf loss propagates
                to the optimizer with no diagnostic.
            loss_reduction: How to aggregate the per-sample values (expectation
                values when unsupervised, or per-sample losses when ``labels``
                is set) into the scalar loss the optimizer sees. ``"mean"``
                (default), ``"sum"``, or a callable ``(n_samples,) -> float``.
                Ignored when ``data_param_indices`` is unset.
            max_iterations: Maximum number of optimization iterations.
            **kwargs: Additional keyword arguments passed to the parent
                class, including backend and optimizer.

        Raises:
            TypeError: If ``qscript`` is not a supported type.
            ValueError: If the input has an invalid measurement, no
                trainable parameters, or inconsistent data-binding args.
        """
        super().__init__(**kwargs)

        if data_arg is not None and data_param_indices is not None:
            raise ValueError(
                "Specify the data axis with either data_arg (by argument name) "
                "or data_param_indices (by index), not both."
            )
        if (arg_shapes is not None or data_arg is not None) and not isinstance(
            qscript, QNode
        ):
            raise ValueError("arg_shapes and data_arg are only valid for QNode inputs.")

        if isinstance(qscript, QNode):
            if data_arg is None and data_param_indices is None:
                data_arg = self._infer_data_arg_from_batch_input(qscript)

            sig_args = list(inspect.signature(qscript.func).parameters)
            unknown = set(arg_shapes or {}) - set(sig_args)
            if unknown:
                raise ValueError(
                    f"arg_shapes names {sorted(unknown)} that are not QNode "
                    f"arguments; expected a subset of {sig_args}."
                )
            if data_arg is not None and data_arg not in sig_args:
                raise ValueError(
                    f"data_arg {data_arg!r} is not a QNode argument; expected "
                    f"one of {sig_args}."
                )
            shapes = dict(arg_shapes) if arg_shapes else {}
            if data_arg is not None:
                if feature_batch is None:
                    raise ValueError("data_arg requires feature_batch.")
                n_features = np.atleast_2d(np.asarray(feature_batch)).shape[1]
                shapes.setdefault(data_arg, (n_features,))
            qscript = _qnode_to_symbolic_qscript(
                qscript, arg_shapes=shapes if shapes else None
            )

        self.qscript = qscript
        self.n_layers = 1
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.measured_wires: tuple[int, ...] = ()
        self._qiskit_circuit: QuantumCircuit | None = None
        # Set by ``_prepare_qiskit_input`` / ``_prepare_pennylane_input``
        # before they return; both the cost-pipeline factory and
        # DataBindingStage read it.
        self._composed_circuit: QuantumCircuit

        if isinstance(qscript, QuantumCircuit):
            base_params = self._prepare_qiskit_input(qscript)
        elif isinstance(qscript, qp.tape.QuantumScript):
            base_params = self._prepare_pennylane_input(qscript)
        else:
            raise TypeError(
                "qscript must be a PennyLane QuantumScript, PennyLane "
                "QNode, or a Qiskit QuantumCircuit."
            )

        if not len(base_params):
            raise ValueError("QuantumScript does not contain any trainable parameters.")

        self._base_params = base_params

        if data_arg is not None:
            data_param_indices = [
                i
                for i, p in enumerate(base_params)
                if _symbol_arg_name(str(p)) == data_arg
            ]
            if not data_param_indices:
                raise ValueError(
                    f"data_arg {data_arg!r} contributed no trainable parameters; "
                    f"check the argument name and arg_shapes."
                )

        self._configure_data_binding(
            data_param_indices=data_param_indices,
            feature_batch=feature_batch,
            loss_reduction=loss_reduction,
            param_shape=param_shape,
            labels=labels,
            loss_fn=loss_fn,
        )

        self._pipelines = self._build_pipelines()

    @staticmethod
    def _infer_data_arg_from_batch_input(qnode: QNode) -> str | None:
        """Read the data axis from a ``@qml.batch_input`` decorator, if present.

        Returns the single batched argument name, or ``None`` when the QNode
        has no detectable batch_input transform. Raises if more than one
        argument is batched, since a single data axis is supported.
        """
        detected = _detect_batch_input_argnames(qnode)
        if not detected:
            return None
        if len(set(detected)) > 1:
            raise ValueError(
                f"@qml.batch_input marks multiple batched arguments {detected}; "
                f"pass data_arg explicitly (one data axis is supported)."
            )
        return detected[0]

    @property
    def n_params_per_layer(self) -> int:
        return int(np.prod(self._param_shape))

    @property
    def param_shape(self) -> tuple[int, ...]:
        """Shape of a single parameter set."""
        return self._param_shape

    def _set_cost_hamiltonian(self, observable: SparsePauliOp) -> None:
        """Clean and store ``cost_hamiltonian`` / ``loss_constant``.

        Raises if the observable reduces to a constant after stripping
        Identity terms — CustomVQA has nothing to optimize over.
        """
        self.cost_hamiltonian, self.loss_constant = _clean_hamiltonian_spo(
            observable, raise_on_constant=True
        )

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
        self._composed_circuit = self._qiskit_circuit
        self._set_cost_hamiltonian(_z_sum_observable(qc.num_qubits, measured_wires))
        return np.array(list(qc.parameters), dtype=object)

    def _prepare_pennylane_input(self, qs: qp.tape.QuantumScript) -> np.ndarray:
        """Set state for a PennyLane ``QuantumScript`` input; return base parameters.

        Trainable operation parameters are made symbolic, then a single
        :func:`~divi.circuits.qscript_to_meta` conversion yields the observable,
        the composed circuit, and the Qiskit parameter vector — the conversion
        is the sole authority for sympy→Qiskit, dedup, and ordering.
        """
        _validate_single_measurement(
            qs,
            allowed=(ExpectationMP,),
            caller="CustomVQA",
            description="expectation-value (expval())",
        )
        if qs.measurements[0].obs is None:
            raise ValueError(
                "CustomVQA requires the QuantumScript's expectation-value "
                "measurement to declare an observable; got expval() with "
                "obs=None."
            )
        self.n_qubits = qs.num_wires

        meta = qscript_to_meta(
            self._symbolize_trainable_ops(qs), precision=self._precision
        )
        observable = meta.observable
        if observable is None:
            raise ValueError("Converted QuantumScript has no observable to optimize.")
        self._set_cost_hamiltonian(observable[0])
        self._composed_circuit = dag_to_circuit(meta.circuit_bodies[0][1])
        return np.array(meta.parameters, dtype=object)

    def _symbolize_trainable_ops(
        self, qs: qp.tape.QuantumScript
    ) -> qp.tape.QuantumScript:
        """Ensure the trainable operation parameters are symbolic before conversion.

        A QNode-derived script is already symbolic and passes through unchanged
        (its ties and expressions intact). A fully concrete script has its
        trainable operation slots replaced with fresh symbols so the conversion
        exposes them as parameters; concrete slots that are not trainable stay
        baked in.
        """
        n_op_params = sum(len(op.data) for op in qs.operations)
        trainable = [i for i in qs.trainable_params if i < n_op_params]
        if qs.trainable_params and not trainable:
            raise ValueError(
                "QuantumScript's trainable_params point only at observable "
                "coefficients; CustomVQA only trains operation parameters. "
                "Remove observable-coefficient indices from qs.trainable_params."
            )
        values = qs.get_parameters(trainable_only=False)
        already_symbolic = any(
            isinstance(values[i], (sp.Expr, ParameterExpression)) for i in trainable
        )
        if already_symbolic or not trainable:
            return qs
        symbols = sp.symbols(f"p0:{len(trainable)}")
        return qs.bind_new_parameters(list(symbols), trainable)

    def _build_pipelines(self) -> dict:
        return {"cost": self._build_cost_pipeline(CircuitSpecStage())}

    def _configure_data_binding(
        self,
        *,
        data_param_indices: Sequence[int] | None,
        feature_batch: npt.ArrayLike | None,
        loss_reduction: LossReductionFn,
        param_shape: tuple[int, ...] | int | None,
        labels: npt.ArrayLike | None,
        loss_fn: SampleLossFn,
    ) -> None:
        """Validate the data-binding kwargs and populate the derived state.

        Sets ``_param_shape``, ``_param_symbols``, and — when data binding is
        active (``feature_batch is not None``) — the ``_data_symbols`` /
        ``_weight_symbols`` partition and the optional ``labels`` /
        ``_sample_loss_fn`` supervised state used by ``DataBindingStage``.
        """
        n_params = len(self._base_params)
        self.labels = None
        self._sample_loss_fn = None

        if (data_param_indices is None) != (feature_batch is None):
            raise ValueError(
                "data_param_indices and feature_batch must both be provided "
                "or both be None."
            )

        if data_param_indices is None:
            if labels is not None:
                raise ValueError(
                    "labels require data binding; provide feature_batch with "
                    "data_arg or data_param_indices."
                )
            # No data axis: nothing to supervise, but still warn if loss_fn was
            # set, since it is ignored.
            self.labels, self._sample_loss_fn = self._resolve_supervision(
                None, loss_fn, 0
            )
            self.feature_batch = None
            self.loss_reduction = loss_reduction
            self._param_shape = self._resolve_param_shape(param_shape, n_params)
            self._param_symbols = self._base_params.reshape(self._param_shape)
            return

        if param_shape is not None:
            raise ValueError(
                "param_shape is not supported when data_param_indices is set; "
                "the optimizer view is automatically flat over the remaining "
                "weight parameters."
            )

        data_indices = self._validate_data_indices(data_param_indices, n_params)
        data_set = set(data_indices)
        weight_indices = [i for i in range(n_params) if i not in data_set]
        if not weight_indices:
            raise ValueError(
                "data_param_indices marks every circuit parameter — no "
                "trainable weights left for the optimizer."
            )

        self._data_symbols = tuple(self._base_params[i] for i in data_indices)
        self._weight_symbols = tuple(self._base_params[i] for i in weight_indices)
        # feature_batch is non-None here: the XOR guard above pairs it with
        # data_param_indices, which is set in this branch.
        self.feature_batch = self._validate_feature_batch(
            cast(npt.ArrayLike, feature_batch), len(self._data_symbols)
        )
        self._loss_reduction_fn = resolve_loss_reduction(loss_reduction)
        self.loss_reduction = loss_reduction
        self.labels, self._sample_loss_fn = self._resolve_supervision(
            labels, loss_fn, self.feature_batch.shape[0]
        )
        self._param_shape = (len(self._weight_symbols),)
        self._param_symbols = np.asarray(self._weight_symbols, dtype=object)
        self._loss_constant_consumed = True

    @staticmethod
    def _validate_data_indices(
        data_param_indices: Sequence[int], n_params: int
    ) -> list[int]:
        raw = list(data_param_indices)
        if not raw:
            raise ValueError("data_param_indices must contain at least one index.")
        indices: list[int] = []
        for idx in raw:
            # Accept numpy integers (np.arange / np.where outputs are natural
            # inputs) but reject bools and non-integers; coerce to plain int.
            if isinstance(idx, bool) or not isinstance(idx, (int, np.integer)):
                raise TypeError(
                    f"data_param_indices entries must be ints; got {idx!r}."
                )
            indices.append(int(idx))
        if len(set(indices)) != len(indices):
            raise ValueError(f"data_param_indices has duplicate entries: {indices}.")
        for idx in indices:
            if idx < 0 or idx >= n_params:
                raise ValueError(
                    f"data_param_indices index {idx} is out of range for a "
                    f"circuit with {n_params} parameters."
                )
        return indices

    @staticmethod
    def _validate_feature_batch(
        feature_batch: npt.ArrayLike, n_data_params: int
    ) -> np.ndarray:
        arr = np.asarray(feature_batch, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(
                f"feature_batch must be 2D (n_samples, n_data_params); got "
                f"shape {arr.shape}."
            )
        if arr.shape[1] != n_data_params:
            raise ValueError(
                f"feature_batch has {arr.shape[1]} columns but "
                f"data_param_indices declares {n_data_params}."
            )
        if arr.shape[0] == 0:
            raise ValueError("feature_batch must contain at least one sample.")
        return arr

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
        """Create the cost meta-circuit factory for CustomVQA.

        Uniform across input paths and data-binding mode:
        ``_composed_circuit`` always carries the full operation-parametric
        circuit, ``_base_params`` is the canonical parameter ordering, and
        ``cost_hamiltonian`` is the cleaned observable (Identity terms
        already factored into ``loss_constant``). DataBindingStage, when
        active, swaps the data parameters out into per-sample variants
        downstream.
        """
        return {
            "cost_circuit": MetaCircuit(
                circuit_bodies=(((), circuit_to_dag(self._composed_circuit)),),
                parameters=tuple(self._base_params),
                observable=self.cost_hamiltonian,
                precision=self._precision,
            )
        }
