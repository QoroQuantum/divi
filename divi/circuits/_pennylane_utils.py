# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""PennyLane adapter helpers: trace ``QNode``\\ s into symbolic
``QuantumScript``\\ s, convert them to :class:`~divi.circuits.MetaCircuit`,
and validate their measurements."""

import inspect
import warnings
from collections.abc import Callable, Mapping

import numpy as np
import pennylane as qp
import sympy as sp
from pennylane.tape import QuantumScript
from pennylane.workflow.qnode import QNode

from divi.circuits._conversions import _PL_TO_QISKIT_GATE, qscript_to_meta
from divi.circuits._core import DEFAULT_PRECISION, MetaCircuit

_PROBE_SIZE = 100

#: Shared hint appended to QNode-conversion failures. Single 1-D-array
#: templates (e.g. ``AngleEmbedding``, ``IQPEmbedding``) are supported, but a
#: structured multi-dimensional weight shape (e.g.
#: ``StronglyEntanglingLayers``'s ``(layers, wires, 3)``) or multiple array
#: arguments can't have their shapes inferred from the function signature.
_SHAPE_HINT = (
    "Automatic conversion couldn't infer the array shape(s). This happens with "
    "templates that need a structured multi-dimensional shape (e.g. "
    "StronglyEntanglingLayers) or with multiple array arguments. Declare the "
    "shapes explicitly — pass arg_shapes={'<arg>': <shape>, ...} (to "
    "qnode_to_meta, or via CustomVQA's arg_shapes/data_arg) — or pass a "
    "QuantumScript with sympy symbols."
)


def _warn_on_device_settings(qnode: QNode) -> None:
    """Warn when a QNode declares a shot count divi will not honor.

    Divi runs every QNode against its own configured backend with its own
    shot count, so device-level shots are silently ignored — flag them so
    users don't expect them to take effect.
    """
    if getattr(qnode.device, "shots", None):
        warnings.warn(
            "QNode device declares a shot count; divi ignores it and uses "
            "the backend's configured shots instead. Set shots on the divi "
            "backend (e.g. MaestroSimulator(shots=...)) to control sampling.",
            UserWarning,
            stacklevel=3,
        )


def _mark_symbolic_params_trainable(
    qscript: QuantumScript, *, signature_has_defaults: bool
) -> QuantumScript:
    """Restrict ``trainable_params`` to the symbolic gate-parameter slots.

    ``make_qscript`` marks every gate parameter trainable (sympy symbols carry
    no ``requires_grad``, so PennyLane falls back to "all trainable"). Marking
    only the slots whose value is a sympy expression leaves frozen
    Python-default values out of the trainable set, as PennyLane does for
    plain-default arguments.

    When the signature had default-valued parameters *and* a non-symbolic
    (frozen) value reached a gate, that default angle is being baked in as a
    constant; warn so the user isn't surprised it never trains.
    """
    full_params = qscript.get_parameters(trainable_only=False)
    symbolic = [i for i, p in enumerate(full_params) if isinstance(p, sp.Basic)]
    if signature_has_defaults and len(symbolic) < len(full_params):
        warnings.warn(
            "A default-valued QNode parameter was baked into a gate as a fixed "
            "constant and will not be trained. Remove the default (pass the value "
            "as a required argument) to make it trainable.",
            UserWarning,
            stacklevel=3,
        )
    qscript.trainable_params = symbolic
    return qscript


def _qnode_to_symbolic_qscript(
    qnode: QNode,
    *,
    arg_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> QuantumScript:
    """Convert a ``QNode`` to a ``QuantumScript`` with sympy placeholders.

    Walks the QNode's function signature. Parameters **without** a Python
    default become sympy placeholders (``p0``, ``p1``, ...) and are marked
    trainable; parameters **with** a default keep their declared value and
    are left non-trainable — matching PennyLane, where a plain-Python-default
    argument has ``requires_grad=False`` and is excluded from
    ``trainable_params``. So ``def f(x, n_layers=3)`` is invoked as ``f(p0)``
    with ``n_layers=3`` fixed, and ``def f(theta, phi=0.5)`` trains only
    ``theta``. For a single positional parameter that defies scalar
    invocation, the function falls back to probing with a dummy numpy array.

    Single 1-D-array PennyLane **templates** are supported, including
    nonlinear encoders. A numpy *object* array of sympy symbols reports the
    ``numpy`` interface (bare symbols report ``sympy``, which PennyLane's
    array math has no backend for), so the template traces symbolically;
    decomposing preserves the full angle expression — e.g. ``AngleEmbedding``
    yields ``RY(p_i)`` and ``IQPEmbedding`` yields ``MultiRZ(p_i*p_j)``, which
    divi's sympy→Qiskit conversion turns into ``ParameterExpression``\\ s.

    When ``arg_shapes`` is given, **multiple array arguments** and structured
    multi-dimensional shapes (e.g. ``StronglyEntanglingLayers``'s
    ``(layers, wires, 3)``) are supported: each no-default argument is seeded
    with a symbol array of its declared shape, named ``<arg>__<i>`` so callers
    can map gate parameters back to their originating argument (e.g. to split
    data inputs from trainable weights). Without ``arg_shapes``, structured
    shapes and multi-argument signatures can't be inferred and raise a clear
    error.

    **Structural arguments** (qubit counts, layer counts, and other values used
    only for control flow — ``range(n_qubits)``, loop bounds — never as gate
    angles) are neither data nor weights. Supply them as a Python default or
    close over them in the enclosing scope; a no-default structural argument is
    seeded as a sympy symbol like any other and then breaks control flow (e.g.
    ``range(<symbol>)``). The circuit is traced **one sample at a time**, so the
    data argument is the per-sample shape (1-D for a flat feature vector) —
    index by the structural size (``range(n_qubits)``), not the batch dimension
    (``len(inputs[0])``).

    Args:
        qnode: A PennyLane ``QNode``. Without ``arg_shapes`` its function may
            take any number of scalar parameters or a single array parameter.
        arg_shapes: Optional map of argument name → shape tuple. Supplying it
            enables multi-argument and structured-shape conversion. Every
            *array-valued* argument must be listed — a no-default argument
            absent from the map is seeded as a single scalar symbol, which
            fails for a template/array consumer. (``CustomVQA`` fills in the
            ``data_arg`` shape from ``feature_batch`` automatically.)

    Returns:
        A ``QuantumScript`` with each trainable parameter slot replaced by a
        sympy symbol and ``trainable_params`` restricted to those slots. Symbol
        names are ``p0``/``p[i]`` for the inferred paths, or ``<arg>__<i>``
        when ``arg_shapes`` is given.

    Raises:
        TypeError: If the QNode's parameter shape cannot be reconciled with
            the chosen path.
    """
    if arg_shapes is not None:
        return _qnode_to_qscript_with_shapes(qnode, arg_shapes)

    _warn_on_device_settings(qnode)
    sig = inspect.signature(qnode.func)
    n_params = sum(
        1 for p in sig.parameters.values() if p.default is inspect.Parameter.empty
    )
    has_defaults = any(
        p.default is not inspect.Parameter.empty for p in sig.parameters.values()
    )
    symbols = sp.symbols(f"p0:{n_params}")  # always a tuple (empty when n=0)

    # Phase 1: try scalar symbols. Defaults on the QNode signature stay
    # at their declared values (no symbol substitution), so non-numeric
    # defaults like ``n_layers=3`` survive the conversion intact. Any failure
    # (templates raise ``IndexError``/``ValueError`` on a scalar symbol, or
    # ``ImportError`` when array math has no sympy backend) routes to the
    # array-probe / decompose paths rather than leaking the internal error.
    try:
        return _mark_symbolic_params_trainable(
            qp.tape.make_qscript(qnode.func)(*symbols),
            signature_has_defaults=has_defaults,
        )
    except Exception as exc:
        # Array-parameter QNodes legitimately fail scalar tracing and fall
        # through to the array probes below; keep the cause so it can be
        # chained into the final error rather than discarded.
        scalar_trace_error = exc

    # Phase 2: single trainable parameter — probe to discover array size.
    if n_params != 1:
        raise TypeError(
            "Failed to convert QNode — the function appears to use array "
            "parameters or numpy operations on its arguments. QNodes with "
            "multiple array parameters are not supported. Pass a "
            "QuantumScript with explicit sympy symbols instead."
        ) from scalar_trace_error

    # 2a: flat-array symbol substitution — handles manual indexing
    # (``weights[i]``) and preserves controlled-gate structure (the symbolic
    # gates are decomposed downstream). Falls through on any failure.
    flat_qs = _try_flat_array_symbols(qnode.func, signature_has_defaults=has_defaults)
    if flat_qs is not None:
        return flat_qs

    # 2b: single 1-D-array templates (``AngleEmbedding``, ``IQPEmbedding``, ...).
    # Trace with a numpy object array of symbols so the template runs
    # symbolically, then decompose — preserving angle expressions (including
    # nonlinear ones). Falls through to the clear error for structured shapes.
    template_qs = _try_symbolic_template_single_array(
        qnode, signature_has_defaults=has_defaults
    )
    if template_qs is not None:
        return template_qs

    raise TypeError(
        "Failed to convert QNode with array parameter. " + _SHAPE_HINT
    ) from scalar_trace_error


def qnode_to_meta(
    qnode: QNode,
    *,
    arg_shapes: Mapping[str, tuple[int, ...]] | None = None,
    precision: int = DEFAULT_PRECISION,
) -> MetaCircuit:
    """Convert a ``QNode`` directly to a :class:`~divi.circuits.MetaCircuit`.

    Traces the QNode into a symbolic ``QuantumScript`` (trainable arguments
    seeded with sympy symbols) and applies :func:`qscript_to_meta`. The
    ``arg_shapes`` mapping declares per-argument shapes for multi-argument and
    structured-shape circuits whose shapes can't be inferred automatically.
    """
    return qscript_to_meta(
        _qnode_to_symbolic_qscript(qnode, arg_shapes=arg_shapes),
        precision=precision,
    )


def _symbol_arg_name(symbol_name: str) -> str:
    """Return the originating argument of a ``<arg>__<i>`` conversion symbol.

    Symbols produced with ``arg_shapes`` are named ``<arg>__<flat index>``;
    this strips the trailing ``__<i>`` so callers can group parameters by the
    argument they came from (e.g. to separate data inputs from weights).
    """
    return symbol_name.rsplit("__", 1)[0]


def _detect_batch_input_argnames(qnode: QNode) -> list[str]:
    """Return the argument names a ``@qml.batch_input`` transform batches.

    Reads the QNode's ``compile_pipeline``, matching the batch_input transform
    by identity against ``qml.batch_input.tape_transform`` and mapping each
    batched ``argnum`` to its argument name. Any failure to introspect the
    pipeline yields ``[]``.
    """
    try:
        program = qnode.compile_pipeline
        signature_names = list(inspect.signature(qnode.func).parameters)
    except Exception:
        return []

    target = qp.batch_input.tape_transform

    def _is_batch_input(container) -> bool:
        return getattr(container, "tape_transform", None) is target

    batched: list[str] = []
    for container in program:
        if not _is_batch_input(container):
            continue
        kwargs = getattr(container, "kwargs", None) or {}
        argnum = kwargs.get("argnum")
        if argnum is None:
            args = getattr(container, "args", ())
            argnum = args[0] if args else None
        if argnum is None:
            continue
        indices = [argnum] if isinstance(argnum, int) else list(argnum)
        for i in indices:
            if isinstance(i, int) and 0 <= i < len(signature_names):
                batched.append(signature_names[i])
    return batched


def _qnode_to_qscript_with_shapes(
    qnode: QNode, arg_shapes: Mapping[str, tuple[int, ...]]
) -> QuantumScript:
    """Convert a QNode to a symbolic ``QuantumScript`` using per-argument shapes.

    Each no-default function argument is seeded with a numpy *object* array of
    sympy symbols named ``<arg>__<i>`` of the declared shape (scalar args, or
    args omitted from ``arg_shapes``, get a single bare symbol). The object
    array reports the ``numpy`` interface, so templates — including those with
    structured multi-dimensional weights — trace symbolically; the result is
    decomposed to divi's gate set and 0-d object-array params are unwrapped.
    The ``<arg>__<i>`` naming lets callers map parameters back to arguments via
    :func:`_symbol_arg_name`.
    """
    _warn_on_device_settings(qnode)
    sig = inspect.signature(qnode.func)
    trainable_args = [
        name
        for name, p in sig.parameters.items()
        if p.default is inspect.Parameter.empty
    ]

    call_args: list = []
    for name in trainable_args:
        shape = tuple(arg_shapes.get(name, ()))
        count = int(np.prod(shape)) if shape else 1
        symbols = sp.symbols(f"{name}__0:{count}")  # always a tuple for "0:n"
        if shape:
            call_args.append(np.array(symbols, dtype=object).reshape(shape))
        else:
            call_args.append(symbols[0])  # scalar arg → bare symbol

    has_defaults = any(
        p.default is not inspect.Parameter.empty for p in sig.parameters.values()
    )
    try:
        return _decompose_and_mark(
            qp.tape.make_qscript(qnode.func)(*call_args),
            signature_has_defaults=has_defaults,
        )
    except Exception as e:
        raise TypeError("Failed to convert QNode. " + _SHAPE_HINT) from e


def _unwrap_object_array_params(qscript: QuantumScript) -> QuantumScript:
    """Replace 0-d numpy object-array gate params with their scalar element.

    Tracing a template with a multi-dimensional symbol array (e.g.
    ``StronglyEntanglingLayers``'s ``(layers, wires, 3)``) wraps some gate
    angles in 0-d ``ndarray``\\ s; this returns them as the bare sympy
    expression. Bare-symbol traces (1-D inputs) have nothing wrapped and pass
    through unchanged.
    """

    def unwrap(value):
        if isinstance(value, np.ndarray) and value.ndim == 0:
            return value.item()
        return value

    for op in qscript.operations:
        if any(isinstance(d, np.ndarray) and d.ndim == 0 for d in op.data):
            op.data = tuple(unwrap(d) for d in op.data)
    return qscript


def _decompose_and_mark(
    qscript: QuantumScript, *, signature_has_defaults: bool
) -> QuantumScript:
    """Decompose to divi's gate set, unwrap 0-d params, then mark trainable symbols."""
    [qscript], _ = qp.transforms.decompose(qscript, gate_set=set(_PL_TO_QISKIT_GATE))
    qscript = _unwrap_object_array_params(qscript)
    return _mark_symbolic_params_trainable(
        qscript, signature_has_defaults=signature_has_defaults
    )


def _try_flat_array_symbols(
    func: Callable[..., object], *, signature_has_defaults: bool
) -> QuantumScript | None:
    """Probe a single-array QNode function with a flat dummy array, then bind a
    matching sympy array. Handles manual ``weights[i]`` indexing. Returns
    ``None`` on any failure (e.g. shape-specific templates) so the caller can
    try the symbolic-template path."""
    try:
        probe_qs = qp.tape.make_qscript(func)(np.zeros(_PROBE_SIZE))
    except Exception:
        return None
    n_gate_params = len(probe_qs.get_parameters())
    if n_gate_params == 0 or n_gate_params >= _PROBE_SIZE:
        return None
    sym_array = sp.symarray("p", (n_gate_params,))
    try:
        return _mark_symbolic_params_trainable(
            qp.tape.make_qscript(func)(sym_array),
            signature_has_defaults=signature_has_defaults,
        )
    except Exception:
        return None


def _try_symbolic_template_single_array(
    qnode: QNode, *, signature_has_defaults: bool
) -> QuantumScript | None:
    """Convert a single-array template QNode by symbolic tracing.

    Traces ``qnode.func`` with a numpy *object* array of sympy symbols sized to
    the device wire count. The object array reports the ``numpy`` interface, so
    the template's internal ``qml.math`` calls run and the gate angles come out
    as sympy expressions — covering 1-D encoders (``AngleEmbedding``,
    ``IQPEmbedding``, ...) and preserving nonlinear angle maps. Decomposing to
    divi's gate set yields per-gate scalar symbols/expressions for binding.

    Returns ``None`` (so the caller raises the clear shape error) when the
    wire count can't seed the array or the template needs a structured shape
    (e.g. ``StronglyEntanglingLayers``'s ``(layers, wires, 3)``), which
    trace-fails at a 1-D shape.
    """
    n_wires = len(qnode.device.wires)
    if n_wires == 0:
        return None
    symbols = np.array(sp.symbols(f"p0:{n_wires}"), dtype=object)
    try:
        qscript = _decompose_and_mark(
            qp.tape.make_qscript(qnode.func)(symbols),
            signature_has_defaults=signature_has_defaults,
        )
    except Exception:
        return None
    return qscript if qscript.get_parameters() else None


def _validate_single_measurement(
    qscript: QuantumScript,
    *,
    allowed: tuple[type, ...],
    caller: str,
    description: str | None = None,
) -> None:
    """Validate that ``qscript`` has exactly one measurement of an allowed type.

    Args:
        qscript: The PennyLane ``QuantumScript`` to validate.
        allowed: Tuple of measurement-process classes (e.g.
            ``(ExpectationMP,)`` for strict expval-only callers, or
            ``(ProbabilityMP, ExpectationMP, CountsMP)`` for the general
            pipeline spec stage).
        caller: Name of the calling context, used in the error message
            so users see which surface rejected their input.
        description: Human-readable description of the allowed measurement
            forms (e.g. ``"probs(), expval(), or counts()"``). When
            ``None``, defaults to the class names from ``allowed``.

    Raises:
        ValueError: If the script has anything other than exactly one
            measurement of an allowed type.
    """
    measurements = qscript.measurements
    if len(measurements) != 1 or not isinstance(measurements[0], allowed):
        names = [type(m).__name__ for m in measurements]
        if description is None:
            description = ", ".join(cls.__name__ for cls in allowed)
        raise ValueError(
            f"{caller} requires exactly one measurement of type "
            f"{description}. Got: {names}"
        )
