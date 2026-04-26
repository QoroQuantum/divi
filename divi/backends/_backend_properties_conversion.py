# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Utilities for working with Qiskit BackendProperties and BackendV2 conversion."""

import datetime
from typing import Any

from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.providers.backend import QubitProperties
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import InstructionProperties, Target

# Unit prefixes used by the BackendProperties schema. Empty string is treated
# as the SI base unit (seconds, hertz) so callers can omit the ``unit`` field
# and rely on ``_normalize_nduv`` to fill it in.
_TIME_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12, "": 1.0}
_FREQUENCY_UNITS = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9, "thz": 1e12, "": 1.0}


def _to_seconds(value: float, unit: str) -> float:
    factor = _TIME_UNITS.get(unit.lower())
    if factor is None:
        raise ValueError(f"Unsupported time unit: {unit!r}")
    return float(value) * factor


def _to_hertz(value: float, unit: str) -> float:
    factor = _FREQUENCY_UNITS.get(unit.lower())
    if factor is None:
        raise ValueError(f"Unsupported frequency unit: {unit!r}")
    return float(value) * factor


def _normalize_properties(
    properties: dict[str, Any],
    default_date: datetime.datetime | None = None,
) -> dict[str, Any]:
    """
    Preprocess an incomplete BackendProperties dictionary by filling in missing
    required fields with sensible defaults.

    This function makes it easier to create BackendProperties dictionaries by
    allowing you to omit fields that have obvious defaults, such as:
    - Missing top-level fields: `backend_name`, `backend_version`, `last_update_date`
    - Missing `unit` field for dimensionless parameters (e.g., gate_error)
    - Missing `general` field (empty list)
    - Missing `gates` field (empty list)
    - Missing `qubits` field (empty list)
    - Missing `date` fields in Nduv objects

    Args:
        properties: Incomplete BackendProperties dictionary. Can omit:
            - `unit` field in parameter/qubit Nduv objects (defaults to "" for
              dimensionless quantities like gate_error, or inferred from name)
            - `general` field (defaults to empty list)
            - `gates` field (defaults to empty list)
            - `qubits` field (defaults to empty list)
            - `date` field in Nduv objects (defaults to current time or provided default)
        default_date: Optional datetime to use for missing date fields.
            If None, uses current time.

    Returns:
        Complete BackendProperties dictionary ready for downstream consumption.

    Example:
        >>> props = {
        ...     "backend_name": "test",
        ...     "gates": [{
        ...         "gate": "sx",
        ...         "qubits": [0],
        ...         "parameters": [{
        ...             "name": "gate_error",
        ...             "value": 0.01,
        ...             # unit and date will be added automatically
        ...         }]
        ...     }]
        ... }
        >>> normalized = _normalize_properties(props)
    """
    if default_date is None:
        default_date = datetime.datetime.now()

    # Create a shallow copy to avoid mutating the input
    # (nested structures are rebuilt below to ensure no mutation)
    normalized = properties.copy()

    # Add missing required top-level fields
    if "backend_name" not in normalized:
        normalized["backend_name"] = "custom_backend"
    if "backend_version" not in normalized:
        normalized["backend_version"] = "1.0.0"
    if "last_update_date" not in normalized:
        normalized["last_update_date"] = default_date

    # Add missing general field
    if "general" not in normalized:
        normalized["general"] = []

    # Add missing gates field (required by BackendProperties)
    if "gates" not in normalized:
        normalized["gates"] = []

    # Add missing qubits field (required by BackendProperties)
    if "qubits" not in normalized:
        normalized["qubits"] = []

    # Normalize qubits (list of lists of Nduv objects)
    if "qubits" in normalized:
        normalized["qubits"] = [
            [_normalize_nduv(param, default_date) for param in qubit_params]
            for qubit_params in normalized["qubits"]
        ]

    # Normalize gates (list of gate dicts with parameters)
    if "gates" in normalized:
        normalized["gates"] = [
            {
                **gate,
                "parameters": [
                    _normalize_nduv(param, default_date)
                    for param in gate.get("parameters", [])
                ],
            }
            for gate in normalized["gates"]
        ]

    # Normalize general (list of Nduv objects)
    if "general" in normalized and normalized["general"]:
        normalized["general"] = [
            _normalize_nduv(param, default_date) for param in normalized["general"]
        ]

    return normalized


def _normalize_nduv(
    nduv: dict[str, Any], default_date: datetime.datetime
) -> dict[str, Any]:
    """
    Normalize a single Nduv (Name, Date, Unit, Value) object by adding
    missing required fields.

    Args:
        nduv: Nduv dictionary (may be incomplete)
        default_date: Default date to use if missing

    Returns:
        Complete Nduv dictionary
    """
    normalized = nduv.copy()

    # Add missing date field
    if "date" not in normalized:
        normalized["date"] = default_date

    # Add missing unit field
    if "unit" not in normalized:
        name = normalized.get("name", "").lower()
        # Dimensionless quantities
        if name in ("gate_error", "readout_error", "prob"):
            normalized["unit"] = ""
        # Time-based quantities
        elif name in ("t1", "t2", "gate_length", "readout_length"):
            # Infer unit from common patterns, default to "ns" for gate_length
            if name == "gate_length":
                normalized["unit"] = "ns"
            elif name in ("t1", "t2"):
                normalized["unit"] = "us"  # microseconds is common
            else:
                normalized["unit"] = "ns"
        # Frequency-based quantities
        elif name in ("frequency", "freq"):
            normalized["unit"] = "GHz"
        # Default to empty string for unknown quantities
        else:
            normalized["unit"] = ""

    return normalized


def _qubit_calibration_from_nduv_list(
    nduv_list: list[dict[str, Any]],
) -> tuple[QubitProperties, float | None, float | None]:
    """Extract per-qubit calibration from a list of Nduv dicts.

    Returns:
        ``(qubit_properties, readout_error, readout_length_seconds)``.
        Each readout field is ``None`` when the corresponding Nduv entry is
        missing from the input.
    """
    qp_kwargs: dict[str, float] = {}
    readout_error: float | None = None
    readout_length_s: float | None = None
    for nduv in nduv_list:
        name = str(nduv.get("name", "")).lower()
        value = nduv["value"]
        unit = str(nduv.get("unit", ""))
        if name == "t1":
            qp_kwargs["t1"] = _to_seconds(value, unit)
        elif name == "t2":
            qp_kwargs["t2"] = _to_seconds(value, unit)
        elif name in ("frequency", "freq"):
            qp_kwargs["frequency"] = _to_hertz(value, unit)
        elif name == "readout_error":
            readout_error = float(value)
        elif name == "readout_length":
            readout_length_s = _to_seconds(value, unit)
    return QubitProperties(**qp_kwargs), readout_error, readout_length_s


def _gate_calibration_from_nduv_list(
    nduv_list: list[dict[str, Any]],
) -> tuple[float | None, float | None]:
    """Extract ``(duration_seconds, error)`` from gate parameter Nduv dicts.

    Either field may be ``None`` when the corresponding entry is absent.
    """
    duration_s: float | None = None
    error: float | None = None
    for nduv in nduv_list:
        name = str(nduv.get("name", "")).lower()
        value = nduv["value"]
        unit = str(nduv.get("unit", ""))
        if name == "gate_length":
            duration_s = _to_seconds(value, unit)
        elif name == "gate_error":
            error = float(value)
    return duration_s, error


def _build_target_from_normalized(
    normalized_properties: dict[str, Any], n_qubits: int
) -> Target:
    """Build a :class:`~qiskit.transpiler.Target` from a normalized properties dict.

    The returned target carries the user's calibration into every consumer
    that reads ``backend.target`` — most importantly the transpiler and
    :func:`qiskit_aer.noise.NoiseModel.from_backend`. Qubits beyond the
    length of ``properties["qubits"]`` are filled with default
    :class:`QubitProperties`; gates not listed in ``properties["gates"]``
    are not added.
    """
    qubits_list = normalized_properties.get("qubits", [])
    qubit_properties: list[QubitProperties] = []
    readout_props: dict[tuple[int, ...], InstructionProperties] = {}
    for q in range(n_qubits):
        if q < len(qubits_list):
            qp, readout_error, readout_length_s = _qubit_calibration_from_nduv_list(
                qubits_list[q]
            )
        else:
            qp, readout_error, readout_length_s = QubitProperties(), None, None
        qubit_properties.append(qp)
        readout_props[(q,)] = InstructionProperties(
            duration=readout_length_s if readout_length_s is not None else 0.0,
            error=readout_error if readout_error is not None else 0.0,
        )

    target = Target(num_qubits=n_qubits, qubit_properties=qubit_properties)

    # Group user-specified gates by name so each ``add_instruction`` call
    # carries every (qarg → InstructionProperties) entry for that gate.
    name_to_gate = get_standard_gate_name_mapping()
    grouped: dict[str, dict[tuple[int, ...], InstructionProperties]] = {}
    for gate_dict in normalized_properties.get("gates", []):
        name = str(gate_dict.get("gate", ""))
        if name not in name_to_gate:
            continue
        qargs = tuple(int(q) for q in gate_dict.get("qubits", []))
        duration_s, error = _gate_calibration_from_nduv_list(
            gate_dict.get("parameters", [])
        )
        grouped.setdefault(name, {})[qargs] = InstructionProperties(
            duration=duration_s, error=error
        )

    for name, props_map in grouped.items():
        target.add_instruction(name_to_gate[name], props_map)

    # ``measure`` is required for circuits that contain measurements; readout
    # error/duration are sourced from per-qubit Nduvs above.
    target.add_instruction(name_to_gate["measure"], readout_props)

    # ``reset`` and ``delay`` are universal infrastructure ops that the
    # transpiler/scheduler insert into circuits regardless of basis, so they
    # are added with default (no-error) properties even when the user
    # doesn't list them.
    infra_props: dict[tuple[int, ...], InstructionProperties | None] = {
        (q,): None for q in range(n_qubits)
    }
    target.add_instruction(name_to_gate["reset"], infra_props)
    target.add_instruction(name_to_gate["delay"], infra_props)

    return target


def create_backend_from_properties(
    properties: dict[str, Any],
    n_qubits: int | None = None,
    default_date: datetime.datetime | None = None,
) -> GenericBackendV2:
    """
    Create a populated GenericBackendV2 from a BackendProperties dictionary.

    This function handles the complete workflow:

    - Normalizes the properties dictionary (fills in missing fields).
    - Infers the number of qubits from the properties if not provided.
    - Creates a GenericBackendV2 backend whose ``target`` reflects the
      supplied calibration values (T1/T2, frequency, gate errors, gate
      durations, readout errors).

    The returned backend's ``target`` is the source of truth for downstream
    qiskit consumers — the transpiler reads gate durations and errors from
    it, and ``qiskit_aer.noise.NoiseModel.from_backend`` derives device
    noise from the same target.

    Args:
        properties: BackendProperties dictionary.
            Missing fields will be filled automatically.
        n_qubits: Optional number of qubits. If None, will be inferred from the
            length of the "qubits" list in the properties dictionary.
        default_date: Optional datetime to use for missing date fields.
            If None, uses current time.

    Returns:
        GenericBackendV2 backend whose ``target`` carries the supplied
        calibration values.

    Raises:
        ValueError: If n_qubits is not provided and cannot be inferred from properties
            (i.e., qubits list is empty or missing), or if n_qubits is less than 1.

    Example:
        >>> props = {
        ...     "backend_name": "test",
        ...     "qubits": [[{"name": "T1", "value": 100.0}]],  # 1 qubit
        ...     "gates": [{"gate": "sx", "qubits": [0], "parameters": []}]
        ... }
        >>> # Infer qubit count from properties (will be 1)
        >>> backend = create_backend_from_properties(props)
        >>> backend.num_qubits
        1
        >>> # Override qubit count if needed
        >>> backend_large = create_backend_from_properties(props, n_qubits=120)
        >>> backend_large.num_qubits
        120
    """
    # Normalize the properties first
    normalized_properties = _normalize_properties(properties, default_date)

    # Infer number of qubits from qubits list length if not provided
    if n_qubits is None:
        n_qubits = len(normalized_properties.get("qubits", []))
        if n_qubits == 0:
            raise ValueError(
                "n_qubits must be provided when properties dictionary has no qubits, "
                "or qubits list must contain at least one qubit"
            )

    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1")

    # ``GenericBackendV2`` populates its target with random calibration; the
    # ``basis_gates=["id"]`` placeholder keeps the constructor happy for
    # ``n_qubits=1`` (its default basis includes 2-qubit gates) — we
    # overwrite ``_target`` immediately with one carrying the user's values.
    backend = GenericBackendV2(num_qubits=n_qubits, basis_gates=["id"])
    target = _build_target_from_normalized(normalized_properties, n_qubits)
    # pyrefly: ignore[missing-attribute]
    backend._target = target

    return backend
