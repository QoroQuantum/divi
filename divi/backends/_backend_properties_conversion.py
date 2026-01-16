# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Utilities for working with Qiskit BackendProperties and BackendV2 conversion."""

import datetime
from typing import Any

from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_ibm_runtime.models.backend_properties import BackendProperties


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
        Complete BackendProperties dictionary ready for BackendProperties.from_dict()

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
        >>> backend_props = BackendProperties.from_dict(normalized)
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


def create_backend_from_properties(
    properties: dict[str, Any],
    n_qubits: int | None = None,
    default_date: datetime.datetime | None = None,
) -> GenericBackendV2:
    """
    Create a populated GenericBackendV2 from a BackendProperties dictionary.

    This function handles the complete workflow:
    1. Normalizes the properties dictionary (fills in missing fields)
    2. Infers the number of qubits from the properties if not provided
    3. Creates a GenericBackendV2 backend
    4. Populates it with the normalized properties

    Args:
        properties: BackendProperties dictionary.
            Missing fields will be filled automatically.
        n_qubits: Optional number of qubits. If None, will be inferred from the
            length of the "qubits" list in the properties dictionary.
        default_date: Optional datetime to use for missing date fields.
            If None, uses current time.

    Returns:
        GenericBackendV2 backend populated with the provided properties.

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
        >>> backend.n_qubits
        1
        >>> # Override qubit count if needed
        >>> backend_large = create_backend_from_properties(props, n_qubits=120)
        >>> backend_large.n_qubits
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

    # Create the backend
    backend = GenericBackendV2(num_qubits=n_qubits)

    # Populate with properties
    backend._properties = BackendProperties.from_dict(normalized_properties)

    return backend
