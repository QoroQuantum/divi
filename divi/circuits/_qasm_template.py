# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Pre-split QASM templates for fast parameter substitution."""

import re
from typing import NamedTuple


class QASMTemplate(NamedTuple):
    """Pre-split QASM body for fast parameter substitution.

    Instead of scanning a full QASM string with a regex on every parameter
    binding call, the string is split once at symbol boundaries into
    ``fragments`` and ``slot_indices``.  Rendering then reduces to
    interleaving fragments with looked-up values — no regex needed.

    ``fragments[i]`` is the literal QASM text between the (i-1)-th and
    i-th symbol occurrence.  ``slot_indices[i]`` is the index into the
    parameter values array for the i-th slot.

    Invariant: ``len(fragments) == len(slot_indices) + 1``

    Example::

        Given QASM body ``"rx(w_0) q[0];\\nrz(w_1) q[1];\\n"``
        and symbols ``("w_0", "w_1")``:

            fragments   = ("rx(", ") q[0];\\nrz(", ") q[1];\\n")
            slot_indices = (0, 1)

        Rendering with values ``("1.5", "2.7")`` produces::

            "rx(" + "1.5" + ") q[0];\\nrz(" + "2.7" + ") q[1];\\n"
            = "rx(1.5) q[0];\\nrz(2.7) q[1];\\n"
    """

    fragments: tuple[str, ...]
    slot_indices: tuple[int, ...]


def build_template(qasm_body: str, symbol_names: tuple[str, ...]) -> QASMTemplate:
    """Split a QASM string at symbol boundaries into a :class:`QASMTemplate`.

    Symbol names are matched longest-first to avoid partial-match issues
    (e.g., ``w_1`` matching inside ``w_10``).

    Args:
        qasm_body: The QASM body string containing symbolic parameter names.
        symbol_names: Ordered tuple of symbol name strings.  The index in
            this tuple becomes the slot index used during rendering.

    Returns:
        A :class:`QASMTemplate` ready for :func:`render_template`.
    """
    if not symbol_names:
        return QASMTemplate(fragments=(qasm_body,), slot_indices=())

    name_to_idx = {name: i for i, name in enumerate(symbol_names)}

    escaped = sorted((re.escape(name) for name in symbol_names), key=len, reverse=True)
    pattern = re.compile("|".join(escaped))

    fragments: list[str] = []
    slot_indices: list[int] = []
    last_end = 0

    for match in pattern.finditer(qasm_body):
        fragments.append(qasm_body[last_end : match.start()])
        slot_indices.append(name_to_idx[match.group(0)])
        last_end = match.end()

    fragments.append(qasm_body[last_end:])

    return QASMTemplate(
        fragments=tuple(fragments),
        slot_indices=tuple(slot_indices),
    )


def render_template(template: QASMTemplate, formatted_values: tuple[str, ...]) -> str:
    """Render a :class:`QASMTemplate` with concrete parameter values.

    Args:
        template: The pre-split template.
        formatted_values: Formatted parameter strings, indexed by
            the slot indices stored in *template*.

    Returns:
        The fully-bound QASM body string.
    """
    fragments = template.fragments
    slot_indices = template.slot_indices

    parts: list[str] = []
    for i, slot_idx in enumerate(slot_indices):
        parts.append(fragments[i])
        parts.append(formatted_values[slot_idx])
    parts.append(fragments[-1])

    return "".join(parts)
