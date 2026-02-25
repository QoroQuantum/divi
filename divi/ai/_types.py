# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared data types for the divi-ai subsystem."""

from dataclasses import dataclass


@dataclass
class ChunkMeta:
    """Metadata for a single indexed text chunk."""

    text: str
    source_file: str
    start_line: int
    end_line: int
