# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared data types for the divi-ai subsystem."""

from dataclasses import dataclass


def display_path(path: str) -> str:
    """Derive a short display path from a full source file path.

    Splits on the first ``"divi/"`` occurrence, so works regardless of
    where the repository is cloned.
    """
    if "divi/" in path:
        return path.split("divi/", 1)[-1]
    return path


@dataclass
class ChunkMeta:
    """Metadata for a single indexed text chunk."""

    text: str
    source_file: str
    start_line: int
    end_line: int
    chunk_type: str = "source"  # "source" | "test" | "tutorial" | "doc"
