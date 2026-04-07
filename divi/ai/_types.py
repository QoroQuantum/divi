# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared data types for the divi-ai subsystem."""

from dataclasses import dataclass

# Repo-root prefix used to shorten absolute paths for display.
REPO_MARKER = "Qoro/divi/"


def short_source(path: str) -> str:
    """Strip the repo-root prefix from an absolute source path."""
    idx = path.find(REPO_MARKER)
    if idx >= 0:
        return path[idx + len(REPO_MARKER) :]
    return path


@dataclass
class ChunkMeta:
    """Metadata for a single indexed text chunk."""

    text: str
    source_file: str
    start_line: int
    end_line: int
    chunk_type: str = "source"  # "source" | "test" | "tutorial" | "doc"
