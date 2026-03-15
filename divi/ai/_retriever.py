# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Dense retrieval via FAISS (semantic similarity)."""

import re
from dataclasses import dataclass, replace
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding

from ._types import ChunkMeta


@dataclass
class RetrievedChunk:
    """A chunk returned by the retriever, with its similarity score."""

    text: str
    source_file: str
    start_line: int
    end_line: int
    score: float
    dense_score: float = 0.0  # raw cosine similarity (absolute confidence)


def retrieve(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: list[ChunkMeta],
    embedder: TextEmbedding,
    *,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Search the index for chunks most relevant to *query*.

    Parameters
    ----------
    query:
        The user's question or search string.
    index:
        The FAISS inner-product index built by :func:`build_index`.
    chunks:
        The chunk metadata list corresponding to *index*.
    embedder:
        A ``fastembed.TextEmbedding`` instance (reused across calls).
    top_k:
        Number of chunks to return.

    Returns
    -------
    list[RetrievedChunk]
        The *top_k* most relevant chunks, sorted by descending score.
    """
    n_dense = min(top_k * 4, index.ntotal)
    query_vec = np.array(list(embedder.embed([query])), dtype=np.float32)
    faiss.normalize_L2(query_vec)
    dense_scores_arr, dense_indices_arr = index.search(query_vec, n_dense)

    results: list[RetrievedChunk] = []
    for score, idx in zip(dense_scores_arr[0], dense_indices_arr[0]):
        if idx < 0:
            break
        chunk = chunks[idx]
        results.append(
            RetrievedChunk(
                text=chunk.text,
                source_file=chunk.source_file,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                score=float(score),
                dense_score=float(score),
            )
        )

    top = results[:top_k]
    # Source diversity: ensure at least one .py chunk is present.
    has_code = any(r.source_file.endswith(".py") for r in top)
    if not has_code:
        best_py = next(
            (r for r in results if r.source_file.endswith(".py")),
            None,
        )
        if best_py:
            top = list(top)
            top[-1] = best_py
    return top


# ---------------------------------------------------------------------------
# Two-stage enrichment — replace docstring-only chunks with full source
# ---------------------------------------------------------------------------

# Matches chunks that end with a closing docstring and nothing else after it.
_DOCSTRING_ONLY_RE = re.compile(r'"""\s*$')


def _find_repo_root() -> Path | None:
    """Walk up from this file to find the git repository root."""
    current = Path(__file__).resolve().parent
    for parent in (current, *current.parents):
        if (parent / ".git").exists():
            return parent
    return None


def _resolve_source_path(source_file: str, repo_root: Path | None) -> Path | None:
    """Try to resolve a source_file string to an existing Path."""
    p = Path(source_file)
    if p.is_file():
        return p
    if repo_root is None:
        return None
    # Try stripping absolute prefix to get a repo-relative path
    # e.g. /home/.../divi/divi/qprog/vqe.py → divi/qprog/vqe.py
    marker = "divi/"
    idx = source_file.find(marker)
    if idx >= 0:
        rel = source_file[idx:]
        candidate = repo_root / rel
        if candidate.is_file():
            return candidate
    return None


def enrich_chunks(
    chunks: list[RetrievedChunk],
    *,
    max_enrich: int = 3,
    max_chars: int = 1500,
    max_total_chars: int = 16000,
) -> list[RetrievedChunk]:
    """Replace docstring-only chunks with full source code from disk.

    For the top *max_enrich* chunks that appear to contain only a
    function/class signature + docstring (no implementation body),
    read the actual source file and substitute the full code.

    Parameters
    ----------
    chunks:
        Retrieved chunks, sorted by score (highest first).
    max_enrich:
        Maximum number of chunks to attempt enrichment on.
    max_chars:
        Maximum character length for an enriched chunk (skip if bigger).
    max_total_chars:
        Budget for total context characters — drop lowest-scored
        chunks if exceeded after enrichment.

    Returns
    -------
    list[RetrievedChunk]
        The (possibly enriched) chunk list.
    """
    repo_root = _find_repo_root()
    result = list(chunks)
    enriched = 0

    for i, chunk in enumerate(result):
        if enriched >= max_enrich:
            break
        if not chunk.source_file.endswith(".py"):
            continue
        # Only enrich docstring-only chunks (signature + docstring, no body)
        if not _DOCSTRING_ONLY_RE.search(chunk.text):
            continue

        path = _resolve_source_path(chunk.source_file, repo_root)
        if path is None:
            continue

        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines(
                keepends=True
            )
        except OSError:
            continue

        start = max(0, chunk.start_line - 1)
        end = min(len(lines), chunk.end_line)
        source_text = "".join(lines[start:end]).strip()

        if not source_text or len(source_text) > max_chars:
            continue

        # Preserve the [Function: ...] / [Class: ...] prefix line
        prefix_match = re.match(r"^\[(?:Function|Class|Module): [^\]]+\]\n", chunk.text)
        prefix = prefix_match.group(0) if prefix_match else ""

        result[i] = replace(
            chunk,
            text=prefix + source_text,
        )
        enriched += 1

    # Budget guard: drop lowest-scored chunks if total is too large
    total = sum(len(c.text) for c in result)
    while total > max_total_chars and len(result) > 1:
        dropped = result.pop()
        total -= len(dropped.text)

    return result
