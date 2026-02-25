# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Dense retrieval via FAISS (semantic similarity)."""

from dataclasses import dataclass

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
