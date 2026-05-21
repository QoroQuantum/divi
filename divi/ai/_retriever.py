# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Hybrid BM25 + dense retrieval with cross-encoder reranking.

Three-stage pipeline:

1. **BM25** (``bm25s``) over a code-aware tokenization (camelCase /
   snake_case / dot splits, English stopwords) of the *full* chunk text
   including the ``[Source:|Class:|Function:|Module:]`` prefix — the
   prefix carries the qualified name / section title, which is the
   single strongest BM25 signal for nominal and topical queries.
2. **Dense** (FAISS + ``jinaai/jina-embeddings-v2-base-code``) over the
   *embed-stripped* chunk text (label noise removed). See
   :func:`divi.ai._indexer._strip_embed_prefix`.
3. **Fusion** via Reciprocal Rank Fusion (``k=60``), then a
   **cross-encoder** (``Xenova/ms-marco-MiniLM-L-6-v2``) reranks the top
   ``rerank_candidates`` candidates against the original query.

The reranker is lazily loaded on first call (it adds ~80 MB resident
and ~1 s of startup time the first time the chatbot retrieves).

A chunk is **gated as confident** by its rerank score: cross-encoder
logits are positive for relevant pairs and negative for irrelevant
ones. Non-confident chunks are dropped before returning, so the
downstream chat code can trust that whatever it receives is relevant
context — no separate ``MIN_DENSE_SCORE`` filter required.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import bm25s
import faiss
import numpy as np
from bm25s.tokenization import STOPWORDS_EN_PLUS
from fastembed import TextEmbedding

from ._types import ChunkMeta

if TYPE_CHECKING:
    from fastembed.rerank.cross_encoder import TextCrossEncoder


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """A chunk returned by the retriever with its scores.

    Attributes
    ----------
    score:
        Primary ranking score — the cross-encoder reranker logit.
        Positive means the reranker judged the chunk relevant.
    dense_score:
        Raw cosine similarity from the dense embedder. Kept for
        diagnostics and display; not used as a gating signal anymore.
    """

    text: str
    source_file: str
    start_line: int
    end_line: int
    score: float
    dense_score: float = 0.0


@dataclass
class SearchStack:
    """Bundle of every resource the retriever needs.

    Built once at chatbot startup by :func:`divi.ai._indexer.load_search_stack`
    and passed to every :func:`retrieve` call.

    Three derived lookups (``reranker``, ``vocab``, ``raw_text``) are
    materialised lazily on first access via ``cached_property``. The
    cross-encoder costs ~80 MB / ~1 s to load, and ``vocab`` / ``raw_text``
    each touch every chunk — only the queries that actually need them pay
    the cost. Tests can override any of these by assigning directly to the
    attribute (``stack.reranker = mock``), which populates ``__dict__`` and
    shortcuts the descriptor.

    ``vocab`` and ``raw_text`` back the "out-of-corpus" sanity check: if a
    query contains a distinctive token or CamelCase identifier that does
    not appear *anywhere* in the corpus, the cross-encoder cannot
    meaningfully judge relevance — it tends to hallucinate high scores
    based on query *shape* (e.g. "What's the QFT implementation in Divi?"
    scored chunks at +3.5 even though "QFT" appears in zero chunks). The
    pre-rerank sanity check short-circuits that case.
    """

    index: faiss.IndexFlatIP
    chunks: list[ChunkMeta]
    embedder: TextEmbedding
    bm25: "bm25s.BM25"

    @cached_property
    def reranker(self) -> "TextCrossEncoder":
        """Cross-encoder model, loaded on first access and cached."""
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")

    @cached_property
    def vocab(self) -> frozenset[str]:
        """Set of all tokens (post-tokenization) that appear in any chunk."""
        v: set[str] = set()
        for c in self.chunks:
            v.update(tokenize(c.text))
        return frozenset(v)

    @cached_property
    def raw_text(self) -> str:
        """Lowercased concatenation of every chunk's raw text, for
        case-insensitive substring checks on CamelCase/acronym
        identifiers from the query."""
        return "\n".join(c.text for c in self.chunks).lower()


# ---------------------------------------------------------------------------
# Code-aware tokenizer (camelCase / snake_case / dots → lowercase tokens)
# ---------------------------------------------------------------------------


_ALPHA_RUN = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_CAMEL_BOUNDARY_1 = re.compile(r"([a-z0-9])([A-Z])")
_CAMEL_BOUNDARY_2 = re.compile(r"([A-Z])([A-Z][a-z])")
_STOPWORDS = frozenset(STOPWORDS_EN_PLUS)


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens.

    Splits camelCase, PascalCase, acronym-word boundaries, snake_case, and
    dots. Drops English stopwords and single-character tokens.

    Examples:
        "GraphPartitioningQAOA" → ["graph", "partitioning", "qaoa"]
        "divi.qprog.problems._graphs" → ["divi", "qprog", "problems", "graphs"]
        "How do I configure ZNE?" → ["configure", "zne"]
    """
    tokens: list[str] = []
    for run in _ALPHA_RUN.findall(text):
        split = _CAMEL_BOUNDARY_1.sub(r"\1 \2", run)
        split = _CAMEL_BOUNDARY_2.sub(r"\1 \2", split)
        for t in split.split():
            tl = t.lower()
            if tl in _STOPWORDS or len(tl) < 2:
                continue
            tokens.append(tl)
    return tokens


# ---------------------------------------------------------------------------
# Per-modality retrieval
# ---------------------------------------------------------------------------


def _dense_topn(query: str, stack: SearchStack, n: int) -> dict[int, float]:
    """Return {chunk_idx: cosine} for the FAISS top-n."""
    qv = np.array(list(stack.embedder.embed([query])), dtype=np.float32)
    faiss.normalize_L2(qv)
    scores, idxs = stack.index.search(qv, min(n, stack.index.ntotal))
    return {int(i): float(s) for s, i in zip(scores[0], idxs[0]) if i >= 0}


def _bm25_topn(query: str, stack: SearchStack, n: int) -> dict[int, float]:
    """Return {chunk_idx: bm25_score} for the BM25 top-n.

    ``bm25s.BM25.retrieve`` returns ``(results, scores)`` arrays of shape
    ``(n_queries, k)``. Because the BM25 index was built positionally from
    a ``list[list[str]]`` (each token list keyed by its position in
    ``stack.chunks``), ``results[0][j]`` is the *integer position* of the
    j-th best chunk in ``stack.chunks`` — not a stored object. That's the
    contract we depend on for the ``int(i)`` cast below.
    """
    q_tokens = tokenize(query)
    if not q_tokens:
        return {}
    results, scores = stack.bm25.retrieve(
        [q_tokens], k=min(n, len(stack.chunks)), show_progress=False
    )
    return {int(i): float(s) for i, s in zip(results[0], scores[0])}


def _cosine_for_indices(
    query_vec: np.ndarray, stack: SearchStack, indices: list[int]
) -> dict[int, float]:
    """Compute cosine for arbitrary chunk indices (e.g. BM25-only hits)."""
    out: dict[int, float] = {}
    for idx in indices:
        vec = stack.index.reconstruct(int(idx)).reshape(1, -1)
        # Index vectors are L2-normalized; dot product equals cosine.
        out[idx] = float(np.dot(query_vec[0], vec[0]))
    return out


# ---------------------------------------------------------------------------
# Fusion + rerank
# ---------------------------------------------------------------------------


_RRF_K = 60  # standard RRF constant (Cormack et al.)


def _rrf_fuse(*runs: dict[int, float], k: int = _RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion across an arbitrary number of runs."""
    fused: dict[int, float] = defaultdict(float)
    for run in runs:
        for rank, (idx, _) in enumerate(
            sorted(run.items(), key=lambda x: x[1], reverse=True), start=1
        ):
            fused[idx] += 1.0 / (k + rank)
    return dict(fused)


# ---------------------------------------------------------------------------
# Out-of-corpus sanity check
# ---------------------------------------------------------------------------

# Matches PascalCase (FastVQE), multi-cap acronyms (QFT, ZNE, QAOA), or
# capitalised words ≥ 4 chars (Grover). All three are "this looks like an
# identifier the user expects to find in the codebase" — if it doesn't
# appear anywhere in the corpus, the query is asking about something
# we don't have content for.
_IDENTIFIER_RE = re.compile(
    r"\b(?:[A-Z][a-z]+(?:[A-Z][a-z0-9]*)+"  # CamelCase / PascalCase
    r"|[A-Z]{2,}[A-Za-z0-9]*"  # ALLCAPS acronym
    r"|[A-Z][a-z]{3,})\b"  # Proper noun ≥ 4 chars
)

# Tokens that match the identifier regex but are actually common English
# sentence-starters (or the project name), not identifiers the user
# expects to find in the codebase. Exempted from the out-of-corpus bail.
_VOCAB_CHECK_STOPWORDS = frozenset(
    {
        # Project name — present in every chunk prefix anyway.
        "divi",
        # Common query-starting verbs.
        "explain",
        "show",
        "tell",
        "give",
        "find",
        "list",
        "describe",
        "discuss",
        "make",
        "create",
        "compare",
        "contrast",
        # Question words.
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "which",
        "whose",
        # Modal verbs.
        "can",
        "could",
        "should",
        "would",
        "may",
        "might",
        "will",
        # Adverbs / conjunctions.
        "before",
        "after",
        "during",
        "while",
        "also",
    }
)


def _has_out_of_corpus_identifier(query: str, stack: "SearchStack") -> bool:
    """True if the query contains a CamelCase / acronym / proper-noun
    identifier that doesn't appear *anywhere* in the corpus.

    Restricting the check to capitalised identifiers avoids false bails on
    general English vocabulary (``explain``, ``versus``, ``modeling`` — words
    that are absent from a code corpus but don't signal "out of scope").
    Cross-encoder hallucination on novel-but-quantum-shaped queries
    (``QFT``, ``Grover``, ``FastVQE``) is consistently signalled by the
    user explicitly capitalising the unknown identifier.
    """
    raw_text_lower = stack.raw_text  # already lowercased
    for match in _IDENTIFIER_RE.findall(query):
        m_lower = match.lower()
        if m_lower in _VOCAB_CHECK_STOPWORDS:
            continue
        if m_lower not in raw_text_lower:
            return True
    return False


# Cross-encoder logit threshold for "confident enough to surface."
# Calibrated against our benchmark — three score regimes observed:
#   * Clear on-topic + lexical overlap with chunk text:  top score > +2
#   * Conversationally-phrased on-topic (paraphrases, "choose between
#     X and Y" style):                                   top score -2..0
#   * Genuinely off-topic ("weather forecast"):          top score < -7
# -2.0 separates the first two regimes from the third.
_RERANK_GATE = -2.0


def retrieve(
    query: str,
    stack: SearchStack,
    *,
    top_k: int = 8,
    k_dense: int = 30,
    k_bm25: int = 30,
    rerank_candidates: int = 20,
) -> list[RetrievedChunk]:
    """Run the hybrid + rerank pipeline and return confident chunks.

    Parameters
    ----------
    query:
        The user's question or search string.
    stack:
        The :class:`SearchStack` built by
        :func:`divi.ai._indexer.load_search_stack`.
    top_k:
        Maximum number of chunks to return. Fewer may be returned if not
        enough chunks clear the reranker confidence gate.
    k_dense, k_bm25:
        Candidate pool sizes for each modality before fusion.
    rerank_candidates:
        How many fused candidates to pass through the cross-encoder.

    Returns
    -------
    list[RetrievedChunk]
        Confident chunks only, sorted by descending rerank score. Empty
        list when no chunk clears the gate (off-topic queries).
    """
    # Pre-rerank sanity check: bail immediately if the query mentions an
    # identifier or distinctive token that doesn't appear anywhere in the
    # corpus. The cross-encoder otherwise hallucinates high scores based
    # on query *shape* alone.
    if _has_out_of_corpus_identifier(query, stack):
        return []

    dense_scores = _dense_topn(query, stack, k_dense)
    bm25_scores = _bm25_topn(query, stack, k_bm25)

    if not dense_scores and not bm25_scores:
        return []

    fused = _rrf_fuse(dense_scores, bm25_scores)
    fused_ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    candidate_ids = [idx for idx, _ in fused_ranked[:rerank_candidates]]

    # Cross-encoder rerank
    reranker = stack.reranker
    docs = [stack.chunks[i].text for i in candidate_ids]
    rerank_scores = list(reranker.rerank(query, docs))

    # Apply the rerank-score gate
    survivors = [
        (idx, rs) for idx, rs in zip(candidate_ids, rerank_scores) if rs >= _RERANK_GATE
    ]
    survivors.sort(key=lambda x: x[1], reverse=True)
    survivors = survivors[:top_k]

    if not survivors:
        return []

    # Fill in cosine for any survivor that BM25 surfaced but dense missed,
    # so dense_score is meaningful for display / diagnostics.
    missing = [idx for idx, _ in survivors if idx not in dense_scores]
    extra_cosines: dict[int, float] = {}
    if missing:
        qv = np.array(list(stack.embedder.embed([query])), dtype=np.float32)
        faiss.normalize_L2(qv)
        extra_cosines = _cosine_for_indices(qv, stack, missing)

    return [
        RetrievedChunk(
            text=stack.chunks[idx].text,
            source_file=stack.chunks[idx].source_file,
            start_line=stack.chunks[idx].start_line,
            end_line=stack.chunks[idx].end_line,
            score=float(rs),
            dense_score=dense_scores.get(idx, extra_cosines.get(idx, 0.0)),
        )
        for idx, rs in survivors
    ]


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
    function/class signature + docstring (no implementation body), read
    the actual source file and substitute the full code.
    """
    repo_root = _find_repo_root()
    result = list(chunks)
    enriched = 0

    for i, chunk in enumerate(result):
        if enriched >= max_enrich:
            break
        if not chunk.source_file.endswith(".py"):
            continue
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

        prefix_match = re.match(r"^\[(?:Function|Class|Module): [^\]]+\]\n", chunk.text)
        prefix = prefix_match.group(0) if prefix_match else ""

        result[i] = replace(
            chunk,
            text=prefix + source_text,
        )
        enriched += 1

    # Budget guard
    total = sum(len(c.text) for c in result)
    while total > max_total_chars and len(result) > 1:
        dropped = result.pop()
        total -= len(dropped.text)

    return result
