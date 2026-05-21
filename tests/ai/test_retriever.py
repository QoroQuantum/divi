# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import bm25s
import faiss
import numpy as np

from divi.ai._retriever import (
    RetrievedChunk,
    SearchStack,
    _rrf_fuse,
    enrich_chunks,
    retrieve,
    tokenize,
)
from divi.ai._types import ChunkMeta


class TestTokenize:
    def test_splits_camel_case(self):
        assert tokenize("GraphPartitioningQAOA") == ["graph", "partitioning", "qaoa"]

    def test_splits_acronym_word_boundary(self):
        assert tokenize("QAOAAnsatz") == ["qaoa", "ansatz"]
        assert tokenize("XMLParser") == ["xml", "parser"]

    def test_splits_snake_and_dot(self):
        assert tokenize("divi.qprog.problems._graphs") == [
            "divi",
            "qprog",
            "problems",
            "graphs",
        ]

    def test_drops_stopwords_and_single_chars(self):
        out = tokenize("What is a ZNE?")
        assert "what" not in out
        assert "is" not in out
        assert "zne" in out

    def test_empty_input(self):
        assert tokenize("") == []


class TestRRFFuse:
    def test_combines_two_runs(self):
        run_a = {1: 10.0, 2: 5.0, 3: 1.0}
        run_b = {3: 20.0, 4: 15.0}
        out = _rrf_fuse(run_a, run_b)
        assert set(out) == {1, 2, 3, 4}
        # 3 appears in both → highest fused score (additivity of RRF).
        assert max(out, key=out.get) == 3
        # Verify RRF actually *adds* contributions, not just selects the
        # winner: 3 is rank 3 in run_a + rank 1 in run_b; 1 is only rank 1
        # in run_a. 3's fused score must exceed 1's.
        assert out[3] > out[1]
        # And a single-modality top (rank 1 in run_b) is beaten by a
        # double-modality lower-rank hit:
        assert out[3] > out[4]

    def test_single_run_preserves_order(self):
        run = {1: 10.0, 2: 5.0, 3: 1.0}
        out = _rrf_fuse(run)
        assert sorted(out, key=out.get, reverse=True) == [1, 2, 3]


class TestRetrieve:
    """End-to-end retrieve() with a mocked reranker."""

    @staticmethod
    def _stack(chunks, embedder, mock_rerank_scores, mocker):
        """Build a SearchStack with a 4-dim dense index and a real BM25 index."""
        dim = 4
        rng = np.random.default_rng(0)
        vecs = rng.standard_normal((len(chunks), dim)).astype(np.float32)
        faiss.normalize_L2(vecs)
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)

        bm25 = bm25s.BM25()
        bm25.index([tokenize(c.text) for c in chunks], show_progress=False)

        stack = SearchStack(index=index, chunks=chunks, embedder=embedder, bm25=bm25)
        # Inject mock reranker via direct attribute assignment — cached_property
        # reads __dict__ first, so this shortcuts the lazy-load path.
        reranker = mocker.MagicMock()
        reranker.rerank.side_effect = lambda q, docs: list(mock_rerank_scores)[
            : len(docs)
        ]
        stack.reranker = reranker
        return stack

    @staticmethod
    def _make_chunks(n: int) -> list[ChunkMeta]:
        return [
            ChunkMeta(
                text=f"[Module: mod{i}]\nbody mod {i}",
                source_file=f"divi/mod{i}.py",
                start_line=1,
                end_line=1,
            )
            for i in range(n)
        ]

    @staticmethod
    def _make_query_embedder(mocker):
        """Mock embedder that returns a fresh iterator on every call.

        ``embedder.embed(...)`` is consumed by ``list(...)`` in the
        retriever, so each invocation needs its own iterator — a single
        ``return_value=iter([...])`` would exhaust after the first call.
        """
        embedder = mocker.MagicMock()
        embedder.embed.side_effect = lambda *_args, **_kwargs: iter(
            [np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)]
        )
        return embedder

    def test_returns_only_chunks_above_rerank_gate(self, mocker):
        """Chunks with rerank score below the gate (-2.0) are dropped."""
        chunks = self._make_chunks(5)
        embedder = self._make_query_embedder(mocker)
        # Rerank scores: first 3 above gate (-2.0), last 2 below
        stack = self._stack(chunks, embedder, [2.5, 1.0, -1.5, -2.5, -5.0], mocker)

        results = retrieve("mod body", stack, top_k=5)
        assert all(r.score >= -2.0 for r in results)
        assert len(results) == 3

    def test_gate_boundary_is_inclusive(self, mocker):
        """A chunk scoring exactly the gate value (-2.0) must survive (>=)."""
        chunks = self._make_chunks(3)
        embedder = self._make_query_embedder(mocker)
        # One chunk at the exact boundary; another just above; one well below.
        stack = self._stack(chunks, embedder, [1.0, -2.0, -2.001], mocker)

        results = retrieve("mod body", stack, top_k=3)
        scores = sorted(r.score for r in results)
        assert -2.0 in scores  # the boundary chunk is admitted
        assert all(s >= -2.0 for s in scores)
        assert len(results) == 2

    def test_empty_when_all_below_gate(self, mocker):
        chunks = self._make_chunks(3)
        embedder = self._make_query_embedder(mocker)
        # All rerank scores below the -2.0 gate
        stack = self._stack(chunks, embedder, [-3.0, -5.0, -10.0], mocker)

        assert retrieve("mod body", stack, top_k=5) == []

    def test_bails_on_out_of_corpus_identifier(self, mocker):
        """Query with a CamelCase identifier absent from corpus → empty.

        Also asserts the reranker is not invoked: the sanity check should
        short-circuit before the cross-encoder is consulted.
        """
        chunks = self._make_chunks(3)
        embedder = self._make_query_embedder(mocker)
        # Rerank scores set high so that if the bail logic regresses, the
        # chunks would clearly pass the gate and the test would fail.
        stack = self._stack(chunks, embedder, [5.0, 5.0, 5.0], mocker)

        # "GPUSimulator" is a CamelCase identifier not in the corpus.
        assert retrieve("How do I use GPUSimulator?", stack, top_k=5) == []
        stack.reranker.rerank.assert_not_called()

    def test_exempt_starter_words_do_not_trigger_bail(self, mocker):
        """Common English query starters that match _IDENTIFIER_RE but live
        in _VOCAB_CHECK_STOPWORDS (e.g. ``Explain``, ``Show``) must NOT
        trigger the out-of-corpus bail, even when absent from the corpus.

        Regression guard: a future edit that empties or narrows
        _VOCAB_CHECK_STOPWORDS would falsely bail on legitimate questions
        about real symbols (e.g. "Explain the ParameterBindingStage").
        """
        chunks = self._make_chunks(3)
        embedder = self._make_query_embedder(mocker)
        stack = self._stack(chunks, embedder, [2.5, 1.0, 0.5], mocker)

        # "Explain", "Show", "Compare", and "Describe" all match the
        # identifier regex (Proper-noun ≥ 4 chars) but are exempt. The
        # tokens "mod" and "body" in the query ensure tokenisation produces
        # in-corpus terms; the test verifies the *exempt* identifiers do
        # not cause a bail.
        for opener in ("Explain", "Show", "Compare", "Describe"):
            results = retrieve(f"{opener} the mod body", stack, top_k=3)
            assert results, f"unexpected bail for query starter {opener!r}"

    def test_results_sorted_by_rerank_score(self, mocker):
        chunks = self._make_chunks(4)
        embedder = self._make_query_embedder(mocker)
        # Out-of-order: highest at index 2
        stack = self._stack(chunks, embedder, [1.0, 0.5, 5.0, 2.0], mocker)

        results = retrieve("mod body", stack, top_k=4)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score
        assert isinstance(results[0], RetrievedChunk)


class TestSearchStackReranker:
    """The reranker is lazy-loaded on first access via ``cached_property``."""

    def test_lazy_load_constructs_once_and_caches(self, mocker):
        # Patch the class at its real import site (inside the cached_property
        # body). The patch is set up *before* the cached_property accesses it.
        mock_cls = mocker.patch("fastembed.rerank.cross_encoder.TextCrossEncoder")
        stack = SearchStack(
            index=mocker.MagicMock(),
            chunks=[],
            embedder=mocker.MagicMock(),
            bm25=mocker.MagicMock(),
        )
        # First access — triggers construction.
        r1 = stack.reranker
        # Second access — returns the cached instance, does not re-construct.
        r2 = stack.reranker
        assert r1 is r2
        mock_cls.assert_called_once_with(model_name="Xenova/ms-marco-MiniLM-L-6-v2")


class TestEnrichChunks:
    def _make_chunk(self, text, source_file, score=0.8):
        return RetrievedChunk(
            text=text,
            source_file=source_file,
            start_line=1,
            end_line=5,
            score=score,
            dense_score=score,
        )

    def test_preserves_non_docstring_chunk(self):
        chunk = self._make_chunk(
            '[Function: mod.func]\ndef func():\n    """Doc."""\n    return 42',
            "divi/mod.py",
        )
        result = enrich_chunks([chunk])
        assert result[0].text == chunk.text

    def test_skips_non_python(self):
        chunk = self._make_chunk(
            'Some RST content ending with """',
            "docs/guide.rst",
        )
        result = enrich_chunks([chunk])
        assert result[0].text == chunk.text

    def test_handles_missing_file_gracefully(self):
        chunk = self._make_chunk(
            '[Function: mod.func]\ndef func():\n    """Doc only."""',
            "/nonexistent/path/mod.py",
        )
        result = enrich_chunks([chunk])
        assert len(result) == 1

    def test_budget_guard_drops_lowest(self):
        chunks = [
            self._make_chunk("x" * 5000, "a.rst", score=0.9),
            self._make_chunk("y" * 5000, "b.rst", score=0.8),
            self._make_chunk("z" * 5000, "c.rst", score=0.7),
        ]
        result = enrich_chunks(chunks, max_total_chars=12000)
        assert len(result) < 3
        assert result[0].text.startswith("x")

    def test_enriches_docstring_only_chunk_from_disk(self, tmp_path):
        source = tmp_path / "mod.py"
        source.write_text(
            "def func():\n" '    """The docstring."""\n' "    return 42\n"
        )
        chunk = self._make_chunk(
            '[Function: mod.func]\ndef func():\n    """The docstring."""',
            str(source),
        )
        result = enrich_chunks([chunk])
        assert "return 42" in result[0].text
        assert result[0].text.startswith("[Function: mod.func]")

    def test_max_enrich_limit(self):
        chunks = [
            self._make_chunk(
                f'[Function: m.f{i}]\ndef f{i}():\n    """Doc."""',
                f"/nonexistent/f{i}.py",
            )
            for i in range(10)
        ]
        result = enrich_chunks(chunks, max_enrich=2)
        assert len(result) == 10
