# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import faiss
import numpy as np
import pytest

from divi.ai._retriever import RetrievedChunk, enrich_chunks, retrieve
from divi.ai._types import ChunkMeta


class TestRetrieve:
    @pytest.fixture(autouse=True)
    def _setup_mocker(self, mocker):
        self._mocker = mocker

    def _make_mock_embedder(self, dim):
        """Create a mock embedder that returns a normalised random vector."""
        embedder = self._mocker.MagicMock()
        rng = np.random.default_rng(99)
        vec = rng.standard_normal((1, dim)).astype(np.float32)
        faiss.normalize_L2(vec)
        embedder.embed.return_value = iter([vec[0]])
        return embedder

    def test_returns_top_k_results(self, mini_faiss_index):
        index, chunks = mini_faiss_index
        embedder = self._make_mock_embedder(index.d)
        results = retrieve("test query", index, chunks, embedder, top_k=3)
        assert len(results) == 3

    def test_results_are_retrieved_chunks(self, mini_faiss_index):
        index, chunks = mini_faiss_index
        embedder = self._make_mock_embedder(index.d)
        results = retrieve("test", index, chunks, embedder, top_k=2)
        for r in results:
            assert isinstance(r, RetrievedChunk)
            assert r.dense_score == r.score  # score equals dense_score

    def test_top_k_larger_than_index(self, mini_faiss_index):
        index, chunks = mini_faiss_index
        embedder = self._make_mock_embedder(index.d)
        results = retrieve("test", index, chunks, embedder, top_k=100)
        assert len(results) == len(chunks)

    def test_source_diversity_injects_py(self):
        """If no .py chunk in top_k, the last slot is replaced with best .py match."""
        dim = 4

        # Construct deterministic geometry: query is [1,0,0,0].
        # 3 .rst vectors are close to query, .py vector is orthogonal.
        chunk_list = [
            ChunkMeta(
                text=f"doc {i}", source_file=f"doc{i}.rst", start_line=1, end_line=1
            )
            for i in range(3)
        ]
        chunk_list.append(
            ChunkMeta(text="code", source_file="code.py", start_line=1, end_line=1)
        )

        vectors = np.array(
            [
                [0.95, 0.05, 0.0, 0.0],  # rst 0: very close to query
                [0.90, 0.10, 0.0, 0.0],  # rst 1: close to query
                [0.85, 0.15, 0.0, 0.0],  # rst 2: close to query
                [0.0, 0.0, 1.0, 0.0],  # py: orthogonal to query
            ],
            dtype=np.float32,
        )
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        # Mock embedder returns query vector [1,0,0,0]
        embedder = self._mocker.MagicMock()
        query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        embedder.embed.return_value = iter([query_vec])

        results = retrieve("test", index, chunk_list, embedder, top_k=3)
        # .py chunk has near-zero score but diversity injection should include it
        has_py = any(r.source_file.endswith(".py") for r in results)
        assert has_py


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
        # Already has a body → should not be changed
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
        # Should not crash, chunk preserved as-is
        assert len(result) == 1

    def test_budget_guard_drops_lowest(self):
        """When total chars exceed max_total_chars, drop lowest-scored chunks."""
        chunks = [
            self._make_chunk("x" * 5000, "a.rst", score=0.9),
            self._make_chunk("y" * 5000, "b.rst", score=0.8),
            self._make_chunk("z" * 5000, "c.rst", score=0.7),
        ]
        result = enrich_chunks(chunks, max_total_chars=12000)
        assert len(result) < 3
        # Highest-scored chunks should be kept
        assert result[0].text.startswith("x")

    def test_enriches_docstring_only_chunk_from_disk(self, tmp_path):
        """Core enrichment path: docstring-only chunk is replaced with full source."""
        # Create a real Python file on disk
        source = tmp_path / "mod.py"
        source.write_text(
            "def func():\n" '    """The docstring."""\n' "    return 42\n"
        )
        chunk = self._make_chunk(
            '[Function: mod.func]\ndef func():\n    """The docstring."""',
            str(source),
        )
        result = enrich_chunks([chunk])
        # Should have replaced with full source including the body
        assert "return 42" in result[0].text
        # Prefix should be preserved
        assert result[0].text.startswith("[Function: mod.func]")

    def test_max_enrich_limit(self):
        """Only max_enrich chunks should be attempted for enrichment."""
        chunks = [
            self._make_chunk(
                f'[Function: m.f{i}]\ndef f{i}():\n    """Doc."""',
                f"/nonexistent/f{i}.py",
            )
            for i in range(10)
        ]
        # Should not crash even with many docstring-only chunks
        result = enrich_chunks(chunks, max_enrich=2)
        assert len(result) == 10
