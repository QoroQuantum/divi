# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

import numpy as np
import pytest

from divi.ai import _indexer
from divi.ai._indexer import _strip_embed_prefix, build_index

DIM = 8


class RecordingEmbedder:
    """Deterministic stand-in for ``TextEmbedding`` that logs what it embeds."""

    calls: list[list[str]] = []

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def vector_for(text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        return np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)

    def embed(self, texts, batch_size=16):
        texts = list(texts)
        RecordingEmbedder.calls.append(texts)
        for text in texts:
            yield self.vector_for(text)


@pytest.fixture()
def embedder(mocker):
    """Patch the indexer's embedder and expose the record of embedded texts."""
    RecordingEmbedder.calls = []
    mocker.patch.object(_indexer, "TextEmbedding", RecordingEmbedder)
    return RecordingEmbedder


@pytest.fixture()
def corpus(tmp_path):
    """A minimal indexable repo: two documented modules under ``divi/``."""
    pkg = tmp_path / "divi" / "qprog"
    pkg.mkdir(parents=True)
    _write_module(pkg / "alpha.py", "Alpha", "Run the alpha routine.")
    _write_module(pkg / "beta.py", "Beta", "Run the beta routine.")
    return tmp_path


def _write_module(path: Path, name: str, summary: str) -> None:
    path.write_text(
        f'"""Module docstring for {name}."""\n'
        "\n"
        f"def {name.lower()}_entry(x: int) -> int:\n"
        f'    """{summary}\n'
        "\n"
        "    Returns\n"
        "    -------\n"
        "    int\n"
        "        The result value for this routine.\n"
        '    """\n'
        "    return x\n",
        encoding="utf-8",
    )


def _build(corpus: Path, tmp_path: Path, cache_path: Path | None):
    return build_index(
        [corpus],
        output_dir=tmp_path / "out",
        batch_size=2,
        threads=1,
        cache_path=cache_path,
    )


def _embedded_texts(embedder) -> list[str]:
    return [text for call in embedder.calls for text in call]


def _assert_rows_match_chunks(index, chunks, embedder) -> None:
    rows = index.reconstruct_n(0, index.ntotal)
    assert index.ntotal == len(chunks)
    for i, chunk in enumerate(chunks):
        expected = embedder.vector_for(_strip_embed_prefix(chunk.text))
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(rows[i], expected, rtol=1e-6, atol=1e-6)


def test_second_build_embeds_nothing(embedder, corpus, tmp_path):
    cache = tmp_path / "cache.npz"
    _build(corpus, tmp_path, cache)
    assert _embedded_texts(embedder)

    embedder.calls = []
    _build(corpus, tmp_path, cache)

    assert _embedded_texts(embedder) == []


def test_only_changed_chunk_is_re_embedded(embedder, corpus, tmp_path):
    cache = tmp_path / "cache.npz"
    _build(corpus, tmp_path, cache)

    _write_module(
        corpus / "divi" / "qprog" / "alpha.py", "Alpha", "Run alpha, revised."
    )
    embedder.calls = []
    index, chunks = _build(corpus, tmp_path, cache)

    embedded = _embedded_texts(embedder)
    assert len(embedded) == 1
    assert "Run alpha, revised." in embedded[0]

    # Rows interleave one freshly embedded vector among cached ones.
    _assert_rows_match_chunks(index, chunks, embedder)


def test_warm_build_matches_cold_build(embedder, corpus, tmp_path):
    cache = tmp_path / "cache.npz"
    _build(corpus, tmp_path, cache)
    cold_index, cold_chunks = _build(corpus, tmp_path, None)
    warm_index, warm_chunks = _build(corpus, tmp_path, cache)

    assert [c.text for c in warm_chunks] == [c.text for c in cold_chunks]
    np.testing.assert_array_equal(
        warm_index.reconstruct_n(0, warm_index.ntotal),
        cold_index.reconstruct_n(0, cold_index.ntotal),
    )


def test_vector_rows_align_with_chunk_order(embedder, corpus, tmp_path):
    cache = tmp_path / "cache.npz"
    _build(corpus, tmp_path, cache)
    index, chunks = _build(corpus, tmp_path, cache)

    _assert_rows_match_chunks(index, chunks, embedder)


def test_deleted_chunks_are_pruned_from_cache(embedder, corpus, tmp_path):
    cache = tmp_path / "cache.npz"
    _, chunks_before = _build(corpus, tmp_path, cache)

    (corpus / "divi" / "qprog" / "beta.py").unlink()
    _, chunks_after = _build(corpus, tmp_path, cache)

    assert len(chunks_after) < len(chunks_before)
    with np.load(cache, allow_pickle=False) as data:
        assert len(data["keys"]) == len(chunks_after)


def test_cache_disabled_ignores_existing_cache(embedder, corpus, tmp_path):
    cache = tmp_path / "cache.npz"
    _build(corpus, tmp_path, cache)

    embedder.calls = []
    _, chunks = _build(corpus, tmp_path, None)

    assert len(_embedded_texts(embedder)) == len(chunks)
    assert list(tmp_path.glob("*.npz")) == [cache]


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(lambda raw: b"not an npz file", id="garbage"),
        pytest.param(lambda raw: raw[: len(raw) // 2], id="truncated"),
        pytest.param(lambda raw: b"", id="empty"),
    ],
)
def test_corrupt_cache_falls_back_to_full_embed(embedder, corpus, tmp_path, damage):
    cache = tmp_path / "cache.npz"
    _build(corpus, tmp_path, cache)
    cache.write_bytes(damage(cache.read_bytes()))

    embedder.calls = []
    _, chunks = _build(corpus, tmp_path, cache)

    assert len(_embedded_texts(embedder)) == len(chunks)


def test_new_embedder_version_invalidates_cache(embedder, corpus, tmp_path, mocker):
    cache = tmp_path / "cache.npz"
    _build(corpus, tmp_path, cache)

    mocker.patch.object(_indexer.metadata, "version", return_value="99.0.0")
    embedder.calls = []
    _, chunks = _build(corpus, tmp_path, cache)

    assert len(_embedded_texts(embedder)) == len(chunks)
