# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import faiss
import numpy as np
import pytest

from divi.ai._retriever import RetrievedChunk
from divi.ai._types import ChunkMeta


@pytest.fixture()
def sample_python_source():
    """Multi-class Python source with docstrings and a __name__ block."""
    return '''\
"""Module docstring for the sample module."""

import numpy as np


class Foo(Bar):
    """A sample class that does Foo things."""

    def __init__(self, x: int, y: str = "hello"):
        """Initialise Foo.

        Args:
            x: The x value.
            y: The greeting.
        """
        self.x = x
        self.y = y

    def compute(self, values: list[int]) -> float:
        """Compute the result from values.

        Returns:
            The computed float.
        """
        return sum(values) / len(values)

    def _private(self):
        pass


def top_level_func(a: np.ndarray, b: int | str) -> bool:
    """A top-level function.

    Args:
        a: An array.
        b: An int or str.
    """
    return True


def undocumented():
    pass


async def async_helper(data: list[int]) -> None:
    """An async helper function."""
    pass


if __name__ == "__main__":
    f = Foo(1)
    print(f.compute([1, 2, 3]))
'''


@pytest.fixture()
def sample_rst_source():
    """RST document with multiple sections including skip-worthy ones."""
    return """\
Main Title
==========

This is the introduction paragraph with enough content to be
above the minimum chunk length threshold for testing purposes.

Getting Started
---------------

Here is how you get started with the library. You need to install
it first and then import the module to begin working with it.

.. code-block:: python

   import divi

See Also
--------

- Link to other resources
- More links here

Advanced Usage
--------------

Advanced users can configure the system with additional options
that provide more control over the execution and output format.
"""


@pytest.fixture()
def sample_markdown_source():
    """Markdown document with headers at various levels."""
    return """\
# Main Title

Introduction paragraph that should be long enough to pass the minimum
chunk length filter for the markdown chunking tests.

## Getting Started

Getting started content with instructions on how to use the library
and set up the development environment for testing.

## See Also

Navigation links that should be skipped by the chunker.

### Subsection

A subsection under See Also that should also be skipped.

## API Reference

Details about the API that should be included as a separate chunk
with enough content to pass the minimum length threshold.
"""


@pytest.fixture()
def sample_chunks():
    """Pre-built list of ChunkMeta for retriever/chat tests."""
    return [
        ChunkMeta(
            text="[Module: divi.qprog.vqe]\nVQE implementation for quantum chemistry.",
            source_file="/repo/divi/qprog/vqe.py",
            start_line=1,
            end_line=10,
        ),
        ChunkMeta(
            text="[Function: divi.qprog.vqe.VQE.run]\ndef run(self):\n"
            '    """Run the VQE algorithm."""',
            source_file="/repo/divi/qprog/vqe.py",
            start_line=50,
            end_line=80,
        ),
        ChunkMeta(
            text="[Source: docs/user_guide/vqe.rst § Getting Started]\n"
            "How to run VQE with Divi.",
            source_file="/repo/docs/user_guide/vqe.rst",
            start_line=1,
            end_line=20,
            chunk_type="doc",
        ),
        ChunkMeta(
            text="[Source: docs/api_reference/qprog.rst § VQE]\n"
            "API reference for the VQE class.",
            source_file="/repo/docs/api_reference/qprog.rst",
            start_line=1,
            end_line=15,
            chunk_type="doc",
        ),
        ChunkMeta(
            text="[Source: tutorials/vqe_quickstart.md § Quick Start]\n"
            "Tutorial on running VQE quickly.",
            source_file="/repo/tutorials/vqe_quickstart.md",
            start_line=1,
            end_line=30,
            chunk_type="doc",
        ),
    ]


@pytest.fixture()
def sample_retrieved_chunks():
    """Retrieved chunks with varying scores for filtering tests."""
    return [
        RetrievedChunk(
            text="High relevance chunk about VQE.",
            source_file="/repo/divi/qprog/vqe.py",
            start_line=1,
            end_line=10,
            score=0.85,
            dense_score=0.85,
        ),
        RetrievedChunk(
            text="Medium relevance chunk about QAOA.",
            source_file="/repo/docs/user_guide/qaoa.rst",
            start_line=1,
            end_line=20,
            score=0.65,
            dense_score=0.65,
        ),
        RetrievedChunk(
            text="Low relevance chunk about cats.",
            source_file="/repo/docs/random.md",
            start_line=1,
            end_line=5,
            score=0.50,
            dense_score=0.50,
        ),
    ]


@pytest.fixture()
def mock_llm(mocker):
    """Lightweight mock of llama_cpp.Llama for prompt/history tests."""
    llm = mocker.MagicMock()
    llm.tokenize.side_effect = lambda content, add_bos=False: list(
        range(len(content) // 4)
    )
    llm.n_ctx.return_value = 4096
    return llm


@pytest.fixture()
def mini_faiss_index(sample_chunks):
    """Real FAISS IndexFlatIP with 5 normalised vectors + chunk list."""
    dim = 8
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((len(sample_chunks), dim)).astype(np.float32)
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index, sample_chunks
