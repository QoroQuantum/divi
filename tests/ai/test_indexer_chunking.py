# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from divi.ai._indexer import (
    _chunk_markdown,
    _chunk_rst,
    _chunk_toml,
    _split_long_section,
    _strip_embed_prefix,
)


class TestStripEmbedPrefix:
    def test_strips_source_prefix(self):
        text = "[Source: foo.py § Bar]\ncontent here"
        assert _strip_embed_prefix(text) == "content here"

    def test_strips_module_prefix(self):
        text = "[Module: divi.core]\nmodule docs"
        assert _strip_embed_prefix(text) == "module docs"

    def test_strips_function_prefix(self):
        text = "[Function: divi.core.foo]\ndef foo(): pass"
        assert _strip_embed_prefix(text) == "def foo(): pass"

    def test_strips_class_prefix(self):
        text = "[Class: divi.core.Bar]\nclass Bar: pass"
        assert _strip_embed_prefix(text) == "class Bar: pass"

    def test_no_prefix_unchanged(self):
        text = "plain text content"
        assert _strip_embed_prefix(text) == "plain text content"

    def test_empty_string(self):
        assert _strip_embed_prefix("") == ""

    def test_prefix_only(self):
        text = "[Source: foo.py]"
        assert _strip_embed_prefix(text) == ""


class TestChunkMarkdown:
    def test_splits_by_headers(self, sample_markdown_source):
        chunks = _chunk_markdown(sample_markdown_source, "docs/test.md")
        assert len(chunks) >= 2

    def test_skips_navigation_sections(self, sample_markdown_source):
        chunks = _chunk_markdown(sample_markdown_source, "docs/test.md")
        all_text = " ".join(c.text for c in chunks)
        # "See Also" section should be skipped
        assert "Navigation links that should be skipped" not in all_text

    def test_preserves_source_prefix(self, sample_markdown_source):
        chunks = _chunk_markdown(sample_markdown_source, "docs/test.md")
        for chunk in chunks:
            assert chunk.text.startswith("[Source:")

    def test_empty_document(self):
        chunks = _chunk_markdown("", "empty.md")
        assert chunks == []

    def test_short_content_skipped(self):
        md = "# Title\n\nHi"  # Too short
        chunks = _chunk_markdown(md, "short.md")
        assert len(chunks) == 0

    def test_chunk_source_file_set(self, sample_markdown_source):
        chunks = _chunk_markdown(sample_markdown_source, "docs/test.md")
        for chunk in chunks:
            assert chunk.source_file == "docs/test.md"


class TestChunkRst:
    def test_parses_sections(self, sample_rst_source):
        chunks = _chunk_rst(sample_rst_source, "docs/test.rst")
        assert len(chunks) >= 2

    def test_skips_see_also(self, sample_rst_source):
        chunks = _chunk_rst(sample_rst_source, "docs/test.rst")
        all_text = " ".join(c.text for c in chunks)
        assert "Link to other resources" not in all_text

    def test_fallback_no_sections(self):
        """RST without section titles falls back to single chunk."""
        plain = "Just a paragraph with enough content to pass the minimum " * 3
        chunks = _chunk_rst(plain, "plain.rst")
        assert len(chunks) >= 1

    def test_preserves_source_prefix(self, sample_rst_source):
        chunks = _chunk_rst(sample_rst_source, "docs/test.rst")
        for chunk in chunks:
            assert chunk.text.startswith("[Source:")

    def test_handles_sphinx_roles(self):
        """RST with Sphinx-specific roles should not crash or produce empty chunks."""
        rst = """\
Using Sphinx Roles
==================

Use :class:`VQE` to run variational quantum eigensolver problems.
See :doc:`user_guide/vqe` for a full walkthrough. You can also
reference :ref:`advanced-usage` for more details on configuring
the :class:`QiskitSimulator` backend.
"""
        chunks = _chunk_rst(rst, "docs/test.rst")
        assert len(chunks) >= 1
        # The meaningful text should survive despite unknown roles
        all_text = " ".join(c.text for c in chunks)
        assert "VQE" in all_text
        assert "variational quantum eigensolver" in all_text


class TestChunkToml:
    def test_splits_by_section_headers(self):
        toml = (
            "[build-system]\n"
            "requires = ['setuptools']\n"
            "build-backend = 'setuptools.build_meta'\n"
            "# Some additional config for the build system here\n"
            "# More content to ensure minimum chunk length is met\n"
            "\n"
            "[project]\n"
            "name = 'myproject'\n"
            "version = '1.0.0'\n"
            "description = 'A sample project for testing'\n"
            "# Additional project metadata for chunk length\n"
        )
        chunks = _chunk_toml(toml, "pyproject.toml")
        assert len(chunks) == 2

    def test_double_bracket_headers(self):
        toml = (
            "[[tool.pytest.ini_options]]\n"
            "testpaths = ['tests']\n"
            "python_files = 'test_*.py'\n"
            "# Configuration for pytest test discovery and settings\n"
            "\n"
            "[[tool.black]]\n"
            "line-length = 88\n"
            "target-version = ['py311']\n"
            "# Black formatter configuration for code style\n"
        )
        chunks = _chunk_toml(toml, "pyproject.toml")
        assert len(chunks) == 2


class TestSplitLongSection:
    def test_produces_multiple_chunks(self):
        # Create a section longer than CHUNK_SIZE
        lines = [
            f"Line {i} with enough content to fill the buffer.\n" for i in range(100)
        ]
        chunks = []
        _split_long_section(lines, "[Source: test]\n", "test.rst", 1, chunks)
        assert len(chunks) >= 2

    def test_all_chunks_have_prefix(self):
        lines = [
            f"Line {i} with enough content to fill the buffer.\n" for i in range(100)
        ]
        chunks = []
        _split_long_section(lines, "[Source: test]\n", "test.rst", 1, chunks)
        for chunk in chunks:
            assert chunk.text.startswith("[Source: test]")

    def test_remainder_included(self):
        """Trailing content below CHUNK_SIZE still gets a chunk."""
        lines = ["x" * 600 + "\n", "y" * 600 + "\n", "remainder\n"]
        chunks = []
        _split_long_section(lines, "", "test.rst", 1, chunks)
        all_text = " ".join(c.text for c in chunks)
        assert "remainder" in all_text
