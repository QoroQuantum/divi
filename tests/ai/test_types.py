# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from divi.ai._types import ChunkMeta, display_path


class TestDisplayPath:
    def test_strips_absolute_prefix(self):
        path = "/home/user/Desktop/Coding/Qoro/divi/divi/qprog/vqe.py"
        assert display_path(path) == "divi/qprog/vqe.py"

    def test_strips_relative_prefix(self):
        path = "some/path/divi/docs/guide.rst"
        assert display_path(path) == "docs/guide.rst"

    def test_no_divi_marker_unchanged(self):
        path = "/other/path/file.py"
        assert display_path(path) == path

    def test_empty_string(self):
        assert display_path("") == ""

    def test_uses_first_occurrence(self):
        path = "/a/divi/b/divi/c.py"
        assert display_path(path) == "b/divi/c.py"


class TestChunkMeta:
    def test_default_chunk_type(self):
        chunk = ChunkMeta(text="hello", source_file="f.py", start_line=1, end_line=1)
        assert chunk.chunk_type == "source"

    def test_custom_chunk_type(self):
        chunk = ChunkMeta(
            text="hello",
            source_file="f.py",
            start_line=1,
            end_line=1,
            chunk_type="test",
        )
        assert chunk.chunk_type == "test"

    def test_fields_accessible(self):
        chunk = ChunkMeta(text="body", source_file="a.py", start_line=5, end_line=10)
        assert chunk.text == "body"
        assert chunk.source_file == "a.py"
        assert chunk.start_line == 5
        assert chunk.end_line == 10
