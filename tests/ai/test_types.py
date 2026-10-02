# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from divi.ai._types import display_path


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
