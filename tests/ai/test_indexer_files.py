# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from divi.ai._indexer import _should_skip


class TestShouldSkip:
    @pytest.mark.parametrize(
        "path,expected",
        [
            (Path("CHANGELOG.md"), True),
            (Path("README.md"), True),
            (Path("CODE_OF_CONDUCT.md"), True),
            (Path("CONTRIBUTING.md"), True),
            # conftest outside tests/ → skip
            (Path("divi/conftest.py"), True),
            # test_ prefix outside tests/ → skip
            (Path("divi/test_foo.py"), True),
            # AI module itself → skip
            (Path("divi/ai/_chat.py"), True),
            (Path("divi/ai/_indexer.py"), True),
            # divi-ai docs page → skip (circular)
            (Path("docs/source/tools/divi_ai.rst"), True),
            # __pycache__ → skip
            (Path("divi/__pycache__/foo.py"), True),
            # _build dir → skip
            (Path("docs/_build/index.rst"), True),
            # Outside INCLUDE_DIRS → skip
            (Path("scripts/deploy.py"), True),
            (Path("random/file.py"), True),
            # Valid divi source → keep
            (Path("divi/pipeline/_core.py"), False),
            (Path("divi/qprog/vqe.py"), False),
            # Valid test file → keep
            (Path("tests/test_foo.py"), False),
            (Path("tests/conftest.py"), False),
            # Valid docs → keep
            (Path("docs/user_guide/intro.rst"), False),
            (Path("docs/source/quickstart.rst"), False),
            # Valid tutorials → keep
            (Path("tutorials/vqe_example.py"), False),
        ],
    )
    def test_should_skip(self, path, expected):
        assert (
            _should_skip(path) == expected
        ), f"_should_skip({path}) should be {expected}"
