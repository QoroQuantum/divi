# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from divi.ai._indexer import _collect_files, _should_skip


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
def test_should_skip(path, expected):
    assert _should_skip(path) == expected, f"_should_skip({path}) should be {expected}"


def test_collect_files_scopes_paths_to_the_scanned_root(tmp_path):
    """Every directory check has to be made on a root-relative path.

    A checkout directory named ``divi`` puts that component into every absolute
    path below it, which satisfies the ``INCLUDE_DIRS`` check for any file and
    makes ``divi/ai`` look like it sits outside the AI module.
    """
    root = tmp_path / "divi"
    for relative in (
        "divi/qprog/kept.py",
        "divi/ai/_retriever.py",
        ".claude/CLAUDE.md",
        "scratch/notes.md",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("content", encoding="utf-8")

    collected = {path.relative_to(root).as_posix() for path in _collect_files([root])}

    assert collected == {"divi/qprog/kept.py"}


def test_collect_files_indexes_an_explicitly_named_file(tmp_path):
    """Naming a file is a request for it, so the top-level allowlist does not apply.

    Only the name-based exclusions still hold, since a lone filename carries no
    directory to scope.
    """
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    wanted = scratch / "notes.md"
    wanted.write_text("content", encoding="utf-8")
    excluded_by_name = scratch / "CHANGELOG.md"
    excluded_by_name.write_text("content", encoding="utf-8")

    assert _collect_files([wanted]) == [wanted]
    assert _collect_files([excluded_by_name]) == []
