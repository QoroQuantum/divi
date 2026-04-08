# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import textwrap

from divi.ai._indexer import (
    _extract_import_block,
    _extract_python_units,
    _extract_test_usage,
)


class TestExtractPythonUnits:
    def test_extracts_module_docstring(self, sample_python_source):
        chunks = _extract_python_units(sample_python_source, "divi/sample.py")
        module_chunks = [
            c
            for c in chunks
            if c.text.startswith("[Module:") and "Module docstring" in c.text
        ]
        assert len(module_chunks) == 1

    def test_module_doc_includes_public_symbols(self, sample_python_source):
        chunks = _extract_python_units(sample_python_source, "divi/sample.py")
        module_chunks = [c for c in chunks if c.text.startswith("[Module:")]
        assert "Defines:" in module_chunks[0].text
        assert "Foo" in module_chunks[0].text
        assert "top_level_func" in module_chunks[0].text
        assert "async_helper" in module_chunks[0].text

    def test_does_not_list_private_in_defines(self, sample_python_source):
        chunks = _extract_python_units(sample_python_source, "divi/sample.py")
        module_chunks = [c for c in chunks if c.text.startswith("[Module:")]
        # undocumented and _private should not be in Defines
        assert "_private" not in module_chunks[0].text

    def test_extracts_class(self, sample_python_source):
        chunks = _extract_python_units(sample_python_source, "divi/sample.py")
        class_chunks = [c for c in chunks if c.text.startswith("[Class:")]
        assert len(class_chunks) == 1
        assert "Foo" in class_chunks[0].text
        assert "Bar" in class_chunks[0].text  # base class

    def test_extracts_class_docstring(self, sample_python_source):
        chunks = _extract_python_units(sample_python_source, "divi/sample.py")
        class_chunks = [c for c in chunks if c.text.startswith("[Class:")]
        assert "A sample class that does Foo things" in class_chunks[0].text

    def test_extracts_methods(self, sample_python_source):
        chunks = _extract_python_units(sample_python_source, "divi/sample.py")
        func_chunks = [c for c in chunks if c.text.startswith("[Function:")]
        func_names = [c.text for c in func_chunks]
        # __init__ and compute have docstrings → should be extracted
        assert any("__init__" in n for n in func_names)
        assert any("compute" in n for n in func_names)

    def test_skips_undocumented_functions(self, sample_python_source):
        """Undocumented functions should not get their own [Function:] chunk."""
        chunks = _extract_python_units(sample_python_source, "divi/sample.py")
        func_chunks = [c for c in chunks if c.text.startswith("[Function:")]
        assert not any("undocumented" in c.text for c in func_chunks)

    def test_skips_private_methods(self, sample_python_source):
        chunks = _extract_python_units(sample_python_source, "divi/sample.py")
        func_chunks = [c for c in chunks if c.text.startswith("[Function:")]
        assert not any("_private" in c.text for c in func_chunks)

    def test_top_level_func_extracted(self, sample_python_source):
        chunks = _extract_python_units(sample_python_source, "divi/sample.py")
        func_chunks = [c for c in chunks if c.text.startswith("[Function:")]
        assert any("top_level_func" in c.text for c in func_chunks)

    def test_async_function_prefix(self, sample_python_source):
        chunks = _extract_python_units(sample_python_source, "divi/sample.py")
        async_chunks = [
            c
            for c in chunks
            if c.text.startswith("[Function:") and "async_helper" in c.text
        ]
        assert len(async_chunks) == 1
        assert "async def" in async_chunks[0].text

    def test_captures_name_main_block(self, sample_python_source):
        chunks = _extract_python_units(sample_python_source, "divi/sample.py")
        main_chunks = [c for c in chunks if "if __name__" in c.text]
        assert len(main_chunks) == 1

    def test_tutorial_name_main_includes_full_file(self, sample_python_source):
        chunks = _extract_python_units(
            sample_python_source, "tutorials/vqe_tutorial.py"
        )
        example_chunks = [c for c in chunks if c.text.startswith("[Example:")]
        assert len(example_chunks) == 1
        # Should contain imports from the top of the file
        assert "import numpy" in example_chunks[0].text

    def test_syntax_error_returns_empty(self):
        assert _extract_python_units("def broken(", "bad.py") == []

    def test_empty_file_returns_empty(self):
        assert _extract_python_units("", "empty.py") == []

    def test_includes_type_annotations_in_signature(self, sample_python_source):
        chunks = _extract_python_units(sample_python_source, "divi/sample.py")
        init_chunks = [c for c in chunks if "__init__" in c.text]
        assert len(init_chunks) == 1
        assert "x: int" in init_chunks[0].text
        assert "y: str" in init_chunks[0].text


class TestGetSignature:
    """Test signature reconstruction using _extract_python_units on small snippets."""

    def _extract_sig(self, func_source):
        """Helper: extract the signature from a function source."""
        chunks = _extract_python_units(func_source, "test.py")
        func_chunks = [c for c in chunks if "[Function:" in c.text]
        assert len(func_chunks) == 1
        return func_chunks[0].text

    def test_simple_args(self):
        src = 'def foo(a, b, c):\n    """Doc."""\n    pass'
        text = self._extract_sig(src)
        assert "(a, b, c)" in text

    def test_with_annotations(self):
        src = 'def foo(x: int, y: str) -> bool:\n    """Doc."""\n    pass'
        text = self._extract_sig(src)
        assert "x: int" in text
        assert "y: str" in text

    def test_varargs(self):
        src = 'def foo(*args, **kwargs):\n    """Doc."""\n    pass'
        text = self._extract_sig(src)
        assert "*args" in text
        assert "**kwargs" in text

    def test_keyword_only(self):
        src = 'def foo(*, key: int):\n    """Doc."""\n    pass'
        text = self._extract_sig(src)
        assert "(*, key: int)" in text


class TestAnnotationStr:
    """Test annotation rendering via _extract_python_units."""

    def _extract_annotation(self, annotation_str):
        """Helper: render an annotation by extracting from a function."""
        src = f'def foo(x: {annotation_str}):\n    """Doc."""\n    pass'
        chunks = _extract_python_units(src, "test.py")
        func_chunks = [c for c in chunks if "[Function:" in c.text]
        return func_chunks[0].text

    def test_simple_name(self):
        text = self._extract_annotation("int")
        assert "x: int" in text

    def test_attribute(self):
        text = self._extract_annotation("np.ndarray")
        assert "x: np.ndarray" in text

    def test_subscript(self):
        text = self._extract_annotation("list[int]")
        assert "x: list[int]" in text

    def test_union(self):
        text = self._extract_annotation("int | str")
        assert "x: int | str" in text


class TestExtractImportBlock:
    def test_extracts_imports(self):
        lines = [
            "import os\n",
            "from pathlib import Path\n",
            "\n",
            "x = 1\n",
        ]
        result = _extract_import_block(lines)
        assert "import os" in result
        assert "from pathlib import Path" in result

    def test_stops_at_non_import(self):
        lines = [
            "import os\n",
            "x = 1\n",
            "import sys\n",  # after code → should not be included
        ]
        result = _extract_import_block(lines)
        assert "import os" in result
        assert "import sys" not in result

    def test_skips_single_line_docstring(self):
        lines = [
            '"""Module doc."""\n',
            "import os\n",
            "\n",
            "class Foo: pass\n",
        ]
        result = _extract_import_block(lines)
        assert "import os" in result
        assert "Module doc" not in result

    def test_skips_multi_line_docstring(self):
        lines = [
            '"""Module docstring\n',
            "that spans multiple lines.\n",
            '"""\n',
            "import os\n",
            "\n",
            "class Foo: pass\n",
        ]
        result = _extract_import_block(lines)
        assert "import os" in result
        assert "spans multiple" not in result

    def test_caps_at_max_chars(self):
        lines = [f"from mod{i} import thing{i}\n" for i in range(100)]
        result = _extract_import_block(lines)
        assert len(result) <= 400 + 50  # some tolerance for line boundary


class TestExtractTestUsage:
    def test_extracts_test_functions(self):
        src = textwrap.dedent("""\
            import pytest
            from divi.qprog import VQE

            def test_vqe_runs():
                vqe = VQE()
                vqe.run()
                assert vqe.result is not None
        """)
        chunks = _extract_test_usage(src, "tests/test_vqe.py")
        assert len(chunks) == 1
        assert "test_vqe_runs" in chunks[0].text
        assert chunks[0].chunk_type == "test"

    def test_skips_non_test_functions(self):
        src = textwrap.dedent("""\
            def helper():
                return 42

            def test_foo():
                assert helper() == 42
        """)
        chunks = _extract_test_usage(src, "tests/test_helpers.py")
        assert len(chunks) == 1
        assert "test_foo" in chunks[0].text
        assert "helper" not in chunks[0].text.split("[Test:")[0]

    def test_skips_heavily_mocked(self):
        src = textwrap.dedent("""\
            def test_mocked(mocker):
                mocker.patch("a.b")
                mocker.patch("c.d")
                mocker.patch("e.f")
                assert True
        """)
        chunks = _extract_test_usage(src, "tests/test_mock.py")
        assert len(chunks) == 0

    def test_skips_long_functions(self):
        body = "\n".join(f"    x{i} = {i}" for i in range(200))
        src = f"def test_long():\n{body}\n"
        chunks = _extract_test_usage(src, "tests/test_long.py")
        assert len(chunks) == 0

    def test_includes_import_block(self):
        src = textwrap.dedent("""\
            import pytest
            from divi.qprog import VQE

            def test_import_visible():
                pass
        """)
        chunks = _extract_test_usage(src, "tests/test_imports.py")
        assert len(chunks) == 1
        assert "import pytest" in chunks[0].text

    def test_syntax_error_returns_empty(self):
        assert _extract_test_usage("def broken(", "tests/test_bad.py") == []
