# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.circuits._qasm_template."""

import pytest

from divi.circuits import build_template, render_template


class TestBuildTemplate:
    @pytest.mark.parametrize(
        "body, symbols, fragments, slot_indices",
        [
            pytest.param(
                "OPENQASM 2.0;\nrx(1.0) q[0];\n",
                (),
                ("OPENQASM 2.0;\nrx(1.0) q[0];\n",),
                (),
                id="no_symbols",
            ),
            pytest.param(
                "rx(w_0) q[0];\n",
                ("w_0",),
                ("rx(", ") q[0];\n"),
                (0,),
                id="single_symbol",
            ),
            pytest.param(
                "rx(w_0) q[0];\nry(w_0) q[1];\n",
                ("w_0",),
                ("rx(", ") q[0];\nry(", ") q[1];\n"),
                (0, 0),
                id="repeated_symbol",
            ),
            pytest.param(
                "rx(a) q[0];\nry(b) q[1];\n",
                ("a", "b"),
                ("rx(", ") q[0];\nry(", ") q[1];\n"),
                (0, 1),
                id="multiple_symbols",
            ),
        ],
    )
    def test_splits_body_into_fragments_and_slots(
        self, body, symbols, fragments, slot_indices
    ):
        template = build_template(body, symbols)
        assert template.fragments == fragments
        assert template.slot_indices == slot_indices

    def test_overlapping_names_longest_first(self):
        """w_1 must not match inside w_10."""
        body = "rx(w_1) q[0];\nry(w_10) q[1];\n"
        template = build_template(body, ("w_1", "w_10"))
        assert template.slot_indices == (0, 1)
        rendered = render_template(template, ("A", "B"))
        assert rendered == "rx(A) q[0];\nry(B) q[1];\n"

    def test_invariant_len_fragments(self):
        body = "rx(x) q[0];\nry(y) q[1];\nrz(x) q[2];\n"
        template = build_template(body, ("x", "y"))
        assert len(template.fragments) == len(template.slot_indices) + 1

    def test_short_symbol_does_not_match_inside_keyword(self):
        """A parameter named ``x`` must not match the ``x`` in ``cx``,
        ``rx``, or any other QASM identifier where it is a substring."""
        body = "rx(x) q[0];\ncx q[0],q[1];\n"
        template = build_template(body, ("x",))
        rendered = render_template(template, ("1.5",))
        # Only the parametric slot in `rx(x)` should be replaced; the
        # `x` inside `rx` and `cx` must survive verbatim.
        assert rendered == "rx(1.5) q[0];\ncx q[0],q[1];\n"

    def test_symbol_with_brackets_does_not_match_qubit_index(self):
        """A parameter named ``x[0]`` must not match the ``[0]`` in
        ``q[0]`` qubit indexing."""
        body = "rx(x[0]) q[0];\nry(x[1]) q[1];\n"
        template = build_template(body, ("x[0]", "x[1]"))
        rendered = render_template(template, ("1.0", "2.0"))
        assert rendered == "rx(1.0) q[0];\nry(2.0) q[1];\n"

    def test_longest_symbol_wins_when_prefix_is_followed_by_non_identifier(self):
        body = "rx(x[0]) q[0];\nrz(x) q[1];\n"
        template = build_template(body, ("x", "x[0]"))
        assert render_template(template, ("A", "B")) == "rx(B) q[0];\nrz(A) q[1];\n"

    def test_letters_of_either_case_block_a_match(self):
        body = "rx(Ax) q[0];\nry(xB) q[0];\nrz(xb) q[0];\nrz(x) q[0];\n"
        template = build_template(body, ("x",))
        assert render_template(template, ("1.5",)) == (
            "rx(Ax) q[0];\nry(xB) q[0];\nrz(xb) q[0];\nrz(1.5) q[0];\n"
        )


class TestRenderTemplate:
    def test_render_replaces_all_symbols_with_values(self):
        """All symbol names are replaced and each value appears the expected number of times."""
        body = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\nry(w_0) q[0];\nry(w_1) q[1];\nry(w_2) q[2];\ncx q[0],q[1];\nrz(w_0) q[1];\n'
        symbols = ("w_0", "w_1", "w_2")
        values = ("1.5", "2.7", "0.3")

        template = build_template(body, symbols)
        result = render_template(template, values)

        # No symbol names remain in the output
        for sym in symbols:
            assert sym not in result

        # Each value appears the same number of times its symbol did in the body
        assert result.count("1.5") == 2  # w_0 appears twice (ry + rz)
        assert result.count("2.7") == 1  # w_1 appears once
        assert result.count("0.3") == 1  # w_2 appears once

        # Non-parameterized structure is preserved
        assert result.startswith('OPENQASM 2.0;\ninclude "qelib1.inc";')
        assert "cx q[0],q[1];" in result

    def test_overlapping_prefix_symbols_substitute_correctly(self):
        """Symbols that are prefixes of each other (w, w_0, w_01) are each replaced independently."""
        body = "rx(w) q[0];\nry(w_0) q[1];\nrz(w_01) q[2];\n"
        symbols = ("w", "w_0", "w_01")
        values = ("A", "B", "C")

        template = build_template(body, symbols)
        result = render_template(template, values)

        assert result == "rx(A) q[0];\nry(B) q[1];\nrz(C) q[2];\n"

    def test_empty_slots(self):
        body = "rx(1.0) q[0];\n"
        template = build_template(body, ())
        assert render_template(template, ()) == body

    def test_values_with_special_chars(self):
        """Numeric values should be inserted verbatim, no regex escaping needed."""
        body = "rx(w_0) q[0];\n"
        template = build_template(body, ("w_0",))
        result = render_template(template, ("-3.14159",))
        assert result == "rx(-3.14159) q[0];\n"
