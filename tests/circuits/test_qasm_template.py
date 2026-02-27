# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.circuits._qasm_template."""

import re

from divi.circuits._qasm_template import build_template, render_template


class TestBuildTemplate:
    def test_no_symbols(self):
        body = "OPENQASM 2.0;\nrx(1.0) q[0];\n"
        template = build_template(body, ())
        assert template.fragments == (body,)
        assert template.slot_indices == ()

    def test_single_symbol(self):
        body = "rx(w_0) q[0];\n"
        template = build_template(body, ("w_0",))
        assert template.fragments == ("rx(", ") q[0];\n")
        assert template.slot_indices == (0,)

    def test_repeated_symbol(self):
        body = "rx(w_0) q[0];\nry(w_0) q[1];\n"
        template = build_template(body, ("w_0",))
        assert template.fragments == ("rx(", ") q[0];\nry(", ") q[1];\n")
        assert template.slot_indices == (0, 0)

    def test_multiple_symbols(self):
        body = "rx(a) q[0];\nry(b) q[1];\n"
        template = build_template(body, ("a", "b"))
        assert template.fragments == ("rx(", ") q[0];\nry(", ") q[1];\n")
        assert template.slot_indices == (0, 1)

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


class TestRenderTemplate:
    def test_roundtrip_matches_regex(self):
        """Rendered template must produce the same string as a correctly-ordered regex sub."""
        body = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\nry(w_0) q[0];\nry(w_1) q[1];\nry(w_2) q[2];\ncx q[0],q[1];\nrz(w_0) q[1];\n'
        symbols = ("w_0", "w_1", "w_2")
        values = ("1.5", "2.7", "0.3")

        # Reference: longest-first regex substitution
        escaped = sorted((re.escape(s) for s in symbols), key=len, reverse=True)
        pattern = re.compile("|".join(escaped))
        mapping = dict(zip(symbols, values))
        expected = pattern.sub(lambda m: mapping[m.group(0)], body)

        template = build_template(body, symbols)
        result = render_template(template, values)
        assert result == expected

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
