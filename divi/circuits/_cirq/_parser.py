# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sympy
from cirq.contrib.qasm_import._lexer import QasmLexer
from cirq.contrib.qasm_import._parser import QasmParser
from cirq.contrib.qasm_import.exception import QasmException
from ply import yacc


class ExtendedQasmLexer(QasmLexer):
    """Extended lexer with INPUT and ANGLE keywords."""

    reserved = {
        **QasmLexer.reserved,
        "input": "INPUT",
        "angle": "ANGLE",
    }
    # Rebuild tokens list to include new reserved keywords
    tokens = [
        "FORMAT_SPEC",
        "NUMBER",
        "NATURAL_NUMBER",
        "STDGATESINC",
        "QELIBINC",
        "ID",
        "ARROW",
        "EQ",
    ] + list(reserved.values())

    # PLY quirk: When overriding t_ID, PLY's method discovery doesn't find inherited
    # token methods. These three have multi-character patterns that MUST be checked
    # before t_ID (PLY matches longest patterns first). Without explicit definitions,
    # "OPENQASM" and "include" would be tokenized as ID instead of their proper types.
    # We keep the regex in the docstring (required by PLY) but delegate to parent.
    def t_FORMAT_SPEC(self, t):
        r"""OPENQASM(\s+)([^\s\t\;]*);"""
        return super().t_FORMAT_SPEC(t)

    def t_QELIBINC(self, t):
        r"""include(\s+)"qelib1.inc";"""
        return super().t_QELIBINC(t)

    def t_STDGATESINC(self, t):
        r"""include(\s+)"stdgates.inc";"""
        return super().t_STDGATESINC(t)

    # Override t_ID to check both parent and extended reserved dicts
    def t_ID(self, t):
        r"""[^\W\d_][\w_]*"""
        # This regex matches any Unicode letter (not digit/underscore) at the start,
        # followed by any number of Unicode word characters or underscores.
        # Check extended reserved first, then parent's
        if t.value in self.reserved:
            t.type = self.reserved[t.value]
        elif t.value in QasmLexer.reserved:
            t.type = QasmLexer.reserved[t.value]
        return t


class ExtendedQasmParser(QasmParser):
    """Extended parser with QASM 3.0 parameter support.

    Only adds support for: input angle[32] param_name;
    """

    # Add new tokens
    tokens = QasmParser.tokens + ["INPUT", "ANGLE"]

    def __init__(self) -> None:
        # Initialize parameter storage
        self.input_params: dict[str, sympy.Symbol] = {}

        # Call parent to set up everything (this will create its own lexer and parser)
        super().__init__()

        self.lexer = ExtendedQasmLexer()

        self.parser = yacc.yacc(module=self, debug=False, write_tables=False)

    # Add parser rule for input angle declarations
    def p_circuit_input_angle(self, p):
        """circuit : input_angle circuit"""
        p[0] = self.circuit

    def p_input_angle(self, p):
        """input_angle : INPUT ANGLE '[' NATURAL_NUMBER ']' ID ';'"""
        param_name = p[6]
        self.input_params[param_name] = sympy.Symbol(param_name)
        p[0] = None

    # Override to check input_params first
    def p_expr_identifier(self, p):
        """expr : ID"""
        # Check input_params first (QASM 3.0 parameters)
        if p[1] in self.input_params:
            p[0] = self.input_params[p[1]]
            return

        # Fall back to parent logic (custom gate parameters)
        if not self.in_custom_gate_scope:
            raise QasmException(
                f"Parameter '{p[1]}' in line {p.lineno(1)} not supported"
            )
        if p[1] not in self.custom_gate_scoped_params:
            raise QasmException(f"Undefined parameter '{p[1]}' in line {p.lineno(1)}'")
        p[0] = sympy.Symbol(p[1])
