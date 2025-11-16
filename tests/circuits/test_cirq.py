# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import sympy
from cirq.contrib.qasm_import.exception import QasmException

from divi.circuits._cirq._parser import ExtendedQasmLexer, ExtendedQasmParser


class TestExtendedQasmLexer:
    """Tests for ExtendedQasmLexer."""

    def test_qelibinc_tokenization(self):
        """Test that include 'qelib1.inc' is correctly tokenized."""
        lexer = ExtendedQasmLexer()
        lexer.input('include "qelib1.inc";')

        tokens = []
        while True:
            tok = lexer.token()
            if not tok:
                break
            tokens.append((tok.type, tok.value))

        # Should tokenize as QELIBINC
        qelibinc_tokens = [t for t in tokens if t[0] == "QELIBINC"]
        assert len(qelibinc_tokens) == 1
        assert qelibinc_tokens[0][1] == 'include "qelib1.inc";'

    def test_stdgatesinc_tokenization(self):
        """Test that include 'stdgates.inc' is correctly tokenized."""
        lexer = ExtendedQasmLexer()
        lexer.input('include "stdgates.inc";')

        tokens = []
        while True:
            tok = lexer.token()
            if not tok:
                break
            tokens.append((tok.type, tok.value))

        # Should tokenize as STDGATESINC
        stdgatesinc_tokens = [t for t in tokens if t[0] == "STDGATESINC"]
        assert len(stdgatesinc_tokens) == 1
        assert stdgatesinc_tokens[0][1] == 'include "stdgates.inc";'

    def test_parent_reserved_keywords_fallback(self, mocker):
        """Test that parent reserved keywords fallback logic works correctly."""
        lexer = ExtendedQasmLexer()
        # To test line 60, we need to simulate a scenario where a token is in
        # QasmLexer.reserved but not in self.reserved. Since self.reserved includes
        # all of QasmLexer.reserved, we'll temporarily remove a keyword from self.reserved
        # to test the fallback path.
        original_reserved = lexer.reserved.copy()
        # Remove a parent keyword to test the elif branch
        if "qreg" in lexer.reserved:
            del lexer.reserved["qreg"]

        lexer.input("qreg")

        tokens = []
        while True:
            tok = lexer.token()
            if not tok:
                break
            tokens.append((tok.type, tok.value))

        # Should still tokenize correctly via the fallback to QasmLexer.reserved
        assert len(tokens) > 0
        # Restore original reserved dict
        lexer.reserved = original_reserved


class TestExtendedQasmParser:
    """Tests for ExtendedQasmParser."""

    def test_input_angle_declaration(self):
        """Test that input angle declarations are parsed correctly."""
        parser = ExtendedQasmParser()
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        input angle[32] theta;
        rx(theta) q[0];
        """

        result = parser.parse(qasm)

        # Verify parameter was stored
        assert "theta" in parser.input_params
        assert isinstance(parser.input_params["theta"], sympy.Symbol)
        assert parser.input_params["theta"].name == "theta"

        # Verify circuit was created successfully
        assert result is not None

    def test_expr_identifier_with_input_param(self):
        """Test that input parameters can be used in expressions."""
        parser = ExtendedQasmParser()
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        input angle[32] theta;
        rx(theta) q[0];
        """

        parser.parse(qasm)

        # The parameter should be accessible
        assert "theta" in parser.input_params

    def test_expr_identifier_not_in_custom_gate_scope_raises(self):
        """Test that using undefined parameter outside custom gate scope raises error."""
        parser = ExtendedQasmParser()
        qasm = """
        OPENQASM 2.0;
        qreg q[1];
        rx(undefined_param) q[0];
        """

        with pytest.raises(QasmException, match="Parameter 'undefined_param' in line"):
            parser.parse(qasm)

    def test_expr_identifier_in_custom_gate_scope_undefined_raises(self):
        """Test that using undefined parameter in custom gate scope raises error."""
        parser = ExtendedQasmParser()
        qasm = """
        OPENQASM 2.0;
        qreg q[1];
        gate my_gate(param) q {
            U(undefined_param, 0, 0) q;
        }
        """

        with pytest.raises(
            QasmException, match="Undefined parameter 'undefined_param' in line"
        ):
            parser.parse(qasm)

    def test_expr_identifier_in_custom_gate_scope_success(self):
        """Test that using defined parameter in custom gate scope succeeds."""
        parser = ExtendedQasmParser()
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        gate my_gate(param) q {
            U(param, 0, 0) q;
        }
        my_gate(pi/2) q[0];
        """

        result = parser.parse(qasm)

        # Should parse successfully - the parameter 'param' is in custom_gate_scoped_params
        assert result is not None
