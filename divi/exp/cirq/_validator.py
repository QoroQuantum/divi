# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import NamedTuple

# ---------- Lexer ----------
_WS_RE = re.compile(r"\s+")
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)

TOKEN_SPECS = [
    ("ARROW", r"->"),
    ("EQ", r"=="),
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACKET", r"\["),
    ("RBRACKET", r"\]"),
    ("COMMA", r","),
    ("SEMI", r";"),
    ("STAR", r"\*"),
    ("SLASH", r"/"),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("CARET", r"\^"),
    ("STRING", r"\"[^\"\n]*\""),
    ("NUMBER", r"\d+(?:\.\d+)?"),
    ("ID", r"[A-Za-z_][A-Za-z0-9_]*"),
]
TOKEN_REGEX = re.compile("|".join(f"(?P<{n}>{p})" for n, p in TOKEN_SPECS))

KEYWORDS = {
    "OPENQASM",
    "include",
    "qreg",
    "creg",
    "qubit",
    "bit",
    "gate",
    "barrier",
    "measure",
    "reset",
    "if",
    "pi",
    "sin",
    "cos",
    "tan",
    "exp",
    "ln",
    "sqrt",
    "acos",
    "atan",
    "asin",
}


class Tok(NamedTuple):
    type: str
    value: str
    pos: int
    line: int
    col: int


def _strip_comments(src: str) -> str:
    return _LINE_COMMENT_RE.sub("", _BLOCK_COMMENT_RE.sub("", src))


def _lex(src: str) -> list[Tok]:
    src = _strip_comments(src)
    i, n = 0, len(src)
    line, line_start = 1, 0
    out: list[Tok] = []
    while i < n:
        m = _WS_RE.match(src, i)
        if m:
            chunk = src[i : m.end()]
            nl = chunk.count("\n")
            if nl:
                line += nl
                line_start = m.end() - (len(chunk) - chunk.rfind("\n") - 1)
            i = m.end()
            continue
        m = TOKEN_REGEX.match(src, i)
        if not m:
            snippet = src[i : i + 20].replace("\n", "\\n")
            raise SyntaxError(
                f"Illegal character at {line}:{i-line_start+1}: {snippet!r}"
            )
        kind = m.lastgroup
        val = m.group(kind)
        col = i - line_start + 1
        if kind == "ID" and val in KEYWORDS:
            kind = val.upper()
        out.append(Tok(kind, val, i, line, col))
        i = m.end()
    out.append(Tok("EOF", "", i, line, i - line_start + 1))
    return out


# ---------- Built-in gates (name -> (num_params, num_qubits)) ----------
BUILTINS: dict[str, tuple[int, int]] = {
    # 1q
    "id": (0, 1),
    "x": (0, 1),
    "y": (0, 1),
    "z": (0, 1),
    "h": (0, 1),
    "s": (0, 1),
    "sdg": (0, 1),
    "t": (0, 1),
    "tdg": (0, 1),
    "sx": (0, 1),
    "sxdg": (0, 1),
    "rx": (1, 1),
    "ry": (1, 1),
    "rz": (1, 1),
    "u1": (1, 1),
    "u2": (2, 1),
    "u3": (3, 1),
    "u": (3, 1),  # allow 'u' alias
    "U": (3, 1),
    # 2q
    "cx": (0, 2),
    "cy": (0, 2),
    "cz": (0, 2),
    "iswap": (0, 2),
    "swap": (0, 2),
    "rxx": (1, 2),
    "ryy": (1, 2),
    "rzz": (1, 2),
    "crx": (1, 2),
    "cry": (1, 2),
    "crz": (1, 2),
    "cu1": (1, 2),
    "cu3": (3, 2),
    "ch": (0, 2),
    # 3q
    "ccx": (0, 3),
    "cswap": (0, 3),
}

_MATH_FUNCS = {"SIN", "COS", "TAN", "EXP", "LN", "SQRT", "ACOS", "ATAN", "ASIN"}


# ---------- Parser with symbol checks ----------
class Parser:
    def __init__(self, toks: list[Tok]):
        self.toks = toks
        self.i = 0
        # symbols
        self.qregs: dict[str, int] = {}
        self.cregs: dict[str, int] = {}
        self.user_gates: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
        # gate-def scope
        self.in_gate_def = False
        self.g_params: set[str] = set()
        self.g_qubits: set[str] = set()

    # -- helpers --
    def peek(self, k=0) -> Tok:
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else self.toks[-1]

    def match(self, *types: str) -> Tok:
        t = self.peek()
        if t.type in types:
            self.i += 1
            return t
        exp = " or ".join(types)
        raise SyntaxError(f"Expected {exp} at {t.line}:{t.col}, got {t.type}")

    def accept(self, *types: str) -> Tok | None:
        if self.peek().type in types:
            return self.match(*types)
        return None

    # -- entry --
    def parse(self):
        self.header()
        while self.accept("INCLUDE"):
            self.include_stmt()
        while self.peek().type != "EOF":
            start_line = self.peek().line
            self.statement()
            # After statement, check if it ended correctly
            prev_tok = self.toks[self.i - 1] if self.i > 0 else None
            # A statement is valid if it ends in a semicolon OR a closing brace (for gates)
            if not prev_tok or (prev_tok.type != "SEMI" and prev_tok.type != "RBRACE"):
                raise SyntaxError(
                    f"Statement at line {start_line} must end with a semicolon or a closing brace."
                )
        self.match("EOF")

    # OPENQASM 2.0 ;
    def header(self):
        self.match("OPENQASM")
        v = self.match("NUMBER")
        if v.value not in ("2.0", "2", "3.0", "3"):
            raise SyntaxError(
                f"Unsupported OPENQASM version '{v.value}' at {v.line}:{v.col}"
            )
        self.match("SEMI")

    def include_stmt(self):
        self.match("STRING")
        self.match("SEMI")

    def statement(self):
        t = self.peek().type
        if t == "QREG":
            self.qreg_decl()
        elif t == "CREG":
            self.creg_decl()
        elif t == "QUBIT":
            self.qubit_decl()
        elif t == "BIT":
            self.bit_decl()
        elif t == "GATE":
            self.gate_def()
        elif t == "MEASURE":
            self.measure_stmt()
        elif t == "RESET":
            self.reset_stmt()
        elif t == "BARRIER":
            self.barrier_stmt()
        elif t == "IF":
            self.if_stmt()
        elif t == "ID":
            self.gate_op_stmt_top()
        else:
            tok = self.peek()
            raise SyntaxError(f"Unexpected token {tok.type} at {tok.line}:{tok.col}")

    # ---- declarations ----
    def qreg_decl(self):
        self.match("QREG")
        name = self.match("ID").value
        self.match("LBRACKET")
        n = self.natural_number_tok()
        self.match("RBRACKET")
        self.match("SEMI")
        if name in self.qregs or name in self.cregs:
            self._dupe(name)
        self.qregs[name] = n

    def creg_decl(self):
        self.match("CREG")
        name = self.match("ID").value
        self.match("LBRACKET")
        n = self.natural_number_tok()
        self.match("RBRACKET")
        self.match("SEMI")
        if name in self.qregs or name in self.cregs:
            self._dupe(name)
        self.cregs[name] = n

    def qubit_decl(self):
        self.match("QUBIT")
        if self.accept("LBRACKET"):
            n = self.natural_number_tok()
            self.match("RBRACKET")
            name = self.match("ID").value
        else:
            name = self.match("ID").value
            n = 1
        self.match("SEMI")
        if name in self.qregs or name in self.cregs:
            self._dupe(name)
        self.qregs[name] = n

    def bit_decl(self):
        self.match("BIT")
        if self.accept("LBRACKET"):
            n = self.natural_number_tok()
            self.match("RBRACKET")
            name = self.match("ID").value
        else:
            name = self.match("ID").value
            n = 1
        self.match("SEMI")
        if name in self.qregs or name in self.cregs:
            self._dupe(name)
        self.cregs[name] = n

    # ---- gate definitions ----
    def gate_def(self):
        self.match("GATE")
        gname = self.match("ID").value
        if gname in self.user_gates:
            self._dupe(gname)
        params: tuple[str, ...] = ()
        if self.accept("LPAREN"):
            params = self._id_list_tuple()
            self.match("RPAREN")
        qubits = self._id_list_tuple()
        # enter scope
        saved = (self.in_gate_def, self.g_params.copy(), self.g_qubits.copy())
        self.in_gate_def = True
        self.g_params = set(params)
        self.g_qubits = set(qubits)
        self.match("LBRACE")
        # body: only gate ops; they can use local qubit ids and local params in expr
        while self.peek().type == "ID":
            self.gate_op_stmt_in_body()
        self.match("RBRACE")
        # leave scope
        self.in_gate_def, self.g_params, self.g_qubits = saved
        self.user_gates[gname] = (params, qubits)

    def _id_list_tuple(self) -> tuple[str, ...]:
        ids = [self.match("ID").value]
        while self.accept("COMMA"):
            ids.append(self.match("ID").value)
        return tuple(ids)

    # ---- gate operations ----
    def gate_op_stmt_top(self):
        name_tok = self.match("ID")
        gname = name_tok.value
        param_count = None
        arity = None

        if self.accept("LPAREN"):
            n_params = self._expr_list_count(allow_id=False)  # top-level: no free IDs
            self.match("RPAREN")
        else:
            n_params = 0

        # resolve gate signature
        if gname in BUILTINS:
            param_count, arity = BUILTINS[gname]
        elif gname in self.user_gates:
            param_count, arity = len(self.user_gates[gname][0]), len(
                self.user_gates[gname][1]
            )
        else:
            self._unknown_gate(name_tok)

        if n_params != param_count:
            raise SyntaxError(
                f"Gate '{gname}' expects {param_count} params, got {n_params} at {name_tok.line}:{name_tok.col}"
            )

        args, reg_sizes = self.qarg_list_top(arity)
        # broadcast check: all full-register sizes must match if >1
        sizes = {s for s in reg_sizes if s > 1}
        if len(sizes) > 1:
            raise SyntaxError(
                f"Mismatched register sizes in arguments to '{gname}' at {name_tok.line}:{name_tok.col}"
            )

        self.match("SEMI")

    def gate_op_stmt_in_body(self):
        name_tok = self.match("ID")
        gname = name_tok.value

        if self.accept("LPAREN"):
            n_params = self._expr_list_count(allow_id=True)  # may use local params
            self.match("RPAREN")
        else:
            n_params = 0

        if gname in BUILTINS:
            param_count, arity = BUILTINS[gname]
        elif gname in self.user_gates:
            param_count, arity = len(self.user_gates[gname][0]), len(
                self.user_gates[gname][1]
            )
        else:
            self._unknown_gate(name_tok)

        if n_params != param_count:
            raise SyntaxError(
                f"Gate '{gname}' expects {param_count} params, got {n_params} at {name_tok.line}:{name_tok.col}"
            )

        # In gate bodies, qargs must be local gate-qubit identifiers (no indexing)
        qids = [self._gate_body_qid()]
        while self.accept("COMMA"):
            qids.append(self._gate_body_qid())
        if len(qids) != arity:
            raise SyntaxError(
                f"Gate '{gname}' expects {arity} qubit args in body, got {len(qids)} at {name_tok.line}:{name_tok.col}"
            )
        self.match("SEMI")

    def _gate_body_qid(self) -> str:
        if self.peek().type != "ID":
            t = self.peek()
            raise SyntaxError(f"Expected gate-qubit id at {t.line}:{t.col}")
        name = self.match("ID").value
        if name not in self.g_qubits:
            t = self.peek(-1)
            raise SyntaxError(
                f"Unknown gate-qubit '{name}' in gate body at {t.line}:{t.col}"
            )
        return name

    # qarg list at top-level: IDs may be full registers or indexed bits q[i]
    def qarg_list_top(
        self, expected_arity: int
    ) -> tuple[list[tuple[str, int | None]], list[int]]:
        args = [self.qarg_top()]
        while self.accept("COMMA"):
            args.append(self.qarg_top())
        if len(args) != expected_arity:
            t = self.peek()
            raise SyntaxError(
                f"Expected {expected_arity} qubit args, got {len(args)} at {t.line}:{t.col}"
            )
        # return sizes for broadcast check
        reg_sizes = [(self.qregs[name] if idx is None else 1) for (name, idx) in args]
        return args, reg_sizes

    def qarg_top(self) -> tuple[str, int | None]:
        name_tok = self.match("ID")
        name = name_tok.value
        if name not in self.qregs:
            raise SyntaxError(
                f"Unknown qreg '{name}' at {name_tok.line}:{name_tok.col}"
            )
        if self.accept("LBRACKET"):
            idx_tok = self.natural_number_tok_tok()
            self.match("RBRACKET")
            if int(idx_tok.value) >= self.qregs[name]:
                raise SyntaxError(
                    f"Qubit index {idx_tok.value} out of range for '{name}[{self.qregs[name]}]' at {idx_tok.line}:{idx_tok.col}"
                )
            return (name, int(idx_tok.value))
        return (name, None)  # full register

    # ---- measure/reset/barrier/if ----
    def measure_stmt(self):
        # two forms: measure qarg -> carg ;  |  carg = measure qarg ;
        if self.peek().type == "MEASURE":
            self.match("MEASURE")
            q_t, q_sz = self._measure_qarg()
            self.match("ARROW")
            c_t, c_sz = self._measure_carg()
        else:
            # handled only when starts with MEASURE in statement(), so unreachable
            raise SyntaxError("Internal: measure_stmt misuse")
        if q_sz != c_sz:
            t = self.peek()
            raise SyntaxError(
                f"Measurement size mismatch {q_sz} -> {c_sz} at {t.line}:{t.col}"
            )
        self.match("SEMI")

    def _measure_qarg(self) -> tuple[str, int]:
        name_tok = self.match("ID")
        name = name_tok.value
        if name not in self.qregs:
            raise SyntaxError(
                f"Unknown qreg '{name}' at {name_tok.line}:{name_tok.col}"
            )
        if self.accept("LBRACKET"):
            idx = self.natural_number_tok()
            self.match("RBRACKET")
            if idx >= self.qregs[name]:
                raise SyntaxError(f"Qubit index {idx} out of range for '{name}'")
            return (f"{name}[{idx}]", 1)
        return (name, self.qregs[name])

    def _measure_carg(self) -> tuple[str, int]:
        name_tok = self.match("ID")
        name = name_tok.value
        if name not in self.cregs:
            raise SyntaxError(
                f"Unknown creg '{name}' at {name_tok.line}:{name_tok.col}"
            )
        if self.accept("LBRACKET"):
            idx = self.natural_number_tok()
            self.match("RBRACKET")
            if idx >= self.cregs[name]:
                raise SyntaxError(f"Bit index {idx} out of range for '{name}'")
            return (f"{name}[{idx}]", 1)
        return (name, self.cregs[name])

    def reset_stmt(self):
        self.match("RESET")
        # allow full reg or single index
        name_tok = self.match("ID")
        name = name_tok.value
        if name not in self.qregs:
            raise SyntaxError(
                f"Unknown qreg '{name}' at {name_tok.line}:{name_tok.col}"
            )
        if self.accept("LBRACKET"):
            idx = self.natural_number_tok()
            self.match("RBRACKET")
            if idx >= self.qregs[name]:
                raise SyntaxError(f"Qubit index {idx} out of range for '{name}'")
        self.match("SEMI")

    def barrier_stmt(self):
        self.match("BARrier".upper())  # tolerate case in tokenization
        # barrier accepts one or more qargs (full regs and/or indices)
        self.qarg_top()
        while self.accept("COMMA"):
            self.qarg_top()
        self.match("SEMI")

    def if_stmt(self):
        self.match("IF")
        self.match("LPAREN")
        cname_tok = self.match("ID")
        cname = cname_tok.value
        if cname not in self.cregs:
            raise SyntaxError(
                f"Unknown creg '{cname}' at {cname_tok.line}:{cname_tok.col}"
            )
        self.match("EQ")
        val_tok = self.natural_number_tok_tok()
        self.match("RPAREN")
        if int(val_tok.value) >= (1 << self.cregs[cname]):
            raise SyntaxError(
                f"if() value {val_tok.value} exceeds creg width {self.cregs[cname]}"
            )
        # must be a single gate op
        self.gate_op_stmt_top()

    # ---- expressions (with symbol policy) ----
    def _expr_list_count(self, *, allow_id: bool) -> int:
        # count expressions in list; expressions may reference IDs only if allow_id
        count = 0
        self._expr(allow_id)
        count += 1
        while self.accept("COMMA"):
            self._expr(allow_id)
            count += 1
        return count

    def _expr(self, allow_id: bool):
        self._expr_addsub(allow_id)

    def _expr_addsub(self, allow_id: bool):
        self._expr_muldiv(allow_id)
        while self.peek().type in ("PLUS", "MINUS"):
            self.match(self.peek().type)
            self._expr_muldiv(allow_id)

    def _expr_muldiv(self, allow_id: bool):
        self._expr_power(allow_id)
        while self.peek().type in ("STAR", "SLASH"):
            self.match(self.peek().type)
            self._expr_power(allow_id)

    def _expr_power(self, allow_id: bool):
        self._expr_unary(allow_id)
        if self.peek().type == "CARET":
            self.match("CARET")
            self._expr_power(allow_id)

    def _expr_unary(self, allow_id: bool):
        if self.peek().type in ("PLUS", "MINUS"):
            self.match(self.peek().type)
        self._expr_atom(allow_id)

    def _expr_atom(self, allow_id: bool):
        t = self.peek()
        if t.type == "NUMBER":
            self.match("NUMBER")
            return
        if t.type == "PI":
            self.match("PI")
            return
        # ---- NEW BLOCK TO HANDLE MATH FUNCTIONS ----
        if t.type in _MATH_FUNCS:
            self.match(t.type)  # Consume the function name (e.g., COS)
            self.match("LPAREN")
            self._expr(allow_id)  # Parse the inner expression
            # Note: QASM 2.0 math functions only take one argument
            self.match("RPAREN")
            return
        # --------------------------------------------
        if t.type == "ID":
            # function call or plain ID
            id_tok = self.match("ID")
            ident = id_tok.value
            if self.accept("LPAREN"):
                # This now correctly handles user-defined functions (if any)
                if self.peek().type != "RPAREN":
                    self._expr(allow_id)
                    while self.accept("COMMA"):
                        self._expr(allow_id)
                self.match("RPAREN")
                return
            # bare identifier: only allowed if in gate body params and allow_id=True
            if not allow_id or ident not in self.g_params:
                raise SyntaxError(
                    f"Unknown symbol '{ident}' in expression at {id_tok.line}:{id_tok.col}"
                )
            return
        if t.type == "LPAREN":
            self.match("LPAREN")
            self._expr(allow_id)
            self.match("RPAREN")
            return
        raise SyntaxError(
            f"Unexpected token {t.type} in expression at {t.line}:{t.col}"
        )

    # ---- numbers / errors ----
    def natural_number_tok(self) -> int:
        t = self.match("NUMBER")
        if "." in t.value:
            raise SyntaxError(
                f"Expected natural number at {t.line}:{t.col}, got {t.value}"
            )
        return int(t.value)

    def natural_number_tok_tok(self) -> Tok:
        t = self.match("NUMBER")
        if "." in t.value:
            raise SyntaxError(
                f"Expected natural number at {t.line}:{t.col}, got {t.value}"
            )
        return t

    def _dupe(self, name: str):
        t = self.peek()
        raise SyntaxError(f"Redefinition of '{name}' at {t.line}:{t.col}")

    def _unknown_gate(self, tok: Tok):
        raise SyntaxError(f"Unknown gate '{tok.value}' at {tok.line}:{tok.col}")


# ---------- Public API ----------
def validate_qasm_raise(src: str) -> None:
    toks = _lex(src)
    Parser(toks).parse()


def is_valid_qasm(src: str) -> bool:
    try:
        validate_qasm_raise(src)
        return True
    except SyntaxError:
        return False
