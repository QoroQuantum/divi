# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Binary QASM encoder/decoder.

Provides a compact binary representation of OpenQASM 2.0 circuits using
Python's ``struct`` module. The binary format eliminates text boilerplate
and encodes gate operations as fixed-size records with 1-byte opcodes,
compact qubit indices, and float32 parameters.

The format is designed to compress significantly better than raw QASM text,
especially for large circuits with many parameterized gates.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Gate opcode table
# ---------------------------------------------------------------------------
# Opcode ranges encode gate arity and parameter count implicitly:
#   0x00-0x0F: 1-qubit, 0-param
#   0x10-0x1F: 1-qubit, 1-param
#   0x20-0x2F: 1-qubit, 2-param
#   0x30-0x3F: 1-qubit, 3-param
#   0x40-0x4F: 2-qubit, 0-param
#   0x50-0x5F: 2-qubit, 1-param
#   0x60-0x6F: 2-qubit, 3-param
#   0x70-0x7F: 3-qubit, 0-param
#   0xF0-0xFF: special instructions

GATE_OPCODES: dict[str, int] = {
    # 1-qubit, 0-param
    "id": 0x00,
    "x": 0x01,
    "y": 0x02,
    "z": 0x03,
    "h": 0x04,
    "s": 0x05,
    "sdg": 0x06,
    "t": 0x07,
    "tdg": 0x08,
    # 1-qubit, 1-param
    "rx": 0x10,
    "ry": 0x11,
    "rz": 0x12,
    "u1": 0x13,
    # 1-qubit, 2-param
    "u2": 0x20,
    # 1-qubit, 3-param
    "u3": 0x30,
    # 2-qubit, 0-param
    "cx": 0x40,
    "cz": 0x41,
    "swap": 0x42,
    "ch": 0x43,
    "iswap": 0x44,
    # 2-qubit, 1-param
    "crx": 0x50,
    "cry": 0x51,
    "crz": 0x52,
    "cu1": 0x53,
    "rxx": 0x54,
    "ryy": 0x55,
    "rzz": 0x56,
    # 2-qubit, 3-param
    "cu3": 0x60,
    # 3-qubit, 0-param
    "ccx": 0x70,
    "cswap": 0x71,
    # Special
    "barrier": 0xF0,
    "reset": 0xF1,
    "measure": 0xF2,
}

OPCODE_TO_GATE: dict[int, str] = {v: k for k, v in GATE_OPCODES.items()}

# Header flags
_FLAG_HAS_MEASUREMENTS = 0x01
_FLAG_MEASURE_ALL = 0x02

# Magic bytes
_MAGIC = b"BQ"
_VERSION = 0x01


def _gate_arity(opcode: int) -> int:
    """Return the number of qubit operands for *opcode*."""
    hi = opcode & 0xF0
    if hi in (0x00, 0x10, 0x20, 0x30):
        return 1
    if hi in (0x40, 0x50, 0x60):
        return 2
    if hi == 0x70:
        return 3
    # specials
    if opcode == 0xF0:  # barrier — variable, encoded separately
        return 0
    if opcode == 0xF1:  # reset
        return 1
    if opcode == 0xF2:  # measure
        return 1
    return 0


def _gate_n_params(opcode: int) -> int:
    """Return the number of float parameters for *opcode*."""
    hi = opcode & 0xF0
    if hi == 0x10 or hi == 0x50:
        return 1
    if hi == 0x20:
        return 2
    if hi == 0x30 or hi == 0x60:
        return 3
    return 0


# ---------------------------------------------------------------------------
# Intermediate representation
# ---------------------------------------------------------------------------


@dataclass
class Instruction:
    gate: str
    qubits: list[int]
    params: list[float] = field(default_factory=list)


@dataclass
class Measurement:
    qubit: int
    clbit: int


@dataclass
class CircuitIR:
    """Intermediate representation of a parsed QASM circuit."""

    n_qubits: int
    n_clbits: int
    instructions: list[Instruction]
    measurements: list[Measurement]


# ---------------------------------------------------------------------------
# QASM text → IR parser (regex-based, fast)
# ---------------------------------------------------------------------------

_RE_QREG = re.compile(r"qreg\s+\w+\[(\d+)\]\s*;")
_RE_CREG = re.compile(r"creg\s+\w+\[(\d+)\]\s*;")
_RE_MEASURE = re.compile(
    r"measure\s+\w+\[(\d+)\]\s*->\s*\w+\[(\d+)\]\s*;"
)
# Gate pattern: gate_name(params) qubit_args;
# Handles optional params and multiple comma-separated qubit args.
_RE_GATE = re.compile(
    r"([a-z][a-z0-9]*)"  # gate name
    r"(?:\(([^)]*)\))?"  # optional params in parens
    r"\s+"
    r"(\w+\[\d+\](?:\s*,\s*\w+\[\d+\])*)"  # qubit args
    r"\s*;"
)
_RE_QUBIT_IDX = re.compile(r"\w+\[(\d+)\]")
_RE_BARRIER = re.compile(r"barrier\b[^;]*;")
_RE_RESET = re.compile(r"reset\s+\w+\[(\d+)\]\s*;")

# Lines to skip during parsing
_SKIP_PREFIXES = ("OPENQASM", "include", "qreg", "creg", "//", "gate", "input")


def _eval_param(s: str) -> float:
    """Evaluate a QASM parameter expression to a float.

    Handles numeric literals, ``pi``, and simple arithmetic involving pi.
    """
    import math

    s = s.strip()
    # Replace 'pi' with its value for eval (safe: only numeric + pi expressions)
    expr = s.replace("pi", str(math.pi))
    try:
        return float(expr)
    except ValueError:
        # Fall back to simple eval for expressions like "pi/2", "2*pi"
        # Only allow safe characters
        if re.fullmatch(r"[\d.eE+\-*/() ]+", expr):
            return float(eval(expr))  # noqa: S307
        raise ValueError(f"Cannot evaluate parameter: {s!r}")


def parse_qasm(qasm: str) -> CircuitIR:
    """Parse an OpenQASM 2.0 string into a :class:`CircuitIR`."""
    n_qubits = 0
    n_clbits = 0
    instructions: list[Instruction] = []
    measurements: list[Measurement] = []

    for m in _RE_QREG.finditer(qasm):
        n_qubits += int(m.group(1))
    for m in _RE_CREG.finditer(qasm):
        n_clbits += int(m.group(1))

    # Strip header/register lines and parse instruction lines
    for line in qasm.splitlines():
        line = line.strip()
        if not line or any(line.startswith(p) for p in _SKIP_PREFIXES):
            continue

        # Barrier
        bm = _RE_BARRIER.match(line)
        if bm:
            instructions.append(Instruction("barrier", []))
            continue

        # Reset
        rm = _RE_RESET.match(line)
        if rm:
            instructions.append(Instruction("reset", [int(rm.group(1))]))
            continue

        # Measurement
        mm = _RE_MEASURE.match(line)
        if mm:
            measurements.append(Measurement(int(mm.group(1)), int(mm.group(2))))
            continue

        # Gate instruction
        gm = _RE_GATE.match(line)
        if gm:
            gate_name = gm.group(1)
            params_str = gm.group(2)
            qubits_str = gm.group(3)

            params: list[float] = []
            if params_str:
                for p in params_str.split(","):
                    params.append(_eval_param(p))

            qubits = [int(x.group(1)) for x in _RE_QUBIT_IDX.finditer(qubits_str)]
            instructions.append(Instruction(gate_name, qubits, params))

    return CircuitIR(n_qubits, n_clbits, instructions, measurements)


# ---------------------------------------------------------------------------
# IR → QASM text reconstruction
# ---------------------------------------------------------------------------


def ir_to_qasm(ir: CircuitIR) -> str:
    """Reconstruct an OpenQASM 2.0 string from a :class:`CircuitIR`."""
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{ir.n_qubits}];",
        f"creg c[{ir.n_clbits}];",
    ]

    for inst in ir.instructions:
        if inst.gate == "barrier":
            lines.append("barrier;")
            continue
        if inst.gate == "reset":
            lines.append(f"reset q[{inst.qubits[0]}];")
            continue

        qubit_args = ",".join(f"q[{q}]" for q in inst.qubits)
        if inst.params:
            param_str = ",".join(f"{p:.8f}" for p in inst.params)
            lines.append(f"{inst.gate}({param_str}) {qubit_args};")
        else:
            lines.append(f"{inst.gate} {qubit_args};")

    for meas in ir.measurements:
        lines.append(f"measure q[{meas.qubit}] -> c[{meas.clbit}];")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Binary encoder (struct-based)
# ---------------------------------------------------------------------------


def encode(qasm: str) -> bytes:
    """Encode an OpenQASM 2.0 string into the compact binary format.

    Returns raw bytes (not compressed — apply a compressor on top).
    """
    ir = parse_qasm(qasm)
    return encode_ir(ir)


def encode_ir(ir: CircuitIR) -> bytes:
    """Encode a :class:`CircuitIR` into the compact binary format."""
    # Determine if all qubits are measured to all matching classical bits
    measure_all = (
        len(ir.measurements) == ir.n_qubits
        and all(m.qubit == m.clbit for m in ir.measurements)
        and len(set(m.qubit for m in ir.measurements)) == ir.n_qubits
    )
    has_measurements = len(ir.measurements) > 0

    flags = 0
    if has_measurements:
        flags |= _FLAG_HAS_MEASUREMENTS
    if measure_all:
        flags |= _FLAG_MEASURE_ALL

    # Use 1-byte qubit indices if n_qubits <= 255, else 2-byte
    wide = ir.n_qubits > 255 or ir.n_clbits > 255
    idx_fmt = ">H" if wide else ">B"
    idx_size = 2 if wide else 1

    if wide:
        flags |= 0x04  # bit 2: wide indices

    parts: list[bytes] = []

    # Header: magic(2) + version(1) + n_qubits(2) + n_clbits(2) + n_gates(4) + flags(1) = 12 bytes
    parts.append(
        struct.pack(
            ">2sBHHIB",
            _MAGIC,
            _VERSION,
            ir.n_qubits,
            ir.n_clbits,
            len(ir.instructions),
            flags,
        )
    )

    # Instructions
    for inst in ir.instructions:
        opcode = GATE_OPCODES.get(inst.gate)
        if opcode is None:
            raise ValueError(f"Unsupported gate: {inst.gate!r}")

        parts.append(struct.pack(">B", opcode))

        # Qubit operands
        for q in inst.qubits:
            parts.append(struct.pack(idx_fmt, q))

        # Float32 parameters
        for p in inst.params:
            parts.append(struct.pack(">f", p))

    # Measurements (only if not measure_all)
    if has_measurements and not measure_all:
        parts.append(struct.pack(">H", len(ir.measurements)))
        for meas in ir.measurements:
            parts.append(struct.pack(idx_fmt, meas.qubit))
            parts.append(struct.pack(idx_fmt, meas.clbit))

    return b"".join(parts)


# ---------------------------------------------------------------------------
# Binary decoder (struct-based)
# ---------------------------------------------------------------------------


def decode(data: bytes) -> str:
    """Decode binary data back into an OpenQASM 2.0 string."""
    ir = decode_to_ir(data)
    return ir_to_qasm(ir)


def decode_to_ir(data: bytes) -> CircuitIR:
    """Decode binary data into a :class:`CircuitIR`."""
    offset = 0

    # Header
    magic, version, n_qubits, n_clbits, n_gates, flags = struct.unpack_from(
        ">2sBHHIB", data, offset
    )
    offset += struct.calcsize(">2sBHHIB")

    if magic != _MAGIC:
        raise ValueError(f"Invalid magic bytes: {magic!r}")
    if version != _VERSION:
        raise ValueError(f"Unsupported version: {version}")

    has_measurements = bool(flags & _FLAG_HAS_MEASUREMENTS)
    measure_all = bool(flags & _FLAG_MEASURE_ALL)
    wide = bool(flags & 0x04)

    idx_fmt = ">H" if wide else ">B"
    idx_size = 2 if wide else 1

    # Instructions
    instructions: list[Instruction] = []
    for _ in range(n_gates):
        (opcode,) = struct.unpack_from(">B", data, offset)
        offset += 1

        gate_name = OPCODE_TO_GATE.get(opcode)
        if gate_name is None:
            raise ValueError(f"Unknown opcode: 0x{opcode:02X}")

        arity = _gate_arity(opcode)
        n_params = _gate_n_params(opcode)

        qubits = []
        for _ in range(arity):
            (q,) = struct.unpack_from(idx_fmt, data, offset)
            offset += idx_size
            qubits.append(q)

        params = []
        for _ in range(n_params):
            (p,) = struct.unpack_from(">f", data, offset)
            offset += 4
            params.append(p)

        instructions.append(Instruction(gate_name, qubits, params))

    # Measurements
    measurements: list[Measurement] = []
    if has_measurements:
        if measure_all:
            measurements = [Measurement(i, i) for i in range(n_qubits)]
        else:
            (n_meas,) = struct.unpack_from(">H", data, offset)
            offset += 2
            for _ in range(n_meas):
                (qubit,) = struct.unpack_from(idx_fmt, data, offset)
                offset += idx_size
                (clbit,) = struct.unpack_from(idx_fmt, data, offset)
                offset += idx_size
                measurements.append(Measurement(qubit, clbit))

    return CircuitIR(n_qubits, n_clbits, instructions, measurements)


# ---------------------------------------------------------------------------
# Columnar encoder (v2) — splits data into homogeneous streams
# ---------------------------------------------------------------------------
# Instead of interleaving [opcode, qubits, params] per gate, we separate
# the circuit into three arrays:
#   1. opcode stream   — uint8 array (very low alphabet → compresses well)
#   2. qubit idx stream — uint8/uint16 array (small ints)
#   3. param stream     — float32 array (IEEE 754 floats)
#
# Each stream is internally homogeneous, which lets general-purpose
# compressors exploit patterns within each data type far more effectively.

_MAGIC_COL = b"BX"
_VERSION_COL = 0x02

# Flag bits for columnar format
_COL_FLAG_HAS_MEAS = 0x01
_COL_FLAG_MEAS_ALL = 0x02
_COL_FLAG_WIDE_IDX = 0x04
_COL_FLAG_SHUFFLED = 0x08  # float byte-shuffle enabled


def _zigzag_encode(values: list[int]) -> list[int]:
    """Zigzag-encode signed integers to unsigned (protobuf-style)."""
    return [(v << 1) ^ (v >> 31) for v in values]


def _zigzag_decode(values: list[int]) -> list[int]:
    """Reverse zigzag encoding."""
    return [(v >> 1) ^ -(v & 1) for v in values]


def _byte_shuffle(data: bytes, element_size: int) -> bytes:
    """Transpose byte order within fixed-size elements.

    For N elements of *element_size* bytes, rearrange so that all byte-0s
    come first, then all byte-1s, etc.  This clusters similar-valued bytes
    together (e.g. IEEE 754 exponent bytes) and dramatically improves
    compression ratios.

    This is the same technique used by HDF5/blosc for numeric arrays.
    """
    n = len(data) // element_size
    if n == 0:
        return data
    arr = np.frombuffer(data, dtype=np.uint8).reshape(n, element_size)
    return arr.T.tobytes()


def _byte_unshuffle(data: bytes, element_size: int, n_elements: int) -> bytes:
    """Reverse :func:`_byte_shuffle`."""
    if n_elements == 0:
        return data
    arr = np.frombuffer(data, dtype=np.uint8).reshape(element_size, n_elements)
    return arr.T.tobytes()


def encode_columnar(ir: CircuitIR, shuffle_floats: bool = True) -> bytes:
    """Columnar binary encoder — separates opcodes, indices, and params.

    Parameters
    ----------
    ir : CircuitIR
        The circuit to encode.
    shuffle_floats : bool
        If True, apply byte-shuffling to the float parameter stream.
        This reorders bytes so that the exponent bytes (low entropy) are
        grouped together, dramatically improving compression.

    Returns
    -------
    bytes
        Raw columnar binary (apply a compressor on top).
    """
    measure_all = (
        len(ir.measurements) == ir.n_qubits
        and all(m.qubit == m.clbit for m in ir.measurements)
        and len(set(m.qubit for m in ir.measurements)) == ir.n_qubits
    )
    has_meas = len(ir.measurements) > 0
    wide = ir.n_qubits > 255 or ir.n_clbits > 255

    flags = 0
    if has_meas:
        flags |= _COL_FLAG_HAS_MEAS
    if measure_all:
        flags |= _COL_FLAG_MEAS_ALL
    if wide:
        flags |= _COL_FLAG_WIDE_IDX
    if shuffle_floats:
        flags |= _COL_FLAG_SHUFFLED

    # Build separate streams
    opcodes = bytearray()
    qubit_indices: list[int] = []
    params: list[float] = []

    for inst in ir.instructions:
        opc = GATE_OPCODES.get(inst.gate)
        if opc is None:
            raise ValueError(f"Unsupported gate: {inst.gate!r}")
        opcodes.append(opc)
        qubit_indices.extend(inst.qubits)
        params.extend(inst.params)

    # Pack qubit indices as numpy array for speed
    idx_dtype = np.uint16 if wide else np.uint8
    qubit_arr = np.array(qubit_indices, dtype=idx_dtype)
    qubit_bytes = qubit_arr.tobytes()

    # Pack parameters as float32 numpy array
    param_arr = np.array(params, dtype=np.float32)
    param_bytes = param_arr.tobytes()

    # Apply byte shuffling to float stream
    if shuffle_floats and len(params) > 0:
        param_bytes = _byte_shuffle(param_bytes, 4)

    # Header: magic(2) + version(1) + flags(1) + n_qubits(2) + n_clbits(2)
    #       + n_gates(4) + n_qubit_indices(4) + n_params(4) = 20 bytes
    header = struct.pack(
        ">2sBBHHIII",
        _MAGIC_COL,
        _VERSION_COL,
        flags,
        ir.n_qubits,
        ir.n_clbits,
        len(ir.instructions),
        len(qubit_indices),
        len(params),
    )

    parts = [header, bytes(opcodes), qubit_bytes, param_bytes]

    # Measurement stream (only if not measure_all)
    if has_meas and not measure_all:
        meas_qubits = np.array([m.qubit for m in ir.measurements], dtype=idx_dtype)
        meas_clbits = np.array([m.clbit for m in ir.measurements], dtype=idx_dtype)
        parts.append(struct.pack(">H", len(ir.measurements)))
        parts.append(meas_qubits.tobytes())
        parts.append(meas_clbits.tobytes())

    return b"".join(parts)


def decode_columnar(data: bytes) -> CircuitIR:
    """Decode columnar binary back into a :class:`CircuitIR`."""
    off = 0

    # Header
    magic, version, flags, n_qubits, n_clbits, n_gates, n_qidx, n_params = (
        struct.unpack_from(">2sBBHHIII", data, off)
    )
    off += struct.calcsize(">2sBBHHIII")

    if magic != _MAGIC_COL:
        raise ValueError(f"Invalid magic: {magic!r}")
    if version != _VERSION_COL:
        raise ValueError(f"Unsupported version: {version}")

    has_meas = bool(flags & _COL_FLAG_HAS_MEAS)
    measure_all = bool(flags & _COL_FLAG_MEAS_ALL)
    wide = bool(flags & _COL_FLAG_WIDE_IDX)
    shuffled = bool(flags & _COL_FLAG_SHUFFLED)

    idx_dtype = np.uint16 if wide else np.uint8
    idx_size = 2 if wide else 1

    # Opcode stream
    opcodes = data[off : off + n_gates]
    off += n_gates

    # Qubit index stream
    qubit_arr = np.frombuffer(data[off : off + n_qidx * idx_size], dtype=idx_dtype)
    off += n_qidx * idx_size

    # Param stream
    param_raw = data[off : off + n_params * 4]
    off += n_params * 4
    if shuffled and n_params > 0:
        param_raw = _byte_unshuffle(param_raw, 4, n_params)
    param_arr = np.frombuffer(param_raw, dtype=np.float32)

    # Reconstruct instructions
    qi = 0  # qubit index cursor
    pi = 0  # param cursor
    instructions: list[Instruction] = []
    for i in range(n_gates):
        opc = opcodes[i]
        gate_name = OPCODE_TO_GATE.get(opc)
        if gate_name is None:
            raise ValueError(f"Unknown opcode: 0x{opc:02X}")
        arity = _gate_arity(opc)
        n_p = _gate_n_params(opc)

        qubits = qubit_arr[qi : qi + arity].tolist()
        qi += arity

        params = param_arr[pi : pi + n_p].tolist() if n_p > 0 else []
        pi += n_p

        instructions.append(Instruction(gate_name, qubits, params))

    # Measurements
    measurements: list[Measurement] = []
    if has_meas:
        if measure_all:
            measurements = [Measurement(i, i) for i in range(n_qubits)]
        else:
            (n_meas,) = struct.unpack_from(">H", data, off)
            off += 2
            mq = np.frombuffer(data[off : off + n_meas * idx_size], dtype=idx_dtype)
            off += n_meas * idx_size
            mc = np.frombuffer(data[off : off + n_meas * idx_size], dtype=idx_dtype)
            off += n_meas * idx_size
            measurements = [
                Measurement(int(mq[i]), int(mc[i])) for i in range(n_meas)
            ]

    return CircuitIR(n_qubits, n_clbits, instructions, measurements)


# ---------------------------------------------------------------------------
# Delta-columnar encoder (v3) — columnar + delta on qubit indices
# ---------------------------------------------------------------------------

_MAGIC_DCOL = b"BD"
_VERSION_DCOL = 0x03


def encode_delta_columnar(ir: CircuitIR, shuffle_floats: bool = True) -> bytes:
    """Columnar encoder with delta + zigzag encoding on qubit indices.

    Qubit index deltas tend to be small in structured circuits (e.g., VQE
    ansätze operate on adjacent qubits). Zigzag encoding maps signed deltas
    to small unsigned values, further reducing byte entropy.
    """
    measure_all = (
        len(ir.measurements) == ir.n_qubits
        and all(m.qubit == m.clbit for m in ir.measurements)
        and len(set(m.qubit for m in ir.measurements)) == ir.n_qubits
    )
    has_meas = len(ir.measurements) > 0
    wide = ir.n_qubits > 255 or ir.n_clbits > 255

    flags = 0
    if has_meas:
        flags |= _COL_FLAG_HAS_MEAS
    if measure_all:
        flags |= _COL_FLAG_MEAS_ALL
    if wide:
        flags |= _COL_FLAG_WIDE_IDX
    if shuffle_floats:
        flags |= _COL_FLAG_SHUFFLED

    # Build streams
    opcodes = bytearray()
    qubit_indices: list[int] = []
    params: list[float] = []

    for inst in ir.instructions:
        opc = GATE_OPCODES.get(inst.gate)
        if opc is None:
            raise ValueError(f"Unsupported gate: {inst.gate!r}")
        opcodes.append(opc)
        qubit_indices.extend(inst.qubits)
        params.extend(inst.params)

    # Delta + zigzag encode qubit indices
    if qubit_indices:
        deltas = [qubit_indices[0]]
        for i in range(1, len(qubit_indices)):
            deltas.append(qubit_indices[i] - qubit_indices[i - 1])
        zz = _zigzag_encode(deltas)
    else:
        zz = []

    # Pack — use uint16 for zigzag values if wide or if deltas are large
    # For safety, always use uint16 since zigzag-encoded deltas can exceed 255
    zz_arr = np.array(zz, dtype=np.uint16) if zz else np.array([], dtype=np.uint16)
    qubit_bytes = zz_arr.tobytes()

    # Float params
    param_arr = np.array(params, dtype=np.float32)
    param_bytes = param_arr.tobytes()
    if shuffle_floats and len(params) > 0:
        param_bytes = _byte_shuffle(param_bytes, 4)

    header = struct.pack(
        ">2sBBHHIII",
        _MAGIC_DCOL,
        _VERSION_DCOL,
        flags,
        ir.n_qubits,
        ir.n_clbits,
        len(ir.instructions),
        len(qubit_indices),
        len(params),
    )

    parts = [header, bytes(opcodes), qubit_bytes, param_bytes]

    if has_meas and not measure_all:
        idx_dtype = np.uint16 if wide else np.uint8
        mq = np.array([m.qubit for m in ir.measurements], dtype=idx_dtype)
        mc = np.array([m.clbit for m in ir.measurements], dtype=idx_dtype)
        parts.append(struct.pack(">H", len(ir.measurements)))
        parts.append(mq.tobytes())
        parts.append(mc.tobytes())

    return b"".join(parts)


def decode_delta_columnar(data: bytes) -> CircuitIR:
    """Decode delta-columnar binary."""
    off = 0
    magic, version, flags, n_qubits, n_clbits, n_gates, n_qidx, n_params = (
        struct.unpack_from(">2sBBHHIII", data, off)
    )
    off += struct.calcsize(">2sBBHHIII")

    if magic != _MAGIC_DCOL:
        raise ValueError(f"Invalid magic: {magic!r}")

    has_meas = bool(flags & _COL_FLAG_HAS_MEAS)
    measure_all = bool(flags & _COL_FLAG_MEAS_ALL)
    wide = bool(flags & _COL_FLAG_WIDE_IDX)
    shuffled = bool(flags & _COL_FLAG_SHUFFLED)

    idx_dtype_meas = np.uint16 if wide else np.uint8
    idx_size_meas = 2 if wide else 1

    # Opcodes
    opcodes = data[off : off + n_gates]
    off += n_gates

    # Qubit indices (delta + zigzag encoded, always uint16)
    zz_arr = np.frombuffer(data[off : off + n_qidx * 2], dtype=np.uint16)
    off += n_qidx * 2
    zz = zz_arr.tolist()
    deltas = _zigzag_decode(zz)
    # Prefix sum to recover absolute indices
    qubit_indices: list[int] = []
    if deltas:
        qubit_indices.append(deltas[0])
        for i in range(1, len(deltas)):
            qubit_indices.append(qubit_indices[-1] + deltas[i])

    # Params
    param_raw = data[off : off + n_params * 4]
    off += n_params * 4
    if shuffled and n_params > 0:
        param_raw = _byte_unshuffle(param_raw, 4, n_params)
    param_arr = np.frombuffer(param_raw, dtype=np.float32)

    # Reconstruct
    qi = 0
    pi = 0
    instructions: list[Instruction] = []
    for i in range(n_gates):
        opc = opcodes[i]
        gate_name = OPCODE_TO_GATE.get(opc)
        if gate_name is None:
            raise ValueError(f"Unknown opcode: 0x{opc:02X}")
        arity = _gate_arity(opc)
        n_p = _gate_n_params(opc)
        qubits = qubit_indices[qi : qi + arity]
        qi += arity
        params = param_arr[pi : pi + n_p].tolist() if n_p > 0 else []
        pi += n_p
        instructions.append(Instruction(gate_name, qubits, params))

    measurements: list[Measurement] = []
    if has_meas:
        if measure_all:
            measurements = [Measurement(i, i) for i in range(n_qubits)]
        else:
            (n_meas,) = struct.unpack_from(">H", data, off)
            off += 2
            mq = np.frombuffer(
                data[off : off + n_meas * idx_size_meas], dtype=idx_dtype_meas
            )
            off += n_meas * idx_size_meas
            mc = np.frombuffer(
                data[off : off + n_meas * idx_size_meas], dtype=idx_dtype_meas
            )
            off += n_meas * idx_size_meas
            measurements = [
                Measurement(int(mq[i]), int(mc[i])) for i in range(n_meas)
            ]

    return CircuitIR(n_qubits, n_clbits, instructions, measurements)


# ---------------------------------------------------------------------------
# Per-stream encoder (v4) — the actually effective approach
# ---------------------------------------------------------------------------
# Key insight: random float32 parameters are ~47% of compressed output and
# are fundamentally incompressible.  The only way to shrink them is to use
# fewer bits per parameter.
#
# This encoder:
#   1. Separates opcodes / qubit indices / parameters into independent streams
#   2. Compresses each stream with the *caller's* chosen compressor
#   3. Supports uint16 quantization of params: [-pi, pi] → [0, 65535]
#      with max error ~4.8e-05 (enough for VQE/QAOA angles)
#   4. Packs the compressed sub-blobs into a self-describing container
#
# The container format (all big-endian):
#   magic        2B  "BP"
#   version      1B  0x04
#   flags        1B  (bit 0: has_meas, bit 1: meas_all, bit 2: wide_idx,
#                      bit 3: float32 params, bit 4: uint16 quantized params)
#   n_qubits     2B  uint16
#   n_clbits     2B  uint16
#   n_gates      4B  uint32
#   n_qidx       4B  uint32  (total qubit index count)
#   n_params     4B  uint32
#   len_opc      4B  compressed opcodes blob length
#   len_qub      4B  compressed qubit-idx blob length
#   len_par      4B  compressed params blob length
#   [opc_blob]   variable — compressed opcodes
#   [qub_blob]   variable — compressed qubit indices
#   [par_blob]   variable — compressed params
#   [meas_blob]  variable — only if partial measurements

import math as _math

_MAGIC_PS = b"BP"
_VERSION_PS = 0x04

_PS_FLAG_HAS_MEAS = 0x01
_PS_FLAG_MEAS_ALL = 0x02
_PS_FLAG_WIDE_IDX = 0x04
_PS_FLAG_F32 = 0x08
_PS_FLAG_QUANT16 = 0x10

# Quantization constants
_PARAM_MIN = -_math.pi
_PARAM_MAX = _math.pi
_QUANT_RANGE = _PARAM_MAX - _PARAM_MIN
_QUANT_SCALE = 65535.0 / _QUANT_RANGE


def _quantize_params(params: np.ndarray) -> np.ndarray:
    """Quantize float params to uint16 over [-pi, pi]."""
    return np.clip(
        np.round((params - _PARAM_MIN) * _QUANT_SCALE), 0, 65535
    ).astype(np.uint16)


def _dequantize_params(q: np.ndarray) -> np.ndarray:
    """Dequantize uint16 back to float32."""
    return (q.astype(np.float32) / _QUANT_SCALE + _PARAM_MIN).astype(np.float32)


CompressFn = type(lambda b: b)  # Callable[[bytes], bytes]


def encode_perstream(
    ir: CircuitIR,
    compress: CompressFn,
    quantize_params: bool = True,
) -> bytes:
    """Per-stream encoder with optional uint16 parameter quantization.

    Parameters
    ----------
    ir : CircuitIR
        Circuit to encode.
    compress : callable
        ``(bytes) -> bytes`` compression function (e.g. ``zstd.compress``).
    quantize_params : bool
        If True, quantize float params to uint16 over [-pi, pi].
        Saves ~40% on parameter bytes with max error ~4.8e-05.
    """
    measure_all = (
        len(ir.measurements) == ir.n_qubits
        and all(m.qubit == m.clbit for m in ir.measurements)
        and len(set(m.qubit for m in ir.measurements)) == ir.n_qubits
    )
    has_meas = len(ir.measurements) > 0
    wide = ir.n_qubits > 255 or ir.n_clbits > 255

    flags = 0
    if has_meas:
        flags |= _PS_FLAG_HAS_MEAS
    if measure_all:
        flags |= _PS_FLAG_MEAS_ALL
    if wide:
        flags |= _PS_FLAG_WIDE_IDX
    if quantize_params:
        flags |= _PS_FLAG_QUANT16
    else:
        flags |= _PS_FLAG_F32

    # Build raw streams
    opcodes = bytearray()
    qubit_indices: list[int] = []
    params: list[float] = []
    for inst in ir.instructions:
        opc = GATE_OPCODES.get(inst.gate)
        if opc is None:
            raise ValueError(f"Unsupported gate: {inst.gate!r}")
        opcodes.append(opc)
        qubit_indices.extend(inst.qubits)
        params.extend(inst.params)

    # Pack and compress each stream independently
    opc_raw = bytes(opcodes)
    idx_dtype = np.uint16 if wide else np.uint8
    qub_raw = np.array(qubit_indices, dtype=idx_dtype).tobytes()

    if quantize_params and params:
        par_np = np.array(params, dtype=np.float32)
        par_raw = _quantize_params(par_np).tobytes()
    elif params:
        par_raw = np.array(params, dtype=np.float32).tobytes()
    else:
        par_raw = b""

    opc_blob = compress(opc_raw) if opc_raw else b""
    qub_blob = compress(qub_raw) if qub_raw else b""
    par_blob = compress(par_raw) if par_raw else b""

    # Header
    header = struct.pack(
        ">2sBBHHIIIIII",
        _MAGIC_PS,
        _VERSION_PS,
        flags,
        ir.n_qubits,
        ir.n_clbits,
        len(ir.instructions),
        len(qubit_indices),
        len(params),
        len(opc_blob),
        len(qub_blob),
        len(par_blob),
    )

    parts = [header, opc_blob, qub_blob, par_blob]

    # Measurements (uncompressed, small)
    if has_meas and not measure_all:
        mq = np.array([m.qubit for m in ir.measurements], dtype=idx_dtype)
        mc = np.array([m.clbit for m in ir.measurements], dtype=idx_dtype)
        parts.append(struct.pack(">H", len(ir.measurements)))
        parts.append(mq.tobytes())
        parts.append(mc.tobytes())

    return b"".join(parts)


def decode_perstream(
    data: bytes,
    decompress: CompressFn,
) -> CircuitIR:
    """Decode per-stream binary."""
    off = 0
    hdr_fmt = ">2sBBHHIIIIII"
    hdr_size = struct.calcsize(hdr_fmt)
    (
        magic, version, flags,
        n_qubits, n_clbits, n_gates, n_qidx, n_params,
        len_opc, len_qub, len_par,
    ) = struct.unpack_from(hdr_fmt, data, off)
    off += hdr_size

    if magic != _MAGIC_PS:
        raise ValueError(f"Invalid magic: {magic!r}")

    has_meas = bool(flags & _PS_FLAG_HAS_MEAS)
    measure_all = bool(flags & _PS_FLAG_MEAS_ALL)
    wide = bool(flags & _PS_FLAG_WIDE_IDX)
    is_quant16 = bool(flags & _PS_FLAG_QUANT16)

    idx_dtype = np.uint16 if wide else np.uint8
    idx_size = 2 if wide else 1

    # Decompress streams
    opc_raw = decompress(data[off : off + len_opc]) if len_opc else b""
    off += len_opc
    qub_raw = decompress(data[off : off + len_qub]) if len_qub else b""
    off += len_qub
    par_raw = decompress(data[off : off + len_par]) if len_par else b""
    off += len_par

    opcodes = opc_raw
    qubit_arr = np.frombuffer(qub_raw, dtype=idx_dtype) if qub_raw else np.array([], dtype=idx_dtype)

    if is_quant16 and par_raw:
        q16 = np.frombuffer(par_raw, dtype=np.uint16)
        param_arr = _dequantize_params(q16)
    elif par_raw:
        param_arr = np.frombuffer(par_raw, dtype=np.float32)
    else:
        param_arr = np.array([], dtype=np.float32)

    # Reconstruct instructions
    qi = 0
    pi = 0
    instructions: list[Instruction] = []
    for i in range(n_gates):
        opc = opcodes[i]
        gate_name = OPCODE_TO_GATE.get(opc)
        if gate_name is None:
            raise ValueError(f"Unknown opcode: 0x{opc:02X}")
        arity = _gate_arity(opc)
        n_p = _gate_n_params(opc)
        qubits = qubit_arr[qi : qi + arity].tolist()
        qi += arity
        params = param_arr[pi : pi + n_p].tolist() if n_p > 0 else []
        pi += n_p
        instructions.append(Instruction(gate_name, qubits, params))

    # Measurements
    measurements: list[Measurement] = []
    if has_meas:
        if measure_all:
            measurements = [Measurement(i, i) for i in range(n_qubits)]
        else:
            (n_meas,) = struct.unpack_from(">H", data, off)
            off += 2
            mq = np.frombuffer(data[off : off + n_meas * idx_size], dtype=idx_dtype)
            off += n_meas * idx_size
            mc = np.frombuffer(data[off : off + n_meas * idx_size], dtype=idx_dtype)
            off += n_meas * idx_size
            measurements = [
                Measurement(int(mq[i]), int(mc[i])) for i in range(n_meas)
            ]

    return CircuitIR(n_qubits, n_clbits, instructions, measurements)
