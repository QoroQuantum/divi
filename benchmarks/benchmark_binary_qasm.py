#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark binary QASM encoding vs compressed QASM text.

Compares multiple serialization formats × compression algorithms across
circuit sizes. Run standalone:

    python benchmarks/benchmark_binary_qasm.py

Requires: zstandard, lz4, brotli, msgpack, cbor2, construct, rich
    pip install zstandard lz4 brotli msgpack cbor2 construct rich
"""

from __future__ import annotations

import base64
import bz2
import gzip
import io
import lzma
import math
import random
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass

import brotli
import cbor2
import construct as cs
import lz4.frame
import msgpack
import zstandard as zstd
from rich.console import Console
from rich.table import Table

from divi.circuits._binary_qasm import (
    GATE_OPCODES,
    CircuitIR,
    Instruction,
    Measurement,
    _gate_arity,
    _gate_n_params,
    encode_columnar,
    encode_delta_columnar,
    encode_ir,
    ir_to_qasm,
    parse_qasm,
)

# ---------------------------------------------------------------------------
# Circuit generation
# ---------------------------------------------------------------------------

# Gate pools with weights reflecting typical VQE/QAOA circuits
_1Q_GATES = ["h", "x", "y", "z", "s", "t", "sdg", "tdg", "id"]
_1Q_PARAM_GATES = ["ry", "rz", "rx", "u1"]
_2Q_GATES = ["cx", "cz", "swap"]
_2Q_PARAM_GATES = ["crx", "cry", "crz", "rxx", "ryy", "rzz"]
_3Q_GATES = ["ccx", "cswap"]


def generate_circuit(n_qubits: int, n_gates: int, seed: int = 42) -> CircuitIR:
    """Generate a realistic random circuit with the given parameters."""
    rng = random.Random(seed)
    instructions: list[Instruction] = []

    for _ in range(n_gates):
        r = rng.random()
        if r < 0.15:
            # 1-qubit, no param
            gate = rng.choice(_1Q_GATES)
            q = rng.randint(0, n_qubits - 1)
            instructions.append(Instruction(gate, [q]))
        elif r < 0.50:
            # 1-qubit, 1 param (most common in VQE)
            gate = rng.choice(_1Q_PARAM_GATES)
            q = rng.randint(0, n_qubits - 1)
            param = rng.uniform(-math.pi, math.pi)
            instructions.append(Instruction(gate, [q], [param]))
        elif r < 0.85:
            # 2-qubit, no param (CNOT-heavy)
            gate = rng.choice(_2Q_GATES)
            q1 = rng.randint(0, n_qubits - 1)
            q2 = rng.randint(0, n_qubits - 2)
            if q2 >= q1:
                q2 += 1
            instructions.append(Instruction(gate, [q1, q2]))
        elif r < 0.95:
            # 2-qubit, 1 param
            gate = rng.choice(_2Q_PARAM_GATES)
            q1 = rng.randint(0, n_qubits - 1)
            q2 = rng.randint(0, n_qubits - 2)
            if q2 >= q1:
                q2 += 1
            param = rng.uniform(-math.pi, math.pi)
            instructions.append(Instruction(gate, [q1, q2], [param]))
        else:
            # 3-qubit gate
            if n_qubits >= 3:
                gate = rng.choice(_3Q_GATES)
                qs = rng.sample(range(n_qubits), 3)
                instructions.append(Instruction(gate, qs))
            else:
                gate = rng.choice(_1Q_GATES)
                q = rng.randint(0, n_qubits - 1)
                instructions.append(Instruction(gate, [q]))

    measurements = [Measurement(i, i) for i in range(n_qubits)]
    return CircuitIR(n_qubits, n_qubits, instructions, measurements)


# ---------------------------------------------------------------------------
# Serialization formats
# ---------------------------------------------------------------------------


def _ir_to_dict(ir: CircuitIR) -> dict:
    """Convert IR to a plain dict for msgpack/cbor serialization."""
    gates = []
    for inst in ir.instructions:
        opcode = GATE_OPCODES[inst.gate]
        entry = [opcode, inst.qubits]
        if inst.params:
            entry.append(inst.params)
        gates.append(entry)

    meas = [[m.qubit, m.clbit] for m in ir.measurements]
    return {
        "nq": ir.n_qubits,
        "nc": ir.n_clbits,
        "g": gates,
        "m": meas,
    }


def _dict_to_ir(d: dict) -> CircuitIR:
    """Reconstruct IR from a plain dict."""
    from divi.circuits._binary_qasm import OPCODE_TO_GATE

    instructions = []
    for entry in d["g"]:
        opcode = entry[0]
        qubits = list(entry[1])
        params = list(entry[2]) if len(entry) > 2 else []
        gate = OPCODE_TO_GATE[opcode]
        instructions.append(Instruction(gate, qubits, params))

    measurements = [Measurement(m[0], m[1]) for m in d["m"]]
    return CircuitIR(d["nq"], d["nc"], instructions, measurements)


# --- construct-based encoder ---

def _build_construct_format(n_qubits: int) -> cs.Construct:
    """Build a construct schema for the given qubit count."""
    idx_type = cs.Int16ub if n_qubits > 255 else cs.Int8ub

    instruction = cs.Struct(
        "opcode" / cs.Int8ub,
        "qubits" / cs.Array(lambda ctx: _gate_arity(ctx.opcode), idx_type),
        "params" / cs.Array(lambda ctx: _gate_n_params(ctx.opcode), cs.Float32b),
    )

    measurement = cs.Struct(
        "qubit" / idx_type,
        "clbit" / idx_type,
    )

    return cs.Struct(
        "magic" / cs.Const(b"BC"),
        "n_qubits" / cs.Int16ub,
        "n_clbits" / cs.Int16ub,
        "n_gates" / cs.Int32ub,
        "n_meas" / cs.Int16ub,
        "instructions" / cs.Array(cs.this.n_gates, instruction),
        "measurements" / cs.Array(cs.this.n_meas, measurement),
    )


def encode_construct(ir: CircuitIR) -> bytes:
    """Encode using the construct library."""
    fmt = _build_construct_format(ir.n_qubits)
    data = {
        "magic": b"BC",
        "n_qubits": ir.n_qubits,
        "n_clbits": ir.n_clbits,
        "n_gates": len(ir.instructions),
        "n_meas": len(ir.measurements),
        "instructions": [
            {
                "opcode": GATE_OPCODES[inst.gate],
                "qubits": inst.qubits,
                "params": inst.params,
            }
            for inst in ir.instructions
        ],
        "measurements": [
            {"qubit": m.qubit, "clbit": m.clbit}
            for m in ir.measurements
        ],
    }
    return fmt.build(data)


def decode_construct(data: bytes, n_qubits_hint: int) -> CircuitIR:
    """Decode using the construct library."""
    from divi.circuits._binary_qasm import OPCODE_TO_GATE

    fmt = _build_construct_format(n_qubits_hint)
    parsed = fmt.parse(data)
    instructions = [
        Instruction(
            OPCODE_TO_GATE[inst.opcode],
            list(inst.qubits),
            list(inst.params),
        )
        for inst in parsed.instructions
    ]
    measurements = [
        Measurement(m.qubit, m.clbit)
        for m in parsed.measurements
    ]
    return CircuitIR(parsed.n_qubits, parsed.n_clbits, instructions, measurements)


# ---------------------------------------------------------------------------
# Compression engines
# ---------------------------------------------------------------------------

@dataclass
class Compressor:
    name: str
    compress: Callable[[bytes], bytes]
    decompress: Callable[[bytes], bytes]


def _zstd_compress(data: bytes) -> bytes:
    cctx = zstd.ZstdCompressor()
    return cctx.compress(data)


def _zstd_decompress(data: bytes) -> bytes:
    dctx = zstd.ZstdDecompressor()
    return dctx.decompress(data)


COMPRESSORS = [
    Compressor("gzip", gzip.compress, gzip.decompress),
    Compressor("zlib", lambda d: __import__("zlib").compress(d), lambda d: __import__("zlib").decompress(d)),
    Compressor("bz2", bz2.compress, bz2.decompress),
    Compressor("lzma", lzma.compress, lzma.decompress),
    Compressor("zstd", _zstd_compress, _zstd_decompress),
    Compressor("lz4", lz4.frame.compress, lz4.frame.decompress),
    Compressor("brotli", brotli.compress, brotli.decompress),
]

# Focused set for v2 benchmark — only the top performers
COMPRESSORS_FOCUSED = [
    Compressor("zstd", _zstd_compress, _zstd_decompress),
    Compressor("brotli", brotli.compress, brotli.decompress),
    Compressor("zlib", lambda d: __import__("zlib").compress(d), lambda d: __import__("zlib").decompress(d)),
    Compressor("lz4", lz4.frame.compress, lz4.frame.decompress),
]


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    format_name: str
    compressor_name: str
    circuit_label: str
    raw_size: int
    compressed_size: int
    compress_ms: float
    decompress_ms: float
    baseline_compressed_size: int  # gzip+base64 on QASM text


def _time_fn(fn: Callable[[], bytes], n_iters: int = 50) -> tuple[bytes, float]:
    """Run *fn* n_iters times and return (result, avg_ms)."""
    result = fn()
    start = time.perf_counter()
    for _ in range(n_iters):
        fn()
    elapsed = (time.perf_counter() - start) / n_iters * 1000
    return result, elapsed


def benchmark_circuit(
    label: str,
    ir: CircuitIR,
    n_iters: int = 50,
) -> list[BenchResult]:
    """Run all format × compressor combinations for one circuit."""
    qasm_text = ir_to_qasm(ir)
    qasm_bytes = qasm_text.encode("utf-8")

    # Baseline: current QoroService approach (gzip + base64)
    baseline = base64.b64encode(gzip.compress(qasm_bytes))
    baseline_size = len(baseline)

    # Prepare serialized formats
    formats: list[tuple[str, bytes, Callable[[bytes], None]]] = []

    # 1. Raw QASM text
    formats.append(("qasm_text", qasm_bytes, lambda d: d))

    # 2. struct-based binary (v1 — interleaved)
    struct_binary = encode_ir(ir)
    formats.append(("struct_binary", struct_binary, lambda d: d))

    # 3. construct-based binary
    construct_binary = encode_construct(ir)
    formats.append(("construct_binary", construct_binary, lambda d: d))

    # 4. columnar binary (v2 — separated streams, byte-shuffled floats)
    columnar_binary = encode_columnar(ir, shuffle_floats=True)
    formats.append(("columnar_shuf", columnar_binary, lambda d: d))

    # 5. columnar binary without byte shuffle (to measure shuffle impact)
    columnar_noshuf = encode_columnar(ir, shuffle_floats=False)
    formats.append(("columnar_plain", columnar_noshuf, lambda d: d))

    # 6. delta-columnar binary (v3 — delta+zigzag qubit indices + shuffled)
    delta_col = encode_delta_columnar(ir, shuffle_floats=True)
    formats.append(("delta_col_shuf", delta_col, lambda d: d))

    # 7. msgpack
    ir_dict = _ir_to_dict(ir)
    msgpack_data = msgpack.packb(ir_dict, use_bin_type=True)
    formats.append(("msgpack", msgpack_data, lambda d: d))

    # 8. cbor2
    cbor_data = cbor2.dumps(ir_dict)
    formats.append(("cbor", cbor_data, lambda d: d))

    results: list[BenchResult] = []

    # Add baseline as a special entry
    _, compress_ms = _time_fn(
        lambda: base64.b64encode(gzip.compress(qasm_bytes)), n_iters
    )
    _, decompress_ms = _time_fn(
        lambda: gzip.decompress(base64.b64decode(baseline)), n_iters
    )
    results.append(
        BenchResult(
            format_name="BASELINE (gzip+b64)",
            compressor_name="gzip+base64",
            circuit_label=label,
            raw_size=len(qasm_bytes),
            compressed_size=baseline_size,
            compress_ms=compress_ms,
            decompress_ms=decompress_ms,
            baseline_compressed_size=baseline_size,
        )
    )

    for fmt_name, raw_data, _verify in formats:
        for comp in COMPRESSORS_FOCUSED:
            compressed, compress_ms = _time_fn(
                lambda _d=raw_data, _c=comp: _c.compress(_d), n_iters
            )
            _, decompress_ms = _time_fn(
                lambda _d=compressed, _c=comp: _c.decompress(_d), n_iters
            )
            results.append(
                BenchResult(
                    format_name=fmt_name,
                    compressor_name=comp.name,
                    circuit_label=label,
                    raw_size=len(raw_data),
                    compressed_size=len(compressed),
                    compress_ms=compress_ms,
                    decompress_ms=decompress_ms,
                    baseline_compressed_size=baseline_size,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------------------------


def print_results(all_results: list[BenchResult]) -> None:
    console = Console()

    # Group by circuit label
    labels = sorted(set(r.circuit_label for r in all_results))

    for label in labels:
        circuit_results = [r for r in all_results if r.circuit_label == label]
        circuit_results.sort(key=lambda r: r.compressed_size)

        table = Table(
            title=f"Circuit: {label}",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Format", style="white", min_width=18)
        table.add_column("Compressor", style="white", min_width=10)
        table.add_column("Raw (B)", justify="right")
        table.add_column("Compressed (B)", justify="right")
        table.add_column("vs Baseline", justify="right")
        table.add_column("Compress (ms)", justify="right")
        table.add_column("Decompress (ms)", justify="right")

        for r in circuit_results:
            ratio = r.compressed_size / r.baseline_compressed_size
            pct = (1 - ratio) * 100
            vs_baseline = f"{pct:+.1f}%" if r.format_name != "BASELINE (gzip+b64)" else "-"
            style = ""
            if r.format_name == "BASELINE (gzip+b64)":
                style = "bold yellow"
            elif pct > 20:
                style = "bold green"
            elif pct > 0:
                style = "green"
            elif pct < -10:
                style = "red"

            table.add_row(
                r.format_name,
                r.compressor_name,
                str(r.raw_size),
                str(r.compressed_size),
                vs_baseline,
                f"{r.compress_ms:.3f}",
                f"{r.decompress_ms:.3f}",
                style=style,
            )

        console.print(table)
        console.print()

    # Summary: best for each circuit
    console.print("[bold]Summary: Best compression for each circuit size[/bold]")
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Circuit")
    summary_table.add_column("Best Format")
    summary_table.add_column("Best Compressor")
    summary_table.add_column("Compressed (B)")
    summary_table.add_column("vs Baseline")
    summary_table.add_column("Baseline (B)")

    for label in labels:
        circuit_results = [
            r for r in all_results
            if r.circuit_label == label and r.format_name != "BASELINE (gzip+b64)"
        ]
        best = min(circuit_results, key=lambda r: r.compressed_size)
        ratio = best.compressed_size / best.baseline_compressed_size
        pct = (1 - ratio) * 100
        summary_table.add_row(
            label,
            best.format_name,
            best.compressor_name,
            str(best.compressed_size),
            f"{pct:+.1f}%",
            str(best.baseline_compressed_size),
        )

    console.print(summary_table)

    # Also print fastest compression for each circuit
    console.print()
    console.print("[bold]Summary: Fastest compression (within 10% of best size)[/bold]")
    speed_table = Table(show_header=True, header_style="bold magenta")
    speed_table.add_column("Circuit")
    speed_table.add_column("Format")
    speed_table.add_column("Compressor")
    speed_table.add_column("Compressed (B)")
    speed_table.add_column("Compress (ms)")
    speed_table.add_column("Decompress (ms)")

    for label in labels:
        circuit_results = [
            r for r in all_results
            if r.circuit_label == label and r.format_name != "BASELINE (gzip+b64)"
        ]
        best_size = min(r.compressed_size for r in circuit_results)
        # Within 10% of best size
        near_best = [
            r for r in circuit_results
            if r.compressed_size <= best_size * 1.10
        ]
        fastest = min(near_best, key=lambda r: r.compress_ms)
        speed_table.add_row(
            label,
            fastest.format_name,
            fastest.compressor_name,
            str(fastest.compressed_size),
            f"{fastest.compress_ms:.3f}",
            f"{fastest.decompress_ms:.3f}",
        )

    console.print(speed_table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CIRCUIT_CONFIGS = [
    ("small_4q_30g", 4, 30),
    ("medium_20q_200g", 20, 200),
    ("large_100q_2000g", 100, 2000),
    ("xl_100q_10000g", 100, 10000),
]


def main():
    console = Console()
    console.print("[bold]Binary QASM Benchmark[/bold]")
    console.print(f"Formats: qasm_text, struct_binary, construct_binary, columnar_shuf, columnar_plain, delta_col_shuf, msgpack, cbor")
    console.print(f"Compressors: {', '.join(c.name for c in COMPRESSORS_FOCUSED)}")
    console.print()

    all_results: list[BenchResult] = []

    for label, n_qubits, n_gates in CIRCUIT_CONFIGS:
        console.print(f"[cyan]Generating circuit: {label} ({n_qubits} qubits, {n_gates} gates)...[/cyan]")
        ir = generate_circuit(n_qubits, n_gates)
        qasm_text = ir_to_qasm(ir)
        console.print(f"  QASM text size: {len(qasm_text.encode('utf-8')):,} bytes")
        console.print(f"  struct binary size: {len(encode_ir(ir)):,} bytes")
        console.print(f"  columnar+shuf size: {len(encode_columnar(ir)):,} bytes")
        console.print(f"  delta_col+shuf size: {len(encode_delta_columnar(ir)):,} bytes")

        results = benchmark_circuit(label, ir, n_iters=50)
        all_results.extend(results)

    console.print()
    print_results(all_results)


if __name__ == "__main__":
    main()
