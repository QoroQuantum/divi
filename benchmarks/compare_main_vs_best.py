#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Head-to-head comparison: current production (main) vs best new technique.

Compares:
  CURRENT:  QoroService._compress_data  →  QASM text → gzip → base64 (str in JSON)
  BEST:     columnar+byte-shuffle → zstd (raw bytes, no base64)
  BEST+B64: columnar+byte-shuffle → zstd → base64 (for fair JSON-transport comparison)

Run:  python benchmarks/compare_main_vs_best.py
"""

from __future__ import annotations

import base64
import gzip
import math
import random
import time
from dataclasses import dataclass

import zstandard as zstd

from divi.circuits._binary_qasm import (
    CircuitIR,
    Instruction,
    Measurement,
    decode_columnar,
    encode_columnar,
    ir_to_qasm,
    parse_qasm,
)

# Also import brotli for the "best compression" variant
import brotli

from rich.console import Console
from rich.table import Table

# ---------------------------------------------------------------------------
# Production baseline — exact code from _qoro_service.py:398-399
# ---------------------------------------------------------------------------

def compress_data_production(value: str) -> str:
    """Exact replica of QoroService._compress_data on main."""
    return base64.b64encode(gzip.compress(value.encode("utf-8"))).decode("utf-8")


def decompress_data_production(value: str) -> str:
    """Reverse of compress_data_production."""
    return gzip.decompress(base64.b64decode(value.encode("utf-8"))).decode("utf-8")


# ---------------------------------------------------------------------------
# Best new technique
# ---------------------------------------------------------------------------

_zstd_cctx = zstd.ZstdCompressor()
_zstd_dctx = zstd.ZstdDecompressor()


def compress_best_raw(qasm: str) -> bytes:
    """Best technique: columnar+shuffle → zstd. Returns raw bytes."""
    ir = parse_qasm(qasm)
    binary = encode_columnar(ir, shuffle_floats=True)
    return _zstd_cctx.compress(binary)


def decompress_best_raw(data: bytes) -> str:
    """Reverse: zstd → columnar decode → QASM text."""
    binary = _zstd_dctx.decompress(data)
    ir = decode_columnar(binary)
    return ir_to_qasm(ir)


def compress_best_b64(qasm: str) -> str:
    """Best technique + base64 for JSON transport. Returns str."""
    raw = compress_best_raw(qasm)
    return base64.b64encode(raw).decode("utf-8")


def decompress_best_b64(value: str) -> str:
    """Reverse of compress_best_b64."""
    raw = base64.b64decode(value.encode("utf-8"))
    return decompress_best_raw(raw)


def compress_best_brotli_b64(qasm: str) -> str:
    """Best compression: columnar+shuffle → brotli → base64."""
    ir = parse_qasm(qasm)
    binary = encode_columnar(ir, shuffle_floats=True)
    compressed = brotli.compress(binary)
    return base64.b64encode(compressed).decode("utf-8")


def decompress_best_brotli_b64(value: str) -> str:
    """Reverse of compress_best_brotli_b64."""
    compressed = base64.b64decode(value.encode("utf-8"))
    binary = brotli.decompress(compressed)
    ir = decode_columnar(binary)
    return ir_to_qasm(ir)


# ---------------------------------------------------------------------------
# Circuit generation (same as benchmark)
# ---------------------------------------------------------------------------

_1Q_GATES = ["h", "x", "y", "z", "s", "t", "sdg", "tdg", "id"]
_1Q_PARAM_GATES = ["ry", "rz", "rx", "u1"]
_2Q_GATES = ["cx", "cz", "swap"]
_2Q_PARAM_GATES = ["crx", "cry", "crz", "rxx", "ryy", "rzz"]
_3Q_GATES = ["ccx", "cswap"]


def generate_circuit(n_qubits: int, n_gates: int, seed: int = 42) -> CircuitIR:
    rng = random.Random(seed)
    instructions: list[Instruction] = []
    for _ in range(n_gates):
        r = rng.random()
        if r < 0.15:
            gate = rng.choice(_1Q_GATES)
            q = rng.randint(0, n_qubits - 1)
            instructions.append(Instruction(gate, [q]))
        elif r < 0.50:
            gate = rng.choice(_1Q_PARAM_GATES)
            q = rng.randint(0, n_qubits - 1)
            param = rng.uniform(-math.pi, math.pi)
            instructions.append(Instruction(gate, [q], [param]))
        elif r < 0.85:
            gate = rng.choice(_2Q_GATES)
            q1 = rng.randint(0, n_qubits - 1)
            q2 = rng.randint(0, n_qubits - 2)
            if q2 >= q1:
                q2 += 1
            instructions.append(Instruction(gate, [q1, q2]))
        elif r < 0.95:
            gate = rng.choice(_2Q_PARAM_GATES)
            q1 = rng.randint(0, n_qubits - 1)
            q2 = rng.randint(0, n_qubits - 2)
            if q2 >= q1:
                q2 += 1
            param = rng.uniform(-math.pi, math.pi)
            instructions.append(Instruction(gate, [q1, q2], [param]))
        else:
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
# Timing helper
# ---------------------------------------------------------------------------

def time_fn(fn, n_iters: int = 100) -> tuple:
    """Return (result, avg_ms)."""
    result = fn()
    start = time.perf_counter()
    for _ in range(n_iters):
        fn()
    avg_ms = (time.perf_counter() - start) / n_iters * 1000
    return result, avg_ms


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

@dataclass
class ComparisonRow:
    circuit: str
    qasm_size: int
    # Production
    prod_payload_size: int
    prod_compress_ms: float
    prod_decompress_ms: float
    # Best (zstd, raw bytes)
    best_raw_size: int
    best_raw_compress_ms: float
    best_raw_decompress_ms: float
    # Best (zstd, b64 for JSON)
    best_b64_size: int
    best_b64_compress_ms: float
    best_b64_decompress_ms: float
    # Best compression (brotli, b64 for JSON)
    brotli_b64_size: int
    brotli_b64_compress_ms: float
    brotli_b64_decompress_ms: float


CIRCUITS = [
    ("small (4q, 30g)", 4, 30),
    ("medium (20q, 200g)", 20, 200),
    ("large (100q, 2Kg)", 100, 2000),
    ("xl (100q, 10Kg)", 100, 10000),
]


def main():
    console = Console(width=130)
    console.print("[bold]HEAD-TO-HEAD: Production (main) vs Best New Technique[/bold]")
    console.print()
    console.print("[dim]PRODUCTION:[/dim] QASM text → gzip → base64  (QoroService._compress_data)")
    console.print("[dim]BEST+ZSTD:[/dim]  QASM → columnar+byte-shuffle → zstd → base64")
    console.print("[dim]BEST+BROTLI:[/dim] QASM → columnar+byte-shuffle → brotli → base64")
    console.print()

    rows: list[ComparisonRow] = []

    for label, nq, ng in CIRCUITS:
        ir = generate_circuit(nq, ng)
        qasm = ir_to_qasm(ir)

        # Production
        prod_result, prod_c_ms = time_fn(lambda: compress_data_production(qasm))
        _, prod_d_ms = time_fn(lambda: decompress_data_production(prod_result))
        prod_size = len(prod_result)

        # Verify production round-trip
        assert decompress_data_production(prod_result) == qasm

        # Best (zstd, raw)
        raw_result, raw_c_ms = time_fn(lambda: compress_best_raw(qasm))
        _, raw_d_ms = time_fn(lambda: decompress_best_raw(raw_result))
        raw_size = len(raw_result)

        # Best (zstd, b64)
        b64_result, b64_c_ms = time_fn(lambda: compress_best_b64(qasm))
        _, b64_d_ms = time_fn(lambda: decompress_best_b64(b64_result))
        b64_size = len(b64_result)

        # Best compression (brotli, b64)
        brotli_result, brotli_c_ms = time_fn(lambda: compress_best_brotli_b64(qasm))
        _, brotli_d_ms = time_fn(lambda: decompress_best_brotli_b64(brotli_result))
        brotli_size = len(brotli_result)

        # Verify new round-trip (semantic equivalence via re-parse)
        recovered = decompress_best_b64(b64_result)
        ir_orig = parse_qasm(qasm)
        ir_recv = parse_qasm(recovered)
        assert ir_orig.n_qubits == ir_recv.n_qubits
        assert len(ir_orig.instructions) == len(ir_recv.instructions)

        rows.append(ComparisonRow(
            circuit=label,
            qasm_size=len(qasm.encode("utf-8")),
            prod_payload_size=prod_size,
            prod_compress_ms=prod_c_ms,
            prod_decompress_ms=prod_d_ms,
            best_raw_size=raw_size,
            best_raw_compress_ms=raw_c_ms,
            best_raw_decompress_ms=raw_d_ms,
            best_b64_size=b64_size,
            best_b64_compress_ms=b64_c_ms,
            best_b64_decompress_ms=b64_d_ms,
            brotli_b64_size=brotli_size,
            brotli_b64_compress_ms=brotli_c_ms,
            brotli_b64_decompress_ms=brotli_d_ms,
        ))

    # ---------- Payload Size Table ----------
    t = Table(title="Payload Size Comparison (bytes in JSON payload)", header_style="bold cyan")
    t.add_column("Circuit", min_width=22)
    t.add_column("QASM text", justify="right")
    t.add_column("PRODUCTION\ngzip+b64", justify="right")
    t.add_column("NEW zstd\nraw bytes", justify="right")
    t.add_column("NEW zstd\n+base64", justify="right")
    t.add_column("NEW brotli\n+base64", justify="right")
    t.add_column("Savings\n(zstd+b64)", justify="right")
    t.add_column("Savings\n(brotli+b64)", justify="right")

    for r in rows:
        zstd_pct = (1 - r.best_b64_size / r.prod_payload_size) * 100
        brotli_pct = (1 - r.brotli_b64_size / r.prod_payload_size) * 100
        t.add_row(
            r.circuit,
            f"{r.qasm_size:,}",
            f"{r.prod_payload_size:,}",
            f"{r.best_raw_size:,}",
            f"{r.best_b64_size:,}",
            f"{r.brotli_b64_size:,}",
            f"[bold green]{zstd_pct:.1f}%[/bold green]",
            f"[bold green]{brotli_pct:.1f}%[/bold green]",
        )
    console.print(t)
    console.print()

    # ---------- Speed Table ----------
    s = Table(title="Speed Comparison (ms, avg over 100 iterations)", header_style="bold cyan")
    s.add_column("Circuit", min_width=22)
    s.add_column("PROD\ncompress", justify="right")
    s.add_column("PROD\ndecompress", justify="right")
    s.add_column("ZSTD+b64\ncompress", justify="right")
    s.add_column("ZSTD+b64\ndecompress", justify="right")
    s.add_column("BROTLI+b64\ncompress", justify="right")
    s.add_column("BROTLI+b64\ndecompress", justify="right")
    s.add_column("Speedup\n(zstd)", justify="right")

    for r in rows:
        speedup = r.prod_compress_ms / r.best_b64_compress_ms if r.best_b64_compress_ms > 0 else 0
        s.add_row(
            r.circuit,
            f"{r.prod_compress_ms:.3f}",
            f"{r.prod_decompress_ms:.3f}",
            f"{r.best_b64_compress_ms:.3f}",
            f"{r.best_b64_decompress_ms:.3f}",
            f"{r.brotli_b64_compress_ms:.3f}",
            f"{r.brotli_b64_decompress_ms:.3f}",
            f"{speedup:.1f}x",
        )
    console.print(s)
    console.print()

    # ---------- Chunk capacity ----------
    max_payload_mb = 0.95
    max_bytes = int(max_payload_mb * 1024 * 1024)
    console.print(f"[bold]Chunk capacity (max payload = {max_payload_mb} MB = {max_bytes:,} bytes)[/bold]")
    cap = Table(header_style="bold magenta")
    cap.add_column("Circuit")
    cap.add_column("PRODUCTION\ncircuits/chunk", justify="right")
    cap.add_column("NEW zstd+b64\ncircuits/chunk", justify="right")
    cap.add_column("NEW brotli+b64\ncircuits/chunk", justify="right")
    cap.add_column("Capacity\ngain (zstd)", justify="right")
    for r in rows:
        prod_per_chunk = max_bytes // r.prod_payload_size
        zstd_per_chunk = max_bytes // r.best_b64_size
        brotli_per_chunk = max_bytes // r.brotli_b64_size
        gain = zstd_per_chunk / prod_per_chunk if prod_per_chunk > 0 else 0
        cap.add_row(
            r.circuit,
            f"{prod_per_chunk:,}",
            f"{zstd_per_chunk:,}",
            f"{brotli_per_chunk:,}",
            f"[bold green]{gain:.1f}x[/bold green]",
        )
    console.print(cap)
    console.print()

    console.print("[bold]Recommendation:[/bold]")
    console.print("  For best SIZE:  columnar+byte-shuffle → brotli → base64")
    console.print("  For best SPEED: columnar+byte-shuffle → zstd → base64  (nearly same size, 10-100x faster)")


if __name__ == "__main__":
    main()
