# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Reproducible evaluation harness for divi-ai.

Runs a fixed set of queries against the retrieval + LLM pipeline and
saves results to JSON for before/after comparison across iterations.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ._chat import build_prompt
from ._indexer import load_search_stack
from ._retriever import enrich_chunks, retrieve

# ---------------------------------------------------------------------------
# Eval queries
# ---------------------------------------------------------------------------

EVAL_QUERIES: list[dict] = [
    {
        "id": "vqe_basic",
        "query": "How do I create and run a VQE problem?",
        "tags": ["code_gen"],
    },
    {
        "id": "optimizers",
        "query": "What are the available optimizers in divi?",
        "tags": ["api_lookup"],
    },
    {
        "id": "qaoa_qubo",
        "query": "Show me how to set up QAOA for a QUBO problem",
        "tags": ["code_gen"],
    },
    {
        "id": "checkpointing",
        "query": "How do I use checkpointing?",
        "tags": ["code_gen"],
    },
    {
        "id": "ensemble",
        "query": "What is a ProgramEnsemble?",
        "tags": ["api_lookup"],
    },
    {
        "id": "trotter",
        "query": "How to do time evolution with Trotter?",
        "tags": ["code_gen"],
    },
    {
        "id": "backends",
        "query": "What backends does divi support?",
        "tags": ["api_lookup"],
    },
    {
        "id": "qaoa_graph",
        "query": "Write code to run QAOA on a graph partitioning problem",
        "tags": ["code_gen"],
    },
    {
        "id": "zne",
        "query": "How do I configure ZNE error mitigation?",
        "tags": ["code_gen"],
    },
    {
        "id": "sim_vs_service",
        "query": "What is the difference between ParallelSimulator and QoroService?",
        "tags": ["api_lookup"],
    },
    {
        "id": "vqe_sweep",
        "query": "How to run a hyperparameter sweep with VQE?",
        "tags": ["code_gen"],
    },
    {
        "id": "qdrift",
        "query": "Show me how to use QDrift for time evolution",
        "tags": ["code_gen"],
    },
]

# Multi-turn conversations (list of user messages, run with accumulated history)
MULTI_TURN_SEQUENCES: list[dict] = [
    {
        "id": "vqe_followup",
        "turns": [
            "How do I run VQE in divi?",
            "Can you show me an example with a custom ansatz?",
            "How do I extract the optimal parameters from the result?",
        ],
        "tags": ["multi_turn", "code_gen"],
    },
    {
        "id": "qaoa_workflow",
        "turns": [
            "What is QAOA in divi?",
            "How do I define a QUBO for it?",
            "Show me the full code to solve it",
        ],
        "tags": ["multi_turn", "code_gen"],
    },
]

# ---------------------------------------------------------------------------
# Result storage
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent / "_eval_results"


def _results_path(label: str) -> Path:
    """Return the path for a result file with the given label."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    return RESULTS_DIR / f"{label}_{ts}.json"


def _find_latest_result(label: str) -> Path | None:
    """Find the most recent result file for a given label."""
    if not RESULTS_DIR.is_dir():
        return None
    candidates = sorted(RESULTS_DIR.glob(f"{label}_*.json"), reverse=True)
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------


def run_eval(
    label: str,
    *,
    model_path: Path,
    top_k: int = 8,
    max_tokens: int = 1024,
    debug: bool = False,
) -> Path:
    """Run all eval queries and save results to a JSON file.

    Parameters
    ----------
    label:
        A short name for this eval run (e.g. ``"baseline"``, ``"step1"``).
    model_path:
        Path to the GGUF model file.
    top_k:
        Number of chunks to retrieve per query.
    max_tokens:
        Maximum tokens to generate per response.
    debug:
        Show debug output during model loading.

    Returns
    -------
    Path
        The path to the saved results JSON file.
    """
    from ._models import load_llm

    console = Console()

    console.print(f"[bold]Eval run:[/bold] {label}")
    console.print(f"[bold]Model:[/bold]  {model_path.name}")

    # Load stack
    console.print("[dim]Loading search stack...[/dim]")
    index, chunks, embedder = load_search_stack()
    console.print(f"[dim]Index: {len(chunks)} chunks[/dim]")

    console.print("[dim]Loading LLM...[/dim]")
    llm = load_llm(model_path, debug=debug)

    results: list[dict] = []

    # Single-turn queries
    console.print(f"\n[bold]Running {len(EVAL_QUERIES)} single-turn queries...[/bold]")
    for i, q in enumerate(EVAL_QUERIES, 1):
        console.print(f"  [{i}/{len(EVAL_QUERIES)}] {q['id']}: {q['query'][:60]}")

        relevant = retrieve(q["query"], index, chunks, embedder, top_k=top_k)
        relevant = enrich_chunks(relevant)
        messages = build_prompt(relevant, history=[], user_query=q["query"], llm=llm)

        t0 = time.monotonic()
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=0.9,
        )
        elapsed = time.monotonic() - t0

        answer = response["choices"][0]["message"]["content"]

        results.append(
            {
                "type": "single",
                "id": q["id"],
                "query": q["query"],
                "tags": q["tags"],
                "response": answer,
                "generation_time_s": round(elapsed, 2),
                "chunks": [
                    {
                        "source": c.source_file,
                        "score": round(c.dense_score, 4),
                    }
                    for c in relevant
                ],
            }
        )

    # Multi-turn sequences
    console.print(
        f"\n[bold]Running {len(MULTI_TURN_SEQUENCES)} multi-turn sequences...[/bold]"
    )
    for seq in MULTI_TURN_SEQUENCES:
        console.print(f"  Sequence: {seq['id']}")
        history: list[dict[str, str]] = []
        turn_results: list[dict] = []

        for turn_idx, user_msg in enumerate(seq["turns"]):
            console.print(f"    Turn {turn_idx + 1}: {user_msg[:60]}")

            relevant = retrieve(user_msg, index, chunks, embedder, top_k=top_k)
            relevant = enrich_chunks(relevant)
            messages = build_prompt(relevant, history, user_query=user_msg, llm=llm)

            t0 = time.monotonic()
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=0.9,
            )
            elapsed = time.monotonic() - t0

            answer = response["choices"][0]["message"]["content"]
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": answer})

            turn_results.append(
                {
                    "turn": turn_idx + 1,
                    "query": user_msg,
                    "response": answer,
                    "generation_time_s": round(elapsed, 2),
                }
            )

        results.append(
            {
                "type": "multi_turn",
                "id": seq["id"],
                "tags": seq["tags"],
                "turns": turn_results,
            }
        )

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _results_path(label)
    meta = {
        "label": label,
        "model": model_path.name,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "num_chunks_in_index": len(chunks),
        "top_k": top_k,
        "max_tokens": max_tokens,
        "results": results,
    }
    out_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    console.print(f"\n[green]Results saved:[/green] {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_evals(label_a: str, label_b: str) -> None:
    """Print a side-by-side comparison of two eval runs.

    Parameters
    ----------
    label_a:
        Label of the first (baseline) eval run.
    label_b:
        Label of the second (improved) eval run.
    """
    console = Console()

    path_a = _find_latest_result(label_a)
    path_b = _find_latest_result(label_b)

    if path_a is None:
        console.print(f"[red]No results found for label '{label_a}'[/red]")
        return
    if path_b is None:
        console.print(f"[red]No results found for label '{label_b}'[/red]")
        return

    data_a = json.loads(path_a.read_text(encoding="utf-8"))
    data_b = json.loads(path_b.read_text(encoding="utf-8"))

    console.print(f"\n[bold]Comparing:[/bold] {label_a} vs {label_b}")
    console.print(f"  A: {path_a.name} (model: {data_a['model']})")
    console.print(f"  B: {path_b.name} (model: {data_b['model']})")

    # Index results by id
    results_a = {r["id"]: r for r in data_a["results"]}
    results_b = {r["id"]: r for r in data_b["results"]}

    # Single-turn comparison table
    table = Table(
        title="Single-Turn Comparison",
        show_lines=True,
        width=min(console.width, 200),
    )
    table.add_column("ID", style="cyan", width=14)
    table.add_column(f"{label_a} (response)", max_width=60, overflow="fold")
    table.add_column(f"{label_b} (response)", max_width=60, overflow="fold")
    table.add_column("Time", width=12)

    all_ids = list(
        dict.fromkeys(
            [r["id"] for r in data_a["results"] if r["type"] == "single"]
            + [r["id"] for r in data_b["results"] if r["type"] == "single"]
        )
    )

    for qid in all_ids:
        ra = results_a.get(qid)
        rb = results_b.get(qid)

        resp_a = (ra["response"][:200] + "...") if ra else "(missing)"
        resp_b = (rb["response"][:200] + "...") if rb else "(missing)"

        time_a = f"{ra['generation_time_s']:.1f}s" if ra else "-"
        time_b = f"{rb['generation_time_s']:.1f}s" if rb else "-"

        table.add_row(qid, resp_a, resp_b, f"{time_a} / {time_b}")

    console.print()
    console.print(table)

    # Multi-turn comparison
    multi_a = {r["id"]: r for r in data_a["results"] if r["type"] == "multi_turn"}
    multi_b = {r["id"]: r for r in data_b["results"] if r["type"] == "multi_turn"}

    if multi_a or multi_b:
        console.print("\n[bold]Multi-Turn Sequences:[/bold]")
        for seq_id in dict.fromkeys(list(multi_a) + list(multi_b)):
            console.print(f"\n  [cyan]{seq_id}[/cyan]")
            sa = multi_a.get(seq_id, {}).get("turns", [])
            sb = multi_b.get(seq_id, {}).get("turns", [])

            for i in range(max(len(sa), len(sb))):
                ta = sa[i] if i < len(sa) else None
                tb = sb[i] if i < len(sb) else None
                query = (ta or tb)["query"]
                console.print(f"    [bold]Turn {i + 1}:[/bold] {query}")
                if ta:
                    console.print(f"      {label_a}: {ta['response'][:150]}...")
                if tb:
                    console.print(f"      {label_b}: {tb['response'][:150]}...")
