# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""CLI tools for divi-ai development.

Usage::

    python -m divi.ai help      # Show commands and workflow overview
    python -m divi.ai build     # Build the FAISS index from the local repo
    python -m divi.ai search    # Search the index interactively
    python -m divi.ai inspect   # Inspect assembled prompts (no LLM)
    python -m divi.ai eval      # Run eval queries and save results
    python -m divi.ai compare   # Compare two eval runs side-by-side
"""

import argparse
from collections.abc import Callable
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ._chat import build_prompt
from ._eval import compare_evals, run_eval
from ._indexer import DATA_DIR, build_index, build_project_meta, load_search_stack
from ._models import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL_SIZE,
    ensure_model,
    load_preferred_model,
)
from ._retriever import enrich_chunks, retrieve
from ._types import display_path

_SAMPLE_QUERIES = [
    "What is a ProgramEnsemble?",
    "How to use QoroService?",
    "How do I run VQE?",
    "How to cancel a running job?",
    "difference between VQE and QAOA",
    "JobConfig options",
    "How to create a quantum circuit?",
    "time evolution simulation",
    "early stopping criteria",
    "QAOA max clique",
]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

console = Console()

# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


_COMMANDS: dict[str, str] = {}


def cmd_help(args: argparse.Namespace) -> None:
    """Print an overview of available commands and the typical workflow."""
    console.print("\n[bold magenta]divi-ai developer tools[/bold magenta]\n")

    console.print("[bold]Commands:[/bold]")
    for name, description in _COMMANDS.items():
        console.print(f"  [cyan]{name:<10}[/cyan]{description}")
    console.print()

    console.print("[bold]Workflow:[/bold]")
    console.print("  1. [bold]Build the index[/bold] after changing source or docs:")
    console.print("     [dim]python -m divi.ai build[/dim]\n")
    console.print("  2. [bold]Verify retrieval quality[/bold] with search or inspect:")
    console.print("     [dim]python -m divi.ai search[/dim]")
    console.print("     [dim]python -m divi.ai inspect -q 'How do I run VQE?'[/dim]\n")
    console.print("  3. [bold]Launch the chatbot[/bold] to test end-to-end:")
    console.print("     [dim]divi-ai[/dim]\n")


def cmd_build(args: argparse.Namespace) -> None:
    """Build the FAISS index from the local repository."""
    console.print(f"[bold]Repo root:[/bold]  {REPO_ROOT}")
    console.print(f"[bold]Output dir:[/bold] {DATA_DIR}\n")

    index, chunks = build_index(
        [REPO_ROOT], output_dir=DATA_DIR, batch_size=args.batch_size
    )
    build_project_meta(REPO_ROOT, output_dir=DATA_DIR)

    console.print(
        f"\n[green]Done.[/green] {index.ntotal} vectors, dimension {index.d}."
    )


def cmd_search(args: argparse.Namespace) -> None:
    """Search the index with a text query and display the top-k chunks."""
    index, chunks, embedder = load_search_stack()

    console.print(
        f"[dim]Loaded {len(chunks)} chunks. Type a query (Ctrl-C to quit).[/dim]\n"
    )

    while True:
        try:
            query = console.input("[bold cyan]Query:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/dim]")
            break

        if not query:
            continue

        results = retrieve(query, index, chunks, embedder, top_k=args.top_k)

        table = Table(title=f"Top {len(results)} results", show_lines=True)
        table.add_column("#", style="bold", width=3)
        table.add_column("Score", width=7)
        table.add_column("Source")
        table.add_column("Lines", width=10)
        table.add_column("Text", max_width=80)

        for i, chunk in enumerate(results, start=1):
            # Truncate text for display
            preview = chunk.text[:200].replace("\n", "↵ ")
            if len(chunk.text) > 200:
                preview += " …"
            table.add_row(
                str(i),
                f"{chunk.score:.4f}",
                chunk.source_file,
                f"L{chunk.start_line}-{chunk.end_line}",
                preview,
            )

        console.print(table)
        console.print()


def cmd_inspect(args: argparse.Namespace) -> None:
    """Assemble prompts for sample queries and print them (no LLM)."""
    index, chunks, embedder = load_search_stack()

    queries = [args.query] if args.query else _SAMPLE_QUERIES

    for query in queries:
        relevant = retrieve(query, index, chunks, embedder, top_k=args.top_k)
        relevant = enrich_chunks(relevant)
        messages = build_prompt(relevant, history=[], user_query=query, llm=None)
        system = messages[0]["content"]
        total_chars = sum(len(m["content"]) for m in messages)

        console.rule(f"[bold]{query}[/bold]")

        # Chunk summary
        console.print("[dim]Retrieved chunks:[/dim]")
        for i, c in enumerate(relevant, 1):
            src = display_path(c.source_file)
            console.print(
                f"  [{i}] score={c.score:.3f}  dense={c.dense_score:.3f}  {src}"
            )

        # Full system message
        console.print("\n[dim]System message:[/dim]")
        console.print(system)

        # Stats
        est_tokens = total_chars // 4
        console.print(
            f"\n[dim]chars={total_chars:,}  ~tokens={est_tokens:,}  "
            f"remaining=~{4096 - est_tokens:,}[/dim]\n"
        )


def cmd_eval(args: argparse.Namespace) -> None:
    """Run eval queries against the LLM and save results."""
    model_size = load_preferred_model() or DEFAULT_MODEL_SIZE
    spec = AVAILABLE_MODELS[model_size]
    model_path = ensure_model(model_size)
    run_eval(
        args.label,
        model_path=model_path,
        n_ctx=spec.n_ctx,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        debug=args.debug,
    )


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare two eval runs side-by-side."""
    compare_evals(args.label_a, args.label_b)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _add_command(
    sub: argparse._SubParsersAction,
    name: str,
    *,
    help: str,
    func: "Callable[[argparse.Namespace], None] | None" = None,
    **kwargs,
) -> argparse.ArgumentParser:
    """Register a subparser and record it in :data:`_COMMANDS` for ``help``."""
    _COMMANDS[name] = help
    p = sub.add_parser(name, help=help, **kwargs)
    if func is not None:
        p.set_defaults(func=func)
    return p


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m divi.ai",
        description="Development tools for divi-ai.",
    )
    sub = parser.add_subparsers(dest="command")

    _add_command(
        sub, "help", help="Show commands and workflow overview.", func=cmd_help
    )
    build_parser = _add_command(
        sub, "build", help="Build the FAISS index from the local repo.", func=cmd_build
    )
    build_parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding batch size; lower to reduce memory usage (default: 16).",
    )

    search_parser = _add_command(
        sub, "search", help="Search the index interactively.", func=cmd_search
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to show (default: 5).",
    )

    inspect_parser = _add_command(
        sub,
        "inspect",
        help="Inspect assembled prompts without running the LLM.",
        func=cmd_inspect,
    )
    inspect_parser.add_argument(
        "--query", "-q", help="Single query to inspect (default: sample set)."
    )
    inspect_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5).",
    )

    eval_parser = _add_command(
        sub, "eval", help="Run eval queries and save results to JSON.", func=cmd_eval
    )
    eval_parser.add_argument(
        "--label",
        required=True,
        help="Label for this eval run (e.g. 'baseline', 'step1').",
    )
    eval_parser.add_argument(
        "--top-k", type=int, default=8, help="Chunks to retrieve (default: 8)."
    )
    eval_parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens per response (default: 1024).",
    )
    eval_parser.add_argument("--debug", action="store_true", help="Show debug output.")

    compare_parser = _add_command(
        sub, "compare", help="Compare two eval runs side-by-side.", func=cmd_compare
    )
    compare_parser.add_argument("label_a", help="First eval label (baseline).")
    compare_parser.add_argument("label_b", help="Second eval label (improved).")

    args = parser.parse_args()

    func = getattr(args, "func", cmd_help)
    func(args)


if __name__ == "__main__":
    main()
