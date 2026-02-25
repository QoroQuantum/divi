# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""CLI tools for divi-ai development.

Usage::

    python -m divi.ai help      # Show commands and workflow overview
    python -m divi.ai build     # Build the FAISS index from the local repo
    python -m divi.ai search    # Search the index interactively
    python -m divi.ai inspect   # Inspect assembled prompts (no LLM)
"""

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ._chat import build_prompt
from ._indexer import DATA_DIR, build_index, build_project_meta, load_search_stack
from ._retriever import retrieve

_SAMPLE_QUERIES = [
    "What is a ProgramBatch?",
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

    index, chunks = build_index([REPO_ROOT], output_dir=DATA_DIR)
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
        messages = build_prompt(relevant, history=[], user_query=query)
        system = messages[0]["content"]
        total_chars = sum(len(m["content"]) for m in messages)

        console.rule(f"[bold]{query}[/bold]")

        # Chunk summary
        console.print("[dim]Retrieved chunks:[/dim]")
        for i, c in enumerate(relevant, 1):
            src = c.source_file
            if "Qoro/divi/" in src:
                src = src.split("Qoro/divi/")[-1]
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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _add_command(
    sub: argparse._SubParsersAction, name: str, *, help: str, **kwargs
) -> argparse.ArgumentParser:
    """Register a subparser and record it in :data:`_COMMANDS` for ``help``."""
    _COMMANDS[name] = help
    return sub.add_parser(name, help=help, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m divi.ai",
        description="Development tools for divi-ai.",
    )
    sub = parser.add_subparsers(dest="command")

    _add_command(sub, "help", help="Show commands and workflow overview.")
    _add_command(sub, "build", help="Build the FAISS index from the local repo.")

    search_parser = _add_command(sub, "search", help="Search the index interactively.")
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to show (default: 5).",
    )

    inspect_parser = _add_command(
        sub, "inspect", help="Inspect assembled prompts without running the LLM."
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

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    else:
        cmd_help(args)


if __name__ == "__main__":
    main()
