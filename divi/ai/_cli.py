# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Interactive terminal chatbot for divi-ai.

Entry point registered as the ``divi-ai`` console script.
"""

import argparse
from pathlib import Path

from rich.console import Console

from ._indexer import load_search_stack
from ._models import (
    AVAILABLE_MODELS,
    ensure_model,
    load_llm,
    load_preferred_model,
    select_model_interactive,
)

console = Console()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="divi-ai",
        description="Chat with a divi-knowledgeable AI assistant (offline, CPU).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a local .gguf model file (bypasses model selection).",
    )
    parser.add_argument(
        "--model-size",
        choices=list(AVAILABLE_MODELS),
        default=None,
        help=(
            "Model size to use. If omitted, uses your saved preference or "
            "prompts for selection on first run."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of context chunks to retrieve per query (default: 8).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate per response (default: 1024).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug output (index load, library messages).",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Developer mode: show FAISS chunks, sources, and debug info in the TUI.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``divi-ai`` console script."""
    args = _parse_args(argv)

    # ── Model selection ────────────────────────────────────────────────
    # Priority: --model path > HuggingFace model by size
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            console.print(f"[red]Model not found: {model_path}[/red]")
            raise SystemExit(1)
        console.print(f"[bold]Model:[/bold] {model_path.name} (custom)")
        display_name = model_path.stem
    else:
        model_size = args.model_size or load_preferred_model()
        if model_size is None:
            model_size = select_model_interactive()
        spec = AVAILABLE_MODELS[model_size]
        console.print(f"[bold]Model:[/bold] {spec.label} ({spec.description})")
        model_path = ensure_model(model_size)
        # e.g. "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF" → "Qwen2.5-Coder 3B"
        base = spec.repo_id.split("/")[-1]  # "Qwen2.5-Coder-3B-Instruct-GGUF"
        for suffix in ("-Instruct-GGUF", "-GGUF"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        # "Qwen2.5-Coder-3B" → split on last dash to separate param count
        parts = base.rsplit("-", 1)
        display_name = f"{parts[0]} {parts[1]}" if len(parts) == 2 else base

    # ── Load model ────────────────────────────────────────────────────
    console.print("[dim]Loading model...[/dim]")
    llm = load_llm(model_path, debug=args.debug)

    # ── Load index + retrieval stack ─────────────────────────────────────
    console.print("[dim]Loading search index...[/dim]")
    index, chunks, embedder = load_search_stack()
    if args.debug:
        console.print(f"[dim]Index: {len(chunks)} chunks loaded[/dim]")

    # ── Launch TUI ────────────────────────────────────────────────────
    from ._tui import DiviAIApp

    app = DiviAIApp(
        llm=llm,
        index=index,
        chunks=chunks,
        embedder=embedder,
        model_name=display_name,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        dev_mode=args.dev,
    )
    app.run()


if __name__ == "__main__":
    main()
