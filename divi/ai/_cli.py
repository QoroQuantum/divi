# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Interactive terminal chatbot for divi-ai.

Entry point registered as the ``divi-ai`` console script.
"""

import argparse
import io
from contextlib import redirect_stderr
from pathlib import Path

from llama_cpp import Llama
from rich.console import Console

from ._chat import (
    build_prompt,
    generate_stream,
    get_hardware_redirect_response,
    is_history_trimmed,
)
from ._indexer import load_search_stack
from ._models import (
    AVAILABLE_MODELS,
    ensure_model,
    find_local_gguf,
    load_preferred_model,
    select_model_interactive,
)
from ._retriever import retrieve

console = Console()

# ANSI: move cursor up one line, then clear that line (used to replace "Thinking…")
_CLEAR_LINE_ABOVE = "\033[A\r\033[K"


def _clear_thinking_line() -> None:
    """Overwrite the previous line (e.g. 'Thinking…') so it can be replaced."""
    print(_CLEAR_LINE_ABOVE, end="", flush=True)


def _is_context_window_error(exc: BaseException) -> bool:
    """Heuristic: llama-cpp raises when prompt + context exceed n_ctx."""
    msg = str(exc).lower()
    return "exceed" in msg or ("context" in msg and "window" in msg)


def _load_llm(model_path: Path, *, debug: bool) -> Llama:
    """Load Llama model; show progress and library output only when debug is True."""
    if debug:
        console.print(f"[dim]Loading model from {model_path} …[/dim]")
        return Llama(
            model_path=str(model_path),
            n_ctx=8192,
            n_threads=0,
            verbose=False,
        )
    with redirect_stderr(io.StringIO()):
        return Llama(
            model_path=str(model_path),
            n_ctx=8192,
            n_threads=0,
            verbose=False,
        )


def _stream_assistant_response(
    llm: Llama, messages: list[dict], max_tokens: int
) -> list[str] | None:
    """Stream response with 'Thinking…' UX; clear that line when first token arrives.

    Returns list of token strings on success; None if interrupted or context-window error
    (caller already sees [Stopped.] or context message).
    """
    console.print("[dim]Thinking…[/dim]")
    response_parts: list[str] = []
    first_token = True
    try:
        for token in generate_stream(llm, messages, max_tokens=max_tokens):
            response_parts.append(token)
            if first_token:
                _clear_thinking_line()
                console.print("[bold green]Assistant:[/bold green] " + token, end="")
                first_token = False
            else:
                print(token, end="", flush=True)
        if first_token:
            _clear_thinking_line()
            console.print("[bold green]Assistant:[/bold green] ")
        print("\n")
        return response_parts
    except KeyboardInterrupt:
        if first_token:
            _clear_thinking_line()
        print("\n")
        console.print("[dim][Stopped.][/dim]")
        console.print(
            "[dim]Say [bold]reset[/bold] to clear history, or ask something else.[/dim]\n"
        )
        return None
    except Exception as e:
        if _is_context_window_error(e):
            if first_token:
                _clear_thinking_line()
            print("\n")
            console.print(
                "[yellow]This conversation is too long for the context window.[/yellow]"
            )
            console.print(
                "[dim]Say [bold]reset[/bold] to start fresh, or ask a shorter question.[/dim]\n"
            )
            return None
        raise


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
        "--base",
        action="store_true",
        help="Use the base model from HuggingFace; skip local fine-tune if present.",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``divi-ai`` console script."""
    args = _parse_args(argv)

    # ── Model selection ────────────────────────────────────────────────
    # Priority: --model path > (local fine-tune unless --base) > HuggingFace base
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            console.print(f"[red]Model not found: {model_path}[/red]")
            raise SystemExit(1)
        console.print(f"[bold]Model:[/bold] {model_path.name} (custom)")
    elif args.base or (local := find_local_gguf()) is None:
        model_size = args.model_size or load_preferred_model()
        if model_size is None:
            model_size = select_model_interactive()
        spec = AVAILABLE_MODELS[model_size]
        console.print(f"[bold]Model:[/bold] {spec.label} ({spec.description})")
        model_path = ensure_model(model_size)
    else:
        model_path = local
        console.print(f"[bold]Model:[/bold] {model_path.name} (local fine-tune)")

    # ── Load model ────────────────────────────────────────────────────
    llm = _load_llm(model_path, debug=args.debug)

    # ── Load index + retrieval stack ─────────────────────────────────────
    index, chunks, embedder = load_search_stack()
    if args.debug:
        console.print(f"[dim]Index: {len(chunks)} chunks loaded[/dim]")

    # ── Chat loop ──────────────────────────────────────────────────────
    history: list[dict[str, str]] = []

    console.print()
    console.rule("[bold magenta]divi-ai[/bold magenta]")
    console.print(
        "\n[yellow]⚠  Experimental:[/yellow] divi-ai uses a small local LLM "
        "and may produce inaccurate or hallucinated responses.\n"
        "   Always verify code against the official documentation.\n"
        "   Feedback → [link=https://github.com/QoroQuantum/divi/issues]"
        "github.com/QoroQuantum/divi/issues[/link]\n"
    )
    console.print(
        "[dim]Type your question. Commands: "
        "[bold]reset[/bold] (clear history), "
        "[bold]quit[/bold] (exit)[/dim]\n"
    )

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        command = user_input.lower()
        if command in ("quit", "exit"):
            console.print("[dim]Goodbye![/dim]")
            break
        if command == "reset":
            history.clear()
            console.print("[yellow]— conversation history cleared —[/yellow]\n")
            continue

        # Always redirect questions about real hardware / cloud providers to QoroService
        redirect = get_hardware_redirect_response(user_input)
        if redirect is not None:
            console.print("[bold green]Assistant:[/bold green] ", end="")
            console.print(redirect + "\n")
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": redirect})
            continue

        # For short follow-ups like "like what?", augment the search query.
        last_user = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"),
            None,
        )
        search_query = (
            f"{last_user} {user_input}"
            if last_user and len(user_input.split()) <= 5
            else user_input
        )

        relevant = retrieve(search_query, index, chunks, embedder, top_k=args.top_k)

        messages = build_prompt(relevant, history, user_input)

        if is_history_trimmed(history):
            console.print(
                "[dim]Older messages are omitted from context. "
                "Say [bold]reset[/bold] to start a fresh conversation.[/dim]"
            )

        response_parts = _stream_assistant_response(
            llm, messages, max_tokens=args.max_tokens
        )
        if response_parts is not None:
            full_response = "".join(response_parts)
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
