# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Textual-based TUI for divi-ai.

A rich terminal interface with a branded header, scrollable chat area,
auto syntax-checking of code blocks, and slash commands (/save, /check).
"""

import py_compile
import re
import tempfile
import time
import webbrowser
from pathlib import Path

import faiss
import pyperclip
from fastembed import TextEmbedding
from llama_cpp import Llama
from rich.console import Group
from rich.markdown import Markdown
from rich.markup import escape
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Input, Static

from ._chat import (
    build_prompt,
    generate_stream,
    get_hardware_redirect_response,
    is_history_trimmed,
)
from ._retriever import enrich_chunks, retrieve
from ._types import ChunkMeta, short_source


class ChatInput(Input):
    """Input that treats the space key as inserting a space character.

    Textual sends the space bar as key='space' with character=None, so the
    default Input does not insert it. This subclass inserts a space when
    key is 'space' so the prompt can contain spaces.
    """

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "space" and event.character is None:
            self._restart_blink()
            event.stop()
            selection = self.selection
            if selection.is_empty:
                self.insert_text_at_cursor(" ")
            else:
                self.replace(" ", *selection)
            event.prevent_default()
            return
        await super()._on_key(event)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_code_blocks(text: str) -> list[str]:
    """Extract all fenced python code blocks from a markdown response."""
    return _CODE_BLOCK_RE.findall(text)


def _safe_display(text: str) -> str:
    """Escape Rich markup in user/LLM content so it is shown literally."""
    return escape(str(text))


def _format_assistant_response(text: str) -> Group:
    """Render an assistant response as 'Assistant:' label + markdown body."""
    label = Text.from_markup("[bold #FF93FA]Assistant:[/bold #FF93FA]")
    body = Markdown(text)
    return Group(label, body)


def _syntax_check(code: str) -> str | None:
    """Return None if code compiles, else the error message."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        f.flush()
        try:
            py_compile.compile(f.name, doraise=True)
            return None
        except py_compile.PyCompileError as e:
            return str(e)
        finally:
            Path(f.name).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


# Brand colors for the thinking pulse animation
_PULSE_COLORS = [
    "#FF93FA",
    "#F47CF5",
    "#E964F0",
    "#DE4DEB",
    "#D435E6",
    "#C92AE0",
    "#B02FD1",
    "#9734C3",
    "#7E39B4",
    "#653EA6",
    "#4D4397",
    "#3E4E8E",
    "#2F5985",
    "#2967A2",
    "#2F5985",
    "#3E4E8E",
    "#4D4397",
    "#653EA6",
    "#7E39B4",
    "#9734C3",
    "#B02FD1",
    "#C92AE0",
    "#D435E6",
    "#DE4DEB",
    "#E964F0",
    "#F47CF5",
]

_IDLE_COLOR = "#FF93FA"


class BrandBar(Static):
    """Top bar with branded text and pulse animation."""

    DEFAULT_CSS = """
    BrandBar {
        dock: top;
        height: 3;
        background: #1E1F20;
        color: #E9EEF6;
        padding: 1 2;
    }
    """

    def __init__(self, model_name: str = "", disclaimer: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._model_name = model_name
        self._disclaimer = disclaimer
        self._context_usage = ""
        self._pulse_index = 0
        self._pulse_timer = None

    def on_mount(self) -> None:
        self.start_pulse()

    def set_context_usage(self, used: int, total: int) -> None:
        pct = used * 100 // total if total else 0
        color = "#50fa7b" if pct < 75 else "#FFB86C" if pct < 90 else "#ff5555"
        self._context_usage = f" [{color}](Context Used: {pct}%)[/{color}]"
        self._update_bar(
            _IDLE_COLOR
            if self._pulse_timer is None
            else _PULSE_COLORS[self._pulse_index % len(_PULSE_COLORS)]
        )

    def _update_bar(self, color: str) -> None:
        model = (
            f"  [#94A0AA]·[/#94A0AA]  [#E9EEF6]{self._model_name}[/#E9EEF6]"
            if self._model_name
            else ""
        )
        left = f"[bold {color}]Qoro[/bold {color}]  [#94A0AA]·[/#94A0AA]  [bold #E9EEF6]divi-ai[/bold #E9EEF6]{model}{self._context_usage}"
        disclaimer = (
            f"[#94A0AA]{self._disclaimer}[/#94A0AA]" if self._disclaimer else ""
        )

        left_text = Text.from_markup(left)
        disc_text = Text.from_markup(disclaimer)
        # Available width minus padding (2 each side)
        width = self.size.width - 4 if self.size.width > 0 else 120
        gap = max(4, width - left_text.cell_len - disc_text.cell_len)
        self.update(f"{left}{' ' * gap}{disclaimer}")

    def start_pulse(self) -> None:
        """Start the color pulse animation."""
        self._pulse_index = 0
        if self._pulse_timer is None:
            self._pulse_timer = self.set_interval(0.08, self._tick_pulse)

    def stop_pulse(self) -> None:
        """Stop the animation and reset to idle color."""
        if self._pulse_timer is not None:
            self._pulse_timer.stop()
            self._pulse_timer = None
        self._update_bar(_IDLE_COLOR)

    def _tick_pulse(self) -> None:
        color = _PULSE_COLORS[self._pulse_index % len(_PULSE_COLORS)]
        self._update_bar(color)
        self._pulse_index += 1


class ChatMessage(Static):
    """A single message in the chat log."""

    DEFAULT_CSS = """
    ChatMessage {
        padding: 0 2 1 2;
    }
    """


_COMMAND_HELP = [
    ("/save <file>", "Save the last code block to a file and syntax-check it."),
    ("/copy", "Copy the last code block to the clipboard."),
    ("/check", "Syntax-check the last saved file or the latest code blocks."),
    ("/retry", "Generate a new answer for your last request."),
    ("/reset", "Clear the current conversation history."),
    ("/clear", "Clear the chat and reset the current conversation history."),
    ("/quit", "Exit the TUI."),
    ("/exit", "Alias for /quit."),
]

_KNOWN_COMMANDS = {command.split()[0] for command, _ in _COMMAND_HELP}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class DiviAIApp(App):
    """Textual TUI for divi-ai."""

    TITLE = "divi-ai"
    CSS = """
    Screen {
        background: #151616;
    }

    #chat-scroll {
        height: 1fr;
        background: #151616;
        scrollbar-size: 1 1;
        scrollbar-color: #2967A2;
        scrollbar-color-hover: #ED56E6;
    }

    #footer {
        dock: bottom;
        height: auto;
        width: 100%;
        background: #151616;
    }

    #chat-input {
        box-sizing: border-box;
        height: 3;
        background: #1E1F20;
        border: tall #2967A2;
        padding: 0 1;
    }

    #chat-input:focus {
        border: tall #FF93FA;
    }

    #slash-help {
        display: none;
        background: #1E1F20;
        color: #CDD2DA;
        padding: 0 2 1 2;
    }

    .user-msg {
        color: #FDB9FD;
    }

    .assistant-msg {
        color: #E9EEF6;
    }

    .system-msg {
        color: #94A0AA;
        text-style: italic;
    }

    .error-msg {
        color: #ff5555;
    }

    .success-msg {
        color: #50fa7b;
    }

    .sources-msg {
        color: #94A0AA;
    }
    """

    ENABLE_COMMAND_PALETTE = False

    def __init__(
        self,
        llm: Llama,
        index: faiss.IndexFlatIP,
        chunks: list[ChunkMeta],
        embedder: TextEmbedding,
        *,
        model_name: str = "",
        top_k: int = 8,
        max_tokens: int = 1024,
        dev_mode: bool = False,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.index = index
        self.chunks = chunks
        self.embedder = embedder
        self.model_name = model_name
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.dev_mode = dev_mode

        self.history: list[dict[str, str]] = []
        self.last_response: str = ""
        self.last_saved_path: Path | None = None
        self._generating = False
        self._cancel_requested = False
        self._thinking_msg: ChatMessage | None = None
        self._thinking_timer = None
        self._thinking_dots = 0

    def compose(self) -> ComposeResult:
        model_info = self.model_name
        if self.dev_mode:
            model_info += f"  |  {len(self.chunks)} chunks"
        disclaimer = (
            "Experimental local assistant — responses may be inaccurate. "
            "[@click='app.open_link(\"https://qoroquantum.github.io/divi/\")']"
            "[#FF93FA]Official docs ↗[/#FF93FA][/]"
        )
        yield BrandBar(model_name=model_info, disclaimer=disclaimer, id="brand-bar")
        scroll = VerticalScroll(id="chat-scroll")
        scroll.can_focus = False
        yield scroll
        with Vertical(id="footer"):
            yield Static("", id="slash-help")
            yield ChatInput(
                placeholder="Ask about Divi or type / for commands",
                id="chat-input",
            )

    _WELCOME_EXAMPLES = [
        "How do I create and run a VQE problem?",
        "What optimizers are available in Divi?",
        "Show me how to set up QAOA for a QUBO problem",
        "How do I use checkpointing?",
    ]

    def on_mount(self) -> None:
        examples = "  ".join(
            f"[#94A0AA]•[/#94A0AA] [italic #CDD2DA]{q}[/italic #CDD2DA]"
            for q in self._WELCOME_EXAMPLES
        )
        self._append_message(
            f"[bold #FF93FA]Welcome to divi-ai![/bold #FF93FA]\n\n"
            f"Ask anything about the Divi quantum computing library. "
            f"Type [bold #FF93FA]/[/bold #FF93FA] to see available commands.\n\n"
            f"Try asking:\n{examples}",
            "system-msg",
        )
        self.query_one("#chat-input", ChatInput).focus()

    # ------------------------------------------------------------------
    # Chat display helpers
    # ------------------------------------------------------------------

    def _append_message(self, text: str, css_class: str = "") -> ChatMessage:
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        msg = ChatMessage(text, classes=css_class)
        scroll.mount(msg)
        scroll.scroll_end(animate=False)
        return msg

    def _append_system(self, text: str) -> None:
        self._append_message(text, "system-msg")

    def _start_thinking_indicator(self) -> ChatMessage:
        self._stop_thinking_indicator()
        self._thinking_dots = 1
        self._thinking_msg = self._append_message(
            "[#94A0AA]Thinking.[/#94A0AA]", "system-msg"
        )
        self._thinking_timer = self.set_interval(0.35, self._tick_thinking_indicator)
        return self._thinking_msg

    def _tick_thinking_indicator(self) -> None:
        if self._thinking_msg is None:
            return
        dots = "." * self._thinking_dots
        self._thinking_msg.update(f"[#94A0AA]Thinking{dots}[/#94A0AA]")
        self._thinking_dots = 1 if self._thinking_dots >= 3 else self._thinking_dots + 1

    def _stop_thinking_indicator(self) -> None:
        if self._thinking_timer is not None:
            self._thinking_timer.stop()
            self._thinking_timer = None
        self._thinking_msg = None
        self._thinking_dots = 0

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape" and self._generating:
            self._cancel_requested = True
            event.prevent_default()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "chat-input":
            self._update_slash_help(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        user_input = event.value.strip()

        if not user_input or self._generating:
            return

        mixed_command = self._find_mixed_slash_command(user_input)
        if mixed_command is not None:
            self._update_slash_help(
                mixed_command,
                error="Slash commands can't be mixed with a normal prompt.",
            )
            return

        # Slash commands
        lower = user_input.lower()
        if lower.startswith("/"):
            parsed = self._parse_slash_command(user_input)
            if isinstance(parsed, str):
                self._update_slash_help(user_input, error=parsed)
                return

            command, args = parsed
            event.input.value = ""
            self._update_slash_help("")

            if command in {"/quit", "/exit"}:
                self.exit()
                return
            if command == "/reset":
                self._handle_reset()
                return
            if command == "/clear":
                self.action_clear_chat()
                return
            if command == "/check":
                self._handle_check()
                return
            if command == "/retry":
                self._handle_retry()
                return
            if command == "/save":
                self._handle_save(args[0])
                return
            if command == "/copy":
                self._handle_copy()
                return

        event.input.value = ""
        self._update_slash_help("")
        self._append_message(
            f"[bold #ED56E6]You:[/bold #ED56E6] {_safe_display(user_input)}",
            "user-msg",
        )
        self._run_query(user_input)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _split_command_input(self, user_input: str) -> tuple[str, str]:
        parts = user_input.split(maxsplit=1)
        command = parts[0].lower()
        remainder = parts[1].strip() if len(parts) > 1 else ""
        return command, remainder

    def _parse_slash_command(self, user_input: str) -> tuple[str, list[str]] | str:
        command, remainder = self._split_command_input(user_input)

        if command in {
            "/quit",
            "/exit",
            "/reset",
            "/clear",
            "/check",
            "/retry",
            "/copy",
        }:
            if remainder:
                return "Slash commands must be entered on their own."
            return command, []

        if command == "/save":
            if not remainder:
                return "Use /save <file>."
            if (
                len(remainder) >= 2
                and remainder[0] == remainder[-1]
                and remainder[0] in {'"', "'"}
            ):
                remainder = remainder[1:-1]
            if not remainder:
                return "Use /save <file>."
            return command, [remainder]

        return "Unknown slash command. Type / for commands."

    def _find_mixed_slash_command(self, user_input: str) -> str | None:
        if user_input.startswith("/"):
            return None

        parts = user_input.split()
        for part in parts[1:]:
            token = part.lower()
            if token in _KNOWN_COMMANDS:
                return token

        return None

    def _update_slash_help(self, value: str, error: str | None = None) -> None:
        helper = self.query_one("#slash-help", Static)
        query = value.strip()
        lowered = query.lower()

        if not lowered.startswith("/"):
            if error is not None:
                helper.update(
                    f"[bold #ff5555]Warning[/bold #ff5555] [#ff5555]{error}[/#ff5555]"
                )
                helper.display = True
                return
            helper.display = False
            helper.update("")
            return

        token = lowered.split(maxsplit=1)[0]
        matches = [
            (command, description)
            for command, description in _COMMAND_HELP
            if token == "/" or command.startswith(token)
        ]

        lines: list[str] = []
        if error is not None:
            lines.append(
                f"[bold #ff5555]Warning[/bold #ff5555] [#ff5555]{error}[/#ff5555]"
            )

        if not matches:
            lines.append("[#94A0AA]No matching slash commands.[/#94A0AA]")
            helper.update("\n".join(lines))
            helper.display = True
            return

        lines.append("[bold #FF93FA]Commands[/bold #FF93FA]")
        for command, description in matches:
            lines.append(
                f"[#FF93FA]{command}[/#FF93FA]  [#94A0AA]{description}[/#94A0AA]"
            )

        helper.update("\n".join(lines))
        helper.display = True

    def _handle_reset(self) -> None:
        self.history.clear()
        self.last_response = ""
        self._append_system("[#CDD2DA]-- conversation history cleared --[/#CDD2DA]")

    def _get_last_code_block(self) -> str | None:
        """Extract the last code block from the most recent response.

        Shows a system message and returns ``None`` if no code blocks exist.
        """
        blocks = _extract_code_blocks(self.last_response)
        if not blocks:
            self._append_system(
                "[#CDD2DA]No code blocks found in the last response.[/#CDD2DA]"
            )
            return None
        return blocks[-1]

    def _handle_save(self, filename: str) -> None:
        code = self._get_last_code_block()
        if code is None:
            return

        path = Path(filename)
        try:
            path.write_text(code, encoding="utf-8")
            self.last_saved_path = path
            self._append_system(
                f"[#50fa7b]Saved to {path} ({len(code)} chars)[/#50fa7b]"
            )

            # Auto-check syntax
            err = _syntax_check(code)
            if err is None:
                self._append_message("[#50fa7b]  Syntax OK[/#50fa7b]", "success-msg")
            else:
                self._append_message(
                    f"[#ff5555]  Syntax error: {err}[/#ff5555]", "error-msg"
                )
        except OSError as e:
            self._append_message(
                f"[#ff5555]Failed to save: {_safe_display(str(e))}[/#ff5555]",
                "error-msg",
            )

    def _handle_copy(self) -> None:
        code = self._get_last_code_block()
        if code is None:
            return

        try:
            pyperclip.copy(code)
            self._append_system(
                f"[#50fa7b]Copied to clipboard ({len(code)} chars)[/#50fa7b]"
            )
        except pyperclip.PyperclipException as e:
            self._append_message(
                f"[#ff5555]Clipboard error: {_safe_display(str(e))}[/#ff5555]",
                "error-msg",
            )

    def _handle_check(self) -> None:
        if self.last_saved_path and self.last_saved_path.exists():
            code = self.last_saved_path.read_text(encoding="utf-8")
            err = _syntax_check(code)
            if err is None:
                self._append_message(
                    f"[#50fa7b]{self.last_saved_path}: Syntax OK[/#50fa7b]",
                    "success-msg",
                )
            else:
                self._append_message(
                    f"[#ff5555]{_safe_display(err)}[/#ff5555]", "error-msg"
                )
            return

        # Fall back to checking last response code blocks
        blocks = _extract_code_blocks(self.last_response)
        if not blocks:
            self._append_system(
                "[#CDD2DA]Nothing to check. No code blocks or saved files.[/#CDD2DA]"
            )
            return
        for i, block in enumerate(blocks, 1):
            err = _syntax_check(block)
            label = f"Block {i}" if len(blocks) > 1 else "Code"
            if err is None:
                self._append_message(
                    f"[#50fa7b]{label}: Syntax OK[/#50fa7b]", "success-msg"
                )
            else:
                self._append_message(
                    f"[#ff5555]{label}: {_safe_display(err)}[/#ff5555]",
                    "error-msg",
                )

    def _handle_retry(self) -> None:
        last_user = next(
            (m["content"] for m in reversed(self.history) if m["role"] == "user"),
            None,
        )
        if last_user is None:
            self._append_system("[#CDD2DA]Nothing to retry.[/#CDD2DA]")
            return

        # Remove last exchange
        if len(self.history) >= 2:
            self.history.pop()  # assistant
            self.history.pop()  # user

        self._append_message(
            f"[bold #8be9fd]You (retry):[/bold #8be9fd] {_safe_display(last_user)}",
            "user-msg",
        )
        self._run_query(last_user, temperature=0.4)

    # ------------------------------------------------------------------
    # Query execution (async worker)
    # ------------------------------------------------------------------

    @work(exclusive=True, thread=True)
    def _run_query(self, user_input: str, *, temperature: float = 0.2) -> None:
        self._generating = True
        self._cancel_requested = False
        try:
            self._run_query_impl(user_input, temperature)
        finally:
            self._generating = False
            self._cancel_requested = False

    def _do_retrieval(self, user_input: str) -> list:
        """Retrieve and enrich context chunks for *user_input*.

        Augments short follow-ups by prepending the previous user query.
        """
        last_user = next(
            (m["content"] for m in reversed(self.history) if m["role"] == "user"),
            None,
        )
        search_query = (
            f"{last_user} {user_input}"
            if last_user and len(user_input.split()) <= 5
            else user_input
        )

        relevant = retrieve(
            search_query, self.index, self.chunks, self.embedder, top_k=self.top_k
        )
        return enrich_chunks(relevant)

    def _stream_response(
        self,
        messages: list[dict[str, str]],
        thinking_msg: ChatMessage,
        temperature: float,
    ) -> tuple[str, bool]:
        """Stream LLM tokens into *thinking_msg*.

        Returns ``(full_text, cancelled)`` — *cancelled* is ``True`` if the
        user pressed Escape mid-stream.  Raises on LLM errors.
        """
        response_parts: list[str] = []
        cancelled = False
        for token in generate_stream(
            self.llm, messages, max_tokens=self.max_tokens, temperature=temperature
        ):
            if self._cancel_requested:
                cancelled = True
                break
            if not response_parts:
                self.call_from_thread(self._stop_thinking_indicator)
            response_parts.append(token)

            current_text = "".join(response_parts)
            self.call_from_thread(
                thinking_msg.update,
                _format_assistant_response(current_text),
            )

        return "".join(response_parts), cancelled

    def _post_response(
        self,
        user_input: str,
        full_response: str,
        messages: list[dict[str, str]],
        relevant: list,
        elapsed: float,
        thinking_msg: ChatMessage,
    ) -> None:
        """Finalize a successful response: history, context bar, syntax check, sources."""
        tps = len(full_response.split()) / elapsed if elapsed > 0 else 0

        # Final markdown render
        self.call_from_thread(self._stop_thinking_indicator)
        self.call_from_thread(
            thinking_msg.update,
            _format_assistant_response(full_response),
        )

        # Update context usage in header
        prompt_tokens = sum(
            len(self.llm.tokenize(m["content"].encode(), add_bos=False))
            for m in messages
        )
        response_tokens = len(self.llm.tokenize(full_response.encode(), add_bos=False))
        brand_bar = self.query_one("#brand-bar", BrandBar)
        self.call_from_thread(
            brand_bar.set_context_usage,
            prompt_tokens + response_tokens,
            self.llm.n_ctx(),
        )

        # Record in history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": full_response})
        self.last_response = full_response

        # Auto syntax-check code blocks
        blocks = _extract_code_blocks(full_response)
        for i, block in enumerate(blocks):
            err = _syntax_check(block)
            label = f"Block {i + 1}" if len(blocks) > 1 else "Code"
            if err is None:
                self.call_from_thread(
                    self._append_message,
                    f"[#50fa7b]  {label}: syntax OK[/#50fa7b]",
                    "success-msg",
                )
            else:
                short_err = err.split("\n")[-1] if "\n" in err else err
                self.call_from_thread(
                    self._append_message,
                    f"[#ff5555]  {label}: syntax error - {_safe_display(short_err)}[/#ff5555]",
                    "error-msg",
                )

        # Show sources (dev mode only)
        if self.dev_mode:
            sources = [short_source(c.source_file) for c in relevant[:4]]
            source_line = "  ".join(sources)
            self.call_from_thread(
                self._append_message,
                f"[#94A0AA]Sources: {source_line}  |  {tps:.1f} tok/s[/#94A0AA]",
                "sources-msg",
            )

        # Scroll to bottom
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        self.call_from_thread(scroll.scroll_end, animate=False)

    def _run_query_impl(self, user_input: str, temperature: float) -> None:
        # Hardware redirect (no LLM)
        redirect = get_hardware_redirect_response(user_input)
        if redirect is not None:
            self.call_from_thread(
                self._append_message,
                _format_assistant_response(redirect),
                "assistant-msg",
            )
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": redirect})
            self.last_response = redirect
            return

        thinking_msg = self.call_from_thread(self._start_thinking_indicator)

        relevant = self._do_retrieval(user_input)

        if is_history_trimmed(self.history, self.llm):
            self.call_from_thread(
                self._append_system,
                "[#94A0AA]Older messages trimmed. Use /reset to start fresh.[/#94A0AA]",
            )

        messages = build_prompt(relevant, self.history, user_input, self.llm)

        t0 = time.monotonic()
        try:
            full_response, cancelled = self._stream_response(
                messages, thinking_msg, temperature
            )
        except Exception as e:
            msg = str(e).lower()
            if "exceed" in msg or ("context" in msg and "window" in msg):
                self.call_from_thread(
                    self._append_system,
                    "[#CDD2DA]Context window exceeded. Use /reset to start fresh.[/#CDD2DA]",
                )
            else:
                self.call_from_thread(
                    self._append_message,
                    f"[#ff5555]Error: {_safe_display(str(e))}[/#ff5555]",
                    "error-msg",
                )
            self.call_from_thread(self._stop_thinking_indicator)
            self.call_from_thread(thinking_msg.update, "")
            return
        elapsed = time.monotonic() - t0

        if cancelled:
            self.call_from_thread(self._stop_thinking_indicator)
            if full_response:
                # Keep partial response visible
                self.call_from_thread(
                    thinking_msg.update,
                    _format_assistant_response(full_response),
                )
            else:
                self.call_from_thread(thinking_msg.update, "")
            self.call_from_thread(
                self._append_system,
                "[#94A0AA]-- generation cancelled --[/#94A0AA]",
            )
            return

        self._post_response(
            user_input, full_response, messages, relevant, elapsed, thinking_msg
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_open_link(self, url: str) -> None:
        webbrowser.open(url)

    def action_clear_chat(self) -> None:
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        scroll.remove_children()
        self._handle_reset()
