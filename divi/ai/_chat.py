# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Prompt construction and LLM generation for divi-ai.

Builds ChatML-formatted prompts with retrieved context and streams
responses from a local GGUF model via llama-cpp-python.
"""

import re
from collections.abc import Iterator

from llama_cpp import Llama

from ._indexer import load_project_meta
from ._retriever import RetrievedChunk
from ._types import display_path

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_BASE = """\
You are **divi-ai**, a helpful coding assistant for the Divi quantum \
computing library by Qoro Quantum.

Use the supplied CONTEXT as the sole source of truth. A question about a Divi \
API, feature, or workflow shown in CONTEXT is in scope: answer it directly and \
never give a generic scope refusal. If the context is insufficient, say what \
information is missing and point to https://divi.readthedocs.io or \
github.com/QoroQuantum/divi/issues. Only use the generic refusal "I can only \
help with the Divi quantum computing library." when the question is clearly \
unrelated to Divi and no relevant documentation was retrieved.

ACCURACY:
  - Do not invent APIs, arguments, return values, attributes, or import paths.
  - Keep examples for different classes separate. Never copy an argument or \
method from one class's example into another class's example.
  - Before using a method or argument, verify that CONTEXT shows it on that \
same class. If CONTEXT documents different patterns for different classes, \
explain them separately instead of merging them.
  - When writing code, follow one coherent example from CONTEXT and include \
the imports and definitions needed to run it. Mark unavoidable user-supplied \
values explicitly instead of silently assuming them.
  - Preserve the example's execution order: perform the run or submission \
before reading or aggregating its results. Every referenced public name must \
either be imported or clearly marked as a user-supplied value.
  - Never import from the repository's ``tutorials`` package in user code. \
Tutorial helpers are not part of Divi's installed public API; use a public \
Divi backend or leave the backend as an explicit user-supplied value.
  - For list or availability questions, include the relevant public items \
shown in CONTEXT. Do not present state objects, configuration objects, \
ansatzes, or algorithms as members of another category.
  - Do not invent category headings. If CONTEXT does not explicitly classify \
the listed items, present a flat list.
  - Answer directly without echoing the question. Prefer concise, complete \
sentences over fragments.
  - When the user explicitly asks for a concise example, give one minimal \
example. Omit introductory prose, summaries, key-point lists, optional \
extensions, and unrelated APIs.

If the requested algorithm does not fit the problem, explain the mismatch and \
recommend the appropriate Divi workflow only when CONTEXT supports that \
recommendation. Do not add unsolicited hardware or provider advice. When the \
user explicitly asks about real hardware or third-party quantum cloud \
providers, mention only QoroService and suggest they contact Qoro.
"""


# The LLM occasionally leaks the internal scope label as a "SCOPE: IN" /
# "**SCOPE: REDIRECT**" preamble line. Strip those before showing the
# response to users.
_SCOPE_PREFIX_RE = re.compile(
    r"\A\**\s*SCOPE\s*:\s*(IN|OUT|REDIRECT|WRONG[- ]?TOOL)\**\s*\n+",
    re.IGNORECASE,
)


def strip_scope_preamble(response: str) -> str:
    """Remove any leading ``SCOPE: …`` scaffolding from an LLM response."""
    return _SCOPE_PREFIX_RE.sub("", response, count=1)


def _build_system_prompt() -> str:
    """Build the system prompt, injecting dynamic project metadata."""
    meta = load_project_meta()
    if meta is None:
        return _SYSTEM_PROMPT_BASE

    parts = [_SYSTEM_PROMPT_BASE]

    # Project info
    project = meta.get("project", {})
    info_lines: list[str] = []
    if "name" in project:
        info_lines.append(f"- Package name: {project['name']}")
    if "python" in project:
        info_lines.append(f"- Python: {project['python']}")
    info_lines.append("- Install: pip install divi")
    if info_lines:
        parts.append("PROJECT INFO:\n" + "\n".join(info_lines))

    return "\n\n".join(parts) + "\n"


SYSTEM_PROMPT = _build_system_prompt()

# Fraction of the model's context window reserved for conversation history.
# The rest goes to the system prompt + RAG chunks + the current query.
HISTORY_BUDGET_FRACTION = 0.25

# Absolute floor so tiny context windows still keep at least one exchange.
MIN_HISTORY_TOKENS = 512


def _history_budget(llm: Llama) -> int:
    """Compute the token budget for conversation history based on model context size."""
    return max(MIN_HISTORY_TOKENS, int(llm.n_ctx() * HISTORY_BUDGET_FRACTION))


def _trim_history(
    history: list[dict[str, str]],
    llm: Llama,
    max_tokens: int | None = None,
) -> list[dict[str, str]]:
    """Keep newest messages that fit within *max_tokens*.

    Walks the history from newest to oldest, accumulating token counts.
    Stops when adding another message would exceed the budget.  Always
    keeps messages in pairs (user + assistant) to avoid orphaned turns.
    """
    if not history:
        return []

    if max_tokens is None:
        max_tokens = _history_budget(llm)

    # Pre-compute token counts for every message (cheap, no inference).
    counts = [len(llm.tokenize(m["content"].encode(), add_bos=False)) for m in history]

    # Walk backwards in pairs (assistant, user).
    total = 0
    keep_from = len(history)
    i = len(history) - 1
    while i >= 1:
        pair_cost = counts[i] + counts[i - 1]
        if total + pair_cost > max_tokens:
            break
        total += pair_cost
        keep_from = i - 1
        i -= 2

    return history[keep_from:]


def is_history_trimmed(history: list[dict[str, str]], llm: "Llama") -> bool:
    """True if history would be trimmed by the token budget."""
    if not history:
        return False
    return len(_trim_history(history, llm)) < len(history)


# ---------------------------------------------------------------------------
# Hardware / provider redirect (always redirect, no LLM)
# ---------------------------------------------------------------------------

_HARDWARE_REDIRECT_KEYWORDS = (
    "real quantum hardware",
    "real hardware",
    "quantum hardware",
    "azure quantum",
    "aws braket",
    "ibm quantum",
    "ibm q",
    "quantum provider",
    "cloud provider",
    "third-party backend",
    "run on hardware",
    "execute on hardware",
)

HARDWARE_REDIRECT_MESSAGE = (
    "For execution on real quantum hardware or quantum cloud providers "
    "(e.g. Azure Quantum, AWS Braket, IBM Quantum), please use **QoroService** — "
    "Divi's cloud offering by Qoro Quantum. Get in touch with us for more details."
)


def get_hardware_redirect_response(user_query: str) -> str | None:
    """If the query is about real hardware or external providers, return the redirect message; else None."""
    q = user_query.lower().strip()
    if not q:
        return None
    for kw in _HARDWARE_REDIRECT_KEYWORDS:
        if kw in q:
            return HARDWARE_REDIRECT_MESSAGE
    return None


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _is_overview_query(query: str) -> bool:
    """True if the user is asking for a high-level list (algorithms, features, etc.)."""
    q = query.lower().strip()
    if not q:
        return False
    patterns = [
        r"what\s+(quantum\s+)?algorithms",
        r"what\s+algorithms\s+does\s+divi",
        r"which\s+algorithms",
        r"what\s+features\s+does\s+divi",
        r"what\s+does\s+divi\s+support",
        r"list\s+(the\s+)?(algorithms|features|backends|optimizers)",
        r"which\s+(features|backends|optimizers)",
    ]
    return any(re.search(p, q) for p in patterns)


def _filter_chunks_for_overview(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """For overview queries, drop API ref and source code; keep all other docs."""
    exclude_dirs = ("api_reference",)
    result = []
    for c in chunks:
        path_lower = c.source_file.lower()
        if any(x in path_lower for x in exclude_dirs):
            continue
        if path_lower.endswith(".py"):
            continue
        result.append(c)
    return result


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context block.

    Confidence gating is done by the retriever (see
    :func:`divi.ai._retriever.retrieve`); an empty list means off-topic.
    """
    if not chunks:
        return "(No relevant documentation found.)"

    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        source = display_path(chunk.source_file)
        text = chunk.text
        normalized_source = chunk.source_file.replace("\\", "/")
        if "/tutorials/" in normalized_source or normalized_source.startswith(
            "tutorials/"
        ):
            text = text.replace(
                "from tutorials._backend import get_backend",
                "from divi.backends import MaestroSimulator",
            )
            text = re.sub(r"\bget_backend\(", "MaestroSimulator(", text)
        parts.append(f"[{i}] {source}:\n{text}")

    return "\n\n".join(parts)


def _format_relevant_imports(context: str) -> str:
    """Return compact import guidance for symbols present in the context."""
    meta = load_project_meta()
    if meta is None:
        return ""

    imports: list[str] = []
    for import_line in meta.get("import_lines", []):
        match = re.fullmatch(r"from (\S+) import (.+)", import_line)
        if match is None:
            continue
        module, raw_names = match.groups()
        names = []
        for raw_name in raw_names.split(","):
            name = raw_name.strip().split(" as ", maxsplit=1)[0]
            if name in {"Any", "TYPE_CHECKING"}:
                continue
            if re.search(rf"\b{re.escape(name)}\b", context):
                names.append(raw_name.strip())
        if names:
            imports.append(f"from {module} import {', '.join(names)}")

    if not imports:
        return ""
    formatted = "\n".join(f"- {line}" for line in imports)
    return (
        "\n\nRELEVANT IMPORT PATHS (grounding hints, not a category list):\n"
        f"{formatted}"
    )


def build_prompt(
    chunks: list[RetrievedChunk],
    history: list[dict[str, str]],
    user_query: str,
    llm: "Llama | None" = None,
) -> list[dict[str, str]]:
    """Build a ChatML message list for the LLM.

    Parameters
    ----------
    chunks:
        Retrieved context chunks from the vector index.
    history:
        Previous conversation turns as ``{"role": ..., "content": ...}``
        dicts.
    user_query:
        The current user message.
    llm:
        Llama instance used for token-budgeted history trimming.

    Returns
    -------
    list[dict[str, str]]
        A message list ready for ``Llama.create_chat_completion``.
    """
    # For "what algorithms/features does Divi support?" use overview guides
    # rather than API reference listings.
    if _is_overview_query(user_query):
        filtered = _filter_chunks_for_overview(chunks)
        if filtered:
            chunks = filtered
    context = _format_context(chunks)
    import_hints = _format_relevant_imports(context)
    system_content = f"{SYSTEM_PROMPT}{import_hints}\n\nCONTEXT:\n{context}"

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_content},
    ]
    if history and llm is not None:
        messages.extend(_trim_history(history, llm))
    messages.append({"role": "user", "content": user_query})

    return messages


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_stream(
    llm: Llama,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> Iterator[str]:
    """Stream tokens from the local LLM.

    Parameters
    ----------
    llm:
        A loaded ``llama_cpp.Llama`` instance.
    messages:
        The ChatML message list from :func:`build_prompt`.
    max_tokens:
        Maximum number of tokens to generate.
    temperature:
        Sampling temperature (0.0 = greedy, higher = more creative).

    Yields
    ------
    str
        Individual token strings as they are generated.
    """
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        stream=True,
    )

    for chunk in response:
        delta = chunk["choices"][0].get("delta", {})
        token = delta.get("content", "")
        if token:
            yield token
