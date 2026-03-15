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
from ._types import short_source

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_BASE = """\
You are **divi-ai**, a helpful coding assistant for the Divi quantum \
computing library by Qoro Quantum.

SCOPE — Answer only about Divi, quantum computing with Divi, or related \
Python. If the question is off-topic (unrelated to Divi), reply only: \
"I can only help with the Divi quantum computing library." \
If the question is about Divi but CONTEXT does not contain the answer, \
say you don't have enough information and point to \
https://divi.readthedocs.io or github.com/QoroQuantum/divi/issues.

ACCURACY — Base answers ONLY on the CONTEXT below. Do not invent APIs. \
Only name algorithms, features, or backends that appear in CONTEXT; do not \
add names that are not mentioned there. Answer directly without echoing the \
question. Use clear, complete sentences; when listing items, introduce them \
briefly (e.g. "Divi supports the following optimizers: …") rather than \
bare comma-separated fragments. For "what does Divi support?" style \
questions, list only the main items from CONTEXT, not every class or config. \
When asked specifically for **algorithms**, list only the top-level algorithm \
names: VQE, QAOA, and time evolution. Do not list ansatzes (e.g. Hartree-Fock, \
UCCSD), optimizers (e.g. Monte Carlo, PCE), or other building blocks.

REDIRECT — For questions about real hardware or third-party quantum cloud \
providers, direct the user to **QoroService** (Divi's cloud offering) and \
suggest they contact Qoro. Do not provide setup or code for other providers.
"""


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

    # Import map
    import_lines = meta.get("import_lines", [])
    if import_lines:
        formatted = "\n".join(f"- {line}" for line in import_lines)
        parts.append(
            "KEY IMPORTS (use these exact paths when writing code):\n" + formatted
        )

    return "\n\n".join(parts) + "\n"


SYSTEM_PROMPT = _build_system_prompt()

# Chunks with raw cosine similarity below this threshold are dropped.
# Off-topic queries ("What is a cat?") peak around 0.54-0.58;
# the lowest valid query ("QAOA max clique") hits 0.59.
MIN_DENSE_SCORE = 0.55

# Token budget for conversation history.  Keeps prompt size predictable
# regardless of whether responses are short Q&A or long code blocks.
MAX_HISTORY_TOKENS = 2048


def _trim_history(
    history: list[dict[str, str]],
    llm: Llama,
    max_tokens: int = MAX_HISTORY_TOKENS,
) -> list[dict[str, str]]:
    """Keep newest messages that fit within *max_tokens*.

    Walks the history from newest to oldest, accumulating token counts.
    Stops when adding another message would exceed the budget.  Always
    keeps messages in pairs (user + assistant) to avoid orphaned turns.
    """
    if not history:
        return []

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
    "azure",
    "aws braket",
    "aws",
    "ibm quantum",
    "ibm q",
    "ibm",
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
    """For overview queries, keep only user-guide/quickstart chunks; drop API ref and .py."""
    overview_dirs = ("user_guide", "quickstart", "core_concepts", "tutorials")
    exclude = ("api_reference",)
    result = []
    for c in chunks:
        path_lower = c.source_file.lower()
        if any(x in path_lower for x in exclude):
            continue
        if path_lower.endswith(".py"):
            continue
        if any(x in path_lower for x in overview_dirs):
            result.append(c)
    return result


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context block."""
    # Drop chunks whose cosine similarity is below threshold.
    relevant = [c for c in chunks if c.dense_score >= MIN_DENSE_SCORE]

    if not relevant:
        return "(No relevant documentation found.)"

    parts: list[str] = []
    for i, chunk in enumerate(relevant, start=1):
        source = short_source(chunk.source_file)
        parts.append(f"[{i}] {source}:\n{chunk.text}")

    return "\n\n".join(parts)


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
    # For "what algorithms/features does Divi support?" use only user-guide
    # chunks so the model sees overview content, not API reference listings.
    if _is_overview_query(user_query):
        filtered = _filter_chunks_for_overview(chunks)
        if filtered:
            chunks = filtered
    context = _format_context(chunks)
    system_content = f"{SYSTEM_PROMPT}\nCONTEXT:\n{context}"

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
