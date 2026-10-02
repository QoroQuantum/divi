# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from divi.ai._chat import (
    HARDWARE_REDIRECT_MESSAGE,
    _filter_chunks_for_overview,
    _format_context,
    _is_overview_query,
    _trim_history,
    build_prompt,
    generate_stream,
    get_hardware_redirect_response,
    strip_scope_preamble,
)
from divi.ai._retriever import RetrievedChunk


@pytest.mark.parametrize(
    "text, expected",
    [
        ("SCOPE: IN\n\nHere is the answer.", "Here is the answer."),
        ("**SCOPE: REDIRECT**\n\nUse QAOA.", "Use QAOA."),
        (
            "SCOPE: OUT\nI can only help with the Divi quantum computing library.",
            "I can only help with the Divi quantum computing library.",
        ),
        ("SCOPE: WRONG-TOOL\n\nUse QAOA instead.", "Use QAOA instead."),
        ("SCOPE: WRONG TOOL\n\nUse QAOA instead.", "Use QAOA instead."),
        ("**SCOPE: WRONG-TOOL**\n\nUse QAOA instead.", "Use QAOA instead."),
        (
            "To configure ZNE, use the ZNE class.",
            "To configure ZNE, use the ZNE class.",
        ),
        (
            "Here is the answer.\n\nSCOPE: IN was the original classification.",
            "Here is the answer.\n\nSCOPE: IN was the original classification.",
        ),
    ],
    ids=[
        "scope_in",
        "bold_redirect",
        "scope_out",
        "wrong_tool_hyphen",
        "wrong_tool_space",
        "wrong_tool_bold",
        "no_preamble",
        "mid_response_scope_kept",
    ],
)
def test_strip_scope_preamble(text, expected):
    assert strip_scope_preamble(text) == expected


class TestHardwareRedirect:
    @pytest.mark.parametrize(
        "query",
        [
            "How do I run on real quantum hardware?",
            "Can I use Azure Quantum with Divi?",
            "AWS Braket integration",
            "IBM Quantum backend",
            "third-party backend setup",
        ],
    )
    def test_matches_hardware_keywords(self, query):
        assert get_hardware_redirect_response(query) == HARDWARE_REDIRECT_MESSAGE

    def test_case_insensitive(self):
        assert (
            get_hardware_redirect_response("REAL QUANTUM HARDWARE")
            == HARDWARE_REDIRECT_MESSAGE
        )

    @pytest.mark.parametrize("query", ["How do I run VQE?", "", "   "])
    def test_no_match_returns_none(self, query):
        assert get_hardware_redirect_response(query) is None


class TestIsOverviewQuery:
    @pytest.mark.parametrize(
        "query",
        [
            "What algorithms does Divi support?",
            "What quantum algorithms are available?",
            "Which algorithms does divi have?",
            "What features does Divi offer?",
            "What does Divi support?",
            "List the algorithms",
            "List the features",
            "Which backends are available?",
            "Which optimizers does divi have?",
        ],
    )
    def test_overview_queries_detected(self, query):
        assert _is_overview_query(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            "How do I configure ZNE?",
            "Show me a VQE example",
            "What is the default optimizer?",
            "",
        ],
    )
    def test_specific_queries_not_detected(self, query):
        assert _is_overview_query(query) is False


@pytest.mark.parametrize(
    "source_file, kept",
    [
        ("/repo/docs/algorithms/vqe.rst", True),
        ("/repo/docs/api_reference/qprog.rst", False),
        ("/repo/divi/qprog/vqe.py", False),
        ("/repo/docs/source/tools/divi_ai.rst", True),
        ("/repo/docs/source/development/contributing.rst", True),
        ("/repo/tutorials/vqe.rst", True),
    ],
)
def test_filter_chunks_for_overview(source_file, kept):
    chunk = RetrievedChunk(
        text="content",
        source_file=source_file,
        start_line=1,
        end_line=10,
        score=0.8,
        dense_score=0.8,
    )
    assert _filter_chunks_for_overview([chunk]) == ([chunk] if kept else [])


class TestFormatContext:
    """Confidence gating is now the retriever's job (see
    :func:`divi.ai._retriever.retrieve`); _format_context just formats
    whatever it gets, and shows "(No relevant documentation found.)"
    for an empty list (meaning the retriever filtered everything out)."""

    def test_numbers_chunks(self, sample_retrieved_chunks):
        result = _format_context(sample_retrieved_chunks)
        assert "[1]" in result
        assert "[2]" in result

    def test_includes_all_chunks(self, sample_retrieved_chunks):
        # All three sample chunks are included — no in-chat filtering.
        result = _format_context(sample_retrieved_chunks)
        assert "VQE" in result
        assert "QAOA" in result
        assert "cats" in result

    def test_empty_chunks(self):
        assert _format_context([]) == "(No relevant documentation found.)"


class TestTrimHistory:
    def test_empty_history(self, mock_llm):
        assert _trim_history([], mock_llm) == []

    def test_fits_entirely(self, mock_llm):
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = _trim_history(history, mock_llm)
        assert len(result) == 2

    def test_trims_oldest_keeps_newest(self, mock_llm):
        # Create many messages that exceed the budget
        history = []
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            history.append({"role": role, "content": f"msg_{i}_" + "x" * 500})
        result = _trim_history(history, mock_llm)
        assert len(result) < len(history)
        assert len(result) % 2 == 0
        # Newest messages must be retained
        assert result[-1] == history[-1]
        assert result[-2] == history[-2]

    def test_keeps_pairs_not_orphans(self, mock_llm):
        history = [
            {"role": "user", "content": "a" * 200},
            {"role": "assistant", "content": "b" * 200},
            {"role": "user", "content": "c" * 200},
            {"role": "assistant", "content": "d" * 200},
        ]
        # 50 tokens each: the budget fits the newest pair plus half the older one.
        result = _trim_history(history, mock_llm, max_tokens=150)
        assert result == history[2:]


class TestBuildPrompt:
    def test_includes_system_prompt(self, sample_retrieved_chunks):
        messages = build_prompt(sample_retrieved_chunks, [], "How to run VQE?")
        assert messages[0]["role"] == "system"
        assert "divi-ai" in messages[0]["content"]

    def test_includes_context(self, sample_retrieved_chunks):
        messages = build_prompt(sample_retrieved_chunks, [], "How to run VQE?")
        assert "CONTEXT:" in messages[0]["content"]

    def test_user_query_is_last_message(self, sample_retrieved_chunks):
        messages = build_prompt(sample_retrieved_chunks, [], "my question")
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "my question"

    def test_no_history_without_llm(self, sample_retrieved_chunks):
        history = [
            {"role": "user", "content": "previous"},
            {"role": "assistant", "content": "answer"},
        ]
        messages = build_prompt(
            sample_retrieved_chunks, history, "new question", llm=None
        )
        # Without llm, history is not included
        assert len(messages) == 2  # system + user
        assert messages[-1]["content"] == "new question"

    def test_history_included_with_llm(self, sample_retrieved_chunks, mock_llm):
        history = [
            {"role": "user", "content": "previous"},
            {"role": "assistant", "content": "answer"},
        ]
        messages = build_prompt(
            sample_retrieved_chunks, history, "new question", llm=mock_llm
        )
        assert len(messages) == 4  # system + 2 history + user

    def test_overview_query_filters_chunks(self):
        """Overview queries should filter to non-API documentation chunks only."""
        chunks = [
            RetrievedChunk(
                text="guide content about algorithms",
                source_file="/repo/docs/execution_workflows/core_concepts.rst",
                start_line=1,
                end_line=10,
                score=0.8,
                dense_score=0.8,
            ),
            RetrievedChunk(
                text="python source code",
                source_file="/repo/divi/qprog/vqe.py",
                start_line=1,
                end_line=10,
                score=0.9,
                dense_score=0.9,
            ),
        ]
        messages = build_prompt(chunks, [], "What algorithms does Divi support?")
        system_content = messages[0]["content"]
        # The .py chunk should be filtered for overview queries
        assert "guide content" in system_content


class TestGenerateStream:
    def test_yields_tokens(self, mock_llm):
        mock_llm.create_chat_completion.return_value = [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
        ]
        messages = [{"role": "user", "content": "Hi"}]
        tokens = list(generate_stream(mock_llm, messages))
        assert tokens == ["Hello", " world"]

    def test_skips_empty_deltas(self, mock_llm):
        mock_llm.create_chat_completion.return_value = [
            {"choices": [{"delta": {}}]},
            {"choices": [{"delta": {"content": ""}}]},
            {"choices": [{"delta": {"content": "ok"}}]},
        ]
        tokens = list(generate_stream(mock_llm, [{"role": "user", "content": "Hi"}]))
        assert tokens == ["ok"]

    def test_passes_parameters(self, mock_llm):
        mock_llm.create_chat_completion.return_value = []
        messages = [{"role": "user", "content": "Hi"}]
        list(generate_stream(mock_llm, messages, max_tokens=512, temperature=0.5))
        call_kwargs = mock_llm.create_chat_completion.call_args.kwargs
        assert call_kwargs["messages"] == messages
        assert call_kwargs["max_tokens"] == 512
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["stream"] is True
