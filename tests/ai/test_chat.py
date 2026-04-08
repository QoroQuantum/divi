# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from divi.ai._chat import (
    HARDWARE_REDIRECT_MESSAGE,
    MIN_DENSE_SCORE,
    _filter_chunks_for_overview,
    _format_context,
    _is_overview_query,
    _trim_history,
    build_prompt,
    generate_stream,
    get_hardware_redirect_response,
)
from divi.ai._retriever import RetrievedChunk


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

    def test_no_match_returns_none(self):
        assert get_hardware_redirect_response("How do I run VQE?") is None

    def test_empty_query_returns_none(self):
        assert get_hardware_redirect_response("") is None

    def test_whitespace_only_returns_none(self):
        assert get_hardware_redirect_response("   ") is None


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


class TestFilterChunksForOverview:
    def test_keeps_user_guide_chunks(self):
        chunks = [
            RetrievedChunk(
                text="guide content",
                source_file="/repo/docs/user_guide/vqe.rst",
                start_line=1,
                end_line=10,
                score=0.8,
                dense_score=0.8,
            ),
        ]
        assert len(_filter_chunks_for_overview(chunks)) == 1

    def test_drops_api_reference(self):
        chunks = [
            RetrievedChunk(
                text="api ref",
                source_file="/repo/docs/api_reference/qprog.rst",
                start_line=1,
                end_line=10,
                score=0.8,
                dense_score=0.8,
            ),
        ]
        assert len(_filter_chunks_for_overview(chunks)) == 0

    def test_drops_py_files(self):
        chunks = [
            RetrievedChunk(
                text="python source",
                source_file="/repo/divi/qprog/vqe.py",
                start_line=1,
                end_line=10,
                score=0.8,
                dense_score=0.8,
            ),
        ]
        assert len(_filter_chunks_for_overview(chunks)) == 0

    def test_keeps_other_doc_directories(self):
        chunks = [
            RetrievedChunk(
                text="tools content",
                source_file="/repo/docs/source/tools/divi_ai.rst",
                start_line=1,
                end_line=10,
                score=0.8,
                dense_score=0.8,
            ),
            RetrievedChunk(
                text="development guide",
                source_file="/repo/docs/source/development/contributing.rst",
                start_line=1,
                end_line=10,
                score=0.7,
                dense_score=0.7,
            ),
        ]
        assert len(_filter_chunks_for_overview(chunks)) == 2

    def test_keeps_tutorial_chunks(self):
        chunks = [
            RetrievedChunk(
                text="tutorial content",
                source_file="/repo/tutorials/vqe.rst",
                start_line=1,
                end_line=10,
                score=0.8,
                dense_score=0.8,
            ),
        ]
        assert len(_filter_chunks_for_overview(chunks)) == 1


class TestFormatContext:
    def test_numbers_chunks(self, sample_retrieved_chunks):
        result = _format_context(sample_retrieved_chunks)
        assert "[1]" in result
        assert "[2]" in result

    def test_filters_below_min_dense_score(self, sample_retrieved_chunks):
        result = _format_context(sample_retrieved_chunks)
        # The third chunk (score 0.50) is below MIN_DENSE_SCORE (0.55)
        assert sample_retrieved_chunks[2].dense_score < MIN_DENSE_SCORE
        assert "cats" not in result

    def test_keeps_above_threshold(self, sample_retrieved_chunks):
        result = _format_context(sample_retrieved_chunks)
        assert "VQE" in result
        assert "QAOA" in result

    def test_all_below_threshold(self):
        below = MIN_DENSE_SCORE - 0.1
        chunks = [
            RetrievedChunk(
                text="irrelevant",
                source_file="f.py",
                start_line=1,
                end_line=1,
                score=below,
                dense_score=below,
            ),
        ]
        assert _format_context(chunks) == "(No relevant documentation found.)"

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
        result = _trim_history(history, mock_llm)
        assert len(result) % 2 == 0

    def test_respects_explicit_max_tokens(self, mock_llm):
        history = [
            {"role": "user", "content": "old_" + "x" * 1000},
            {"role": "assistant", "content": "old_" + "y" * 1000},
            {"role": "user", "content": "new_short"},
            {"role": "assistant", "content": "new_short"},
        ]
        # Small budget → should drop the large old pair, keep the short new pair
        result = _trim_history(history, mock_llm, max_tokens=20)
        assert len(result) == 2
        assert result[-1] == history[-1]


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
        """Overview queries should filter to user-guide/tutorial chunks only."""
        chunks = [
            RetrievedChunk(
                text="guide content about algorithms",
                source_file="/repo/docs/user_guide/core_concepts.rst",
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
