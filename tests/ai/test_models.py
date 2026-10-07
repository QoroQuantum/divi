# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from divi.ai._models import AVAILABLE_MODELS, get_recommended_models, load_llm


class TestGetRecommendedModels:
    @pytest.mark.parametrize(
        "arch,ram,expected",
        [
            ("apple_silicon", 16.0, {"4b", "9b"}),
            ("apple_silicon", 32.0, {"4b", "9b"}),
            ("apple_silicon", 8.0, {"4b"}),
            ("x86_64", 32.0, {"4b", "9b"}),
            ("x86_64", 64.0, {"4b", "9b"}),
            ("x86_64", 16.0, {"4b", "9b"}),
            ("x86_64", 8.0, {"4b"}),
            ("arm64", 8.0, {"4b"}),
            ("arm64", 16.0, {"4b", "9b"}),
        ],
    )
    def test_recommendations(self, arch, ram, expected):
        assert get_recommended_models(arch, ram) == expected

    def test_none_ram_returns_empty(self):
        assert get_recommended_models("x86_64", None) == set()

    def test_all_recommended_keys_are_valid(self):
        """Every recommended model key must exist in AVAILABLE_MODELS."""
        for arch in ("apple_silicon", "x86_64", "arm64"):
            for ram in (8.0, 16.0, 32.0):
                recommended = get_recommended_models(arch, ram)
                for key in recommended:
                    assert key in AVAILABLE_MODELS, f"{key} not in AVAILABLE_MODELS"


def test_load_llm_uses_detected_cpu_threads(mocker, tmp_path):
    mocker.patch("divi.ai._models.detect_cpu_threads", return_value=6)
    llama = mocker.patch("divi.ai._models.Llama")
    model_path = tmp_path / "model.gguf"

    load_llm(model_path, n_ctx=4096, debug=True)

    llama.assert_called_once_with(
        model_path=str(model_path),
        n_ctx=4096,
        n_threads=6,
        n_threads_batch=6,
        verbose=False,
    )
