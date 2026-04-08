# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from divi.ai._models import AVAILABLE_MODELS, get_recommended_models


class TestGetRecommendedModels:
    @pytest.mark.parametrize(
        "arch,ram,expected",
        [
            ("apple_silicon", 16.0, {"7b", "14b"}),
            ("apple_silicon", 32.0, {"7b", "14b"}),
            ("apple_silicon", 8.0, {"1.5b", "3b", "e2b", "e4b", "7b"}),
            ("x86_64", 32.0, {"7b", "14b"}),
            ("x86_64", 64.0, {"7b", "14b"}),
            ("x86_64", 16.0, {"e4b", "7b", "14b"}),
            ("x86_64", 8.0, {"1.5b", "3b", "e2b"}),
            ("arm64", 8.0, {"1.5b", "3b", "e2b"}),
            ("arm64", 16.0, {"e4b", "7b", "14b"}),
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
