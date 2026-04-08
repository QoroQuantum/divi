# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import pytest

from divi.ai._system import detect_arch, detect_ram_gb


class TestDetectArch:
    @pytest.mark.parametrize(
        "machine,system,expected",
        [
            ("arm64", "Darwin", "apple_silicon"),
            ("aarch64", "Darwin", "apple_silicon"),
            ("aarch64", "Linux", "arm64"),
            ("arm64", "Linux", "arm64"),
            ("x86_64", "Linux", "x86_64"),
            ("x86_64", "Darwin", "x86_64"),
            ("amd64", "Windows", "x86_64"),
            ("riscv64", "Linux", "riscv64"),
        ],
    )
    def test_arch_detection(self, mocker, machine, system, expected):
        mocker.patch("divi.ai._system.platform.machine", return_value=machine)
        mocker.patch("divi.ai._system.platform.system", return_value=system)
        assert detect_arch() == expected


class TestDetectRamGb:
    def test_returns_float(self, mocker):
        VMemory = namedtuple("VMemory", ["total"])
        mocker.patch(
            "divi.ai._system.psutil.virtual_memory",
            return_value=VMemory(total=16 * 1024**3),
        )
        result = detect_ram_gb()
        assert isinstance(result, float)
        assert result == pytest.approx(16.0, abs=0.01)

    def test_returns_none_on_failure(self, mocker):
        mocker.patch(
            "divi.ai._system.psutil.virtual_memory",
            side_effect=RuntimeError("no psutil"),
        )
        assert detect_ram_gb() is None
