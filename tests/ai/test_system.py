# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import pytest

from divi.ai._system import detect_arch, detect_cpu_threads, detect_ram_gb


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
def test_arch_detection(mocker, machine, system, expected):
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
        assert result == 16.0

    def test_returns_none_on_failure(self, mocker):
        mocker.patch(
            "divi.ai._system.psutil.virtual_memory",
            side_effect=RuntimeError("no psutil"),
        )
        assert detect_ram_gb() is None


class TestDetectCpuThreads:
    def test_uses_physical_core_count(self, mocker):
        mocker.patch(
            "divi.ai._system.psutil.cpu_count",
            side_effect=lambda logical: 32 if logical else 16,
        )
        process = mocker.patch("divi.ai._system.psutil.Process").return_value
        process.cpu_affinity.return_value = list(range(32))

        assert detect_cpu_threads() == 16

    def test_respects_process_affinity(self, mocker):
        mocker.patch(
            "divi.ai._system.psutil.cpu_count",
            side_effect=lambda logical: 32 if logical else 16,
        )
        process = mocker.patch("divi.ai._system.psutil.Process").return_value
        process.cpu_affinity.return_value = list(range(8))

        assert detect_cpu_threads() == 8

    def test_falls_back_to_logical_count(self, mocker):
        mocker.patch(
            "divi.ai._system.psutil.cpu_count",
            side_effect=lambda logical: 6 if logical else None,
        )
        mocker.patch(
            "divi.ai._system.psutil.Process",
            side_effect=RuntimeError("affinity unavailable"),
        )

        assert detect_cpu_threads() == 6
