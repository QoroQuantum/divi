# Copyright 2018 The Cirq Developers
# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


class QasmException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
