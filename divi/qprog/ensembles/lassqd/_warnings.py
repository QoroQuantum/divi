# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Attribution of a warning to the first frame outside divi."""

import os
import sys
from pathlib import Path

# Frames under this prefix are divi's own, never the code a warning is for.
_DIVI_PREFIX = os.path.join(Path(__file__).parents[3], "")


def _outside_divi_stacklevel() -> int:
    """``stacklevel`` for a warning raised by this function's caller that names
    the first frame outside divi, however deep the call path into divi is."""
    frame = sys._getframe(1)
    stacklevel = 1
    while frame is not None and frame.f_code.co_filename.startswith(_DIVI_PREFIX):
        frame = frame.f_back
        stacklevel += 1
    return stacklevel
