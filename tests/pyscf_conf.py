# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""PySCF configuration for the test suite, read through ``PYSCF_CONFIG_FILE``."""

# Each mean field would hold a temporary checkpoint file open until garbage
# collection, which then warns in whatever test is running; no test reads one.
scf_hf_SCF_mute_chkfile = True
