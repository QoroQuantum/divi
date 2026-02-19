# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared backend factory for tutorials.

Parses ``--local`` / ``--maestro`` from the command line and provides a
backend constructor so each tutorial can remain backend-agnostic.

Usage in a tutorial::

    from tutorials._backend import get_backend

    backend = get_backend(shots=5000)
"""

import argparse

from divi.backends import CircuitRunner, JobConfig, ParallelSimulator, QoroService


def _parse_mode() -> str:
    """Parse --local / --maestro from sys.argv without interfering with the tutorial."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--local",
        dest="mode",
        action="store_const",
        const="local",
    )
    parser.add_argument(
        "--maestro",
        dest="mode",
        action="store_const",
        const="maestro",
    )
    args, _ = parser.parse_known_args()

    return args.mode or "local"


def get_backend(
    *, shots: int = 5000, track_depth: bool = False, **kwargs
) -> CircuitRunner:
    """Create a backend based on the CLI flag.

    Args:
        shots: Number of measurement shots (used by both backends).
        track_depth: If True, record circuit depth for each submitted batch.
        **kwargs: Extra keyword arguments forwarded to ``ParallelSimulator``
            (e.g. ``n_processes``, ``qiskit_backend``).
            These are silently ignored when ``--maestro`` is selected.

    Returns:
        A ``ParallelSimulator`` (``--local``, default) or
        ``QoroService`` (``--maestro``) instance.
    """
    mode = _parse_mode()

    if mode == "maestro":
        config = JobConfig(shots=shots, qpu_system="qoro_maestro")
        service = QoroService(config=config)
        service.track_depth = track_depth
        return service

    return ParallelSimulator(shots=shots, track_depth=track_depth, **kwargs)
