# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared backend factory for tutorials.

Parses ``--local`` / ``--maestro`` / ``--maestro-local`` and ``--force-sampling``
from the command line and provides a backend constructor so each tutorial can
remain backend-agnostic.

Usage in a tutorial::

    from tutorials._backend import get_backend

    backend = get_backend(shots=5000)
"""

import argparse
import os

from divi.backends import (
    CircuitRunner,
    JobConfig,
    MaestroSimulator,
    ParallelSimulator,
    QoroService,
)


def _parse_args() -> argparse.Namespace:
    """Parse CLI flags from sys.argv without interfering with the tutorial."""
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
    parser.add_argument(
        "--maestro-local",
        dest="mode",
        action="store_const",
        const="maestro-local",
    )
    parser.add_argument(
        "--force-sampling",
        action="store_true",
        default=False,
    )
    args, _ = parser.parse_known_args()
    args.mode = args.mode or "local"

    return args


def get_backend(
    *,
    shots: int = 5000,
    track_depth: bool = False,
    force_sampling: bool = False,
    **kwargs,
) -> CircuitRunner:
    """Create a backend based on the CLI flag.

    Args:
        shots: Number of measurement shots (used by both backends).
        track_depth: If True, record circuit depth for each submitted batch.
        force_sampling: If True, disable expval mode on ``ParallelSimulator``
            and ``QoroService``, forcing shot-based sampling instead.
            Ignored for ``MaestroSimulator``.
        **kwargs: Extra keyword arguments forwarded to ``ParallelSimulator``
            (e.g. ``n_processes``, ``qiskit_backend``).
            These are silently ignored when ``--maestro`` or
            ``--maestro-local`` is selected.

    Returns:
        A ``ParallelSimulator`` (``--local``, default),
        ``MaestroSimulator`` (``--maestro-local``), or
        ``QoroService`` (``--maestro``) instance.
    """
    ci_max_shots = os.environ.get("DIVI_CI_MAX_SHOTS")
    if ci_max_shots is not None:
        shots = min(shots, int(ci_max_shots))

    cli = _parse_args()
    force_sampling = force_sampling or cli.force_sampling

    if cli.mode == "maestro":
        config = JobConfig(shots=shots, simulator_cluster="qoro_maestro")
        service = QoroService(job_config=config, force_sampling=force_sampling)
        service.track_depth = track_depth
        return service

    if cli.mode == "maestro-local":
        return MaestroSimulator(shots=shots, track_depth=track_depth)

    return ParallelSimulator(
        shots=shots, track_depth=track_depth, force_sampling=force_sampling, **kwargs
    )
