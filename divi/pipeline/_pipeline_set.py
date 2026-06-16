# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Named registry of the auxiliary pipelines a program exposes."""

from collections.abc import Callable, ItemsView, Iterator
from typing import Any

from divi.pipeline._core import CircuitPipeline


class PipelineSet:
    """The named pipelines a program runs over its ansatz.

    A thin, typed wrapper over a ``name -> (CircuitPipeline, spec_factory)``
    mapping. Programs define entries in ``_build_pipelines`` (``"cost"`` drives
    optimization, ``"sample"`` extracts the solution distribution); callers reach
    the pipeline they need by name via ``__getitem__`` and its initial spec via
    :meth:`spec_for`. Registering both together keeps a pipeline and its seed from
    drifting apart; subclasses and mixins extend a set cooperatively with
    :meth:`with_`.
    """

    def __init__(
        self, pipelines: dict[str, tuple[CircuitPipeline, Callable[[], Any]]]
    ) -> None:
        self._pipelines = {name: pipe for name, (pipe, _) in pipelines.items()}
        self._spec_factories = {name: spec for name, (_, spec) in pipelines.items()}

    def with_(
        self, name: str, pipeline: CircuitPipeline, spec_factory: Callable[[], Any]
    ) -> "PipelineSet":
        """Return a new set with ``name`` added (or replaced) — for cooperative
        ``super()._build_pipelines().with_(...)`` extension by mixins/subclasses."""
        specs = {n: (p, self._spec_factories[n]) for n, p in self._pipelines.items()}
        specs[name] = (pipeline, spec_factory)
        return PipelineSet(specs)

    def _missing(self, name: str) -> KeyError:
        return KeyError(
            f"No {name!r} pipeline; this program exposes {sorted(self._pipelines)}."
        )

    def __getitem__(self, name: str) -> CircuitPipeline:
        try:
            return self._pipelines[name]
        except KeyError:
            raise self._missing(name) from None

    def get(self, name: str) -> CircuitPipeline | None:
        return self._pipelines.get(name)

    def spec_for(self, name: str) -> Any:
        """Resolve the initial spec seeding pipeline ``name`` (calls its factory)."""
        try:
            factory = self._spec_factories[name]
        except KeyError:
            raise self._missing(name) from None
        return factory()

    def __contains__(self, name: object) -> bool:
        return name in self._pipelines

    def __iter__(self) -> Iterator[str]:
        return iter(self._pipelines)

    def __len__(self) -> int:
        return len(self._pipelines)

    def items(self) -> ItemsView[str, CircuitPipeline]:
        return self._pipelines.items()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({sorted(self._pipelines)})"
