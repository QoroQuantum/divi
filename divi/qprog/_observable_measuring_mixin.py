# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Mixin adding Pauli-observable measurement configuration knobs.

Provides ``grouping_strategy`` and ``shot_distribution`` kwargs to any
:class:`~divi.qprog.QuantumProgram` subclass that mixes it in. Programs
that hit the expval branch of
:class:`~divi.pipeline.stages.MeasurementStage` honour both knobs;
sampling-only programs hit the probs branch (which ignores both) and the
kwargs become silent no-ops.
"""

from warnings import warn

from divi.pipeline import GroupingStrategy, ShotDistStrategy

# Distinguishes "user accepted the default" from "user explicitly passed
# a value" so the override warning only fires for explicit choices.
_UNSET: object = object()

# User-facing subset of ``GroupingStrategy``.  ``"_backend_expval"`` is an
# internal MeasurementStage strategy and is excluded.
_ALLOWED_GROUPING_STRATEGIES: frozenset = frozenset({"qwc", "default", "wires", None})


class ObservableMeasuringMixin:
    """Mixin adding measurement-stage configuration to a quantum program.

    Mix in alongside :class:`~divi.qprog.QuantumProgram` (the mixin must
    come first so its cooperative ``super().__init__()`` resolves to the
    program base) — for example::

        class MyProgram(ObservableMeasuringMixin, QuantumProgram):
            ...

    The mixin stores the two measurement-stage knobs verbatim.  Runtime
    shape-aware decisions — including the auto-flip from ``"qwc"`` to
    the internal ``"_backend_expval"`` strategy when the backend supports
    analytic expvals and every MetaCircuit carries a single observable —
    live inside :class:`~divi.pipeline.stages.MeasurementStage`.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        from divi.qprog.quantum_program import QuantumProgram

        mro = cls.__mro__
        if ObservableMeasuringMixin in mro and QuantumProgram in mro:
            mixin_idx = mro.index(ObservableMeasuringMixin)
            base_idx = mro.index(QuantumProgram)
            if mixin_idx > base_idx:
                raise TypeError(
                    f"{cls.__name__}: ObservableMeasuringMixin must precede "
                    f"QuantumProgram in the base list so the mixin's "
                    f"__init__ runs before QuantumProgram's strict-kwargs "
                    f"check rejects ``grouping_strategy=`` / "
                    f"``shot_distribution=``."
                )

    def __init__(
        self,
        *args,
        grouping_strategy: GroupingStrategy | None = _UNSET,  # type: ignore[assignment]
        shot_distribution: ShotDistStrategy | None = None,
        **kwargs,
    ):
        """Initialize the measurement-config layer.

        Args:
            grouping_strategy: Strategy for partitioning Hamiltonian terms
                into compatible measurement groups; one circuit is
                executed per group. Options: ``"qwc"`` (qubit-wise-
                commuting — most compact, default), ``"wires"`` (group
                by support wires), or ``None`` (one circuit per term).
                :class:`~divi.pipeline.stages.MeasurementStage` may
                further auto-flip ``"qwc"`` to its internal
                ``"_backend_expval"`` mode at runtime when the backend
                supports analytic expvals and the batch is single-
                observable; ``"_backend_expval"`` is not user-passable.
            shot_distribution: Focus the backend's shot budget on the
                Hamiltonian terms that matter most. Without this option,
                every measurement group is sampled with the backend's
                full shot count, even tiny terms with little impact on
                the final energy. With ``shot_distribution`` set, the
                same total budget is split across groups according to
                their importance — reducing variance without spending
                more shots.

                Available strategies:

                - ``"uniform"`` — equal split across groups.
                - ``"weighted"`` — proportional to per-group coefficient
                  L1 norm; dominant Hamiltonian terms get more shots.
                - ``"weighted_random"`` — multinomial sample of the same
                  probabilities; may drop more low-weight groups than
                  the deterministic ``"weighted"`` for the same budget.
                - A callable
                  ``(group_l1_norms, total_shots) -> per_group_shots``
                  for fully custom allocation.

                Defaults to ``None`` (every group receives the full shot
                budget).
            ``*args``, ``**kwargs``: Forwarded to the next class in the
                MRO (typically :class:`~divi.qprog.QuantumProgram`).
        """
        super().__init__(*args, **kwargs)

        if (
            grouping_strategy is not _UNSET
            and grouping_strategy not in _ALLOWED_GROUPING_STRATEGIES
        ):
            raise ValueError(
                f"Invalid grouping_strategy={grouping_strategy!r}. "
                f"Choose 'qwc', 'wires', 'default', or None."
            )

        if (
            grouping_strategy not in (_UNSET, None)
            and shot_distribution is None
            and self.backend.supports_expval  # type: ignore[attr-defined]
        ):
            warn(
                "Backend supports analytic expectation values; "
                f"grouping_strategy={grouping_strategy!r} may be auto-"
                "overridden to use the backend's native expval path for "
                "single-observable expval-mode circuits.",
                UserWarning,
                stacklevel=2,
            )

        self._grouping_strategy: GroupingStrategy = (
            "qwc" if grouping_strategy is _UNSET else grouping_strategy
        )
        self._shot_distribution = shot_distribution
