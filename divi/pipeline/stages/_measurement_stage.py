# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Callable, Hashable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from typing import Any, get_args

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import measurement_qasms_from_groups
from divi.circuits._conversions import _sparse_pauli_op_to_ham_string
from divi.pipeline._grouping import (
    BACKEND_EXPVAL,
    BACKEND_EXPVAL_GROUPS,
    GroupingStrategy,
    _compute_measurement_groups,
    _create_postprocessing_fn,
    _ResolvedGroupingStrategy,
    _union_labels,
)
from divi.pipeline._postprocessing import (
    _counts_to_cost_variance,
    _counts_to_wrs_cost_variance,
)
from divi.pipeline._result_keys_operations import (
    PARAM_SET_AXIS,
    group_by_base_key,
    reduce_postprocess_ordered,
    strip_axis_from_label,
)
from divi.pipeline._shadows import (
    _count_unique_rows,
    _sample_shadow_covers,
    _shadow_term_weights,
)
from divi.pipeline._shot_distribution import (
    ShotDistStrategy,
    _compute_group_l1_norms,
    _compute_shot_distribution,
)
from divi.pipeline.abc import (
    BundleStage,
    ChildResults,
    MetaCircuitBatch,
    PipelineEnv,
    ResultFormat,
    StageOutput,
    StageToken,
)

OBS_GROUP_AXIS = "obs_group"
PROBS_MEAS_AXIS = "meas"
# ``GroupingStrategy`` is ``Literal[...] | None``; unwrap the literal's names.
_GROUPING_STRATEGY_NAMES: tuple[str, ...] = get_args(get_args(GroupingStrategy)[0])


def _split_identity_offsets(
    observables: Sequence[SparsePauliOp],
) -> tuple[tuple[SparsePauliOp, ...], tuple[float, ...]]:
    """Remove identity terms and return each observable's exact real-valued offset."""
    stripped: list[SparsePauliOp] = []
    offsets: list[float] = []
    for observable in observables:
        identity = ~(observable.paulis.x | observable.paulis.z).any(axis=1)
        offsets.append(float(np.asarray(observable.coeffs).real[identity].sum()))
        stripped.append(observable[np.flatnonzero(~identity)])
    if not any(len(observable) for observable in stripped):
        raise ValueError(
            "The observable is identity-only, so its expectation is an exact "
            "constant and there is nothing to sample."
        )
    return tuple(stripped), tuple(offsets)


@dataclass(frozen=True)
class _GroupShotPlan:
    """Shot allocation plus the adjustments needed during reduction."""

    surviving_indices: list[int]
    shots_by_group: dict[int, int] | None
    missing_group_results: dict[int, object] = field(default_factory=dict)
    scales_by_group: dict[int, float] = field(default_factory=dict)
    term_weights_by_group: dict[int, np.ndarray] = field(default_factory=dict)
    """Inverse inclusion probability of each group's terms, aligned with the
    group; empty means all ones."""
    random_group_selection: bool = False
    """Whether groups are drawn at random, making selection part of the noise."""
    empty_draws: int = 0
    """Random draws that measured no term: zero-valued samples never submitted."""


def _scale_group_result(
    result: Any, scale: float, term_weights: np.ndarray | None
) -> Any:
    """Scale a scalar or indexed group expectation result, term by term."""
    if term_weights is None:
        if scale == 1.0:
            return result
        term_weights = np.ones(len(result) if isinstance(result, dict) else 1)
    if isinstance(result, dict):
        return {idx: scale * term_weights[idx] * value for idx, value in result.items()}
    return scale * term_weights[0] * result


def _apply_group_shot_plan(
    group_results: dict[int, Any], plan: _GroupShotPlan | None
) -> dict[int, Any]:
    """Apply missing-group defaults and estimator scales in one pass."""
    if plan is None:
        return group_results
    measured_results = {
        group_idx: _scale_group_result(
            result,
            plan.scales_by_group.get(group_idx, 1.0),
            plan.term_weights_by_group.get(group_idx),
        )
        for group_idx, result in group_results.items()
    }
    return {**plan.missing_group_results, **measured_results}


def _plan_group_shots(
    spec_key: object,
    observable,
    measurement_groups: Sequence[tuple[object, ...]],
    partition_indices: Sequence[list[int]],
    env: PipelineEnv,
    shot_distribution: ShotDistStrategy | None,
    *,
    total_shots: int | None = None,
) -> _GroupShotPlan:
    """Plan per-group shots and the estimator adjustments applied during reduction.

    ``observable`` is the flattened union, with real coefficients only.
    """
    n_groups = len(measurement_groups)
    if shot_distribution is None:
        # Every group gets the full budget; an override becomes uniform group
        # shots so it reaches the backend.
        if env.shots_override is None:
            return _GroupShotPlan(list(range(n_groups)), None)
        return _GroupShotPlan(
            surviving_indices=list(range(n_groups)),
            shots_by_group=dict.fromkeys(range(n_groups), env.effective_shots),
        )

    effective_shots = env.effective_shots if total_shots is None else total_shots
    coefficients = np.asarray(observable.coeffs).real.astype(np.float64)
    group_norms = _compute_group_l1_norms(coefficients, partition_indices)
    per_group_shots = _compute_shot_distribution(
        group_norms,
        effective_shots,
        shot_distribution,
        rng=env.rng,
    )

    surviving_shots: dict[int, int] = {
        idx: shots for idx, shots in enumerate(per_group_shots) if shots > 0
    }
    surviving_indices: list[int] = list(surviving_shots)
    dropped_indices: list[int] = [
        idx for idx, shots in enumerate(per_group_shots) if shots <= 0
    ]

    if dropped_indices and shot_distribution != "weighted_random":
        # |bias| <= L1 norm of the dropped groups, since |<h_i>| <= 1.
        dropped_norm = sum(group_norms[i] for i in dropped_indices)
        total_norm = sum(group_norms)
        dropped_fraction = dropped_norm / total_norm if total_norm > 0 else 0.0
        warnings.warn(
            f"Shot distribution assigned zero shots to "
            f"{len(dropped_indices)}/{n_groups} measurement group(s) for "
            f"spec {spec_key!r}; those groups are skipped and contribute "
            f"zero to the final expectation value, biasing the estimate by "
            f"at most {dropped_norm:.4g} ({dropped_fraction:.2%} of the "
            f"Hamiltonian's L1 norm). To reduce the bias: raise "
            f"``backend.shots``, switch to the deterministic "
            f"``shot_distribution='weighted'`` strategy, or accept it if "
            f"the fraction is negligible. Per-group allocations are "
            f"recorded on each circuit's ``group_shots``.",
            UserWarning,
            stacklevel=2,
        )
    # Zero per observable in the group (an absent draw for weighted_random).
    missing_group_results: dict[int, object] = {
        idx: dict.fromkeys(range(len(measurement_groups[idx])), 0.0)
        for idx in dropped_indices
    }

    scales_by_group: dict[int, float] = {}
    term_weights_by_group: dict[int, np.ndarray] = {}
    if shot_distribution == "weighted_random":
        total_weight = float(sum(group_norms))
        probabilities = (
            [group_norm / total_weight for group_norm in group_norms]
            if total_weight > 0
            else [1.0 / n_groups] * n_groups
        )
        scales_by_group = {
            idx: shots / effective_shots for idx, shots in surviving_shots.items()
        }
        # Zero-probability groups hold only zero-coefficient terms; weight moot.
        term_weights_by_group = {
            idx: np.full(len(indices), 1.0 / probability if probability > 0 else 1.0)
            for idx, (indices, probability) in enumerate(
                zip(partition_indices, probabilities)
            )
        }

    return _GroupShotPlan(
        surviving_indices=surviving_indices,
        shots_by_group=surviving_shots,
        missing_group_results=missing_group_results,
        scales_by_group=scales_by_group,
        term_weights_by_group=term_weights_by_group,
        random_group_selection=shot_distribution == "weighted_random",
    )


def _plans_for_spec(
    spec_key: tuple,
    observable: SparsePauliOp,
    measurement_groups: Sequence[tuple[object, ...]],
    partition_indices: Sequence[list[int]],
    env: PipelineEnv,
    shot_distribution: ShotDistStrategy | None,
    sample_budgets: Sequence[int] | None,
) -> dict[tuple, _GroupShotPlan]:
    """Build the execution plans for one observable specification."""
    if sample_budgets is None:
        return {
            spec_key: _plan_group_shots(
                spec_key,
                observable,
                measurement_groups,
                partition_indices,
                env,
                shot_distribution,
            )
        }

    return {
        (*spec_key, (PARAM_SET_AXIS, param_idx)): _plan_group_shots(
            (*spec_key, (PARAM_SET_AXIS, param_idx)),
            observable,
            measurement_groups,
            partition_indices,
            env,
            "weighted_random",
            total_shots=budget,
        )
        for param_idx, budget in enumerate(sample_budgets)
    }


def _plan_shadow_shots(
    spec_key: tuple,
    labels: Sequence[str],
    env: PipelineEnv,
    sample_budgets: Sequence[int] | None,
) -> tuple[tuple[tuple[str, ...], ...], list[list[int]], dict[tuple, _GroupShotPlan]]:
    """Draw classical-shadow snapshots and build one plan per draw.

    Each parameter set with an estimator budget draws independently; the groups
    are the union of every draw's cover patterns, and a plan treats the groups
    it did not draw as missing. Returns ``(groups, cover_indices, plans)``.
    """
    rng = env.rng if env.rng is not None else np.random.default_rng()
    if sample_budgets is None:
        budget_by_key = {spec_key: env.effective_shots}
    else:
        budget_by_key = {
            (*spec_key, (PARAM_SET_AXIS, param_idx)): budget
            for param_idx, budget in enumerate(sample_budgets)
        }
    draws = {
        key: _sample_shadow_covers(labels, budget, rng)
        for key, budget in budget_by_key.items()
    }

    patterns, _ = _count_unique_rows(
        np.concatenate([covers for covers, _, _ in draws.values()])
    )
    if not len(patterns):
        patterns = np.eye(1, len(labels), dtype=bool)
    group_by_pattern = {pattern.tobytes(): idx for idx, pattern in enumerate(patterns)}
    cover_indices = [np.flatnonzero(pattern).tolist() for pattern in patterns]
    groups = tuple(tuple(labels[term] for term in terms) for terms in cover_indices)
    term_weights = _shadow_term_weights(labels)
    term_weights_by_group = {
        idx: term_weights[terms] for idx, terms in enumerate(cover_indices)
    }

    plans: dict[tuple, _GroupShotPlan] = {}
    for key, (covers, counts, empty_draws) in draws.items():
        shots_by_group = {
            group_by_pattern[pattern.tobytes()]: int(count)
            for pattern, count in zip(covers, counts)
        }
        budget = budget_by_key[key]
        plan_weights = term_weights_by_group
        if not shots_by_group:
            # One empty draw runs group 0 at zero weight, so the plan has a result.
            shots_by_group = {0: 1}
            empty_draws -= 1
            plan_weights = {**term_weights_by_group, 0: np.zeros(len(cover_indices[0]))}
        plans[key] = _GroupShotPlan(
            surviving_indices=sorted(shots_by_group),
            shots_by_group=shots_by_group,
            missing_group_results={
                idx: dict.fromkeys(range(len(terms)), 0.0)
                for idx, terms in enumerate(cover_indices)
                if idx not in shots_by_group
            },
            scales_by_group={
                idx: shots / budget for idx, shots in shots_by_group.items()
            },
            term_weights_by_group=plan_weights,
            random_group_selection=True,
            empty_draws=empty_draws,
        )
    return groups, cover_indices, plans


@dataclass(frozen=True)
class MeasurementToken:
    """Token carrying measurement metadata from expand to reduce."""

    postprocess_fn_by_spec: dict[object, Callable] = field(default_factory=dict)
    """Per-spec postprocessing functions (combine groups with coefficients)."""

    group_shot_plans_by_spec: dict[tuple, _GroupShotPlan] = field(default_factory=dict)
    """Per-spec execution and estimator-adjustment plans."""

    constant_offsets_by_spec: dict[tuple, tuple[float, ...]] = field(
        default_factory=dict
    )
    """Exact per-observable identity contributions excluded from random sampling."""

    is_probs: bool = False
    """True when this is a probabilities measurement (no observable grouping)."""

    effective_strategy: str | None = None
    """Strategy actually used (may differ from user-configured when auto-promoted
    to ``_backend_expval``). ``None`` for the probs path."""

    n_observable_terms: int | None = None
    """Number of single-Pauli terms in the source observable (only set for
    the ``_backend_expval`` path, where ``measurement_groups`` is a sentinel
    empty group)."""

    def scoped_to(
        self, foreign_key: tuple, foreign_axes: AbstractSet[str]
    ) -> "MeasurementToken":
        """Select and strip metadata for one downstream-axis reduction group."""
        foreign_set = set(foreign_key)

        def scoped(entries: dict) -> dict:
            return {
                tuple(axis for axis in key if axis[0] not in foreign_axes): value
                for key, value in entries.items()
                if {
                    axis
                    for axis in key
                    if isinstance(axis, tuple) and axis[0] in foreign_axes
                }
                <= foreign_set
            }

        return replace(
            self,
            postprocess_fn_by_spec=scoped(self.postprocess_fn_by_spec),
            group_shot_plans_by_spec=scoped(self.group_shot_plans_by_spec),
            constant_offsets_by_spec=scoped(self.constant_offsets_by_spec),
        )


class MeasurementStage(BundleStage):
    """Unified measurement stage for all circuit measurement types.

    Handles both ``probs()`` and ``expval(H)`` measurements, auto-detecting
    the measurement type from the MetaCircuit.  For ``expval(H)``, groups
    observables using the configured strategy (or delegates to expval-native
    backends via ``ham_ops``).

    During ``expand``, records ``result_format`` on each circuit to communicate
    the expected result format to ``pipeline.run()``:

    - ``ResultFormat.PROBS`` for ``probs()`` measurements
    - ``ResultFormat.EXPVALS`` for ``expval(H)`` measurements
    - ``ResultFormat.COUNTS`` is never set here (reserved for PCE)

    During ``reduce``, combines measurement groups using the postprocessing
    function from ``_compute_measurement_groups``.
    """

    @property
    def axis_name(self) -> str:
        return OBS_GROUP_AXIS

    @property
    def handles_measurement(self) -> bool:
        return True

    @property
    def consumes_dag_bodies(self) -> bool:
        # Reads only observable, measured_wires and n_qubits.
        return False

    @property
    def volatile(self) -> bool:
        """True for strategies that draw from ``env.rng``.

        That is classical shadows, ``"weighted_random"`` and user-supplied
        shot-distribution callables.
        """
        return (
            self._grouping_strategy == "shadows"
            or self._shot_distribution == "weighted_random"
            or callable(self._shot_distribution)
        )

    def is_volatile_for(self, env: PipelineEnv) -> bool:
        """Estimator sampling redraws operators on every evaluation."""
        return self.volatile or env.estimator_samples is not None

    def cache_key_extras(self, env) -> tuple[Hashable, ...]:
        """Extra forward-pass cache key parts.

        Includes the effective shot budget when classical shadows, a shot
        distribution or ``shots_override`` is active, and ``collect_variance``
        when set.
        """
        extras: tuple[Hashable, ...] = ()
        if (
            self._grouping_strategy == "shadows"
            or self._shot_distribution is not None
            or env.shots_override is not None
        ):
            extras += (env.effective_shots,)
        if env.collect_variance:
            extras += (env.collect_variance,)
        return extras

    def __init__(
        self,
        grouping_strategy: GroupingStrategy = "qwc",
        result_format_override: ResultFormat | None = None,
        shot_distribution: ShotDistStrategy | None = None,
        measure_all: bool = False,
    ) -> None:
        """
        Args:
            grouping_strategy: How expval observables are measured; see
                :ref:`measuring-observables`. Ignored for ``probs()``
                measurements. Defaults to ``"qwc"``.
            result_format_override: If set, overrides the auto-detected result
                format. For example, pass ``ResultFormat.COUNTS`` to get raw
                shot counts even when an observable is present.
            measure_all: When ``False`` (default), measure only the qubits each
                group acts on non-trivially, reducing the number of distinct
                outcomes without changing the expectation value (identity
                positions are dropped by postprocessing). ``True`` measures
                every qubit. Affects only the expval path (the probs path
                always measures its declared wires); a ``COUNTS``/``PROBS``
                override forces full measurement.
            shot_distribution: How to split the backend's total shot budget
                across measurement groups (``"uniform"``, ``"weighted"``,
                ``"weighted_random"``, or a callable). Only affects results on
                a sampling backend (``supports_expval=False``); a backend that
                computes expectation values analytically has no measurement groups
                to split shots across, so it is not applied there and ``expand``
                emits a ``UserWarning`` saying so. When ``None`` (default),
                every group is submitted with the backend's full shot count.
        """
        super().__init__(name=type(self).__name__)
        if grouping_strategy not in (*_GROUPING_STRATEGY_NAMES, None):
            raise ValueError(
                f"Invalid grouping_strategy={grouping_strategy!r}. Choose "
                f"{', '.join(map(repr, _GROUPING_STRATEGY_NAMES))}, or None."
            )
        if grouping_strategy == "shadows" and shot_distribution is not None:
            raise ValueError(
                "shot_distribution cannot be combined with grouping_strategy="
                "'shadows': the random measurement bases already decide each "
                "circuit's shots."
            )
        self._grouping_strategy = grouping_strategy
        self._result_format_override = result_format_override
        self._shot_distribution = shot_distribution
        self._measure_all = measure_all

    # ------------------------------------------------------------------ #
    # Real-path QASM factories (shared inside :meth:`expand` / :meth:`dry_expand`).
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_wires_qasm(wires: Sequence[int]) -> str:
        """Build the ``measure q[i] -> c[i];`` sequence for a probs circuit."""
        return "".join(f"measure q[{q}] -> c[{q}];\n" for q in wires)

    @staticmethod
    def _dry_wires_qasm(wires: Sequence[int]) -> str:
        """Dry placeholder: probs measurement is always one tagged entry."""
        return ""

    @staticmethod
    def _dry_expval_qasms(
        surviving_groups: Sequence[tuple[object, ...]],
        n_qubits: int,
        measure_all: bool = False,
    ) -> tuple[str, ...]:
        """Dry placeholder: one empty string per surviving measurement group."""
        return ("",) * len(surviving_groups)

    def expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> StageOutput[MetaCircuitBatch]:
        """Set up measurements and declare the result format on env."""
        return self._dispatch(
            batch,
            env,
            expval_qasm_factory=measurement_qasms_from_groups,
            probs_qasm_factory=self._build_wires_qasm,
        )

    def dry_expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> StageOutput[MetaCircuitBatch]:
        """Same grouping and shot allocation as :meth:`expand`, with placeholder QASM."""
        return self._dispatch(
            batch,
            env,
            expval_qasm_factory=self._dry_expval_qasms,
            probs_qasm_factory=self._dry_wires_qasm,
        )

    def _dispatch(
        self,
        batch: MetaCircuitBatch,
        env: PipelineEnv,
        *,
        expval_qasm_factory: Callable,
        probs_qasm_factory: Callable,
    ) -> StageOutput[MetaCircuitBatch]:
        """Pick the expval or probs path and apply ``result_format_override``."""
        sample_meta = next(iter(batch.values()))
        if sample_meta.observable is not None:
            result = self._expand_expval(batch, env, expval_qasm_factory)
        elif sample_meta.measured_wires is not None:
            result = self._expand_probs(batch, env, probs_qasm_factory)
        else:
            raise ValueError(
                "MeasurementStage: MetaCircuit has neither an observable "
                "nor measured_wires set. Did the spec stage run?"
            )

        if self._result_format_override is None:
            return result
        return result._replace(
            batch={
                key: meta.set_result_format(self._result_format_override)
                for key, meta in result.batch.items()
            }
        )

    # ------------------------------------------------------------------ #
    # Probs path
    # ------------------------------------------------------------------ #

    def _expand_probs(
        self,
        batch: MetaCircuitBatch,
        env: PipelineEnv,
        qasm_factory: Callable[[tuple[int, ...]], str],
    ) -> StageOutput[MetaCircuitBatch]:
        """Generate measurement QASM for ``probs()`` / ``counts()`` circuits."""
        out: MetaCircuitBatch = {}

        for key, meta in batch.items():
            wires = meta.measured_wires
            if wires is None:
                raise ValueError(
                    f"MeasurementStage (probs path): key '{key}' has no "
                    "measured_wires set."
                )
            measure_qasm = qasm_factory(wires)
            tagged_measurement_qasms = ((((PROBS_MEAS_AXIS, 0),), measure_qasm),)
            out[key] = meta.set_measurement_bodies(
                tagged_measurement_qasms
            ).set_result_format(ResultFormat.PROBS)

        token = MeasurementToken(is_probs=True)
        return StageOutput(batch=out, token=token)

    def _resolve_expval_mode(
        self,
        batch: MetaCircuitBatch,
        env: PipelineEnv,
    ) -> tuple[_ResolvedGroupingStrategy, tuple[int, ...] | None]:
        """Resolve backend-aware grouping and optional estimator budgets."""
        if self._grouping_strategy == "shadows" and env.backend.supports_expval:
            raise ValueError(
                "grouping_strategy='shadows' requires a sampling backend; "
                f"{type(env.backend).__name__} computes expectation values "
                "analytically and does not consume shots."
            )
        estimator_samples = env.estimator_samples
        if estimator_samples is not None:
            if env.backend.supports_expval:
                raise ValueError(
                    "estimator_samples requires a sampling backend; analytic "
                    "expectation-value backends do not consume shots."
                )
            if self._shot_distribution not in (None, "weighted_random"):
                raise ValueError(
                    "estimator_samples selects weighted-random sampling and cannot "
                    "be combined with another shot_distribution strategy."
                )
            sample_budgets = estimator_samples.by_param_set
        else:
            sample_budgets = None

        strategy: _ResolvedGroupingStrategy = self._grouping_strategy
        # The analytic path emits one shared ham_ops for the whole batch.
        first_obs = next(iter(batch.values())).observable
        if env.backend.supports_expval and all(
            meta.observable == first_obs for meta in batch.values()
        ):
            strategy = BACKEND_EXPVAL

        if self._shot_distribution is not None and strategy == BACKEND_EXPVAL:
            warnings.warn(
                f"shot_distribution is set but backend "
                f"{type(env.backend).__name__} computes expectation values "
                "analytically (supports_expval=True), so per-group shot "
                "allocation does not change the (exact) result. Use a "
                "sampling backend for shot_distribution to take effect — e.g. "
                "MaestroSimulator(force_sampling=True), "
                "QiskitSimulator(force_sampling=True), or QoroService with "
                "JobConfig(force_sampling=True).",
                UserWarning,
                stacklevel=3,
            )

        return strategy, sample_budgets

    # ------------------------------------------------------------------ #
    # Expval path
    # ------------------------------------------------------------------ #

    def _expand_expval(
        self,
        batch: MetaCircuitBatch,
        env: PipelineEnv,
        qasm_factory: Callable[..., tuple[str, ...]],
    ) -> StageOutput[MetaCircuitBatch]:
        """Group observables and generate measurement QASM (or ham_ops).

        ``qasm_factory`` maps ``(surviving_groups, n_qubits, measure_all=...)``
        to one measurement QASM per group.
        """
        strategy, sample_budgets = self._resolve_expval_mode(batch, env)

        result: MetaCircuitBatch = {}
        postprocess_fn_by_spec: dict[object, Callable] = {}
        n_observable_terms: int | None = None
        group_shot_plans_by_spec: dict[tuple, _GroupShotPlan] = {}
        constant_offsets_by_spec: dict[tuple, tuple[float, ...]] = {}
        sample_union: SparsePauliOp | None = None

        # Full measurement when opted out, COUNTS/PROBS override, or the analytic
        # _backend_expval path (its all-identity sentinel would be rejected).
        measure_all = (
            self._measure_all
            or strategy == BACKEND_EXPVAL
            or self._result_format_override in (ResultFormat.COUNTS, ResultFormat.PROBS)
        )

        for key, meta in batch.items():
            if meta.observable is None:
                raise ValueError(
                    f"MeasurementStage (expval path): key '{key}' has no "
                    "observable set."
                )
            # MetaCircuit.__post_init__ normalises a bare SparsePauliOp to a
            # 1-tuple via object.__setattr__, which type checkers can't track.
            observable = (
                meta.observable
                if isinstance(meta.observable, tuple)
                else (meta.observable,)
            )
            if sample_budgets is not None and len(observable) != 1:
                raise ValueError(
                    "estimator_samples requires one scalar observable per circuit."
                )
            # Random sampling handles identity terms exactly, outside the draws.
            constant_offsets: tuple[float, ...] | None = None
            if sample_budgets is not None or strategy == "shadows":
                observable, constant_offsets = _split_identity_offsets(observable)

            if strategy == "shadows":
                union_obs, per_obs_term_indices, labels = _union_labels(
                    observable, meta.n_qubits
                )
                measurement_groups, partition_indices, plans = _plan_shadow_shots(
                    key, labels, env, sample_budgets
                )
            else:
                (
                    measurement_groups,
                    partition_indices,
                    union_obs,
                    per_obs_term_indices,
                ) = _compute_measurement_groups(observable, strategy, meta.n_qubits)
                plans = _plans_for_spec(
                    key,
                    union_obs,
                    measurement_groups,
                    partition_indices,
                    env,
                    self._shot_distribution,
                    sample_budgets,
                )
            if strategy == BACKEND_EXPVAL and n_observable_terms is None:
                n_observable_terms = sum(len(p) for p in partition_indices)

            if sample_union is None:
                sample_union = union_obs
            group_shot_plans_by_spec.update(plans)
            postprocessing_fn = _create_postprocessing_fn(
                observable, per_obs_term_indices, partition_indices, len(union_obs)
            )
            surviving_indices = sorted(
                {
                    group_idx
                    for plan in plans.values()
                    for group_idx in plan.surviving_indices
                }
            )
            if not surviving_indices:
                raise ValueError(
                    f"shot_distribution assigned zero shots to every measurement "
                    f"group of '{key}', so no circuit can be submitted. Raise the "
                    "backend's shot count or use an allocation that keeps at least "
                    "one group."
                )

            surviving_groups = tuple(measurement_groups[i] for i in surviving_indices)
            measurement_qasms = qasm_factory(
                surviving_groups, meta.n_qubits, measure_all=measure_all
            )
            tagged_measurement_qasms = tuple(
                (((OBS_GROUP_AXIS, orig_idx),), meas_qasm)
                for orig_idx, meas_qasm in zip(surviving_indices, measurement_qasms)
            )

            # Keep all groups: results are indexed by their original obs_group tag.
            measurement_meta = (
                meta if constant_offsets is None else meta.set_observable(observable)
            )
            new_meta = (
                measurement_meta.set_measurement_bodies(tagged_measurement_qasms)
                .set_measurement_groups(measurement_groups)
                .set_result_format(ResultFormat.EXPVALS)
            )
            if sample_budgets is not None:
                new_meta = new_meta.set_param_group_shots(
                    {
                        param_idx: plans[
                            (*key, (PARAM_SET_AXIS, param_idx))
                        ].shots_by_group
                        or {}
                        for param_idx in range(len(sample_budgets))
                    }
                )
            else:
                shot_plan = plans[key]
                # Analytic expval ignores shots and rejects shot_groups.
                if shot_plan.shots_by_group is not None and strategy != BACKEND_EXPVAL:
                    new_meta = new_meta.set_group_shots(shot_plan.shots_by_group)
            result[key] = new_meta
            postprocess_fn_by_spec.update(dict.fromkeys(plans, postprocessing_fn))
            if constant_offsets is not None:
                constant_offsets_by_spec.update(dict.fromkeys(plans, constant_offsets))

        if strategy == BACKEND_EXPVAL and sample_union is not None:
            ham_ops = _sparse_pauli_op_to_ham_string(sample_union)
            result = {
                key: meta.set_backend_ham_ops(ham_ops) for key, meta in result.items()
            }

        token = MeasurementToken(
            postprocess_fn_by_spec=postprocess_fn_by_spec,
            effective_strategy=strategy,
            n_observable_terms=n_observable_terms,
            group_shot_plans_by_spec=group_shot_plans_by_spec,
            constant_offsets_by_spec=constant_offsets_by_spec,
        )
        return StageOutput(batch=result, token=token)

    # ------------------------------------------------------------------ #
    def introspect(
        self,
        batch: MetaCircuitBatch,
        env: PipelineEnv,
        token: StageToken,
    ) -> dict[str, Any]:
        meta = next(iter(batch.values()), None)
        effective_strategy = getattr(token, "effective_strategy", None)
        strategy = effective_strategy or self._grouping_strategy
        info: dict[str, Any] = {
            "strategy": "analytic" if strategy == BACKEND_EXPVAL else strategy
        }
        if meta is None:
            return info

        # Analytic expval carries sentinel groups; check them too, since the
        # strategy is only known from the token.
        if (
            effective_strategy == BACKEND_EXPVAL
            or meta.measurement_groups == BACKEND_EXPVAL_GROUPS
        ):
            info["strategy"] = "analytic"
            info["n_groups"] = 1
            if getattr(token, "n_observable_terms", None) is not None:
                info["n_pauli_terms"] = token.n_observable_terms
            return info

        groups = meta.measurement_groups
        if not groups:
            # Computational-basis readout: nothing was grouped.
            info.pop("strategy", None)
            info["readout"] = "computational basis (no observable)"
            return info

        info["n_groups"] = len(groups)
        info["n_pauli_terms"] = len({label for g in groups for label in g})
        if effective_strategy == "shadows":
            # Empty snapshots count toward the estimate but are never submitted.
            plans = getattr(token, "group_shot_plans_by_spec", {}).values()
            empty_snapshots = sum(plan.empty_draws for plan in plans)
            info["snapshots"] = empty_snapshots + sum(
                sum((plan.shots_by_group or {}).values()) for plan in plans
            )
            info["empty_snapshots"] = empty_snapshots
        if env.estimator_samples is not None:
            info["estimator_samples_per_param_set"] = list(
                env.estimator_samples.by_param_set
            )
            plans = getattr(token, "group_shot_plans_by_spec", {}).values()
            info["device_shots"] = sum(
                sum((plan.shots_by_group or {}).values()) for plan in plans
            )
        info["measured_qubits"] = (
            "all" if self._measure_all else "per group (observable support only)"
        )
        # Most terms in one group, and most qubits any group's basis change touches.
        info["largest_group_size"] = max(len(g) for g in groups)
        info["largest_group_width"] = max(
            len({q for label in g for q, c in enumerate(str(label)) if c != "I"})
            for g in groups
        )

        spec_pgs = next((m.group_shots for m in batch.values() if m.group_shots), None)
        if spec_pgs:
            info["shots_per_group_range"] = [
                min(spec_pgs.values()),
                max(spec_pgs.values()),
            ]
            # Zero-shot groups are dropped, so fewer than n_groups may be submitted.
            if len(spec_pgs) != len(groups):
                info["n_groups_submitted"] = len(spec_pgs)
        else:
            info["shots_per_circuit"] = env.effective_shots
        return info

    # ------------------------------------------------------------------ #
    # Reduce
    # ------------------------------------------------------------------ #

    def _estimate_cost_variance(
        self,
        raw: ChildResults,
        batch: MetaCircuitBatch,
        token: StageToken,
    ) -> dict[tuple, float]:
        """Estimate variance according to this stage's sampling strategy."""
        random_plans = {
            key: plan
            for key, plan in token.group_shot_plans_by_spec.items()
            if plan.random_group_selection
        }
        if random_plans:
            return _counts_to_wrs_cost_variance(
                raw,
                batch,
                {key: plan.term_weights_by_group for key, plan in random_plans.items()},
                {key: plan.empty_draws for key, plan in random_plans.items()},
            )
        return _counts_to_cost_variance(raw, batch)

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        """Combine measurement groups (expval) or strip meas axis (probs)."""
        if token.is_probs:
            return self._reduce_probs(results)
        if self._result_format_override in (ResultFormat.COUNTS, ResultFormat.PROBS):
            return self._reduce_raw(results)
        return self._reduce_expval(results, token)

    @staticmethod
    def _reduce_strip_axis(results: ChildResults, axis_name: str) -> ChildResults:
        """Identity reduce: drop ``axis_name`` from each key, values untouched."""
        return {
            strip_axis_from_label(key, axis_name): value
            for key, value in results.items()
        }

    @classmethod
    def _reduce_probs(cls, results: ChildResults) -> ChildResults:
        """Strip the meas axis from probs results (identity reduce)."""
        return cls._reduce_strip_axis(results, PROBS_MEAS_AXIS)

    @classmethod
    def _reduce_raw(cls, results: ChildResults) -> ChildResults:
        """Strip the obs_group axis without postprocessing.

        Used when the expand path grouped observables but the caller
        requested raw counts or probabilities instead of expectation values.
        """
        return cls._reduce_strip_axis(results, OBS_GROUP_AXIS)

    def _reduce_expval(
        self, results: ChildResults, token: MeasurementToken
    ) -> ChildResults:
        """Combine expval results across measurement groups."""
        grouped = group_by_base_key(results, OBS_GROUP_AXIS, indexed=True)
        grouped = {
            base_key: _apply_group_shot_plan(
                group_dict, token.group_shot_plans_by_spec.get(base_key)
            )
            for base_key, group_dict in grouped.items()
        }

        postprocess_fn_by_base = {
            base_key: token.postprocess_fn_by_spec[base_key] for base_key in grouped
        }
        reduced = reduce_postprocess_ordered(grouped, postprocess_fn_by_base)
        return {
            base_key: (
                [
                    obs_value + offset
                    for obs_value, offset in zip(
                        value, token.constant_offsets_by_spec[base_key], strict=True
                    )
                ]
                if base_key in token.constant_offsets_by_spec
                else value
            )
            for base_key, value in reduced.items()
        }
