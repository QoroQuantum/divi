# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from divi.circuits import MetaCircuit
from divi.circuits._conversions import (
    measurement_qasms_from_groups,
    sparse_pauli_op_to_ham_string,
)
from divi.pipeline._grouping import GroupingStrategy, compute_measurement_groups
from divi.pipeline._shot_distribution import (
    ShotDistStrategy,
    compute_group_l1_norms,
    compute_shot_distribution,
)
from divi.pipeline.abc import (
    BundleStage,
    ChildResults,
    ExpansionResult,
    MetaCircuitBatch,
    PipelineEnv,
    ResultFormat,
    StageToken,
)
from divi.pipeline.transformations import group_by_base_key, reduce_postprocess_ordered

OBS_GROUP_AXIS = "obs_group"
PROBS_MEAS_AXIS = "meas"


def _allocate_per_group_shots(
    spec_key: object,
    observable,
    measurement_groups: tuple[tuple[object, ...], ...],
    partition_indices: list[list[int]],
    env: PipelineEnv,
    shot_distribution: ShotDistStrategy | None,
) -> tuple[list[int], dict[int, object], dict[int, int] | None]:
    """Compute per-group shots and identify dropped (zero-shot) groups.

    Pure helper — no instance state. Reads ``env.backend.shots`` and
    ``env.rng`` from the pipeline environment, and ``shot_distribution``
    from the caller.

    Returns:
        surviving_indices: Original indices of groups to actually submit.
        zero_shot_groups: ``{orig_idx: zero_result_for_group}`` for each
            group that received zero shots; the reduce step injects these
            so the postprocessing function still receives one entry per group.
        surviving_shots: ``{orig_idx: shots}`` for each surviving group,
            or ``None`` when ``shot_distribution`` is not configured.
    """
    n_groups = len(measurement_groups)
    if shot_distribution is None or n_groups == 0:
        return list(range(n_groups)), {}, None

    raw_coeffs = np.asarray(observable.coeffs)
    # SparsePauliOp stores complex coefficients. For a Hermitian operator the
    # imaginary parts are zero up to construction round-off (~1e-15 relative).
    # A non-trivial imaginary component means the operator isn't Hermitian,
    # and silently dropping it would produce wrong group norms — warn so the
    # user can symmetrize the operator or inspect their construction.
    max_abs = float(np.max(np.abs(raw_coeffs))) if raw_coeffs.size else 0.0
    if max_abs > 0:
        max_imag = float(np.max(np.abs(raw_coeffs.imag)))
        if max_imag > 1e-8 * max_abs:
            warnings.warn(
                f"Observable for spec {spec_key!r} has non-negligible imaginary "
                f"coefficients (max |Im(c)| = {max_imag:.3g}, max |c| = "
                f"{max_abs:.3g}). Shot allocation uses only the real parts; if "
                f"the operator is meant to be Hermitian, symmetrize it as "
                f"0.5 * (O + O.adjoint()) before constructing the program.",
                UserWarning,
                stacklevel=2,
            )
    coefficients = raw_coeffs.real.astype(np.float64)
    group_norms = compute_group_l1_norms(coefficients, partition_indices)
    per_group_shots = compute_shot_distribution(
        group_norms,
        env.backend.shots,
        shot_distribution,
        rng=env.rng,
    )

    surviving_indices: list[int] = []
    dropped_indices: list[int] = []
    surviving_shots: dict[int, int] = {}
    for idx, shots in enumerate(per_group_shots):
        if shots > 0:
            surviving_indices.append(idx)
            surviving_shots[idx] = shots
        else:
            dropped_indices.append(idx)

    zero_shot_groups: dict[int, object] = {}
    if dropped_indices:
        # Quantify the bias introduced by dropping these groups. The
        # estimator is biased by sum(c_i * <h_i>) over the dropped terms;
        # |bias| <= sum_{dropped} ||c_i||_1 = sum(group_norms[dropped]),
        # since |<h_i>| <= 1 for any Pauli string. Reporting the dropped
        # fraction of total L1 norm tells the user whether the skipped terms
        # are negligible or load-bearing.
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
            f"available in ``env.artifacts['per_group_shots']``.",
            UserWarning,
            stacklevel=2,
        )
        for idx in dropped_indices:
            # Match the shape that the postprocessing function expects:
            # a dict {obs_idx_within_group: 0.0} works for both single-
            # and multi-observable groups.
            zero_shot_groups[idx] = {
                j: 0.0 for j in range(len(measurement_groups[idx]))
            }

    return surviving_indices, zero_shot_groups, surviving_shots


@dataclass(frozen=True)
class MeasurementToken:
    """Token carrying measurement metadata from expand to reduce."""

    postprocess_fn_by_spec: dict[object, Callable] = field(default_factory=dict)
    """Per-spec postprocessing functions (combine groups with coefficients)."""

    zero_shot_groups_by_spec: dict[object, dict[int, object]] = field(
        default_factory=dict
    )
    """Per-spec ``{obs_group_idx: zero_result_for_group}`` for groups that
    adaptive shot allocation assigned zero shots.  Injected back into the
    grouped results before postprocessing so those groups contribute zero
    to the final energy."""

    is_probs: bool = False
    """True when this is a probabilities measurement (no observable grouping)."""

    effective_strategy: str | None = None
    """Strategy actually used (may differ from user-configured when auto-promoted
    to ``_backend_expval``). ``None`` for the probs path."""

    n_observable_terms: int | None = None
    """Number of single-Pauli terms in the source observable (only set for
    the ``_backend_expval`` path, where ``measurement_groups`` is a sentinel
    empty group)."""


class MeasurementStage(BundleStage):
    """Unified measurement stage for all circuit measurement types.

    Handles both ``probs()`` and ``expval(H)`` measurements, auto-detecting
    the measurement type from the MetaCircuit.  For ``expval(H)``, groups
    observables using the configured strategy (or delegates to expval-native
    backends via ``ham_ops``).

    During ``expand``, sets ``env.result_format`` to communicate the expected
    result format to ``pipeline.run()``:

    - ``ResultFormat.PROBS`` for ``probs()`` measurements
    - ``ResultFormat.EXPVALS`` for ``expval(H)`` measurements
    - ``ResultFormat.COUNTS`` is never set here (reserved for PCE)

    During ``reduce``, combines measurement groups using the postprocessing
    function from ``compute_measurement_groups``.
    """

    @property
    def axis_name(self) -> str | None:
        return OBS_GROUP_AXIS

    @property
    def handles_measurement(self) -> bool:
        return True

    @property
    def consumes_dag_bodies(self) -> bool:
        # Reads only ``meta.observable`` / ``meta.measured_wires`` /
        # ``meta.n_qubits`` — never touches body DAG contents.
        return False

    @property
    def stateful(self) -> bool:
        """True only for genuinely non-deterministic shot-distribution strategies.

        ``"weighted_random"`` and user-supplied callables draw from ``env.rng``
        and must re-run on every call. The built-in deterministic strategies
        (``"uniform"``, ``"weighted"``) are pure functions of the observable
        and the shot count, so they stay cacheable; cache invalidation on
        shot-count changes is delegated to :meth:`cache_key_extras`.
        """
        return self._shot_distribution == "weighted_random" or callable(
            self._shot_distribution
        )

    def cache_key_extras(self, env) -> tuple:
        """Fold ``env.backend.shots`` into the forward-pass cache key.

        Any configured shot distribution (even the deterministic ones) reads
        ``env.backend.shots`` during :meth:`expand` to compute the per-group
        allocation. Including the current shot budget in the cache key means
        a re-run with a different budget triggers fresh allocation rather
        than replaying the stale one. Returns ``()`` when
        ``shot_distribution`` is ``None``.
        """
        if self._shot_distribution is None:
            return ()
        return (env.backend.shots,)

    def __init__(
        self,
        grouping_strategy: GroupingStrategy = "qwc",
        result_format_override: ResultFormat | None = None,
        shot_distribution: ShotDistStrategy | None = None,
    ) -> None:
        """
        Args:
            grouping_strategy: Grouping strategy for expval observables.
                Ignored for ``probs()`` measurements.
                Defaults to "qwc" (qubit-wise commuting).
            result_format_override: If set, overrides the auto-detected result
                format. For example, pass ``ResultFormat.COUNTS`` to get raw
                shot counts even when an observable is present.
            shot_distribution: How to split the backend's total shot budget
                across measurement groups (``"uniform"``, ``"weighted"``,
                ``"weighted_random"``, or a callable). Only valid for
                sampling-based grouping strategies (``"qwc"``, ``"wires"``,
                or ``None``); rejected for ``"_backend_expval"`` because
                the backend computes expectation values analytically and
                ignores shots. When ``None`` (default), every group is
                submitted with the backend's full shot count.
        """
        super().__init__(name=type(self).__name__)
        self._grouping_strategy = grouping_strategy
        self._result_format_override = result_format_override
        self._shot_distribution = shot_distribution

    # ------------------------------------------------------------------ #
    # Real-path QASM factories (shared inside :meth:`expand` / :meth:`dry_expand`).
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_wires_qasm(wires: tuple[int, ...]) -> str:
        """Build the ``measure q[i] -> c[i];`` sequence for a probs circuit."""
        return "".join(f"measure q[{q}] -> c[{q}];\n" for q in wires)

    @staticmethod
    def _dry_wires_qasm(wires: tuple[int, ...]) -> str:
        """Dry placeholder: probs measurement is always one tagged entry."""
        return ""

    @staticmethod
    def _dry_expval_qasms(
        surviving_groups: tuple[tuple[object, ...], ...], n_qubits: int
    ) -> tuple[str, ...]:
        """Dry placeholder: one empty string per surviving measurement group."""
        return ("",) * len(surviving_groups)

    def expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> tuple[ExpansionResult, StageToken]:
        """Set up measurements and declare the result format on env."""
        return self._dispatch(
            batch,
            env,
            expval_qasm_factory=measurement_qasms_from_groups,
            probs_qasm_factory=self._build_wires_qasm,
        )

    def dry_expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> tuple[ExpansionResult, StageToken]:
        """Analytic path: keep grouping + shot allocation, skip QASM rendering.

        Group count is the source of truth for the measurement fan-out, so
        we always run ``compute_measurement_groups`` and
        ``_allocate_per_group_shots`` (both cheap, pure numpy / analytic).
        Only the per-group QASM string generation is swapped for a
        placeholder so the emitted batch has correct shape without
        materialising measurement circuits.
        """
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
    ) -> tuple[ExpansionResult, StageToken]:
        """Shared front-end for both real and dry paths.

        Picks the expval vs probs branch from the first MetaCircuit's shape
        and applies the ``result_format_override``, exactly as the legacy
        ``expand`` did — only the QASM factory differs between modes.
        """
        sample_meta = next(iter(batch.values()))
        # Probs/counts circuits carry measured_wires; expval circuits carry
        # observable.  Exactly one is expected to be set.
        if sample_meta.observable is not None:
            result = self._expand_expval(batch, env, expval_qasm_factory)
        elif sample_meta.measured_wires is not None:
            result = self._expand_probs(batch, env, probs_qasm_factory)
        else:
            raise ValueError(
                "MeasurementStage: MetaCircuit has neither an observable "
                "nor measured_wires set. Did the spec stage run?"
            )

        if self._result_format_override is not None:
            env.result_format = self._result_format_override

        return result

    # ------------------------------------------------------------------ #
    # Probs path
    # ------------------------------------------------------------------ #

    def _expand_probs(
        self,
        batch: MetaCircuitBatch,
        env: PipelineEnv,
        qasm_factory: Callable[[tuple[int, ...]], str],
    ) -> tuple[ExpansionResult, StageToken]:
        """Generate measurement QASM for ``probs()`` / ``counts()`` circuits."""
        env.result_format = ResultFormat.PROBS

        out: dict[object, MetaCircuit] = {}

        for key, meta in batch.items():
            wires = meta.measured_wires
            if wires is None:
                # Defensive: expand() already dispatched based on observable
                # vs measured_wires, but the batch might be heterogeneous.
                raise ValueError(
                    f"MeasurementStage (probs path): key '{key}' has no "
                    "measured_wires set."
                )
            measure_qasm = qasm_factory(wires)
            tagged_measurement_qasms = ((((PROBS_MEAS_AXIS, 0),), measure_qasm),)
            out[key] = meta.set_measurement_bodies(tagged_measurement_qasms)

        token = MeasurementToken(is_probs=True)
        return ExpansionResult(batch=out), token

    # ------------------------------------------------------------------ #
    # Expval path
    # ------------------------------------------------------------------ #

    def _expand_expval(
        self,
        batch: MetaCircuitBatch,
        env: PipelineEnv,
        qasm_factory: Callable[[tuple[tuple[object, ...], ...], int], tuple[str, ...]],
    ) -> tuple[ExpansionResult, StageToken]:
        """Group observables and generate measurement QASM (or ham_ops).

        ``qasm_factory`` turns ``(surviving_groups, n_qubits)`` into a tuple of
        per-group measurement QASMs. :meth:`expand` passes
        :func:`measurement_qasms_from_groups`; :meth:`dry_expand` passes a
        placeholder factory so the batch shape is preserved without
        serialising diagonalising gates + ``measure`` instructions.
        """
        env.result_format = ResultFormat.EXPVALS

        # Setting shot_distribution declares sampling intent — skip the
        # analytical fallback entirely so the user's per-group allocation
        # is actually used.
        strategy = self._grouping_strategy
        if (
            self._shot_distribution is None
            and strategy in ("qwc", "_backend_expval")
            and env.backend.supports_expval
        ):
            first_obs = next(iter(batch.values())).observable
            all_same_obs = all(meta.observable == first_obs for meta in batch.values())
            strategy = "_backend_expval" if all_same_obs else "qwc"

        if self._shot_distribution is not None and strategy == "_backend_expval":
            raise ValueError(
                "shot_distribution is incompatible with the '_backend_expval' grouping "
                "strategy: the backend computes expectation values analytically and "
                "ignores shots. Set grouping_strategy to 'qwc', 'wires', or None."
            )

        result: dict[object, MetaCircuit] = {}
        postprocess_fn_by_spec: dict[object, Callable] = {}
        n_observable_terms: int | None = None
        zero_shot_groups_by_spec: dict[object, dict[int, object]] = {}
        per_group_shots_by_spec: dict[object, dict[int, int]] = {}

        for key, meta in batch.items():
            if meta.observable is None:
                raise ValueError(
                    f"MeasurementStage (expval path): key '{key}' has no "
                    "observable set."
                )

            measurement_groups, partition_indices, postprocessing_fn = (
                compute_measurement_groups(meta.observable, strategy, meta.n_qubits)
            )
            if strategy == "_backend_expval" and n_observable_terms is None:
                n_observable_terms = sum(len(p) for p in partition_indices)

            # Decide which groups to actually submit and with how many shots.
            surviving_indices, zero_shot_groups, surviving_shots = (
                _allocate_per_group_shots(
                    key,
                    meta.observable,
                    measurement_groups,
                    partition_indices,
                    env,
                    self._shot_distribution,
                )
            )
            if zero_shot_groups:
                zero_shot_groups_by_spec[key] = zero_shot_groups
            if surviving_shots is not None:
                per_group_shots_by_spec[key] = surviving_shots

            surviving_groups = tuple(measurement_groups[i] for i in surviving_indices)
            measurement_qasms = qasm_factory(surviving_groups, meta.n_qubits)
            tagged_measurement_qasms = tuple(
                (((OBS_GROUP_AXIS, orig_idx),), meas_qasm)
                for orig_idx, meas_qasm in zip(surviving_indices, measurement_qasms)
            )

            # Keep the *full* measurement_groups on the MetaCircuit so that
            # _counts_to_expvals can index into it by the original obs_group
            # tag carried on each surviving label.
            result[key] = meta.set_measurement_bodies(
                tagged_measurement_qasms
            ).set_measurement_groups(measurement_groups)
            postprocess_fn_by_spec[key] = postprocessing_fn

        if per_group_shots_by_spec:
            env.artifacts["per_group_shots"] = per_group_shots_by_spec
        else:
            env.artifacts.pop("per_group_shots", None)

        # For expval-native backends, compute ham_ops from the SparsePauliOp
        # and store it in env.artifacts so _default_execute_fn can use it.
        if strategy == "_backend_expval":
            sample_meta = next(iter(batch.values()))
            env.artifacts["ham_ops"] = sparse_pauli_op_to_ham_string(
                sample_meta.observable
            )
        else:
            env.artifacts.pop("ham_ops", None)

        token = MeasurementToken(
            postprocess_fn_by_spec=postprocess_fn_by_spec,
            effective_strategy=strategy,
            n_observable_terms=n_observable_terms,
            zero_shot_groups_by_spec=zero_shot_groups_by_spec,
        )
        return ExpansionResult(batch=result), token

    # ------------------------------------------------------------------ #
    def introspect(
        self,
        batch: MetaCircuitBatch,
        env: PipelineEnv,
        token: StageToken,
    ) -> dict[str, Any]:
        meta = next(iter(batch.values()), None)
        effective_strategy = getattr(token, "effective_strategy", None)
        info: dict[str, Any] = {
            "strategy": effective_strategy or self._grouping_strategy
        }
        if meta is None:
            return info

        # Backend-native expval: measurement_groups is a sentinel empty group
        # because the backend evaluates the full observable directly. Surface
        # the source observable instead of the empty placeholder.
        if effective_strategy == "_backend_expval":
            info["n_groups"] = 1
            if getattr(token, "n_observable_terms", None) is not None:
                info["n_terms"] = token.n_observable_terms
            obs_str = str(meta.observable) if meta.observable is not None else ""
            if len(obs_str) > 80:
                obs_str = obs_str[:77] + "..."
            info["observable"] = obs_str
            return info

        groups = meta.measurement_groups
        info["n_groups"] = len(groups)
        n_terms = sum(len(g) for g in groups)
        info["n_terms"] = n_terms
        # Largest group by term count
        if groups:
            largest = max(groups, key=len)
            info["largest_group"] = ", ".join(str(op) for op in largest[:5])
            if len(largest) > 5:
                info["largest_group"] += f" ... ({len(largest)} terms)"
        return info

    # Reduce
    # ------------------------------------------------------------------ #

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        """Combine measurement groups (expval) or strip meas axis (probs)."""
        if token.is_probs:
            return self._reduce_probs(results)
        if env.result_format in (ResultFormat.COUNTS, ResultFormat.PROBS):
            return self._reduce_raw(results)
        return self._reduce_expval(results, token)

    @staticmethod
    def _reduce_probs(results: ChildResults) -> ChildResults:
        """Strip the meas axis from probs results (identity reduce)."""
        out: ChildResults = {}
        for key, value in results.items():
            base_key = tuple(ax for ax in key if ax[0] != PROBS_MEAS_AXIS)
            out[base_key] = value
        return out

    @staticmethod
    def _reduce_raw(results: ChildResults) -> ChildResults:
        """Strip the obs_group axis without postprocessing.

        Used when the expand path grouped observables but the caller
        requested raw counts or probabilities instead of expectation values.
        """
        out: ChildResults = {}
        for key, value in results.items():
            base_key = tuple(ax for ax in key if ax[0] != OBS_GROUP_AXIS)
            out[base_key] = value
        return out

    def _reduce_expval(
        self, results: ChildResults, token: MeasurementToken
    ) -> ChildResults:
        """Combine expval results across measurement groups."""
        grouped = group_by_base_key(results, OBS_GROUP_AXIS, indexed=True)

        # Inject zero results for groups that adaptive shot allocation
        # assigned zero shots. The postprocessing function expects one entry
        # per original measurement group; the zero fills make those terms
        # contribute zero to the final sum. Built as an explicit merge
        # (zero-shot groups ∪ measured results, with measured results winning
        # on collisions) so the input dicts are never mutated in place.
        if token.zero_shot_groups_by_spec:
            grouped = {
                base_key: {
                    **token.zero_shot_groups_by_spec.get(base_key, {}),
                    **group_dict,
                }
                for base_key, group_dict in grouped.items()
            }

        postprocess_fn_by_base = {
            base_key: token.postprocess_fn_by_spec[base_key] for base_key in grouped
        }
        return reduce_postprocess_ordered(grouped, postprocess_fn_by_base)
