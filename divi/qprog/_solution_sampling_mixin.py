# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Mixin adding the discrete-solution-sampling capability to a quantum program.

Sampling a solution — running a circuit, measuring it as a probability
distribution over bitstrings, then ranking/decoding those bitstrings — is not
tied to the variational parameter model. This mixin owns that capability's own
state (the measured ``_best_probs`` distribution and the ``_decode_solution_fn``
decode hook) and leans only on the shared :class:`~divi.qprog.QuantumProgram`
seams (:meth:`~divi.qprog.QuantumProgram._assemble_pipeline`,
:meth:`~divi.qprog.QuantumProgram._run_pipeline`).

The host program must provide three things:

* a cooperative ``_build_pipelines`` returning a
  :class:`~divi.pipeline.PipelineSet` — the mixin extends it with the ``"sample"``
  entry via ``super()._build_pipelines().with_(...)``;
* a ``meta_circuit_factories`` mapping carrying a ``"cost_circuit"`` (and
  optionally a ``"sample_circuit"``) :class:`~divi.circuits.MetaCircuit` to seed
  the sample pipeline;
* a ``_coerce_sample_params`` method turning caller-supplied parameters into the
  array fed to the pipeline.

The third hook is where any model-specific parameter handling lives — e.g.
:class:`~divi.qprog.variational_quantum_algorithm.VariationalQuantumAlgorithm`
implements it with a shape check against ``n_layers * n_params_per_layer`` and a
fallback to the trained ``_best_params``.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple
from warnings import warn

import numpy as np
import numpy.typing as npt

from divi.pipeline import CircuitPipeline, PipelineSet, ResultFormat
from divi.pipeline._compilation import _extract_param_set_idx
from divi.pipeline.stages import CircuitSpecStage, MeasurementStage

if TYPE_CHECKING:
    # Type-check the mixin as if mixed into its host, so ``super()`` calls and the
    # inherited seams it relies on resolve. At runtime the base is ``object`` — it
    # is a genuine mixin. The host today is always a VariationalQuantumAlgorithm.
    from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm

    _SamplingMixinBase = VariationalQuantumAlgorithm
else:
    _SamplingMixinBase = object


class SolutionEntry(NamedTuple):
    """A solution entry with bitstring, probability, and optional decoded value.

    Args:
        bitstring: Binary string representing a computational basis state.
        prob: Measured probability in range [0.0, 1.0].
        decoded: Optional problem-specific decoded representation. Defaults to None.
        energy: Optional objective energy for this solution. Defaults to None.
    """

    bitstring: str
    prob: float
    decoded: Any | None = None
    energy: float | None = None


class SolutionSamplingMixin(_SamplingMixinBase):
    """Adds the discrete-solution-sampling capability to a quantum program.

    Mix in before the host program (e.g. VQE/QAOA/PCE) for programs that extract a
    bitstring solution. It registers the ``"sample"`` pipeline and exposes the
    solution API (:meth:`sample_solution`, :meth:`get_top_solutions`,
    :attr:`best_probs`). Programs without it (e.g. data-bound QNN/CustomVQA) simply
    do not have these members — calling them raises ``AttributeError`` rather than
    silently returning nothing.

    The mixin owns its result state (``_best_probs``) and decode hook
    (``_decode_solution_fn``); the host supplies ``meta_circuit_factories``, a
    cooperative ``_build_pipelines``, and ``_coerce_sample_params`` (see the module
    docstring for the full contract).
    """

    def __init__(
        self,
        *args,
        decode_solution_fn: Callable[[str], Any] | None = None,
        **kwargs,
    ):
        """Initialize the solution-sampling state.

        Args:
            decode_solution_fn: Function mapping a bitstring (e.g. ``"0101"``) to a
                problem-specific decoded representation (e.g. a list of indices, a
                numpy array, or a custom object). Called by
                :meth:`get_top_solutions` when ``include_decoded=True`` and by
                subclass solution decoding. Defaults to the identity function.
            ``*args``, ``**kwargs``: Forwarded to the next class in the MRO (the
                host program).
        """
        super().__init__(*args, **kwargs)
        self._best_probs: dict[str, dict[str, float]] = {}
        self._decode_solution_fn = decode_solution_fn or (lambda bitstring: bitstring)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # The mixin's cooperative ``_build_pipelines`` must run before the host's
        # base one — i.e. the mixin must precede the host in the MRO — or the
        # "sample" pipeline is silently dropped. Reject the misordering at
        # class-definition time.
        from divi.qprog.variational_quantum_algorithm import (
            VariationalQuantumAlgorithm,
        )

        mro = cls.__mro__
        if VariationalQuantumAlgorithm in mro and mro.index(
            SolutionSamplingMixin
        ) > mro.index(VariationalQuantumAlgorithm):
            raise TypeError(
                f"{cls.__name__} must list SolutionSamplingMixin before "
                f"VariationalQuantumAlgorithm in its bases; otherwise the mixin's "
                f"_build_pipelines is bypassed and the sample pipeline is omitted."
            )

    def _build_pipelines(self) -> PipelineSet:
        return (
            super()
            ._build_pipelines()
            .with_(
                "sample",
                self._assemble_pipeline(
                    CircuitSpecStage(),
                    MeasurementStage(),
                    result_format=ResultFormat.PROBS,
                ),
                lambda: self.meta_circuit_factories.get(
                    "sample_circuit", self.meta_circuit_factories["cost_circuit"]
                ),
            )
        )

    @property
    def _sample_pipeline(self) -> CircuitPipeline:
        """The solution-sampling pipeline."""
        return self._pipelines["sample"]

    @property
    def best_probs(self) -> dict[str, dict[str, float]]:
        """Get normalized probabilities for the best parameters.

        This property provides access to the probability distribution computed
        by running measurement circuits with the best parameters found during
        optimization. The distribution maps bitstrings (computational basis states)
        to their measured probabilities.

        The probabilities are normalized and have deterministic ordering when
        iterated (dictionary insertion order is preserved in Python 3.7+).

        Returns:
            dict[str, dict[str, float]]: Dictionary mapping parameter-set keys to
                bitstring probability dictionaries. Bitstrings are binary strings
                (e.g., "0101"), values are probabilities in range [0.0, 1.0].
                Returns an empty dict if final computation has not been performed.

        Raises:
            RuntimeError: If attempting to access probabilities before running
                the algorithm with final computation enabled.

        Note:
            To populate this distribution, you must run the algorithm with
            `perform_final_computation=True` (the default):

            >>> program.run(perform_final_computation=True)
            >>> probs = program.best_probs

        Example:
            >>> program.run()
            >>> probs = program.best_probs
            >>> for bitstring, prob in probs.items():
            ...     print(f"{bitstring}: {prob:.2%}")
            0101: 42.50%
            1010: 31.20%
            ...
        """
        if not self._best_probs:
            warn(
                "best_probs is empty. Either optimization has not been run yet, "
                "or final computation was not performed. Call run() to execute "
                "the optimization.",
                UserWarning,
                stacklevel=2,
            )
        return self._best_probs.copy()

    def get_top_solutions(
        self, n: int = 10, *, min_prob: float = 0.0, include_decoded: bool = False
    ) -> list[SolutionEntry]:
        """Get the top-N solutions sorted by probability.

        This method extracts the most probable solutions from the measured
        probability distribution. Solutions are sorted by probability (descending)
        with deterministic tie-breaking using lexicographic ordering of bitstrings.

        Args:
            n (int): Maximum number of solutions to return. Must be non-negative.
                If n is 0 or negative, returns an empty list. If n exceeds the
                number of available solutions (after filtering), returns all
                available solutions. Defaults to 10.
            min_prob (float): Minimum probability threshold for including solutions.
                Only solutions with probability >= min_prob will be included.
                Must be in range [0.0, 1.0]. Defaults to 0.0 (no filtering).
            include_decoded (bool): Whether to populate the `decoded` field of
                each SolutionEntry by calling the `decode_solution_fn` provided
                in the constructor. If False, the decoded field will be None.
                Defaults to False.

        Returns:
            list[SolutionEntry]: List of solution entries sorted by probability
                (descending), then by bitstring (lexicographically ascending)
                for deterministic tie-breaking. Returns an empty list if no
                probability distribution is available or n <= 0.

        Raises:
            RuntimeError: If probability distribution is not available because
                optimization has not been run or final computation was not performed.
            ValueError: If min_prob is not in range [0.0, 1.0] or n is negative.

        Note:
            The probability distribution must be computed by running the algorithm
            with `perform_final_computation=True` (the default):

            >>> program.run(perform_final_computation=True)
            >>> top_10 = program.get_top_solutions(n=10)

        Example:
            >>> # Get top 5 solutions with probability >= 5%
            >>> program.run()
            >>> solutions = program.get_top_solutions(n=5, min_prob=0.05)
            >>> for sol in solutions:
            ...     print(f"{sol.bitstring}: {sol.prob:.2%}")
            1010: 42.50%
            0101: 31.20%
            1100: 15.30%
            0011: 8.50%
            1111: 2.50%

            >>> # Get solutions with decoding
            >>> solutions = program.get_top_solutions(n=3, include_decoded=True)
            >>> for sol in solutions:
            ...     print(f"{sol.bitstring} -> {sol.decoded}")
            1010 -> [0, 2]
            0101 -> [1, 3]
            ...
        """
        # Validate inputs
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if not (0.0 <= min_prob <= 1.0):
            raise ValueError(f"min_prob must be in range [0.0, 1.0], got {min_prob}")

        # Handle edge case: n == 0
        if n == 0:
            return []

        # Require probability distribution to exist
        if not self._best_probs:
            raise RuntimeError(
                "No probability distribution available. The final computation step "
                "must be performed to compute the probability distribution. "
                "Call run(perform_final_computation=True) to execute optimization "
                "and compute the distribution."
            )
        # Extract the probability distribution (nested by parameter set)
        # _best_probs structure: {tag: {bitstring: prob}}
        probs_dict = next(iter(self._best_probs.values()))

        # Filter by minimum probability and get top n sorted by probability (descending),
        # then bitstring (ascending) for deterministic tie-breaking
        top_items = sorted(
            filter(
                lambda bitstring_prob: bitstring_prob[1] >= min_prob, probs_dict.items()
            ),
            key=lambda bitstring_prob: (-bitstring_prob[1], bitstring_prob[0]),
        )[:n]

        # Build result list (decode on demand)
        return [
            SolutionEntry(
                bitstring=bitstring,
                prob=prob,
                decoded=(
                    self._decode_solution_fn(bitstring) if include_decoded else None
                ),
            )
            for bitstring, prob in top_items
        ]

    def sample_solution(
        self,
        params: npt.NDArray[np.float64] | None = None,
        **kwargs,
    ) -> "SolutionSamplingMixin":
        """Run the final measurement and decode the solution.

        Called by ``run()`` (with ``params=None``, falling back to the host's
        trained parameters) after optimization completes. It can also be called
        directly with externally-provided ``params`` when you already have trained
        parameters (e.g. from a prior ``run()``, a checkpoint, or external
        training) and only need to sample the circuit — skipping the EXPECTATION
        jobs that ``run()`` would otherwise dispatch during optimization.

        When called with explicit ``params``, this method does NOT mutate the
        host's optimizer state. Only the measurement-side attributes are updated:
        ``_best_probs``, ``_total_circuit_count``, ``_total_run_time``, and
        subclass-specific solution fields (e.g. ``solution_bitstring`` for QAOA,
        ``_eigenstate`` for VQE).

        Parameter coercion (default-fallback and shape validation) is delegated to
        the host's :meth:`_coerce_sample_params`.

        Args:
            params: Optional parameter set to evaluate. When ``None`` (the
                default), the host falls back to its trained parameters.
            **kwargs: Subclass-specific keyword arguments.

        Returns:
            The program itself, for method chaining.

        Note:
            Subclasses override this method to add their algorithm-specific
            decoding step. They should call ``super().sample_solution(params)``
            to perform coercion and the measurement-pipeline dispatch, then read
            from ``self._best_probs`` to extract algorithm-specific solution state.
        """
        params_arr = self._coerce_sample_params(params)
        self._run_solution_measurement_for(np.atleast_2d(params_arr))
        return self

    def _run_solution_measurement_for(
        self, param_sets: npt.NDArray[np.float64]
    ) -> None:
        """Execute sample circuits via the pipeline for the provided parameter sets."""
        result = self._run_pipeline("sample", param_sets=np.atleast_2d(param_sets))

        indexed = {
            _extract_param_set_idx(key, default=0): value
            for key, value in result.items()
        }
        self._best_probs = dict(sorted(indexed.items()))
