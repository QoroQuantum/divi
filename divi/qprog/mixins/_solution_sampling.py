# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Mixin adding the discrete-solution-sampling capability to a quantum program.

Sampling a solution — running a circuit, measuring it as a probability
distribution over bitstrings, then ranking/decoding those bitstrings — is not
tied to the variational parameter model. This mixin owns that capability's own
state (the measured ``_best_probs`` distribution and the ``_decode_solution_fn``
decode hook) and leans only on the shared :meth:`~divi.qprog.QuantumProgram.evaluate`
entry point, to which it hands a :func:`~divi.pipeline.sample_preprocessor`.

The host program must provide two things:

* a ``_initial_spec`` method returning the cost
  :class:`~divi.circuits.MetaCircuit` (the program's seed circuit);
* a ``_resolve_sample_params`` method that maps caller-supplied parameters
  (including ``None``) to the numeric array fed to
  :meth:`~divi.qprog.QuantumProgram.evaluate`. The base
  :meth:`SolutionSamplingMixin.sample_solution` calls it for the ``None``
  fallback; a host without it must pass explicit ``params``.

That hook is where any model-specific parameter handling lives — e.g.
:class:`~divi.qprog.variational_quantum_algorithm.VariationalQuantumAlgorithm`
implements it with a shape check against ``n_layers * n_params_per_layer`` and a
fallback to the trained ``_best_params``. VQE/QAOA/PCE additionally call it in
their ``sample_solution`` overrides to validate explicit params before decoding.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, Self, cast
from warnings import warn

import numpy as np
import numpy.typing as npt

from divi.backends import CircuitRunner
from divi.pipeline import CircuitPreprocessor, sample_preprocessor

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
        energy: Optional objective energy for this solution. ``None`` from the
            default probability-ranked path; populated only by feasibility-aware
            retrieval, e.g. :meth:`~divi.qprog.algorithms.QAOA.get_top_solutions`
            with ``feasibility="filter"`` or ``"repair"`` (which scores each
            bitstring via the problem's ``compute_energy``). Defaults to None.
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
    (``_decode_solution_fn``); the host supplies ``_initial_spec`` and
    ``_resolve_sample_params`` (see the module docstring for the full contract).
    """

    def __init__(
        self,
        *args,
        decode_solution_fn: Callable[[str], Any] | None = None,
        sampling_backend: CircuitRunner | None = None,
        **kwargs,
    ):
        """Initialise the solution-sampling state.

        Args:
            decode_solution_fn: Function mapping a bitstring (e.g. ``"0101"``) to a
                problem-specific decoded representation (e.g. a list of indices, a
                numpy array, or a custom object). Called by
                :meth:`get_top_solutions` when ``include_decoded=True`` and by
                subclass solution decoding. Defaults to the identity function.
            sampling_backend: Backend used only for solution sampling. ``None``
                reuses the host program's backend.
            ``*args``, ``**kwargs``: Forwarded to the next class in the MRO (the
                host program).
        """
        super().__init__(*args, **kwargs)
        self._best_probs: dict[int, dict[str, float]] = {}
        self._decode_solution_fn = decode_solution_fn or (lambda bitstring: bitstring)
        self._sampling_backend = sampling_backend

    def _preprocessors(self) -> tuple[CircuitPreprocessor, ...]:
        """Expose the sample routine for introspection alongside the host's."""
        return (*super()._preprocessors(), self._sample_preprocessor())

    def _sample_preprocessor(self) -> CircuitPreprocessor:
        """The preprocessor sampling the prepared state in the computational basis."""
        return sample_preprocessor()

    @property
    def sampling_backend(self) -> CircuitRunner | None:
        """Backend dedicated to solution sampling, when configured."""
        return self._sampling_backend

    @property
    def best_probs(self) -> dict[int, dict[str, float]]:
        """Get normalised probabilities for the best parameters.

        This property provides access to the probability distribution computed
        by running measurement circuits with the best parameters found during
        optimisation. It maps each parameter-set index to that set's distribution
        over bitstrings (computational basis states).

        The probabilities are normalised and iterate in a deterministic order.

        Returns:
            dict[int, dict[str, float]]: Dictionary mapping each parameter-set
                index to a bitstring probability dictionary. Bitstrings are binary
                strings (e.g., "0101"), values are probabilities in range
                [0.0, 1.0]. Returns an empty dict if final computation has not
                been performed.

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
            >>> for idx, distribution in probs.items():
            ...     print(f"parameter set {idx}:")
            ...     for bitstring, prob in distribution.items():
            ...         print(f"  {bitstring}: {prob:.2%}")
            parameter set 0:
              0101: 42.50%
              1010: 31.20%
            ...
        """
        if not self._best_probs:
            warn(
                "best_probs is empty. Either optimisation has not been run yet, "
                "or final computation was not performed. Call run() to execute "
                "the optimisation.",
                UserWarning,
                stacklevel=2,
            )
        return self._best_probs.copy()

    def _single_distribution(self) -> dict[str, float]:
        """The distribution of the first sampled parameter set.

        The usual flow samples one set (the trained best params); only an
        explicit multi-row ``sample_solution()`` leaves several, in which case
        the first is used and the caller is pointed at ``best_probs``.

        Call this directly from the public method that needs it — the warning
        it emits is reported at that method's own call site.

        Raises:
            RuntimeError: If no distribution has been measured yet.
        """
        if not self._best_probs:
            raise RuntimeError(
                "No probability distribution available. The final computation step "
                "must be performed to compute the probability distribution. "
                "Call run(perform_final_computation=True) to execute optimisation "
                "and compute the distribution."
            )
        if len(self._best_probs) > 1:
            warn(
                f"{len(self._best_probs)} parameter sets were sampled; "
                "only the first (lowest-index) set is used. "
                "Access best_probs for the per-set distributions.",
                UserWarning,
                stacklevel=3,
            )
        return next(iter(self._best_probs.values()))

    def get_correlations(self) -> npt.NDArray[np.float64]:
        r"""Get the two-point spin correlations of the sampled state.

        Returns the matrix :math:`Z_{ij} = \langle Z_i Z_j \rangle`, evaluated
        over the measured distribution as
        :math:`\sum_x p(x)\, s_i(x)\, s_j(x)` with bits read as spins in the
        Ising convention (``0`` maps to ``+1``, ``1`` maps to ``-1``). Entries
        near ``+1`` mark wires that agree across the distribution, near ``-1``
        wires that disagree, and near ``0`` wires the state leaves undecided.

        Index ``i`` refers to position ``i`` of the measured bitstrings, the
        same ordering ``decode_solution_fn`` uses, so for a graph problem it is
        the wire label at that position.

        Returns:
            npt.NDArray[np.float64]: Symmetric ``(n_wires, n_wires)`` matrix
                whose diagonal is exactly 1.0.

        Raises:
            RuntimeError: If no distribution is available because optimisation
                has not been run or final computation was not performed.
            ValueError: If the distribution mixes bitstring widths.

        Note:
            Probabilities are used as measured. If several parameter sets were
            sampled, only the first is used and a warning is emitted.

        Example:
            >>> program.run(perform_final_computation=True)
            >>> correlations = program.get_correlations()
            >>> correlations[0, 1]  # how wires 0 and 1 relate
            0.87
        """
        return _spin_moments(self._single_distribution())[1]

    def get_magnetisations(self) -> npt.NDArray[np.float64]:
        r"""Get the single-site spin expectations of the sampled state.

        Returns the vector :math:`m_i = \langle Z_i \rangle`, evaluated over
        the measured distribution as :math:`\sum_x p(x)\, s_i(x)` under the
        same spin convention and wire ordering as :meth:`get_correlations`.
        Entries near ``+/-1`` mark wires the state has decided; entries near
        ``0`` mark wires it has not.

        Returns:
            npt.NDArray[np.float64]: Vector of length ``n_wires`` with entries
                in range [-1.0, 1.0].

        Raises:
            RuntimeError: If no distribution is available because optimisation
                has not been run or final computation was not performed.
            ValueError: If the distribution mixes bitstring widths.

        Note:
            Probabilities are used as measured. If several parameter sets were
            sampled, only the first is used and a warning is emitted.

        Example:
            >>> program.run(perform_final_computation=True)
            >>> program.get_magnetisations()
            array([ 0.92, -0.88,  0.04])
        """
        return _spin_moments(self._single_distribution())[0]

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
                optimisation has not been run or final computation was not performed.
            ValueError: If min_prob is not in range [0.0, 1.0] or n is negative.

        Note:
            The probability distribution must be computed by running the algorithm
            with `perform_final_computation=True` (the default):

            >>> program.run(perform_final_computation=True)
            >>> top_10 = program.get_top_solutions(n=10)

            If several parameter sets were sampled (an explicit multi-row
            ``sample_solution(params=...)``), ranking uses only the first
            (lowest-index) set and emits a warning; use :attr:`best_probs` to
            access every set's distribution.

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

        probs_dict = self._single_distribution()

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
        *,
        backend: CircuitRunner | None = None,
        **kwargs,
    ) -> Self:
        """Run the final measurement and decode the solution.

        Called by ``run()`` (with ``params=None``, falling back to the host's
        trained parameters) after optimisation completes. It can also be called
        directly with externally-provided ``params`` when you already have trained
        parameters (e.g. from a prior ``run()``, a checkpoint, or external
        training) and only need to sample the circuit — skipping the EXPECTATION
        jobs that ``run()`` would otherwise dispatch during optimisation.

        When called with explicit ``params``, this method does NOT mutate the
        host's optimizer state. Only the measurement-side attributes are updated:
        ``_best_probs``, ``_total_circuit_count``, ``_total_run_time``, and
        subclass-specific solution fields (e.g. ``solution_bitstring`` for QAOA,
        ``_eigenstate`` for VQE).

        Args:
            params: Parameter set to evaluate. Must be a numeric array; pass
                ``None`` only when the host has an override that resolves the
                fallback (e.g. :class:`~divi.qprog.VariationalQuantumAlgorithm`
                falls back to ``_best_params``).
            backend: One-call sampling backend override. ``None`` uses the
                configured sampling backend, falling back to the host backend.
            **kwargs: Subclass-specific keyword arguments.

        Returns:
            The program itself, for method chaining.

        Note:
            Subclasses override this method to add their algorithm-specific
            decoding step. They should call ``super().sample_solution(params)``
            to perform the measurement-pipeline dispatch, then read from
            ``self._best_probs`` to extract algorithm-specific solution state.
        """
        if params is None:
            # Defer the fallback to the host's resolution hook (VQA maps None to
            # the trained _best_params). A host with no resolver and no explicit
            # params is a contract violation — fail loud rather than sampling at
            # ``np.asarray(None)`` == NaN.
            resolver = getattr(self, "_resolve_sample_params", None)
            if resolver is None:
                raise TypeError(
                    "sample_solution() received params=None on a host that does "
                    "not define _resolve_sample_params. Pass explicit params or "
                    "mix in a host (e.g. VariationalQuantumAlgorithm) that resolves "
                    "the trained-parameter fallback."
                )
            params = resolver(None)
        params_arr = np.asarray(params, dtype=np.float64)
        selected_backend = backend if backend is not None else self._sampling_backend
        self._run_solution_measurement_for(
            np.atleast_2d(params_arr), backend=selected_backend
        )
        return self

    def _run_solution_measurement_for(
        self,
        param_sets: npt.NDArray[np.float64],
        *,
        backend: CircuitRunner | None = None,
    ) -> None:
        """Sample the prepared state for the provided parameter sets."""
        result = cast(
            "dict[int, dict[str, float] | list[dict[str, float]]]",
            self.evaluate(
                np.atleast_2d(param_sets),
                self._sample_preprocessor(),
                backend=backend,
            ),
        )
        self._best_probs = {
            idx: _average_probabilities(value) for idx, value in result.items()
        }


def _spin_moments(
    probs_dict: dict[str, float],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """First and second spin moments of a bitstring distribution.

    Bits map to spins in the Ising convention (``0 -> +1``, ``1 -> -1``), with
    bitstring position ``i`` denoting wire ``i`` — the same ordering
    ``decode_solution_fn`` uses.

    Returns:
        The magnetisation vector and the correlation matrix, in that order.

    Raises:
        ValueError: If the distribution mixes bitstring widths.
    """
    bitstrings = list(probs_dict)
    widths = {len(bitstring) for bitstring in bitstrings}
    if len(widths) > 1:
        raise ValueError(
            f"All bitstrings must have the same length to index wires "
            f"consistently, but the distribution mixes widths {sorted(widths)}."
        )

    probabilities = np.fromiter(probs_dict.values(), dtype=np.float64)
    bits = np.frombuffer("".join(bitstrings).encode(), dtype=np.uint8).reshape(
        len(bitstrings), widths.pop()
    ) - ord("0")
    spins = 1.0 - 2.0 * bits.astype(np.float64)

    magnetisations = probabilities @ spins
    correlations = spins.T @ (probabilities[:, np.newaxis] * spins)
    np.fill_diagonal(correlations, 1.0)
    return magnetisations, correlations


def _average_probabilities(
    value: dict[str, float] | list[dict[str, float]],
) -> dict[str, float]:
    """Average one or more probability distributions."""
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    bitstrings = set().union(*(probs.keys() for probs in value))
    return {
        bitstring: sum(probs.get(bitstring, 0.0) for probs in value) / len(value)
        for bitstring in sorted(bitstrings)
    }
