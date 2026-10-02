# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""``hybrid`` decomposer that partitions a QUBO by community structure."""

from hybrid import traits
from hybrid.core import Runnable
from hybrid.exceptions import EndOfStream
from hybrid.utils import bqm_induced_by

from divi.qprog.problems._partitioning_config import QUBOPartitioningConfig
from divi.qprog.problems._qubo_partitioning_utils import (
    bqm_to_sparse,
    partition_by_method,
)


class CommunityDecomposer(traits.ProblemDecomposer, traits.SISO, Runnable):
    """Structure-aware QUBO decomposer that partitions by community structure.

    A drop-in ``hybrid`` decomposer — like D-Wave's ``EnergyImpactDecomposer`` or
    ``ComponentDecomposer`` — that groups strongly-coupled variables and cuts weak
    couplings, so little energy is lost at partition boundaries. Connected components
    are separated first, then each component is clustered to honour the size budget.
    Successive calls roll through the resulting clusters, one subproblem per iteration.

    The default ``"modularity"`` method (Louvain) picks the community count itself
    and is the strongest general-purpose choice across structured, dense, and
    constrained QUBOs; ``"spectral"`` suits mainly sparse-geometric instances.

    Best on problems with community structure; for featureless (dense, unstructured)
    QUBOs, D-Wave's ``EnergyImpactDecomposer`` is also a reasonable choice.

    Args:
        config: Size limits, clustering method and seed.
        silent_rewind: If ``False``, raise ``hybrid.exceptions.EndOfStream`` once
            all clusters are exhausted (used by ``hybrid.Unwind``, which is how
            :meth:`BinaryOptimizationProblem.decompose` drives this decomposer).

    Raises:
        ImportError: If the ``qubo-decompose`` extra is not installed.
        TypeError: If ``config`` is not a
            :class:`~divi.qprog.problems.QUBOPartitioningConfig`.
    """

    _reproducible = True

    def __init__(
        self,
        config: QUBOPartitioningConfig,
        *,
        silent_rewind: bool = True,
        **runopts,
    ):
        super().__init__(**runopts)
        if not isinstance(config, QUBOPartitioningConfig):
            raise TypeError(
                "config must be a QUBOPartitioningConfig, got "
                f"{type(config).__name__}."
            )
        self.config = config
        self.silent_rewind = silent_rewind
        self._rolling_bqm = None
        self._iter_clusters = None

    def __repr__(self):
        return f"{self}(config={self.config!r}, silent_rewind={self.silent_rewind!r})"

    def _get_iter_clusters(self, bqm):
        variables, _h, sigma = bqm_to_sparse(bqm)
        clusters = partition_by_method(sigma, self.config)
        return iter([[variables[i] for i in cl] for cl in clusters])

    def next(self, state, **runopts):
        """Emit the next cluster as the subproblem (one hybrid decomposition step)."""
        silent_rewind = runopts.get("silent_rewind", self.silent_rewind)
        bqm = state.problem

        if bqm.num_variables <= 1:
            # One trivial cluster per pass, so ``Unwind`` sees its end.
            if bqm == self._rolling_bqm and not silent_rewind:
                self._rolling_bqm = None
                raise EndOfStream
            self._rolling_bqm = bqm
            return state.updated(subproblem=bqm)

        # Content equality, not identity: hybrid.State.updated() deep-copies
        # ``problem`` each call, so an identity check would never reach EndOfStream.
        if bqm != self._rolling_bqm:
            self._rolling_bqm = bqm
            self._iter_clusters = self._get_iter_clusters(bqm)
        assert self._iter_clusters is not None  # set above or on a prior call
        try:
            cluster = next(self._iter_clusters)
        except StopIteration:
            if not silent_rewind:
                self._rolling_bqm = None
                raise EndOfStream
            self._iter_clusters = self._get_iter_clusters(bqm)
            cluster = next(self._iter_clusters)

        sample = state.samples.change_vartype(bqm.vartype).first.sample
        return state.updated(subproblem=bqm_induced_by(bqm, cluster, sample))
