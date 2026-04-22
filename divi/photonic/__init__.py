# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Middleware for photonic hardware: loop-based time-bin interferometers.

This subpackage is divi's entry point for Orca-style multi-loop TBI
photonic processors. It is deliberately a parallel shelf to the qubit
stack — photonic samples are photon-count tuples, not bitstrings with
Pauli-eigenvalue semantics, so this subpackage does not go through
``CircuitRunner`` / ``MetaCircuit`` / the Pauli-group adaptive-shot
allocator.

Public API:

- :class:`PhotonicProgram`, :class:`PhotonicSamples` — the IR and result
  types.
- :class:`PhotonicSampler` (Protocol) + :class:`SimulatedTBISampler` — the
  sampler surface. Future remote / CUDA-Q / Orca-SDK backends plug in
  here.
- :class:`PhotonicObservable` (Protocol) + :class:`ParityQUBO` and
  :class:`MMDSampleLoss` — loss / observable evaluators.
- :class:`TBIVariationalQUBO` — ported from ``orcacomputing/quantumqubo``
  (Apache-2.0).
- :class:`TBIBornMachine` with :class:`BarsAndStripes` — a Boson Sampling
  Born Machine generative model.

The underlying simulator is the vendored subset of
``orcacomputing/loop-progressive-simulator`` (arXiv:2411.16873).
"""

from divi.photonic._ir import (
    PhotonicProgram,
    PhotonicSamples,
    count_beamsplitters,
)
from divi.photonic._observables import (
    MMDSampleLoss,
    ObservableResult,
    ParityQUBO,
    PhotonicObservable,
    parity_bitstring,
)
from divi.photonic._samplers import PhotonicSampler, SimulatedTBISampler
from divi.photonic._tbi_born_machine import BarsAndStripes, TBIBornMachine
from divi.photonic._tbi_qubo import TBIVariationalQUBO

__all__ = [
    "BarsAndStripes",
    "MMDSampleLoss",
    "ObservableResult",
    "ParityQUBO",
    "PhotonicObservable",
    "PhotonicProgram",
    "PhotonicSampler",
    "PhotonicSamples",
    "SimulatedTBISampler",
    "TBIBornMachine",
    "TBIVariationalQUBO",
    "count_beamsplitters",
    "parity_bitstring",
]
