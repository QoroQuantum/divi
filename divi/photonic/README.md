# divi.photonic — middleware for TBI photonic hardware

A self-contained proof-of-concept demonstrating divi as middleware for
loop-based time-bin interferometer (TBI) photonic processors — the
architecture behind Orca Computing's PT-series.

## Why a parallel subpackage

TBI samplers are not universal unitary machines. They return
**photon-count tuples per mode**, not bitstrings with Pauli-eigenvalue
semantics, so this subpackage deliberately does *not* reuse
`CircuitRunner`, `MetaCircuit`, or divi's Pauli-grouping adaptive-shot
allocator — those abstractions are structurally bound to the gate model.
Photonic programs flow through their own IR, sampler protocol, and
observable evaluators. Ensemble-level scheduling and result aggregation
can be adopted later if the parallelism story demands it.

## What's included

Two algorithm families, both running on a multi-loop TBI (PT-2-style
nested loops), share a common set of primitives:

- **Variational QUBO** (`TBIVariationalQUBO`). Ports the four-config
  `(n_photons × parity)` sweep from [`orcacomputing/quantumqubo`][qq]
  (Apache-2.0), generalised to multi-loop via the vendored simulator.
  Adam + parameter-shift rule at `±π/6`. Exact match on small QUBO
  instances vs. brute force.
- **Boson Sampling Born Machine** (`TBIBornMachine` + `BarsAndStripes`).
  Generative model training with a mixture-of-Gaussians MMD loss —
  "train classically, deploy quantumly". Toy problem: 3×3 bars-and-
  stripes (12 valid patterns in a 512-bitstring search space;
  trained model beats the random baseline by ~6×).

## Public API

```python
from divi.photonic import (
    # IR
    PhotonicProgram,
    PhotonicSamples,
    # Sampler protocol
    PhotonicSampler,
    SimulatedTBISampler,
    # Observables
    PhotonicObservable,
    ParityQUBO,
    MMDSampleLoss,
    # Algorithms
    TBIVariationalQUBO,
    TBIBornMachine,
    BarsAndStripes,
)
```

See [`tutorials/tbi_qubo.py`](tutorials/tbi_qubo.py) and
[`tutorials/tbi_born_machine.py`](tutorials/tbi_born_machine.py) for
end-to-end examples.

## Layout

```
divi/photonic/
├── README.md                 ← this file
├── __init__.py               ← public API
├── _ir.py                    ← PhotonicProgram + PhotonicSamples
├── _observables.py           ← ParityQUBO, MMDSampleLoss, protocol
├── _samplers.py              ← SimulatedTBISampler (wraps vendored sim)
├── _adam.py                  ← numpy Adam (no torch dep)
├── _tbi_qubo.py              ← TBIVariationalQUBO
├── _tbi_born_machine.py      ← TBIBornMachine + BarsAndStripes
├── tutorials/
│   ├── tbi_qubo.py           ← P1 demo
│   └── tbi_born_machine.py   ← P2 demo
└── _vendor/
    └── loop_progressive_simulator/  ← Apache-2.0 vendored subset
```

## Simulator provenance

The TBI simulator is a vendored subset of
[`orcacomputing/loop-progressive-simulator`][lps] (Apache-2.0, active)
pinned at commit
[`941400e28594270391ec9bde14cb1bbf0af2a8a2`][lps-commit] (2026-02-11),
accompanying arXiv:2411.16873 on progressive simulation of loop-based
boson sampling. Vendored (not pip-installed) because upstream uses
absolute imports and is not published as an installable distribution.
Only the modules needed for the public API are included
(`bscircuits`, `step_simulator`, `fock_states`, `number_basis`, `utils`,
`factorial_table`), with imports rewritten to relative and SPDX headers
added. Upstream license is preserved at `_vendor/loop_progressive_simulator/LICENSE.upstream`.

The QUBO algorithm layer (four-config sweep, parameter-shift loop,
parity mapping) is ported with attribution from
[`orcacomputing/quantumqubo`][qq] (Apache-2.0, stale since 2021 — ported,
not depended on, to avoid 2021-era numpy/numba constraints).

## Future seams

Any class satisfying the `PhotonicSampler` protocol slots in via
`sampler=`. The obvious next implementations:

- `RemoteTBISampler` — Orca cloud endpoint, once access is available.
- `CUDAQTBISampler` — routed through Orca's contribution to NVIDIA
  CUDA-Q, once their photonic simulator open-sources in coordination
  with that release.

## Running the tutorials

```bash
.venv/bin/python -m divi.photonic.tutorials.tbi_qubo
.venv/bin/python -m divi.photonic.tutorials.tbi_born_machine
```

On a laptop:

- [`tutorials/tbi_qubo.py`](tutorials/tbi_qubo.py) — 4 variables,
  `loop_lengths=(1, 2)`, 20 updates × 4 configs × 200 shots. Recovers
  the brute-force optimum. ~3 min.
- [`tutorials/tbi_born_machine.py`](tutorials/tbi_born_machine.py) —
  3×3 bars-and-stripes (12 valid patterns in a 512-bitstring search
  space), `loop_lengths=(1, 2)`, 9 modes, 3 photons, 20 updates × 80
  shots. MMD halves, model places ~13% of mass on target patterns vs
  ~2.3% random baseline (**~6× uplift**). ~80 s.

## Theoretical caveats baked into the design

1. TBI is a sampler, not a universal unitary machine — no gate-model
   API surface exposed.
2. The bitstring-coverage argument for the four-config QUBO sweep was
   originally derived for single-loop TBI; for multi-loop geometries we
   verify it empirically via `TBIQUBOResult.bitstring_coverage`.
3. Parity mapping is algorithm-specific: QUBO uses it, BSBM doesn't.
   The `PhotonicObservable` tagged-protocol lives outside the sampler
   so either can be used without friction.
4. Parameter-shift shot cost: `(2·n_bs + 1)·shots` per Adam step.
   Reported explicitly in result objects.

[qq]: https://github.com/orcacomputing/quantumqubo
[lps]: https://github.com/orcacomputing/loop-progressive-simulator
[lps-commit]: https://github.com/orcacomputing/loop-progressive-simulator/tree/941400e28594270391ec9bde14cb1bbf0af2a8a2
