# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Trotterization inputs, results and the strategy protocol."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
from warnings import warn

import numpy as np
from qiskit import transpile
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter

from divi.hamiltonians._term_ops import _spo_to_qiskit_basis_gates

if TYPE_CHECKING:
    from divi.qprog.initial_states import InitialState


@dataclass(frozen=True)
class TrotterSpec:
    """What a trotterization strategy works on: the Hamiltonian and, for a time
    evolution, the state it starts from, what is measured and for how long.

    :class:`~divi.pipeline.CircuitPipeline` caches forward passes by the
    identity of this object, so a program builds its spec once and reuses it.
    """

    hamiltonian: SparsePauliOp
    """Hamiltonian to split into Trotter steps."""

    initial_state: "InitialState | None" = None
    """State the evolution starts from; ``None`` outside a time evolution."""

    observable: SparsePauliOp | tuple[SparsePauliOp, ...] | None = None
    """Observable(s) measured after the evolution; ``None`` when none is."""

    time: float | Parameter | None = None
    """Evolution time, symbolic for a parametric template; ``None`` outside a
    time evolution."""


@dataclass(frozen=True)
class TrotterizationResult:
    """Output of one trotterization step: the Hamiltonian to build the circuit
    from, plus (for sampling strategies) the exact sequence of sampled terms."""

    effective_hamiltonian: SparsePauliOp
    """Simplified Hamiltonian the observable/circuit is built from. For a
    sampling result, the first step's draw."""

    sampled_terms: tuple[SparsePauliOp, ...] | None = None
    """For sampling strategies such as QDrift, one independent draw per Trotter
    step, each holding the drawn terms in order with repeats kept (one entry
    per draw). ``None`` for deterministic strategies."""

    def synthesize_evolution(
        self,
        qc: QuantumCircuit,
        *,
        time: float,
        n_steps: int,
        order: int,
        qubits: Sequence[int],
        basis_gates: Sequence[str],
    ) -> QuantumCircuit:
        """Append this result's time-evolution gates to ``qc``.

        Every path approximates ``exp(-i t H)``. A sampling result
        (``sampled_terms`` set) is first order and carries its own step count:
        each draw is one step at ``time / len(sampled_terms)``, applying one
        evolution gate per drawn term (sampling-with-replacement multiplicities
        kept). A deterministic result synthesises ``effective_hamiltonian`` via
        :class:`~qiskit.circuit.library.PauliEvolutionGate` over ``n_steps``
        steps (:class:`~qiskit.synthesis.LieTrotter` for ``order == 1``, else
        :class:`~qiskit.synthesis.SuzukiTrotter`). Every path is then lowered
        to ``basis_gates``; the returned circuit is a new object.

        Raises:
            ValueError: If a sampling result is synthesised with ``order != 1``.
            RuntimeError: If the circuit cannot be lowered to ``basis_gates``.
        """
        if self.sampled_terms is not None:
            if order != 1:
                raise ValueError(
                    f"Sampled Trotterization supports only order=1, got order={order}."
                )
            # Concatenating keeps every step's terms in order, one gate per draw.
            _spo_to_qiskit_basis_gates(
                qc,
                SparsePauliOp.sum(list(self.sampled_terms)),
                time / len(self.sampled_terms),
                qubits,
            )
        elif self.effective_hamiltonian.size >= 2:
            synthesis = (
                LieTrotter(reps=n_steps, preserve_order=True)
                if order == 1
                else SuzukiTrotter(order=order, reps=n_steps, preserve_order=True)
            )
            qc.append(
                PauliEvolutionGate(
                    self.effective_hamiltonian, time=time, synthesis=synthesis
                ),
                qubits,
            )
        else:
            _spo_to_qiskit_basis_gates(qc, self.effective_hamiltonian, time, qubits)

        # ``optimization_level=0`` keeps lowering to a gate-by-gate substitution.
        try:
            return transpile(qc, basis_gates=list(basis_gates), optimization_level=0)
        except Exception as exc:
            raise RuntimeError(
                "Failed to lower the Trotter-synthesised circuit to the "
                "requested basis-gate set. This usually means "
                "PauliEvolutionGate synthesis emitted a gate the QASM2 "
                f"emitter does not handle. Supported gates: {sorted(basis_gates)}."
            ) from exc


def _warn_truncation_no_op(
    keep_fraction: float | None, keep_top_n: int | None, n_terms: int
) -> bool:
    """Emit the truncation-is-a-no-op warning if applicable. Returns True when warned."""
    if keep_fraction is not None and keep_fraction == 1.0:
        warn(
            "keep_fraction is 1.0 (no truncation); returning the full Hamiltonian.",
            UserWarning,
        )
        return True
    if keep_top_n is not None and keep_top_n >= n_terms:
        warn(
            "keep_top_n is greater than or equal to the number of terms; "
            "returning the full Hamiltonian.",
            UserWarning,
        )
        return True
    return False


class TrotterizationStrategy(Protocol):
    """Trotterization strategy protocol."""

    def process_hamiltonian(
        self,
        spec: TrotterSpec,
        *,
        n_steps: int = 1,
        rng: np.random.Generator | None = None,
    ) -> TrotterizationResult:
        """Return a trotterization result of ``spec.hamiltonian`` for an
        ``n_steps`` evolution without retaining call state; sampling strategies
        draw independently per step."""
        ...

    def process_hamiltonian_batch(
        self,
        spec: TrotterSpec,
        n_samples: int,
        *,
        n_steps: int = 1,
        rng: np.random.Generator | None = None,
    ) -> list[TrotterizationResult]:
        """Return ``n_samples`` results from one spec.

        The default calls :meth:`process_hamiltonian` ``n_samples`` times;
        strategies with expensive deterministic preprocessing (e.g. QDrift's
        keep/sample split) override this to share that work across the batch.
        """
        return [
            self.process_hamiltonian(spec, n_steps=n_steps, rng=rng)
            for _ in range(n_samples)
        ]
