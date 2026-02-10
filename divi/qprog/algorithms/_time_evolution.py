# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, get_args

import numpy as np
import pennylane as qml

from divi.backends import convert_counts_to_probs, reverse_dict_endianness
from divi.circuits import CircuitBundle, CircuitTag, MetaCircuit
from divi.qprog._expectation import _batched_expectation
from divi.qprog._hamiltonians import (
    ExactTrotterization,
    TrotterizationStrategy,
    _clean_hamiltonian,
    _get_terms_iterable,
    _is_empty_hamiltonian,
    _is_multi_term_sum,
    convert_hamiltonian_to_pauli_string,
)
from divi.qprog.quantum_program import QuantumProgram

_INITIAL_STATE_LITERAL = Literal["Zeros", "Superposition", "Ones"]


class TimeEvolution(QuantumProgram):
    """Quantum program for Hamiltonian time evolution.

    Simulates the evolution of a quantum state under a Hamiltonian using
    Trotter-Suzuki decomposition. Uses Divi's TrotterizationStrategy
    (ExactTrotterization, QDrift) for term selection and approximation.
    """

    def __init__(
        self,
        hamiltonian: qml.operation.Operator,
        trotterization_strategy: TrotterizationStrategy | None = None,
        time: float = 1.0,
        n_steps: int = 1,
        order: int = 1,
        initial_state: _INITIAL_STATE_LITERAL = "Zeros",
        observable: qml.operation.Operator | None = None,
        **kwargs,
    ):
        """Initialize TimeEvolution.

        Args:
            hamiltonian: Hamiltonian to evolve under.
            trotterization_strategy: Strategy for term selection (ExactTrotterization, QDrift).
                Defaults to ExactTrotterization().
            time: Evolution time t (e^(-iHt)).
            n_steps: Number of Trotter steps.
            order: Suzuki-Trotter order (1 or even).
            initial_state: One of ``"Zeros"`` (``|0...0>``), ``"Superposition"``
                (``|+...+>``), or ``"Ones"`` (``|1...1>``).
            observable: If None, measure qml.probs(); else qml.expval(observable).
            **kwargs: Passed to QuantumProgram (backend, seed, progress_queue, etc.).
        """
        super().__init__(**kwargs)

        if trotterization_strategy is None:
            trotterization_strategy = ExactTrotterization()

        hamiltonian_clean, _ = _clean_hamiltonian(hamiltonian)
        if _is_empty_hamiltonian(hamiltonian_clean):
            raise ValueError("Hamiltonian contains only constant terms.")

        if initial_state not in get_args(_INITIAL_STATE_LITERAL):
            raise ValueError(
                f"initial_state must be one of {get_args(_INITIAL_STATE_LITERAL)}, got {initial_state!r}"
            )

        self._hamiltonian = hamiltonian_clean
        self.trotterization_strategy = trotterization_strategy
        self.time = time
        self.n_steps = n_steps
        self.order = order
        self.initial_state = initial_state
        self.observable = observable
        self._circuit_wires = tuple(hamiltonian_clean.wires)
        self.n_qubits = len(self._circuit_wires)

        self._hamiltonian_samples: list[qml.operation.Operator] | None = None
        self.results: dict[str, Any] = {}

    def run(self, **kwargs) -> tuple[int, float]:
        """Execute time evolution.

        Returns:
            tuple[int, float]: (total_circuit_count, total_run_time).
        """
        strategy = self.trotterization_strategy
        n_samples = getattr(strategy, "n_hamiltonians_per_iteration", 1)

        if (
            n_samples > 1
            and self.observable is not None
            and self.backend.supports_expval
        ):
            raise ValueError(
                "Multi-sample QDrift with observable and expval backend is not supported. "
                "Use a shot-based backend or set observable=None for probs."
            )

        self._hamiltonian_samples = [
            strategy.process_hamiltonian(self._hamiltonian) for _ in range(n_samples)
        ]

        self._curr_circuits = self._generate_circuits(**kwargs)

        if self.observable is not None and self.backend.supports_expval:
            kwargs["ham_ops"] = convert_hamiltonian_to_pauli_string(
                self.observable, self.n_qubits
            )

        self._dispatch_circuits_and_process_results(**kwargs)

        return self.total_circuit_count, self.total_run_time

    def _build_ops(self, hamiltonian: qml.operation.Operator) -> list:
        """Build circuit ops: initial state, evolution, measurement."""
        ops = []

        # Initial state
        if self.initial_state == "Ones":
            for wire in self._circuit_wires:
                ops.append(qml.PauliX(wires=wire))
        elif self.initial_state == "Superposition":
            for wire in self._circuit_wires:
                ops.append(qml.Hadamard(wires=wire))

        # Evolution: e^(-iHt)
        n_terms = len(hamiltonian) if _is_multi_term_sum(hamiltonian) else 1
        if n_terms >= 2:
            evo = qml.adjoint(
                qml.TrotterProduct(
                    hamiltonian, time=self.time, n=self.n_steps, order=self.order
                )
            )
            ops.append(evo)
        else:
            term = (
                hamiltonian
                if not _is_multi_term_sum(hamiltonian)
                else _get_terms_iterable(hamiltonian)[0]
            )
            ops.append(qml.evolve(term, coeff=self.time))

        return ops

    def _generate_circuits(self, **kwargs) -> list[CircuitBundle]:
        """Generate circuits for each Hamiltonian sample."""
        if self._hamiltonian_samples is None:
            raise RuntimeError(
                "_hamiltonian_samples must be set before _generate_circuits; call run() first."
            )

        circuit_bundles: list[CircuitBundle] = []
        use_probs = self.observable is None

        for ham_id, sample_hamiltonian in enumerate(self._hamiltonian_samples):
            ops = self._build_ops(sample_hamiltonian)

            if use_probs:
                measurement = qml.probs()
                meta = MetaCircuit(
                    source_circuit=qml.tape.QuantumScript(
                        ops=ops, measurements=[measurement]
                    ),
                    symbols=np.array([], dtype=object),
                    measurement_groups_override=((),),
                    postprocessing_fn_override=lambda x: x,
                )
            else:
                measurement = qml.expval(self.observable)
                meta = MetaCircuit(
                    source_circuit=qml.tape.QuantumScript(
                        ops=ops, measurements=[measurement]
                    ),
                    symbols=np.array([], dtype=object),
                    grouping_strategy=(
                        "_backend_expval" if self.backend.supports_expval else "wires"
                    ),
                )

            bundle = meta.initialize_circuit_from_params(
                [], param_idx=0, hamiltonian_id=ham_id
            )
            circuit_bundles.append(bundle)

        return circuit_bundles

    def _group_results(
        self, results: dict[CircuitTag, dict[str, int]]
    ) -> dict[int, dict[int, dict[tuple[str, int], list[dict[str, int]]]]]:
        """Group results by param_id, hamiltonian_id, (qem_name, qem_id)."""
        return self._group_results_by_tag(results)

    def _post_process_results(
        self, results: dict[CircuitTag, dict[str, int]], **kwargs
    ) -> dict[str, Any]:
        """Post-process results: convert to probs or expectation, average for QDrift."""
        use_probs = self.observable is None
        grouped = self._group_results(results)

        if use_probs:
            ham_probs: list[dict[str, float]] = []
            for _p, ham_dict in grouped.items():
                for _ham_id, qem_groups in ham_dict.items():
                    qem_probs: list[dict[str, float]] = []
                    for _qem_key, shots_list in sorted(
                        qem_groups.items(), key=lambda x: x[0][1]
                    ):
                        merged_counts = self._merge_shot_histograms(shots_list)
                        probs = convert_counts_to_probs(
                            {"_merged": merged_counts}, self.backend.shots
                        )["_merged"]
                        qem_probs.append(probs)
                    if qem_probs:
                        ham_probs.append(self._average_probabilities(qem_probs))

            merged_probs = self._average_probabilities(ham_probs)
            self.results = {
                "probs": reverse_dict_endianness({"_merged": merged_probs})["_merged"]
            }

            return self.results

        # Expval path
        ops = self._build_ops(self._hamiltonian_samples[0])
        meta = MetaCircuit(
            source_circuit=qml.tape.QuantumScript(
                ops=ops, measurements=[qml.expval(self.observable)]
            ),
            symbols=np.array([], dtype=object),
            grouping_strategy=(
                "_backend_expval" if self.backend.supports_expval else "wires"
            ),
        )

        wire_order = tuple(reversed(self._circuit_wires))
        ham_energies: list[float] = []
        ham_ops = kwargs.get("ham_ops")
        ham_ops_list = ham_ops.split(";") if ham_ops is not None else None

        def _append_expval(meta, marginal_results: list[np.ndarray]) -> float:
            pl_exp = meta.postprocessing_fn(marginal_results)
            return pl_exp.item() if hasattr(pl_exp, "item") else float(pl_exp)

        for _p, ham_dict in grouped.items():
            for _ham_id, qem_groups in ham_dict.items():
                qem_sorted = sorted(qem_groups.items(), key=lambda x: x[0][1])
                qem_energies: list[float] = []

                if self.backend.supports_expval:
                    if ham_ops_list is None:
                        raise ValueError(
                            "ham_ops required for expval backend but not provided."
                        )
                    for _, shots_dicts in qem_sorted:
                        exp_arr = np.array(
                            [[d[op] for op in ham_ops_list] for d in shots_dicts]
                        ).T
                        marginal = exp_arr.flatten()
                        qem_energies.append(_append_expval(meta, [marginal]))
                else:
                    for _, shots_dicts in qem_sorted:
                        marginal_results: list[np.ndarray] = []
                        for shots_dict, obs_group in zip(
                            shots_dicts, meta.measurement_groups
                        ):
                            exp_matrix = _batched_expectation(
                                [shots_dict], list(obs_group), wire_order
                            )
                            marginal_results.append(exp_matrix[:, 0])
                        qem_energies.append(_append_expval(meta, marginal_results))

                if qem_energies:
                    ham_energies.append(float(np.mean(qem_energies)))

        expval = float(np.mean(ham_energies)) if ham_energies else 0.0
        self.results = {"expval": expval}
        return self.results
