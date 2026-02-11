# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, get_args
from warnings import warn

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
        initial_state: _INITIAL_STATE_LITERAL | str = "Zeros",
        observable: qml.operation.Operator | None = None,
        ensemble_size: int | None = None,
        time_points: list[float] | np.ndarray | None = None,
        **kwargs,
    ):
        """Initialize TimeEvolution.

        Args:
            hamiltonian: Hamiltonian to evolve under.
            trotterization_strategy: Strategy for term selection (ExactTrotterization, QDrift).
                Defaults to ExactTrotterization().
            time: Evolution time t (e^(-iHt)). Ignored when ``time_points`` is set.
            n_steps: Number of Trotter steps.
            order: Suzuki-Trotter order (1 or even).
            initial_state: One of ``"Zeros"`` (``|0...0>``), ``"Superposition"``
                (``|+...+>``), or ``"Ones"`` (``|1...1>``), or a string
                (e.g. ``"01+-"``) specifying the state of each qubit.
                Characters: '0' -> |0>, '1' -> |1>, '+' -> |+>, '-' -> |->.
            observable: If None, measure qml.probs(); else qml.expval(observable).
            ensemble_size: Number of Hamiltonian samples to average over.
                If None (default), falls back to the strategy's settings.
            time_points: Optional list of evolution times. When provided, circuits
                for every ``(time_point, ensemble_member)`` pair are generated and
                submitted as a **single batch job**.  Results are stored as a list
                of per-time-point dicts in ``self.results["trajectory"]``.
            **kwargs: Passed to QuantumProgram (backend, seed, progress_queue, etc.).
        """
        super().__init__(**kwargs)

        if trotterization_strategy is None:
            trotterization_strategy = ExactTrotterization()

        hamiltonian_clean, _ = _clean_hamiltonian(hamiltonian)
        if _is_empty_hamiltonian(hamiltonian_clean):
            raise ValueError("Hamiltonian contains only constant terms.")

        self._circuit_wires = tuple(hamiltonian_clean.wires)
        self.n_qubits = len(self._circuit_wires)

        if initial_state not in get_args(_INITIAL_STATE_LITERAL):
            # Check for valid custom string consisting of '0', '1', '+', '-'
            is_valid_custom = isinstance(initial_state, str) and all(
                c in "01+-" for c in initial_state
            )
            if is_valid_custom:
                if len(initial_state) != self.n_qubits:
                    raise ValueError(
                        f"initial_state string length ({len(initial_state)}) "
                        f"must match number of qubits ({self.n_qubits})."
                    )
            else:
                raise ValueError(
                    f"initial_state must be one of {get_args(_INITIAL_STATE_LITERAL)} "
                    f"or a string of '0', '1', '+', '-', got {initial_state!r}"
                )

        self._hamiltonian = hamiltonian_clean
        self.trotterization_strategy = trotterization_strategy
        self.time = time
        self.n_steps = n_steps
        self.order = order
        self.initial_state = initial_state
        self.observable = observable
        self.ensemble_size = ensemble_size
        self.time_points = list(time_points) if time_points is not None else None

        if ensemble_size is not None and ensemble_size < 1:
            raise ValueError(f"ensemble_size must be >= 1, got {ensemble_size}")

        self._hamiltonian_samples: list[qml.operation.Operator] | None = None
        self.results: dict[str, Any] = {}

    def run(self, **kwargs) -> tuple[int, float]:
        """Execute time evolution.

        Returns:
            tuple[int, float]: (total_circuit_count, total_run_time).
        """
        strategy = self.trotterization_strategy
        n_samples = self.ensemble_size

        if n_samples is None:
            n_samples = getattr(strategy, "n_hamiltonians_per_iteration", 1)
        elif (
            hasattr(strategy, "n_hamiltonians_per_iteration")
            and strategy.n_hamiltonians_per_iteration != 1
            and strategy.n_hamiltonians_per_iteration != n_samples
        ):
            warn(
                f"Both TimeEvolution(ensemble_size={n_samples}) and "
                f"{strategy.__class__.__name__}(n_hamiltonians_per_iteration={strategy.n_hamiltonians_per_iteration}) "
                f"are set. ensemble_size={n_samples} will take precedence.",
                UserWarning,
            )

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

        self._n_samples = n_samples
        self._curr_circuits = self._generate_circuits(**kwargs)

        if self.observable is not None and self.backend.supports_expval:
            kwargs["ham_ops"] = convert_hamiltonian_to_pauli_string(
                self.observable, self.n_qubits
            )

        self._dispatch_circuits_and_process_results(**kwargs)

        return self.total_circuit_count, self.total_run_time

    def _build_ops_for_time(
        self, hamiltonian: qml.operation.Operator, t: float
    ) -> list:
        """Build circuit ops for a specific evolution time."""
        ops = []

        # Initial state
        if self.initial_state == "Ones":
            for wire in self._circuit_wires:
                ops.append(qml.PauliX(wires=wire))
        elif self.initial_state == "Superposition":
            for wire in self._circuit_wires:
                ops.append(qml.Hadamard(wires=wire))
        elif all(c in "01+-" for c in self.initial_state):
            for wire, char in zip(self._circuit_wires, self.initial_state):
                if char == "1":
                    ops.append(qml.PauliX(wires=wire))
                elif char == "+":
                    ops.append(qml.Hadamard(wires=wire))
                elif char == "-":
                    ops.append(qml.PauliX(wires=wire))
                    ops.append(qml.Hadamard(wires=wire))

        # Evolution: e^(-iHt)
        n_terms = len(hamiltonian) if _is_multi_term_sum(hamiltonian) else 1
        if n_terms >= 2:
            evo = qml.TrotterProduct(
                hamiltonian, time=t, n=self.n_steps, order=self.order
            )
            ops.append(evo)
        else:
            term = (
                hamiltonian
                if not _is_multi_term_sum(hamiltonian)
                else _get_terms_iterable(hamiltonian)[0]
            )
            ops.append(qml.evolve(term, coeff=t))

        return ops

    def _build_ops(self, hamiltonian: qml.operation.Operator) -> list:
        """Build circuit ops using self.time (backward compat)."""
        return self._build_ops_for_time(hamiltonian, self.time)

    def _generate_circuits(self, **kwargs) -> list[CircuitBundle]:
        """Generate circuits for each (time_point, Hamiltonian sample) pair.

        When ``self.time_points`` is set, circuits are generated for every
        combination and tagged with ``param_id = time_index`` so the
        post-processor can reconstruct per-time-step results.
        """
        if self._hamiltonian_samples is None:
            raise RuntimeError(
                "_hamiltonian_samples must be set before _generate_circuits; call run() first."
            )

        circuit_bundles: list[CircuitBundle] = []
        use_probs = self.observable is None
        time_list = self.time_points if self.time_points is not None else [self.time]

        for time_idx, t in enumerate(time_list):
            if t == 0:
                continue  # t=0 is the initial state; no circuit needed

            for ham_id, sample_hamiltonian in enumerate(self._hamiltonian_samples):
                ops = self._build_ops_for_time(sample_hamiltonian, t)

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
                            "_backend_expval"
                            if self.backend.supports_expval
                            else "wires"
                        ),
                    )

                bundle = meta.initialize_circuit_from_params(
                    [], param_idx=time_idx, hamiltonian_id=ham_id
                )
                circuit_bundles.append(bundle)

        return circuit_bundles

    def _group_results(
        self, results: dict[CircuitTag, dict[str, int]]
    ) -> dict[int, dict[int, dict[tuple[str, int], list[dict[str, int]]]]]:
        """Group results by param_id, hamiltonian_id, (qem_name, qem_id)."""
        return self._group_results_by_tag(results)

    def _maybe_reverse_endianness(self, probs: dict[str, float]) -> dict[str, float]:
        """Reverse bitstring endianness only when the backend returns little-endian."""
        if self.backend.little_endian_bitstrings:
            return reverse_dict_endianness({"_": probs})["_"]
        return probs

    def _post_process_results(
        self, results: dict[CircuitTag, dict[str, int]], **kwargs
    ) -> dict[str, Any]:
        """Post-process results: convert to probs or expectation, average for QDrift.

        When ``time_points`` was used, results are grouped by ``param_id``
        (= time index) and a ``trajectory`` list of per-time-point dicts is
        produced.  The single-time-point path remains backward compatible.
        """
        use_probs = self.observable is None
        grouped = self._group_results(results)

        if use_probs:
            if self.time_points is not None:
                # --- Multi-time-point path ---
                trajectory: list[dict[str, float]] = []
                init_state = self.initial_state
                # Build t=0 probs from the initial state string
                t0_probs = (
                    {init_state: 1.0}
                    if init_state not in get_args(_INITIAL_STATE_LITERAL)
                    else {"0" * self.n_qubits: 1.0}
                )

                for time_idx, t in enumerate(self.time_points):
                    if t == 0:
                        trajectory.append(t0_probs)
                        continue

                    if time_idx not in grouped:
                        trajectory.append(t0_probs)  # fallback
                        continue

                    ham_dict = grouped[time_idx]
                    ham_probs: list[dict[str, float]] = []
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

                    merged = self._average_probabilities(ham_probs)
                    trajectory.append(self._maybe_reverse_endianness(merged))

                self.results = {"trajectory": trajectory}
                return self.results

            # --- Single-time-point path (unchanged) ---
            ham_probs_single: list[dict[str, float]] = []
            for _p, ham_dict in grouped.items():
                for _ham_id, qem_groups in ham_dict.items():
                    qem_probs_s: list[dict[str, float]] = []
                    for _qem_key, shots_list in sorted(
                        qem_groups.items(), key=lambda x: x[0][1]
                    ):
                        merged_counts = self._merge_shot_histograms(shots_list)
                        probs = convert_counts_to_probs(
                            {"_merged": merged_counts}, self.backend.shots
                        )["_merged"]
                        qem_probs_s.append(probs)
                    if qem_probs_s:
                        ham_probs_single.append(
                            self._average_probabilities(qem_probs_s)
                        )

            merged_probs = self._average_probabilities(ham_probs_single)
            self.results = {"probs": self._maybe_reverse_endianness(merged_probs)}

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
