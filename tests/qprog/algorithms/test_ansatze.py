# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import (
    CRXGate,
    CXGate,
    CZGate,
    HGate,
    RXGate,
    RYGate,
    RZGate,
    SwapGate,
    UGate,
)
from qiskit.quantum_info import Operator, SparsePauliOp
from scipy.linalg import expm

from divi.qprog import (
    VQE,
    GenericLayerAnsatz,
    HartreeFockAnsatz,
    LUCJAnsatz,
    QAOAAnsatz,
    QCCAnsatz,
    UCCSDAnsatz,
)
from divi.qprog.algorithms._ansatze import (
    _emit_givens_rotation,
    _resolve_spin_counts,
    _rotation_schedule,
    _uccsd_excitations,
)
from tests.qprog.algorithms._helpers import gate_names, gate_qubits


def _build_circuit(ansatz, params, n_qubits, n_layers, **kwargs) -> QuantumCircuit:
    return ansatz.build(params, n_qubits, n_layers, **kwargs)


def _occupied_from_label(label: str) -> set[int]:
    """Qubit indices set in a measured bitstring (qubit 0 leftmost, as divi
    reports them)."""
    return {i for i, bit in enumerate(label) if bit == "1"}


def _interleaved(blocked: int, n_spatial: int) -> int:
    """Interleaved qubit for a blocked spin-orbital index."""
    spin, spatial = divmod(blocked, n_spatial)
    return 2 * spatial + spin


# --- Test GenericLayerAnsatz ---
class TestGenericLayerAnsatz:
    """Tests for the GenericLayerAnsatz class."""

    @pytest.mark.parametrize(
        "gate_sequence, entangler, layout",
        [
            ([RXGate], CXGate, "linear"),
            ([RYGate, RZGate], CZGate, "circular"),
            ([UGate], None, "all-to-all"),
            ([RXGate], CXGate, [(0, 2), (1, 3)]),
        ],
    )
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_initialization_valid(self, gate_sequence, entangler, layout):
        try:
            GenericLayerAnsatz(
                gate_sequence=gate_sequence,
                entangler=entangler,
                entangling_layout=layout,
            )
        except (ValueError, TypeError):
            pytest.fail("GenericLayerAnsatz initialization failed with valid inputs.")

    def test_initialization_rejects_string_gate(self):
        with pytest.raises(TypeError, match="must be a Qiskit Gate subclass"):
            GenericLayerAnsatz(gate_sequence=[RXGate, "rx"])

    def test_initialization_rejects_gate_instance(self):
        with pytest.raises(TypeError, match="must be a Qiskit Gate subclass"):
            GenericLayerAnsatz(gate_sequence=[RXGate(0.0)])

    def test_initialization_rejects_multi_qubit_in_gate_sequence(self):
        with pytest.raises(ValueError, match="must be a 1-qubit gate"):
            GenericLayerAnsatz(gate_sequence=[CXGate])

    def test_initialization_rejects_single_qubit_entangler(self):
        with pytest.raises(ValueError, match="must be a 2-qubit gate"):
            GenericLayerAnsatz(gate_sequence=[RXGate], entangler=RYGate)

    def test_initialization_rejects_parameterized_entangler(self):
        with pytest.raises(ValueError, match="must take 0 parameters"):
            GenericLayerAnsatz(gate_sequence=[RXGate], entangler=CRXGate)

    def test_initialization_invalid_layout_string(self):
        with pytest.raises(ValueError, match="Unknown entangling_layout:"):
            GenericLayerAnsatz(
                gate_sequence=[RXGate],
                entangler=CXGate,
                entangling_layout="invalid_layout",
            )

    def test_initialization_warns_on_layout_without_entangler(self):
        with pytest.warns(UserWarning, match="`entangler` is None"):
            GenericLayerAnsatz(
                gate_sequence=[RXGate], entangler=None, entangling_layout="linear"
            )

    @pytest.mark.parametrize(
        "gate_sequence, n_qubits, expected_params",
        [
            ([RXGate], 4, 4),
            ([RXGate, RZGate], 4, 8),
            ([UGate], 3, 9),
            ([RYGate, UGate], 2, 8),  # 1 + 3 params per qubit
        ],
    )
    def test_n_params_per_layer(self, gate_sequence, n_qubits, expected_params):
        ansatz = GenericLayerAnsatz(gate_sequence=gate_sequence)
        assert ansatz.n_params_per_layer(n_qubits) == expected_params

    def test_n_params_per_layer_rejects_parameter_free_ansatz(self):
        ansatz = GenericLayerAnsatz(gate_sequence=[HGate])
        with pytest.raises(ValueError, match="must define at least one trainable"):
            ansatz.n_params_per_layer(n_qubits=2)

    def test_build_no_entangler(self):
        n_qubits, n_layers = 2, 2
        ansatz = GenericLayerAnsatz(gate_sequence=[RXGate, RYGate], entangler=None)
        n_params = n_layers * ansatz.n_params_per_layer(n_qubits)
        params = ParameterVector("p", n_params)

        qc = _build_circuit(ansatz, list(params), n_qubits, n_layers)

        names = gate_names(qc)
        # 2 qubits * 2 layers * 2 gates = 8 gates
        assert len(names) == 8
        assert all(name in ("rx", "ry") for name in names)
        # First two gates are on qubit 0 then qubit 1 with the first two params.
        assert qc.data[0].operation.params[0] == params[0]
        assert qc.data[2].operation.params[0] == params[2]

    def test_build_with_entangler(self):
        n_qubits, n_layers = 3, 1
        ansatz = GenericLayerAnsatz(
            gate_sequence=[RXGate], entangler=CXGate, entangling_layout="linear"
        )
        n_params = n_layers * ansatz.n_params_per_layer(n_qubits)
        params = ParameterVector("p", n_params)

        qc = _build_circuit(ansatz, list(params), n_qubits, n_layers)

        # 3 RX + 2 CNOTs for linear layout on 3 qubits
        assert gate_names(qc) == ["rx", "rx", "rx", "cx", "cx"]
        assert gate_qubits(qc)[3:] == [[0, 1], [1, 2]]

    def test_build_with_swap_entangler(self):
        """Non-CX/CZ 2-qubit entanglers work — SwapGate isn't in any whitelist."""
        n_qubits, n_layers = 3, 1
        ansatz = GenericLayerAnsatz(
            gate_sequence=[RXGate], entangler=SwapGate, entangling_layout="linear"
        )
        n_params = n_layers * ansatz.n_params_per_layer(n_qubits)
        qc = _build_circuit(
            ansatz, list(ParameterVector("p", n_params)), n_qubits, n_layers
        )
        assert gate_names(qc) == ["rx", "rx", "rx", "swap", "swap"]


# --- Test QAOAAnsatz ---
class TestQAOAAnsatz:
    """Tests for the QAOAAnsatz class."""

    @pytest.mark.parametrize(
        "n_qubits, expected",
        [(1, 1), (2, 3), (3, 6), (4, 8)],
    )
    def test_n_params_per_layer(self, n_qubits, expected):
        """Per-layer param count: 1 / 3 / 2n for n=1 / 2 / >=3."""
        assert QAOAAnsatz().n_params_per_layer(n_qubits=n_qubits) == expected

    def test_build_structure(self):
        """Each layer = Hadamards + ZZ ring (CX-RZ-CX) + RY field; trailing Hadamards."""
        n_qubits, n_layers = 4, 3
        ansatz = QAOAAnsatz()
        n_params = n_layers * ansatz.n_params_per_layer(n_qubits)
        params = ParameterVector("p", n_params)

        qc = _build_circuit(ansatz, list(params), n_qubits, n_layers)
        names = gate_names(qc)

        # (n_layers + 1) Hadamard layers of size n_qubits.
        assert names.count("h") == (n_layers + 1) * n_qubits
        # Ring of n_qubits ZZ rotations per layer, each decomposed to CX-RZ-CX.
        assert names.count("cx") == 2 * n_layers * n_qubits
        assert names.count("rz") == n_layers * n_qubits
        # One local-field RY per qubit per layer.
        assert names.count("ry") == n_layers * n_qubits

    def test_build_n_qubits_one(self):
        """n=1 special case: only local-field rotations between Hadamards."""
        ansatz = QAOAAnsatz()
        n_qubits, n_layers = 1, 2
        params = ParameterVector("p", n_layers * ansatz.n_params_per_layer(n_qubits))
        qc = _build_circuit(ansatz, list(params), n_qubits, n_layers)
        names = gate_names(qc)
        assert names.count("h") == n_layers + 1
        assert names.count("ry") == n_layers
        # No two-qubit interaction emitted for the single-qubit case.
        assert "cx" not in names

    def test_build_n_qubits_two(self):
        """n=2 special case: one ZZ rotation (no wrap) and two RYs per layer."""
        ansatz = QAOAAnsatz()
        n_qubits, n_layers = 2, 2
        params = ParameterVector("p", n_layers * ansatz.n_params_per_layer(n_qubits))
        qc = _build_circuit(ansatz, list(params), n_qubits, n_layers)
        names = gate_names(qc)
        assert names.count("h") == (n_layers + 1) * n_qubits
        # One ZZ rotation per layer → CX-RZ-CX.
        assert names.count("cx") == 2 * n_layers
        assert names.count("rz") == n_layers
        assert names.count("ry") == n_layers * n_qubits

    def test_custom_local_field(self):
        """``local_field=RXGate`` swaps RY for RX in the field layer."""
        ansatz = QAOAAnsatz(local_field=RXGate)
        n_qubits, n_layers = 3, 1
        params = ParameterVector("p", n_layers * ansatz.n_params_per_layer(n_qubits))
        qc = _build_circuit(ansatz, list(params), n_qubits, n_layers)
        names = gate_names(qc)
        assert names.count("rx") == n_qubits
        assert "ry" not in names

    def test_invalid_local_field(self):
        with pytest.raises(ValueError, match="local_field must be"):
            QAOAAnsatz(local_field=HGate)


# --- Test Chemistry Ansaetze ---
@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda a: a.n_params_per_layer(n_qubits=4), id="n_params_per_layer"
        ),
        pytest.param(lambda a: a.build([0.1, 0.2, 0.3], 4, 1), id="build"),
    ],
)
@pytest.mark.parametrize("ansatz", [UCCSDAnsatz(), HartreeFockAnsatz()], ids=type)
def test_chemistry_ansatz_names_a_missing_electron_count(ansatz, call):
    """A chemistry ansatz builds excitations from a reference state, so it cannot do
    anything without ``n_electrons``. Omitting it used to surface as a comparison
    against ``NoneType`` from inside PennyLane, naming neither the missing setting
    nor the ansatz that needed it."""
    with pytest.raises(ValueError, match="requires n_electrons"):
        call(ansatz)


def test_qcc_ansatz_names_a_missing_electron_count():
    """QCC derives its parameter count from the register alone, so only ``build``
    needs the electron count."""
    with pytest.raises(ValueError, match="requires n_electrons"):
        QCCAnsatz().build([0.1] * 6, 4, 1)


class TestUCCSDAnsatz:
    """Tests for the UCCSDAnsatz class."""

    def test_n_params_per_layer(self):
        assert UCCSDAnsatz().n_params_per_layer(n_qubits=4, n_electrons=2) == 3

    def test_build_emits_qiskit_circuit(self):
        n_electrons, n_qubits, n_layers = 2, 4, 1
        ansatz = UCCSDAnsatz()
        n_params = n_layers * ansatz.n_params_per_layer(
            n_qubits, n_electrons=n_electrons
        )
        params = ParameterVector("p", n_params)
        qc = _build_circuit(
            ansatz, list(params), n_qubits, n_layers, n_electrons=n_electrons
        )
        # Should produce a non-empty Qiskit circuit on the requested qubit count.
        assert qc.num_qubits == n_qubits
        assert len(qc.data) > 0

    def test_reference_state_occupies_the_interleaved_spin_orbitals(
        self, default_test_simulator, default_optimizer
    ):
        """Alpha occupies ``2p``, beta ``2p + 1``. With zero angles the circuit
        is deterministic, so every shot lands on the reference determinant.

        A total-electron-count check would pass with the two spins transposed,
        so this asserts the occupied set.
        """
        n_qubits, n_alpha, n_beta = 8, 3, 1
        vqe = _make_uccsd_vqe(
            n_qubits,
            n_alpha + n_beta,
            default_test_simulator,
            default_optimizer,
            n_alpha=n_alpha,
            n_beta=n_beta,
        )

        vqe.sample_solution(params=np.zeros(vqe.n_params))

        probs = next(iter(vqe.best_probs.values()))
        assert len(probs) == 1
        occupied = _occupied_from_label(next(iter(probs)))
        assert occupied == {2 * p for p in range(n_alpha)} | {
            2 * p + 1 for p in range(n_beta)
        }

    @pytest.mark.parametrize("index", range(26))
    def test_each_excitation_rotates_only_the_modes_it_names(
        self, index, default_test_simulator, default_optimizer
    ):
        """One parameter at ``theta`` must put support on exactly two
        determinants: the reference, and the one reached by moving electrons
        between the *interleaved* qubits named by ``excitation_list[index]``.

        This localizes the class of error a qubit permutation introduced.
        Jordan-Wigner parity strings are built over the mapper's mode order, so
        relabeling qubits afterwards left each excitation's Z-string covering
        the wrong modes -- 25 of 26 excitations were then neither
        ``exp(-theta A)`` nor ``exp(+theta A)``. The 14 whose spurious Z factors
        happened to evaluate to +1 on the reference determinant still produced
        the right state, so only an energy assertion caught it, and it pointed
        at nothing in particular.

        Run through the backend rather than a bare ``Statevector`` so this
        exercises divi's full circuit-submission path, including the QASM2
        lowering the excitation gates require.

        Also pins the positional alignment between ``excitation_list[index]``
        and parameter ``index``, which ``_uccsd_amplitude_seed`` indexes on.
        """
        n_qubits, n_electrons, theta = 8, 4, 0.37
        excitations = _uccsd_excitations(n_qubits // 2, (2, 2))
        assert len(excitations) == 26

        vqe = _make_uccsd_vqe(
            n_qubits, n_electrons, default_test_simulator, default_optimizer
        )
        params = np.zeros(vqe.n_params)
        params[index] = theta

        vqe.sample_solution(params=params)
        probs = next(iter(vqe.best_probs.values()))

        assert len(probs) == 2, "a single excitation must span two determinants"
        reference = {0, 1, 2, 3}
        occupied, unoccupied = excitations[index]
        excited = (reference - {_interleaved(i, 4) for i in occupied}) | {
            _interleaved(i, 4) for i in unoccupied
        }
        assert {frozenset(_occupied_from_label(label)) for label in probs} == {
            frozenset(reference),
            frozenset(excited),
        }

        # Shot-noise tolerance on 5000 shots; the support above is the sharp part.
        excited_label = next(
            label for label in probs if _occupied_from_label(label) == excited
        )
        assert probs[excited_label] == pytest.approx(np.sin(theta) ** 2, abs=0.02)


class TestHartreeFockAnsatz:
    """Tests for the HartreeFockAnsatz class."""

    def test_n_params_per_layer(self):
        assert HartreeFockAnsatz().n_params_per_layer(n_qubits=4, n_electrons=2) == 3

    def test_build_emits_qiskit_circuit(self):
        n_electrons, n_qubits, n_layers = 2, 4, 2
        ansatz = HartreeFockAnsatz()
        n_params = n_layers * ansatz.n_params_per_layer(
            n_qubits, n_electrons=n_electrons
        )
        params = ParameterVector("p", n_params)
        qc = _build_circuit(
            ansatz, list(params), n_qubits, n_layers, n_electrons=n_electrons
        )
        assert qc.num_qubits == n_qubits
        assert len(qc.data) > 0


# --- Test QCCAnsatz ---
class TestQCCAnsatz:
    """Tests for the QCCAnsatz class."""

    def test_n_params_per_layer(self):
        # n_qubits RY + 3*(n_qubits-1) entanglers.
        assert QCCAnsatz().n_params_per_layer(n_qubits=4) == 13
        assert QCCAnsatz().n_params_per_layer(n_qubits=2) == 5

    def test_build_structure(self):
        n_electrons, n_qubits, n_layers = 2, 4, 1
        ansatz = QCCAnsatz()
        n_params = n_layers * ansatz.n_params_per_layer(n_qubits)
        params = ParameterVector("p", n_params)

        qc = _build_circuit(
            ansatz, list(params), n_qubits, n_layers, n_electrons=n_electrons
        )

        names = gate_names(qc)
        # Hartree-Fock prep: 2 X gates (n_electrons=2 bits set in hf_state).
        # Then 4 RY rotations, then 9 two-qubit Pauli rotations
        # (XX/YY/ZZ for each adjacent pair) each emitted as basis gates.
        assert names[:2] == ["x", "x"]
        assert names[2:6] == ["ry", "ry", "ry", "ry"]
        # Each XX block: 2 H + cx + rz + cx + 2 H = 7 gates.
        # Each YY block: 2 RX + cx + rz + cx + 2 RX = 7 gates.
        # Each ZZ block: cx + rz + cx = 3 gates.
        # 3 adjacent pairs × (7 + 7 + 3) = 51 entangler gates.
        # Total: 2 + 4 + 51 = 57.
        assert len(names) == 57
        assert names.count("cx") == 6 * 3  # CX appears in all three rotations
        assert names.count("rz") == 9  # one RZ per two-qubit Pauli rotation
        assert names.count("h") == 12  # XX needs 4 H per pair × 3 pairs
        assert names.count("rx") == 12  # YY needs 4 RX per pair × 3 pairs

    def test_build_multi_layer(self):
        n_electrons, n_qubits, n_layers = 2, 4, 2
        ansatz = QCCAnsatz()
        n_params = n_layers * ansatz.n_params_per_layer(n_qubits)
        params = ParameterVector("p", n_params)

        qc = _build_circuit(
            ansatz, list(params), n_qubits, n_layers, n_electrons=n_electrons
        )

        names = gate_names(qc)
        # 2 X (HF prep, once total) + 2 layers × (4 RY + 51 entangler basis gates).
        assert len(names) == 2 + 2 * (4 + 51)
        assert names.count("ry") == 8


def _make_uccsd_vqe(n_qubits, n_electrons, backend, optimizer, **spin_counts):
    """Build a VQE(UCCSDAnsatz()) over a dummy n_qubits-wide Hamiltonian.

    Same rationale as :func:`_make_lucj_vqe`: ``sample_solution`` measures in
    the computational basis regardless of the attached observable.
    """
    return VQE(
        hamiltonian={"Z" + "I" * (n_qubits - 1): 1.0},
        n_electrons=n_electrons,
        ansatz=UCCSDAnsatz(),
        backend=backend,
        optimizer=optimizer,
        **spin_counts,
    )


def _make_lucj_vqe(
    n_qubits,
    n_electrons,
    backend,
    optimizer,
    n_layers=1,
    ansatz_kwargs=None,
    **spin_counts,
):
    """Build a VQE(LUCJAnsatz()) over a dummy n_qubits-wide Hamiltonian.

    The Hamiltonian's value is irrelevant here — ``sample_solution`` measures
    the prepared state in the computational basis regardless of what
    observable is attached — it only needs to span ``n_qubits`` wires.
    """
    hamiltonian = {"Z" + "I" * (n_qubits - 1): 1.0}
    return VQE(
        hamiltonian=hamiltonian,
        n_electrons=n_electrons,
        n_layers=n_layers,
        ansatz=LUCJAnsatz(),
        backend=backend,
        optimizer=optimizer,
        ansatz_kwargs=ansatz_kwargs,
        **spin_counts,
    )


def _finite_difference_gradient(vqe, params, step=1e-4):
    """Central-difference gradient of the exact expectation value."""
    gradient = np.zeros_like(params)
    for index in range(len(params)):
        plus, minus = params.copy(), params.copy()
        plus[index] += step
        minus[index] -= step
        forward = next(iter(vqe._evaluate_cost_param_sets(plus).values()))
        backward = next(iter(vqe._evaluate_cost_param_sets(minus).values()))
        gradient[index] = (forward - backward) / (2.0 * step)
    return gradient


@pytest.mark.parametrize(
    "ansatz_factory, n_qubits, n_electrons, n_layers, ansatz_kwargs",
    [
        # Ansaetze on the default (1, 1) rule: measured over a 4*pi window to
        # carry frequency 1 exactly, so the two-term rule is right for them.
        (lambda: GenericLayerAnsatz([RYGate, RZGate]), 4, 2, 1, None),
        (QCCAnsatz, 4, 2, 1, None),
        (QCCAnsatz, 4, 2, 2, None),
        # Ansaetze that declare their own frequencies. Hartree-Fock is exact
        # under the default rule at 4 qubits / 1 layer — the excitation acts on
        # the untouched reference, where the 1/2 harmonic has zero amplitude —
        # so the wider cases are what actually pin its declaration.
        (HartreeFockAnsatz, 4, 2, 1, None),
        (HartreeFockAnsatz, 6, 2, 1, None),
        (HartreeFockAnsatz, 4, 2, 2, None),
        (UCCSDAnsatz, 4, 2, 1, None),
        (UCCSDAnsatz, 4, 2, 2, None),
        # Both LUCJ orbital regimes, since two orbitals take a reduced family.
        (LUCJAnsatz, 4, 2, 1, None),
        (LUCJAnsatz, 4, 2, 2, None),
        (LUCJAnsatz, 6, 2, 1, None),
        (LUCJAnsatz, 6, 2, 2, None),
        (LUCJAnsatz, 4, 2, 1, {"trailing_rotation": True}),
        (LUCJAnsatz, 6, 2, 1, {"trailing_rotation": True}),
    ],
)
def test_declared_frequencies_give_exact_gradients(
    ansatz_factory,
    n_qubits,
    n_electrons,
    n_layers,
    ansatz_kwargs,
    default_test_simulator,
    default_optimizer,
):
    """An ansatz's declared frequencies must reproduce the true derivative.

    The two-term ``+-pi/2`` rule silently returns a wrong gradient on an ansatz
    whose parameters carry other frequencies — it returned numerically *zero* for
    UCCSD, whose amplitudes sit at frequency 2 where that shift is a null step.
    This pins each declaration against finite differences, so a structural change
    to an ansatz that moves its frequencies fails here instead of quietly
    corrupting every gradient-based optimizer.

    Qubit counts and layer counts are part of the contract: an excitation gate's
    higher harmonics only carry amplitude once earlier gates have moved the state
    out of the gate's invariant subspace, so a narrow single-layer case can pass
    under a wrong rule.
    """
    hamiltonian = {
        "Z" + "I" * (n_qubits - 1): 1.0,
        "XX" + "I" * (n_qubits - 2): 0.4,
        "I" * (n_qubits - 2) + "ZZ": -0.7,
    }
    vqe = VQE(
        hamiltonian=hamiltonian,
        n_electrons=n_electrons,
        n_layers=n_layers,
        ansatz=ansatz_factory(),
        backend=default_test_simulator,
        optimizer=default_optimizer,
        ansatz_kwargs=ansatz_kwargs,
    )

    params = np.random.default_rng(5).uniform(0.2, 1.2, vqe.n_params)
    analytic = vqe._evaluate_gradient_at(params)
    numerical = _finite_difference_gradient(vqe, params)

    np.testing.assert_allclose(analytic, numerical, atol=1e-6)


def test_vqe_ansatz_kwargs_reach_both_the_count_and_the_circuit(
    default_test_simulator, default_optimizer
):
    """An ansatz option must reach ``n_params_per_layer`` *and* ``build``.

    If only ``build`` saw it, the circuit would carry unbound parameters; if
    only the count saw it, the optimizer would tune parameters no gate reads.
    Either way the mismatch is silent, so assert the two agree.
    """
    n_qubits, n_electrons = 6, 2
    plain = _make_lucj_vqe(
        n_qubits, n_electrons, default_test_simulator, default_optimizer
    )
    extended = _make_lucj_vqe(
        n_qubits,
        n_electrons,
        default_test_simulator,
        default_optimizer,
        ansatz_kwargs={"trailing_rotation": True},
    )

    n_orb = n_qubits // 2
    n_rotation = n_orb * (n_orb - 1)
    assert extended.n_params_per_layer - plain.n_params_per_layer == n_rotation

    assert len(extended.cost_circuit.parameters) == extended.n_params
    assert len(plain.cost_circuit.parameters) == plain.n_params


@pytest.mark.parametrize(
    "n_qubits,n_electrons,n_alpha,n_beta,expected",
    [
        (4, 2, None, None, (2, (1, 1))),
        (8, 4, None, None, (4, (2, 2))),
        (8, 4, 3, 1, (4, (3, 1))),
        (8, 4, 4, 0, (4, (4, 0))),
    ],
)
def test_resolve_spin_counts_accepts(n_qubits, n_electrons, n_alpha, n_beta, expected):
    """Without explicit counts the closed-shell split is used; with them, they
    pass through untouched."""
    assert (
        _resolve_spin_counts(n_qubits, n_electrons, n_alpha, n_beta, "Ansatz")
        == expected
    )


@pytest.mark.parametrize(
    "n_qubits,n_electrons,n_alpha,n_beta,message",
    [
        (5, 2, None, None, "even qubit count"),
        (8, 4, 2, None, "together"),
        (8, 4, None, 2, "together"),
        (8, 3, None, None, "cannot split"),
        (8, 4, 3, 2, "not n_electrons"),
        (4, 4, 3, 1, "outside the range"),
    ],
)
def test_resolve_spin_counts_rejects(n_qubits, n_electrons, n_alpha, n_beta, message):
    with pytest.raises(ValueError, match=message):
        _resolve_spin_counts(n_qubits, n_electrons, n_alpha, n_beta, "Ansatz")


@pytest.mark.parametrize("n_orb", [2, 3, 4, 5, 8])
def test_rotation_schedule_spans_the_rotation_group(n_orb):
    """A general orbital rotation needs ``n_orb * (n_orb - 1) / 2`` Givens
    rotations -- the dimension of the group. A single adjacent chain has only
    ``n_orb - 1`` and cannot reach an arbitrary rotation, which is what confined
    LUCJ to a fraction of its fragment correlation energy."""
    schedule = _rotation_schedule(n_orb)
    assert len(schedule) == n_orb * (n_orb - 1) // 2
    # Every adjacent pair participates, and each half-layer is disjoint.
    assert set(schedule) == set(range(n_orb - 1))


@pytest.mark.parametrize("orbital, spin", [(0, 0), (0, 1), (1, 0), (2, 1)])
def test_givens_rotation_is_a_real_fermionic_orbital_rotation(orbital, spin):
    """The gate must equal the JW image of ``exp(t (a+_j a_l - a+_l a_j))``.

    Two independent ways to get a particle- and Sz-conserving gate that is *not*
    an orbital rotation, both of which this pins:

    * dropping the ``Z`` the intervening opposite-spin qubit contributes, which
      leaves a plain qubit XY exchange;
    * exponentiating the Hermitian hop (phase argument ``0``), whose one-particle
      action ``exp(-i t sigma_x)`` is complex, so the network spans a
      phase-conjugated rotation group containing no real orbital rotation -- and
      no real CCSD amplitude can then be mapped onto it.
    """
    n_qubits = 8
    angle = 0.7
    circuit = QuantumCircuit(n_qubits)
    _emit_givens_rotation(circuit, angle, orbital, spin)

    lower = 2 * orbital + spin
    middle = lower + 1
    upper = 2 * (orbital + 1) + spin

    def label(paulis: dict[int, str]) -> str:
        return "".join(paulis.get(q, "I") for q in reversed(range(n_qubits)))

    # a+_j a_l - a+_l a_j under Jordan-Wigner, with the string on the qubit
    # between the two hopped spin-orbitals.
    generator = SparsePauliOp.from_list(
        [
            (label({lower: "X", middle: "Z", upper: "Y"}), 0.5j),
            (label({lower: "Y", middle: "Z", upper: "X"}), -0.5j),
        ]
    )
    expected = expm(0.5 * angle * generator.to_matrix())

    np.testing.assert_allclose(Operator(circuit).data, expected, atol=1e-12)


# --- Test LUCJAnsatz ---
class TestLUCJAnsatz:
    """LUCJ must conserve particle number and Sz for SQD sampling to work."""

    def test_requires_even_qubit_count(self):
        with pytest.raises(ValueError, match="even"):
            LUCJAnsatz().build(np.zeros(6), n_qubits=5, n_layers=1, n_electrons=2)

    def test_requires_n_electrons(self):
        with pytest.raises(ValueError, match="n_electrons"):
            LUCJAnsatz().build(np.zeros(6), n_qubits=4, n_layers=1)

    @pytest.mark.parametrize("n_qubits", [2, 4, 6, 8])
    def test_param_count_matches_built_circuit(self, n_qubits):
        """``build`` only rejects too *few* parameters, so an
        ``n_params_per_layer`` that over-reports the real count would still
        pass a check that only inspects ``circuit.num_qubits``. Feed exactly
        ``n_params`` distinct values and confirm the last one is actually
        consumed by some gate, proving the reported count matches what
        ``build`` reads rather than over- or under-counting it."""
        n_params = LUCJAnsatz.n_params_per_layer(n_qubits)
        circuit = LUCJAnsatz().build(
            np.arange(n_params, dtype=float),
            n_qubits=n_qubits,
            n_layers=1,
            n_electrons=2,
        )
        assert circuit.num_qubits == n_qubits
        consumed_values = {
            abs(float(instr.operation.params[0]))
            for instr in circuit.data
            if instr.operation.name in ("xx_plus_yy", "rzz")
        }
        assert consumed_values == set(range(n_params))

    @pytest.mark.parametrize("n_qubits", [4, 6, 10])
    def test_trailing_rotation_adds_one_independent_orbital_rotation(self, n_qubits):
        """``trailing_rotation`` must add exactly one more orbital rotation, with
        its own angles, after the layer's closing inverse rotation.

        This is the difference between our circuit and the one both SQD papers
        run (``exp(K2) exp(-K1) exp(iJ1) exp(K1)``), so pin the gate counts and
        the ordering rather than only the parameter total: appending the extra
        rotation *before* the inverse would collapse against it.
        """
        n_orb = n_qubits // 2
        n_rotation = n_orb * (n_orb - 1)
        plain = LUCJAnsatz.n_params_per_layer(n_qubits)
        extended = LUCJAnsatz.n_params_per_layer(n_qubits, trailing_rotation=True)
        assert extended - plain == n_rotation

        def rotation_gates(**kwargs):
            n_params = LUCJAnsatz.n_params_per_layer(n_qubits, **kwargs)
            circuit = LUCJAnsatz().build(
                np.arange(1, n_params + 1, dtype=float),
                n_qubits=n_qubits,
                n_layers=1,
                n_electrons=2,
                **kwargs,
            )
            gates = [
                instr for instr in circuit.data if instr.operation.name == "xx_plus_yy"
            ]
            return circuit, gates

        _, plain_gates = rotation_gates()
        circuit, extended_gates = rotation_gates(trailing_rotation=True)
        assert len(extended_gates) - len(plain_gates) == n_rotation

        # Every reported parameter reaches a gate, so the count cannot over-report.
        consumed = {
            abs(float(instr.operation.params[0]))
            for instr in circuit.data
            if instr.operation.name in ("xx_plus_yy", "rzz")
        }
        assert consumed == set(float(v) for v in range(1, extended + 1))

        # The trailing rotation comes last: each Givens is a CZ-wrapped hop, so
        # the tail is that triple repeated, with no Jastrow gate after it.
        names = [instr.operation.name for instr in circuit.data]
        assert names[-3 * n_rotation :] == ["cz", "xx_plus_yy", "cz"] * n_rotation

    def test_trailing_rotation_angles_are_independent_of_the_opening_rotation(self):
        """The added rotation is ``exp(K2)``, a *new* rotation -- not a repeat of
        ``exp(K1)``. Sharing angles with the opening rotation would make the extra
        parameters redundant and the flag pointless."""
        n_qubits = 6
        n_orb = n_qubits // 2
        n_rotation = n_orb * (n_orb - 1)
        n_params = LUCJAnsatz.n_params_per_layer(n_qubits, trailing_rotation=True)
        params = np.zeros(n_params)
        # Only the trailing block is non-zero, so any gate that fires must
        # belong to it.
        trailing = np.arange(1, n_rotation + 1, dtype=float) / 10.0
        params[-n_rotation:] = trailing
        circuit = LUCJAnsatz().build(
            params,
            n_qubits=n_qubits,
            n_layers=1,
            n_electrons=2,
            trailing_rotation=True,
        )
        firing = [
            float(instr.operation.params[0])
            for instr in circuit.data
            if instr.operation.name == "xx_plus_yy"
            and abs(float(instr.operation.params[0])) > 0.0
        ]
        assert sorted(firing) == sorted(trailing)

    def test_trailing_rotation_defaults_off(self):
        """Existing callers must see the previous circuit unchanged."""
        n_qubits = 6
        n_params = LUCJAnsatz.n_params_per_layer(n_qubits)
        rng = np.random.default_rng(11)
        params = rng.uniform(0, 2 * np.pi, n_params)
        without = LUCJAnsatz().build(
            params, n_qubits=n_qubits, n_layers=1, n_electrons=2
        )
        explicit = LUCJAnsatz().build(
            params,
            n_qubits=n_qubits,
            n_layers=1,
            n_electrons=2,
            trailing_rotation=False,
        )
        assert [instr.operation.name for instr in without.data] == [
            instr.operation.name for instr in explicit.data
        ]

    @pytest.mark.parametrize("n_alpha,n_beta", [(2, 2), (2, 0), (3, 1)])
    def test_reference_honours_the_requested_spin_counts(self, n_alpha, n_beta):
        """LUCJ used to ignore ``n_alpha``/``n_beta`` and fill the lowest
        ``n_electrons`` qubits, which in interleaved ordering silently prepared
        ``(1, 1)`` for a requested ``(2, 0)`` -- the wrong Sz sector, with no
        error."""
        n_qubits = 8
        circuit = LUCJAnsatz().build(
            np.zeros(LUCJAnsatz.n_params_per_layer(n_qubits)),
            n_qubits,
            1,
            n_electrons=n_alpha + n_beta,
            n_alpha=n_alpha,
            n_beta=n_beta,
        )

        occupied = sorted(
            circuit.qubits.index(instruction.qubits[0])
            for instruction in circuit.data
            if instruction.operation.name == "x"
        )
        assert occupied == sorted(
            [2 * p for p in range(n_alpha)] + [2 * p + 1 for p in range(n_beta)]
        )

    @pytest.mark.parametrize("n_alpha,n_beta", [(2, 0), (3, 1)])
    def test_vqe_forwards_the_spin_counts(
        self, n_alpha, n_beta, default_test_simulator, default_optimizer
    ):
        """The spin counts must survive the trip through ``VQE`` into
        ``build``, not just be honoured when ``build`` is called directly.

        With zero angles the circuit is deterministic, so every shot lands on
        the reference determinant and the occupied set is observable.
        """
        n_qubits = 8
        vqe = _make_lucj_vqe(
            n_qubits,
            n_alpha + n_beta,
            default_test_simulator,
            default_optimizer,
            n_alpha=n_alpha,
            n_beta=n_beta,
        )

        vqe.sample_solution(params=np.zeros(vqe.n_params))

        probs = next(iter(vqe.best_probs.values()))
        assert len(probs) == 1
        assert _occupied_from_label(next(iter(probs))) == {
            2 * p for p in range(n_alpha)
        } | {2 * p + 1 for p in range(n_beta)}

    def test_hartree_fock_reference_is_embedded(
        self, default_test_simulator, default_optimizer
    ):
        """With zero angles the circuit is deterministic: every shot lands on
        the same bitstring, and that bitstring is exactly the HF determinant.

        Executed through the actual backend (rather than a bare
        ``Statevector``) so this exercises divi's full circuit-submission
        path, including the QASM2 lowering ``xx_plus_yy`` requires.
        """
        n_qubits, n_electrons = 4, 2
        n_params = LUCJAnsatz.n_params_per_layer(n_qubits)
        vqe = _make_lucj_vqe(
            n_qubits, n_electrons, default_test_simulator, default_optimizer
        )

        vqe.sample_solution(params=np.zeros(n_params))

        probs = next(iter(vqe.best_probs.values()))
        # Deterministic circuit: exactly one bitstring across every shot.
        assert len(probs) == 1
        occupied = next(iter(probs))
        assert probs[occupied] == pytest.approx(1.0, abs=1e-9)
        # Interleaved placement (qubits 0, 1 occupied), not a popcount-only
        # check: a blocked-HF regression ("1010") has the same popcount as
        # the correct interleaved one ("1100") but must still fail here.
        assert occupied == "1" * n_electrons + "0" * (n_qubits - n_electrons)

    def test_hopping_gates_stay_within_one_spin_sector(self):
        """Every XXPlusYY hop connects same-parity (same-spin) qubits two apart.

        A statevector-only check can miss this: at closed-shell HF, an
        on-site XXPlusYY between qubits 2p and 2p+1 acts as the identity
        (both are occupied or both empty), so a regression that hops across
        spin sectors on-site would still pass a probability-based test. This
        asserts the gate placement directly instead.
        """
        n_qubits, n_electrons, n_layers = 6, 4, 2
        n_params = n_layers * LUCJAnsatz.n_params_per_layer(n_qubits)
        rng = np.random.default_rng(3)
        circuit = LUCJAnsatz().build(
            rng.uniform(0, 2 * np.pi, n_params),
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_electrons=n_electrons,
        )
        hop_gates = [
            instr for instr in circuit.data if instr.operation.name == "xx_plus_yy"
        ]
        assert hop_gates
        for instr in hop_gates:
            lower, upper = (circuit.find_bit(q).index for q in instr.qubits)
            assert lower % 2 == upper % 2
            assert abs(upper - lower) == 2

    def test_conserves_particle_number_and_sz(
        self, default_test_simulator, default_optimizer
    ):
        """Every sampled basis state keeps both alpha and beta counts.

        Executed through the actual backend (rather than a bare
        ``Statevector``) so this exercises divi's full circuit-submission
        path, including the QASM2 lowering ``xx_plus_yy`` requires.
        """
        n_qubits, n_electrons, n_layers = 6, 4, 2
        n_params = n_layers * LUCJAnsatz.n_params_per_layer(n_qubits)
        rng = np.random.default_rng(2)
        vqe = _make_lucj_vqe(
            n_qubits,
            n_electrons,
            default_test_simulator,
            default_optimizer,
            n_layers=n_layers,
        )

        vqe.sample_solution(params=rng.uniform(0, 2 * np.pi, n_params))

        probs = next(iter(vqe.best_probs.values()))
        n_orb = n_qubits // 2
        assert probs
        for bitstring, prob in probs.items():
            if prob < 1e-12:
                continue
            # divi normalizes measurement bitstrings so character k is qubit k.
            alpha = sum(int(bitstring[2 * p]) for p in range(n_orb))
            beta = sum(int(bitstring[2 * p + 1]) for p in range(n_orb))
            assert alpha == n_electrons // 2
            assert beta == n_electrons // 2
