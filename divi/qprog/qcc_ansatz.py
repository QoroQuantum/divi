# nh3_qcc_vqe.py
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer

########################################
# 1. Problem setup: NH3 configs & active space
########################################

active_electrons = 8
active_orbitals = 6  # -> 12 spin-orbitals / qubits

nh3_config1_coords = np.array(
    [
        (0.0, 0.0, 0.0),   # N
        (1.01, 0.0, 0.0),  # H1
        (-0.5, 0.87, 0.0), # H2
        (-0.5, -0.87, 0.0) # H3
    ]
)

nh3_config2_coords = np.array(
    [
        (0.0, 0.0, 0.0),   # N (inverted)
        (-1.01, 0.0, 0.0), # H1
        (0.5, -0.87, 0.0), # H2
        (0.5, 0.87, 0.0),  # H3
    ]
)

def make_nh3_hamiltonian(coords):
    """Build molecular Hamiltonian and HF reference state."""
    mol = qml.qchem.Molecule(
        symbols=["N", "H", "H", "H"],
        coordinates=coords,
    )
    H, n_qubits = qml.qchem.molecular_hamiltonian(
        mol,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )
    # Hartree–Fock basis state
    hf_state = qml.qchem.hf_state(active_electrons, n_qubits)
    return H, n_qubits, hf_state

########################################
# 2. QCC ansatz definition
########################################

def build_qcc_entangler_pool(n_qubits):
    """Build a simple QCC entangler pool with Pauli words.

    Here we choose:
      - XX, YY, ZZ on *adjacent* qubits (i, i+1)

    This is not the chemically optimal full QCC pool,
    but it is a genuine QCC-style product of Pauli exponentials.
    """
    pool = []
    for i in range(n_qubits - 1):
        pair = (i, i + 1)
        pool.append(("XX", pair))
        pool.append(("YY", pair))
        pool.append(("ZZ", pair))
    return pool

def qcc_ansatz(params, hf_state, entangler_pool, wires):
    """QCC circuit:

      |ψ(θ)> = ∏_k exp(-i θ_k P_k / 2) ∏_j RY(α_j) |HF>

    params = [α_0,...,α_{n_qubits-1}, θ_0,...,θ_{n_ent-1}]
    """
    n_qubits = len(wires)
    n_singles = n_qubits
    n_ent = len(entangler_pool)

    # safety check
    assert len(params) == n_singles + n_ent

    # Prepare HF reference
    qml.BasisState(hf_state, wires=wires)

    # Single-qubit RY rotations
    for i in range(n_qubits):
        qml.RY(params[i], wires=wires[i])

    # QCC entanglers: Pauli exponentials via PauliRot
    ent_params = params[n_singles:]
    for (pauli_string, pair), theta in zip(entangler_pool, ent_params):
        qml.PauliRot(theta, pauli_string, wires=pair)

########################################
# 3. VQE wrapper for a given geometry
########################################

def run_qcc_vqe_for_geometry(coords, max_steps=150, stepsize=0.05, seed=0):
    """Run QCC-VQE on a single NH3 geometry."""
    H, n_qubits, hf_state = make_nh3_hamiltonian(coords)
    wires = list(range(n_qubits))

    # Build entangler pool for QCC
    entangler_pool = build_qcc_entangler_pool(n_qubits)

    n_singles = n_qubits
    n_ent = len(entangler_pool)
    n_params = n_singles + n_ent

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        qcc_ansatz(params, hf_state, entangler_pool, wires)
        return qml.expval(H)

    def cost(params):
        return circuit(params)

    # Initialize parameters
    rng = np.random.default_rng(seed)
    init_params = pnp.array(
        rng.normal(loc=0.0, scale=0.1, size=(n_params,)),
        requires_grad=True,
    )

    opt = AdamOptimizer(stepsize=stepsize)
    params = init_params

    for it in range(max_steps):
        params, energy = opt.step_and_cost(cost, params)

        if (it + 1) % 10 == 0 or it == 0:
            print(f"[NH3] step {it+1:4d}/{max_steps}   E = {energy:.8f} Ha")

    return float(energy), params, H, entangler_pool, hf_state

########################################
# 4. Run on both configs and compare
########################################

def main():
    print("=== QCC VQE for NH3 inversion ===")

    print("\n--- Config 1 (upright) ---")
    E1, params1, H1, pool1, hf1 = run_qcc_vqe_for_geometry(
        nh3_config1_coords,
        max_steps=150,
        stepsize=0.05,
        seed=1,
    )

    print("\n--- Config 2 (inverted) ---")
    # IMPORTANT: use same ansatz structure (same n_qubits, same pool pattern)
    # so we re-build Hamiltonian but keep pool pattern consistent
    E2, params2, H2, pool2, hf2 = run_qcc_vqe_for_geometry(
        nh3_config2_coords,
        max_steps=150,
        stepsize=0.05,
        seed=2,
    )

    delta = abs(E1 - E2)

    print("\n=== Results (true QCC ansatz) ===")
    print(f"E(config1) = {E1:.8f} Ha")
    print(f"E(config2) = {E2:.8f} Ha")
    print(f"|ΔE|        = {delta:.8f} Ha")

    if delta < 1e-3:
        print("Degeneracy confirmed within 1 mHa.")
    else:
        print("Degeneracy NOT within 1 mHa – increase steps, adjust pool, or refine ansatz.")

if __name__ == "__main__":
    main()