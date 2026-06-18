# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Compute-uncompute state-overlap circuit construction.

QN-SPSA estimates the Fubini–Study metric from state fidelities
:math:`F(\\theta_1, \\theta_2) = |\\langle\\psi(\\theta_1)|\\psi(\\theta_2)\\rangle|^2`.
:func:`build_overlap_meta` turns an ansatz into the compute-uncompute circuit
:math:`U(\\theta_\\text{fwd})\\,U(\\theta_\\text{bwd})^\\dagger` whose all-zeros
measurement probability equals that fidelity.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.converters import circuit_to_dag, dag_to_circuit

from divi.circuits._core import MetaCircuit


def build_overlap_meta(cost_circuit: MetaCircuit) -> MetaCircuit:
    """Build the compute-uncompute overlap circuit from a cost ansatz.

    The forward block is the ansatz on its original parameters; the backward
    block is the ansatz inverted (``QuantumCircuit.inverse()`` flips each
    rotation, e.g. ``RX(θ) → RX(-θ)``) on a disjoint parameter namespace. The
    returned MetaCircuit measures all qubits as a probability distribution; its
    parameters are ordered ``(*θ_fwd, *θ_bwd)`` (each of length ``d``), so binding
    a flat ``2d`` vector ``[θ_fwd | θ_bwd]`` gives

    .. math::
        P(0^n) = |\\langle 0|U(\\theta_\\text{bwd})^\\dagger U(\\theta_\\text{fwd})|0\\rangle|^2
               = F(\\theta_\\text{fwd}, \\theta_\\text{bwd}),

    which is symmetric in its arguments. The result format is left unset — the
    probs ``MeasurementStage`` assigns it during expansion, as for the sample
    pipeline.
    """
    forward = dag_to_circuit(cost_circuit.circuit_bodies[0][1])
    fwd_params = list(cost_circuit.parameters)
    n_qubits = forward.num_qubits

    # Backward block lives in its own namespace so the flat (*θ_fwd, *θ_bwd)
    # binding is unambiguous. The name must not clash with any forward parameter
    # name, or compose() sees two distinct parameters sharing a name and raises.
    fwd_names = {p.name for p in fwd_params}
    bwd_prefix = "theta_uncompute"
    bwd_params = ParameterVector(bwd_prefix, len(fwd_params))
    while not fwd_names.isdisjoint(p.name for p in bwd_params):
        bwd_prefix += "_"
        bwd_params = ParameterVector(bwd_prefix, len(fwd_params))
    backward = forward.assign_parameters(
        dict(zip(fwd_params, bwd_params)), inplace=False
    )

    overlap = QuantumCircuit(n_qubits)
    overlap.compose(forward, inplace=True)
    overlap.compose(backward.inverse(), inplace=True)

    return MetaCircuit(
        circuit_bodies=(((), circuit_to_dag(overlap)),),
        parameters=(*fwd_params, *tuple(bwd_params)),
        observable=None,
        measured_wires=tuple(range(n_qubits)),
    )
