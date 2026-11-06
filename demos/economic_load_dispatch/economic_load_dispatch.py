# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Economic Load Dispatch with Prohibited Operating Zones via PCE-VQE.

Builds the dispatch problem for three generators, runs PCE-VQE on it, then
repeats for six generators.
"""

import time

import dimod
from qiskit.circuit.library import CXGate, RYGate, RZGate

from divi.backends import JobConfig, MaestroSimulator, QoroService
from divi.qprog.algorithms import PCE, GenericLayerAnsatz, PolyEncoding, SoftObjective
from divi.qprog.optimizers import PymooMethod, PymooOptimizer
from divi.qprog.problems import BinaryOptimizationProblem, LinearConstraint


def define_generators():
    """Define the 3-generator microgrid.

    Each generator has:
      - a, b, c:          fuel cost curve  Cost = a + b·P + c·P²
      - P_min, P_max:     operating range in MW
      - poz_low, poz_high: bounds of the prohibited operating zone in MW

    Returns a list of generator dicts.
    """
    return [
        {
            "name": "Gen 1",
            "a": 20,
            "b": 2.0,
            "c": 0.010,
            "P_min": 40,
            "P_max": 115,
            "poz_low": 60,
            "poz_high": 75,
        },
        {
            "name": "Gen 2",
            "a": 15,
            "b": 1.5,
            "c": 0.020,
            "P_min": 20,
            "P_max": 95,
            "poz_low": 40,
            "poz_high": 55,
        },
        {
            "name": "Gen 3",
            "a": 25,
            "b": 1.8,
            "c": 0.015,
            "P_min": 30,
            "P_max": 105,
            "poz_low": 50,
            "poz_high": 65,
        },
    ]


def define_generators_large():
    """Define a six-generator variant for exploring the same formulation."""
    return define_generators() + [
        {
            "name": "Gen 4",
            "a": 18,
            "b": 1.7,
            "c": 0.012,
            "P_min": 35,
            "P_max": 110,
            "poz_low": 55,
            "poz_high": 70,
        },
        {
            "name": "Gen 5",
            "a": 22,
            "b": 2.2,
            "c": 0.009,
            "P_min": 50,
            "P_max": 120,
            "poz_low": 70,
            "poz_high": 85,
        },
        {
            "name": "Gen 6",
            "a": 12,
            "b": 1.3,
            "c": 0.018,
            "P_min": 25,
            "P_max": 90,
            "poz_low": 45,
            "poz_high": 60,
        },
    ]


STEP_MW = 5  # power resolution per qubit level
N_QUBITS_PER_GEN = 4  # 2^4 = 16 discrete levels per generator
BIT_WEIGHTS = [2**b for b in range(N_QUBITS_PER_GEN)]  # [1, 2, 4, 8]


def compute_fuel_cost(gen, power):
    """Compute fuel cost ($) for a generator at a given power (MW)."""
    return gen["a"] + gen["b"] * power + gen["c"] * power**2


def _build_qubit_name(gen_idx, bit_idx):
    """Variable name for qubit `bit_idx` of generator `gen_idx`."""
    return f"q_{gen_idx}_{bit_idx}"


def decode_power(gen_idx, generators, bit_values):
    """Convert binary qubit values back to MW for one generator."""
    integer_val = sum(
        BIT_WEIGHTS[b] * bit_values[_build_qubit_name(gen_idx, b)]
        for b in range(N_QUBITS_PER_GEN)
    )
    return generators[gen_idx]["P_min"] + STEP_MW * integer_val


def build_problem(generators, demand, penalty_weight=2000):
    """Build the ELD problem: fuel-cost objective plus demand and zone constraints.

    The objective is the fuel cost alone. Two kinds of constraint carry the rest:
      1. Demand: total generation must equal the demand (an equality, so it
         adds no qubits).
      2. Prohibited operating zones: a generator's second-highest bit may not
         be set while its highest bit is clear (one slack qubit each).

    A violation of ``v`` units costs ``v**2 * penalty_weight``.

    Args:
        generators:     list of generator dicts from define_generators()
        demand:         target load in MW
        penalty_weight: weight of every constraint penalty

    Returns:
        The BinaryOptimizationProblem.
    """
    bqm = dimod.BinaryQuadraticModel(vartype="BINARY")

    cost_offset = 0.0
    for g, gen in enumerate(generators):
        a, b_coeff, c = gen["a"], gen["b"], gen["c"]
        p_min = gen["P_min"]
        cost_offset += a + b_coeff * p_min + c * p_min**2

        for k in range(N_QUBITS_PER_GEN):
            w_k = STEP_MW * BIT_WEIGHTS[k]
            var_k = _build_qubit_name(g, k)
            bqm.add_linear(var_k, b_coeff * w_k + c * (2 * p_min * w_k + w_k**2))
            for m in range(k + 1, N_QUBITS_PER_GEN):
                w_m = STEP_MW * BIT_WEIGHTS[m]
                bqm.add_quadratic(var_k, _build_qubit_name(g, m), c * 2 * w_k * w_m)
    bqm.offset += cost_offset

    demand_constraint = LinearConstraint(
        {
            _build_qubit_name(g, k): STEP_MW * BIT_WEIGHTS[k]
            for g in range(len(generators))
            for k in range(N_QUBITS_PER_GEN)
        },
        "==",
        demand - sum(gen["P_min"] for gen in generators),
    )
    zone_constraints = [
        LinearConstraint(
            {
                _build_qubit_name(g, N_QUBITS_PER_GEN - 1): 1,
                _build_qubit_name(g, N_QUBITS_PER_GEN - 2): -1,
            },
            ">=",
            0,
        )
        for g in range(len(generators))
    ]
    return BinaryOptimizationProblem(
        bqm,
        constraints=[demand_constraint, *zone_constraints],
        penalty_weight=penalty_weight,
    )


def build_penalised_bqm(problem):
    """Return the problem's penalised objective, slack qubits included, as a BQM."""
    bqm = dimod.BinaryQuadraticModel(vartype="BINARY")
    for term, coeff in problem.raw_problem.items():
        if not term:
            bqm.offset += coeff
        elif len(term) == 1:
            bqm.add_linear(term[0], coeff)
        else:
            bqm.add_quadratic(term[0], term[1], coeff)
    return bqm


def solve_brute_force(generators, demand):
    """Enumerate all 4,096 configurations and return the cheapest valid one.

    Returns:
        (P1, P2, P3, cost) for the best valid dispatch.
    """
    best = None

    for i1 in range(16):
        p1 = generators[0]["P_min"] + STEP_MW * i1
        poz1 = generators[0]["poz_low"] <= p1 <= generators[0]["poz_high"]
        for i2 in range(16):
            p2 = generators[1]["P_min"] + STEP_MW * i2
            poz2 = generators[1]["poz_low"] <= p2 <= generators[1]["poz_high"]
            for i3 in range(16):
                p3 = generators[2]["P_min"] + STEP_MW * i3
                poz3 = generators[2]["poz_low"] <= p3 <= generators[2]["poz_high"]

                if p1 + p2 + p3 != demand:
                    continue
                if poz1 or poz2 or poz3:
                    continue

                cost = sum(
                    compute_fuel_cost(generators[g], p)
                    for g, p in enumerate([p1, p2, p3])
                )
                if best is None or cost < best[3]:
                    best = (p1, p2, p3, cost)

    return best


def solve_with_annealing(generators, demand, problem, num_reads=1000):
    """Simulated annealing on the penalised BQM.

    Returns the cheapest decoded sample within ``STEP_MW`` of the demand.
    """
    sampler = dimod.SimulatedAnnealingSampler()
    sampleset = sampler.sample(build_penalised_bqm(problem), num_reads=num_reads)

    best = None
    for sample, energy in sampleset.data(["sample", "energy"]):
        powers = []
        valid = True
        for g, gen in enumerate(generators):
            p = gen["P_min"] + STEP_MW * sum(
                BIT_WEIGHTS[b] * int(sample.get(_build_qubit_name(g, b), 0))
                for b in range(N_QUBITS_PER_GEN)
            )
            if gen["poz_low"] <= p <= gen["poz_high"]:
                valid = False
                break
            powers.append(p)

        if not valid or abs(sum(powers) - demand) > STEP_MW:
            continue

        cost = sum(compute_fuel_cost(gen, p) for gen, p in zip(generators, powers))
        if best is None or cost < best[-1]:
            best = tuple(powers) + (cost,)

    return best


def solve_with_pce(
    problem,
    n_layers=3,
    max_iterations=20,
    alpha=3.0,
    population_size=30,
    shots=10000,
    backend=None,
):
    """Run the PCE-VQE quantum solver on the problem.

    PCE (Pauli Correlation Encoding) encodes the problem's variables, slack
    included, with ``PolyEncoding``: each variable is the parity of one or two
    qubits.

    Args:
        problem:          the BinaryOptimizationProblem to solve
        n_layers:         depth of the variational quantum circuit
        max_iterations:   number of Differential Evolution generations
        alpha:            binary activation hardness (higher = sharper)
        population_size:  DE population per generation
        shots:            measurement samples per circuit evaluation
        backend:          Divi backend (MaestroSimulator or QoroService)

    Returns:
        pce_solver:  the solved PCE object (access .solution, .get_top_solutions)
    """
    if backend is None:
        backend = MaestroSimulator(shots=shots)

    ansatz = GenericLayerAnsatz(
        gate_sequence=[RYGate, RZGate],
        entangler=CXGate,
        entangling_layout="all-to-all",
    )

    pce_solver = PCE(
        problem,
        ansatz=ansatz,
        n_layers=n_layers,
        encoding=PolyEncoding(),
        optimizer=PymooOptimizer(
            method=PymooMethod.DE, population_size=population_size
        ),
        max_iterations=max_iterations,
        objective=SoftObjective(alpha=alpha),
        backend=backend,
    )

    print(
        f"\n   PCE qubits: {pce_solver.n_qubits}  "
        f"(poly encoding of {len(problem.canonical_problem.variable_order)} variables)"
    )
    pce_solver.run()
    return pce_solver


def repair_solution(powers, generators, demand):
    """Move a decoded dispatch out of prohibited zones and towards the demand.

    Stage 1 — Move any generator inside a prohibited operating zone to the
              nearest allowed power level.
    Stage 2 — Step the generator with the smallest marginal cost by
              ``STEP_MW`` towards the demand, for at most 50 steps.

    Args:
        powers:     list of MW values [P1, P2, P3]
        generators: list of generator dicts
        demand:     target load in MW

    Returns:
        Repaired list of powers, or None if the demand is not reached.
    """
    ps = list(powers)

    # Stage 1: fix POZ violations
    for g, gen in enumerate(generators):
        if gen["poz_low"] <= ps[g] <= gen["poz_high"]:
            # Find all allowed power levels for this generator
            allowed = [
                gen["P_min"] + STEP_MW * idx
                for idx in range(2**N_QUBITS_PER_GEN)
                if not (
                    gen["poz_low"] <= gen["P_min"] + STEP_MW * idx <= gen["poz_high"]
                )
            ]
            ps[g] = min(allowed, key=lambda lv: abs(lv - ps[g]))

    # Stage 2: fix demand mismatch
    for _ in range(50):
        gap = demand - sum(ps)
        if gap == 0:
            break

        step = STEP_MW if gap > 0 else -STEP_MW
        best_g, best_cost = None, float("inf")

        for g, gen in enumerate(generators):
            new_p = ps[g] + step
            if new_p < gen["P_min"] or new_p > gen["P_max"]:
                continue
            if gen["poz_low"] <= new_p <= gen["poz_high"]:
                continue
            marginal = abs(
                compute_fuel_cost(gen, new_p) - compute_fuel_cost(gen, ps[g])
            )
            if marginal < best_cost:
                best_cost = marginal
                best_g = g

        if best_g is None:
            return None
        ps[best_g] += step

    return ps if sum(ps) == demand else None


def build_feasibility_report(powers, generators, demand):
    """Return constraint diagnostics for a decoded generator dispatch."""
    powers = [int(power) for power in powers]
    prohibited = [
        i
        for i, (power, gen) in enumerate(zip(powers, generators))
        if gen["poz_low"] <= power <= gen["poz_high"]
    ]
    out_of_bounds = [
        i
        for i, (power, gen) in enumerate(zip(powers, generators))
        if power < gen["P_min"] or power > gen["P_max"]
    ]
    total = sum(powers)
    return {
        "total_generation": total,
        "demand_gap": demand - total,
        "prohibited_generators": prohibited,
        "out_of_bounds_generators": out_of_bounds,
        "feasible": total == demand and not prohibited and not out_of_bounds,
    }


def sweep_penalty(generators, demand, penalties=(200, 2_000, 20_000)):
    """Show how the penalty weight changes the penalised ground sample."""
    rows = []
    for penalty in penalties:
        problem = build_problem(generators, demand, penalty_weight=penalty)
        bqm = build_penalised_bqm(problem)
        sample = dimod.ExactSolver().sample(bqm).first.sample
        powers = [decode_power(g, generators, sample) for g in range(len(generators))]
        report = build_feasibility_report(powers, generators, demand)
        rows.append((penalty, report))
    return rows


def print_dispatch_report(label, powers, generators, demand):
    """Print the demand and generator-constraint status of one dispatch."""
    report = build_feasibility_report(powers, generators, demand)
    print(f"\n   {label}")
    print(
        f"   total={report['total_generation']:.0f} MW, "
        f"demand gap={report['demand_gap']:.0f} MW"
    )
    print(
        f"   prohibited zones: {report['prohibited_generators'] or 'none'}; "
        f"limits: {report['out_of_bounds_generators'] or 'none'}; "
        f"feasible: {'yes' if report['feasible'] else 'no'}"
    )


def find_best_repaired_solution(pce_solver, generators, demand, top_n=20):
    """Scan the top quantum candidates and return the best repaired dispatch.

    For each candidate in the quantum probability distribution:
      1. Decode the bitstring to MW values
      2. Repair any constraint violations
      3. Keep the cheapest valid result

    Returns:
        (powers, cost, probability) or None
    """
    top_solutions = pce_solver.get_top_solutions(n=top_n, include_decoded=True)
    best = None

    print("\n   Top quantum candidates:")
    for i, sol in enumerate(top_solutions, 1):
        if sol.decoded is None:
            continue

        ps = [decode_power(g, generators, sol.decoded) for g in range(len(generators))]
        report = build_feasibility_report(ps, generators, demand)
        cost = sum(
            compute_fuel_cost(generators[g], ps[g]) for g in range(len(generators))
        )
        tot = sum(ps)
        valid = tot == demand and all(
            not (generators[g]["poz_low"] <= ps[g] <= generators[g]["poz_high"])
            for g in range(len(generators))
        )
        print(
            f"     {i:2d}. P=[{ps[0]:.0f},{ps[1]:.0f},{ps[2]:.0f}]  "
            f"Tot={tot:.0f}  Cost={cost:.0f}$  "
            f"Prob={sol.prob:.2%}  {'valid' if valid else 'invalid'}"
        )
        if i == 1:
            print(
                f"         demand gap={report['demand_gap']:.0f} MW; "
                f"prohibited zones={report['prohibited_generators'] or 'none'}"
            )

        repaired = repair_solution(ps, generators, demand)
        if repaired is not None:
            rep_cost = sum(
                compute_fuel_cost(generators[g], repaired[g])
                for g in range(len(generators))
            )
            if best is None or rep_cost < best[1]:
                best = (repaired, rep_cost, sol.prob)

    return best


def print_comparison(quantum_result, classical_result, classical_is_optimal=True):
    """Print a side-by-side comparison of quantum and classical solutions.

    Args:
        quantum_result:       (powers, cost, probability) from the quantum solver
        classical_result:     (P1, ..., cost) from the classical solver
        classical_is_optimal: True for an exhaustive search; False when the
                              classical result is not one
    """
    q_powers, q_cost, q_prob = quantum_result
    c_powers, c_cost = classical_result[:-1], classical_result[-1]
    label = "Classical optimum" if classical_is_optimal else "Annealing baseline"

    c_str = ", ".join(f"P{i + 1}={p:.0f}" for i, p in enumerate(c_powers))
    q_str = ", ".join(f"P{i + 1}={p:.0f}" for i, p in enumerate(q_powers))

    print("\n" + "=" * 70)
    print("  Quantum vs Classical Comparison")
    print("=" * 70)
    print(f"\n   {label + ':':<20}{c_str} MW  → Cost = {c_cost:.1f} $")
    print(f"   {'PCE-VQE result:':<20}{q_str} MW  → Cost = {q_cost:.1f} $")

    delta = q_cost - c_cost
    if not classical_is_optimal:
        print(f"\n   Cost relative to the annealing baseline: {delta:+.1f} $")
    else:
        print(f"\n   Cost difference to exhaustive search: {delta:+.1f} $")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    DEMAND = 195  # MW — how much power the grid needs

    # Backend selection
    USE_QORO_SERVICE = False  # Optional backend; credentials must be configured.

    if USE_QORO_SERVICE:
        backend = QoroService(job_config=JobConfig(shots=10_000))
        print("Using QoroService backend")
    else:
        backend = MaestroSimulator(shots=10_000)
        print("Using local MaestroSimulator")

    # 1. Define the generators and their constraints
    generators = define_generators()

    # 2. Encode the fuel cost, demand and zone rules as a problem with constraints
    problem = build_problem(generators, demand=DEMAND)
    n_decision = len(generators) * N_QUBITS_PER_GEN
    n_variables = len(problem.canonical_problem.variable_order)
    print(
        f"Built problem: {n_decision} decision variables, "
        f"{n_variables - n_decision} slack variables"
    )
    print("Penalty-weight sweep (penalised ground samples):")
    for penalty, report in sweep_penalty(generators, DEMAND):
        print(
            f"  λ={penalty:>5}: gap={report['demand_gap']:>4.0f} MW, "
            f"prohibited={report['prohibited_generators'] or 'none'}"
        )

    # 3. Find the classical optimum (for comparison)
    classical_best = solve_brute_force(generators, DEMAND)
    print(
        f"Classical optimum: P1={classical_best[0]}, P2={classical_best[1]}, "
        f"P3={classical_best[2]} MW  → Cost = {classical_best[3]:.1f} $"
    )

    # 4. Solve with quantum computing
    print("\nRunning quantum solver (PCE-VQE)...")
    t0 = time.time()
    pce_solver = solve_with_pce(problem, backend=backend)
    local_time = time.time() - t0

    # 5. Repair the quantum solution to make it fully valid
    result = find_best_repaired_solution(pce_solver, generators, DEMAND)

    if result is not None:
        powers, cost, prob = result
        print(
            f"\n   → Best repaired solution: P1={powers[0]:.0f}, "
            f"P2={powers[1]:.0f}, P3={powers[2]:.0f} MW, "
            f"Cost={cost:.1f} $  (quantum seed prob={prob:.2%})"
        )
        print_dispatch_report(
            "Repaired-solution feasibility", powers, generators, DEMAND
        )

        # 6. Compare quantum vs classical
        print_comparison(result, classical_best)
    else:
        print("\n   No candidate could be repaired to a feasible dispatch.")

    print("\n" + "=" * 70)
    print("  Six-generator variant")
    print("=" * 70)

    generators_large = define_generators_large()
    DEMAND_LARGE = 390  # MW

    problem_large = build_problem(generators_large, demand=DEMAND_LARGE)
    n_decision_large = len(generators_large) * N_QUBITS_PER_GEN
    n_variables_large = len(problem_large.canonical_problem.variable_order)
    print(
        f"Built problem: {n_decision_large} decision variables "
        f"(was {n_decision}), {n_variables_large - n_decision_large} slack variables"
    )

    classical_large = solve_with_annealing(
        generators_large, DEMAND_LARGE, problem_large, num_reads=100
    )
    if classical_large:
        print(f"Classical SA baseline: Cost = {classical_large[-1]:.1f} $")

    print("\nRunning the six-generator PCE-VQE variant...")

    t0 = time.time()
    pce_solver_large = solve_with_pce(
        problem_large,
        n_layers=3,
        max_iterations=10,
        population_size=50,
        backend=backend,
    )
    large_time = time.time() - t0

    result_large = find_best_repaired_solution(
        pce_solver_large, generators_large, DEMAND_LARGE
    )

    if result_large is not None:
        powers, cost, prob = result_large
        gen_str = ", ".join(f"P{i + 1}={p:.0f}" for i, p in enumerate(powers))
        print(f"\n   → Best repaired solution: {gen_str} MW")
        print(
            f"     Total={sum(powers):.0f} MW, Cost={cost:.1f} $, "
            f"quantum seed prob={prob:.2%}"
        )
        if classical_large:
            print_comparison(result_large, classical_large, classical_is_optimal=False)

    print(f"\n   Three-generator run: {local_time:.1f}s ({n_decision} variables)")
    print(f"   Six-generator run: {large_time:.1f}s ({n_decision_large} variables)")
