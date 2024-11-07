from divi.qprog import VQE, Ansatze, Optimizers

# q_service = QoroService("71ec99c9c94cf37499a2b725244beac1f51b8ee4")

vqe_problem = VQE(
    symbols=["H", "H"],
    bond_lengths=[0.5],
    coordinate_structure=[(0, 0, -0.5), (0, 0, 0.5)],
    ansatze=[Ansatze.HARTREE_FOCK],
    optimizer=Optimizers.MONTE_CARLO,
    shots=500,
    max_interations=4,
    qoro_service=None,  # Run through the local simulator
)

vqe_problem.run()
energies = vqe_problem.energies[vqe_problem.current_iteration - 1]
ansatz = vqe_problem.ansatze[0]
print(energies)
for i in range(len(vqe_problem.bond_lengths)):
    print(energies[i][ansatz][0])
vqe_problem.visualize_results()

# data = []
# for energy in vqe_problem.energies:
#     data.append(energy[Ansatze.HARTREE_FOCK][0])

c = 0
for circuits in vqe_problem.circuits.values():
    for circuit in circuits:
        c += 1

print(f"Total circuits: {c}")
