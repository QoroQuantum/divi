from divi.qprog import VQEHyperparameterSweep, VQEAnsatze
from divi.qprog.optimizers import Optimizers


batch = VQEHyperparameterSweep(
    bond_lengths=[0.5, 0.75, 1],
    ansatze=[VQEAnsatze.HARTREE_FOCK],
    symbols=["H", "H"],
    coordinate_structure=[(0, 0, -0.5), (0, 0, 0.5)],
    optimizer=Optimizers.MONTE_CARLO,
    shots=5000,
    max_iterations=4,
    qoro_service=None,  # Run through the local simulator
)

batch.create_programs()
batch.run()
batch.wait_for_all()
result = batch.aggregate_results()
batch.visualize_results()
