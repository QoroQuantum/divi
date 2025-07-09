from divi.parallel_simulator import ParallelSimulator
from divi.qprog import MLAE

if __name__ == "__main__":

    mlae_problem = MLAE(
        grovers=[2, 3, 4],
        qubits_to_measure=0,
        probability=0.2,
        backend=ParallelSimulator(),
    )

    mlae_problem.run()

    print(f"Amplitude Estimate: {mlae_problem.estimate_amplitude(-(10e30))}")
