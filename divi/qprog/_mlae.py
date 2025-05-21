from functools import reduce

import numpy as np
import scipy.optimize as optimize
from qiskit import QuantumCircuit
from qiskit_algorithms import EstimationProblem, MaximumLikelihoodAmplitudeEstimation

from divi.circuits import Circuit
from divi.qprog.quantum_program import QuantumProgram


class BernoulliA(QuantumCircuit):
    """A circuit representing the Bernoulli A operator."""

    def __init__(self, probability):
        super().__init__(1)

        theta_p = 2 * np.arcsin(np.sqrt(probability))
        self.ry(theta_p, 0)


class BernoulliQ(QuantumCircuit):
    """A circuit representing the Bernoulli Q operator."""

    def __init__(self, probability):
        super().__init__(1)

        self._theta_p = 2 * np.arcsin(np.sqrt(probability))
        self.ry(2 * self._theta_p, 0)

    def power(self, k):
        # implement the efficient power of Q
        q_k = QuantumCircuit(1)
        q_k.ry(2 * k * self._theta_p, 0)
        return q_k


class MLAE(QuantumProgram):
    """
    An implementation of the Maximum Likelihood Amplitude Estimateion described in
    https://arxiv.org/pdf/1904.10246
    """

    def __init__(
        self,
        grovers: list[int],
        qubits_to_measure: list[int],
        probability: float,
        **kwargs,
    ):
        """
        Initializes the MLAE problem.
        args:
            grovers (list): A list of non-negative integers corresponding to the powers of the Grover
            operator for each iteration
            qubits: An integer or list of integers containing the index of the qubits to measure
            probability: The probability of being in the good state to estimate
            shots: The number of shots to run for each circuit. Default set at 5000.
        """

        super().__init__(**kwargs)

        self.grovers = grovers
        self.qubits_to_measure = qubits_to_measure
        self.probability = probability
        self.likelihood_functions = []

    def _create_meta_circuits_dict(self):
        return super()._create_meta_circuits_dict()

    def _generate_circuits(self, params=None, **kwargs):
        """
        Generates the circuits that perform step one of the MLAE algorithm,
            the quantum amplitude amplification.

        Inputs a selection of m values corresponding to the powers of the
            Grover operatorfor each iteration.

        Returns:
            A list of QASM circuits to run on various devices
        """
        self.circuits.clear()

        A = BernoulliA(self.probability)
        Q = BernoulliQ(self.probability)

        problem = EstimationProblem(
            state_preparation=A,
            grover_operator=Q,
            objective_qubits=self.qubits_to_measure,
        )

        qiskit_circuits = MaximumLikelihoodAmplitudeEstimation(
            self.grovers
        ).construct_circuits(problem)

        for circuit, grover in zip(qiskit_circuits, self.grovers):
            circuit.measure_all()
            self.circuits.append(Circuit(circuit, tags=[f"{grover}"]))

    def run(self, store_data=False, data_file=None):
        self._generate_circuits()
        self._dispatch_circuits_and_process_results(
            store_data=store_data, data_file=data_file
        )

    def _post_process_results(self, results):
        """
        Generates the likelihood function for each circuit of the quantum
        amplitude amplification. These likelihood functions will then
        be combined to create a maximum likelihood function to analyze.

        Returns:
            A callable maximum likelihood function
        """

        # Define the necessary variables Nk, Mk, Lk
        for label, shots_dict in results.items():
            mk = int(label)
            Nk = 0
            hk = 0
            for key, shots in shots_dict.items():
                Nk += shots
                hk += shots if key.count("1") == len(key) else 0

            def likelihood_function(theta, mk=mk, hk=hk, Nk=Nk):
                as_theta = np.arcsin(np.sqrt(theta))
                return ((np.sin((2 * mk + 1) * as_theta)) ** (2 * hk)) * (
                    (np.cos((2 * mk + 1) * as_theta)) ** (2 * (Nk - hk))
                )

            self.likelihood_functions.append(likelihood_function)

    def generate_maximum_likelihood_function(self, factor=1.0):
        """
        Post-processing takes in likelihood functions.

        A large factor (e.g. 1e200) should be used for visualization purposes.
        Returns:
            The maximum likelihood function.
        """

        def combined_likelihood_function(theta):
            return (
                reduce(
                    lambda result, f: result * f(theta), self.likelihood_functions, 1.0
                )
                * factor
            )

        self.maximum_likelihood_fn = combined_likelihood_function

        return combined_likelihood_function

    def estimate_amplitude(self, factor):
        """
        Uses the maximum likelihood function to ascertain
        a value for the amplitude.

        Returns
            Estimation of the amplitude
        """

        def minimum_likelihood_function(theta):
            # The factor to set to -10e30 in the older branch
            return (
                reduce(
                    lambda result, f: result * f(theta), self.likelihood_functions, 1.0
                )
                * factor
            )

        # create the range of possible amplitudes
        amplitudes = np.linspace(0, 1, 100)

        bounds = [(min(amplitudes), max(amplitudes))]

        return optimize.differential_evolution(minimum_likelihood_function, bounds).x[0]
