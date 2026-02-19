Utilities
=========

Utility functions for quantum computing operations have been organized into specialized modules based on their functionality.

Result Processing Utilities
----------------------------

The ``divi.backends._results_processing`` module provides functions for processing quantum measurement results.

Public functions are available directly from the ``divi.backends`` module:

.. autofunction:: divi.backends.convert_counts_to_probs

.. autofunction:: divi.backends.reverse_dict_endianness

Hamiltonian Utilities
---------------------

The ``divi.hamiltonians`` module provides functions for working with Hamiltonians, including QUBO conversion and Hamiltonian manipulation.

.. autofunction:: divi.hamiltonians.convert_hamiltonian_to_pauli_string

.. autofunction:: divi.hamiltonians.convert_qubo_matrix_to_pennylane_ising

Expectation Value Computation
-----------------------------

The ``divi.qprog._expectation`` module provides internal functions for efficiently computing expectation values from quantum measurement results. These functions are primarily used internally by variational quantum algorithms.

.. note::
   The functions in this module are internal implementation details and are not part of the public API. They are documented here for completeness but should not be used directly by end users.
