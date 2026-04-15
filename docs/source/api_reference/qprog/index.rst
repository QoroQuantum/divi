Quantum Programs (qprog)
========================

The ``divi.qprog`` module contains the core quantum programming abstractions for
building and executing quantum algorithms. It provides a high-level interface
for quantum algorithm development, supporting both single-instance problems and
large-scale hyperparameter sweeps. At its core is the
:class:`~divi.qprog.QuantumProgram` abstract base class that defines the common
interface for all quantum algorithms.

.. toctree::
   :maxdepth: 1

   core
   algorithms
   problems
   optimizers
   early_stopping
   checkpointing
