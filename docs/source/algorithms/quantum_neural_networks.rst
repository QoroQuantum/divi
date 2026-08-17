Quantum Neural Networks
=======================

:class:`~divi.qprog.algorithms.QNN` is Divi's first-class primitive for
quantum machine learning. It trains the weights of a parameterized circuit over
a batch of classical feature vectors. By default it minimizes the expectation
value of a chosen observable averaged across the batch (unsupervised); pass
``labels`` to train a **supervised** loss instead. The quick start uses this
supervised path; unsupervised reduction is described under
`The Observable and the Loss`_.

Unlike :doc:`../execution_workflows/framework_integration` (where you bring your own circuit via
:class:`~divi.qprog.algorithms.CustomVQA`), ``QNN`` **builds the circuit for
you** from two composable pieces:

- a :class:`~divi.qprog.algorithms.FeatureMap` — the *data* layer. It encodes
  each feature vector into circuit parameters that are bound from data, never
  optimized.
- an :class:`~divi.qprog.algorithms.Ansatz` — the *trainable* layer. Any Divi
  ansatz works; its parameters are the weights the optimizer updates.

Data parameters and weights remain disjoint. In unsupervised mode, the pipeline
reduces per-sample expectations directly. With labels, ``loss_fn`` first turns
each prediction-label pair into a loss, then ``loss_reduction`` aggregates it.
The optimizer sees one scalar per weight candidate, never the data axis.

Quick Start
-----------

Train a tiny 2-qubit classifier on four labeled samples:

.. code-block:: python

   import numpy as np
   from qiskit.circuit.library import CXGate, RYGate, RZGate

   from divi.qprog import QNN, AngleEmbedding, GenericLayerAnsatz
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import MaestroSimulator

   n_qubits = 2

   # Two loose clusters: (n_samples, n_features). n_features must match the
   # feature map's parameter count — AngleEmbedding uses one feature per qubit.
   X_train = np.array([[0.1, 0.2], [0.3, 0.5], [2.0, 2.1], [2.3, 2.4]])
   y_train = np.array([-1.0, -1.0, 1.0, 1.0])

   program = QNN(
       n_qubits=n_qubits,
       feature_map=AngleEmbedding(rotation="Y"),
       ansatz=GenericLayerAnsatz(
           gate_sequence=[RYGate, RZGate],
           entangler=CXGate,
           entangling_layout="linear",
       ),
       feature_batch=X_train,
       labels=y_train,
       loss_fn="squared_error",
       n_layers=2,
       loss_reduction="mean",
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       max_iterations=5,
       backend=MaestroSimulator(),
       seed=1997,
   )

   # QNNs return their answer as the trained weights, so the solution-sampling
   # "final computation" step that combinatorial VQAs use is unnecessary.
   program.run(perform_final_computation=False)

   print(f"Best loss: {program.best_loss:.4f}")
   best_weights = program.best_params

The optimizer view (:attr:`~divi.qprog.VariationalQuantumAlgorithm.best_params`)
contains only the **weights** —
``ansatz.n_params_per_layer(n_qubits) * n_layers`` of them. The feature columns
never appear in the parameter vector.

**Loss trajectory.** After ``run()`` completes, the training history is exposed
on the program via
:attr:`~divi.qprog.VariationalQuantumAlgorithm.min_losses_per_iteration` and
:attr:`~divi.qprog.VariationalQuantumAlgorithm.losses_history` — see
:ref:`reading results <reading-results>` in core concepts for their
semantics and types.

Per-iteration INFO log lines are emitted to the ``divi`` logger during
optimization.  To silence them: ``from divi.reporting import disable_logging``
(call ``enable_logging()`` to restore).

Inference
---------

After ``run()``, score a fresh feature batch with
:meth:`~divi.qprog.algorithms.DataBindingMixin.predict`. It binds each row's
features with the trained weights, estimates the cost observable per sample, and
returns the **sign** as a class label in ``{-1, +1}``.  For continuous /
regression output — or to apply a custom threshold — pass
``return_scores=True`` to get the raw ``⟨H⟩`` value per sample instead.  Either
way ``predict`` returns a ``numpy.ndarray`` of shape ``(n_samples,)``:

.. invisible-code-block: python

   import numpy as np
   from qiskit.circuit.library import CXGate, RYGate, RZGate
   from divi.qprog import QNN, AngleEmbedding, GenericLayerAnsatz
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import MaestroSimulator

   program = QNN(
       n_qubits=2,
       feature_map=AngleEmbedding(rotation="Y"),
       ansatz=GenericLayerAnsatz(
           gate_sequence=[RYGate, RZGate], entangler=CXGate,
           entangling_layout="linear",
       ),
       feature_batch=np.array([[0.1, 0.2], [2.0, 2.1]]),
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       backend=MaestroSimulator(),
   )
   trained_weights = np.array([0.5, 1.0, 1.5, 2.0])

.. code-block:: python

   X_new = np.array([[0.15, 0.25], [2.1, 2.2]])

   labels = program.predict(X_new, params=trained_weights)   # {-1, +1} per row
   scores = program.predict(X_new, params=trained_weights, return_scores=True)  # raw ⟨H⟩ per row

When called after training, ``params`` defaults to ``best_params``. Pass
``return_scores=True`` to :meth:`~divi.qprog.algorithms.DataBindingMixin.predict` for the
continuous scores if you want to apply your own decision threshold. The same method is
available on :class:`~divi.qprog.algorithms.CustomVQA` when it has a data axis.

**Regression use.** ``return_scores=True`` returns the raw ⟨H⟩ value per sample —
a continuous output in the observable's readout range.  For the default all-qubit
parity observable (``Z ⊗ Z ⊗ … ⊗ Z``) that range is ``[-1, 1]``.  When training
a regressor, scale your continuous targets into this range. Targets outside it
are unattainable, forcing predictions toward the boundary and potentially
slowing or preventing convergence. For a
custom observable, inspect its eigenvalue range and scale accordingly.

**Low-dimensional inputs.** The built-in feature maps
(:class:`~divi.qprog.algorithms.AngleEmbedding` and
:class:`~divi.qprog.algorithms.ZZFeatureMap`) consume exactly **one feature per
qubit**: ``feature_batch`` must have ``n_qubits`` columns.  If your dataset has
fewer features than qubits, pad the feature vectors or use a custom
:class:`~divi.qprog.algorithms.FeatureMap` subclass that maps fewer inputs to more
qubits (e.g. via data re-uploading).

Feature Maps
------------

A feature map encodes a feature vector into a parametric circuit. Built-in maps:

- :class:`~divi.qprog.algorithms.AngleEmbedding` — one single-qubit rotation per
  feature, ``R(x_i)`` on qubit ``i``. Choose the axis with
  ``rotation="X" | "Y" | "Z"`` (default ``"Y"``).
- :class:`~divi.qprog.algorithms.ZZFeatureMap` — the ZZ entangling encoding of
  Havlíček et al. (Hadamards, ``RZ(2·x_i)`` per qubit, then
  ``RZZ(2·(π−x_i)(π−x_j))`` over an entangling layout). Select the pair pattern
  with ``entangling_layout="linear" | "circular" | "all-to-all"``. Requires at
  least two qubits.

Both built-in maps consume **one feature per qubit**, so ``feature_batch`` must
have ``n_qubits`` columns. More generally, its column count must equal
``feature_map.n_params(n_qubits)``; a mismatch raises ``ValueError`` at
construction. Feature maps are *not* layered — a single application encodes the
vector once. For data re-uploading (interleaving encoding with
variational layers), subclass :class:`~divi.qprog.algorithms.FeatureMap` and
implement ``n_params`` and ``build``.

The Observable and the Loss
---------------------------

``observable`` produces one prediction per sample, as a
:class:`~qiskit.quantum_info.SparsePauliOp` acting on ``n_qubits`` qubits. It
defaults to the all-qubit parity ``Z ⊗ Z ⊗ … ⊗ Z``, which gives a single
readout in ``[-1, 1]`` informed by every qubit. Pass your own to change the
readout, e.g. ``SparsePauliOp.from_list([("ZI", 1.0)])`` to read a single qubit.
Unsupervised mode minimizes this expectation directly; supervised mode compares
it with each label through ``loss_fn``.

``loss_reduction`` aggregates per-sample expectations in unsupervised mode and
per-sample losses in supervised mode:

- ``"mean"`` (default) — average over the batch.
- ``"sum"`` — total over the batch.
- a callable ``np.ndarray (n_samples,) -> float`` — any custom aggregation.

Supervised Loss and Readout
---------------------------

The quick start passes ``labels`` with shape ``(n_samples,)``, aligned with
``feature_batch``'s rows. Each sample's prediction — the cost observable's
expectation value, in ``[-1, 1]`` for the default parity observable — is
compared to its label by ``loss_fn``, and those per-sample losses are then
aggregated by ``loss_reduction``. The default ``loss_fn="squared_error"`` with
the default ``"mean"`` reduction is mean-squared error; encode labels to match
the readout range (e.g. ``-1`` / ``+1`` for the parity observable):

.. code-block:: python

   import numpy as np
   from qiskit.circuit.library import CXGate, RYGate, RZGate

   from divi.qprog import QNN, AngleEmbedding, GenericLayerAnsatz
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
   from divi.backends import MaestroSimulator

   X_train = np.array([[0.1, 0.2], [0.3, 0.5], [2.0, 2.1], [2.3, 2.4]])
   y_train = np.array([-1.0, -1.0, 1.0, 1.0])  # one label per sample

   clf = QNN(
       n_qubits=2,
       feature_map=AngleEmbedding(rotation="Y"),
       ansatz=GenericLayerAnsatz(
           gate_sequence=[RYGate, RZGate],
           entangler=CXGate,
           entangling_layout="linear",
       ),
       feature_batch=X_train,
       labels=y_train,
       loss_fn="squared_error",  # default; or a callable (pred, label) -> float
       n_layers=2,
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       max_iterations=5,
       backend=MaestroSimulator(),
       seed=1997,
   )
   clf.run(perform_final_computation=False)

A supervised loss requires a single :class:`~qiskit.quantum_info.SparsePauliOp`
observable whose terms sum to one scalar prediction per sample.  A multi-term
``SparsePauliOp`` is supported — its terms are evaluated together and their
weighted sum is the prediction.  A **list** of observables is not supported on
the supervised path; keep the default parity observable or pass a single
``SparsePauliOp``.  The same ``labels`` / ``loss_fn`` pair is available on
:class:`~divi.qprog.algorithms.CustomVQA`'s data-binding path
(:doc:`../execution_workflows/framework_integration`) when you bring your own circuit.

When to use QNN vs CustomVQA
----------------------------

- Reach for :class:`~divi.qprog.algorithms.QNN` when you want a curated
  feature-map + ansatz workflow and Divi to compose and bind the circuit for
  you.
- Reach for :class:`~divi.qprog.algorithms.CustomVQA` (see
  :doc:`../execution_workflows/framework_integration`) when you already have a PennyLane or Qiskit
  circuit and want full control over its structure, marking data parameters
  yourself with ``data_arg`` / ``arg_shapes`` / ``data_param_indices``.

Next Steps
----------

- :doc:`../execution_workflows/framework_integration` — bring-your-own-circuit data binding with
  ``CustomVQA``
- :doc:`../execution_workflows/optimizers` — optimizer choice and early stopping
- :doc:`../api_reference/qprog/algorithms` — full ``QNN``, ``FeatureMap``, and
  ``Ansatz`` API
- `qnn_classifier.py <https://github.com/QoroQuantum/divi/blob/main/tutorials/advanced/qnn_classifier.py>`_
  — the runnable end-to-end tutorial
