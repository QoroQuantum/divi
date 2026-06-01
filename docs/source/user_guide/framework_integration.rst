PennyLane & Qiskit Integration
==============================

Divi is framework-agnostic at its input boundary. You can author circuits in
either PennyLane or Qiskit and hand them to Divi, which compiles, batches, and
runs them through the same pipeline the built-in algorithms use. This guide
covers how external circuits enter Divi and how to wrap your own circuit as an
optimizable program — including QML-style data binding for quantum neural
networks.

.. note::
   For the logical circuit IR these inputs are converted into, see the
   :class:`~divi.circuits.MetaCircuit` discussion in :doc:`core_concepts`.

.. tip::
   **Two ways to do QML in Divi.** For most quantum-machine-learning work, reach
   for the curated :class:`~divi.qprog.algorithms.QNN` primitive
   (:doc:`quantum_neural_networks`) — it composes a feature map and ansatz for
   you. Use the ``CustomVQA`` data-binding path on this page only when you bring
   your own circuit and want to mark its data parameters by hand.

How External Circuits Enter Divi
--------------------------------

Every pipeline begins with a :class:`~divi.pipeline.SpecStage` that converts a
circuit spec into a :class:`~divi.circuits.MetaCircuit` batch. Two of these
stages bridge directly from the major quantum frameworks:

- :class:`~divi.pipeline.stages.PennyLaneSpecStage` — converts a PennyLane
  ``QuantumScript`` or ``QNode`` into a MetaCircuit. QNodes are traced into a
  symbolic tape, with the trainable arguments seeded as ``sympy`` symbols.
- :class:`~divi.pipeline.stages.QiskitSpecStage` — converts a Qiskit
  ``QuantumCircuit`` into a MetaCircuit, mapping ``measure`` instructions to a
  probability measurement over the measured wires.

You rarely call these stages by hand. :class:`~divi.qprog.algorithms.CustomVQA`
selects the right one based on the object you pass it, then drives the resulting
circuit through binding, execution, and reduction — the same machinery used
throughout Divi. For a manual, stage-by-stage walkthrough see :doc:`pipelines`.

Bring Your Own Circuit with CustomVQA
-------------------------------------

Built-in algorithms like :class:`~divi.qprog.algorithms.VQE`,
:class:`~divi.qprog.algorithms.QAOA`, and
:class:`~divi.qprog.algorithms.TimeEvolution` generate their circuits
automatically — you don't need to build circuits manually for most use cases.

When you need a **custom ansatz or circuit**, use
:class:`~divi.qprog.algorithms.CustomVQA`. It lets you define your own circuit
template and Hamiltonian while Divi handles compilation, execution, and
optimization:

.. code-block:: python

   import pennylane as qp
   from divi.qprog import CustomVQA
   from divi.backends import MaestroSimulator

   qscript = qp.tape.QuantumScript(
       ops=[
           qp.RY(0.0, wires=0),
           qp.RX(0.0, wires=1),
           qp.CNOT(wires=[0, 1]),
       ],
       measurements=[qp.expval(qp.Z(0) @ qp.Z(1) + 0.5 * qp.X(0))],
   )

   # Freeze the Hamiltonian coefficient so only gate parameters are trainable
   qscript.trainable_params = [0, 1]

   program = CustomVQA(
       qscript=qscript,
       backend=MaestroSimulator(),
   )
   program.run(perform_final_computation=False)

In this example, the ``0.0`` values in ``ops`` are placeholders. ``CustomVQA``
replaces trainable slots with internal symbols and optimizes them. By default
the optimizer sees a flat vector of those parameters; pass the optional
``param_shape`` only to reshape them (for example, into per-layer angle
matrices).

``CustomVQA`` also accepts a PennyLane ``QNode`` (its trainable arguments become
the parameters) and a Qiskit ``QuantumCircuit`` (computational-basis
measurements map to a sum-of-Z observable).

Data Binding (Quantum Neural Networks)
--------------------------------------

For QML-style workflows, some parameters are *data* — bound from a classical
feature batch every iteration — and only the rest are trained. For a
multi-argument QNode (a feature map plus a trainable ansatz), declare the weight
shapes with ``arg_shapes`` and name the data argument with ``data_arg`` (its
shape is taken from ``feature_batch``). Only the weights reach the optimizer;
the per-sample losses are aggregated by ``loss_reduction`` (``"mean"``,
``"sum"``, or a callable):

.. code-block:: python

   import numpy as np
   import pennylane as qp
   from divi.qprog import CustomVQA
   from divi.backends import MaestroSimulator

   n_qubits = 3

   @qp.qnode(qp.device("default.qubit", wires=n_qubits))
   def circuit(inputs, weights):
       qp.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
       qp.StronglyEntanglingLayers(weights, wires=range(n_qubits))
       return qp.expval(qp.Z(0) @ qp.Z(1) @ qp.Z(2))

   feature_batch = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

   program = CustomVQA(
       qscript=circuit,
       arg_shapes={"weights": (1, n_qubits, 3)},
       data_arg="inputs",
       feature_batch=feature_batch,
       backend=MaestroSimulator(),
       max_iterations=2,
   )
   program.run(perform_final_computation=False)

PennyLane templates — including nonlinear feature maps such as
``qp.IQPEmbedding`` — are supported. If the circuit is decorated with
``@qp.batch_input(argnum=...)``, the data argument is detected automatically
and ``data_arg`` can be omitted. For a Qiskit circuit, mark data parameters by
index with ``data_param_indices`` instead of ``arg_shapes``/``data_arg``.

.. note::
   **Structural arguments** — qubit or layer counts used only for control flow
   (the ``n_qubits`` in ``range(n_qubits)`` above, loop bounds, and similar) —
   are neither data nor weights. Close over them in the enclosing scope (as
   ``n_qubits`` is above) or give them a Python default; a *no-default*
   structural argument is symbolized like a weight and then breaks (e.g.
   ``range(<symbol>)``). The QNode is traced one sample at a time, so index by
   the structural size (``range(n_qubits)``) rather than the batch dimension
   (``len(inputs[0])``).

Data binding and ``param_shape`` are mutually exclusive: when you mark data
parameters, the optimizer's weight view is automatically flat over the
remaining parameters.

For **supervised** training, pass ``labels`` (shape ``(n_samples,)``) alongside
the feature batch: each sample's expectation value is compared to its label via
``loss_fn`` (``"squared_error"`` by default, or a callable) before
``loss_reduction`` aggregates. This is the same supervised interface as
:class:`~divi.qprog.algorithms.QNN` — see :doc:`quantum_neural_networks`.

For the full tutorial, see `custom_vqa.py <https://github.com/QoroQuantum/divi/blob/main/tutorials/advanced/custom_vqa.py>`_.

Next Steps
----------

- :doc:`quantum_neural_networks` — the curated ``QNN`` primitive for QML
- :doc:`core_concepts` — the :class:`~divi.circuits.MetaCircuit` IR these inputs
  compile to, and the variational run lifecycle
- :doc:`pipelines` — the stage-by-stage view, including a manual ``CustomVQA``
  pipeline walkthrough
- :doc:`../api_reference/qprog/algorithms` — full
  :class:`~divi.qprog.algorithms.CustomVQA` API
