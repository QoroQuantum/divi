Pipeline
========

The ``divi.pipeline`` module provides the circuit pipeline engine that orchestrates
circuit generation, transformation, execution, and result reduction.

.. seealso::
   For a conceptual introduction, see the User Guide: :doc:`../user_guide/pipelines`.

Core
----

.. autoclass:: divi.pipeline.CircuitPipeline
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: divi.pipeline.PipelineEnv
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: divi.pipeline.PipelineTrace
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: divi.pipeline.ExpansionResult
   :members:
   :undoc-members:
   :show-inheritance:

Abstract Base Classes
---------------------

.. autoclass:: divi.pipeline.Stage
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: divi.pipeline.SpecStage
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: divi.pipeline.BundleStage
   :members:
   :undoc-members:
   :show-inheritance:

Built-in Stages
---------------

.. autoclass:: divi.pipeline.stages.CircuitSpecStage
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: divi.pipeline.stages.TrotterSpecStage
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: divi.pipeline.stages.MeasurementStage
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: divi.pipeline.stages.ParameterBindingStage
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: divi.pipeline.stages.QEMStage
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: divi.pipeline.stages.PCECostStage
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Utility Functions
-----------------

.. autofunction:: divi.pipeline.format_pipeline_tree

.. autofunction:: divi.pipeline.reduce_merge_histograms
