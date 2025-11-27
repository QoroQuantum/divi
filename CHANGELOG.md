<!-- markdownlint-disable MD024 -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### ✨ Added

* Checkpointing support for variational quantum algorithms: added comprehensive checkpointing functionality with `CheckpointConfig` class (including `with_timestamped_dir()` method) using Pydantic for JSON validation, enabling state saving and resuming of optimization runs. Includes `save_state()` and `load_state()` methods on optimizer classes, comprehensive user guide (`docs/source/user_guide/checkpointing.rst`), and tutorial example (`tutorials/checkpointing.py`)
* `precision` parameter to `VariationalQuantumAlgorithm`: added configurable precision for QASM parameter formatting (defaults to 8 decimal places). The precision parameter controls the number of decimal places used when converting circuit parameters to QASM strings, affecting the size of QASM circuits sent to cloud backends. Higher precision values result in longer QASM strings and increased data transfer overhead
* Job cancellation support in `QoroService`: added `cancel_job()` method to cancel pending or running jobs on the Qoro Service API. The method returns a `requests.Response` object containing cancellation details (status, job_id, circuits_cancelled). Includes comprehensive test coverage for successful cancellation, permission errors (403), and conflict errors (409) when attempting to cancel non-cancellable jobs

### 🔄 Changed

* Refactored type hints to use `numpy.typing` for improved type safety: updated type annotations across the codebase to use numpy's typing module for better type checking and IDE support
* Adapted `ProgramBatch` workflows to support stateful optimizers: updated workflow classes (`GraphPartitioningQAOA`, `QUBOPartitioningQAOA`, `VQEHyperparameterSweep`) to properly handle optimizer state persistence
* Thread-safe QuantumScript creation: refactored circuit creation to avoid using `make_qscript` for improved thread safety in parallel execution scenarios
* Refactored logging infrastructure to use Rich library: replaced custom `OverwriteStreamHandler` with `RichHandler` from the Rich library for improved log formatting and colorization. `LoggingProgressReporter` now uses Rich's `Console.status()` for message overwriting with spinners, providing better visual feedback during job polling and iteration updates. Removed ANSI escape sequence handling in favor of Rich's markup system

### ⚠️ Deprecated

### 🗑️ Removed

### 🐛 Fixed

* Fixed overflow issue with batched expectation values: corrected numerical overflow problem in expectation value calculations when processing large batches
* Fixed documentation issues: corrected various documentation hiccups and formatting problems
* Fixed `_raise_with_details()` to preserve response object: updated error handling to attach the HTTP response object to `HTTPError` exceptions, enabling proper error inspection in tests and error handling code

## [0.4.2] - 2025-11-18

### 🐛 Fixed

* Fixed `ProgramBatch` bug where `QuantumScript` operations got intermangled due to thread-unsafe implementation.

## [0.4.1] - 2025-11-16

### ✨ Added

* New test files for `CircuitRunner` (54 lines), `ParallelSimulator` (320 lines), Cirq integration (160 lines), QASM validation (338 lines), progress bars (170 lines), loggers (293 lines), and reporters (190 lines)
* Python 3.12 support in CI/CD test matrix alongside Python 3.11
* Autoflake to pre-commit hooks for automatic unused import and variable removal
* Ability to disable final computation step in variational quantum algorithms via `perform_final_computation` parameter

### 🔄 Changed

* Updated Poetry from version 1.8.5 to 2.2.1
* Replaced `Circuit` class with `ExecutableQASMCircuit` and `CircuitBundle` dataclasses for immutable circuit representation
* Replaced Pennylane's `split_non_commuting` transform with manual wire-based grouping implementation for improved observable grouping performance
* Changed `is_valid_qasm()` return type from `bool | str` to `bool` for clearer API, and renamed `validate_qasm_raise()` to `validate_qasm()`
* Moved `optimizer` parameter from `VQE` and `QAOA` constructors to parent `VariationalQuantumAlgorithm` class for consistent interface
* Refactored Cirq QASM parser and export functionality: moved from `divi/extern/cirq/` to `divi/circuits/_cirq/` with rewritten parser implementation
* Reorganized QASM validation and conversion files: moved from `divi/extern/cirq/` to `divi/circuits/`
* Updated Mitiq, Cirq, and Scipy dependencies
* Enhanced API documentation: improved docstrings and updated API reference documentation

### 🗑️ Removed

* Removed redundant `extern` module (Cirq code moved to `circuits/`, Scipy COBYLA implementation removed in favor of external dependency)

### 🐛 Fixed

* Fixed QAOA bug for non-integer graph labels
* Fixed parameter initialization in `VariationalQuantumAlgorithm` where user-provided `initial_params` were being overwritten by automatic initialization
* Fixed test failures related to removed `_meta_circuits` attribute and incorrect parameter ordering
* Fixed Sphinx documentation configuration to read from `project` section instead of deprecated `tool.poetry` section in pyproject.toml

## [0.4.0] - 2025-11-10

### ✨ Added

* Support for expectation value calculation in QoroService (Maestro backend): ability to compute expectation values directly without sampling when backend supports it
* `JobConfig` dataclass for improved job management in QoroService: provides structured configuration with validation, override capabilities, and QPU system resolution
* Ability to maintain older best parameters in new Monte-Carlo population via `keep_best_params` parameter in `MonteCarloOptimizer`
* Comprehensive Sphinx documentation: user guides covering core concepts, backends, optimizers, program batches, QAOA, VQE, and error mitigation, plus complete API reference
* GitHub Actions workflow for automated Sphinx documentation deployment to GitHub Pages
* Matrix-aware dependency caching for CI/CD workflows: caching keys that account for Python version matrix
* Coverage configuration for tests: `.coveragerc` file to track test coverage
* `reverse_dict_endianness` utility function for converting probability dictionaries between endianness conventions
* `supports_expval` property to `QPUSystem` class to expose expectation value calculation capabilities
* `VariationalQuantumAlgorithm` base class: new abstract base class extracted from `QuantumProgram` containing 872 lines of common variational algorithm logic (optimization loop, parameter management, loss tracking) for better code organization and reusability
* Comprehensive test suite for v0.4.0: new test files for `Optimizer` classes, `ProgramBatch`, `QuantumProgram`, `VariationalQuantumAlgorithm`, plus reorganized existing tests into `tests/backends/`, `tests/circuits/`, and `tests/reporting/` subdirectories

### 🔄 Changed

* Converted `QPU` and `QPUSystem` classes to frozen dataclasses for immutability and cleaner code
* Refactored `CircuitRunner` backend interface: replaced `isinstance()` checks with abstract properties `is_async` and `supports_expval` to enable polymorphism without concrete class dependencies
* Enhanced optimizer callback functions: ensured consistent parameter and cost array shapes across all optimizers, and updated type hints from `Callable | None` to `Callable[[OptimizeResult], Any] | None` for better type safety
* Refactored circuit management: renamed `_circuits` to `_curr_circuits` in `QuantumProgram` and `VariationalQuantumAlgorithm` to better reflect that it stores only the current iteration's circuits
* Simplified probability storage in `VariationalQuantumAlgorithm`: changed from storing `_best_probs` and `_final_probs` dictionaries to computing them on-demand using new `_process_probability_results()` method with `reverse_dict_endianness` for consistent endianness handling, and refactored `_perform_final_computation()` to accept keyword arguments
* Updated documentation and examples: changed `best_loss` retrieval method for VQE and QAOA algorithms, and enhanced user guide with backend selection and error handling strategies
* Improved Sphinx documentation configuration: enhanced `conf.py` with better metadata extraction and API reference organization
* Refactored `QuantumProgram` class: extracted variational algorithm logic into `VariationalQuantumAlgorithm` base class, reducing `QuantumProgram` from 759 lines to a more focused abstract base class
* Streamlined CI/CD workflows: consolidated and simplified GitHub Actions workflows for testing and documentation deployment
* Refactored progress reporting: extracted reporting logic from `QuantumProgram` into abstract `ProgressReporter` base class with `QueueProgressReporter` and `LoggingProgressReporter` implementations for better separation of concerns
* Enhanced progress bar display: improved polling status to show job ID with colored formatting and removed hardcoded `max_retries` parameter in favor of dynamic task fields
* Improved logger message handling: replaced `\c` prefix magic string with cleaner `append` attribute on log records for better message overwriting control
* Changed `use_circuit_packing` parameter handling in `QoroService` to explicitly default to `False` when `None`
* Simplified GitHub Pages deployment workflow: removed redundant artifact steps and enabled cancel-in-progress for concurrent deployments
* Updated Poetry dependency group syntax in CI/CD workflows: changed from `--only docs` to `--with docs` and `--with tutorials` to `--with testing`
* Updated tutorial examples: changed `MonteCarloOptimizer` to use `population_size` parameter instead of deprecated `n_param_sets`

### 🐛 Fixed

* Fixed `QoroService.submit_circuits()` bug where `override_config` was incorrectly applied, causing wrong config values to be used for job initialization and circuit submission

## [0.3.5] - 2025-10-15

### ✨ Added

* Pymoo optimizers integration: added `PymooOptimizer` class supporting CMAES (Covariance Matrix Adaptation Evolution Strategy) and DE (Differential Evolution) algorithms from the pymoo library
* `batched_expectation()` function: vectorized expectation value calculation for multiple observables across multiple shot histograms using matrix operations, replacing loop-based `ExpectationMP.process_counts()` calls for improved performance
* New job results endpoint support in `QoroService`: added JSON decompression routine for handling compressed job result payloads from Qoro API
* `initial_params` parameter to `QuantumProgram.run()`: ability to initialize program parameters instead of using random initialization
* Deterministic execution mode in `ParallelSimulator`: added `_deterministic_execution` parameter and `_execute_circuits_deterministically()` method for debugging non-deterministic behavior
* Graceful shutdown handling in `ProgramBatch`: improved cancellation logic that properly tracks which futures were successfully cancelled vs unstoppable

### 🔄 Changed

* Refactored `ParallelSimulator` to use Qiskit's built-in parallelism: replaced custom parallelization with Qiskit Aer's native parallel execution capabilities
* Swapped `ProcessPoolExecutor` with `ThreadPoolExecutor` in `ParallelSimulator`: changed from process-based to thread-based parallelism for better performance and resource usage
* Improved `submit_circuits` in `ParallelSimulator`: enhanced batch transpilation and execution with better error handling
* Enhanced runtime estimation: improved variable naming and clarity in execution time calculation
* Updated all docstrings and added missing ones across the codebase
* Standardized optimizer callback function signatures and parameter handling to support Pymoo optimizers

### 🗑️ Removed

* Removed static `simulate_circuit` method from `ParallelSimulator`: consolidated circuit execution into instance methods

### 🐛 Fixed

* Fixed QoroService tests: updated tests to match new init/add circuit submission flow
* Fixed test failures after optimizers interface change: updated test expectations for new optimizer signatures
* Fixed and simplified Qoro API tests behavior: improved test reliability and reduced flakiness

## [0.3.4] - 2025-10-04

### ✨ Added

* `GenericLayerAnsatz` class: flexible ansatz that alternates single-qubit gates with optional entanglers, supporting custom gate sequences and entangling layouts (linear, brick, circular, all-to-all)
* `Ansatz` abstract base class: new abstraction for all VQE ansaetze with `n_params_per_layer()` and `build()` methods for better extensibility
* Comprehensive test suite for ansaetze: 241 lines of tests in `test_ansatze.py` covering all ansatz implementations

### 🔄 Changed

* Restructured project for better modularity: moved `exp/` directory to `extern/`, moved backend-related files (`interfaces.py`, `qoro_service.py`, `qpu_system.py`, `_parallel_simulator.py`) to `backends/` subdirectory, moved circuit-related files to `circuits/` subdirectory, and reorganized test files to match new structure
* Improved QASM validation API: changed `is_valid_qasm()` return type from `bool` to `bool | str` to return error messages, and updated `QoroService.submit_circuits()` to use this improved validation with clearer error messages
* Enhanced QASM parser error handling: added validation to prevent redefinition of built-in gates with clearer error messages including line and column numbers
* Refactored VQE to use new Ansatz abstraction: extracted ansatz logic into `_ansatze.py` module, reducing VQE code complexity and enabling better extensibility
* Updated tutorial examples: modified tutorial examples to use new ansatz interface

### 🐛 Fixed

* Fixed wire order handling in expectation value calculation: updated to use `cost_hamiltonian.wires` when available instead of assuming integer wire indices, enabling support for non-integer wire labels
* Fixed sparse input bug in QUBO conversion: replaced `.A1` attribute access with `.tocoo()` method for proper sparse matrix handling in `convert_qubo_matrix_to_pennylane_ising()`
* Fixed test failures related to ansatz refactoring: updated VQE and VQE sweep tests to work with new Ansatz abstraction interface

## [0.3.3] - 2025-09-18

### ✨ Added

* `Optimizer` abstract base class: new abstraction replacing the `Optimizer` enum, with `ScipyOptimizer` and `MonteCarloOptimizer` concrete implementations for improved flexibility and extensibility
* `ScipyMethod` enum: new enum for scipy optimization methods (Nelder-Mead, COBYLA, L-BFGS-B) used by `ScipyOptimizer`
* `ProgressReporter` abstract base class: new abstraction for progress reporting with `QueueProgressReporter` and `LoggingProgressReporter` implementations, extracted from `QuantumProgram` for better separation of concerns
* Comprehensive tests for optimizer behavior and callback consistency: added tests in `test_core.py` to verify optimizer interfaces and callback function behavior
* ZNE tests: added comprehensive test suite for Zero Noise Extrapolation functionality in `test_core.py`

### 🔄 Changed

* Refactored optimizer interface: replaced `Optimizer` enum usage throughout codebase with new `Optimizer` abstract base class and concrete implementations (`ScipyOptimizer`, `MonteCarloOptimizer`)
* Extracted progress reporting logic: moved reporting functionality from `QuantumProgram` into separate `ProgressReporter` abstraction in new `reporter.py` module
* Enhanced `poll_callback` API in `QoroService.get_job_status()`: added `status` parameter to function signature to expose job status during polling
* Optimized result processing performance: refactored `_post_process_results()` in `QuantumProgram` to use `itertools.groupby` for more efficient grouping of execution results by parameter ID and QEM index
* Updated Pennylane version to fix autoray dependency issue
* Improved VQE sweep tutorial: minor enhancements to tutorial examples

### 🐛 Fixed

* Fixed ZNE errors: corrected ZNE implementation issues in `_post_process_results()` method
* Fixed progress bar reporter bugs: corrected lambda function signature in `QoroService.get_job_status()` default callback to accept both retry count and status parameters, and fixed conditional logic in `ProgramBatch.join()` method
* Fixed expected fail detection in tests: corrected test framework configuration for proper handling of expected test failures

## [0.3.2b0] - 2025-08-28

### ✨ Added

* Long description for PyPI package: added comprehensive package description to `pyproject.toml` for better PyPI presentation
* `QPU` and `QPUSystem` dataclasses: new frozen dataclasses for representing Quantum Processing Units and QPU systems with parsing functionality from Qoro API JSON responses, including `qpu_systems()` method in `QoroService` to fetch available systems
* `MoleculeTransformer` class: new utility class exported from `_vqe_sweep` module for molecular transformations
* `_raise_with_details()` method in `QoroService`: improved error handling helper that provides detailed error information when circuit submission fails

### 🔄 Changed

* Refactored `QoroService` class structure: moved QPU parsing logic to `_parse_qpu_systems()` function, added support for loading auth token from `.env` file via python-dotenv, improved constructor parameter handling, and integrated new `QPU` and `QPUSystem` dataclasses
* Updated `VQE` class API: changed constructor to accept `hamiltonian` (Pennylane Operator) and `molecule` (Pennylane Molecule) objects instead of `symbols`, `bond_length`, and `coordinate_structure` parameters, with optional `n_electrons` parameter
* Revamped `VQEHyperparameterSweep` class: improved molecular manipulation accuracy through z-matrices for more precise molecular geometry transformations
* Replaced Qiskit expectation value calculation functions with Pennylane equivalents: updated codebase to use Pennylane's native expectation value calculation methods
* Replaced all instances of `Optional` type hint with pipe operator (`|`): migrated codebase-wide from `Optional[T]` to `T | None` syntax for Python 3.10+ compatibility
* Updated tutorial examples: modified all tutorial examples to use new VQE API with `Molecule` objects and `Hamiltonian` inputs, and corrected ZNE tutorial to work with updated codebase

### 🗑️ Removed

* Deleted all MLAE (Maximum Likelihood Amplitude Estimation) functionality: removed `_mlae.py` module (182 lines) and `MLAE` class, along with `mlae_example.py` tutorial

### 🐛 Fixed

* Fixed Qoro API tests: updated test expectations to match new API behavior and QASM validation integration

## [0.3.1b0] - 2025-08-22

### ✨ Added

* `QUBOPartitioningQAOA` class: new `ProgramBatch` implementation for solving QUBO problems using partitioning and QAOA, supporting `QUBOProblemTypes` and `BinaryQuadraticModel` inputs with configurable decomposer and composer
* `is_valid_qasm()` function: lightweight QASM validation function in `divi/exp/cirq/_validator.py` (645 lines) that returns boolean indicating QASM syntax validity using lexer and parser implementation
* QASM validation integration in `QoroService.submit_circuits()`: added automatic QASM validation check that raises `ValueError` for invalid QASM strings before submission
* Comprehensive test suite for QUBO partitioning: 218 lines of tests in `test_qubo_partitioning.py` covering `QUBOPartitioningQAOA` functionality
* Enhanced graph partitioning tests: expanded test coverage in `test_graph_partitioning.py` with 230 additional lines
* Extended VQE sweep tests: added comprehensive tests for new VQE input formats (Molecule objects and Hamiltonian inputs) in `test_vqe_sweep.py` with 473 additional lines
* `_task_fn` attribute in `ProgramBatch`: added configurable task function support for custom program execution workflows
* `reset()` method validation in `ProgramBatch.create_programs()`: added check to prevent creating programs when dictionary is not empty, requiring explicit reset

### 🔄 Changed

* Updated VQE tests: modified `test_vqe.py` (163 additional lines) to work with new VQE API accepting `Molecule` objects and `Hamiltonian` inputs instead of symbols/coords/bond_length
* Improved `ProgramBatch` cleanup logic: enhanced `__del__` method with proper manager shutdown, better queue cleanup, and improved progress bar termination handling
* Refactored `linear_aggregation()` function in graph partitioning: improved type hints and function signature for better clarity and type safety
* Enhanced error handling in `QoroService`: improved error messages and exception handling, particularly in `submit_circuits()` method
* Updated graph partitioning implementation: improved `_node_partition_graph()` function and related partitioning logic

### 🐛 Fixed

* Fixed QoroService endpoint: corrected API endpoint from `/qpu_systems/` to `/qpusystem/` in `qpu_systems()` method
* Fixed QPU system name property: corrected property setter logic to properly handle `None` values and prevent infinite recursion when setting system name
* Fixed system name retrieval: improved logic to use system name from root QPUSystem object when available
* Fixed potential issues in QoroService: corrected conditional logic for QPU system name handling in `submit_circuits()` method

## [0.2.2b1] - 2025-08-08

### ✨ Added

* COBYLA optimizer implementation: added `fmin_cobyla()` function in `divi/exp/scipy/_cobyla.py` (342 lines) using pure-Python PRIMA implementation for constrained optimization, integrated into optimizer enum as `Optimizer.COBYLA`
* `use_circuit_packing` parameter to `QoroService` constructor: added boolean flag to enable circuit packing functionality for more efficient job submission
* `draw_partitions()` method in `GraphPartitioningQAOA`: added visualization function to draw graph partitions with nodes colored by partition assignment using matplotlib
* Maximum size constraint for subgraph partitioning: added `max_n_nodes_per_cluster` parameter to `PartitioningConfig` dataclass to limit subgraph size in `_node_partition_graph()` function
* Support for pymetis and Kernighan-Lin partitioning methods: added support for additional graph partitioning algorithms beyond spectral clustering
* Apache 2.0 license headers across the codebase: added comprehensive license headers to all source files with copyright notice and full Apache 2.0 license text
* Pre-commit hook for license header insertion: added automated license header insertion to pre-commit configuration for consistent license management
* `charge` parameter to VQE constructor: added optional charge argument for molecular calculations (defaults to 0)
* Improved job naming for partitioning: enhanced job name generation in graph partitioning workflows for better identification
* Progress bar behavior improvements for Jupyter: added `is_jupyter` parameter to `make_progress_bar()` function with manual refresh mode and improved terminal size handling in `OverwriteStreamHandler` for Jupyter notebook compatibility
* Python-dotenv support: added `python-dotenv` dependency and support for loading environment variables from `.env` files, including `.env` in `.gitignore`

### 🔄 Changed

* Renamed `Optimizers` enum to `Optimizer`: refactored enum name for consistency across codebase
* Added `use_packing` parameter to `QoroService.submit_circuits()` method: method-level parameter to override constructor-level circuit packing setting
* Refactored partitioning method parameter handling: improved parameter validation and error messages for unsupported partitioning methods
* Improved partitioning logic: refactored `_node_partition_graph()` and `perform_partitioning()` functions to return list of subgraphs instead of partition labels, added `partition_graph_with_max_nodes()` function for recursive partitioning with size constraints, and improved edge count validation

### 🐛 Fixed

* Fixed progress bar behavior in Jupyter notebooks: corrected terminal size detection and line clearing logic in `OverwriteStreamHandler` to work properly in Jupyter environments
* Fixed logger message formatting: corrected string slicing in `OverwriteStreamHandler.emit()` to properly handle message overwriting (changed from `[2:-1]` to `[2:-2]`)
* Fixed partitioning logic: corrected edge counting and subgraph validation in `_node_partition_graph()` to properly handle empty subgraphs and edge preservation

## [0.2.1b1] - 2025-08-07

### ✨ Added

* Circuit cutting as a service: added `CIRCUIT_CUT` job type to `QoroService` with validation to ensure only one circuit is submitted for circuit-cutting jobs
* Noisy simulation support in `ParallelSimulator`: added `qiskit_backend` parameter (supports "auto" for automatic backend selection or specific backend names) and `noise_model` parameter for simulating noisy quantum circuits using Qiskit Aer
* Zero Noise Extrapolation (ZNE) implementation: added `ZNE` class in `divi/qem.py` (187 lines) using Mitiq library with `modify_circuit()` method for circuit folding and `postprocess_results()` method for extrapolation, supporting configurable scale factors and extrapolation factories
* QASM import/export functionality for QEM support: added `cirq_circuit_from_qasm()` and `cirq_circuit_to_qasm()` functions in `divi/exp/cirq/` module for converting between Cirq circuits and QASM strings, enabling QEM protocols like ZNE to work with circuit representations
* Cirq QASM parser and lexer for QEM support: added `_lexer.py` (126 lines) and `_parser.py` (889 lines) modules with PLY-based lexer and parser implementation for parsing QASM 2.0 syntax, including support for parameterized gates with `input` and `angle` keywords, required for QEM circuit manipulation
* Progress bars with polling info: added `_pbar.py` module (73 lines) using Rich library with `make_progress_bar()` function, `ConditionalSpinnerColumn`, and `PhaseStatusColumn` classes that display job progress, polling attempts, and final status
* Enhanced logging module (`qlogger`): improved `OverwriteStreamHandler` class (85 lines) with message overwriting support using `\c` prefix for updating progress messages on the same line, and better terminal size handling
* Circuit folding with modified Cirq conversion code: enhanced QASM export to properly handle parameterized circuits by modifying Cirq's QASM conversion to preserve Sympy symbols for parameters
* `CircuitRunner` abstract base class: added `divi/interfaces.py` (21 lines) with abstract `submit_circuits()` method for abstracting circuit execution, implemented by `ParallelSimulator` and `QoroService`
* Reporter infrastructure for progress tracking: added progress reporting system in `ProgramBatch` and `QuantumProgram` classes that integrates with progress bars and logging for real-time status updates

### 🔄 Changed

* Refactored `ParallelSimulator` to implement `CircuitRunner` interface: renamed `simulate()` method to `submit_circuits()` and made class inherit from `CircuitRunner` for consistent API
* Updated `QoroService` to implement `CircuitRunner` interface: renamed `send_circuits()` method to `submit_circuits()` and made class inherit from `CircuitRunner` for consistent API
* Enhanced circuit generation and parameter handling: improved `MetaCircuit` class to properly handle parameterized circuits with `measure_all=True` and symbol preservation in QASM export
* Improved error handling and logging: enhanced error messages and logging throughout codebase, particularly in circuit execution and progress reporting
* Updated tutorial examples: modified grouping tutorial to work with new API changes

### 🐛 Fixed

* Fixed instance variable conflict in `QuantumProgram`: resolved variable naming conflicts that were causing issues in program execution
* Fixed issues with standalone Sympy symbols: corrected QASM export to properly handle Sympy symbols in parameterized circuits, preventing Pennylane errors from unresolved symbols
* Fixed bug with postprocessing function in `ProgramBatch`: corrected postprocessing logic to properly handle results from batch execution

## [0.1.0-beta.1] - 2025-06-18

### ✨ Added

* Initial release of Divi quantum programming library
* `QuantumProgram` abstract base class: core abstraction for managing quantum computations with abstract methods for circuit generation, parameter initialization, and result processing
* `VQE` (Variational Quantum Eigensolver) class: implementation supporting multiple ansatze (UCCSD, RY, RYRZ, Hardware Efficient, QAOA, Hartree-Fock) with molecular chemistry support via symbols, bond lengths, and coordinate structures
* `QAOA` (Quantum Approximate Optimization Algorithm) class: implementation for solving combinatorial optimization problems on graphs
* `GraphPartitioningQAOA` class: `ProgramBatch` implementation for solving large graph problems by partitioning into smaller subproblems using QAOA
* `ProgramBatch` abstract base class: framework for higher-order computations requiring multiple quantum programs, with abstract `create_programs()` method
* `VQEHyperparameterSweep` class: functionality for performing hyperparameter sweeps on VQE problems with molecular transformations
* `MLAE` (Maximum Likelihood Amplitude Estimation) class: implementation for amplitude estimation (later removed in v0.3.2b0)
* `Optimizers` enum: support for Nelder-Mead, Monte Carlo, and L-BFGS-B optimization methods with parameter shift rule for gradient-based optimization
* `QoroService` class: cloud-based quantum execution backend via REST API with job submission, polling, and result retrieval
* `ParallelSimulator` class: local quantum circuit simulator using Qiskit Aer with parallel execution support
* Observable grouping: efficient expectation value calculation for VQE using Pennylane's `split_non_commuting` transform to group commuting observables
* QUBO and Quadratic Program support: integration with Qiskit Optimization for solving Quadratic Unconstrained Binary Optimization problems and general Quadratic Programs via `QuadraticProgramToQubo` converter
* QASM circuit generation: basic QASM string generation from quantum circuits (import/export functionality added in v0.2.1b1)
* Seeding support: reproducible simulations via `seed` parameter in `QuantumProgram` and `ParallelSimulator`
* Progress reporting: basic progress tracking infrastructure integrated into `QuantumProgram` and `ProgramBatch` classes
* Logging infrastructure: centralized logging module (`qlogger`) with `OverwriteStreamHandler` for progress message updates
* Test suite: comprehensive pytest-based test coverage for core classes including `QuantumProgram`, `VQE`, `QAOA`, `GraphPartitioningQAOA`, `ProgramBatch`, and backend classes
* Tutorial examples: demonstration scripts for VQE, QAOA, and graph partitioning workflows
* Documentation: Sphinx-based documentation framework with API reference generation
* Development tooling: Poetry for dependency management and package building, pre-commit hooks with black and isort for code formatting

### 🐛 Fixed

* Fixed measurement phase bug: corrected measurement handling in quantum circuit execution
* Fixed Monte Carlo missing losses bug: resolved issue where losses were not properly tracked in Monte Carlo optimization
* Fixed QAOA partitioning bugs: corrected graph partitioning logic for QAOA-based workflows
* Fixed circuit chunking for large payloads: improved handling of large circuit submissions to QoroService API

[Unreleased]: https://github.com/QoroQuantum/divi/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/QoroQuantum/divi/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/QoroQuantum/divi/compare/v0.3.5...v0.4.0
[0.3.5]: https://github.com/QoroQuantum/divi/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/QoroQuantum/divi/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/QoroQuantum/divi/compare/v0.3.2b0...v0.3.3
[0.3.2b0]: https://github.com/QoroQuantum/divi/compare/v0.3.1b0...v0.3.2b0
[0.3.1b0]: https://github.com/QoroQuantum/divi/compare/v0.2.2b1...v0.3.1b0
[0.2.2b1]: https://github.com/QoroQuantum/divi/compare/v0.2.1b1...v0.2.2b1
[0.2.1b1]: https://github.com/QoroQuantum/divi/compare/v0.1.0-beta.1...v0.2.1b1
[0.1.0-beta.1]: https://github.com/QoroQuantum/divi/releases/tag/v0.1.0-beta.1
