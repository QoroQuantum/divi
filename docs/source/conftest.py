# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Sybil config for testing RST code blocks.

Replaces :class:`~divi.backends.QoroService` with a local stub (no network),
configures matplotlib for headless rendering, and injects a few shared objects
into the snippet namespace.

Run: ``cd docs && make test-snippets``
"""

import os
import shutil
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import matplotlib

# Headless backend must be selected before pyplot is imported.
os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 — after matplotlib.use
import numpy as np
import pennylane as qml
import pytest
from sybil import Sybil
from sybil.parsers.rest import PythonCodeBlockParser, SkipParser
from sybil.region import Lexeme

from divi.backends import (
    CircuitRunner,
    ExecutionConfig,
    ExecutionResult,
    JobConfig,
    JobStatus,
    MaestroSimulator,
    SimulatorCluster,
)
from divi.qprog import SuperpositionState

# ---------------------------------------------------------------------------
# Runtime patching for snippet tests
# ---------------------------------------------------------------------------
# Docs show realistic, instructive parameter values (max_iterations=25, shots=
# 10000, 5 bond modifiers, etc.).  Running all of that at full cost in CI
# makes the snippet suite take many minutes with per-test pauses that look
# like hangs.  Mirror the tutorials/_ci_runner.py approach: rewrite the
# snippet source text at exec time to reduce iteration / shot / population
# budgets before running.  The rendered HTML still shows the original values.
SNIPPET_PATCHES: list[tuple[str, str]] = [
    # User-guide snippets use standardized parameter values so the patch
    # list stays small.  If you add a new snippet, match these defaults
    # rather than inventing new values.
    #   - ``shots=5000``                    → ``shots=500``
    #   - ``max_iterations=10``             → ``max_iterations=3``
    #   - ``max_iterations_per_depth=10``   → ``max_iterations_per_depth=3``
    ("shots=5000", "shots=500"),
    ("max_iterations=10", "max_iterations=3"),
    ("max_iterations_per_depth=10", "max_iterations_per_depth=3"),
    # ``optimizers.rst`` deliberately uses ``max_iterations=200`` to show
    # early-stopping; patch so CI doesn't spend the full budget if the
    # early-stopping criterion fails to trigger.
    ("max_iterations=200", "max_iterations=10"),
    # VQEHyperparameterSweep — 5 bonds x 2 ansatze collapsed to 2 bonds x 1 ansatz.
    (
        "bond_modifiers=[-0.4, -0.25, 0, 0.25, 0.4]",
        "bond_modifiers=[0, 0.25]",
    ),
    (
        "ansatze=[HartreeFockAnsatz(), UCCSDAnsatz()]",
        "ansatze=[HartreeFockAnsatz()]",
    ),
    # IterativeQAOA depth — 5 QAOA depths → 2.
    ("max_depth=5", "max_depth=2"),
    # MonteCarloOptimizer population.
    ("population_size=10", "population_size=3"),
    # ZNE: fewer scale factors → fewer circuits per iteration.
    ("scale_factors = [1.0, 1.5, 2.0]", "scale_factors = [1.0, 1.5]"),
    # Circuit depth.
    ("n_layers=2", "n_layers=1"),
]


class PatchingPythonCodeBlockParser(PythonCodeBlockParser):
    """Sybil code-block parser that rewrites snippet source before execution.

    Applies :data:`SNIPPET_PATCHES` to each region's ``parsed`` lexeme.  The
    rendered HTML is unaffected — only the source that Sybil compiles and
    executes is shrunk, so CI finishes quickly while docs stay instructive.
    """

    def __init__(self, patches, future_imports=()):
        super().__init__(future_imports=future_imports)
        self._patches = patches

    def __call__(self, document):
        for region in super().__call__(document):
            lex = region.parsed
            text = str(lex)
            for needle, replacement in self._patches:
                text = text.replace(needle, replacement)
            if text != str(lex):
                region.parsed = Lexeme(text, lex.offset, lex.line_offset)
            yield region


_QASM_STRING = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
    "h q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];"
)


class DocStubQoroService(CircuitRunner):
    """Stand-in for :class:`~divi.backends.QoroService` — no network, deterministic."""

    def __init__(
        self,
        auth_token=None,
        job_config=None,
        execution_config=None,
        polling_interval=3.0,
        max_retries=5000,
        track_depth=False,
    ):
        jc = job_config or JobConfig(shots=1000)
        if jc.simulator_cluster is None and jc.qpu_system is None:
            jc = replace(jc, simulator_cluster=SimulatorCluster(name="qoro_maestro"))
        super().__init__(shots=jc.shots, track_depth=track_depth)
        self._job_config = jc
        self._execution_config = execution_config
        self._pending_circuits: dict[str, str] | None = None
        self._last_ham_ops: str | None = None

    @property
    def job_config(self) -> JobConfig:
        return self._job_config

    @job_config.setter
    def job_config(self, value: JobConfig) -> None:
        self._job_config = value

    @property
    def execution_config(self) -> ExecutionConfig | None:
        return self._execution_config

    @execution_config.setter
    def execution_config(self, value: ExecutionConfig | None) -> None:
        self._execution_config = value

    @property
    def supports_expval(self) -> bool:
        return False

    @property
    def is_async(self) -> bool:
        return True

    def submit_circuits(
        self,
        circuits: Mapping[str, str],
        ham_ops=None,
        circuit_ham_map=None,
        job_type=None,
        override_execution_config=None,
        override_job_config=None,
        **kwargs,
    ) -> ExecutionResult:
        self._pending_circuits = dict(circuits)
        self._last_ham_ops = ham_ops
        return ExecutionResult(job_id="doc-sybil-qoro-job")

    def poll_job_status(self, execution_result, **kwargs):
        return JobStatus.COMPLETED

    def get_job_results(self, execution_result):
        circuits = self._pending_circuits or {}
        if self._last_ham_ops:
            rows = [
                {"label": k, "results": {"XYZ": 0.5, "XXZ": -0.3, "ZIZ": 1.0}}
                for k in circuits
            ]
        else:
            rows = [{"label": k, "results": {"0011": 2000}} for k in circuits]
        return execution_result.with_results(rows)

    def cancel_job(self, execution_result):
        class _R:
            status_code = 200

            def json(self):
                return {"status": "CANCELLED"}

        return _R()

    def get_execution_config(self, execution_result):
        return ExecutionConfig(bond_dimension=512)

    def set_execution_config(self, execution_result, config):
        return {"status": "ok", "job_id": execution_result.job_id}


def setup(namespace):
    """Called once before each .rst document's code blocks are executed."""
    os.environ["MPLBACKEND"] = "Agg"

    _original_show = plt.show
    plt.show = lambda *_a, **_kw: None
    namespace["_original_plt_show"] = _original_show

    namespace["SuperpositionState"] = SuperpositionState
    namespace["MaestroSimulator"] = MaestroSimulator
    namespace["qasm_string"] = _QASM_STRING
    namespace["qasm_string_1"] = _QASM_STRING
    namespace["qasm_string_2"] = _QASM_STRING
    namespace["circuits"] = {"c0": _QASM_STRING}
    namespace["molecule"] = qml.qchem.Molecule(
        symbols=["H", "H"],
        coordinates=np.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]]),
    )
    namespace["backend"] = MaestroSimulator()


def teardown(namespace):
    """Called once after each .rst document's code blocks finish."""
    if "_original_plt_show" in namespace:
        plt.show = namespace["_original_plt_show"]
    plt.close("all")

    for name in ("my_checkpoints", "qaoa_checkpoints"):
        p = Path(name)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    for p in Path(".").glob("checkpoint_*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


@pytest.fixture(scope="session")
def _doc_snippet_qoro_stub(session_mocker):
    """Replace ``QoroService`` so RST snippets never hit the network."""
    session_mocker.patch("divi.backends.QoroService", DocStubQoroService)
    session_mocker.patch("divi.backends._qoro_service.QoroService", DocStubQoroService)


pytest_collect_file = Sybil(
    parsers=[PatchingPythonCodeBlockParser(SNIPPET_PATCHES), SkipParser()],
    patterns=["*.rst"],
    setup=setup,
    teardown=teardown,
    fixtures=("_doc_snippet_qoro_stub",),
).pytest()
