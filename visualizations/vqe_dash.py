# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback_context, dcc, html
from dash.exceptions import PreventUpdate
from qiskit_ibm_runtime.fake_provider import FakeQuitoV2

from divi.backends import ParallelSimulator
from divi.qprog import (
    HartreeFockAnsatz,
    ScipyMethod,
    ScipyOptimizer,
    VQEHyperparameterSweep,
)

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    html.Div(
        [
            html.H2("VQE Results"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Button(
                                "Start Noiseless VQE", id="noiseless-button", n_clicks=0
                            ),
                            html.Button(
                                "Start Noisy VQE", id="noisy-button", n_clicks=0
                            ),
                            html.Button(
                                "Start Noisy+ZNE VQE", id="zne-button", n_clicks=0
                            ),
                        ],
                        style={
                            "display": "flex",
                            "flexDirection": "row",
                            "justifyContent": "center",
                            "alignItems": "center",
                            "gap": "16px",
                        },
                    ),
                    dcc.Loading(
                        id="loading-1",
                        type="default",
                        children=html.Div(id="loading-output-1"),
                    ),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "justifyContent": "center",
                    "alignItems": "center",
                    "gap": "16px",
                    "marginTop": "24px",
                },
            ),
            html.Div(
                [
                    dcc.Graph(id="energy-graph", figure={}),
                    dcc.Graph(id="iterations", figure={}),
                    dcc.Graph(id="run_time", figure={}),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "row",
                    "flexWrap": "wrap",
                    "justifyContent": "center",
                    "alignItems": "stretch",
                    "width": "100%",
                    "gap": "24px",
                },
            ),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",
            "justifyContent": "center",
            "minHeight": "100vh",
        },
    )
)

BOND_LENGTHS = np.linspace(0.1, 2.7, 4)
OPTIMIZER = ScipyOptimizer(method=ScipyMethod.NELDER_MEAD)
MAX_ITERATIONS = 5

# backend = QoroService("71ec99c9c94cf37499a2b725244beac1f51b8ee4", shots=500)
exact_backend = ParallelSimulator(shots=500)
noisy_backend = ParallelSimulator(shots=500, qiskit_backend="auto")

args = dict(
    symbols=["H", "H"],
    bond_lengths=BOND_LENGTHS,
    coordinate_structure=[(0, 0, -0.5), (0, 0, 0.5)],
    ansatze=[HartreeFockAnsatz()],
    optimizer=OPTIMIZER,
    max_iterations=MAX_ITERATIONS,
)

noiseless_vqe_problem = VQEHyperparameterSweep(backend=exact_backend, **args)
noisy_vqe_problem = VQEHyperparameterSweep(backend=noisy_backend, **args)
zne_vqe_problem = VQEHyperparameterSweep(
    backend=noisy_backend,
    **args,
    #   zne=True,
)


@app.callback(
    [
        Output(
            component_id="energy-graph",
            component_property="figure",
            allow_duplicate=True,
        ),
        Output(
            component_id="iterations", component_property="figure", allow_duplicate=True
        ),
        Output(
            component_id="run_time", component_property="figure", allow_duplicate=True
        ),
        Output("loading-1", "children", allow_duplicate=True),
    ],
    inputs=[
        Input("noiseless-button", "n_clicks"),
        Input("noisy-button", "n_clicks"),
        Input("zne-button", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def run_vqe(noiseless_clicks, noisy_clicks, zne_clicks):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    mode = ctx.triggered_id.split("-")[0]
    problem = globals()[f"{mode}_vqe_problem"]

    energy_v_bond_fig = go.Figure()
    energy_v_iteration_fig = go.Figure()
    run_time_fig = go.Figure()

    problem.create_programs()
    problem.run(blocking=True)

    for ansatz in problem.ansatze:
        ys = []
        for bond_length in problem.bond_lengths:
            ys.append(problem.programs[(ansatz, bond_length)].losses[-1][0])
        energy_v_bond_fig.add_trace(
            go.Scatter(
                x=problem.bond_lengths,
                y=ys,
                mode="lines+markers",
                name=ansatz.name,
                line=dict(color="blue"),
            )
        )
    energy_v_bond_fig.update_layout(
        title="Energy vs Bond Length",
        xaxis_title="Bond Length",
        yaxis_title="Energy",
        showlegend=True,
    )
    print("Finished plotting Energy vs. Bond")

    ansatz = problem.ansatze[0]
    for bond_length in problem.bond_lengths:
        losses = problem.programs[(ansatz, bond_length)].losses
        energy_v_iteration_fig.add_trace(
            go.Scatter(
                x=list(range(1, len(losses) + 1)),
                y=list(map(lambda item: item[0], losses)),
                mode="lines+markers",
                name=f"{round(bond_length, 3)}",
            )
        )
    energy_v_iteration_fig.update_layout(
        title="Energy vs Iterations",
        xaxis_title="Iteration",
        yaxis_title="Energy",
        showlegend=True,
        legend_title_text="Bond Length (Å)",
    )
    print("Finished plotting Energy vs. Iteration")

    qasm_circuits = []
    for program in problem.programs.values():
        qasm_circuits.extend(
            program.meta_circuit_factories["cost_circuit"]
            .initialize_circuit_from_params(program.final_params[0])
            .qasm_circuits
        )

    durations = [
        ParallelSimulator.estimate_run_time_single_circuit(
            circuit, qiskit_backend=FakeQuitoV2()
        )
        for circuit in qasm_circuits
    ]
    qpu_range = tuple(range(1, 10))

    run_times = [
        ParallelSimulator.estimate_run_time_batch(
            precomputed_duration=durations, n_qpus=i
        )
        for i in qpu_range
    ]

    run_time_fig.add_trace(
        go.Scatter(
            x=qpu_range,
            y=run_times,
            mode="lines+markers",
            name="Runtimes",
            line=dict(color="blue"),
        )
    )

    run_time_fig.update_layout(
        title="Runtime per Iteration vs QPUs",
        xaxis_title="Num. of QPUs",
        yaxis_title="Runtime per Iteration",
        showlegend=True,
    )
    print("Finished plotting Runtime vs. QPUs")

    return (
        energy_v_bond_fig,
        energy_v_iteration_fig,
        run_time_fig,
        f"{mode.capitalize()} run complete.",
    )


# @app.callback(
#     [
#         Output(
#             component_id="energy-graph",
#             component_property="figure",
#             allow_duplicate=True,
#         ),
#         Output(
#             component_id="iterations", component_property="figure", allow_duplicate=True
#         ),
#         Output(
#             component_id="run_time", component_property="figure", allow_duplicate=True
#         ),
#         Output("loading-1", "children", allow_duplicate=True),
#     ],
#     Input("zne-button", "n_clicks"),
#     prevent_initial_call=True,
# )
# def run_vqe_zne(n_clicks):
#     energy_v_bond_fig = go.Figure()
#     energy_v_iteration_fig = go.Figure()
#     run_time_fig = go.Figure()

#     if n_clicks > 0:
#         zne_vqe_problem.run()
#         energies = zne_vqe_problem.losses[zne_vqe_problem.current_iteration - 1]
#         for ansatz in zne_vqe_problem.ansatze:
#             ys = []
#             for i in range(len(zne_vqe_problem.bond_lengths)):
#                 ys.append(energies[i][ansatz][0])
#             energy_v_bond_fig.add_trace(
#                 go.Scatter(
#                     x=zne_vqe_problem.bond_lengths,
#                     y=ys,
#                     mode="lines+markers",
#                     name=f"{ansatz.name}, ZNE",
#                     line=dict(color="red"),
#                 )
#             )
#         energy_v_bond_fig.update_layout(
#             title="Energy vs Bond Length",
#             xaxis_title="Bond Length",
#             yaxis_title="Energy",
#             showlegend=True,
#         )

#         data = []
#         ansatz = zne_vqe_problem.ansatze[0]
#         for energy in zne_vqe_problem.losses:
#             data.append(energy[0][ansatz][0])
#         energy_v_iteration_fig.add_trace(
#             go.Scatter(
#                 x=list(range(1, len(data) + 1)),
#                 y=data,
#                 mode="lines+markers",
#                 name=f"{ansatz.name}, ZNE",
#                 line=dict(color="red"),
#             )
#         )
#         energy_v_iteration_fig.update_layout(
#             title="Energy vs Iterations",
#             xaxis_title="Iteration",
#             yaxis_title="Energy",
#             showlegend=True,
#         )
#         qasm_circuits = [dumps(circuit) for circuit in zne_vqe_problem.zne_circuits]
#         qpu_list = [i for i in range(1, 10)]
#         simulators = [ParallelSimulator(n_processes=2, qpus=i) for i in qpu_list]
#         run_times = [
#             simulator.estimate_run_time_single_circuit(qasm_circuits)
#             for simulator in simulators
#         ]
#         run_time_fig.add_trace(
#             go.Scatter(
#                 x=qpu_list,
#                 y=run_times,
#                 mode="lines+markers",
#                 name="ZNE Runtimes",
#                 line=dict(color="red"),
#             )
#         )
#         run_time_fig.update_layout(
#             title="Runtime per Iteration vs QPUs",
#             xaxis_title="Num. of QPUs",
#             yaxis_title="Runtime per Iteration",
#             showlegend=True,
#         )

#         return energy_v_bond_fig, energy_v_iteration_fig, run_time_fig, "done"

#     energy_v_bond_fig.update_layout(
#         title="Energy vs Bond Length",
#         xaxis_title="Bond Length",
#         yaxis_title="Energy",
#         showlegend=True,
#     )
#     energy_v_iteration_fig.update_layout(
#         title="Energy vs Iterations",
#         xaxis_title="Iteration",
#         yaxis_title="Energy",
#         showlegend=True,
#     )
#     run_time_fig.update_layout(
#         title="Runtime per Iteration vs QPUs",
#         xaxis_title="Num. of QPUs",
#         yaxis_title="Runtime per Iteration",
#         showlegend=True,
#     )

#     return energy_v_bond_fig, energy_v_iteration_fig, run_time_fig, no_update


if __name__ == "__main__":
    app.run(debug=True)
