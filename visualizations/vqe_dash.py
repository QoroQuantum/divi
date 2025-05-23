from itertools import chain

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html, no_update
from qiskit.qasm2 import dumps

from divi.parallel_simulator import ParallelSimulator
from divi.qprog import Optimizers, VQEAnsatze, VQEHyperparameterSweep

app = Dash()
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div(
    html.Div(
        [
            html.H2("VQE Results"),
            html.H4("Energy vs Bond Length"),
            dcc.Graph(id="energy-graph", figure={}),
            dcc.Graph(id="iterations", figure={}),
            dcc.Graph(id="run_time", figure={}),
            dcc.Loading(
                id="loading-1", type="default", children=html.Div(id="loading-output-1")
            ),
            html.Button("Start VQE", id="start-button", n_clicks=0),
            html.Button("Start ZNE VQE", id="zne-button", n_clicks=0),
            html.Button("Start Noisy VQE", id="noisy-button", n_clicks=0),
            html.Div(id="state", children=None),
        ]
    )
)

# q_service = QoroService("71ec99c9c94cf37499a2b725244beac1f51b8ee4")
q_service = None
noiseless_vqe_problem = VQEHyperparameterSweep(
    symbols=["H", "H"],
    bond_lengths=list(np.linspace(0.1, 2.7, 5)),
    coordinate_structure=[(0, 0, -0.5), (0, 0, 0.5)],
    ansatze=[VQEAnsatze.HARTREE_FOCK],
    optimizer=Optimizers.NELDER_MEAD,
    shots=500,
    max_iterations=1,
    qoro_service=q_service,
)


zne_vqe_problem = VQEHyperparameterSweep(
    symbols=["H", "H"],
    bond_lengths=list(np.linspace(0.1, 2.7, 5)),
    coordinate_structure=[(0, 0, -0.5), (0, 0, 0.5)],
    ansatze=[VQEAnsatze.HARTREE_FOCK],
    optimizer=Optimizers.MONTE_CARLO,
    #   zne=True,
    #   noise=0.01,
    shots=500,
    max_iterations=5,
    qoro_service=q_service,
)

noisy_vqe_problem = VQEHyperparameterSweep(
    symbols=["H", "H"],
    bond_lengths=[0.5, 1.0, 1.5],
    coordinate_structure=[(0, 0, -0.5), (0, 0, 0.5)],
    ansatze=[VQEAnsatze.HARTREE_FOCK],
    optimizer=Optimizers.MONTE_CARLO,
    max_iterations=5,
    # noise=0.01,
    # shots=500,
    qoro_service=q_service,
)


@callback(
    Output(component_id="state", component_property="children"),
    Input("start-button", "n_clicks"),
)
def started(n_clicks):
    if n_clicks > 0:
        return "VQE Execution Started"
    return ""


@callback(
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
    Input("start-button", "n_clicks"),
    prevent_initial_call=True,
)
def run_vqe(n_clicks):
    energy_v_bond_fig = go.Figure()
    energy_v_iteration_fig = go.Figure()
    run_time_fig = go.Figure()

    if n_clicks > 0:
        noiseless_vqe_problem.create_programs()
        noiseless_vqe_problem.run()
        noiseless_vqe_problem.wait_for_all()

        for ansatz in noiseless_vqe_problem.ansatze:
            ys = []
            for bond_length in noiseless_vqe_problem.bond_lengths:
                ys.append(
                    noiseless_vqe_problem.programs[(ansatz, bond_length)].losses[-1][0]
                )
            energy_v_bond_fig.add_trace(
                go.Scatter(
                    x=noiseless_vqe_problem.bond_lengths,
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

        data = []
        ansatz = noiseless_vqe_problem.ansatze[0]
        bond_length = noiseless_vqe_problem.bond_lengths[0]
        for energy in noiseless_vqe_problem.programs[(ansatz, bond_length)].losses:
            data.append(energy[0])
        energy_v_iteration_fig.add_trace(
            go.Scatter(
                x=list(range(1, len(data) + 1)),
                y=data,
                mode="lines+markers",
                name="Hartree Fock",
                line=dict(color="blue"),
            )
        )
        energy_v_iteration_fig.update_layout(
            title="Energy vs Iterations",
            xaxis_title="Iteration",
            yaxis_title="Energy",
            showlegend=True,
        )

        qasm_circuits = []
        for circuit in chain.from_iterable(
            program.circuits for program in noiseless_vqe_problem.programs.values()
        ):
            qasm_circuits.extend(circuit.qasm_circuits)

        durations = [
            ParallelSimulator.estimate_run_time_single_circuit(circuit)
            for circuit in qasm_circuits
        ]
        qpu_range = tuple(range(1, 10))

        run_times = [
            ParallelSimulator(n_processes=2).estimate_run_time_batch(
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

        return energy_v_bond_fig, energy_v_iteration_fig, run_time_fig, "done"

    energy_v_bond_fig.update_layout(
        title="Energy vs Bond Length",
        xaxis_title="Bond Length",
        yaxis_title="Energy",
        showlegend=True,
    )
    energy_v_iteration_fig.update_layout(
        title="Energy vs Iterations",
        xaxis_title="Iteration",
        yaxis_title="Energy",
        showlegend=True,
    )
    run_time_fig.update_layout(
        title="Runtime per Iteration vs QPUs",
        xaxis_title="Num. of QPUs",
        yaxis_title="Runtime per Iteration",
        showlegend=True,
    )

    return energy_v_bond_fig, energy_v_iteration_fig, run_time_fig, no_update


@callback(
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
    Input("zne-button", "n_clicks"),
    Input("energy-graph", "figure"),
    Input("iterations", "figure"),
    Input("run_time", "figure"),
    prevent_initial_call=True,
)
def run_zne_vqe(n_clicks, existing_figure1, existing_figure2, existing_figure3):
    if existing_figure1 is not None:
        fig = go.Figure(existing_figure1)
    else:
        fig = go.Figure()
    if existing_figure2 is not None:
        fig2 = go.Figure(existing_figure2)
    else:
        fig2 = go.Figure()
    if existing_figure3 is not None:
        fig3 = go.Figure(existing_figure3)
    else:
        fig3 = go.Figure()
    if n_clicks > 0:
        zne_vqe_problem.run()
        energies = zne_vqe_problem.losses[zne_vqe_problem.current_iteration - 1]
        for ansatz in zne_vqe_problem.ansatze:
            ys = []
            for i in range(len(zne_vqe_problem.bond_lengths)):
                ys.append(energies[i][ansatz][0])
            fig.add_trace(
                go.Scatter(
                    x=zne_vqe_problem.bond_lengths,
                    y=ys,
                    mode="lines+markers",
                    name=f"{ansatz.name}, ZNE",
                    line=dict(color="red"),
                )
            )
        fig.update_layout(
            title="Energy vs Bond Length",
            xaxis_title="Bond Length",
            yaxis_title="Energy",
            showlegend=True,
        )

        data = []
        ansatz = zne_vqe_problem.ansatze[0]
        for energy in zne_vqe_problem.losses:
            data.append(energy[0][ansatz][0])
        fig2.add_trace(
            go.Scatter(
                x=list(range(1, len(data) + 1)),
                y=data,
                mode="lines+markers",
                name=f"{ansatz.name}, ZNE",
                line=dict(color="red"),
            )
        )
        fig2.update_layout(
            title="Energy vs Iterations",
            xaxis_title="Iteration",
            yaxis_title="Energy",
            showlegend=True,
        )
        qasm_circuits = [dumps(circuit) for circuit in zne_vqe_problem.zne_circuits]
        qpu_list = [i for i in range(1, 10)]
        simulators = [ParallelSimulator(n_processes=2, qpus=i) for i in qpu_list]
        run_times = [
            simulator.estimate_run_time_single_circuit(qasm_circuits)
            for simulator in simulators
        ]
        fig3.add_trace(
            go.Scatter(
                x=qpu_list,
                y=run_times,
                mode="lines+markers",
                name="ZNE Runtimes",
                line=dict(color="red"),
            )
        )
        fig3.update_layout(
            title="Runtime per Iteration vs QPUs",
            xaxis_title="Num. of QPUs",
            yaxis_title="Runtime per Iteration",
            showlegend=True,
        )

        return fig, fig2, fig3, "done"
    fig.update_layout(
        title="Energy vs Bond Length",
        xaxis_title="Bond Length",
        yaxis_title="Energy",
        showlegend=True,
    )
    fig2.update_layout(
        title="Energy vs Iterations",
        xaxis_title="Iteration",
        yaxis_title="Energy",
        showlegend=True,
    )
    fig3.update_layout(
        title="Runtime per Iteration vs QPUs",
        xaxis_title="Num. of QPUs",
        yaxis_title="Runtime per Iteration",
        showlegend=True,
    )

    return fig, fig2, fig3, no_update


@callback(
    [
        Output(
            component_id="energy-graph",
            component_property="figure",
            allow_duplicate=True,
        ),
        Output(
            component_id="iterations", component_property="figure", allow_duplicate=True
        ),
        Output("loading-1", "children", allow_duplicate=True),
    ],
    Input("noisy-button", "n_clicks"),
    Input("energy-graph", "figure"),
    Input("iterations", "figure"),
    prevent_initial_call=True,
)
def run_noisy_vqe(n_clicks, existing_figure1, existing_figure2):
    if existing_figure1 is not None:
        fig = go.Figure(existing_figure1)
    else:
        fig = go.Figure()
    if existing_figure2 is not None:
        fig2 = go.Figure(existing_figure2)
    else:
        fig2 = go.Figure()
    if n_clicks > 0:
        noisy_vqe_problem.run()
        energies = noisy_vqe_problem.losses[noisy_vqe_problem.current_iteration - 1]
        for ansatz in noisy_vqe_problem.ansatze:
            ys = []
            for i in range(len(noisy_vqe_problem.bond_lengths)):
                ys.append(energies[i][ansatz][0])
            fig.add_trace(
                go.Scatter(
                    x=noisy_vqe_problem.bond_lengths,
                    y=ys,
                    mode="lines+markers",
                    name=f"{ansatz.name}, NOISY",
                    line=dict(color="green"),
                )
            )
        fig.update_layout(
            title="Energy vs Bond Length",
            xaxis_title="Bond Length",
            yaxis_title="Energy",
            showlegend=True,
        )

        data = []
        ansatz = noisy_vqe_problem.ansatze[0]
        for energy in noisy_vqe_problem.losses:
            data.append(energy[0][ansatz][0])
        fig2.add_trace(
            go.Scatter(
                x=list(range(1, len(data) + 1)),
                y=data,
                mode="lines+markers",
                name=f"{ansatz.name}, NOISY",
                line=dict(color="green"),
            )
        )
        fig2.update_layout(
            title="Energy vs Iterations",
            xaxis_title="Iteration",
            yaxis_title="Energy",
            showlegend=True,
        )
        return fig, fig2, "done"
    fig.update_layout(
        title="Energy vs Bond Length",
        xaxis_title="Bond Length",
        yaxis_title="Energy",
        showlegend=True,
    )
    fig2.update_layout(
        title="Energy vs Iterations",
        xaxis_title="Iteration",
        yaxis_title="Energy",
        showlegend=True,
    )
    return fig, fig2, no_update


if __name__ == "__main__":
    app.run(debug=True)
