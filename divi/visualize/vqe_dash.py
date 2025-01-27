from divi.qprog import VQE, Ansatze, Optimizers
from qiskit.qasm2 import dumps
from dash import Dash, html, dcc, Input, Output, callback, no_update
from divi.simulator.parallel_simulator import ParallelSimulator
from divi.services.qoro_service import QoroService

import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash
from dash import Input
from dash import Output
from dash import callback
from dash import dcc
from dash import html
from dash import no_update
from qiskit.qasm2 import dumps

from divi.qprog import VQE, Optimizers, VQEAnsatze

app = Dash()
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div(
    html.Div([html.H2("VQE Results"),
              html.H4("Energy vs Bond Length"),
              dcc.Graph(id="energy-graph", figure={}),
              dcc.Graph(id="iterations", figure={}),
              dcc.Graph(id="runtime", figure={}),
              dcc.Loading(id="loading-1", type="default",
                          children=html.Div(id="loading-output-1")),
              html.Button('Start VQE', id='start-button', n_clicks=0),
              html.Button('Start ZNE VQE', id='zne-button', n_clicks=0),
              html.Button('Start Noisy VQE', id='noisy-button', n_clicks=0),
              html.Div(id='state', children=None)
              ])
)

# q_service = QoroService("71ec99c9c94cf37499a2b725244beac1f51b8ee4")
q_service = None
vqe_problem = VQE(symbols=["H", "H"],
                  bond_lengths=[0.5, 1.0, 1.5],
                  coordinate_structure=[(0, 0, -0.5), (0, 0, 0.5)],
                  ansatze=[Ansatze.HARTREE_FOCK],
                  optimizer=Optimizers.NELDER_MEAD,
                  qoro_service=q_service,
                  shots=500,
                  max_interations=1)


zne_vqe_problem = VQE(symbols=["H", "H"],
                      bond_lengths=[0.5, 1.0, 1.5],
                      coordinate_structure=[(0, 0, -0.5), (0, 0, 0.5)],
                      ansatze=[Ansatze.HARTREE_FOCK],
                      optimizer=Optimizers.MONTE_CARLO,
                      qoro_service=q_service,
                      #   zne=True,
                      #   noise=0.01,
                      shots=500,
                      max_interations=5)

noisy_vqe_problem = VQE(symbols=["H", "H"],
                        bond_lengths=[0.5, 1.0, 1.5],
                        coordinate_structure=[(0, 0, -0.5), (0, 0, 0.5)],
                        ansatze=[Ansatze.HARTREE_FOCK],
                        optimizer=Optimizers.MONTE_CARLO,
                        qoro_service=q_service,
                        # noise=0.01,
                        # shots=500,
                        max_interations=5)


@callback(
    Output(component_id="state", component_property="children"),
    Input("start-button", "n_clicks"),
)
def started(n_clicks):
    if n_clicks > 0:
        return "VQE Execution Started"
    return ""


@callback(
    [Output(component_id="energy-graph", component_property="figure", allow_duplicate=True),
     Output(component_id="iterations",
            component_property="figure", allow_duplicate=True),
     Output(component_id="runtime",
            component_property="figure", allow_duplicate=True),
     Output("loading-1", "children", allow_duplicate=True),],
    Input('start-button', 'n_clicks'),
    prevent_initial_call=True,
)
def run_vqe(n_clicks):
    fig = go.Figure()
    fig2 = go.Figure()
    fig3 = go.Figure()
    if n_clicks > 0:
        vqe_problem.run()
        energies = vqe_problem.energies[vqe_problem.current_iteration - 1]
        for ansatz in vqe_problem.ansatze:
            ys = []
            for i in range(len(vqe_problem.bond_lengths)):
                ys.append(energies[i][ansatz][0])
            fig.add_trace(go.Scatter(x=vqe_problem.bond_lengths,
                          y=ys, mode='lines+markers', name=ansatz.name, line=dict(color='blue')))
        fig.update_layout(title="Energy vs Bond Length",
                          xaxis_title="Bond Length", yaxis_title="Energy", showlegend=True)

        data = []
        ansatz = vqe_problem.ansatze[0]
        for energy in vqe_problem.energies:
            data.append(energy[0][ansatz][0])
        fig2.add_trace(go.Scatter(x=list(range(1, len(data) + 1)),
                                  y=data, mode='lines+markers', name="Hartree Fock", line=dict(color='blue')))
        fig2.update_layout(title="Energy vs Iterations",
                           xaxis_title="Iteration", yaxis_title="Energy", showlegend=True)
        qasm_circuits = []
        for circuits in vqe_problem.circuits.values():
            qasm_circuits += [circuit.qasm_circuit for circuit in circuits]
        qpu_list = [i for i in range(1, 10)]
        simulators = [ParallelSimulator(
            num_processes=2, qpus=i) for i in qpu_list]
        runtimes = [simulator.runtime_estimate(
            qasm_circuits) for simulator in simulators]
        fig3.add_trace(go.Scatter(x=qpu_list, y=runtimes, mode='lines+markers', name='Runtimes',
                                  line=dict(color='blue')))
        fig3.update_layout(title="Runtime per Iteration vs QPUs", xaxis_title="Num. of QPUs",
                           yaxis_title="Runtime per Iteration", showlegend=True)

        return fig, fig2, fig3, "done"
    fig.update_layout(title="Energy vs Bond Length",
                      xaxis_title="Bond Length", yaxis_title="Energy", showlegend=True)
    fig2.update_layout(title="Energy vs Iterations",
                       xaxis_title="Iteration", yaxis_title="Energy", showlegend=True)
    fig3.update_layout(title="Runtime per Iteration vs QPUs",
                       xaxis_title="Num. of QPUs", yaxis_title="Runtime per Iteration", showlegend=True)

    return fig, fig2, fig3, no_update


@callback(
    [Output(component_id="energy-graph", component_property="figure", allow_duplicate=True),
     Output(component_id="iterations",
            component_property="figure", allow_duplicate=True),
     Output(component_id="runtime",
            component_property="figure", allow_duplicate=True),
     Output("loading-1", "children", allow_duplicate=True),],
    Input('zne-button', 'n_clicks'),
    Input('energy-graph', 'figure'),
    Input('iterations', 'figure'),
    Input('runtime', 'figure'),
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
        energies = zne_vqe_problem.energies[zne_vqe_problem.current_iteration - 1]
        for ansatz in zne_vqe_problem.ansatze:
            ys = []
            for i in range(len(zne_vqe_problem.bond_lengths)):
                ys.append(energies[i][ansatz][0])
            fig.add_trace(go.Scatter(x=zne_vqe_problem.bond_lengths,
                          y=ys, mode='lines+markers', name=f'{ansatz.name}, ZNE', line=dict(color='red')))
        fig.update_layout(title="Energy vs Bond Length",
                          xaxis_title="Bond Length", yaxis_title="Energy", showlegend=True)

        data = []
        ansatz = zne_vqe_problem.ansatze[0]
        for energy in zne_vqe_problem.energies:
            data.append(energy[0][ansatz][0])
        fig2.add_trace(go.Scatter(x=list(range(1, len(data) + 1)),
                                  y=data, mode='lines+markers', name=f'{ansatz.name}, ZNE', line=dict(color='red')))
        fig2.update_layout(title="Energy vs Iterations",
                           xaxis_title="Iteration", yaxis_title="Energy", showlegend=True)
        qasm_circuits = [dumps(circuit)
                         for circuit in zne_vqe_problem.zne_circuits]
        qpu_list = [i for i in range(1, 10)]
        simulators = [ParallelSimulator(
            num_processes=2, qpus=i) for i in qpu_list]
        runtimes = [simulator.runtime_estimate(
            qasm_circuits) for simulator in simulators]
        fig3.add_trace(go.Scatter(x=qpu_list, y=runtimes, mode='lines+markers', name='ZNE Runtimes',
                                  line=dict(color='red')))
        fig3.update_layout(title="Runtime per Iteration vs QPUs", xaxis_title="Num. of QPUs",
                           yaxis_title="Runtime per Iteration", showlegend=True)

        return fig, fig2, fig3, "done"
    fig.update_layout(title="Energy vs Bond Length",
                      xaxis_title="Bond Length", yaxis_title="Energy", showlegend=True)
    fig2.update_layout(title="Energy vs Iterations",
                       xaxis_title="Iteration", yaxis_title="Energy", showlegend=True)
    fig3.update_layout(title="Runtime per Iteration vs QPUs",
                       xaxis_title="Num. of QPUs", yaxis_title="Runtime per Iteration", showlegend=True)

    return fig, fig2, fig3, no_update


@callback(
    [Output(component_id="energy-graph", component_property="figure", allow_duplicate=True),
     Output(component_id="iterations",
            component_property="figure", allow_duplicate=True),
     Output("loading-1", "children", allow_duplicate=True),],
    Input('noisy-button', 'n_clicks'),
    Input('energy-graph', 'figure'),
    Input('iterations', 'figure'),
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
        energies = noisy_vqe_problem.energies[noisy_vqe_problem.current_iteration - 1]
        for ansatz in noisy_vqe_problem.ansatze:
            ys = []
            for i in range(len(noisy_vqe_problem.bond_lengths)):
                ys.append(energies[i][ansatz][0])
            fig.add_trace(go.Scatter(x=noisy_vqe_problem.bond_lengths,
                          y=ys, mode='lines+markers', name=f'{ansatz.name}, NOISY', line=dict(color='green')))
        fig.update_layout(title="Energy vs Bond Length",
                          xaxis_title="Bond Length", yaxis_title="Energy", showlegend=True)

        data = []
        ansatz = noisy_vqe_problem.ansatze[0]
        for energy in noisy_vqe_problem.energies:
            data.append(energy[0][ansatz][0])
        fig2.add_trace(go.Scatter(x=list(range(1, len(data) + 1)),
                                  y=data, mode='lines+markers', name=f'{ansatz.name}, NOISY', line=dict(color='green')))
        fig2.update_layout(title="Energy vs Iterations",
                           xaxis_title="Iteration", yaxis_title="Energy", showlegend=True)
        return fig, fig2, "done"
    fig.update_layout(title="Energy vs Bond Length",
                      xaxis_title="Bond Length", yaxis_title="Energy", showlegend=True)
    fig2.update_layout(title="Energy vs Iterations",
                       xaxis_title="Iteration", yaxis_title="Energy", showlegend=True)
    return fig, fig2, no_update


if __name__ == "__main__":
    app.run(debug=True)
