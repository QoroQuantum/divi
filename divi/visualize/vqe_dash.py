import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd

from dash import Dash, html, dcc, Input, Output, callback, no_update
from qprog.vqe import VQE, Ansatze, Optimizers
from qoro_service import QoroService


app = Dash()
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div(
    html.Div([html.H2("VQE Results"),
              html.H4("Energy vs Bond Length"),
              html.Div([
                  html.P(id='Ansatze', children="Ansatz: N/A"),
                  html.P(id='Optimizer', children="Optimizer: N/A"),
                  html.P(id='Atoms', children="Atoms: N/A"),
              ], style={'border': '1px solid black', 'padding': '10px', 'margin-bottom': '20px'}),
              dcc.Graph(id="energy-graph", figure={}),
              dcc.Graph(id="iterations", figure={}),
              dcc.Loading(id="loading-1", type="default",
                          children=html.Div(id="loading-output-1")),
              html.Button('Start VQE', id='start-button', n_clicks=0),
              html.Div(id='state', children=None)
              ])
)

# q_service = QoroService("71ec99c9c94cf37499a2b725244beac1f51b8ee4")
q_service = None
vqe_problem = VQE(symbols=["H", "H"],
                  bond_lengths=[0.5, 1, 1.5],
                  coordinate_structure=[(0, 0, -0.5), (0, 0, 0.5)],
                  ansatze=[Ansatze.HARTREE_FOCK],
                  optimizer=Optimizers.NELDER_MEAD,
                  qoro_service=q_service,
                  shots=500,
                  max_interations=5)

@callback(
    Output(component_id="state", component_property="children"),
    Input('start-button', 'n_clicks')
)
def started(n_clicks):
    if n_clicks > 0:
        return f"VQE Execution Started"
    return ""

@callback(
    [Output('ansatze', 'children'),
     Output('optimizer', 'children'),
     Output('atoms', 'children'),],
    Input('start-button', 'n_clicks')
)

def update_metadata(n_clicks):
    if n_clicks > 0:
        ansatz = f"Ansatze: {vqe_problem.ansatze}"
        optimizer = f"Optimizer: {vqe_problem.optimizer}"
        atoms = f"Atoms: {vqe_problem.symbols}"

    else:
        ansatz = "Ansatz: N/A"
        optimizer = "Optimizer: N/A"
        atoms = "Atoms: N/A"

    # Return the updated metadata to the output components
    return ansatz, optimizer, atoms


@callback(
    [Output(component_id="energy-graph", component_property="figure"),
     Output(component_id="iterations", component_property="figure"),
     Output("loading-1", "children"),],
    Input('start-button', 'n_clicks')
)
def run_vqe(n_clicks):
    fig = go.Figure()
    fig2 = go.Figure()
    if n_clicks > 0:
        vqe_problem.run()
        energies = vqe_problem.energies[vqe_problem.current_iteration - 1]
        for ansatz in vqe_problem.ansatze:
            ys = []
            for i in range(len(vqe_problem.bond_lengths)):
                ys.append(energies[i][ansatz][0])
            fig.add_trace(go.Scatter(x=vqe_problem.bond_lengths,
                          y=ys, mode='lines+markers', name=ansatz.name))
        fig.update_layout(title="Energy vs Bond Length",
                          xaxis_title="Bond Length", yaxis_title="Energy")

        data = []
        ansatz = vqe_problem.ansatze[0]
        for energy in vqe_problem.energies:
            data.append(energy[0][ansatz][0])

        print(data)
        fig2.add_trace(go.Scatter(x=list(range(1, len(data) + 1)),
                                  y=data, mode='lines+markers', name="Hartree Fock"))
        fig2.update_layout(title="Energy vs Iterations",
                           xaxis_title="Iteration", yaxis_title="Energy")
        return fig, fig2, "done"

    fig.update_layout(title="Energy vs Bond Length",
                      xaxis_title="Bond Length", yaxis_title="Energy")
    fig2.update_layout(title="Energy vs Iterations",
                       xaxis_title="Iteration", yaxis_title="Energy")
    return fig, fig2, no_update


if __name__ == "__main__":
    app.run(debug=True)
