import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
import sys
sys.path.append("/Users/salahedeenissa/PycharmProjects/project11/divi/divi")
from dash import Dash, html, dcc, Input, Output, callback, no_update, State
from qprog.vqe import VQE, Ansatze, Optimizers
from qoro_service import QoroService


app = Dash()
app = Dash(external_stylesheets=[dbc.themes.CYBORG])
app.layout = dbc.Container(html.Div(
    html.Div([html.H2("Parallelized VQE", style={'padding-top': '20px'}),
              dbc.Row([
                  dbc.Col(html.P(id='ansatze', children="Ansatz: N/A"), width=4),
                  dbc.Col(html.P(id='optimizer',
                          children="Optimizer: N/A"), width=4),
                  dbc.Col(html.P(id='atoms', children="Atoms: N/A"), width=4),
              ], style={'padding': '10px', 'margin-bottom': '20px'}),
              dbc.Row([dbc.Col(dcc.Graph(id="energy-graph", figure={}), width=6),
                       dbc.Col(dcc.Graph(id="iterations", figure={}), width=6)], style={'margin-top': '50px'}),
              dcc.Loading(id="loading-1", type="default",
                          children=html.Div(id="loading-output-1")),
              dcc.Dropdown(id="Bond Length",options = [], value = None,),
              html.Button('Run VQE', id='start-button', n_clicks=0)
              ])
))

# q_service = QoroService("71ec99c9c94cf37499a2b725244beac1f51b8ee4")
q_service = None
vqe_problem = VQE(symbols=["H", "H"],
                  bond_lengths=[0.5, 1, 1.5],
                  coordinate_structure=[(0, 0, -0.5), (0, 0, 0.5)],
                  ansatze=[Ansatze.HARTREE_FOCK],
                  optimizer=Optimizers.MONTE_CARLO,
                  qoro_service=q_service,
                  shots=1500,
                  max_interations=3)


@callback(
    [Output('ansatze', 'children'),
     Output('optimizer', 'children'),
     Output('atoms', 'children'),],
    Input('start-button', 'n_clicks')
)
def update_metadata(n_clicks):
    if n_clicks >= 0:
        ansatz = f"Ansatze: {[v.name for v in vqe_problem.ansatze]}"
        optimizer = f"Optimizer: {vqe_problem.optimizer.name}"
        atoms = f"Atoms: {vqe_problem.symbols}"
    else:
        ansatz = "Ansatz: "
        optimizer = "Optimizer:"
        atoms = "Atoms: "

    # Return the updated metadata to the output components
    return ansatz, optimizer, atoms

@callback(
    Output('Bond Length', 'options'),
    Input('start-button', 'n_clicks')
)
def update_dropdown(n_clicks):
    if n_clicks:
        return [{'label': bond_length, 'value': bond_length} for bond_length in vqe_problem.bond_lengths]
    return []

@callback(
    [Output(component_id="energy-graph", component_property="figure"),
     Output(component_id="iterations", component_property="figure"),
     Output("loading-1", "children"),],
    Input('start-button', 'n_clicks'),
    State('Bond Length', 'value')
)
def run_vqe(n_clicks, bond_length):
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
        idx = 0
        if bond_length is not None:
            idx = vqe_problem.bond_lengths.index(bond_length)
        for energy in vqe_problem.energies:
            data.append(energy[idx][ansatz][0])

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
