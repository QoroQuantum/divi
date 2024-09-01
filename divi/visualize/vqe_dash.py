import plotly.express as px
import plotly.graph_objects as go

import pandas as pd

from dash import Dash, html, dcc, Input, Output, callback
from qprog.vqe import VQE, Ansatze, Optimizers
from qoro_service import QoroService


app = Dash()
app.layout = html.Div(
    html.Div([html.H2("VQE Results"),
              html.H4("Energy vs Bond Length"),
              dcc.Graph(id="energy-graph", figure={}),
              dcc.Graph(id="iterations", figure={}),
              html.Button('Start VQE', id='start-button', n_clicks=0),
              html.Div(id='state', children=None)
              ])
)

q_service = None
vqe_problem = VQE(symbols=["H", "H"],
                  bond_lengths=[0.5],
                  coordinate_structure=[(0, 0, -0.5), (0, 0, 0.5)],
                  ansatze=[Ansatze.HARTREE_FOCK],
                  optimizer=Optimizers.MONTE_CARLO,
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
    [Output(component_id="energy-graph", component_property="figure"),
     Output(component_id="iterations", component_property="figure")],
    Input('start-button', 'n_clicks')
)
def run_vqe(n_clicks):
    if n_clicks > 0:
        vqe_problem.run()
        energies = vqe_problem.energies[vqe_problem.current_iteration - 1]
        fig = go.Figure()
        fig2 = go.Figure()
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
        return fig, fig2
    return {}, {}


if __name__ == "__main__":
    app.run(debug=True)

    # q_service = QoroService("71ec99c9c94cf37499a2b725244beac1f51b8ee4")
    # q_service = None
    # vqe_problem = VQE(symbols=["H", "H"],
    #                   bond_lengths=[0.75, 1, 1.25],
    #                   coordinate_structure=[(0, 0, 0), (0, 0, 1)],
    #                   ansatze=[Ansatze.RY],
    #                   optimizer=Optimizers.MONTE_CARLO,
    #                   qoro_service=q_service,
    #                   max_interations=2)
    # vqe_problem.run(store_data=True, data_file="vqe_data.pkl")
    # energies = vqe_problem.energies[vqe_problem.current_iteration - 1]
    # print(energies)
