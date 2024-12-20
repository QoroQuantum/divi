import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from divi.qprog import VQE, VQEAnsatze, Optimizers
from divi.services.qoro_service import QoroService

from dash import Dash, html, dcc, Input, Output, callback, no_update


app = Dash(__name__)
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
fig = go.Figure()
fig2 = go.Figure()

fig = go.Figure()
fig.update_layout(
    title={"text": "Energy vs Bond Length", "x": 0.5},  # Center the title
    title_font={"size": 20},  # Optional: Adjust the font size
)

fig2 = go.Figure()
fig2.update_layout(
    title={"text": "VQE Iterations", "x": 0.5},  # Center the title
    title_font={"size": 20},  # Optional: Adjust the font size
)

app.layout = html.Div([
    html.H1("Simulating the H2 Molecule with VQE",
            style={"padding": "20px", "textAlign": "center", "color": "#333", "fontFamily": "Arial, sans-serif"}),
    html.Div([
        html.Div(
            [
                dcc.Graph(id="energy-graph", figure=fig)
            ],
            style={"width": "48%", "display": "inline-block",
                   "padding": "10px", "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)"}
        ),
        html.Div(
            [
                dcc.Graph(id="iterations", figure=fig2),
            ],
            style={"width": "48%", "display": "inline-block", "float": "right",
                   "padding": "10px", "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)"}
        )],
        style={"width": "95%", "margin": "auto"}),
    dcc.Loading(
        id="loading-1",
        type="default",
        children=html.Div(id="loading-output-1"),
        style={"padding": "20px", "margin-top": "35px"}
    ),
    html.Div(
        [
            html.Button('Run VQE',
                        id='start-button',
                        n_clicks=0,
                        style={
                            "width": "120px",
                            "padding": "10px 20px",
                            "backgroundColor": "#4CAF50",
                            "color": "white",
                            "border": "none",
                            "margin-top": "35px",
                            "borderRadius": "5px",
                            "cursor": "pointer",
                            "fontSize": "16px",
                            "fontWeight": "bold",
                            "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
                            "transition": "background-color 0.3s, transform 0.2s"
                        }),
        ],
        style={
            "display": "flex",
            "justifyContent": "center",  # Centers horizontally
            "alignItems": "center",  # Centers vertically (if needed)
            "padding": "20px"
        }  # Adds spacing around the button
    ),

])

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

# q_service = QoroService("71ec99c9c94cf37499a2b725244beac1f51b8ee4")
q_service = None
vqe_problem = VQE(
    symbols=["H", "H"],
    bond_length=[0.5, 1, 1.5],
    coordinate_structure=[(0, 0, -0.5), (0, 0, 0.5)],
    ansatz=[VQEAnsatze.HARTREE_FOCK],
    optimizer=Optimizers.NELDER_MEAD,
    shots=8000,
    max_iterations=5,
    qoro_service=None
)


@app.callback(
    Output('start-button', 'disabled'),
    Input('start-button', 'n_clicks'),
    prevent_initial_call=True  # Don't disable the button on page load
)
def disable_button(n_clicks):
    if n_clicks > 0:
        return True  # Disable the button after it is clicked
    return False


@callback(
    [Output(component_id="energy-graph", component_property="figure"),
     Output(component_id="iterations", component_property="figure"),
     Output(component_id="loading-1", component_property="children")],
    Input('start-button', 'n_clicks')
)
def run_vqe(n_clicks):
    if n_clicks > 0:
        vqe_problem.run()
        energies = vqe_problem.energies[vqe_problem.current_iteration - 1]
        for ansatz in vqe_problem.ansatze:
            ys = []
            for i in range(len(vqe_problem.bond_lengths)):
                ys.append(energies[i][ansatz][0])
            fig.add_trace(go.Scatter(x=vqe_problem.bond_lengths,
                          y=ys, mode='lines+markers', name=ansatz.name))

            data = []

            for energy in vqe_problem.energies:
                data.append(energy[0][ansatz][0])

            fig2.add_trace(go.Scatter(x=list(range(1, len(data) + 1)),
                                      y=data, mode='lines+markers', name=ansatz.name))

        fig.update_layout(title="Energy vs Bond Length",
                          xaxis_title="Bond Length", yaxis_title="Energy")
        fig2.update_layout(title="Energy vs Iterations",
                           xaxis_title="Iteration", yaxis_title="Energy")
        return fig, fig2, ""
    return {}, {}, no_update


if __name__ == "__main__":
    app.run(debug=True)
