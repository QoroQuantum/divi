import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html, no_update

from divi.qprog import MLAE

app = Dash()
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    html.Div(
        [
            html.H2("MLAE Results"),
            dcc.Graph(id="individual_amplitude_estimations", figure={}),
            dcc.Graph(id="maximum_likelihood", figure={}),
            dcc.Loading(
                id="loading-1", type="default", children=html.Div(id="loading-output-1")
            ),
            html.Button("Start MLAE", id="start-button", n_clicks=0),
            html.Div(id="state", children=None),
        ]
    )
)

mlae_problem = MLAE(
    grovers=[2, 3, 4],
    qubits_to_measure=0,
    probability=0.2,
    qoro_service=None,
)
mlae_problem.run()
mlae_problem.generate_maximum_likelihood_function(factor=10e30)


@callback(
    Output(component_id="state", component_property="children"),
    Input("start-button", "n_clicks"),
)
def started(n_clicks):
    if n_clicks > 0:
        return f"MLAE Execution Started"
    return ""


@callback(
    [
        Output(
            component_id="individual_amplitude_estimations", component_property="figure"
        ),
        Output(component_id="maximum_likelihood", component_property="figure"),
        Output("loading-1", "children"),
    ],
    Input("start-button", "n_clicks"),
)
def run_mlae(n_clicks):
    grover_fig = go.Figure()
    max_likelihood_fig = go.Figure()

    if n_clicks > 0:
        a = np.linspace(0, 1, 100)  # set of amplitudes

        for func, grover in zip(
            mlae_problem.likelihood_functions, mlae_problem.grovers
        ):
            grover_fig.add_trace(
                go.Scatter(
                    x=a,
                    y=func(a),
                    mode="lines+markers",
                    name=f"{grover} Grover Operators",
                )
            )

        min_y, max_y = min(min(fn(a)) for fn in mlae_problem.likelihood_functions), max(
            max(fn(a)) for fn in mlae_problem.likelihood_functions
        )
        for fig in (grover_fig, max_likelihood_fig):
            fig.add_shape(
                type="line",
                x0=mlae_problem.probability,
                x1=mlae_problem.probability,
                y0=min_y,
                y1=max_y,
                line=dict(color="Red", width=2, dash="dashdot"),
            )

        max_likelihood_fig.add_trace(
            go.Scatter(
                x=a,
                y=mlae_problem.maximum_likelihood_fn(a),
                mode="lines+markers",
                name="Maximum Likelihood Function",
            )
        )
        grover_fig.update_layout(
            title="Individual Amplitude Likelihood Functions",
            xaxis_title="Amplitude",
            yaxis_title="Likelihood",
            showlegend=True,
        )

        max_likelihood_fig.update_layout(
            title="Maximum Likelihood Function",
            xaxis_title="Amplitude",
            yaxis_title="Likelihood",
        )
        return grover_fig, max_likelihood_fig, "done"

    grover_fig.update_layout(
        title="Individual Amplitude Likelihood Functions",
        xaxis_title="Amplitude",
        yaxis_title="Likelihood",
        showlegend=True,
    )
    max_likelihood_fig.update_layout(
        title="Maximum Likelihood Function",
        xaxis_title="Amplitude",
        yaxis_title="Likelihood",
    )

    return grover_fig, max_likelihood_fig, no_update


if __name__ == "__main__":
    app.run(debug=True)
