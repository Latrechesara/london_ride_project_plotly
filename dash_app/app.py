import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin

# --- 1. Define the Custom Transformer Class (CRITICAL for joblib) ---
# This class MUST be defined here exactly as it was defined during training.
class CyclicalFeatureTransformer(BaseEstimator, TransformerMixin):
    """Transforms a single cyclical feature into sin and cos components."""
    
    def __init__(self, feature, period):
        self.feature = feature
        self.period = period
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[self.feature])
            
        X[f'{self.feature}_sin'] = np.sin(2 * np.pi * X[self.feature] / self.period)
        X[f'{self.feature}_cos'] = np.cos(2 * np.pi * X[self.feature] / self.period)
        
        return X[[f'{self.feature}_sin', f'{self.feature}_cos']].values


# --- 2. Import Page Layouts ---
import pages.introduction as introduction
import pages.dataanalysis as dataanalysis
import pages.prediction as prediction 


# --- 3. Load the Model ---
MODEL_PATH = 'best_bike_share_forecaster_pipeline_final.joblib'
forecaster_pipeline = None

if os.path.exists(MODEL_PATH):
    try:
        forecaster_pipeline = joblib.load(MODEL_PATH)
        print("Model pipeline loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model pipeline from {MODEL_PATH}. Check file path/integrity. Error: {e}")
else:
    print(f"WARNING: Model file not found at {MODEL_PATH}. Prediction page will be disabled.")


# --- 4. Initialize App ---
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.CERULEAN], 
                suppress_callback_exceptions=True)
server = app.server

# --- 5. Sidebar Layout (Styled via custom.css) ---
sidebar = html.Div(
    [
        html.H2("LONDON RIDE", className="display-4"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("1. Introduction", href="/", active="exact"),
                dbc.NavLink("2. Data Analysis", href="/analysis", active="exact"),
                dbc.NavLink("3. Prediction App", href="/prediction", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    className="sidebar", # Applies custom.css styling
    style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "18rem",
        "padding": "2rem 1rem",
    },
)

# --- 6. Main Content Layout ---
content = html.Div(id='page-content', style={
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
})

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    sidebar,
    content
])


# --- 7. Routing Callback ---
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return introduction.layout
    elif pathname == '/analysis':
        return dataanalysis.layout
    elif pathname == '/prediction':
        return prediction.create_prediction_layout(forecaster_pipeline)
    else:
        return dbc.Container([
            html.H1("404: Not Found", className="text-danger"),
            html.P(f"The path '{pathname}' was not recognized.")
        ])

# --- 8. Run Server ---
if __name__ == '__main__':
    app.run(debug=True)