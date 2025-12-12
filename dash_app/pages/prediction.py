import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime
import numpy as np
from dash.exceptions import PreventUpdate 
import traceback 

# A dictionary to map weather codes to human-readable names for the dropdown
# Note: You can align these values with the categories used in your OHE transformer if needed.
WEATHER_CODE_MAP = {
    1.0: "Clear/Fog (1)", 
    2.0: "Scattered Clouds (2)", 
    3.0: "Broken Clouds (3)", 
    4.0: "Cloudy (4)", 
    7.0: "Rain/Light Rain (7)",
    10.0: "Rain/Thunderstorm (10)", # Originally 10.0: "Snowfall", but corrected per context list
    26.0: "Snowfall (26)", # Originally 26.0: "Rain/Thunderstorm", but corrected per context list
    94.0: "Freezing Fog (94)" # Added the last code mentioned in dataset context
}

# --- Function to generate the prediction layout (Passed from app.py) ---
def create_prediction_layout(model_pipeline):
    
    # Check if the model failed to load in app.py
    model_load_status = "Ready" if model_pipeline else "Error/Not Loaded"
    
    # --- Input Components ---
    inputs = dbc.Row([
        dbc.Col(html.Div([
            html.Label("Date & Time", className="fw-bold"),
            dcc.DatePickerSingle(
                id='input-date',
                date=datetime.now().date(),
                display_format='YYYY-MM-DD',
                style={'marginBottom': '10px'}
            ),
            dcc.Dropdown(
                id='input-hour',
                options=[{'label': f'{h:02d}:00', 'value': h} for h in range(24)],
                value=datetime.now().hour,
                placeholder="Select Hour"
            ),
        ]), md=3),

        dbc.Col(html.Div([
            html.Label("Feels Like Temp (°C) [t2]", className="fw-bold"),
            dcc.Slider(id='input-temp-feels', min=-10, max=40, step=0.5, value=15, 
                       marks={i: str(i) for i in range(-10, 41, 10)}),
            html.Label("Humidity (%)", className="mt-3 fw-bold"),
            dcc.Slider(id='input-humidity', min=0, max=100, step=5, value=60, 
                       marks={i: str(i) for i in range(0, 101, 20)})
        ]), md=3),

        dbc.Col(html.Div([
            html.Label("Wind Speed (km/h)", className="fw-bold"),
            dcc.Slider(id='input-wind-speed', min=0, max=60, step=5, value=20, 
                       marks={i: str(i) for i in range(0, 61, 20)}),
            html.Label("Weather Condition", className="mt-3 fw-bold"),
            dcc.Dropdown(
                id='input-weather',
                options=[{'label': v, 'value': k} for k, v in WEATHER_CODE_MAP.items()],
                value=3.0, # Broken Clouds as default
                placeholder="Select Weather Code"
            )
        ]), md=3),

        dbc.Col(html.Div([
            html.Label("Holiday Status", className="fw-bold"),
            dcc.RadioItems(
                id='input-holiday',
                options=[{'label': 'No Holiday', 'value': 0.0}, {'label': 'Is Holiday', 'value': 1.0}],
                value=0.0,
                inline=True
            ),
            html.Label("Weekend Status", className="mt-3 fw-bold"),
            dcc.RadioItems(
                id='input-weekend',
                options=[{'label': 'Weekday', 'value': 0.0}, {'label': 'Weekend', 'value': 1.0}],
                value=0.0,
                inline=True
            )
        ]), md=3),
    ], className="mb-4", style={'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '5px'})

    # --- Output Interface ---
    prediction_interface = html.Div([
        html.H1("Bike Share Demand Forecaster", className="mb-4"),
        
        inputs,
        
        dbc.Button("Calculate Prediction", id="predict-button", color="primary", className="mb-4"),
        
        dbc.Alert(
            "Enter input values and press 'Calculate Prediction'.", 
            id="prediction-output",
            color="info",
            className="h2"
        ),
        
        html.Div(id="model-load-status", children=[
            html.P(f"Model Load Status: {model_load_status}", className="mt-3"),
            html.P("Note: This app uses a scikit-learn Pipeline with custom feature transformers (e.g., Cyclical Encoder) and One-Hot Encoding."),
        ])
    ])

    # --- Prediction Callback (FINAL ROBUST VERSION) ---
    # The callback uses the model_pipeline passed from app.py
    @dash.callback(
        Output('prediction-output', 'children'),
        Output('prediction-output', 'color'),
        [Input('predict-button', 'n_clicks')],
        [State('input-date', 'date'),
         State('input-hour', 'value'),
         State('input-temp-feels', 'value'), 
         State('input-humidity', 'value'),
         State('input-wind-speed', 'value'),
         State('input-weather', 'value'),
         State('input-holiday', 'value'),
         State('input-weekend', 'value'),
         # Note: We don't need a manual 'season' input here because we derive it 
         # in the callback using the timestamp, as suggested by your previous logic.
        ]
    )
    def calculate_prediction(n_clicks, date_str, hour, temp_feels, humidity, wind_speed, weather, is_holiday, is_weekend):
        
        # 1. Prevent callback on app load
        if not n_clicks:
            raise PreventUpdate 

        # 2. Check if the pipeline object was successfully loaded
        if model_pipeline is None:
            return "Model not loaded. Check app.py console for loading error.", "danger"
            
        # 3. Check for missing/None inputs
        input_list = [date_str, hour, temp_feels, humidity, wind_speed, weather, is_holiday, is_weekend]
        if any(i is None for i in input_list):
            return "Please ensure all input fields have a valid value selected.", "warning"

        try:
            # 4. Time Calculations and Feature Derivation
            date_time_str = f"{date_str} {hour:02d}:00:00"
            timestamp = pd.to_datetime(date_time_str)

            # Season calculation: 0=Spring, 1=Summer, 2=Fall, 3=Winter. 
            # (month % 12 + 3) // 3 => March(3)=1, June(6)=2, Sep(9)=3, Dec(12)=4 (but the data is 0-3, so we might need correction)
            # The standard mapping is: Dec-Feb=0, Mar-May=1, Jun-Aug=2, Sep-Nov=3
            # Let's use the season column definition from the dataset context: 0-spring; 1-summer; 2-fall; 3-winter.
            
            month = timestamp.month
            if month in [3, 4, 5]: # Mar, Apr, May
                season_calc = 0.0 # Spring
            elif month in [6, 7, 8]: # Jun, Jul, Aug
                season_calc = 1.0 # Summer
            elif month in [9, 10, 11]: # Sep, Oct, Nov
                season_calc = 2.0 # Fall
            else: # Dec, Jan, Feb
                season_calc = 3.0 # Winter
            
            # 5. Define Expected Column Order (Crucial for ColumnTransformer/Pipeline)
            # This must match the original raw features used to train the pipeline.
            EXPECTED_COLUMNS = [
                't1', 't2', 'hum', 'wind_speed', 
                'weather_code', 'is_holiday', 'is_weekend', 'season', 
                'hour', 'day_of_week', 'month' # These are the time features we derived during engineering
            ]
            
            # The original data had 't1' and 't2'. Since we only have one temperature input ('Feels Like Temp'),
            # we must use it for both t1 (real temp) and t2 (feels like temp) to fill the required columns.
            X_new = pd.DataFrame([{
                # Numerical Features
                't1': temp_feels,          # Real Temperature
                't2': temp_feels,          # Feels Like Temperature
                'hum': humidity,
                'wind_speed': wind_speed,
                
                # Categorical Features (use float type as defined in original data)
                'weather_code': float(weather),
                'is_holiday': is_holiday,
                'is_weekend': is_weekend,
                'season': season_calc,
                
                # Time-based features
                'hour': float(timestamp.hour),
                'day_of_week': float(timestamp.dayofweek), # 0=Monday, 6=Sunday
                'month': float(timestamp.month)
                
            }], index=[timestamp])
            
            # --- Enforce the Exact Column Order ---
            X_new = X_new[EXPECTED_COLUMNS]

            # 6. Make Prediction using the loaded pipeline
            prediction_value = model_pipeline.predict(X_new)[0]
            
            # 7. Format Output
            predicted_count = max(0, int(round(prediction_value)))
            
            return f"Predicted Hourly Bike Shares: {predicted_count:,}", "success"
            
        except Exception as e:
            # Print the detailed traceback to the console for debugging
            print("\n--- PREDICTION CALLBACK ERROR TRACEBACK ---")
            traceback.print_exc()
            print("-------------------------------------------\n")
            
            # Return an informative error message
            return f"Prediction Failed ({type(e).__name__}). Please check the console traceback for details.", "danger"

    return prediction_interface