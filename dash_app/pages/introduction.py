import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

# --- Layout for the Introduction Page ---

layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Introduction to the London Bike Share Forecasting Project", className="text-primary mb-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col(html.Div([
            html.H3("Project Overview"),
            html.P(
                "This application predicts hourly bike share demand in London using a machine learning model based on historical data, weather conditions, and time features. "
                "The goal is to provide reliable forecasts to help optimize bike station restocking and operational planning."
            ),
            
            html.H3("The Data Source"),
            html.P(
                "The core dataset is 'london_merged.csv', which aggregates hourly counts of rented bicycles "
                "with corresponding environmental and calendar factors, including:"
            ),
            html.Ul([
                html.Li("Hourly bike rental counts ('count')"),
                html.Li("Real and 'Feels Like' temperature (°C)"),
                html.Li("Humidity and Wind Speed"),
                html.Li("Weather codes (e.g., Clear, Cloudy, Rain)"),
                html.Li("Calendar information (Holiday, Weekend, Season)")
            ]),
        ], className="mb-5")),
    ]),
    
    dbc.Row([
        dbc.Col(html.Div([
            html.H3("Forecasting Model"),
            html.P(
                "After testing several models, a highly-tuned **Random Forest Regressor** was selected for its superior performance in capturing the non-linear relationship between "
                "time/weather and demand. The model achieves an **R-squared score of approximately 0.83** on the holdout test data."
            ),
            html.P(
                "The model pipeline includes robust preprocessing steps such as **Trigonometric Encoding** for cyclical features (hour, day of year) and **Robust Scaling** for temperature and wind speed."
            ),
        ], className="mb-5")),
    ])
])