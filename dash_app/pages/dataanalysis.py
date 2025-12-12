import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

# --- Layout for the Data Analysis Page ---

layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Key Data Analysis and Feature Insights", className="text-info mb-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col(html.Div([
            html.H3("1. Temperature is a Strong Predictor"),
            html.P(
                "There is a clear positive correlation between the 'Feels Like Temperature' and the total number of hourly bike shares. "
                "As the temperature rises, demand generally increases, particularly around the 20°C mark."
            ),
            # FIX: Use the direct relative path /assets/filename
            html.Img(src='/assets/image_440918.png', className="img-fluid mb-4", style={'maxWidth': '800px'}),
        ]), md=12),
    ]),
    
    dbc.Row([
        dbc.Col(html.Div([
            html.H3("2. Impact of Weather Conditions"),
            html.P(
                "The distribution of bike shares varies dramatically across different weather conditions. "
                "Conditions like 'Clear/Fog' and 'Scattered Clouds' show the highest median demand and range, while adverse conditions like 'Snowfall' and 'Rain/Thunderstorm' suppress demand significantly."
            ),
            # FIX: Use the direct relative path /assets/filename
            html.Img(src='/assets/image_440974.png', className="img-fluid mb-4", style={'maxWidth': '800px'}),
        ]), md=12),
    ]),
    
    dbc.Row([
        dbc.Col(html.Div([
            html.H3("3. Cyclical Patterns"),
            html.Ul([
                html.Li("Demand peaks during morning (8-9 AM) and evening (5-7 PM) commute hours on weekdays."),
                html.Li("Demand is highest in Summer and lowest in Winter."),
                html.Li("Weekend and holiday effects are significant, often suppressing commuter peaks but boosting leisure riding.")
            ]),
        ], className="mb-5")),
    ])
])