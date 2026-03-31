import dash
import dash_bootstrap_components as dbc
import os
import sys

# Ensure parent directory is in path for data/simulation imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
app.title = "Monte Carlo Simulator"

# Import layout and callbacks after app initialization
from dashboard.layout import create_layout
app.layout = create_layout()

import dashboard.callbacks  # Registers callbacks

if __name__ == "__main__":
    app.run(debug=True, port=8050)
