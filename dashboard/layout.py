import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table

def _create_metric_card(id_prefix, title):
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="card-subtitle mb-2 text-muted", style={'fontSize': '0.85rem'}),
            html.H4(id=f"{id_prefix}-val", className="card-title", style={'fontFamily': 'monospace'})
        ], className="text-center p-2")
    ], color="dark", outline=True)

def create_layout():
    header = dbc.Row([
        dbc.Col([
            html.H2("ANTIGRAVITY // MONTE CARLO", style={'fontFamily': 'monospace', 'margin': 0}),
            html.Small("Probabilistic Price Simulation Engine", className="text-muted")
        ], width=9),
        dbc.Col(
            html.Div(id="ticker-badge", className="badge bg-secondary fs-5 float-end", style={'fontFamily': 'monospace'}),
            width=3, align="center"
        )
    ], className="mb-4 mt-3 border-bottom pb-3")

    controls = dbc.Card([
        dbc.CardBody([
            html.H5("Parameters", className="card-title"),
            
            html.Label("Ticker"),
            dcc.Dropdown(
                id="dropdown-ticker",
                options=[{'label': t, 'value': t} for t in ['KC=F', 'SPY']],
                value='SPY',
                className="mb-3",
                clearable=False,
                style={"color": "black"} 
            ),
            
            html.Label("Simulation Mode"),
            dcc.Dropdown(
                id="dropdown-mode",
                options=[
                    {'label': 'GBM', 'value': 'gbm'},
                    {'label': 'Student-T', 'value': 'student_t'},
                    {'label': 'Jump Diffusion', 'value': 'jump_diffusion'}
                ],
                value='gbm',
                className="mb-4",
                clearable=False,
                style={"color": "black"}
            ),
            
            html.Label("Number of Paths"),
            dcc.Slider(id="slider-paths", min=1000, max=20000, step=1000, value=5000, 
                       marks={1000: '1k', 10000: '10k', 20000: '20k'}, className="mb-4"),
            
            html.Label("Horizon (days)"),
            dcc.Slider(id="slider-horizon", min=63, max=504, step=63, value=252,
                       marks={63: '3M', 252: '1Y', 504: '2Y'}, className="mb-4"),
            
            html.Div(id="student-t-controls", style={'display': 'none'}, children=[
                html.Label("Student-t df"),
                dcc.Slider(id="slider-df", min=2.1, max=10.0, step=0.1, value=3.0, 
                           marks={2.1: '2.1', 5: '5', 10: '10'}, className="mb-4")
            ]),
            
            html.Div(id="jump-controls", style={'display': 'none'}, children=[
                html.Label("Jump Intensity (lambda)"),
                dbc.Input(id="input-lambda", type="number", value=5.0, className="mb-3"),
                
                html.Label("Jump Mean (mu_j)"),
                dbc.Input(id="input-mu-j", type="number", value=-0.02, className="mb-3"),
                
                html.Label("Jump Std (sigma_j)"),
                dbc.Input(id="input-sigma-j", type="number", value=0.04, className="mb-3"),
            ]),
            
            dbc.Button("RUN SIMULATION", id="btn-run", color="primary", className="w-100", 
                       style={'fontFamily': 'monospace'})
        ])
    ])

    v_spacer = html.Div(style={'height': '20px'})
    
    right_panel = dcc.Loading(type="cube", children=[
        html.Div(id="charts-container", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id="graph-fan"), width=6),
                dbc.Col(dcc.Graph(id="graph-return-dist"), width=6)
            ]),
            v_spacer,
            dbc.Row([
                dbc.Col(dcc.Graph(id="graph-price-hist"), width=6),
                dbc.Col([
                    html.H5("Risk Metrics", style={'fontFamily': 'monospace', 'marginBottom': '15px'}),
                    dash_table.DataTable(
                        id="table-metrics",
                        style_as_list_view=True,
                        style_header={'display': 'none'},
                        style_cell={
                            'backgroundColor': '#1a1a2e',
                            'color': 'white',
                            'border': 'none',
                            'textAlign': 'left',
                            'padding': '8px',
                            'fontFamily': 'monospace'
                        },
                        style_data_conditional=[
                            {
                                'if': {'state': 'active'},
                                'backgroundColor': 'rgba(255, 255, 255, 0.1)',
                                'border': 'none'
                            }
                        ]
                    )
                ], width=6)
            ])
        ])
    ])

    tab_sim = dbc.Tab(label="SIMULATION", tab_id="tab-sim", children=[
        html.Div(style={'marginTop': '20px'}),
        dbc.Row([
            dbc.Col(controls, width=3),
            dbc.Col(right_panel, width=9)
        ])
    ])
    
    # -----------------------
    # POSITION TAB SECTION 1
    # -----------------------
    pos_cards = dbc.Row([
        dbc.Col(_create_metric_card("pos-entry", "Entry Price")),
        dbc.Col(_create_metric_card("pos-curr", "Current Price")),
        dbc.Col(_create_metric_card("pos-units", "Units Held")),
        dbc.Col(_create_metric_card("pos-val", "Position Value")),
        dbc.Col(_create_metric_card("pos-upl", "Unrealised P&L")),
        dbc.Col(_create_metric_card("pos-perc", "P&L %")),
        dbc.Col(_create_metric_card("pos-dry", "Dry Powder")),
    ], className="mb-4")

    # -----------------------
    # POSITION TAB SECTION 2
    # -----------------------
    pos_charts = dcc.Loading(type="cube", children=[
        dbc.Row([
            dbc.Col(dcc.Graph(id="graph-pos-euro-dist"), width=6),
            dbc.Col(dcc.Graph(id="graph-pos-scenarios"), width=6)
        ], className="mb-4")
    ])
    
    # -----------------------
    # POSITION TAB SECTION 3
    # -----------------------
    cc_calculator = dbc.Card([
         dbc.CardBody([
             html.H5("Covered Call Viability Calculator", className="card-title mb-4"),
             dbc.Row([
                 # Left: Inputs
                 dbc.Col([
                     html.Label("Target Monthly Income"),
                     dcc.Dropdown(
                         id="dropdown-cc-target",
                         options=[{'label': f'€{t}', 'value': t} for t in [25, 50, 100, 200]],
                         value=100,
                         className="mb-3",
                         clearable=False,
                         style={"color": "black"}
                     ),
                     
                     html.Label("Expected Monthly Premium Rate"),
                     dcc.Slider(id="slider-cc-rate", min=0.01, max=0.05, step=0.001, value=0.025,
                                marks={0.01: '1%', 0.025: '2.5%', 0.05: '5%'}, className="mb-3"),
                     
                     html.Label("Current Portfolio Size (£/€)"),
                     dbc.Input(id="input-cc-portfolio", type="number", className="mb-3", style={'maxWidth': '200px'}),
                     
                     html.Label("Monthly Contribution"),
                     dcc.Slider(id="slider-cc-savings", min=0, max=500, step=50, value=200,
                                marks={0: '0', 250: '250', 500: '500'}, className="mb-4"),
                                
                     dbc.Button("RECALCULATE", id="btn-pos-calc", color="success", className="w-100", 
                                style={'fontFamily': 'monospace', 'maxWidth': '200px'})
                 ], width=4),
                 
                 # Middle: Tables/Metrics
                 dbc.Col([
                     dbc.Row([
                         dbc.Col([
                             html.H6("Current Position Est."),
                             html.H4(id="cc-est-monthly", style={'color': '#00ff88'}),
                             html.Small(id="cc-est-annual", className="text-muted")
                         ], className="mb-3")
                     ]),
                     html.H6("Capital Required"),
                     dash_table.DataTable(
                        id="table-cc",
                        style_as_list_view=True,
                        style_cell={
                            'backgroundColor': '#1a1a2e',
                            'color': 'white',
                            'border': 'none',
                            'textAlign': 'left',
                            'padding': '8px',
                            'fontFamily': 'monospace'
                        },
                        style_data_conditional=[
                            {
                                'if': {'state': 'active'},
                                'backgroundColor': 'rgba(255, 255, 255, 0.1)',
                                'border': 'none'
                            }
                        ]
                    )
                 ], width=4),
                 
                 # Right: Bar Chart
                 dbc.Col([
                     dcc.Graph(id="graph-cc-months", style={'height': '300px'})
                 ], width=4)
             ])
         ])
    ])
    
    tab_pos = dbc.Tab(label="MY POSITION", tab_id="tab-pos", children=[
        html.Div(style={'marginTop': '20px'}),
        
        dbc.Row([
             dbc.Col([
                 html.Label("Position Configuration:"),
                 dbc.InputGroup([
                     dbc.InputGroupText("Entry Price:"),
                     dbc.Input(id="input-pos-entry", type="number", value=51.75),
                     dbc.InputGroupText("Units:"),
                     dbc.Input(id="input-pos-units", type="number", value=7),
                 ], style={'maxWidth': '400px'})
             ])
        ], className="mb-3"),
        
        pos_cards,
        pos_charts,
        cc_calculator
    ])

    from strategy.rules import STRATEGY_REGISTRY
    
    strat_controls = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Strategy"),
                    dcc.Dropdown(
                        id="strategy-dropdown",
                        options=[{'label': k, 'value': k} for k in STRATEGY_REGISTRY.keys()],
                        value=list(STRATEGY_REGISTRY.keys())[0],
                        clearable=False,
                        style={"color": "black"}
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Ticker"),
                    dcc.Dropdown(
                        id="strategy-ticker",
                        options=[{'label': 'KC=F', 'value': 'KC=F'}, {'label': 'SPY', 'value': 'SPY'}],
                        value='KC=F',
                        clearable=False,
                        style={"color": "black"}
                    )
                ], width=2),
                dbc.Col([
                    html.Label("Backtest Period"),
                    dcc.Dropdown(
                        id="strategy-period",
                        options=[
                            {'label': '1 Year', 'value': '1Y'},
                            {'label': '2 Years', 'value': '2Y'},
                            {'label': '3 Years', 'value': '3Y'},
                            {'label': '5 Years', 'value': '5Y'}
                        ],
                        value='5Y',
                        clearable=False,
                        style={"color": "black"}
                    )
                ], width=2),
                dbc.Col([
                    html.Label("Tx Cost (bps)"),
                    dcc.Slider(id="strategy-cost-slider", min=0, max=50, step=5, value=10, marks={0:'0', 10:'10', 50:'50'})
                ], width=3),
                dbc.Col([
                    html.Label("\u200b"),
                    dbc.Button("RUN OVERLAY", id="run-strategy-btn", color="primary", className="w-100", style={'fontFamily': 'monospace'})
                ], width=2)
            ])
        ])
    ], className="mb-4")

    strat_row2 = dbc.Row([
        dbc.Col(dcc.Loading(type="cube", children=dcc.Graph(id="strategy-equity-chart")), width=7),
        dbc.Col([
            html.H5("Strategy vs Buy & Hold", className="mb-3", style={'fontFamily': 'monospace'}),
            dash_table.DataTable(
                id="strategy-stats-table",
                style_as_list_view=True,
                style_cell={
                    'backgroundColor': '#1a1a2e',
                    'color': 'white', 
                    'border': 'none',
                    'fontFamily': 'monospace',
                    'padding': '8px',
                    'textAlign': 'left'
                },
                style_header={'fontWeight': 'bold', 'backgroundColor': '#0e1117'}
            )
        ], width=5)
    ], className="mb-4")

    strat_row3 = dbc.Row([
        dbc.Col(dcc.Loading(type="cube", children=dcc.Graph(id="mc-strategy-fan")), width=7),
        dbc.Col([
            html.H5("Monte Carlo Probabilities", className="mb-3", style={'fontFamily': 'monospace'}),
            html.Div(id="mc-strategy-cards")
        ], width=5)
    ], className="mb-4")

    strat_row4 = dbc.Row([
        dbc.Col(dcc.Loading(type="cube", children=dcc.Graph(id="monthly-return-hist")), width=12)
    ])

    tab_strat = dbc.Tab(label="STRATEGY", tab_id="tab-strat", children=[
        html.Div(style={'marginTop': '20px'}),
        strat_controls,
        strat_row2,
        strat_row3,
        strat_row4
    ])

    port_controls = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Strategy"),
                    dcc.Dropdown(
                        id="portfolio-strategy",
                        options=[
                            {'label': 'Dual Momentum', 'value': 'dual_momentum'},
                            {'label': 'CS Momentum', 'value': 'cs_momentum'},
                            {'label': 'Equal Weight', 'value': 'equal_weight'}
                        ],
                        value='dual_momentum',
                        clearable=False,
                        style={"color": "black"}
                    )
                ], width=2),
                dbc.Col([
                    html.Label("Lookback"),
                    dcc.Dropdown(
                        id="portfolio-lookback",
                        options=[{'label': f'{d}d', 'value': d} for d in [21, 42, 63, 126]],
                        value=126,
                        clearable=False,
                        style={"color": "black"}
                    )
                ], width=1),
                dbc.Col([
                    html.Label("Top N"),
                    dcc.Slider(id="portfolio-top-n", min=3, max=8, step=1, value=4, marks={i: str(i) for i in range(3, 9)})
                ], width=2),
                dbc.Col([
                    html.Label("Sizing"),
                    dcc.Dropdown(
                        id="portfolio-sizing",
                        options=[
                            {'label': 'Inverse Volatility', 'value': 'inverse_vol'},
                            {'label': 'Equal Weight', 'value': 'equal_weight'}
                        ],
                        value='inverse_vol',
                        clearable=False,
                        style={"color": "black"}
                    )
                ], width=2),
                dbc.Col([
                    html.Label("Period"),
                    dcc.Dropdown(
                        id="portfolio-period",
                        options=[{'label': '3 Years', 'value': '3y'}, {'label': '5 Years', 'value': '5y'}],
                        value='5y',
                        clearable=False,
                        style={"color": "black"}
                    )
                ], width=1),
                dbc.Col([
                    html.Label("Initial (€)"),
                    dbc.Input(id="portfolio-capital", type="number", value=7000)
                ], width=1),
                dbc.Col([
                    html.Label("Monthly (€)"),
                    dbc.Input(id="portfolio-contribution", type="number", value=200)
                ], width=1),
                dbc.Col([
                    html.Label("\u200b"),
                    dbc.Button("RUN", id="run-portfolio-btn", color="success", className="w-100", style={'fontFamily': 'monospace', 'fontSize': '0.9rem'})
                ], width=2)
            ])
        ])
    ], className="mb-4")

    port_row2 = dbc.Row([
        dbc.Col(dcc.Loading(type="cube", children=dcc.Graph(id="portfolio-equity-chart")), width=7),
        dbc.Col([
            html.H5("Portfolio Summary", className="mb-3", style={'fontFamily': 'monospace'}),
            dash_table.DataTable(
                id="portfolio-stats-table",
                style_as_list_view=True,
                style_cell={
                    'backgroundColor': '#1a1a2e',
                    'color': 'white', 
                    'border': 'none',
                    'fontFamily': 'monospace',
                    'padding': '8px',
                    'textAlign': 'left'
                },
                style_header={'display': 'none'}
            )
        ], width=5)
    ], className="mb-4")

    port_row3 = dbc.Row([
        dbc.Col(dcc.Loading(type="cube", children=dcc.Graph(id="portfolio-alloc-chart")), width=7),
        dbc.Col(dcc.Loading(type="cube", children=dcc.Graph(id="portfolio-asset-returns-chart")), width=5)
    ], className="mb-4")

    port_row4 = dbc.Row([
        dbc.Col(dcc.Loading(type="cube", children=dcc.Graph(id="portfolio-monthly-hist")), width=7),
        dbc.Col([
            html.H5("Benchmark Comparison", className="mb-3", style={'fontFamily': 'monospace'}),
            dash_table.DataTable(
                id="portfolio-benchmark-table",
                style_as_list_view=True,
                style_cell={
                    'backgroundColor': '#1a1a2e',
                    'color': 'white', 
                    'border': 'none',
                    'fontFamily': 'monospace',
                    'padding': '8px',
                    'textAlign': 'left'
                },
                style_header={'fontWeight': 'bold', 'backgroundColor': '#0e1117'}
            )
        ], width=5)
    ], className="mb-4")

    tab_port = dbc.Tab(label="PORTFOLIO", tab_id="tab-port", children=[
        html.Div(style={'marginTop': '20px'}),
        port_controls,
        port_row2,
        port_row3,
        port_row4
    ])

    layout = dbc.Container([
        header,
        dbc.Tabs([
            tab_sim,
            tab_pos,
            tab_strat,
            tab_port
        ], id="tabs")
    ], fluid=True, className="p-4")

    return layout
