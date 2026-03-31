from dash import Input, Output, State, callback
from data.fetcher import fetch_data, fetch_etc_price
from data.calibrator import calibrate
from simulation.engine import simulate
from dashboard.charts import create_path_fan_chart, create_return_distribution, create_price_histogram

# Fetch globally for position tab
etc_price = fetch_etc_price()

@callback(
    Output("student-t-controls", "style"),
    Output("jump-controls", "style"),
    Input("dropdown-mode", "value")
)
def toggle_controls(mode):
    if mode == "student_t":
        return {'display': 'block'}, {'display': 'none'}
    elif mode == "jump_diffusion":
        return {'display': 'none'}, {'display': 'block'}
    else: # gbm
        return {'display': 'none'}, {'display': 'none'}

@callback(
    Output("graph-fan", "figure"),
    Output("graph-return-dist", "figure"),
    Output("graph-price-hist", "figure"),
    Output("table-metrics", "data"),
    Output("table-metrics", "columns"),
    Output("ticker-badge", "children"),
    Input("btn-run", "n_clicks"),
    State("dropdown-ticker", "value"),
    State("dropdown-mode", "value"),
    State("slider-paths", "value"),
    State("slider-horizon", "value"),
    State("slider-df", "value"),
    State("input-lambda", "value"),
    State("input-mu-j", "value"),
    State("input-sigma-j", "value"),
    prevent_initial_call=False
)
def update_dashboard(n_clicks, ticker, mode, n_paths, n_days, df_val, lambda_val, mu_j, sigma_j):
    
    # We allow the initial call to run so dashboard is populated immediately.
    df = fetch_data(ticker)
    cal = calibrate(df)
    
    kwargs = {}
    if mode == "student_t":
        kwargs['df'] = df_val
    elif mode == "jump_diffusion":
        kwargs['lambda_'] = lambda_val
        kwargs['mu_j'] = mu_j
        kwargs['sigma_j'] = sigma_j
        
    paths, sim_res = simulate(cal, S0=None, n_paths=n_paths, n_days=n_days, mode=mode, seed=42, **kwargs)
    
    fig_fan = create_path_fan_chart(paths)
    fig_ret = create_return_distribution(sim_res)
    fig_hist = create_price_histogram(sim_res)
    
    table_data = [
        {"Metric": "CAGR Mean", "Value": f"{sim_res.cagr_mean * 100:.2f}%"},
        {"Metric": "CAGR Median", "Value": f"{sim_res.cagr_median * 100:.2f}%"},
        {"Metric": "CAGR Std", "Value": f"{sim_res.cagr_std * 100:.2f}%"},
        {"Metric": "Sharpe Ratio", "Value": f"{sim_res.sharpe_ratio:.2f}"},
        {"Metric": "Max Drawdown Mean", "Value": f"{sim_res.max_drawdown_mean * 100:.2f}%"},
        {"Metric": "VaR 95%", "Value": f"{sim_res.var_95 * 100:.2f}%"},
        {"Metric": "VaR 99%", "Value": f"{sim_res.var_99 * 100:.2f}%"},
        {"Metric": "CVaR 95%", "Value": f"{sim_res.cvar_95 * 100:.2f}%"},
        {"Metric": "CVaR 99%", "Value": f"{sim_res.cvar_99 * 100:.2f}%"},
    ]
    cols = [{"name": "Metric", "id": "Metric"}, {"name": "Value", "id": "Value"}]
    
    badge_text = f"{ticker} · ${cal.last_price:.2f}"
    
    return fig_fan, fig_ret, fig_hist, table_data, cols, badge_text

from dashboard.position import (
    get_position_metrics,
    run_monthly_simulation,
    run_horizon_scenarios,
    covered_call_metrics,
    months_to_target
)
from dashboard.charts import (
    create_euro_pl_distribution,
    create_scenario_fan_chart,
    create_time_to_target_bar_chart
)
import numpy as np

@callback(
    Output("pos-entry-val", "children"),
    Output("pos-curr-val", "children"),
    Output("pos-units-val", "children"),
    Output("pos-val-val", "children"),
    Output("pos-upl-val", "children"),
    Output("pos-perc-val", "children"),
    Output("pos-dry-val", "children"),
    Output("pos-curr-val", "style"),
    Output("pos-val-val", "style"),
    Output("pos-upl-val", "style"),
    Output("pos-perc-val", "style"),
    Output("graph-pos-euro-dist", "figure"),
    Output("graph-pos-scenarios", "figure"),
    Output("input-cc-portfolio", "value"),
    Input("input-pos-entry", "value"),
    Input("input-pos-units", "value"),
    prevent_initial_call=False
)
def update_position_tab(entry_price, units):
    if entry_price is None or units is None:
        import dash
        raise dash.exceptions.PreventUpdate
        
    df = fetch_data("KC=F")
    cal = calibrate(df)
    
    current_price = etc_price if etc_price is not None else cal.last_price
    metrics = get_position_metrics(entry_price, units, current_price)
    
    entry_text = f"€{entry_price:.2f}"
    curr_text = f"€{current_price:.2f}"
    units_text = f"{units}"
    val_text = f"€{metrics['position_value']:.2f}"
    upl_text = f"€{metrics['unrealised_pl']:.2f}"
    perc_text = f"{metrics['pl_percent']:.2f}%"
    dry_text = f"€{metrics['dry_powder']:.2f}"
    
    green = {'color': '#00ff88'}
    red = {'color': '#ff4b4b'}
    
    is_profit = current_price >= entry_price
    color_style = green if is_profit else red
    
    monthly = run_monthly_simulation(cal, entry_price, units, current_price)
    fig_euro_dist = create_euro_pl_distribution(
        monthly['euro_pl'], monthly['median_euro'], 
        monthly['var_95_euro'], monthly['cvar_95_euro']
    )
    
    horizons_data = run_horizon_scenarios(cal, entry_price, units, current_price)
    fig_scenarios = create_scenario_fan_chart(
        horizons_data['horizons'],
        horizons_data[5], horizons_data[25], horizons_data[50], 
        horizons_data[75], horizons_data[95],
        total_loss=-(entry_price * units)
    )
    
    port_val = metrics['position_value']
    
    return (
        entry_text, curr_text, units_text, val_text, upl_text, perc_text, dry_text,
        color_style, color_style, color_style, color_style,
        fig_euro_dist, fig_scenarios, port_val
    )

@callback(
    Output("cc-est-monthly", "children"),
    Output("cc-est-annual", "children"),
    Output("table-cc", "data"),
    Output("table-cc", "columns"),
    Output("graph-cc-months", "figure"),
    Input("dropdown-cc-target", "value"),
    Input("slider-cc-rate", "value"),
    Input("input-cc-portfolio", "value"),
    Input("slider-cc-savings", "value"),
    prevent_initial_call=False
)
def update_cc_calculator(target_income, premium_rate, portfolio_value, monthly_savings):
    if portfolio_value is None or premium_rate is None:
        import dash
        raise dash.exceptions.PreventUpdate
        
    cc_metrics = covered_call_metrics(portfolio_value, premium_rate)
    
    est_monthly = f"€{cc_metrics['monthly_premium']:.2f}/mo"
    est_annual = f"Annual: €{cc_metrics['annual_premium']:.2f}"
    
    df = fetch_data("KC=F")
    cal = calibrate(df)
    cagr_median = np.exp(cal.mu_gbm * 252) - 1.0
    
    targets = cc_metrics['targets']
    req_cap = cc_metrics['required_capital']
    
    table_data = []
    
    curr_t = portfolio_value * premium_rate
    base_price = etc_price if etc_price is not None else cal.last_price
    table_data.append({
        "Monthly Target": f"€{curr_t:.2f} (Current)",
        "Capital Required": f"€{portfolio_value:,.2f}",
        "Units needed": f"~{portfolio_value/base_price:.1f} units"
    })
    
    for t in targets:
        cap = req_cap[t]
        u_need = cap / base_price
        table_data.append({
            "Monthly Target": f"€{t}/month",
            "Capital Required": f"€{cap:,.2f}",
            "Units needed": f"~{int(np.ceil(u_need))} units"
        })
        
    cols = [{"name": i, "id": i} for i in ["Monthly Target", "Capital Required", "Units needed"]]
    
    months_list = []
    for t in targets:
        m = months_to_target(portfolio_value, monthly_savings, t, premium_rate, cagr_median)
        months_list.append(m)
        
    fig_months = create_time_to_target_bar_chart(targets, months_list)
    
    return est_monthly, est_annual, table_data, cols, fig_months

from strategy.rules import STRATEGY_REGISTRY, buy_and_hold
from strategy.backtester import run_backtest
from strategy.mc_strategy import run_mc_strategy
from dashboard.charts import (
    create_equity_curve_chart,
    create_mc_strategy_fan_chart,
    create_monthly_return_hist
)
from datetime import datetime
from dateutil.relativedelta import relativedelta

@callback(
    Output("strategy-equity-chart", "figure"),
    Output("strategy-stats-table", "data"),
    Output("strategy-stats-table", "columns"),
    Output("mc-strategy-fan", "figure"),
    Output("mc-strategy-cards", "children"),
    Output("monthly-return-hist", "figure"),
    Input("run-strategy-btn", "n_clicks"),
    State("strategy-dropdown", "value"),
    State("strategy-ticker", "value"),
    State("strategy-period", "value"),
    State("strategy-cost-slider", "value"),
    prevent_initial_call=False
)
def run_strategy_tab(n_clicks, strategy_name, ticker, period, cost_bps):
    import dash
    from dash import html
    
    cost = cost_bps / 10000.0
    
    entry = STRATEGY_REGISTRY[strategy_name]
    signal_fn = entry["fn"]
    signal_params = entry["params"]
    
    end_date = datetime.today().strftime('%Y-%m-%d')
    years = int(period[0]) # '1Y', '2Y' etc
    start_date = (datetime.today() - relativedelta(years=years)).strftime('%Y-%m-%d')
    
    df = fetch_data(ticker, start_date=start_date, end_date=end_date)
    if len(df) < 50:
        raise dash.exceptions.PreventUpdate
        
    prices = df['Close'].values
    dates = df.index.values
    
    bt_result = run_backtest(prices, signal_fn, signal_params, cost, dates=dates)
    bh_result = run_backtest(prices, buy_and_hold, {}, 0.0, dates=dates)
    
    cal = calibrate(df)
    # Important: use starting S0 = prices[-1] for MC if going forward from now
    # Wait, for MC strategy, S0 is usually derived inside simulate() from cal.last_price natively.
    paths, _ = simulate(cal, S0=cal.last_price, n_paths=10000, n_days=252, mode="gbm", seed=42)
    
    mc_result = run_mc_strategy(paths, signal_fn, signal_params, cost)
    
    fig_equity = create_equity_curve_chart(bt_result, bh_result)
    fig_mc_fan = create_mc_strategy_fan_chart(mc_result)
    fig_hist = create_monthly_return_hist(bt_result, mc_result)
    
    table_data = [
        {"Metric": "CAGR", "Strategy": f"{bt_result.cagr*100:.2f}%", "Buy & Hold": f"{bh_result.cagr*100:.2f}%"},
        {"Metric": "Volatility", "Strategy": f"{bt_result.volatility*100:.2f}%", "Buy & Hold": f"{bh_result.volatility*100:.2f}%"},
        {"Metric": "Sharpe", "Strategy": f"{bt_result.sharpe:.2f}", "Buy & Hold": f"{bh_result.sharpe:.2f}"},
        {"Metric": "Max Drawdown", "Strategy": f"{bt_result.max_drawdown*100:.2f}%", "Buy & Hold": f"{bh_result.max_drawdown*100:.2f}%"},
        {"Metric": "# Trades", "Strategy": f"{bt_result.n_trades}", "Buy & Hold": "—"},
        {"Metric": "Win Rate", "Strategy": f"{bt_result.win_rate*100:.1f}%", "Buy & Hold": "—"},
        {"Metric": "Time in Market", "Strategy": f"{bt_result.time_in_market*100:.1f}%", "Buy & Hold": f"{bh_result.time_in_market*100:.1f}%"},
        {"Metric": "Total Tx Costs", "Strategy": f"{bt_result.total_costs*100:.2f}%", "Buy & Hold": "0.00%"},
    ]
    cols = [{"name": i, "id": i} for i in ["Metric", "Strategy", "Buy & Hold"]]
    
    def _strat_card(title, val, fmt):
        color = '#00ff88' if (val > 0 and title != 'Median Max DD') else ('white' if val == 0 else '#ff4b4b')
        if title == 'Median Max DD':
            color = 'white'
        return html.Div([
            html.H6(title, className="text-muted", style={'fontSize': '0.85rem', 'marginBottom': '2px'}),
            html.H4(fmt.format(val), style={'fontFamily': 'monospace', 'color': color})
        ], className="mb-3")
        
    mc_cards = html.Div([
        _strat_card("Median CAGR", mc_result.median_cagr*100, "{:.2f}%"),
        _strat_card("Median Sharpe", mc_result.median_sharpe, "{:.2f}"),
        _strat_card("Median Max DD", mc_result.median_max_dd*100, "{:.2f}%"),
        _strat_card("P(Profit)", mc_result.prob_profit*100, "{:.1f}%"),
        _strat_card("P(Beat B&H)", mc_result.prob_beat_buyhold*100, "{:.1f}%")
    ])
    
    return fig_equity, table_data, cols, fig_mc_fan, mc_cards, fig_hist

from portfolio import (
    fetch_universe_prices,
    cross_sectional_momentum,
    dual_momentum,
    inverse_volatility_weights,
    equal_weights,
    run_portfolio_backtest
)
from dashboard.charts import (
    create_portfolio_equity_chart,
    create_portfolio_alloc_chart,
    create_portfolio_asset_returns_chart,
    create_portfolio_monthly_hist
)

@callback(
    Output("portfolio-equity-chart", "figure"),
    Output("portfolio-stats-table", "data"),
    Output("portfolio-stats-table", "columns"),
    Output("portfolio-alloc-chart", "figure"),
    Output("portfolio-asset-returns-chart", "figure"),
    Output("portfolio-monthly-hist", "figure"),
    Output("portfolio-benchmark-table", "data"),
    Output("portfolio-benchmark-table", "columns"),
    Input("run-portfolio-btn", "n_clicks"),
    State("portfolio-strategy", "value"),
    State("portfolio-lookback", "value"),
    State("portfolio-top-n", "value"),
    State("portfolio-sizing", "value"),
    State("portfolio-period", "value"),
    State("portfolio-capital", "value"),
    State("portfolio-contribution", "value"),
    prevent_initial_call=False
)
def run_portfolio(n_clicks, strategy, lookback, top_n, sizing, period, capital, contribution):
    import pandas as pd
    
    prices_df = fetch_universe_prices(period)
    if len(prices_df) == 0:
        import dash
        raise dash.exceptions.PreventUpdate
        
    if strategy == "dual_momentum":
        signals_df = dual_momentum(prices_df, lookback, top_n)
    elif strategy == "cs_momentum":
        signals_df = cross_sectional_momentum(prices_df, lookback, top_n)
    else: 
        signals_df = pd.DataFrame(1.0, index=prices_df.index, columns=prices_df.columns)
        
    if sizing == "inverse_vol":
        weights_df = inverse_volatility_weights(prices_df, signals_df)
    else:
        weights_df = equal_weights(signals_df)
        
    rebal_freq = 21
    res = run_portfolio_backtest(
        prices_df, weights_df,
        rebalance_freq=rebal_freq,
        cost_per_trade=0.0005,
        initial_capital=capital,
        monthly_contribution=contribution,
        rf=0.045
    )
    
    spy_weights = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    if 'SPY' in prices_df.columns: spy_weights['SPY'] = 1.0
    spy_res = run_portfolio_backtest(prices_df, spy_weights, initial_capital=capital, monthly_contribution=contribution)
    
    bal_weights = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    if 'SPY' in prices_df.columns: bal_weights['SPY'] = 0.6
    if 'IEF' in prices_df.columns: bal_weights['IEF'] = 0.4
    bal_res = run_portfolio_backtest(prices_df, bal_weights, initial_capital=capital, monthly_contribution=contribution)
    
    fig_equity = create_portfolio_equity_chart(res, capital, contribution, rebal_freq)
    fig_alloc = create_portfolio_alloc_chart(res)
    fig_returns = create_portfolio_asset_returns_chart(res)
    fig_hist = create_portfolio_monthly_hist(res)
    
    stats_data = [
        {"Metric": "Final Value", "Value": f"€{res.final_value:,.2f}"},
        {"Metric": "Total Invested", "Value": f"€{res.total_invested:,.2f}"},
        {"Metric": "Total Return", "Value": f"{res.total_return*100:.1f}%"},
        {"Metric": "CAGR (TWRR)", "Value": f"{res.cagr*100:.1f}%"},
        {"Metric": "Volatility", "Value": f"{res.volatility*100:.1f}%"},
        {"Metric": "Sharpe Ratio", "Value": f"{res.sharpe:.2f}"},
        {"Metric": "Max Drawdown", "Value": f"{res.max_drawdown*100:.1f}%"},
        {"Metric": "Avg Monthly Turnover", "Value": f"{res.avg_turnover*100:.1f}%"},
        {"Metric": "Total Tx Costs", "Value": f"€{res.total_costs:,.2f}"},
        {"Metric": "Rebalances", "Value": f"{res.n_rebalances}"}
    ]
    stats_cols = [{"name": "Metric", "id": "Metric"}, {"name": "Value", "id": "Value"}]
    
    bench_data = [
        {"Metric": "CAGR", "Portfolio": f"{res.cagr*100:.1f}%", "SPY B&H": f"{spy_res.cagr*100:.1f}%", "60/40": f"{bal_res.cagr*100:.1f}%"},
        {"Metric": "Volatility", "Portfolio": f"{res.volatility*100:.1f}%", "SPY B&H": f"{spy_res.volatility*100:.1f}%", "60/40": f"{bal_res.volatility*100:.1f}%"},
        {"Metric": "Sharpe", "Portfolio": f"{res.sharpe:.2f}", "SPY B&H": f"{spy_res.sharpe:.2f}", "60/40": f"{bal_res.sharpe:.2f}"},
        {"Metric": "Max Drawdown", "Portfolio": f"{res.max_drawdown*100:.1f}%", "SPY B&H": f"{spy_res.max_drawdown*100:.1f}%", "60/40": f"{bal_res.max_drawdown*100:.1f}%"}
    ]
    bench_cols = [{"name": i, "id": i} for i in ["Metric", "Portfolio", "SPY B&H", "60/40"]]
    
    return fig_equity, stats_data, stats_cols, fig_alloc, fig_returns, fig_hist, bench_data, bench_cols
