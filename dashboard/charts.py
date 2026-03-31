import plotly.graph_objects as go
import numpy as np
from simulation.results import SimulationResult

BG_COLOR = '#0e1117'

def _base_layout(title: str):
    return go.Layout(
        title=title,
        template='plotly_dark',
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        margin=dict(l=40, r=20, t=50, b=30),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False)
    )

def create_path_fan_chart(paths: np.ndarray) -> go.Figure:
    fig = go.Figure()
    
    n_total = paths.shape[0]
    n_days = paths.shape[1] - 1
    S0 = paths[0, 0]
    
    # Subsample paths for display to avoid browser overload
    max_display = min(300, n_total)
    sample_idx = np.random.choice(n_total, max_display, replace=False)
    paths_sub = paths[sample_idx]
    
    # Determine quartiles based on final price
    final_prices = paths_sub[:, -1]
    q25 = np.percentile(final_prices, 25)
    q75 = np.percentile(final_prices, 75)
    
    x_axis = np.arange(n_days + 1)
    
    for i in range(max_display):
        fp = final_prices[i]
        if fp >= q75:
            color = 'rgba(0, 255, 136, 0.3)' # Top 25% Green
        elif fp <= q25:
            color = 'rgba(255, 75, 75, 0.3)' # Bottom 25% Red
        else:
            color = 'rgba(150, 150, 150, 0.15)' # Middle 50% Grey
            
        fig.add_trace(go.Scatter(
            x=x_axis, y=paths_sub[i], 
            mode='lines', 
            line=dict(color=color, width=1), 
            hoverinfo='skip',
            showlegend=False
        ))
        
    # Percentiles over all paths (not just sample)
    p5 = np.percentile(paths, 5, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    
    fig.add_trace(go.Scatter(x=x_axis, y=p50, mode='lines', line=dict(color='white', width=2.5), name='Median', hoverinfo='x+y'))
    fig.add_trace(go.Scatter(x=x_axis, y=p95, mode='lines', line=dict(color='#00ff88', width=2.5), name='95th Pctl', hoverinfo='x+y'))
    fig.add_trace(go.Scatter(x=x_axis, y=p5, mode='lines', line=dict(color='#ff4b4b', width=2.5), name='5th Pctl', hoverinfo='x+y'))
    
    # S0 dashed line
    fig.add_hline(y=S0, line_dash="dash", line_color="yellow")
    
    fig.update_layout(_base_layout("Monte Carlo Price Paths"))
    fig.update_yaxes(title="Price")
    fig.update_xaxes(title="Days")
    
    return fig

def create_return_distribution(res: SimulationResult) -> go.Figure:
    fig = go.Figure()
    
    returns = res.log_returns_total
    mean_ret = np.mean(returns)
    
    fig.add_trace(go.Histogram(
        x=returns, nbinsx=60, 
        marker_color='rgba(74, 158, 255, 0.7)',
        name='Log Returns',
        showlegend=False
    ))
    
    fig.add_vrect(
        x0=np.min(returns), x1=res.var_95,
        fillcolor="red", opacity=0.15, layer="below", line_width=0
    )
    
    fig.add_vline(x=mean_ret, line_dash="dash", line_color="white", annotation_text="Mean", annotation_y=0.85)
    fig.add_vline(x=res.var_95, line_dash="dash", line_color="#ff4b4b", annotation_text="VaR 95%", annotation_y=0.75)
    fig.add_vline(x=res.var_99, line_dash="dash", line_color="#ff0000", annotation_text="VaR 99%", annotation_y=0.65)
    fig.add_vline(x=res.cvar_95, line_dash="dash", line_color="#ff8c00", annotation_text="CVaR 95%", annotation_y=0.55)

    fig.update_layout(_base_layout("Total Return Distribution"))
    fig.update_xaxes(title="Total Log Return")
    
    return fig

def create_price_histogram(res: SimulationResult) -> go.Figure:
    fig = go.Figure()
    
    prices = res.final_prices
    
    # Calculate S0 based on end logic
    S0 = res.final_prices[0] / np.exp(res.log_returns_total[0])
    
    mean_px = np.mean(prices)
    median_px = np.median(prices)
    var95_px = S0 * np.exp(res.var_95)
    
    fig.add_trace(go.Histogram(
        x=prices, nbinsx=60,
        marker_color='#a78bfa',
        name='Final Prices',
        showlegend=False
    ))
    
    fig.add_vline(x=mean_px, line_color="white", annotation_text=f"Mean: {mean_px:.2f}", annotation_y=0.85)
    fig.add_vline(x=median_px, line_color="cyan", annotation_text=f"Median: {median_px:.2f}", annotation_y=0.75)
    fig.add_vline(x=var95_px, line_color="red", annotation_text=f"VaR 95%: {var95_px:.2f}", annotation_y=0.65)
    
    fig.update_layout(_base_layout("Final Price Distribution"))
    fig.update_xaxes(title="Price")
    
    return fig

def create_euro_pl_distribution(euro_pl: np.ndarray, median: float, var95: float, cvar95: float) -> go.Figure:
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=euro_pl, nbinsx=60,
        marker_color='rgba(74, 158, 255, 0.7)',
        name='1M Euro P&L',
        showlegend=False
    ))
    
    # Shade loss region (left of zero) red, 15% alpha
    fig.add_vrect(
        x0=np.min(euro_pl), x1=0,
        fillcolor="red", opacity=0.15, layer="below", line_width=0
    )
    
    # Vertical lines
    fig.add_vline(x=0, line_dash="dash", line_color="white", annotation_text="Break-even", annotation_y=0.85)
    fig.add_vline(x=median, line_color="cyan", annotation_text=f"Median: €{median:.2f}", annotation_y=0.75)
    fig.add_vline(x=var95, line_dash="dash", line_color="#ff4b4b", annotation_text=f"VaR 95%: €{var95:.2f}", annotation_y=0.65)
    fig.add_vline(x=cvar95, line_dash="dash", line_color="#ff8c00", annotation_text=f"CVaR 95%: €{cvar95:.2f}", annotation_y=0.55)
    
    fig.update_layout(_base_layout("1-Month P&L Distribution (€)"))
    fig.update_xaxes(title="Euro P&L (€)")
    
    return fig

def create_scenario_fan_chart(horizons: list, p5: list, p25: list, p50: list, p75: list, p95: list, total_loss: float) -> go.Figure:
    fig = go.Figure()
    
    # Plotly fill='tonexty' fills from the PREVIOUS trace up to the CURRENT trace.
    # So we plot from bottom to top: p5 -> p25 -> p50 -> p75 -> p95
    
    fig.add_trace(go.Scatter(
        x=horizons, y=p5, mode='lines', line=dict(color='#ff4b4b', width=2),
        name='5th Pctl'
    ))
    fig.add_trace(go.Scatter(
        x=horizons, y=p25, mode='lines', line=dict(color='rgba(255, 75, 75, 0.5)', width=1),
        name='25th Pctl', fill='tonexty', fillcolor='rgba(255, 75, 75, 0.1)'
    ))
    fig.add_trace(go.Scatter(
        x=horizons, y=p50, mode='lines', line=dict(color='white', width=3),
        name='Median', fill='tonexty', fillcolor='rgba(150, 150, 150, 0.1)'
    ))
    fig.add_trace(go.Scatter(
        x=horizons, y=p75, mode='lines', line=dict(color='rgba(0, 255, 136, 0.5)', width=1),
        name='75th Pctl', fill='tonexty', fillcolor='rgba(0, 255, 136, 0.1)'
    ))
    fig.add_trace(go.Scatter(
        x=horizons, y=p95, mode='lines', line=dict(color='#00ff88', width=2),
        name='95th Pctl', fill='tonexty', fillcolor='rgba(0, 255, 136, 0.1)'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="white", annotation_text="Break-even", annotation_position="top right")
    fig.add_hline(y=total_loss, line_dash="dash", line_color="red", annotation_text="Total Loss", annotation_position="bottom right")
    
    fig.update_layout(_base_layout("Position P&L Scenarios Over Time (€)"))
    fig.update_xaxes(title="Trading Days", tickvals=horizons, ticktext=["1M", "3M", "6M", "12M"])
    fig.update_yaxes(title="Euro P&L (€)")
    
    return fig

def create_time_to_target_bar_chart(targets: list, months: list) -> go.Figure:
    fig = go.Figure()
    
    # Cap visualization at 120 so bars don't grow to infinity visually throwing off scale
    capped_months = [min(m, 120) for m in months]
    text = [f"{m:.1f} mo" if m < float('inf') else "Out of reach" for m in months]
    
    fig.add_trace(go.Bar(
        x=[f"€{t}" for t in targets],
        y=capped_months,
        text=text,
        textposition='auto',
        marker_color='rgba(0, 255, 136, 0.7)'
    ))
    
    fig.update_layout(_base_layout("Months to Target Monthly Income"))
    fig.update_xaxes(title="Monthly Income Target")
    fig.update_yaxes(title="Months")
    
    return fig

def create_equity_curve_chart(bt_res, bh_res) -> go.Figure:
    fig = go.Figure()
    
    x_axis = bt_res.dates if bt_res.dates is not None else np.arange(len(bt_res.equity_curve))
    
    # Shade regions where signal is 1
    # We can use vrect or fill between. Plotting filled area is easiest if we define signal regions.
    # To keep it performant, we find contiguous blocks of signal==1
    signals = bt_res.signals
    
    if len(signals) > 0 and type(x_axis[0]) is not np.int64:
        # It's dates
        in_market = False
        start_idx = None
        for i in range(len(signals)):
            if signals[i] > 0 and not in_market:
                in_market = True
                start_idx = i
            elif signals[i] == 0 and in_market:
                in_market = False
                fig.add_vrect(x0=x_axis[start_idx], x1=x_axis[i], fillcolor="green", opacity=0.1, layer="below", line_width=0)
        if in_market:  # close out to end
            fig.add_vrect(x0=x_axis[start_idx], x1=x_axis[-1], fillcolor="green", opacity=0.1, layer="below", line_width=0)
    
    fig.add_trace(go.Scatter(
        x=x_axis, y=bt_res.equity_curve,
        name='Strategy', mode='lines', line=dict(color='#00ff88', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=x_axis, y=bh_res.equity_curve,
        name='Buy & Hold', mode='lines', line=dict(color='white', width=1.5, dash='dash')
    ))
    
    fig.update_layout(_base_layout("Historical Equity Curve"))
    fig.update_yaxes(title="Normalised Equity (Starts at 1.0)")
    
    return fig

def create_mc_strategy_fan_chart(mc_res) -> go.Figure:
    fig = go.Figure()
    
    n_days = len(mc_res.pctile_50) - 1
    x_axis = np.arange(n_days + 1)
    
    # Reusing the 'tonexty' fill layering bottom to top
    fig.add_trace(go.Scatter(x=x_axis, y=mc_res.pctile_5, mode='lines', line=dict(color='#ff4b4b', width=2), name='5th Pctl'))
    fig.add_trace(go.Scatter(x=x_axis, y=mc_res.pctile_25, mode='lines', line=dict(color='rgba(255, 75, 75, 0.5)', width=1), name='25th Pctl', fill='tonexty', fillcolor='rgba(255, 75, 75, 0.1)'))
    fig.add_trace(go.Scatter(x=x_axis, y=mc_res.pctile_50, mode='lines', line=dict(color='#00ff88', width=3), name='Strategy Median', fill='tonexty', fillcolor='rgba(150, 150, 150, 0.1)'))
    fig.add_trace(go.Scatter(x=x_axis, y=mc_res.pctile_75, mode='lines', line=dict(color='rgba(0, 255, 136, 0.5)', width=1), name='75th Pctl', fill='tonexty', fillcolor='rgba(0, 255, 136, 0.1)'))
    fig.add_trace(go.Scatter(x=x_axis, y=mc_res.pctile_95, mode='lines', line=dict(color='#00ff88', width=2), name='95th Pctl', fill='tonexty', fillcolor='rgba(0, 255, 136, 0.1)'))
    
    fig.update_layout(_base_layout("Monte Carlo Strategy Envelope (10k Paths)"))
    fig.update_yaxes(title="Normalised Equity")
    fig.update_xaxes(title="Trading Days")
    
    return fig

def create_monthly_return_hist(bt_res, mc_res) -> go.Figure:
    fig = go.Figure()
    
    hist_monthly = bt_res.monthly_returns
    
    fig.add_trace(go.Histogram(
        x=hist_monthly, nbinsx=40,
        marker_color='rgba(74, 158, 255, 0.8)',
        name='Historical Months',
        histnorm='probability'
    ))
    
    if len(mc_res.monthly_returns_median) > 0:
        fig.add_trace(go.Histogram(
            x=mc_res.monthly_returns_median, nbinsx=40,
            marker_color='rgba(0, 255, 136, 0.5)',
            name='MC Median Months',
            histnorm='probability'
        ))
        
    hist_var95 = np.percentile(hist_monthly, 5) if len(hist_monthly) > 0 else 0
    fig.add_vline(x=hist_var95, line_dash="dash", line_color="#ff4b4b", annotation_text=f"VaR 95%: {hist_var95*100:.1f}%", annotation_y=0.85)

    fig.update_layout(_base_layout("Monthly Return Distribution"))
    fig.update_layout(barmode='overlay')
    fig.update_xaxes(title="Monthly Return", tickformat='.1%')
    
    return fig

def create_portfolio_equity_chart(res, initial_cap, monthly_contrib, rebal_freq) -> go.Figure:
    fig = go.Figure()
    
    invested = np.zeros(len(res.dates))
    invested[0] = initial_cap
    for t in range(1, len(res.dates)):
        invested[t] = invested[t-1]
        if t % rebal_freq == 0:
            invested[t] += monthly_contrib
            
    fig.add_trace(go.Scatter(
        x=res.dates, y=res.equity_curve,
        name='Portfolio Value', mode='lines', line=dict(color='#00ff88', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=res.dates, y=invested,
        name='Total Invested', mode='lines', line=dict(color='white', width=1.5, dash='dash')
    ))
    
    fig.update_layout(_base_layout("Portfolio Equity Curve (€)"))
    fig.update_yaxes(title="Portfolio Value (€)")
    return fig

def create_portfolio_alloc_chart(res) -> go.Figure:
    fig = go.Figure()
    
    for i, t in enumerate(res.tickers):
        weights = res.actual_weights_history[:, i]
        if np.max(weights) > 0.001:
            fig.add_trace(go.Scatter(
                x=res.dates, y=weights,
                name=t,
                mode='none',
                stackgroup='one'
            ))
            
    allocated = np.sum(res.actual_weights_history, axis=1)
    cash_weight = 1.0 - allocated
    cash_weight = np.clip(cash_weight, 0, 1)
    
    fig.add_trace(go.Scatter(
        x=res.dates, y=cash_weight,
        name="Cash",
        mode='none',
        stackgroup='one',
        fillcolor='rgba(150, 150, 150, 0.3)'
    ))
    
    fig.update_layout(_base_layout("Asset Allocation Over Time"))
    fig.update_yaxes(title="Weight", range=[0, 1], tickformat='.0%')
    return fig

def create_portfolio_asset_returns_chart(res) -> go.Figure:
    fig = go.Figure()
    
    sorted_ret = dict(sorted(res.asset_returns.items(), key=lambda item: item[1]))
    
    from portfolio.universe import get_universe_info
    info = {t[0]: t[2] for t in get_universe_info()}
    
    colors = []
    for t in sorted_ret.keys():
        cls = info.get(t, 'equity')
        if cls == 'equity': colors.append('#00ff88')
        elif cls == 'bond': colors.append('#4a9eff')
        elif cls == 'commodity': colors.append('#ffd700')
        else: colors.append('#a78bfa')
        
    fig.add_trace(go.Bar(
        y=list(sorted_ret.keys()),
        x=list(sorted_ret.values()),
        orientation='h',
        marker_color=colors,
        text=[f"{v*100:.1f}%" for v in sorted_ret.values()],
        textposition='auto'
    ))
    
    fig.update_layout(_base_layout("Asset Total Returns"))
    fig.update_xaxes(tickformat='.0%')
    return fig

def create_portfolio_monthly_hist(res) -> go.Figure:
    fig = go.Figure()
    
    colors = ['#00ff88' if r >= 0 else '#ff4b4b' for r in res.monthly_returns]
    x_axis = list(range(1, len(res.monthly_returns) + 1))
    
    fig.add_trace(go.Bar(
        x=x_axis,
        y=res.monthly_returns,
        marker_color=colors,
        name='Monthly Return'
    ))
    
    if len(res.monthly_returns) >= 12:
        rolling = []
        for i in range(len(res.monthly_returns)):
            if i < 11:
                rolling.append(0)
            else:
                r12 = 1.0
                for j in range(12):
                    r12 *= (1.0 + res.monthly_returns[i-j])
                rolling.append(r12 - 1.0)
        
        fig.add_trace(go.Scatter(
            x=x_axis, y=rolling,
            name='Rolling 12M', mode='lines', line=dict(color='white', width=2)
        ))
    
    fig.update_layout(_base_layout("Monthly Portfolio Returns"))
    fig.update_yaxes(tickformat='.1%')
    fig.update_xaxes(title="Month")
    return fig
