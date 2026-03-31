import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple
from portfolio.results import PortfolioResult

def run_portfolio_backtest(
    prices_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    rebalance_freq: int = 21,          # Rebalance every 21 trading days (~monthly)
    cost_per_trade: float = 0.0005,    # 5 bps per trade (IBKR is cheap)
    initial_capital: float = 7000.0,
    monthly_contribution: float = 200.0,
    rf: float = 0.045
) -> PortfolioResult:

    n_days = len(prices_df)
    tickers = prices_df.columns.tolist()
    n_assets = len(tickers)
    dates = prices_df.index.to_numpy()
    
    # State tracking
    holdings = np.zeros(n_assets)      # shares held per asset
    cash = initial_capital
    
    # Daily output arrays
    equity_curve = np.zeros(n_days)
    cash_curve = np.zeros(n_days)
    actual_weights_history = np.zeros((n_days, n_assets))
    total_costs = 0.0
    
    # Trade log
    rebalance_dates = []
    turnover_per_rebalance = []
    
    # Total capital invested (for TWR calculation)
    total_invested = initial_capital
    
    for t in range(n_days):
        current_prices = prices_df.iloc[t].values.astype(float)
        
        # Portfolio value before any rebalance
        position_values = holdings * current_prices
        portfolio_value = position_values.sum() + cash
        
        # Rebalance check
        is_rebalance_day = (t > 0 and t % rebalance_freq == 0)
        
        if is_rebalance_day:
            # Add monthly contribution
            cash += monthly_contribution
            total_invested += monthly_contribution
            portfolio_value += monthly_contribution
            
            # Get target weights
            target_weights = weights_df.iloc[t].values.astype(float)
            
            # Target allocations in EUR
            target_allocations = portfolio_value * target_weights
            current_allocations = holdings * current_prices
            
            # Trades needed
            trade_values = target_allocations - current_allocations
            
            # Transaction costs
            trade_cost = np.sum(np.abs(trade_values)) * cost_per_trade
            total_costs += trade_cost
            cash -= trade_cost
            portfolio_value -= trade_cost
            
            # Recompute target allocations after cost
            target_allocations = portfolio_value * target_weights
            
            # Update holdings
            for j in range(n_assets):
                if current_prices[j] > 0 and target_weights[j] > 0:
                    holdings[j] = target_allocations[j] / current_prices[j]
                else:
                    holdings[j] = 0.0
            
            # Cash = whatever isn't allocated (from cash-like unallocated weight)
            allocated_value = np.sum(holdings * current_prices)
            cash = portfolio_value - allocated_value
            
            # Track turnover
            turnover = np.sum(np.abs(trade_values)) / portfolio_value if portfolio_value > 0 else 0
            turnover_per_rebalance.append(turnover)
            rebalance_dates.append(dates[t])
        
        # Record daily state
        position_values = holdings * current_prices
        total_value = position_values.sum() + cash
        equity_curve[t] = total_value
        cash_curve[t] = cash
        
        if total_value > 0:
            actual_weights_history[t] = position_values / total_value
    
    # === Summary Statistics ===
    
    # Daily returns of the portfolio (for Sharpe, vol, etc.)
    daily_returns = np.zeros(n_days)
    for t in range(1, n_days):
        if equity_curve[t - 1] > 0:
            # Subtract contribution to get pure investment return
            contribution = monthly_contribution if (t % rebalance_freq == 0 and t > 0) else 0
            adjusted_prev = equity_curve[t - 1] + contribution
            daily_returns[t] = equity_curve[t] / adjusted_prev - 1.0
    
    # CAGR
    years = (n_days - 1) / 252.0
    
    # Simple approach: total portfolio value vs total invested
    total_return = equity_curve[-1] / total_invested - 1.0 if total_invested > 0 else 0.0
    
    # TWRR approximation using daily returns
    equity_factor = np.prod(1.0 + daily_returns[1:])
    twrr_total = equity_factor - 1.0
    cagr = (equity_factor ** (1.0 / years) - 1.0) if years > 0 and equity_factor > 0 else 0.0
    
    # Volatility
    daily_vol = np.std(daily_returns[1:], ddof=1)
    volatility = daily_vol * np.sqrt(252)
    
    # Sharpe
    daily_rf = rf / 252.0
    excess = daily_returns[1:] - daily_rf
    sharpe = (np.mean(excess) / np.std(excess, ddof=1)) * np.sqrt(252) if np.std(excess, ddof=1) > 0 else 0.0
    
    # Max drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = np.where(running_max > 0, (running_max - equity_curve) / running_max, 0)
    max_drawdown = np.max(drawdowns)
    
    # Monthly returns
    monthly_returns = _compute_monthly_returns(equity_curve, dates, monthly_contribution, rebalance_freq)
    
    # Average turnover
    avg_turnover = np.mean(turnover_per_rebalance) if turnover_per_rebalance else 0.0
    
    # Per-asset return attribution
    asset_returns = {}
    for j, ticker in enumerate(tickers):
        start_price = prices_df.iloc[0].iloc[j]
        end_price = prices_df.iloc[-1].iloc[j]
        if start_price > 0:
            asset_returns[ticker] = end_price / start_price - 1.0
    
    return PortfolioResult(
        equity_curve=equity_curve,
        cash_curve=cash_curve,
        dates=dates,
        tickers=tickers,
        actual_weights_history=actual_weights_history,
        daily_returns=daily_returns,
        total_return=total_return,
        twrr_total=twrr_total,
        cagr=cagr,
        volatility=volatility,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        n_rebalances=len(rebalance_dates),
        avg_turnover=avg_turnover,
        total_costs=total_costs,
        total_invested=total_invested,
        final_value=equity_curve[-1],
        monthly_returns=monthly_returns,
        asset_returns=asset_returns
    )

def _compute_monthly_returns(equity_curve, dates, contribution, rebal_freq):
    """Compute monthly portfolio returns, adjusting for contributions."""
    import pandas as pd
    dates_pd = pd.to_datetime(dates)
    
    months = {}
    for i, d in enumerate(dates_pd):
        key = (d.year, d.month)
        if key not in months:
            months[key] = {'start': i, 'end': i}
        months[key]['end'] = i
    
    monthly = []
    for key in months:
        s = months[key]['start']
        e = months[key]['end']
        if s > 0 and e > s and equity_curve[s] > 0:
            # Count contributions in this month
            n_rebalances_in_month = sum(1 for t in range(s + 1, e + 1) if t % rebal_freq == 0)
            contrib = n_rebalances_in_month * contribution
            adjusted_start = equity_curve[s] + contrib
            if adjusted_start > 0:
                monthly.append(equity_curve[e] / adjusted_start - 1.0)
    
    return np.array(monthly)
