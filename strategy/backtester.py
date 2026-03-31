import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass
class BacktestResult:
    """Container for backtest output."""
    # Equity curve
    equity_curve: np.ndarray       # daily portfolio value, starts at 1.0
    dates: np.ndarray              # corresponding dates (if available)
    
    # Signals
    signals: np.ndarray            # daily signal array from the rule
    
    # Summary stats
    total_return: float            # cumulative return
    cagr: float                    # annualised return
    volatility: float              # annualised volatility
    sharpe: float                  # annualised Sharpe (rf=0.045)
    max_drawdown: float            # worst peak-to-trough
    n_trades: int                  # number of position changes
    win_rate: float                # % of trades that were profitable
    avg_trade_return: float        # mean return per completed trade
    time_in_market: float          # % of days with position on
    
    # Monthly P&L
    monthly_returns: np.ndarray    # array of monthly returns
    
    # Transaction costs
    total_costs: float             # cumulative transaction cost drag

def run_backtest(
    prices: np.ndarray,
    signal_fn: Callable,
    signal_params: dict = None,
    cost_per_trade: float = 0.001,  # 10 bps per trade (buy or sell)
    rf: float = 0.045,
    dates: np.ndarray = None
) -> BacktestResult:
    """
    Run a historical backtest.
    
    Args:
        prices: 1D array of daily closing prices
        signal_fn: function from rules.py (takes prices, returns signals)
        signal_params: dict of kwargs passed to signal_fn (default: {})
        cost_per_trade: proportional cost per position change (default 10 bps)
        rf: annual risk-free rate for Sharpe (default 4.5%)
        dates: optional array of dates for equity curve labelling
    
    Returns:
        BacktestResult dataclass
    """
    if signal_params is None:
        signal_params = {}
    
    signals = signal_fn(prices, **signal_params)
    
    # Daily price returns
    price_returns = np.zeros(len(prices))
    price_returns[1:] = prices[1:] / prices[:-1] - 1.0
    
    # Strategy returns = signal * price_return, minus costs on signal changes
    strategy_returns = np.zeros(len(prices))
    total_costs = 0.0
    trades = []          # list of (entry_idx, exit_idx, return) for win rate calc
    current_entry_idx = None
    
    for t in range(1, len(prices)):
        strategy_returns[t] = signals[t - 1] * price_returns[t]
        
        # Detect signal change -> apply transaction cost
        if signals[t - 1] != signals[t - 2] and t >= 2:
            strategy_returns[t] -= cost_per_trade
            total_costs += cost_per_trade
            
            # Track trade boundaries
            if signals[t - 1] == 1.0 and signals[t - 2] == 0.0:
                # Entering position
                current_entry_idx = t
            elif signals[t - 1] == 0.0 and signals[t - 2] == 1.0:
                # Exiting position - record trade
                if current_entry_idx is not None:
                    trade_ret = prices[t] / prices[current_entry_idx] - 1.0
                    trades.append(trade_ret)
                    current_entry_idx = None
    
    # Build equity curve
    equity_curve = np.cumprod(1.0 + strategy_returns)
    
    # Summary stats
    n_days = len(prices) - 1
    total_return = equity_curve[-1] / equity_curve[0] - 1.0
    
    # CAGR
    years = n_days / 252.0
    if years > 0 and equity_curve[-1] > 0:
        cagr = (equity_curve[-1] / equity_curve[0]) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0
    
    # Volatility (annualised)
    daily_vol = np.std(strategy_returns[1:], ddof=1)
    volatility = daily_vol * np.sqrt(252)
    
    # Sharpe
    daily_rf = rf / 252.0
    excess_daily = strategy_returns[1:] - daily_rf * (signals[:-1] > 0)
    sharpe = (np.mean(excess_daily) / np.std(excess_daily, ddof=1)) * np.sqrt(252) if np.std(excess_daily, ddof=1) > 0 else 0.0
    
    # Max drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / running_max
    max_drawdown = np.max(drawdowns)
    
    # Trade stats
    n_trades = len(trades)
    win_rate = np.mean([1 if t > 0 else 0 for t in trades]) if trades else 0.0
    avg_trade_return = np.mean(trades) if trades else 0.0
    
    # Time in market
    time_in_market = np.mean(signals[:-1] > 0)
    
    # Monthly returns
    monthly_returns = _compute_monthly_returns(equity_curve, dates)
    
    return BacktestResult(
        equity_curve=equity_curve,
        dates=dates,
        signals=signals,
        total_return=total_return,
        cagr=cagr,
        volatility=volatility,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        n_trades=n_trades,
        win_rate=win_rate,
        avg_trade_return=avg_trade_return,
        time_in_market=time_in_market,
        monthly_returns=monthly_returns,
        total_costs=total_costs
    )

def _compute_monthly_returns(equity_curve, dates=None):
    """Compute monthly returns from equity curve."""
    if dates is not None:
        import pandas as pd
        # Ensure we have datetime
        if not isinstance(dates[0], pd.Timestamp):
            dates = pd.to_datetime(dates)
            
        months = {}
        for i, d in enumerate(dates):
            key = (d.year, d.month)
            if key not in months:
                months[key] = {'start': i, 'end': i}
            months[key]['end'] = i
        
        monthly = []
        for key in months:
            s = months[key]['start']
            e = months[key]['end']
            if s >= 0 and e > s:
                monthly.append(equity_curve[e] / equity_curve[s] - 1.0)
        return np.array(monthly)
    else:
        # 21-day blocks
        block_size = 21
        monthly = []
        for i in range(0, len(equity_curve) - block_size, block_size):
            monthly.append(equity_curve[i + block_size] / equity_curve[i] - 1.0)
        return np.array(monthly)
