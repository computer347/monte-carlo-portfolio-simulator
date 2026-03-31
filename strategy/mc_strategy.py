import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass
class MCStrategyResult:
    """Container for Monte Carlo strategy overlay output."""
    # Per-path summary stats (arrays of length n_paths)
    total_returns: np.ndarray
    cagrs: np.ndarray
    sharpes: np.ndarray
    max_drawdowns: np.ndarray
    
    # Percentile equity curves for fan chart
    # Shape: (n_steps,) for each percentile
    pctile_5: np.ndarray
    pctile_25: np.ndarray
    pctile_50: np.ndarray
    pctile_75: np.ndarray
    pctile_95: np.ndarray
    
    # Aggregated stats
    median_cagr: float
    median_sharpe: float
    median_max_dd: float
    prob_profit: float             # % of paths with positive total return
    prob_beat_buyhold: float       # % of paths where strategy beats buy & hold
    
    # Optional field: we need median monthly returns across all paths for row 4
    # The requirement is "histogram of MC median monthly returns"
    monthly_returns_median: np.ndarray  # Added for chart mapping
    
def run_mc_strategy(
    paths: np.ndarray,
    signal_fn: Callable,
    signal_params: dict = None,
    cost_per_trade: float = 0.001,
    rf: float = 0.045
) -> MCStrategyResult:
    """
    Apply a trading strategy across all Monte Carlo simulated paths.
    """
    if signal_params is None:
        signal_params = {}
    
    n_paths, n_steps = paths.shape
    
    # Storage
    all_total_returns = np.zeros(n_paths)
    all_cagrs = np.zeros(n_paths)
    all_sharpes = np.zeros(n_paths)
    all_max_dds = np.zeros(n_paths)
    all_equity = np.zeros((n_paths, n_steps))  # full equity curves for percentiles
    buyhold_returns = np.zeros(n_paths)
    
    # monthly returns across all paths (for the Row 4 plot)
    block_size = 21
    n_months = (n_steps - 1) // block_size
    all_monthly = np.zeros((n_paths, n_months)) if n_months > 0 else np.zeros((n_paths, 1))
    
    years = (n_steps - 1) / 252.0
    daily_rf = rf / 252.0
    
    for i in range(n_paths):
        price_path = paths[i, :]
        signals = signal_fn(price_path, **signal_params)
        
        # Daily returns
        price_rets = np.zeros(n_steps)
        price_rets[1:] = price_path[1:] / price_path[:-1] - 1.0
        
        # Strategy returns with transaction costs
        strat_rets = np.zeros(n_steps)
        strat_rets[1:] = signals[:-1] * price_rets[1:]
        signal_changes = np.diff(signals[:-1]) != 0
        strat_rets[2:][signal_changes] -= cost_per_trade
        
        # Equity curve
        equity = np.cumprod(1.0 + strat_rets)
        all_equity[i, :] = equity
        
        # Stats
        all_total_returns[i] = equity[-1] - 1.0
        all_cagrs[i] = (equity[-1] ** (1.0 / years) - 1.0) if years > 0 and equity[-1] > 0 else 0.0
        
        daily_vol = np.std(strat_rets[1:], ddof=1)
        if daily_vol > 0:
            excess = strat_rets[1:] - daily_rf * (signals[:-1] > 0)
            all_sharpes[i] = (np.mean(excess) / np.std(excess, ddof=1)) * np.sqrt(252)
        
        running_max = np.maximum.accumulate(equity)
        dd = (running_max - equity) / running_max
        all_max_dds[i] = np.max(dd)
        
        buyhold_returns[i] = price_path[-1] / price_path[0] - 1.0
        
        if n_months > 0:
            month_ret = []
            for m_idx in range(n_months):
                s = m_idx * block_size
                e = s + block_size
                if e < len(equity):
                    month_ret.append(equity[e] / equity[s] - 1.0)
                else:
                    month_ret.append(0.0)
            all_monthly[i, :] = np.array(month_ret)
    
    pctile_5 = np.percentile(all_equity, 5, axis=0)
    pctile_25 = np.percentile(all_equity, 25, axis=0)
    pctile_50 = np.percentile(all_equity, 50, axis=0)
    pctile_75 = np.percentile(all_equity, 75, axis=0)
    pctile_95 = np.percentile(all_equity, 95, axis=0)
    
    monthly_returns_median = np.median(all_monthly, axis=0) if n_months > 0 else np.array([])
    
    return MCStrategyResult(
        total_returns=all_total_returns,
        cagrs=all_cagrs,
        sharpes=all_sharpes,
        max_drawdowns=all_max_dds,
        pctile_5=pctile_5,
        pctile_25=pctile_25,
        pctile_50=pctile_50,
        pctile_75=pctile_75,
        pctile_95=pctile_95,
        median_cagr=float(np.median(all_cagrs)),
        median_sharpe=float(np.median(all_sharpes)),
        median_max_dd=float(np.median(all_max_dds)),
        prob_profit=float(np.mean(all_total_returns > 0)),
        prob_beat_buyhold=float(np.mean(all_total_returns > buyhold_returns)),
        monthly_returns_median=monthly_returns_median
    )
