import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class PortfolioResult:
    """Container for portfolio backtest output."""
    # Time series
    equity_curve: np.ndarray           # daily portfolio value in EUR
    cash_curve: np.ndarray             # daily cash balance in EUR
    dates: np.ndarray                  # date array
    tickers: List[str]                 # list of ticker symbols
    actual_weights_history: np.ndarray # (n_days, n_assets) actual allocation weights
    daily_returns: np.ndarray          # daily portfolio returns (contribution-adjusted)
    
    # Summary stats
    total_return: float                # total return on invested capital
    twrr_total: float                  # time-weighted total return
    cagr: float                        # annualised TWRR
    volatility: float                  # annualised volatility
    sharpe: float                      # annualised Sharpe ratio
    max_drawdown: float                # worst peak-to-trough on equity curve
    
    # Trading stats
    n_rebalances: int                  # number of rebalance events
    avg_turnover: float                # average turnover per rebalance
    total_costs: float                 # cumulative EUR transaction costs
    
    # Capital tracking
    total_invested: float              # total EUR put in (initial + contributions)
    final_value: float                 # ending portfolio EUR value
    
    # Returns
    monthly_returns: np.ndarray        # array of monthly returns
    asset_returns: Dict[str, float]    # per-asset total return over period
