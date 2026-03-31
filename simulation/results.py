from dataclasses import dataclass
import numpy as np

@dataclass
class SimulationResult:
    """Metrics calculated across all simulated paths."""
    final_prices: np.ndarray
    log_returns_total: np.ndarray
    cagr_mean: float
    cagr_median: float
    cagr_std: float
    sharpe_ratio: float
    max_drawdown_mean: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float

    @classmethod
    def from_paths(cls, paths: np.ndarray, rf: float = 0.045, n_days: int = 252) -> 'SimulationResult':
        """
        Creates a SimulationResult from an (n_paths, n_days+1) array of paths.
        
        Args:
            paths (np.ndarray): Price paths shape (n_paths, n_days+1)
            rf (float): Annualised risk-free rate
            n_days (int): Number of trading days in the simulation
        """
        initial_prices = paths[:, 0]
        final_prices = paths[:, -1]
        
        # log(S_T / S_0)
        log_returns_total = np.log(final_prices / initial_prices)
        
        # CAGR = (S_T / S_0) ^ (252 / n_days) - 1
        # Which is equivalent to exp(log_returns_total * (252 / n_days)) - 1
        cagr = np.exp(log_returns_total * (252.0 / n_days)) - 1.0
        
        cagr_mean = float(np.mean(cagr))
        cagr_median = float(np.median(cagr))
        cagr_std = float(np.std(cagr))
        
        # Sharpe Ratio (annualised)
        # We annualise the mean total return and standard deviation
        ann_mean_ret = np.mean(log_returns_total) * (252.0 / n_days)
        ann_std_ret = np.std(log_returns_total) * np.sqrt(252.0 / n_days)
        
        # Guard against zero vol paths
        if ann_std_ret == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = float((ann_mean_ret - rf) / ann_std_ret)
            
        # Max Drawdown
        # Using numpy running maximum
        running_max = np.maximum.accumulate(paths, axis=1)
        drawdowns = (running_max - paths) / running_max
        max_drawdowns = np.max(drawdowns, axis=1)
        max_drawdown_mean = float(np.mean(max_drawdowns))
        
        # VaR and CVaR (left tail of log_returns_total)
        # Note: VaR is typically expressed as a positive number for loss, 
        # or negative representing the cutoff. We'll use the literal percentile value.
        var_95 = float(np.percentile(log_returns_total, 5))
        var_99 = float(np.percentile(log_returns_total, 1))
        
        # CVaR 95 is the mean of returns worse than VaR 95
        tail_returns = log_returns_total[log_returns_total < var_95]
        cvar_95 = float(np.mean(tail_returns)) if len(tail_returns) > 0 else var_95
        
        # CVaR 99 is the mean of returns worse than VaR 99
        tail_returns_99 = log_returns_total[log_returns_total < var_99]
        cvar_99 = float(np.mean(tail_returns_99)) if len(tail_returns_99) > 0 else var_99
        
        return cls(
            final_prices=final_prices,
            log_returns_total=log_returns_total,
            cagr_mean=cagr_mean,
            cagr_median=cagr_median,
            cagr_std=cagr_std,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_mean=max_drawdown_mean,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99
        )
