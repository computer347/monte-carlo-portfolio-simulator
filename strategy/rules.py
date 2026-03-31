import numpy as np

def buy_and_hold(prices):
    """Baseline: always long.
    
    Args:
        prices: 1D np.array of daily prices
    Returns:
        signals: 1D np.array of same length, all 1.0
    """
    return np.ones(len(prices))

def sma_crossover(prices, fast=20, slow=50):
    """Long when fast SMA > slow SMA, flat otherwise.
    
    Args:
        prices: 1D np.array of daily prices
        fast: fast moving average window (default 20)
        slow: slow moving average window (default 50)
    Returns:
        signals: 1D np.array, 1.0 = long, 0.0 = flat
        First `slow` entries are 0.0 (insufficient data)
    """
    signals = np.zeros(len(prices))
    
    # Use cumsum trick for efficient rolling mean
    fast_ma = _rolling_mean(prices, fast)
    slow_ma = _rolling_mean(prices, slow)
    
    for i in range(slow, len(prices)):
        if fast_ma[i] > slow_ma[i]:
            signals[i] = 1.0
    
    return signals

def momentum(prices, lookback=21, threshold=0.0):
    """Long when N-day return exceeds threshold, flat otherwise.
    
    Args:
        prices: 1D np.array of daily prices
        lookback: number of days for return calculation (default 21 = ~1 month)
        threshold: minimum return to trigger long (default 0.0)
    Returns:
        signals: 1D np.array, 1.0 = long, 0.0 = flat
        First `lookback` entries are 0.0
    """
    signals = np.zeros(len(prices))
    
    for i in range(lookback, len(prices)):
        ret = (prices[i] / prices[i - lookback]) - 1.0
        if ret > threshold:
            signals[i] = 1.0
    
    return signals

def mean_reversion(prices, window=20, z_entry=-1.0, z_exit=0.0):
    """Long when price drops below z_entry std devs from rolling mean.
    Exit when price returns to z_exit std devs.
    
    Args:
        prices: 1D np.array of daily prices
        window: rolling window for mean/std (default 20)
        z_entry: z-score to enter long (default -1.0, i.e. 1 std below mean)
        z_exit: z-score to exit (default 0.0, i.e. at the mean)
    Returns:
        signals: 1D np.array, 1.0 = long, 0.0 = flat
    """
    signals = np.zeros(len(prices))
    in_position = False
    
    rolling_mu = _rolling_mean(prices, window)
    rolling_std = _rolling_std(prices, window)
    
    for i in range(window, len(prices)):
        if rolling_std[i] < 1e-8:
            continue
        z = (prices[i] - rolling_mu[i]) / rolling_std[i]
        
        if not in_position and z < z_entry:
            in_position = True
        elif in_position and z > z_exit:
            in_position = False
        
        signals[i] = 1.0 if in_position else 0.0
    
    return signals

# --- Helpers (private) ---

def _rolling_mean(arr, window):
    """Efficient rolling mean using cumsum."""
    out = np.full(len(arr), np.nan)
    cs = np.cumsum(arr)
    out[window - 1:] = (cs[window - 1:] - np.concatenate(([0], cs[:-window]))) / window
    return out

def _rolling_std(arr, window):
    """Rolling standard deviation."""
    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        out[i] = np.std(arr[i - window + 1:i + 1], ddof=1)
    return out

# Registry for dashboard dropdown
STRATEGY_REGISTRY = {
    "Buy & Hold": {"fn": buy_and_hold, "params": {}},
    "SMA Crossover (20/50)": {"fn": sma_crossover, "params": {"fast": 20, "slow": 50}},
    "SMA Crossover (10/30)": {"fn": sma_crossover, "params": {"fast": 10, "slow": 30}},
    "Momentum (21d)": {"fn": momentum, "params": {"lookback": 21}},
    "Momentum (63d)": {"fn": momentum, "params": {"lookback": 63}},
    "Mean Reversion (20d)": {"fn": mean_reversion, "params": {"window": 20, "z_entry": -1.0, "z_exit": 0.0}},
}
