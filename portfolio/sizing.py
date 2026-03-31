import numpy as np
import pandas as pd

def inverse_volatility_weights(prices_df, signals_df, vol_window=21):
    """
    Size positions inversely proportional to their recent volatility.
    """
    log_returns = np.log(prices_df / prices_df.shift(1))
    rolling_vol = log_returns.rolling(window=vol_window).std() * np.sqrt(252)
    rolling_vol = rolling_vol.clip(lower=0.01)
    
    weights = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    
    for i in range(vol_window, len(prices_df)):
        selected = signals_df.iloc[i] > 0
        if selected.sum() == 0:
            continue
        
        vols = rolling_vol.iloc[i, selected.values]
        inv_vol = 1.0 / vols
        weights.iloc[i, selected.values] = inv_vol / inv_vol.sum()
    
    return weights

def equal_weights(signals_df):
    """
    Equal-weight all selected assets.
    """
    weights = pd.DataFrame(0.0, index=signals_df.index, columns=signals_df.columns)
    
    for i in range(len(signals_df)):
        selected = signals_df.iloc[i] > 0
        n = selected.sum()
        if n > 0:
            weights.iloc[i, selected.values] = 1.0 / n
    
    return weights
