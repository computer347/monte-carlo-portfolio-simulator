import numpy as np
import pandas as pd

def cross_sectional_momentum(prices_df, lookback=63, top_n=5):
    """
    Rank all assets by trailing return, select top N.
    """
    signals = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    
    for i in range(lookback, len(prices_df)):
        returns = prices_df.iloc[i] / prices_df.iloc[i - lookback] - 1.0
        
        # Rank descending, select top N
        ranked = returns.nlargest(top_n).index
        signals.loc[signals.index[i], ranked] = 1.0
    
    return signals

def absolute_momentum_filter(prices_df, lookback=63):
    """
    Binary filter: 1 if asset has positive trailing return, 0 otherwise.
    Used as an overlay on cross-sectional momentum to avoid holding
    assets in absolute downtrends.
    """
    filter_df = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    
    for i in range(lookback, len(prices_df)):
        returns = prices_df.iloc[i] / prices_df.iloc[i - lookback] - 1.0
        filter_df.iloc[i] = (returns > 0).astype(float)
    
    return filter_df

def dual_momentum(prices_df, lookback=63, top_n=5):
    """
    Combined cross-sectional + absolute momentum.
    """
    cs_signals = cross_sectional_momentum(prices_df, lookback, top_n)
    abs_filter = absolute_momentum_filter(prices_df, lookback)
    
    # Only hold if both cross-sectional AND absolute momentum agree
    return cs_signals * abs_filter
