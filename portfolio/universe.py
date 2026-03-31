import yfinance as yf
import numpy as np
import pandas as pd

# ETF Universe — all available on IBKR, highly liquid, diversified across asset classes
# Each entry: (ticker, name, asset_class)
ETF_UNIVERSE = [
    # Equities — core global allocation
    ("SPY",  "S&P 500",              "equity"),
    ("QQQ",  "Nasdaq 100",           "equity"),
    ("EFA",  "MSCI EAFE",            "equity"),
    ("EEM",  "MSCI Emerging Markets","equity"),
    ("VNQ",  "US Real Estate",       "equity"),
    
    # Bonds — rate-sensitive diversifiers
    ("TLT",  "20+ Year Treasury",    "bond"),
    ("IEF",  "7-10 Year Treasury",   "bond"),
    ("TIP",  "TIPS (Inflation-Protected)", "bond"),
    
    # Commodities — broad + precious metals only, NO single-commodity ETCs
    ("GLD",  "Gold",                 "commodity"),
    ("DBC",  "Broad Commodities",    "commodity"),
]

def fetch_universe_prices(period="5y"):
    """
    Fetch daily close prices for all tickers in the universe.
    
    Args:
        period: yfinance period string (default "5y")
    
    Returns:
        pd.DataFrame with DatetimeIndex, columns = ticker symbols, values = daily close
        Only includes tickers that returned valid data.
    """
    tickers = [t[0] for t in ETF_UNIVERSE]
    
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    
    # Extract Close prices
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data[['Close']]
        prices.columns = tickers[:1]
    
    # Drop tickers with >20% missing
    threshold = len(prices) * 0.2
    valid_cols = prices.columns[prices.isnull().sum() < threshold]
    prices = prices[valid_cols]
    
    # Forward-fill small gaps, then drop any remaining NaN rows
    prices = prices.ffill(limit=5)
    prices = prices.dropna()
    
    return prices

def get_universe_info():
    """Returns list of (ticker, name, asset_class) for UI display."""
    return ETF_UNIVERSE
