from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, jarque_bera, shapiro


@dataclass
class CalibrationResult:
    """Holds calibration parameters from historical data."""
    mu_historical: float  # Mean log return * 252 (Not for direct use in simulation)
    mu_gbm: float         # Itô-corrected drift for use in GBM simulation
    sigma: float          # Annualised volatility
    skewness: float
    excess_kurtosis: float
    jb_pvalue: float
    sw_pvalue: float
    log_returns: pd.Series
    rolling_vol: pd.Series
    rolling_sharpe: pd.Series
    last_price: float


def calibrate(
    df: pd.DataFrame,
    price_col: str = 'Close',
    risk_free_rate: float = 0.045,
    trading_days: int = 252
) -> CalibrationResult:
    """
    Computes statistical parameters for Monte Carlo simulation from a dataframe.

    Args:
        df (pd.DataFrame): OHLCV DataFrame (e.g., from fetcher.py)
        price_col (str): Column representing price. Use 'Adj Close' if available,
                         otherwise 'Close'.
        risk_free_rate (float): Annualised risk-free rate (e.g., 0.045 for 4.5%).
        trading_days (int): Number of trading days in a year (default 252).

    Returns:
        CalibrationResult: Calibrated metrics ready for simulation.

    Raises:
        ValueError: If DataFrame has fewer than 30 rows.
    """
    # --- Data validation ---
    if len(df) < 30:
        raise ValueError(
            f"Insufficient data: {len(df)} rows. Minimum 30 required for calibration."
        )
    if len(df) < trading_days:
        warnings.warn(
            f"Only {len(df)} rows available — rolling stats and Sharpe ratio will be "
            f"sparse (require {trading_days} rows for a full window). "
            f"Treat rolling outputs with caution.",
            UserWarning
        )

    # --- Price column selection ---
    if 'Adj Close' in df.columns:
        price_col = 'Adj Close'
    elif 'Close' not in df.columns:
        # Fallback if a single-column Series was passed as a DataFrame
        price_col = df.columns[0]

    # --- Log returns ---
    prices = df[price_col]
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # --- Core parameters ---
    daily_mu = log_returns.mean()
    daily_sigma = log_returns.std()

    mu_historical = daily_mu * trading_days
    sigma = daily_sigma * np.sqrt(trading_days)

    # mu_gbm is the Itô-corrected drift for use in GBM simulation.
    # Do not use mu_historical directly in any simulation context.
    mu_gbm = mu_historical - 0.5 * (sigma ** 2)

    # --- Distribution diagnostics ---
    # scipy.stats.kurtosis with fisher=True returns excess kurtosis (0 for normal)
    dist_skewness = skew(log_returns)
    dist_excess_kurtosis = kurtosis(log_returns, fisher=True)

    # Jarque-Bera test (full history)
    _, jb_p = jarque_bera(log_returns)

    # Shapiro-Wilk on the last 252 days only (test is unreliable on large samples)
    last_year_returns = (
        log_returns[-trading_days:] if len(log_returns) > trading_days else log_returns
    )
    try:
        _, sw_p = shapiro(last_year_returns)
    except ValueError:
        sw_p = np.nan

    # --- Rolling statistics (252-day window) ---
    window = trading_days
    rolling_vol = log_returns.rolling(window=window).std() * np.sqrt(trading_days)

    # rf is constant so std(r - rf) == std(r); mean is adjusted, std is not.
    # Explicit: daily risk-free rate = 0.045 / 252, subtracted from each daily log return.
    daily_rf = risk_free_rate / trading_days
    rolling_excess_returns_mean = (log_returns - daily_rf).rolling(window=window).mean()
    rolling_excess_returns_std = log_returns.rolling(window=window).std()

    # Annualised Sharpe ratio
    rolling_sharpe = (
        rolling_excess_returns_mean / rolling_excess_returns_std
    ) * np.sqrt(trading_days)

    return CalibrationResult(
        mu_historical=mu_historical,
        mu_gbm=mu_gbm,
        sigma=sigma,
        skewness=dist_skewness,
        excess_kurtosis=dist_excess_kurtosis,
        jb_pvalue=jb_p,
        sw_pvalue=sw_p,
        log_returns=log_returns,
        rolling_vol=rolling_vol,
        rolling_sharpe=rolling_sharpe,
        last_price=float(prices.iloc[-1]),
    )