import pytest
import numpy as np
import pandas as pd
from data.calibrator import calibrate, CalibrationResult

@pytest.fixture
def dummy_price_data():
    np.random.seed(42)
    # Generate 500 days of random walk data for stability
    returns = np.random.normal(0.0005, 0.01, 500)
    prices = 100 * np.exp(np.cumsum(returns))
    dates = pd.date_range('2020-01-01', periods=500)
    
    # Store in dataframe
    return pd.DataFrame({'Close': prices}, index=dates)

def test_calibrate_output_types(dummy_price_data):
    result = calibrate(dummy_price_data)
    
    assert isinstance(result, CalibrationResult)
    assert isinstance(result.mu_historical, float)
    assert isinstance(result.mu_gbm, float)
    assert isinstance(result.sigma, float)
    assert isinstance(result.skewness, float)
    assert isinstance(result.excess_kurtosis, float)
    assert isinstance(result.jb_pvalue, float)
    assert isinstance(result.sw_pvalue, float)

def test_calibrate_gbm_drift_logic(dummy_price_data):
    result = calibrate(dummy_price_data)
    
    # Check relationship: mu_gbm = mu_historical - 0.5 * sigma^2
    expected_mu_gbm = result.mu_historical - 0.5 * (result.sigma ** 2)
    assert np.isclose(result.mu_gbm, expected_mu_gbm)

def test_calibrate_rolling_stats(dummy_price_data):
    result = calibrate(dummy_price_data)
    
    # Ensure they have NaNs at the beginning due to the 252-day window
    assert result.rolling_vol.iloc[0:251].isna().all()
    assert result.rolling_sharpe.iloc[0:251].isna().all()
    
    # Ensure they have values at the end
    assert not np.isnan(result.rolling_vol.iloc[-1])
    assert not np.isnan(result.rolling_sharpe.iloc[-1])

    # Ensure length matches returns (length-1 from prices)
    assert len(result.rolling_vol) == 499
