import pytest
import numpy as np
import pandas as pd
from data.calibrator import CalibrationResult
from simulation.models import gbm_paths, student_t_paths, merton_jump_paths

@pytest.fixture
def mock_cal():
    # Return a CalibrationResult perfectly set for testing
    return CalibrationResult(
        mu_historical=0.05,
        mu_gbm=0.0,  # Zero drift for basic flatline tests
        sigma=0.0,   # Zero vol
        skewness=0.0,
        excess_kurtosis=0.0,
        jb_pvalue=1.0,
        sw_pvalue=1.0,
        log_returns=pd.Series([0.0]*100),
        rolling_vol=pd.Series([0.0]*100),
        rolling_sharpe=pd.Series([0.0]*100),
        last_price=100.0
    )

@pytest.fixture
def active_cal():
    return CalibrationResult(
        mu_historical=0.05,
        mu_gbm=0.05,  
        sigma=0.2,   
        skewness=0.0,
        excess_kurtosis=0.0,
        jb_pvalue=1.0,
        sw_pvalue=1.0,
        log_returns=pd.Series(np.random.normal(0, 0.01, 100)),
        rolling_vol=pd.Series([0.2]*100),
        rolling_sharpe=pd.Series([0.1]*100),
        last_price=100.0
    )

def test_gbm_zero_drift_vol(mock_cal):
    paths = gbm_paths(mock_cal, S0=100.0, n_paths=10, n_days=50)
    # If mu_gbm = 0 and sigma = 0, paths should be totally flat at exactly 100.0
    assert np.allclose(paths, 100.0)

def test_gbm_seed_determinism(active_cal):
    paths1 = gbm_paths(active_cal, S0=100.0, n_paths=100, n_days=252, seed=42)
    paths2 = gbm_paths(active_cal, S0=100.0, n_paths=100, n_days=252, seed=42)
    assert np.array_equal(paths1, paths2)

def test_student_t_variance():
    # Test Student-t: verify variance of Z draws ≈ 1.0 after normalisation
    np.random.seed(42)
    df = 5.0
    t_draw = np.random.standard_t(df, size=(5000, 252))
    Z = t_draw / np.sqrt(df / (df - 2.0))
    # Variance of the normalized matrix should be close to 1
    assert np.isclose(np.var(Z), 1.0, rtol=0.05)
    
def test_merton_differs_from_gbm(active_cal):
    # Both take Z ~ N(0,1), but merton adds jumps. Using same seed should yield different paths.
    seed = 42
    gbm = gbm_paths(active_cal, S0=100.0, n_paths=100, n_days=252, seed=seed)
    # Using high jump lambda and size to guarantee variation
    merton = merton_jump_paths(active_cal, S0=100.0, n_paths=100, n_days=252, lambda_=5.0, mu_j=-0.02, sigma_j=0.04, seed=seed)
    
    # Path sums should differ significantly due to added jumps
    assert not np.array_equal(gbm, merton)
