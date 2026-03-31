import pytest
import numpy as np
import pandas as pd
from data.calibrator import CalibrationResult
from simulation.engine import simulate
from simulation.results import SimulationResult

@pytest.fixture
def active_cal():
    return CalibrationResult(
        mu_historical=0.05,
        mu_gbm=0.01,  
        sigma=0.3,   
        skewness=0.0,
        excess_kurtosis=0.0,
        jb_pvalue=1.0,
        sw_pvalue=1.0,
        log_returns=pd.Series(np.random.normal(0, 0.01, 100)),
        rolling_vol=pd.Series([0.2]*100),
        rolling_sharpe=pd.Series([0.1]*100),
        last_price=150.0
    )

def test_engine_output_shape(active_cal):
    n_days = 100
    n_paths = 50
    paths, result = simulate(active_cal, S0=150.0, n_paths=n_paths, n_days=n_days, mode="gbm")
    
    assert isinstance(result, SimulationResult)
    # final prices should be of length n_paths
    assert paths.shape == (n_paths, n_days + 1)
    assert result.final_prices.shape == (n_paths,)
    assert result.log_returns_total.shape == (n_paths,)

def test_engine_positivity_constraint(active_cal):
    # Log-normal property verification for standard GBM
    n_days = 252
    n_paths = 1000
    # Simulate a heavily volatile asset
    high_vol = active_cal
    high_vol.sigma = 0.5
    
    # Check that ALL paths strictly stay strictly positive directly from models since engine returns SimulationResult
    paths, _ = simulate(high_vol, S0=50.0, n_paths=n_paths, n_days=n_days, mode="gbm")
    
    assert (paths > 0).all()

def test_engine_initial_price(active_cal):
    # When None, engine should default to last_price from cal
    n_days = 50
    n_paths = 10
    
    paths, result = simulate(active_cal, S0=None, n_paths=n_paths, n_days=n_days, mode="student_t")
    
    assert paths[:, 0].mean() == 150.0
