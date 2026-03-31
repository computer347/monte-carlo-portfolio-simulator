import numpy as np
from data.calibrator import CalibrationResult

def _init_paths(S0: float, n_paths: int, n_days: int) -> np.ndarray:
    """Helper to initialize an array of paths with S0."""
    paths = np.zeros((n_paths, n_days + 1))
    paths[:, 0] = S0
    return paths

def gbm_paths(cal: CalibrationResult, S0: float, n_paths: int, n_days: int, seed: int = None) -> np.ndarray:
    """
    Standard geometric Brownian motion using vectorised Euler-Maruyama.
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = 1.0 / 252.0
    Z = np.random.normal(size=(n_paths, n_days))
    
    # Use mu_gbm directly in exponent (Itô drift correction already applied in calibrate module)
    daily_returns = np.exp(cal.mu_gbm * dt + cal.sigma * np.sqrt(dt) * Z)
    
    paths = _init_paths(S0, n_paths, n_days)
    paths[:, 1:] = S0 * np.cumprod(daily_returns, axis=1)
    
    return paths

def student_t_paths(cal: CalibrationResult, S0: float, n_paths: int, n_days: int, df: float = 3.0, seed: int = None) -> np.ndarray:
    """
    Fat-tailed GBM using a Normalised Student-t distribution for the random driver.
    """
    if seed is not None:
        np.random.seed(seed)
        
    dt = 1.0 / 252.0
    t_draw = np.random.standard_t(df, size=(n_paths, n_days))
    # Normalize to preserve unit variance
    norm_factor = np.sqrt(df / (df - 2.0))
    Z = t_draw / norm_factor
    
    daily_returns = np.exp(cal.mu_gbm * dt + cal.sigma * np.sqrt(dt) * Z)
    
    paths = _init_paths(S0, n_paths, n_days)
    paths[:, 1:] = S0 * np.cumprod(daily_returns, axis=1)
    
    return paths

def merton_jump_paths(cal: CalibrationResult, S0: float, n_paths: int, n_days: int, 
                      lambda_: float = 5.0, mu_j: float = -0.02, sigma_j: float = 0.04, 
                      seed: int = None) -> np.ndarray:
    """
    Merton Jump-Diffusion combining GBM with a compound Poisson jump process.
    """
    if seed is not None:
        np.random.seed(seed)
        
    dt = 1.0 / 252.0
    Z = np.random.normal(size=(n_paths, n_days))
    
    kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    
    # N_t ~ Poisson(lambda*dt)
    N_t = np.random.poisson(lambda_ * dt, size=(n_paths, n_days))
    
    # Jump size is sum of N_t independent normals N(mu_j, sigma_j). 
    # Equivalently, drawn from N(N_t * mu_j, sqrt(N_t) * sigma_j)
    jump_mean = N_t * mu_j
    jump_std = np.sqrt(N_t) * sigma_j
    total_jumps = np.random.normal(jump_mean, jump_std)
    
    daily_returns = np.exp((cal.mu_gbm - lambda_ * kappa) * dt + cal.sigma * np.sqrt(dt) * Z + total_jumps)
    
    paths = _init_paths(S0, n_paths, n_days)
    paths[:, 1:] = S0 * np.cumprod(daily_returns, axis=1)
    
    return paths
