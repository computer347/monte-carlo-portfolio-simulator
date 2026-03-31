from data.calibrator import CalibrationResult
from simulation.results import SimulationResult
from simulation import models

def simulate(cal: CalibrationResult, 
             S0: float = None, 
             n_paths: int = 1000, 
             n_days: int = 252, 
             mode: str = "gbm", 
             **kwargs) -> SimulationResult:
    """
    Main entry point for Monte Carlo paths generation.
    
    Args:
        cal (CalibrationResult): Calibrated parameters.
        S0 (float, optional): Initial price. Defaults to last closing price.
        n_paths (int): Number of separate paths to simulate.
        n_days (int): Number of trading days to simulate ahead.
        mode (str): Simulation model ('gbm', 'student_t', 'jump_diffusion').
        **kwargs: Extensible keyword arguments for models (e.g., seed, df, lambda_).
        
    Returns:
        SimulationResult: Object holding paths and calculated metrics.
    """
    if S0 is None:
        S0 = cal.last_price

    mode = mode.lower()
    
    if mode == "gbm":
        paths = models.gbm_paths(cal, S0, n_paths, n_days, **kwargs)
    elif mode == "student_t":
        paths = models.student_t_paths(cal, S0, n_paths, n_days, **kwargs)
    elif mode == "jump_diffusion":
        paths = models.merton_jump_paths(cal, S0, n_paths, n_days, **kwargs)
    else:
        raise ValueError(f"Unknown simulation mode: '{mode}'")
        
    # Return paths and calculated result wrapper
    return paths, SimulationResult.from_paths(paths, rf=0.045, n_days=n_days)
