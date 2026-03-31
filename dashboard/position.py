import numpy as np
from simulation.engine import simulate
from simulation.results import SimulationResult

DRY_POWDER = 137.75

def get_position_metrics(entry_price, units, current_price, dry_powder=DRY_POWDER):
    """Returns dict of current position metrics in euros."""
    position_value = current_price * units
    unrealised_pl = (current_price - entry_price) * units
    
    if entry_price > 0:
        pl_percent = (current_price / entry_price - 1.0) * 100.0
    else:
        pl_percent = 0.0
        
    return {
        "entry_price": entry_price,
        "current_price": current_price,
        "units": units,
        "position_value": position_value,
        "unrealised_pl": unrealised_pl,
        "pl_percent": pl_percent,
        "dry_powder": dry_powder
    }

def run_monthly_simulation(cal, entry_price, units, etc_price, n_paths=10000):
    """Runs 21-day simulation, returns euro P&L distribution mapped to ETC."""
    paths, sim_res = simulate(cal, S0=None, n_paths=n_paths, n_days=21, mode="gbm")
    
    returns = paths[:, -1] / paths[:, 0]
    projected_prices = etc_price * returns
    euro_pl = (projected_prices - entry_price) * units
    
    var_95_euro = np.percentile(euro_pl, 5)
    
    tail_pl = euro_pl[euro_pl < var_95_euro]
    cvar_95_euro = np.mean(tail_pl) if len(tail_pl) > 0 else var_95_euro
    
    median_euro = np.median(euro_pl)
    
    return {
        "euro_pl": euro_pl,
        "var_95_euro": var_95_euro,
        "cvar_95_euro": cvar_95_euro,
        "median_euro": median_euro
    }

def run_horizon_scenarios(cal, entry_price, units, etc_price):
    """Runs single 252 day simulation, extracts percentiles at 1M, 3M, 6M, 12M mapped to ETC."""
    n_days = 252
    paths, sim_res = simulate(cal, S0=None, n_paths=10000, n_days=n_days, mode="gbm")
    
    horizons = [21, 63, 126, 252]
    percentiles = [5, 25, 50, 75, 95]
    
    results = {p: [] for p in percentiles}
    results['horizons'] = horizons
    
    for h in horizons:
        returns = paths[:, h] / paths[:, 0]
        projected_prices = etc_price * returns
        euro_pls = (projected_prices - entry_price) * units
        
        for p in percentiles:
            results[p].append(np.percentile(euro_pls, p))
            
    # Also want median cagr for the targets later
    results['median_cagr'] = sim_res.cagr_median
    
    return results

def covered_call_metrics(position_value, premium_rate):
    """Returns current monthly premium estimate and capital targets."""
    monthly_premium = position_value * premium_rate
    annual_premium = monthly_premium * 12
    
    targets = [25, 50, 100, 200]
    required_capital = {t: t / premium_rate for t in targets}
    
    return {
        "monthly_premium": monthly_premium,
        "annual_premium": annual_premium,
        "targets": targets,
        "required_capital": required_capital
    }

def months_to_target(current_value, monthly_savings, target_income, premium_rate, median_cagr):
    """
    Calculates months to reach each income target.
    
    FV = PV*(1+r)^n + PMT*[(1+r)^n - 1]/r
    TargetCapital = TargetIncome / PremiumRate
    Solve for n:
    n = log( (TargetCapital * r + PMT) / (PV * r + PMT) ) / log(1 + r)
    """
    target_capital = target_income / premium_rate
    
    # Monthly growth rate
    r = (1.0 + max(0, median_cagr)) ** (1.0 / 12.0) - 1.0 
    
    # If r is effectively zero, simple arithmetic avoids /0 error
    if r < 1e-6:
        if monthly_savings > 0:
            remaining = target_capital - current_value
            months = remaining / monthly_savings if remaining > 0 else 0
            return max(0, months)
        else:
            return float('inf') if current_value < target_capital else 0
    
    if target_capital <= current_value:
        return 0
        
    num = (target_capital * r) + monthly_savings
    den = (current_value * r) + monthly_savings
    
    if den <= 0 or num <= 0:
        return float('inf') # Unable to reach target with current growth/savings vs drag
        
    months = np.log(num / den) / np.log(1 + r)
    
    return max(0, months)
