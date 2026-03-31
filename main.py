import logging
import numpy as np
from data.fetcher import fetch_data
from data.calibrator import calibrate
from data.diagnostics import plot_diagnostics
from simulation.engine import simulate

def run_pipeline(ticker: str):
    print(f"\nProcessing {ticker} pipeline...")
    
    # 1. Fetch
    df = fetch_data(ticker)
    
    if df.empty:
        print(f"Skipping calibration for {ticker} as no data was returned.")
        return
        
    # 2. Calibrate
    result = calibrate(df)
    
    # 3. Output Diagnostics
    plot_diagnostics(result, ticker)
    
    # 4. Simulation Engine
    modes = ["gbm", "student_t", "jump_diffusion"]
    for mode in modes:
        print(f"\n--- Simulating {mode.upper()} for {ticker} ---")
        paths, sim_res = simulate(result, n_paths=10000, n_days=252, mode=mode, seed=42)
        
        print(f"Final Prices -> Mean: {np.mean(sim_res.final_prices):.2f}")
        print(f"CAGR         -> Mean: {sim_res.cagr_mean * 100:.2f}%, Median: {sim_res.cagr_median * 100:.2f}%, Std: {sim_res.cagr_std * 100:.2f}%")
        print(f"Sharpe Ratio -> {sim_res.sharpe_ratio:.4f}")
        print(f"Max Drawdown -> Mean: {sim_res.max_drawdown_mean * 100:.2f}%")
        print(f"VaR (95%)    -> {sim_res.var_95 * 100:.2f}%")
        print(f"CVaR (95%)   -> {sim_res.cvar_95 * 100:.2f}%")
        print(f"CVaR (99%)   -> {sim_res.cvar_99 * 100:.2f}%")

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    
    tickers = ['SPY', 'KC=F']
    for t in tickers:
        run_pipeline(t)
