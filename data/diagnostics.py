import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

def plot_diagnostics(result, ticker: str, output_dir: str = 'data/plots'):
    """
    Generates diagnostic plots and prints a summary table.
    
    Args:
        result (CalibrationResult): The calibrated outputs.
        ticker (str): The ticker symbol.
        output_dir (str): Directory to save the plots.
    """
    # Ensure plots directory exists relative to this file
    base_dir = os.path.dirname(os.path.dirname(__file__))
    plots_path = os.path.join(base_dir, output_dir)
    os.makedirs(plots_path, exist_ok=True)
    
    safe_ticker = ticker.replace('=', '_')
    
    returns = result.log_returns.dropna()
    
    # Plot 1: Histogram + Normal Fit
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=50, density=True, alpha=0.6, color='b', label='Log Returns')
    
    # Fit normal distribution
    mu, std = stats.norm.fit(returns)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label=f'Normal Fit\n(mu={mu:.4f}, std={std:.4f})')
    
    plt.title(f'{ticker} Log Returns Distribution')
    plt.xlabel('Log Returns')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, f'{safe_ticker}_histogram.png'))
    plt.close()

    # Plot 2: Rolling Volatility
    plt.figure(figsize=(10, 6))
    result.rolling_vol.dropna().plot(color='r', title=f'{ticker} 252-Day Rolling Annualised Volatility')
    plt.xlabel('Date')
    plt.ylabel('Annualised Volatility')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, f'{safe_ticker}_rolling_vol.png'))
    plt.close()

    # Plot 3: QQ-Plot
    plt.figure(figsize=(10, 6))
    stats.probplot(returns, dist="norm", plot=plt)
    plt.title(f'{ticker} QQ-Plot')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, f'{safe_ticker}_qqplot.png'))
    plt.close()

    # Print Summary Table
    print("\n" + "="*60)
    print(f"Calibration Summary: {ticker}")
    print("="*60)
    print(f"{'Metric':<25} | {'Value'}")
    print("-" * 60)
    print(f"{'Historical Drift (mu)':<25} | {result.mu_historical:.4f}")
    print(f"{'Itô-Corrected Drift (GBM)':<25} | {result.mu_gbm:.4f}")
    print(f"{'Annualised Volatility':<25} | {result.sigma:.4f}")
    print(f"{'Skewness':<25} | {result.skewness:.4f}")
    print(f"{'Excess Kurtosis':<25} | {result.excess_kurtosis:.4f}")
    print(f"{'Jarque-Bera p-value':<25} | {result.jb_pvalue:.4f}")
    if not np.isnan(result.sw_pvalue):
        print(f"{'Shapiro-Wilk (252d) p-val':<25} | {result.sw_pvalue:.4f}")
    print("="*60 + "\n")
