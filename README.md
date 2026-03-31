# Monte Carlo Portfolio Simulator

A quantitative finance platform for simulating asset prices, backtesting trading strategies, and managing multi-asset portfolios. Built in Python with a Dash/Plotly dashboard.

The project started as a Monte Carlo simulation engine for coffee futures and evolved into a full portfolio analysis toolkit covering single-asset simulation, position tracking, strategy backtesting, and multi-asset systematic portfolio management.

![Dashboard Screenshot](docs/screenshots/simulation_tab.png)

## What It Does

The platform has four modules, each exposed as a dashboard tab:

### 1. Simulation Engine
Generates forward-looking price paths using three stochastic models calibrated from historical market data:

- **Geometric Brownian Motion (GBM)** — baseline model with Itô-corrected drift
- **Student-t** — fat-tailed returns (df=3.0), captures extreme moves invisible at the 95th percentile but visible at the 99th
- **Merton Jump-Diffusion** — discrete jumps overlaid on GBM, suited for commodities with regime shifts

Outputs include fan charts (5th/25th/50th/75th/95th percentile paths), final price distributions, and risk metrics (CAGR, VaR, CVaR, Sharpe, max drawdown).

### 2. Position Tracker
Maps simulation results onto a real trading position. Designed for tracking a WisdomTree Coffee ETC (COFF.MI) position on a European broker account.

Key feature: the simulation engine calibrates on KC=F (coffee futures — deep liquidity, long history) but the position tab uses **return-based scaling** to map simulated percentage returns onto the actual ETC price from Borsa Italiana. This avoids the price-level mismatch between raw futures (~$294) and the ETC (~€52).

Includes a covered call viability calculator and a time-to-target income estimator with monthly savings projections.

### 3. Strategy Backtester
Tests trading rules against historical data and validates them across 10,000 Monte Carlo simulated paths.

Four signal generators:
- **Buy & Hold** — baseline
- **SMA Crossover** — fast/slow moving average (configurable windows)
- **Momentum** — N-day trailing return threshold
- **Mean Reversion** — z-score entry/exit relative to rolling mean

Each strategy is tested in two ways:
1. **Historical backtest** on real price data, producing an equity curve, Sharpe, drawdown, trade stats, and comparison against buy-and-hold
2. **Monte Carlo overlay** applying the same rule across all simulated paths, producing a distribution of outcomes and the probability of beating buy-and-hold

The backtester uses a **one-day signal lag** (`signals[t-1] * returns[t]`) to prevent look-ahead bias — a bug that was caught during development when a momentum strategy showed an unrealistic 51% CAGR that dropped to 21% after the fix.

### 4. Multi-Asset Portfolio
Runs a systematic portfolio across 10 liquid ETFs spanning equities, bonds, and commodities. Designed for a real deployment scenario on Interactive Brokers with monthly contributions.

**Strategy: Dual Momentum** — combines cross-sectional momentum (rank assets by trailing return, select top N) with an absolute momentum filter (only hold assets with positive trailing returns; otherwise sit in cash). Sized by inverse volatility so lower-vol assets get larger allocations.

Monthly rebalancing with configurable parameters (lookback period, top N, transaction costs). Benchmarked against SPY buy-and-hold and a 60/40 portfolio.

The best-performing configuration (126-day lookback, top 4, inverse volatility sizing) produced a 7.2% CAGR with 13.2% max drawdown over 5 years — lower return than SPY (10.2%) but significantly better drawdown protection (SPY: 17.2%).

## Key Technical Decisions

### Itô Correction
The GBM simulation uses `mu_gbm = mu_historical - 0.5 * sigma²` rather than raw historical drift. This accounts for the difference between arithmetic and geometric returns in continuous-time models. Without this correction, simulated paths would systematically overestimate expected returns.

### Return-Based ETC Scaling
The simulation engine calibrates on KC=F futures data (good liquidity, long history) but the position tracker needs prices in EUR for a specific ETC. Rather than simulating the ETC directly (poor yfinance data coverage), the system computes percentage returns from the KC=F simulation and applies them to the live COFF.MI price. The volatility characteristics transfer between instruments since the ETC tracks the same underlying index.

### Look-Ahead Bias Prevention
The backtester applies `signals[t-1] * price_returns[t]` — yesterday's signal determines today's exposure. This prevents the strategy from "seeing" a price move and simultaneously getting credit for it. This matters most for momentum strategies where a big daily move can flip the signal and inflate backtest returns. The fix was validated by comparing a momentum strategy's CAGR before (51.5%) and after (21.1%) the correction.

### Universe Selection
The multi-asset portfolio initially included single-commodity ETFs (USO, UNG, SLV) which produced a -5.1% CAGR and 52.9% max drawdown. These instruments have extreme contango drag and volatility that distorts momentum rankings. Replacing them with broad commodity exposure (DBC, GLD) and adding TIPS brought the portfolio to 7.2% CAGR with 13.2% max drawdown. The lesson: universe design matters more than signal design.

### Transaction Cost Modelling
Single-asset backtests use 10 bps per trade (realistic for Nordnet/European broker pricing). The multi-asset portfolio uses 5 bps (realistic for IBKR tiered pricing on US ETFs). Costs are applied on the day of signal change to the absolute value of the trade.

## Tech Stack

- **Language:** Python 3.11+
- **Data:** yfinance (KC=F, COFF.MI, multi-asset ETF universe)
- **Simulation:** NumPy (vectorised path generation, no loops over time steps)
- **Dashboard:** Dash + Plotly + dash-bootstrap-components (CYBORG dark theme)
- **Stats:** SciPy (distribution fitting, normality tests)

## Project Structure

```
/data
  fetcher.py          # yfinance data fetching (single + multi-asset)
  calibrator.py       # Log-return calibration, annualisation
  diagnostics.py      # QQ plots, rolling vol, histograms
/simulation
  engine.py           # simulate() — main entry point
  models.py           # GBM, Student-t, Merton jump-diffusion paths
  results.py          # SimulationResult dataclass
/strategy
  rules.py            # Signal generators (pure functions: prices in, signals out)
  backtester.py       # Historical backtest engine
  mc_strategy.py      # Monte Carlo strategy overlay
/portfolio
  universe.py         # ETF universe definition
  signals.py          # Cross-sectional + absolute momentum
  sizing.py           # Inverse volatility + equal weight
  rebalancer.py       # Portfolio backtest with contributions
  results.py          # PortfolioResult dataclass
/dashboard
  app.py              # Runs on localhost:8050
  layout.py           # Four tabs: SIMULATION, MY POSITION, STRATEGY, PORTFOLIO
  charts.py           # All Plotly figure generators
  callbacks.py        # All interactivity
  position.py         # Position tracking + ETC price mapping
/tests
  test_fetcher.py
  test_calibrator.py
  test_engine.py
  test_models.py
main.py
requirements.txt
```

## Running It

```bash
pip install -r requirements.txt
python dashboard/app.py
```

Dashboard runs at `http://localhost:8050`.

## Development Approach

This project was built using AI-assisted development (Claude + Antigravity IDE), with a focus on mathematical verification at each stage. Every simulation model, backtest engine, and portfolio function was reviewed for correctness before being integrated into the dashboard. Key catches during review included the KC=F vs COFF.MI price-level mismatch, the momentum look-ahead bias, and the universe composition problem — all identified through analytical reasoning about expected outputs before examining code.

## License

MIT
