# Development Log

Chronological record of what was built, bugs caught, and lessons learned at each stage.

## Task 1-3: Simulation Engine & Dashboard

**Built:** Core simulation engine with three stochastic models (GBM, Student-t, Merton Jump-Diffusion), calibration pipeline from yfinance data, diagnostic plots (QQ, rolling vol, histograms), and a Dash dashboard with the SIMULATION tab.

**Key decision:** Using Itô correction for GBM drift. The historical arithmetic mean of log returns (18.1%) is not the correct drift parameter for the SDE — the corrected value (11.4%) accounts for the variance drag in continuous compounding.

**Validation:** Simulation CAGR median (~11.7%) matches the corrected drift, confirming the model is self-consistent. The mean CAGR (~19.3%) is higher due to log-normal right skew, which is expected behaviour.

## Task 4: Position Tracker

**Built:** MY POSITION tab with metric cards, 1-month P&L histogram, scenario fan chart, covered call calculator, and time-to-target estimator.

**Bug caught — KC=F vs COFF.MI price mismatch:** The initial implementation used `cal.last_price` (KC=F futures price, ~$294) as the current price for a position denominated in EUR with an ETC priced at ~€52. This made the position value show €2,060 instead of ~€362, and inflated all P&L projections by ~5.5x.

**Fix:** Introduced return-based scaling. The simulation runs on KC=F calibration data (good vol/drift estimates) but the position tab maps percentage returns onto the live COFF.MI price from Borsa Italiana. The ETC ticker was identified by searching for the ISIN (JE00BN7KB557) on Yahoo Finance — `COFF.MI` (Milan exchange, EUR) returned a price within €0.10 of the Nordnet quote.

**Lesson:** Always verify that the instrument you're simulating matches the instrument you're trading. Price levels, currencies, and exchange venues all matter.

## Task 5: Strategy Backtester

**Built:** STRATEGY tab with historical equity curve, Monte Carlo strategy overlay, and benchmark comparison. Four signal generators (buy & hold, SMA crossover, momentum, mean reversion) that work identically on historical prices and simulated paths.

**Bug caught — look-ahead bias in momentum backtest:** The initial implementation used `signals[t] * price_returns[t]`, meaning the signal computed from today's close got credit for today's return. For momentum, a big up day that flips the signal from 0 to 1 simultaneously counts as profit. This showed a 51.5% CAGR for 63-day momentum on KC=F.

**Fix:** Lagged signals by one day: `signals[t-1] * price_returns[t]`. After the fix, momentum CAGR dropped to 21.1% — still above buy-and-hold (19.0%) but no longer suspiciously high. The Monte Carlo overlay then showed a median CAGR of -0.51% and P(Beat B&H) of 35.5%, confirming that the historical outperformance was likely luck rather than structural edge.

**Also caught:** Antigravity left its internal reasoning as comments in production code (lines debating the signal alignment convention). Cleaned up before verification.

**Key finding from backtesting results:**
- SMA Crossover (20/50) on coffee: CAGR 3.3% vs B&H 19.0%. Destroyed by whipsaws on a 36% vol asset.
- Momentum (63d) on coffee: CAGR 21.1% vs B&H 19.0%. Marginal edge, but MC overlay shows P(Beat B&H) = 35.5%.
- **Conclusion: No simple price-based trading rule adds structural value on a single high-volatility commodity.** Buy and hold is the dominant strategy at this scale.

## Task 6: Multi-Asset Portfolio

**Built:** PORTFOLIO tab with dual momentum across 10 ETFs, inverse volatility sizing, monthly rebalancing with contributions, and benchmark comparison against SPY and 60/40.

**Bug caught — universe composition:** The initial universe included USO (crude oil), UNG (natural gas), and SLV (silver). These single-commodity ETFs have extreme volatility and contango drag that dominates momentum rankings. Result: -5.1% CAGR, 52.9% max drawdown, Sharpe -0.39. The strategy was correct; the inputs were wrong.

**Fix:** Removed USO, UNG, SLV. Added TIP (inflation-protected bonds). New universe: 5 equity, 3 bond, 2 commodity ETFs. Also changed defaults from 63d lookback / top 5 to 126d lookback / top 4 to reduce turnover.

**Results after fix (best configuration — 126d lookback, top 4, inverse vol):**

| Metric | Portfolio | SPY B&H | 60/40 |
|--------|-----------|---------|-------|
| CAGR | 7.2% | 10.2% | 5.9% |
| Volatility | 12.7% | 16.9% | 10.8% |
| Sharpe | 0.26 | 0.39 | 0.17 |
| Max Drawdown | 13.2% | 17.2% | 10.9% |

**Interpretation:** The dual momentum portfolio delivers higher returns than 60/40 with better drawdown protection than SPY. It doesn't beat SPY on raw return, but the max drawdown reduction (13.2% vs 17.2%) represents real emotional and financial value — on a €7k portfolio, that's a ~€280 difference at the worst point. The strategy is most valuable for someone who would panic-sell during a 17% drawdown but can hold through 13%.

**Key lesson from the full project:** Universe design > signal design > parameter tuning. The single biggest improvement came from removing three bad ETFs, not from optimising lookback periods or top-N selection.
