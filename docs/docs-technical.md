# Technical Decisions & Mathematical Notes

This document records the key mathematical and architectural decisions made during development, along with the reasoning behind each one.

## 1. Calibration

### Log Returns
All calibration uses log returns: `ln(P_t / P_{t-1})`. This ensures additivity across time periods and compatibility with the continuous-time GBM framework.

### Annualisation
252 trading days used throughout:
- `mu_historical = mean(log_returns) × 252`
- `sigma = std(log_returns) × sqrt(252)`

### Itô Correction
The drift term passed to the GBM simulator is:
```
mu_gbm = mu_historical - 0.5 × sigma²
```
This corrects for the difference between the arithmetic mean of log returns (what we observe historically) and the drift parameter in the GBM SDE. Without this, `E[S_T]` would be systematically too high.

For KC=F coffee futures with `mu_historical = 0.1805` and `sigma = 0.3653`:
```
mu_gbm = 0.1805 - 0.5 × 0.3653² = 0.1805 - 0.0667 = 0.1138
```

### Normality Testing
Two tests are run:
- **Jarque-Bera** on full history — KC=F rejects normality (p=0.0000) due to regime behaviour
- **Shapiro-Wilk** on last 252 days — cannot reject (p=0.1366) for the recent window alone

This informs model selection: GBM is a reasonable baseline for short horizons, but Merton Jump-Diffusion better captures the full distribution.

## 2. Simulation Models

### GBM
```
S(t+dt) = S(t) × exp(mu_gbm × dt + sigma × sqrt(dt) × Z)
Z ~ N(0,1)
```
Vectorised: generates all paths simultaneously using a (n_paths, n_days) matrix of standard normal draws.

### Student-t
Same as GBM but replaces Z with:
```
Z = t_draw / sqrt(df / (df - 2))
```
where `t_draw ~ t(df=3.0)`. The normalisation ensures unit variance so the vol calibration stays valid. Fat tails only become visible at the 99th percentile — at the 95th, Student-t and GBM are nearly identical because the base sigma (36.5%) is already high.

### Merton Jump-Diffusion
```
S(t+dt) = S(t) × exp((mu_gbm - lambda×kappa)×dt + sigma×sqrt(dt)×Z + J×N_t)
kappa = exp(mu_j + 0.5×sigma_j²) - 1
N_t ~ Poisson(lambda×dt)
J ~ N(mu_j, sigma_j) when N_t > 0
```
Default parameters: lambda=5 (5 jumps/year), mu_j=-0.02 (slightly negative average jump), sigma_j=0.04. The `lambda×kappa` term in the drift compensates for the expected jump contribution, keeping the overall drift consistent with the calibrated value.

## 3. Risk Metrics

- **CAGR**: `exp(log_return × 252/n_days) - 1` — annualised geometric return
- **VaR 95%/99%**: 5th/1st percentile of total log returns across all paths
- **CVaR 95%/99%**: Mean of returns below the corresponding VaR threshold — expected loss in the tail
- **Max Drawdown**: `max((running_max - price) / running_max)` per path — worst peak-to-trough decline
- **Sharpe**: `(mean(daily_excess_return) / std(daily_excess_return)) × sqrt(252)` with rf=4.5%

The mean/median CAGR gap (~7% for KC=F) is a direct consequence of log-normal asymmetry. The median is always the better anchor for expected outcomes because the mean is pulled up by the right tail.

## 4. Position Tracking: Return-Based ETC Scaling

### The Problem
The simulation calibrates on KC=F (coffee futures, ~$294 c/lb in USD). The actual trading instrument is WisdomTree Coffee ETC (COFF.MI, ~€52 on Borsa Italiana). These track the same underlying commodity but at completely different price levels and in different currencies.

### The Solution
Extract percentage returns from the simulation and apply them to the ETC price:
```python
returns = paths[:, h] / paths[:, 0]      # relative return from KC=F simulation
projected_prices = etc_price * returns     # map onto COFF.MI price
euro_pl = (projected_prices - entry_price) * units
```

This preserves the calibrated volatility and drift characteristics while producing euro-denominated P&L that matches the actual position.

### ETC Price Source
`COFF.MI` (Borsa Italiana, EUR-denominated) via yfinance. Verified against the Nordnet quoted price with <€0.10 discrepancy (bid/ask spread).

## 5. Backtesting: Signal Alignment

### The Convention
```python
strategy_returns[t] = signals[t-1] * price_returns[t]
```

`signals[t]` is computed using prices up to and including `prices[t]`. The one-day lag means: "I observe today's close, decide my position overnight, and I'm exposed to tomorrow's close-to-close return."

### Why This Matters
Without the lag (`signals[t] * price_returns[t]`), the backtester gives the strategy credit for the price move that triggered the signal. For momentum strategies this is catastrophic — a big up day that flips the signal from 0 to 1 is simultaneously captured as return. This produced a 51.5% CAGR for 63-day momentum on KC=F. After the lag fix: 21.1%.

For Buy & Hold (signal always 1), the lag makes no difference. For SMA crossover, the effect is small because the moving averages change slowly. The bias is worst for momentum and mean reversion strategies where single-day moves can flip signals.

### Transaction Costs
Applied on the day of signal change. Cost is proportional to the absolute value of the trade:
- Single-asset (Nordnet): 10 bps per trade
- Multi-asset (IBKR): 5 bps per trade

Sharpe ratio subtracts the risk-free rate only on days when the strategy holds a position (signal > 0), which is conservative — it assumes idle cash earns nothing.

## 6. Multi-Asset Portfolio

### Dual Momentum
Combines two filters:
1. **Cross-sectional momentum**: Rank all assets by trailing N-day return, select top K
2. **Absolute momentum**: Only hold assets with positive trailing return

The multiplication of these two signals means an asset must be both relatively strong AND in an absolute uptrend to be held. When few assets qualify, the portfolio holds more cash — this is the drawdown protection mechanism.

### Inverse Volatility Sizing
Selected assets are weighted inversely proportional to their recent (21-day) annualised volatility:
```
weight_i = (1 / vol_i) / sum(1 / vol_j for j in selected)
```
This naturally allocates more to bonds (vol ~10%) and less to commodities (vol ~20-30%), creating a risk-balanced portfolio without explicitly targeting a volatility level.

### Universe Design
The universe was revised after initial results showed -5.1% CAGR and 52.9% max drawdown.

**Removed**: USO (crude oil), UNG (natural gas), SLV (silver) — single-commodity ETFs with extreme contango drag and volatility that dominates momentum rankings. USO and UNG in particular suffer from persistent negative roll yield that destroys long-term returns regardless of spot price direction.

**Final universe (10 assets)**:
- Equities: SPY, QQQ, EFA, EEM, VNQ
- Bonds: TLT, IEF, TIP
- Commodities: GLD, DBC

This gives a balanced 5/3/2 split across asset classes. The momentum signal rotates between them based on regime — equities in risk-on periods, bonds and gold in risk-off.

### Rebalancing
Monthly (every 21 trading days). On each rebalance:
1. Add monthly contribution to cash
2. Compute target weights
3. Calculate trades needed (target - current allocation)
4. Apply transaction costs on gross trade value
5. Update holdings in share counts

Between rebalances, holdings are fixed — weights drift with prices. This is realistic for a retail account with monthly contributions.

### Performance Return Calculation
Daily returns are adjusted for contributions using time-weighted return methodology:
```
return_t = equity_t / (equity_{t-1} + contribution_t) - 1
```
CAGR is computed from the compounded product of these adjusted daily returns, ensuring contributions don't inflate the performance metric.

## 7. Calibration Results (KC=F, 5-Year History)

```
mu_historical:    0.1805  (18.1% annualised arithmetic mean)
mu_gbm:           0.1138  (11.4% Itô-corrected drift)
sigma:            0.3653  (36.5% annualised volatility)
skewness:         0.0717  (near-symmetric)
excess_kurtosis:  0.7006  (mild fat tails)
JB p-value:       0.0000  (reject normality — regime behaviour)
SW p-value:       0.1366  (recent window alone: can't reject)
```

### Simulation Output (252 days, 10,000 paths, seed=42)

| Model | CAGR Mean | CAGR Median | CVaR 95% | CVaR 99% |
|-------|-----------|-------------|----------|----------|
| GBM | 19.30% | 11.66% | -64.76% | -86.03% |
| Student-t | 19.54% | 12.31% | -63.41% | -92.44% |
| Jump-Diffusion | 19.38% | 11.95% | -67.86% | -90.92% |

The ~7% gap between mean and median CAGR is the log-normal asymmetry in action. Student-t fat tails show at CVaR 99% (-92.44% vs GBM's -86.03%) but not at CVaR 95% — the base volatility is high enough that 95th percentile losses are similar across models.
