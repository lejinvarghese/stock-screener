# Portfolio Management - Quick Reference Card

## PyPortfolioOpt Cheat Sheet

### Basic Workflow
```python
from pypfopt import expected_returns, risk_models, EfficientFrontier

# 1. Get expected returns
mu = expected_returns.mean_historical_return(prices)
mu = expected_returns.ema_historical_return(prices, span=500)

# 2. Get risk model
S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

# 3. Optimize
ef = EfficientFrontier(mu, S)
ef.max_sharpe()  # or min_volatility(), efficient_return(), etc.

# 4. Get weights
weights = ef.clean_weights()

# 5. Performance
perf = ef.portfolio_performance(verbose=True)
```

### Quick Methods Reference

| Method | Purpose | When to Use |
|--------|---------|-------------|
| `max_sharpe()` | Best risk-adjusted return | Default choice |
| `min_volatility()` | Lowest risk | Conservative portfolio |
| `efficient_return(0.20)` | Target 20% return | Specific return goal |
| `efficient_risk(0.15)` | Target 15% volatility | Risk budget constraint |

### Expected Returns

```python
from pypfopt import expected_returns

mu = expected_returns.mean_historical_return(prices)           # Simple average
mu = expected_returns.ema_historical_return(prices, span=500)  # Recent data weighted
mu = expected_returns.capm_return(prices)                      # Market-based
```

### Risk Models

```python
from pypfopt import risk_models, CovarianceShrinkage

S = risk_models.sample_cov(prices)                    # Basic
S = risk_models.exp_cov(prices, span=180)             # Recent data weighted
S = CovarianceShrinkage(prices).ledoit_wolf()         # Best for most cases
S = CovarianceShrinkage(prices).oracle_approximating() # Alternative shrinkage
S = risk_models.semicovariance(prices)                # Downside only
```

### Alternative Optimizers

```python
from pypfopt import EfficientSemivariance, EfficientCVaR, HRPOpt

# Downside risk only
es = EfficientSemivariance(mu, returns)
es.min_semivariance()

# Tail risk (worst 5%)
ecvar = EfficientCVaR(mu, returns, beta=0.95)
ecvar.min_cvar()

# Hierarchical (no expected returns needed)
hrp = HRPOpt(returns)
weights = hrp.optimize()
```

### Constraints

```python
# Sector constraints
ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)

# Custom constraint
ef.add_constraint(lambda w: w[0] >= 0.05)  # Min 5% in first asset

# Objective functions
from pypfopt import objective_functions
ef.add_objective(objective_functions.L2_reg, gamma=0.1)
ef.add_objective(objective_functions.transaction_cost, w_prev=prev_weights, k=0.001)
```

### Discrete Allocation

```python
from pypfopt import DiscreteAllocation, get_latest_prices

latest_prices = get_latest_prices(prices)
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000)
allocation, leftover = da.lp_portfolio()
```

---

## yfinance Quick Reference

### Price Data
```python
import yfinance as yf

# Single ticker
data = yf.download('AAPL', start='2020-01-01')

# Multiple tickers
data = yf.download(['AAPL', 'GOOGL'], start='2020-01-01')['Adj Close']
```

### Fundamental Data
```python
ticker = yf.Ticker("AAPL")

# Quick metrics
info = ticker.info
pe = info['trailingPE']
roe = info['returnOnEquity']
debt_to_equity = info['debtToEquity']

# Financial statements
income = ticker.financials
balance = ticker.balance_sheet
cashflow = ticker.cashflow
```

### Common Metrics
```python
info = ticker.info

# Valuation
info['trailingPE']          # P/E ratio
info['priceToBook']         # P/B ratio
info['enterpriseToEbitda']  # EV/EBITDA

# Profitability
info['profitMargins']       # Net margin
info['returnOnEquity']      # ROE
info['returnOnAssets']      # ROA

# Liquidity
info['currentRatio']
info['debtToEquity']

# Growth
info['revenueGrowth']
info['earningsGrowth']

# Risk
info['beta']
```

---

## FFN Quick Reference (Already Installed)

```python
import ffn

stats = ffn.core.PerformanceStats(returns)
print(stats.display())

# Key metrics
stats.total_return
stats.cagr
stats.sharpe
stats.max_drawdown
```

---

## Recommended Additions

### pandas-ta (Technical Analysis)
```python
# Install: pip install pandas-ta
import pandas_ta as ta

df['RSI'] = ta.rsi(df['close'], length=14)
df['MACD'] = ta.macd(df['close'])['MACD_12_26_9']

# Or add to DataFrame
df.ta.rsi(length=14, append=True)
df.ta.macd(append=True)
```

### bt (Backtesting)
```python
# Install: pip install bt
import bt

strategy = bt.Strategy('EqualWeight', [
    bt.algos.RunMonthly(),
    bt.algos.SelectAll(),
    bt.algos.WeighEqually(),
    bt.algos.Rebalance()
])

test = bt.Backtest(strategy, data)
result = bt.run(test)
result.plot()
```

---

## Common Patterns

### 1. Standard Portfolio Optimization
```python
mu = expected_returns.ema_historical_return(prices)
S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
ef = EfficientFrontier(mu, S)
ef.add_objective(objective_functions.L2_reg, gamma=0.1)
ef.max_sharpe()
weights = ef.clean_weights()
```

### 2. Conservative Portfolio
```python
mu = expected_returns.mean_historical_return(prices)
S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
ef = EfficientFrontier(mu, S)
ef.min_volatility()
weights = ef.clean_weights()
```

### 3. Target Return Portfolio
```python
mu = expected_returns.ema_historical_return(prices)
S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
ef = EfficientFrontier(mu, S)
ef.efficient_return(target_return=0.20)  # 20%
weights = ef.clean_weights()
```

### 4. Downside Risk Portfolio
```python
mu = expected_returns.mean_historical_return(prices)
returns = expected_returns.returns_from_prices(prices)
es = EfficientSemivariance(mu, returns)
es.efficient_return(target_return=0.20)
weights = es.clean_weights()
```

### 5. Hierarchical Risk Parity
```python
returns = expected_returns.returns_from_prices(prices)
hrp = HRPOpt(returns)
weights = hrp.optimize()
```

### 6. With Constraints
```python
mu = expected_returns.ema_historical_return(prices)
S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
ef = EfficientFrontier(mu, S)

# Add constraints
ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
ef.add_constraint(lambda w: w >= 0.05)  # Min 5% per stock

ef.max_sharpe()
weights = ef.clean_weights()
```

### 7. Fundamental Screening + Optimization
```python
# Screen stocks
selected = []
for ticker in tickers:
    info = yf.Ticker(ticker).info
    if info['trailingPE'] < 30 and info['returnOnEquity'] > 0.15:
        selected.append(ticker)

# Optimize
data = yf.download(selected, start='2020-01-01')['Adj Close']
mu = expected_returns.ema_historical_return(data)
S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
ef = EfficientFrontier(mu, S)
ef.max_sharpe()
weights = ef.clean_weights()
```

---

## Troubleshooting

### Optimization Fails
```python
# Try different solvers
ef._solver = 'ECOS'  # or 'SCS', 'CVXOPT'
ef.max_sharpe()

# Or simplify
ef = EfficientFrontier(mu, S)  # No objectives
ef.max_sharpe()
```

### All Weights Zero After Cleaning
```python
# Lower the cutoff
weights = ef.clean_weights(cutoff=0.001)  # Default is 0.0001
```

### Non-positive Definite Matrix
```python
from pypfopt import risk_models
S = risk_models.fix_nonpositive_semidefinite(S)
```

---

## Performance Comparison

```python
# Compare different approaches
results = []
for method in ['max_sharpe', 'min_vol', 'hrp']:
    if method == 'max_sharpe':
        ef = EfficientFrontier(mu, S)
        ef.max_sharpe()
    elif method == 'min_vol':
        ef = EfficientFrontier(mu, S)
        ef.min_volatility()
    else:
        ef = HRPOpt(returns)
        ef.optimize()

    perf = ef.portfolio_performance()
    results.append((method, perf))

for method, (ret, vol, sharpe) in results:
    print(f"{method:15} Return:{ret:>8.2%} Risk:{vol:>8.2%} Sharpe:{sharpe:>6.2f}")
```

---

## Files Created

1. **LIBRARY_RESEARCH.md** - Comprehensive reference
2. **PORTFOLIO_EXAMPLES.py** - 12 working examples
3. **QUICK_REFERENCE.md** - This cheat sheet

## Installation Commands

```bash
# Essential
pip install pandas-ta bt empyrical

# Optional
pip install vectorbt riskfolio-lib
```
