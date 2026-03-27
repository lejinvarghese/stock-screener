# Python Portfolio & Analysis Libraries - Comprehensive Research

## Currently Installed Libraries

```
pyportfolioopt==1.5.6
yfinance==0.2.65
ffn==1.1.2
tradingview-ta==3.3.0
pandas==2.3.1
numpy==2.3.1
scikit-learn==1.7.0
matplotlib==3.10.3
seaborn==0.13.2
plotly==5.24.1
```

---

## 1. PyPortfolioOpt (INSTALLED) - Portfolio Optimization

### Overview
- **Version**: 1.5.6
- **GitHub**: https://github.com/robertmartin8/PyPortfolioOpt
- **Docs**: https://pyportfolioopt.readthedocs.io
- **Purpose**: Modern Portfolio Theory optimization

### Expected Returns Methods

```python
from pypfopt import expected_returns

# Method 1: Mean Historical Return (simple average)
mu = expected_returns.mean_historical_return(prices, frequency=252)

# Method 2: EMA Historical Return (exponentially weighted - recent data matters more)
mu = expected_returns.ema_historical_return(prices, span=500, frequency=252)

# Method 3: CAPM Return (based on market risk premium)
mu = expected_returns.capm_return(
    prices,
    market_prices=None,  # If None, uses equal-weighted portfolio
    risk_free_rate=0.02,
    frequency=252
)

# Helper functions
returns = expected_returns.returns_from_prices(prices)
prices = expected_returns.prices_from_returns(returns)
```

### Risk Models

```python
from pypfopt import risk_models, CovarianceShrinkage

# 1. Sample Covariance (basic)
S = risk_models.sample_cov(prices, frequency=252)

# 2. Exponentially-weighted Covariance (recent data weighted higher)
S = risk_models.exp_cov(prices, span=180, frequency=252)

# 3. Semicovariance (downside risk only)
S = risk_models.semicovariance(prices, benchmark=0, frequency=252)

# 4. Ledoit-Wolf Shrinkage (reduces estimation error) - RECOMMENDED
S = CovarianceShrinkage(prices).ledoit_wolf()

# 5. Oracle Approximating Shrinkage
S = CovarianceShrinkage(prices).oracle_approximating()

# 6. Minimum Covariance Determinant (robust to outliers)
S = risk_models.min_cov_determinant(prices, frequency=252)

# Helper: Fix non-positive semidefinite matrices
S = risk_models.fix_nonpositive_semidefinite(S)
```

### Optimizer Classes

#### A. EfficientFrontier (Main Class)

```python
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions

# Initialize
ef = EfficientFrontier(
    expected_returns=mu,
    cov_matrix=S,
    weight_bounds=(0, 1),  # (min, max) for each asset
    solver='OSQP',  # Options: 'OSQP', 'ECOS', 'SCS', 'CVXOPT'
    verbose=False
)

# Optimization Methods
ef.max_sharpe(risk_free_rate=0.02)                # 1. Maximum Sharpe Ratio
ef.min_volatility()                                # 2. Minimum Volatility
ef.efficient_return(target_return=0.20)            # 3. Target Return
ef.efficient_risk(target_volatility=0.15)          # 4. Target Risk
ef.max_quadratic_utility(risk_aversion=1)          # 5. Utility Function
```

#### B. EfficientSemivariance (Downside Risk)

```python
from pypfopt import EfficientSemivariance

returns = expected_returns.returns_from_prices(prices)
es = EfficientSemivariance(mu, returns)
es.min_semivariance()                              # Minimize downside risk
es.efficient_return(target_return=0.20)
```

#### C. EfficientCVaR (Conditional Value at Risk)

```python
from pypfopt import EfficientCVaR

ecvar = EfficientCVaR(mu, returns, beta=0.95)      # 95th percentile
ecvar.min_cvar()                                   # Minimize tail risk
ecvar.efficient_return(target_return=0.20)
```

#### D. EfficientCDaR (Conditional Drawdown at Risk)

```python
from pypfopt import EfficientCDaR

ecdar = EfficientCDaR(mu, returns, beta=0.95)
ecdar.min_cdar()                                   # Minimize drawdown risk
ecdar.efficient_return(target_return=0.20)
```

#### E. HRPOpt (Hierarchical Risk Parity)

```python
from pypfopt import HRPOpt

hrp = HRPOpt(returns)
weights = hrp.optimize()                           # No optimization needed, returns weights
perf = hrp.portfolio_performance(verbose=True)
```

#### F. CLA (Critical Line Algorithm)

```python
from pypfopt import CLA

cla = CLA(mu, S)
cla.max_sharpe()
cla.min_volatility()
frontier = cla.efficient_frontier()                # Get entire frontier
```

#### G. Black-Litterman Model

```python
from pypfopt import BlackLittermanModel

# Market-implied prior
market_caps = {'AAPL': 2e12, 'GOOGL': 1.5e12, 'MSFT': 1.8e12}
bl = BlackLittermanModel(S, pi='market', market_caps=market_caps)

# Add investor views
viewdict = {
    'AAPL': 0.20,    # Expect 20% return
    'GOOGL': -0.05   # Expect -5% return
}
bl_returns = bl.bl_returns()
bl_cov = bl.bl_cov()

# Use with EfficientFrontier
ef = EfficientFrontier(bl_returns, bl_cov)
ef.max_sharpe()
```

### Objective Functions (Constraints)

```python
# L2 Regularization (promotes diversification)
ef.add_objective(objective_functions.L2_reg, gamma=0.1)

# Transaction Costs
ef.add_objective(
    objective_functions.transaction_cost,
    w_prev=previous_weights,
    k=0.001  # 0.1% transaction cost
)

# Ex-ante Tracking Error (minimize deviation from benchmark)
ef.add_objective(
    objective_functions.ex_ante_tracking_error,
    cov_matrix=S,
    benchmark_weights=benchmark
)

# Custom Objective
def custom_objective(weights, expected_returns):
    return -weights @ expected_returns

ef.add_objective(custom_objective, expected_returns=mu)
```

### Constraints

```python
# General constraint
ef.add_constraint(lambda w: w[0] >= 0.05)  # Min 5% in first asset

# Sector constraints
sector_mapper = {
    'AAPL': 'Tech', 'GOOGL': 'Tech', 'MSFT': 'Tech',
    'JPM': 'Finance', 'BAC': 'Finance'
}
sector_lower = {'Tech': 0.1}   # Min 10% in tech
sector_upper = {'Tech': 0.5}   # Max 50% in tech
ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
```

### Discrete Allocation

```python
from pypfopt import DiscreteAllocation, get_latest_prices

# Get final weights
weights = ef.clean_weights(cutoff=0.01)

# Get latest prices
latest_prices = get_latest_prices(prices)

# Allocate to whole shares
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000)

# Method 1: LP (linear programming - optimal)
allocation, leftover = da.lp_portfolio()

# Method 2: Greedy (faster)
allocation, leftover = da.greedy_portfolio()

print(f"Buy: {allocation}")
print(f"Leftover: ${leftover:.2f}")
```

### Performance Analysis

```python
# Get portfolio metrics
expected_return, volatility, sharpe = ef.portfolio_performance(
    verbose=True,
    risk_free_rate=0.02
)

print(f"Expected Annual Return: {expected_return*100:.2f}%")
print(f"Annual Volatility: {volatility*100:.2f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")
```

### Visualization

```python
from pypfopt import plotting

# Plot efficient frontier
plotting.plot_efficient_frontier(
    ef,
    ef_param='return',
    show_assets=True,
    filename='frontier.png'
)

# Plot covariance matrix
plotting.plot_covariance(S, plot_correlation=False)

# Plot dendrogram (for HRP)
plotting.plot_dendrogram(hrp)

# Plot weights
plotting.plot_weights(weights)
```

---

## 2. yfinance (INSTALLED) - Market Data & Fundamentals

### Price Data

```python
import yfinance as yf

# Single ticker
ticker = yf.Ticker("AAPL")
hist = ticker.history(period="1y")  # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

# Multiple tickers
data = yf.download(['AAPL', 'GOOGL', 'MSFT'], start='2020-01-01', end='2024-01-01')
```

### Fundamental Data Available

```python
ticker = yf.Ticker("AAPL")

# Financial Statements
income = ticker.financials              # Annual income statement
quarterly_income = ticker.quarterly_financials
balance = ticker.balance_sheet          # Annual balance sheet
quarterly_balance = ticker.quarterly_balance_sheet
cashflow = ticker.cashflow              # Annual cash flow
quarterly_cashflow = ticker.quarterly_cashflow

# Key Metrics from .info
info = ticker.info
metrics = {
    # Valuation
    'trailingPE': info.get('trailingPE'),           # P/E ratio
    'forwardPE': info.get('forwardPE'),
    'priceToBook': info.get('priceToBook'),         # P/B ratio
    'priceToSalesTrailing12Months': info.get('priceToSalesTrailing12Months'),
    'enterpriseToRevenue': info.get('enterpriseToRevenue'),
    'enterpriseToEbitda': info.get('enterpriseToEbitda'),

    # Profitability
    'profitMargins': info.get('profitMargins'),     # Net margin
    'grossMargins': info.get('grossMargins'),
    'ebitdaMargins': info.get('ebitdaMargins'),
    'operatingMargins': info.get('operatingMargins'),
    'returnOnEquity': info.get('returnOnEquity'),   # ROE
    'returnOnAssets': info.get('returnOnAssets'),   # ROA

    # Liquidity
    'currentRatio': info.get('currentRatio'),
    'quickRatio': info.get('quickRatio'),
    'debtToEquity': info.get('debtToEquity'),

    # Cash Flow
    'freeCashflow': info.get('freeCashflow'),
    'operatingCashflow': info.get('operatingCashflow'),

    # Growth
    'revenueGrowth': info.get('revenueGrowth'),     # QoQ
    'earningsGrowth': info.get('earningsGrowth'),   # QoQ
    'earningsQuarterlyGrowth': info.get('earningsQuarterlyGrowth'),

    # Dividend
    'dividendYield': info.get('dividendYield'),
    'dividendRate': info.get('dividendRate'),
    'payoutRatio': info.get('payoutRatio'),
    'fiveYearAvgDividendYield': info.get('fiveYearAvgDividendYield'),

    # Risk
    'beta': info.get('beta'),

    # Other
    'marketCap': info.get('marketCap'),
    'enterpriseValue': info.get('enterpriseValue'),
    'sharesOutstanding': info.get('sharesOutstanding'),
    'averageVolume': info.get('averageVolume'),
}

# Analyst Data
recommendations = ticker.recommendations          # Buy/Sell ratings
price_targets = ticker.analyst_price_targets      # Price targets
earnings_estimate = ticker.earnings_estimate
revenue_estimate = ticker.revenue_estimate

# Holders
major_holders = ticker.major_holders
institutional_holders = ticker.institutional_holders

# Calendar
earnings_dates = ticker.earnings_dates
calendar = ticker.calendar

# News
news = ticker.news
```

---

## 3. Alternative Portfolio Libraries

### Riskfolio-Lib (Advanced - Not Installed)

**Install**: `pip install riskfolio-lib`

**Features vs PyPortfolioOpt**:
- ✅ 16+ optimization methods (vs 5)
- ✅ Factor models (Fama-French)
- ✅ Risk parity
- ✅ Robust optimization
- ✅ ESG constraints
- ✅ Better visualization
- ❌ More complex API
- ❌ Heavier dependencies

**When to Use**:
- Advanced risk management
- Multi-period optimization
- Factor investing
- Robust/stress-testing

**Code Example**:
```python
import riskfolio as rp

port = rp.Portfolio(returns=returns)
port.assets_stats(method_mu='hist', method_cov='hist')

# Risk measures: 'MV', 'CVaR', 'EVaR', 'CDaR', 'EDaR', 'UCI', 'WR'
# Objectives: 'MinRisk', 'Sharpe', 'Utility', 'MaxRet'
w1 = port.optimization(model='Classic', rm='MV', obj='Sharpe')
w2 = port.optimization(model='Classic', rm='CVaR', obj='Sharpe')
w3 = port.rp_optimization(model='Classic', rm='MV')  # Risk Parity

# Plot
ax = rp.plot_frontier(w_frontier, mu, cov, returns, rm='MV')
```

### Empyrical (Metrics - Not Installed)

**Install**: `pip install empyrical`

**Features**:
- Performance metrics only (no optimization)
- Alpha, beta, Sharpe, Sortino, Calmar
- Drawdown analysis
- Compatible with pandas

**Code Example**:
```python
import empyrical as emp

sharpe = emp.sharpe_ratio(returns)
sortino = emp.sortino_ratio(returns)
max_dd = emp.max_drawdown(returns)
calmar = emp.calmar_ratio(returns)
alpha, beta = emp.alpha_beta(returns, benchmark_returns)
```

---

## 4. Backtesting Libraries

### FFN (INSTALLED) - Simple Backtesting

**What's Available**:
```python
import ffn

# Performance stats
stats = ffn.core.PerformanceStats(returns)
print(stats.display())

# Metrics
stats.total_return
stats.cagr              # Compound annual growth rate
stats.volatility
stats.sharpe
stats.sortino
stats.max_drawdown
stats.calmar
stats.monthly_returns
stats.yearly_returns
```

### BT - Allocation Backtesting (Not Installed, but Simple)

**Install**: `pip install bt`

**Features**:
- Simple allocation backtests
- Rebalancing strategies
- Tree-based strategy structure
- Clean, minimal code
- **Perfect for testing portfolio optimizations**

**Code Example**:
```python
import bt

# Equal weight rebalancing
strategy = bt.Strategy(
    'EqualWeight',
    [
        bt.algos.RunMonthly(),      # Rebalance monthly
        bt.algos.SelectAll(),        # All tickers
        bt.algos.WeighEqually(),     # Equal weight
        bt.algos.Rebalance()         # Execute
    ]
)

# Custom weights
weights = {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3}
strategy = bt.Strategy(
    'CustomWeight',
    [
        bt.algos.RunQuarterly(),
        bt.algos.SelectAll(),
        bt.algos.WeighSpecified(**weights),
        bt.algos.Rebalance()
    ]
)

# Momentum (top N performers)
strategy = bt.Strategy(
    'Momentum',
    [
        bt.algos.RunMonthly(),
        bt.algos.SelectAll(),
        bt.algos.SelectMomentum(n=2, lookback=pd.DateOffset(months=3)),
        bt.algos.WeighEqually(),
        bt.algos.Rebalance()
    ]
)

# Backtest
test = bt.Backtest(strategy, data)
results = bt.run(test)
results.plot()
print(results.stats)
```

**Integration with PyPortfolioOpt**:
```python
def calculate_pypfopt_weights(target):
    selected = target.temp['selected']
    data = target.universe[selected]

    from pypfopt import expected_returns, risk_models, EfficientFrontier
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe()
    weights = ef.clean_weights()

    target.temp['weights'] = weights
    return True

strategy = bt.Strategy(
    'MeanVariance',
    [
        bt.algos.RunQuarterly(),
        bt.algos.SelectAll(),
        calculate_pypfopt_weights,
        bt.algos.Rebalance()
    ]
)
```

### Vectorbt (Fast - Not Installed)

**Install**: `pip install vectorbt`

**Features**:
- 100-1000x faster than event-driven
- Vectorized operations
- Multi-asset, multi-strategy
- Interactive Plotly charts
- Great for parameter optimization

**When to Use**:
- Testing many strategies
- Walk-forward optimization
- Large universes (100+ stocks)

**Code Example**:
```python
import vectorbt as vbt

data = vbt.YFData.download(['AAPL', 'GOOGL'], start='2020-01-01')

# Moving average crossover
fast_ma = vbt.MA.run(data.close, 10)
slow_ma = vbt.MA.run(data.close, 50)
entries = fast_ma.ma_above(slow_ma, crossed=True)
exits = fast_ma.ma_below(slow_ma, crossed=True)

# Backtest
pf = vbt.Portfolio.from_signals(
    data.close,
    entries,
    exits,
    init_cash=10000,
    fees=0.001
)

print(pf.total_return())
print(pf.sharpe_ratio())
print(pf.max_drawdown())
pf.plot().show()
```

### Backtrader (Event-Driven - Not Installed)

**Install**: `pip install backtrader`

**Features**:
- Realistic trade simulation
- Complex order types
- Live trading integration
- Large community
- Slower for large backtests

**When to Use**:
- Complex trading logic
- Realistic order execution
- Transitioning to live trading

---

## 5. Technical Analysis Libraries

### pandas-ta (Best Choice - Not Installed)

**Install**: `pip install pandas-ta`

**Features**:
- 130+ indicators
- Pandas integration
- Vectorized (fast)
- Clean API
- Custom strategies

**Indicators Available**:
- **Momentum**: RSI, MACD, Stochastic, CCI, ROC, Williams %R, MOM
- **Trend**: SMA, EMA, WMA, VWAP, SuperTrend, ADX, Aroon, PSAR
- **Volatility**: Bollinger Bands, ATR, Keltner, Donchian, UI
- **Volume**: OBV, CMF, MFI, VWAP, PVT, AD
- **Candlestick**: 65+ patterns (Doji, Hammer, Engulfing, etc.)

**Code Example**:
```python
import pandas_ta as ta

# Single indicators
df['RSI'] = ta.rsi(df['close'], length=14)
df['MACD'] = ta.macd(df['close'])['MACD_12_26_9']
bbands = ta.bbands(df['close'])
df['BB_LOWER'] = bbands['BBL_5_2.0']
df['BB_UPPER'] = bbands['BBU_5_2.0']

# Add to DataFrame directly
df.ta.rsi(length=14, append=True)
df.ta.macd(append=True)
df.ta.bbands(append=True)
df.ta.atr(append=True)

# Strategy (multiple indicators at once)
MyStrategy = ta.Strategy(
    name="Momo and Vol",
    ta=[
        {"kind": "rsi", "length": 14},
        {"kind": "macd", "fast": 12, "slow": 26},
        {"kind": "bbands", "length": 20},
        {"kind": "atr", "length": 14},
        {"kind": "sma", "length": 50},
        {"kind": "ema", "length": 20},
    ]
)
df.ta.strategy(MyStrategy)

# All indicators
df.ta.indicators(as_list=True)  # List all available

# Categories
df.ta.candles(append=True)      # All candlestick patterns
df.ta.momentum(append=True)     # All momentum indicators
df.ta.volatility(append=True)   # All volatility indicators
```

### TA-Lib (Fastest - Not Installed)

**Install**: `pip install TA-Lib` (requires C libraries)
**Easier**: `conda install -c conda-forge ta-lib`

**Features**:
- 150+ indicators
- Extremely fast (C implementation)
- Industry standard
- Installation can be tricky

**Code Example**:
```python
import talib as ta

# Indicators
rsi = ta.RSI(close, timeperiod=14)
macd, signal, hist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
upper, middle, lower = ta.BBANDS(close, timeperiod=20)

# Pattern recognition
doji = ta.CDLDOJI(open, high, low, close)
hammer = ta.CDLHAMMER(open, high, low, close)
engulfing = ta.CDLENGULFING(open, high, low, close)
```

### TradingView-TA (INSTALLED)

**Usage**:
```python
from tradingview_ta import TA_Handler, Interval

handler = TA_Handler(
    symbol="AAPL",
    screener="america",
    exchange="NASDAQ",
    interval=Interval.INTERVAL_1_DAY
)

analysis = handler.get_analysis()
print(analysis.summary)      # {'RECOMMENDATION': 'BUY', 'BUY': 10, 'SELL': 5, 'NEUTRAL': 12}
print(analysis.indicators)   # All indicator values
```

---

## 6. Fundamental Data Libraries

### yfinance (INSTALLED) - Sufficient

See section 2 for complete coverage.

### financetoolkit (Advanced - Not Installed)

**Install**: `pip install financetoolkit`

**Features**:
- 100+ financial metrics
- Financial modeling
- DuPont analysis
- Altman Z-Score
- Better than yfinance for deep analysis

**Code Example**:
```python
from financetoolkit import Toolkit

companies = Toolkit(["AAPL", "GOOGL"], api_key="FMP_KEY")

# Ratios
companies.ratios.get_price_earnings_ratio()
companies.ratios.get_return_on_equity()
companies.ratios.get_debt_to_equity()

# Models
companies.models.get_dupont_analysis()
companies.models.get_altman_z_score()
```

### fundamentalanalysis (Not Installed)

**Install**: `pip install fundamentalanalysis`

**Requires**: API key from financialmodelingprep.com (free tier available)

---

## RECOMMENDATIONS FOR THIS PROJECT

### Keep (Already Installed)
1. **PyPortfolioOpt** - Excellent portfolio optimization
2. **yfinance** - Good fundamental data coverage
3. **ffn** - Basic performance metrics
4. **tradingview-ta** - Technical analysis aggregate

### Add (High Priority)
1. **pandas-ta** - Best technical analysis library
   ```bash
   pip install pandas-ta
   ```

2. **bt** - Perfect for backtesting portfolio allocations
   ```bash
   pip install bt
   ```

3. **empyrical** - Better performance metrics
   ```bash
   pip install empyrical
   ```

### Add (Optional)
1. **vectorbt** - If you need speed for parameter optimization
   ```bash
   pip install vectorbt
   ```

2. **riskfolio-lib** - If you need advanced portfolio features
   ```bash
   pip install riskfolio-lib
   ```

3. **financetoolkit** - If you need advanced fundamental analysis
   ```bash
   pip install financetoolkit
   ```

### Quick Install Command
```bash
# Essential additions
pip install pandas-ta bt empyrical

# Optional (advanced features)
pip install vectorbt riskfolio-lib financetoolkit
```

---

## Summary Table

| Library | Category | Installed | Priority | Use Case |
|---------|----------|-----------|----------|----------|
| PyPortfolioOpt | Portfolio Opt | ✅ | Keep | Mean-variance optimization |
| yfinance | Data | ✅ | Keep | Price & fundamental data |
| ffn | Backtesting | ✅ | Keep | Simple performance stats |
| tradingview-ta | Technical | ✅ | Keep | Technical analysis aggregate |
| pandas-ta | Technical | ❌ | HIGH | Best TA library |
| bt | Backtesting | ❌ | HIGH | Portfolio allocation testing |
| empyrical | Metrics | ❌ | MEDIUM | Better performance metrics |
| vectorbt | Backtesting | ❌ | MEDIUM | Fast vectorized backtesting |
| riskfolio-lib | Portfolio Opt | ❌ | LOW | Advanced optimization |
| financetoolkit | Fundamentals | ❌ | LOW | Advanced ratios |
