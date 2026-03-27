# Stock Screener Enhancement Plan

## Goals
1. ✅ Better buy portfolios (multi-factor analysis)
2. ✅ Reduce beta/volatility (add min-vol, HRP optimization)
3. ✅ Sell recommendations (stop-loss, fundamental deterioration)
4. ✅ More robust analysis (technical + fundamental combo)

---

## Phase 1: Risk Management & Sell Signals (Priority 1) 🔴

**Why First**: Protect capital, most impactful

### 1.1 Technical Exits
```python
# core/sell_analyzer.py (NEW FILE)
- Stop-loss: 8% fixed
- Trailing stop: ATR-based (2x ATR from 20-day high)
- Technical breakdown: MA death cross + RSI<30 + MACD crossover
- Library: pandas-ta for indicators
```

### 1.2 Fundamental Exits
```python
# Add to core/analyzer.py
- ROE decline >20%
- Negative earnings growth
- P/E > 30 or PEG > 2
- Debt/Equity increase >50%
```

### 1.3 Portfolio Exits
```python
# core/portfolio_manager.py (NEW FILE)
- Position limit: max 15% per stock
- Concentration: top 3 < 50%
- Profit-taking: trim 50% if gain >100%
```

**Implementation**: 2-3 days
**Libraries**: pandas-ta, yfinance (existing)

---

## Phase 2: Enhanced Technical Analysis (Priority 2) 🟡

### 2.1 Better Indicators
```python
# Upgrade core/analyzer.py
Add pandas-ta indicators:
- ADX (trend strength) - require >25
- Stochastic RSI (momentum) - 40-60 range
- OBV (volume confirmation) - rising trend
- MFI (money flow) - >50 bullish
```

### 2.2 Multi-Timeframe Confirmation
```python
def multi_timeframe_buy(ticker):
    daily: 50 > 200 MA (trend)
    weekly: 20 > 50 MA (strength)
    4hr: RSI < 60 (entry timing)
    all_aligned = daily AND weekly AND 4hr
```

**Implementation**: 2-3 days
**New Dependency**: pandas-ta==0.3.14b

---

## Phase 3: Fundamental Screening (Priority 3) 🟡

### 3.1 Quality Filters
```python
# core/fundamental.py (NEW FILE)
Metrics from yfinance:
- ROE > 15% (profitability)
- Debt/Equity < 2 (leverage)
- Current Ratio > 1.5 (liquidity)
- Earnings Growth > 15%
- Revenue Growth > 10%
- PEG < 1.5 (value)

Score: require 4/6 criteria
```

### 3.2 Combined Filter
```python
def robust_screening(ticker):
    tech_score = technical_analysis(ticker)  # 0-5
    fund_score = fundamental_analysis(ticker)  # 0-6
    return (tech_score >= 3) AND (fund_score >= 4)
```

**Implementation**: 3-4 days
**Data Source**: yfinance (free, existing)

---

## Phase 4: Advanced Optimization (Priority 4) 🟢

### 4.1 Add HRP (Hierarchical Risk Parity)
```python
# core/optimizer.py - add method
from pypfopt.hierarchical_portfolio import HRPOpt

def optimize(..., method='max_sharpe'):
    if method == 'hrp':
        returns = expected_returns.returns_from_prices(prices)
        hrp = HRPOpt(returns)
        weights = hrp.optimize()
    elif method == 'min_vol':
        ef = EfficientFrontier(mu, cov)
        weights = ef.min_volatility()
```

### 4.2 Factor Tilting
```python
# core/factors.py (NEW FILE)
Calculate factor scores:
- Value: 1/P/E
- Growth: earnings growth
- Momentum: 6-month return
- Quality: ROE

Tilt weights by factor exposure
```

**Implementation**: 3-4 days
**Libraries**: pypfopt (existing)

---

## Phase 5: Backtesting & Validation (Priority 5) 🟢

### 5.1 Simple Backtest
```python
# core/backtest.py (NEW FILE)
- Walk-forward testing (rolling 1-year train, 3-month test)
- Include transaction costs (0.1%)
- Track: Sharpe, max drawdown, win rate
- Library: vectorbt or simple pandas
```

### 5.2 Performance Tracking
```python
# Store in SQLite
- Entry price, date
- Current P&L
- Sell trigger status
- Performance metrics
```

**Implementation**: 4-5 days
**New Dependency**: vectorbt==0.26.1 (optional)

---

## Implementation Roadmap

### Sprint 1 (Week 1): Risk Management
- [ ] Create sell_analyzer.py with stop-loss logic
- [ ] Add fundamental exit checks
- [ ] Add portfolio position limits
- [ ] New API endpoint: `/check_sells/`
- [ ] UI: Sell recommendations panel

### Sprint 2 (Week 2): Better Analysis
- [ ] Add pandas-ta dependency
- [ ] Implement ADX, Stochastic RSI, OBV, MFI
- [ ] Multi-timeframe confirmation
- [ ] Add fundamental.py with quality filters
- [ ] Combined technical + fundamental scoring

### Sprint 3 (Week 3): Advanced Portfolio
- [ ] Add HRP optimization method
- [ ] Add min-volatility method
- [ ] Implement factor scoring
- [ ] Factor-tilted portfolio construction
- [ ] Rebalancing trigger logic

### Sprint 4 (Week 4): Validation
- [ ] Simple backtest framework
- [ ] Walk-forward testing
- [ ] Performance tracking database
- [ ] Metrics dashboard
- [ ] Documentation & examples

---

## New Dependencies

```toml
# Add to pyproject.toml
dependencies = [
    # ... existing ...
    "pandas-ta==0.3.14b",  # Technical indicators
    "vectorbt==0.26.1",    # Backtesting (optional)
]
```

---

## File Structure (After Implementation)

```
core/
├── analyzer.py          # Enhanced with pandas-ta indicators
├── fundamental.py       # NEW: Fundamental screening
├── sell_analyzer.py     # NEW: Sell signal logic
├── optimizer.py         # Add HRP, min-vol methods
├── factors.py           # NEW: Factor scoring
├── portfolio_manager.py # NEW: Position/risk management
├── backtest.py          # NEW: Backtesting framework
├── watchlist.py         # Existing
└── utils.py             # Existing

app.py                   # New endpoints: /check_sells/, /rebalance/
```

---

## API Changes

### New Endpoints

```python
# Check sell signals for current holdings
POST /check_sells/
{
  "holdings": [
    {"symbol": "AAPL", "entry_price": 150, "shares": 10},
    {"symbol": "MSFT", "entry_price": 300, "shares": 5}
  ]
}
→ Returns sell recommendations with reasons

# Portfolio rebalancing check
POST /check_rebalance/
{
  "current_weights": {"AAPL": 0.35, "MSFT": 0.25, ...},
  "target_weights": {"AAPL": 0.30, "MSFT": 0.25, ...}
}
→ Returns rebalancing actions if drift >5%
```

### Enhanced Endpoint

```python
# Add parameters to /recommend_stocks/
POST /recommend_stocks/
{
  "budget": 10000,
  "method": "hrp",  # NEW: hrp, min_vol, max_sharpe, semivariance
  "threshold": 0.05,
  "include_fundamentals": true,  # NEW: Apply fundamental filters
  "factor_tilt": "momentum"  # NEW: momentum, value, growth, quality
}
```

---

## Testing Checklist

- [ ] Unit tests for sell signals
- [ ] Unit tests for fundamental scoring
- [ ] Integration test: full pipeline with new filters
- [ ] Backtest: compare old vs new approach
- [ ] Performance: ensure <5 min for 150 stocks

---

## Success Metrics

**Before (Current)**:
- 4/150 stocks pass filter (2.7%)
- Only technical analysis
- No sell signals
- Max Sharpe only

**After (Target)**:
- 10-20/150 stocks pass (7-13%) - better quality filter
- Technical + Fundamental scoring
- Automated sell recommendations
- 4 optimization methods (max_sharpe, hrp, min_vol, semivariance)
- Backtested performance metrics
- Position/concentration risk management

**Key Improvement**: Reduce false positives, add sell discipline, better risk management
