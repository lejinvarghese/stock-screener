# Tactical Implementation Plan - Stock Screener Enhancement

## Research Summary

**Current Stack Analysis:**
- ✅ PyPortfolioOpt v1.5.6 - Already using best practices (Ledoit-Wolf, EMA returns, max_sharpe)
- ✅ yfinance v0.2.65 - Sufficient for fundamentals (P/E, ROE, growth, debt ratios)
- ✅ tradingview-ta v3.3.0 - Good for aggregated signals
- ❌ Missing: pandas-ta (technical indicators)
- ❌ Missing: bt (backtesting)
- ❌ Missing: empyrical (better metrics)

**Recommended Additions:**
```bash
pip install pandas-ta==0.3.14b bt empyrical
```

---

## Phase 1: Sell Signals (2-3 days) 🔴

### 1.1 Technical Sell Signals - `core/sell_analyzer.py`

**Libraries**: pandas-ta (new), pandas (existing)

```python
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from typing import Dict, List, Tuple

class SellAnalyzer:
    """Technical & fundamental sell signal detection"""

    def __init__(self, stop_loss_pct=0.08, trailing_atr_mult=2):
        self.stop_loss_pct = stop_loss_pct
        self.trailing_atr_mult = trailing_atr_mult

    def analyze_position(self, symbol: str, entry_price: float,
                        entry_date: str) -> Dict:
        """
        Check all sell signals for a position

        Returns:
            {
                'symbol': 'AAPL',
                'sell_signal': True/False,
                'reasons': ['stop_loss', 'tech_breakdown'],
                'current_price': 150.0,
                'gain_loss': -0.08,
                'recommendation': 'SELL' | 'HOLD'
            }
        """
        # Download recent data
        df = yf.download(symbol, period='6mo', progress=False)

        # Current price
        current_price = df['Close'].iloc[-1]
        gain_loss = (current_price - entry_price) / entry_price

        # Check all signals
        signals = []

        # 1. Stop Loss (fixed 8%)
        if current_price < entry_price * (1 - self.stop_loss_pct):
            signals.append('stop_loss')

        # 2. Trailing Stop (ATR-based)
        if self._check_trailing_stop(df):
            signals.append('trailing_stop')

        # 3. Technical Breakdown
        if self._check_technical_breakdown(df):
            signals.append('technical_breakdown')

        # 4. Profit Target (optional - 100% gain)
        if gain_loss > 1.0:
            signals.append('profit_target_trim')  # Consider trimming

        return {
            'symbol': symbol,
            'sell_signal': len(signals) > 0,
            'reasons': signals,
            'current_price': float(current_price),
            'entry_price': entry_price,
            'gain_loss': float(gain_loss),
            'recommendation': 'SELL' if any(s in signals for s in
                ['stop_loss', 'trailing_stop', 'technical_breakdown']) else 'HOLD'
        }

    def _check_trailing_stop(self, df: pd.DataFrame) -> bool:
        """ATR-based trailing stop"""
        # Calculate ATR(14)
        df.ta.atr(length=14, append=True)

        # 20-day high
        high_20 = df['Close'].rolling(20).max()

        # Trailing stop level = 20-day high - (2 * ATR)
        trailing_level = high_20 - (self.trailing_atr_mult * df['ATR_14'])

        # Triggered if current < trailing level
        return df['Close'].iloc[-1] < trailing_level.iloc[-1]

    def _check_technical_breakdown(self, df: pd.DataFrame) -> bool:
        """MA death cross + RSI oversold breakdown + MACD bearish"""
        # Add indicators using pandas-ta
        df.ta.sma(length=50, append=True)
        df.ta.sma(length=200, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)

        # Death cross: 50 MA < 200 MA
        death_cross = df['SMA_50'].iloc[-1] < df['SMA_200'].iloc[-1]

        # RSI breakdown: crossed below 30 recently (not bouncing)
        rsi_break = df['RSI_14'].iloc[-1] < 30 and df['RSI_14'].iloc[-2] >= 30

        # MACD bearish: MACD < Signal
        macd_bear = df['MACD_12_26_9'].iloc[-1] < df['MACDs_12_26_9'].iloc[-1]

        # All three must align for breakdown
        return death_cross and macd_bear and rsi_break

# API endpoint in app.py
@app.route("/check_sells/", methods=["POST"])
def check_sell_signals():
    """
    POST /check_sells/
    {
        "holdings": [
            {"symbol": "AAPL", "entry_price": 150, "entry_date": "2024-01-01", "shares": 10},
            {"symbol": "MSFT", "entry_price": 300, "entry_date": "2024-02-01", "shares": 5}
        ]
    }
    """
    data = request.get_json()
    holdings = data.get('holdings', [])

    analyzer = SellAnalyzer()
    results = []

    for holding in holdings:
        result = analyzer.analyze_position(
            holding['symbol'],
            holding['entry_price'],
            holding.get('entry_date', '')
        )
        result['shares'] = holding.get('shares', 0)
        results.append(result)

    return jsonify({'sell_recommendations': results})
```

**Test:**
```bash
curl -X POST http://localhost:5004/check_sells/ \
  -H "Content-Type: application/json" \
  -d '{
    "holdings": [
      {"symbol": "AAPL", "entry_price": 250, "shares": 10},
      {"symbol": "TSLA", "entry_price": 400, "shares": 5}
    ]
  }'
```

### 1.2 Fundamental Sell Signals - Add to `SellAnalyzer`

```python
def check_fundamental_deterioration(self, symbol: str) -> Dict:
    """Check if fundamentals have deteriorated"""
    stock = yf.Ticker(symbol)
    info = stock.info

    signals = []

    # 1. Negative earnings growth
    earnings_growth = info.get('earningsGrowth', 0)
    if earnings_growth < -0.10:  # -10% decline
        signals.append('earnings_decline')

    # 2. ROE decline (compare to industry avg or historical)
    roe = info.get('returnOnEquity', 0)
    if roe < 0.05:  # Less than 5%
        signals.append('low_roe')

    # 3. High debt
    debt_to_equity = info.get('debtToEquity', 0)
    if debt_to_equity > 200:  # >2.0 ratio
        signals.append('high_debt')

    # 4. Overvaluation
    pe = info.get('forwardPE', 0)
    peg = info.get('pegRatio', 999)
    if pe > 30 or (peg > 2 and peg < 999):
        signals.append('overvalued')

    return {
        'fundamental_issues': signals,
        'recommend_review': len(signals) >= 2,
        'metrics': {
            'earnings_growth': earnings_growth,
            'roe': roe,
            'debt_to_equity': debt_to_equity,
            'pe': pe,
            'peg': peg
        }
    }
```

### 1.3 Portfolio-Level Sell Rules - `core/portfolio_manager.py`

```python
class PortfolioManager:
    """Portfolio-level position and risk management"""

    def __init__(self, max_position=0.15, max_top3=0.50):
        self.max_position = max_position
        self.max_top3 = max_top3

    def check_rebalancing_needed(self, current_portfolio: Dict[str, float],
                                 target_portfolio: Dict[str, float],
                                 drift_threshold: float = 0.05) -> Dict:
        """
        Check if portfolio needs rebalancing

        Args:
            current_portfolio: {'AAPL': 0.35, 'MSFT': 0.25, ...}
            target_portfolio: {'AAPL': 0.30, 'MSFT': 0.25, ...}
            drift_threshold: 0.05 = 5% drift triggers rebalance

        Returns:
            {
                'needs_rebalance': True/False,
                'actions': [
                    {'symbol': 'AAPL', 'action': 'SELL', 'amount': 0.05},
                    {'symbol': 'GOOGL', 'action': 'BUY', 'amount': 0.03}
                ],
                'reason': 'drift' | 'concentration' | 'position_limit'
            }
        """
        actions = []

        # Check position limits
        for symbol, weight in current_portfolio.items():
            if weight > self.max_position:
                actions.append({
                    'symbol': symbol,
                    'action': 'TRIM',
                    'current': weight,
                    'target': self.max_position,
                    'amount': weight - self.max_position,
                    'reason': 'position_limit'
                })

        # Check concentration (top 3 positions)
        sorted_positions = sorted(current_portfolio.items(),
                                 key=lambda x: x[1], reverse=True)
        top3_weight = sum([w for _, w in sorted_positions[:3]])

        if top3_weight > self.max_top3:
            actions.append({
                'action': 'REBALANCE',
                'reason': f'concentration: top 3 = {top3_weight:.1%} > {self.max_top3:.1%}',
                'suggestion': 'Reduce top 3 positions'
            })

        # Check drift from target
        for symbol in target_portfolio:
            current = current_portfolio.get(symbol, 0)
            target = target_portfolio[symbol]
            drift = abs(current - target)

            if drift > drift_threshold:
                action = 'BUY' if current < target else 'SELL'
                actions.append({
                    'symbol': symbol,
                    'action': action,
                    'current': current,
                    'target': target,
                    'amount': abs(current - target),
                    'reason': 'drift'
                })

        return {
            'needs_rebalance': len(actions) > 0,
            'actions': actions
        }
```

**Checkpoint 1 Commit:**
```bash
git add core/sell_analyzer.py core/portfolio_manager.py
git commit -m "Phase 1: Add sell signals (stop-loss, ATR trailing, technical breakdown, fundamental checks)"
```

---

## Phase 2: Enhanced Technical Analysis (2-3 days) 🟡

### 2.1 Upgrade Analyzer with pandas-ta - Modify `core/analyzer.py`

**Add multi-factor confirmation:**

```python
import pandas_ta as ta

def enhanced_technical_analysis(ticker: str) -> Dict:
    """
    Multi-factor technical scoring system

    Returns score 0-5:
    - Trend (ADX > 25): +1
    - Momentum (Stochastic RSI 40-60): +1
    - Volume (OBV rising): +1
    - MACD (bullish crossover): +1
    - Support (Price > VWAP): +1
    """
    df = yf.download(ticker, period='6mo', progress=False)

    # Add all indicators at once (pandas-ta magic!)
    df.ta.adx(length=14, append=True)
    df.ta.stochrsi(length=14, append=True)
    df.ta.obv(append=True)
    df.ta.macd(append=True)
    df.ta.vwap(append=True)

    score = 0
    signals = []

    # 1. Trend Strength: ADX > 25
    if df['ADX_14'].iloc[-1] > 25:
        score += 1
        signals.append('strong_trend')

    # 2. Momentum: Stochastic RSI in 40-60 range (not overbought)
    stoch_rsi = df['STOCHRSIk_14_14_3_3'].iloc[-1]
    if 40 < stoch_rsi < 60:
        score += 1
        signals.append('momentum_good')

    # 3. Volume: OBV rising (20-day trend)
    obv_trend = df['OBV'].iloc[-1] > df['OBV'].iloc[-20]
    if obv_trend:
        score += 1
        signals.append('volume_confirmation')

    # 4. MACD: Bullish crossover
    macd_bullish = df['MACD_12_26_9'].iloc[-1] > df['MACDs_12_26_9'].iloc[-1]
    if macd_bullish:
        score += 1
        signals.append('macd_bullish')

    # 5. Price > VWAP (above average)
    if df['Close'].iloc[-1] > df['VWAP'].iloc[-1]:
        score += 1
        signals.append('above_vwap')

    return {
        'ticker': ticker,
        'technical_score': score,
        'max_score': 5,
        'signals': signals,
        'recommendation': 'BUY' if score >= 4 else ('HOLD' if score >= 3 else 'PASS'),
        'indicators': {
            'adx': float(df['ADX_14'].iloc[-1]),
            'stoch_rsi': float(stoch_rsi),
            'obv_trend': obv_trend,
            'macd': float(df['MACD_12_26_9'].iloc[-1]),
            'price': float(df['Close'].iloc[-1]),
            'vwap': float(df['VWAP'].iloc[-1])
        }
    }

# Replace TradingView analysis with this
def get_buy_candidates(tickers: List[str], min_score: int = 4) -> List[str]:
    """Filter tickers by technical score"""
    candidates = []

    with ThreadPool(N_PROCESS) as pool:
        results = pool.map(enhanced_technical_analysis, tickers)

    for result in results:
        if result['technical_score'] >= min_score:
            candidates.append(result['ticker'])
            console.print(f"✓ {result['ticker']}: {result['technical_score']}/5 - {result['signals']}")

    return candidates
```

### 2.2 Multi-Timeframe Confirmation

```python
def multi_timeframe_analysis(ticker: str) -> Dict:
    """
    Confirm signals across timeframes:
    - Daily: Trend direction (50 > 200 MA)
    - Weekly: Strength confirmation (20 > 50 MA)
    - 4-hour: Entry timing (RSI < 60)
    """
    # Daily timeframe (trend)
    daily = yf.download(ticker, period='1y', interval='1d', progress=False)
    daily.ta.sma(length=50, append=True)
    daily.ta.sma(length=200, append=True)
    daily_bullish = daily['SMA_50'].iloc[-1] > daily['SMA_200'].iloc[-1]

    # Weekly timeframe (strength)
    weekly = yf.download(ticker, period='2y', interval='1wk', progress=False)
    weekly.ta.sma(length=20, append=True)
    weekly.ta.sma(length=50, append=True)
    weekly_bullish = weekly['SMA_20'].iloc[-1] > weekly['SMA_50'].iloc[-1]

    # 4-hour timeframe (entry)
    hourly = yf.download(ticker, period='60d', interval='1h', progress=False)
    hourly.ta.rsi(length=14, append=True)
    entry_ok = hourly['RSI_14'].iloc[-1] < 60  # Not overbought

    aligned = daily_bullish and weekly_bullish and entry_ok

    return {
        'ticker': ticker,
        'timeframe_aligned': aligned,
        'daily_trend': 'bullish' if daily_bullish else 'bearish',
        'weekly_strength': 'strong' if weekly_bullish else 'weak',
        'entry_timing': 'good' if entry_ok else 'overbought',
        'recommendation': 'BUY' if aligned else 'WAIT'
    }
```

**Checkpoint 2 Commit:**
```bash
git add core/analyzer.py
git commit -m "Phase 2: Enhanced technical analysis with pandas-ta (ADX, StochRSI, OBV, VWAP) + multi-timeframe"
```

---

## Phase 3: Fundamental Screening (2 days) 🟡

### 3.1 Fundamental Analyzer - `core/fundamental.py`

```python
import yfinance as yf
from typing import Dict, List

class FundamentalAnalyzer:
    """Fundamental quality screening using yfinance data"""

    def __init__(self):
        # Thresholds
        self.min_roe = 0.15
        self.max_debt_equity = 2.0
        self.min_current_ratio = 1.5
        self.min_earnings_growth = 0.15
        self.min_revenue_growth = 0.10
        self.max_peg = 1.5

    def analyze(self, ticker: str) -> Dict:
        """
        Score fundamental quality (0-6 points)

        Metrics:
        1. ROE > 15% (profitability)
        2. Debt/Equity < 2.0 (leverage)
        3. Current Ratio > 1.5 (liquidity)
        4. Earnings Growth > 15% (growth)
        5. Revenue Growth > 10% (top-line)
        6. PEG < 1.5 (valuation)
        """
        stock = yf.Ticker(ticker)
        info = stock.info

        score = 0
        passed = []
        failed = []

        # 1. ROE
        roe = info.get('returnOnEquity', 0)
        if roe > self.min_roe:
            score += 1
            passed.append('roe')
        else:
            failed.append('roe')

        # 2. Debt/Equity
        debt_equity = info.get('debtToEquity', 999)
        if debt_equity < self.max_debt_equity * 100:  # yfinance returns as %
            score += 1
            passed.append('leverage')
        else:
            failed.append('leverage')

        # 3. Current Ratio
        current_ratio = info.get('currentRatio', 0)
        if current_ratio > self.min_current_ratio:
            score += 1
            passed.append('liquidity')
        else:
            failed.append('liquidity')

        # 4. Earnings Growth
        earnings_growth = info.get('earningsGrowth', 0)
        if earnings_growth > self.min_earnings_growth:
            score += 1
            passed.append('earnings_growth')
        else:
            failed.append('earnings_growth')

        # 5. Revenue Growth
        revenue_growth = info.get('revenueGrowth', 0)
        if revenue_growth > self.min_revenue_growth:
            score += 1
            passed.append('revenue_growth')
        else:
            failed.append('revenue_growth')

        # 6. PEG Ratio
        peg = info.get('pegRatio', 999)
        if 0 < peg < self.max_peg:
            score += 1
            passed.append('valuation')
        else:
            failed.append('valuation')

        return {
            'ticker': ticker,
            'fundamental_score': score,
            'max_score': 6,
            'passed_criteria': passed,
            'failed_criteria': failed,
            'recommendation': 'BUY' if score >= 4 else ('HOLD' if score >= 3 else 'PASS'),
            'metrics': {
                'roe': roe,
                'debt_to_equity': debt_equity / 100,
                'current_ratio': current_ratio,
                'earnings_growth': earnings_growth,
                'revenue_growth': revenue_growth,
                'peg': peg
            }
        }

    def screen_tickers(self, tickers: List[str], min_score: int = 4) -> List[str]:
        """Filter tickers by fundamental score"""
        candidates = []

        for ticker in tickers:
            try:
                result = self.analyze(ticker)
                if result['fundamental_score'] >= min_score:
                    candidates.append(ticker)
                    console.print(f"✓ {ticker}: {result['fundamental_score']}/6 - {result['passed_criteria']}")
            except Exception as e:
                console.print(f"✗ {ticker}: Error - {e}")

        return candidates
```

### 3.2 Combined Screening - Modify `core/analyzer.py`

```python
def combined_screening(tickers: List[str],
                      min_technical_score: int = 3,
                      min_fundamental_score: int = 4) -> List[str]:
    """
    Two-stage filter:
    1. Technical score >= 3 (out of 5)
    2. Fundamental score >= 4 (out of 6)
    """
    from core.fundamental import FundamentalAnalyzer

    console.print("[blue]Stage 1: Technical Analysis[/blue]")
    tech_candidates = []
    for ticker in tickers:
        result = enhanced_technical_analysis(ticker)
        if result['technical_score'] >= min_technical_score:
            tech_candidates.append(ticker)

    console.print(f"[green]Technical pass: {len(tech_candidates)}/{len(tickers)}[/green]")

    console.print("[blue]Stage 2: Fundamental Analysis[/blue]")
    fund_analyzer = FundamentalAnalyzer()
    final_candidates = fund_analyzer.screen_tickers(tech_candidates, min_fundamental_score)

    console.print(f"[green]Final candidates: {len(final_candidates)}/{len(tech_candidates)}[/green]")

    return final_candidates

# Update run() function
def run(send_telegram=True, use_fundamentals=True):
    watchlist = get_custom_watchlist()

    if use_fundamentals:
        selected = combined_screening(watchlist, min_technical_score=3, min_fundamental_score=4)
    else:
        selected = get_buy_candidates(watchlist, min_score=4)

    return selected
```

**Checkpoint 3 Commit:**
```bash
git add core/fundamental.py core/analyzer.py
git commit -m "Phase 3: Add fundamental screening (ROE, debt, growth, valuation) + combined filter"
```

---

## Phase 4: Advanced Optimization (2-3 days) 🟢

### 4.1 Expand Optimizer Methods - Modify `core/optimizer.py`

**PyPortfolioOpt already installed - add new methods:**

```python
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.cla import CLA
from pypfopt.efficient_frontier import EfficientCVaR

def optimize(prices, budget, method='max_sharpe', **kwargs):
    """
    Enhanced optimization with 5 methods:
    - max_sharpe: Maximum Sharpe ratio (current)
    - min_vol: Minimum volatility (conservative)
    - hrp: Hierarchical Risk Parity (diversification)
    - cvar: Conditional Value at Risk (tail risk)
    - semivariance: Downside risk (current)
    """
    mu = expected_returns.ema_historical_return(prices)

    if method == 'max_sharpe':
        # Current implementation
        cov = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        ef = EfficientFrontier(mu, cov)
        ef.add_objective(objective_functions.L2_reg, gamma=0.01)
        weights = ef.max_sharpe()

    elif method == 'min_vol':
        # Minimum volatility (lowest beta)
        cov = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        ef = EfficientFrontier(mu, cov)
        ef.add_objective(objective_functions.L2_reg, gamma=0.01)
        weights = ef.min_volatility()

    elif method == 'hrp':
        # Hierarchical Risk Parity (best diversification)
        returns = expected_returns.returns_from_prices(prices)
        hrp = HRPOpt(returns)
        weights = hrp.optimize()

    elif method == 'cvar':
        # Conditional Value at Risk (95% confidence)
        returns = expected_returns.returns_from_prices(prices)
        ef_cvar = EfficientCVaR(mu, returns, beta=0.95)
        ef_cvar.add_objective(objective_functions.L2_reg, gamma=0.01)
        weights = ef_cvar.min_cvar()

    elif method == 'semivariance':
        # Current semivariance implementation
        returns = expected_returns.returns_from_prices(prices)
        ef_semi = EfficientSemivariance(mu, returns)
        ef_semi.add_objective(objective_functions.L2_reg, gamma=0.01)
        weights = ef_semi.efficient_return(target_return=0.20)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Clean weights and get discrete allocation
    cleaned = ef.clean_weights() if method != 'hrp' else hrp.clean_weights()

    # Performance metrics
    if method == 'hrp':
        perf = hrp.portfolio_performance(verbose=False)
    else:
        perf = ef.portfolio_performance(verbose=False)

    # Discrete allocation
    latest_prices = get_latest_prices(prices)
    da = discrete_allocation.DiscreteAllocation(
        cleaned, latest_prices, total_portfolio_value=budget
    )
    allocation, leftover = da.lp_portfolio()

    # Plotting
    plot_efficient_frontier(mu, cov, cleaned, method)
    plot_covariance_matrix(cov, prices.columns)

    return {
        'weights': cleaned,
        'allocation': allocation,
        'performance': list(perf),  # [return, volatility, sharpe]
        'leftover': leftover,
        'method': method
    }

# Add to app.py endpoint
@app.route("/recommend_stocks/", methods=["POST"])
def recommend_stocks():
    data = request.get_json() or {}
    method = data.get("method", "max_sharpe")  # NEW: Support all 5 methods
    use_fundamentals = data.get("include_fundamentals", True)  # NEW

    # Run analysis with fundamental filter
    pre_selected_stocks = analyze(send_telegram=False, use_fundamentals=use_fundamentals)

    # Optimize with selected method
    optimized = optimize(pre_selected_stocks, budget=budget, method=method)

    return jsonify(optimized)
```

### 4.2 Factor Tilting (Optional Advanced) - `core/factors.py`

```python
def calculate_factor_scores(tickers: List[str]) -> Dict[str, Dict]:
    """
    Calculate factor exposures:
    - Value: 1 / P/E (higher = more value)
    - Growth: Earnings Growth
    - Momentum: 6-month return
    - Quality: ROE
    """
    scores = {}

    for ticker in tickers:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period='6mo')

        # Value factor
        pe = info.get('forwardPE', 999)
        value_score = 1 / pe if pe > 0 and pe < 999 else 0

        # Growth factor
        growth_score = info.get('earningsGrowth', 0)

        # Momentum factor
        if len(hist) > 0:
            momentum_score = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
        else:
            momentum_score = 0

        # Quality factor
        quality_score = info.get('returnOnEquity', 0)

        scores[ticker] = {
            'value': value_score,
            'growth': growth_score,
            'momentum': momentum_score,
            'quality': quality_score
        }

    return scores

def tilt_portfolio(base_weights: Dict, factor_scores: Dict,
                  tilt_factor: str = 'momentum', tilt_strength: float = 0.5) -> Dict:
    """
    Tilt portfolio toward a factor

    Args:
        base_weights: {'AAPL': 0.30, 'MSFT': 0.25, ...}
        factor_scores: From calculate_factor_scores()
        tilt_factor: 'value', 'growth', 'momentum', or 'quality'
        tilt_strength: 0.0-1.0 (0 = no tilt, 1 = full tilt)
    """
    # Extract factor scores
    scores = {ticker: factor_scores[ticker][tilt_factor] for ticker in base_weights}

    # Normalize scores to 0-1
    min_score = min(scores.values())
    max_score = max(scores.values())
    norm_scores = {t: (s - min_score) / (max_score - min_score)
                   for t, s in scores.items()}

    # Tilt weights
    tilted = {}
    for ticker, weight in base_weights.items():
        # weight * (1 + tilt_strength * normalized_score)
        tilted[ticker] = weight * (1 + tilt_strength * norm_scores[ticker])

    # Renormalize to sum to 1
    total = sum(tilted.values())
    tilted = {t: w / total for t, w in tilted.items()}

    return tilted
```

**Checkpoint 4 Commit:**
```bash
git add core/optimizer.py core/factors.py
git commit -m "Phase 4: Add HRP, min-vol, CVaR optimization + factor tilting"
```

---

## Phase 5: Backtesting (3-4 days) 🟢

### 5.1 Simple Backtesting with bt Library

**Install:** `pip install bt`

```python
# core/backtest.py
import bt
import pandas as pd
import yfinance as yf
from typing import Dict, List

class PortfolioBacktester:
    """Backtest portfolio strategies using bt library"""

    def backtest_allocation(self, tickers: List[str], weights: Dict[str, float],
                           start_date: str = '2020-01-01',
                           end_date: str = '2024-01-01') -> Dict:
        """
        Backtest a fixed-weight allocation

        Args:
            tickers: ['AAPL', 'MSFT', 'GOOGL']
            weights: {'AAPL': 0.40, 'MSFT': 0.35, 'GOOGL': 0.25}
            start_date: '2020-01-01'
            end_date: '2024-01-01'

        Returns:
            Performance metrics, equity curve, drawdowns
        """
        # Download price data
        prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

        # Create strategy (rebalance monthly)
        strategy = bt.Strategy('portfolio',
                              [bt.algos.RunMonthly(),
                               bt.algos.SelectAll(),
                               bt.algos.WeighSpecified(**weights),
                               bt.algos.Rebalance()])

        # Create backtest
        test = bt.Backtest(strategy, prices)
        result = bt.run(test)

        # Extract metrics
        stats = result.stats

        return {
            'total_return': stats.loc['total_return'][0],
            'cagr': stats.loc['cagr'][0],
            'max_drawdown': stats.loc['max_drawdown'][0],
            'sharpe': stats.loc['daily_sharpe'][0],
            'sortino': stats.loc['daily_sortino'][0],
            'calmar': stats.loc['calmar'][0],
            'equity_curve': result.prices.to_dict(),
            'monthly_returns': stats.loc['monthly_mean'][0]
        }

    def walk_forward_optimization(self, tickers: List[str],
                                 train_months: int = 12,
                                 test_months: int = 3,
                                 start_date: str = '2020-01-01') -> List[Dict]:
        """
        Walk-forward optimization

        Process:
        1. Train on 12 months, optimize weights
        2. Test on next 3 months (out-of-sample)
        3. Roll forward, repeat
        """
        from core.optimizer import optimize

        # Download all data
        prices = yf.download(tickers, start=start_date)['Adj Close']

        results = []
        start_idx = 0

        while start_idx + train_months + test_months < len(prices):
            # Training period
            train_end_idx = start_idx + train_months
            train_data = prices.iloc[start_idx:train_end_idx]

            # Optimize on training data
            optimized = optimize(train_data, budget=10000, method='max_sharpe')
            weights = optimized['weights']

            # Test period
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + test_months
            test_data = prices.iloc[test_start_idx:test_end_idx]

            # Calculate test performance
            # (portfolio value with optimized weights)
            returns = test_data.pct_change().dropna()
            portfolio_returns = sum(returns[ticker] * weight
                                   for ticker, weight in weights.items()
                                   if ticker in returns.columns)

            test_sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
            test_return = (1 + portfolio_returns).prod() - 1

            results.append({
                'train_period': f"{train_data.index[0]} to {train_data.index[-1]}",
                'test_period': f"{test_data.index[0]} to {test_data.index[-1]}",
                'weights': weights,
                'test_return': float(test_return),
                'test_sharpe': float(test_sharpe)
            })

            # Roll forward
            start_idx += test_months

        return results

# API endpoint
@app.route("/backtest/", methods=["POST"])
def run_backtest():
    """
    POST /backtest/
    {
        "tickers": ["AAPL", "MSFT", "GOOGL"],
        "weights": {"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25},
        "start_date": "2020-01-01",
        "end_date": "2024-01-01"
    }
    """
    data = request.get_json()

    backtester = PortfolioBacktester()
    results = backtester.backtest_allocation(
        data['tickers'],
        data['weights'],
        data.get('start_date', '2020-01-01'),
        data.get('end_date', '2024-01-01')
    )

    return jsonify(results)
```

**Checkpoint 5 Commit:**
```bash
git add core/backtest.py requirements.txt
git commit -m "Phase 5: Add backtesting with bt library + walk-forward optimization"
```

---

## Final Dependencies Update

```bash
# Install new packages
pip install pandas-ta bt empyrical

# Generate new lock file
pip freeze > requirements-lock.txt

# Update pyproject.toml
```

```toml
dependencies = [
    # ... existing ...
    "pandas-ta==0.3.14b",
    "bt==1.0.2",
    "empyrical==0.5.5",
]
```

**Final Commit:**
```bash
git add requirements-lock.txt pyproject.toml
git commit -m "Update dependencies: pandas-ta, bt, empyrical"
git push
```

---

## Testing Checklist

```bash
# Phase 1: Sell signals
curl -X POST http://localhost:5004/check_sells/ \
  -d '{"holdings": [{"symbol": "AAPL", "entry_price": 250}]}'

# Phase 2: Enhanced technical
# (Integrated into /recommend_stocks/)

# Phase 3: Fundamentals
curl -X POST http://localhost:5004/recommend_stocks/ \
  -d '{"include_fundamentals": true, "method": "max_sharpe"}'

# Phase 4: New optimization methods
curl -X POST http://localhost:5004/recommend_stocks/ \
  -d '{"method": "hrp", "budget": 10000}'

curl -X POST http://localhost:5004/recommend_stocks/ \
  -d '{"method": "min_vol", "budget": 10000}'

# Phase 5: Backtesting
curl -X POST http://localhost:5004/backtest/ \
  -d '{"tickers": ["AAPL", "MSFT"], "weights": {"AAPL": 0.6, "MSFT": 0.4}}'
```

---

## Summary: What Changed

### Before:
- TradingView only (2.7% pass rate)
- Max Sharpe + Semivariance
- No sell signals
- No backtesting

### After:
- Technical: pandas-ta indicators + multi-timeframe (7-13% pass rate)
- Fundamental: Quality scoring (ROE, debt, growth)
- Sell signals: Stop-loss, trailing, technical breakdown, fundamental
- Optimization: 5 methods (max_sharpe, min_vol, hrp, cvar, semivariance)
- Backtesting: bt library with walk-forward
- Factor tilting: momentum, value, growth, quality

### Effort:
- **Phase 1**: 2-3 days
- **Phase 2**: 2-3 days
- **Phase 3**: 2 days
- **Phase 4**: 2-3 days
- **Phase 5**: 3-4 days
- **Total**: 11-15 days (~2-3 weeks)
