# Portfolio Optimization Results
**Generated**: 2026-03-27
**Watchlist Source**: data/inputs/my_watchlist.csv
**Total Stocks Analyzed**: 57
**Budget**: $10,000
**Optimization Method**: Maximum Sharpe Ratio

---

## 🎯 Pre-Selected Stocks (Technical Analysis)

From 57 stocks, TradingView technical analysis identified **4 stocks with BUY signals**:

| Symbol | Company | Sector | Industry |
|--------|---------|--------|----------|
| **AEP** | American Electric Power Company | Utilities | Regulated Electric |
| **LITE** | Lumentum Holdings Inc. | Technology | Communication Equipment |
| **XOM** | Exxon Mobil Corporation | Energy | Oil & Gas Integrated |
| **ALB** | Albemarle Corporation | Materials | Specialty Chemicals |

---

## 💼 Optimized Portfolio Allocation

### Discrete Allocation ($10,000 Budget)

| Stock | Company | Weight | Shares | Estimated Value |
|-------|---------|--------|--------|-----------------|
| **AEP** | American Electric Power | 42.07% | 32 | $4,207 |
| **LITE** | Lumentum Holdings | 27.80% | 4 | $2,780 |
| **XOM** | Exxon Mobil | 30.13% | 17 | $3,013 |

**Total Invested**: ~$10,000
**Remaining Cash**: Minimal (optimized for full deployment)

---

## 📈 Expected Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Expected Annual Return** | 217.91% | Very high expected returns |
| **Annual Volatility** | 21.86% | Moderate risk level |
| **Sharpe Ratio** | 9.97 | Excellent risk-adjusted returns |

### What These Numbers Mean:

- **Expected Annual Return (217.91%)**: The portfolio is expected to more than double in value annually based on historical data
- **Annual Volatility (21.86%)**: The portfolio's returns may deviate by ±21.86% from the expected return
- **Sharpe Ratio (9.97)**: For every unit of risk taken, you earn ~10 units of excess return (>1 is good, >3 is excellent)

⚠️ **Important Note**: These metrics are based on historical data (2020 prices from your CSV). Actual future performance will vary. Past performance does not guarantee future results.

---

## 🎨 Sector Diversification

```
Utilities (AEP):     42.07% ███████████████████████▌
Technology (LITE):   27.80% ███████████████▋
Energy (XOM):        30.13% █████████████████
```

**Diversification Analysis**:
- **Defensive (Utilities)**: 42% - Provides stability and dividend income
- **Cyclical (Energy)**: 30% - Benefits from economic growth and commodity prices
- **Growth (Technology)**: 28% - High growth potential with higher volatility

This balanced mix provides exposure to different market sectors, reducing concentration risk.

---

## 📊 Generated Visualizations

The following charts were created in `data/outputs/`:

1. **pf_optimizer.png** (98KB)
   - Efficient Frontier plot
   - Shows the optimal risk-return trade-off
   - Your portfolio is marked on the efficient frontier

2. **pf_cov_clusters.png** (42KB)
   - Covariance cluster map
   - Shows correlation between stocks
   - Helps visualize diversification benefits

3. **pf_cov_matrix.png** (153KB)
   - Detailed covariance matrix heatmap
   - Shows how stocks move together
   - Darker colors indicate higher correlation

---

## 🔍 Technical Analysis Details

### Pre-Selection Criteria
- TradingView 1-day interval technical indicators
- Combined oscillators and moving averages
- Only stocks with "BUY" or "STRONG_BUY" recommendations selected
- Out of 57 stocks analyzed, 4 met the criteria (~7% pass rate)

### Optimization Method
- **Algorithm**: Modern Portfolio Theory (Markowitz)
- **Objective**: Maximize Sharpe Ratio
- **Constraints**:
  - Minimum position weight: 5%
  - Long-only (no short positions)
  - Sum of weights = 100%
- **Risk Model**: Ledoit-Wolf covariance shrinkage
- **Expected Returns**: EMA historical returns with shrinkage

---

## 💡 Investment Recommendations

### What to Buy (for $10,000 budget):
```bash
Buy 32 shares of AEP  @ ~$131.47 = $4,207.04
Buy  4 shares of LITE @ ~$695.00 = $2,780.00
Buy 17 shares of XOM  @ ~$177.24 = $3,013.08
                       Total     = $10,000.12
```

### Risk Considerations:
1. **Market Risk**: All stocks exposed to general market movements
2. **Sector Risk**: Energy sector (XOM) sensitive to oil prices
3. **Concentration**: Only 3 positions (limited diversification)
4. **Data Age**: Based on 2020 prices - refresh with current data before investing

### Next Steps:
1. ✅ Verify current stock prices (data is from 2020)
2. ✅ Check your risk tolerance against 21.86% volatility
3. ✅ Review each company's latest fundamentals
4. ✅ Consider your investment time horizon
5. ✅ Ensure proper asset allocation within your overall portfolio

---

## 🛠️ How to Run This Analysis Again

```bash
# With current market data
source .venv/bin/activate
python app.py --port 5004

# Then via web interface: http://localhost:5004
# Or via API:
curl -X POST http://localhost:5004/recommend_stocks/ \
  -H "Content-Type: application/json" \
  -d '{
    "threshold": 0.05,
    "budget": 10000,
    "method": "max_sharpe"
  }'
```

### Adjust Parameters:
- **threshold**: Minimum position weight (0.01-0.10)
  - Lower = more positions
  - Higher = fewer, more concentrated positions
- **budget**: Investment amount in dollars
- **method**: "max_sharpe" or "semivariance"
  - max_sharpe: Best risk-adjusted returns
  - semivariance: Focuses on downside risk protection

---

## ⚠️ Disclaimer

This portfolio optimization is for educational and informational purposes only. It is NOT financial advice. The analysis is based on:
- Historical price data (2020)
- Technical indicators
- Mathematical optimization

**Before investing**:
- Consult with a qualified financial advisor
- Do your own research on each company
- Consider your personal financial situation
- Understand that you can lose money in the stock market
- Past performance does not guarantee future results

---

*Generated by Stock Screener - A portfolio optimization tool using Modern Portfolio Theory and technical analysis*
