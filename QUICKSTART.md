# Quick Start Guide

## Setup (One Time)

```bash
# 1. Create venv with Python 3.11
uv venv --python 3.11

# 2. Activate venv
source .venv/bin/activate

# 3. Install all dependencies
uv pip install -r requirements.txt

# 4. Verify installation
python -c "import bt; import empyrical; print('✓ All packages installed')"
```

## Run

```bash
# Activate venv
source .venv/bin/activate

# Start app (default port 5004)
python app.py

# Or custom port
python app.py --port 8000
```

## Access

Open browser: **http://localhost:8000** (or your port)

## Test New Features

### 1. Check Sell Signals
```bash
curl -X POST http://localhost:8000/check_sells/ \
  -H "Content-Type: application/json" \
  -d '{
    "holdings": [
      {"symbol": "AAPL", "entry_price": 250, "shares": 10}
    ]
  }'
```

### 2. Try Different Optimization Methods
```bash
# Minimum Volatility (lowest risk)
curl -X POST http://localhost:8000/recommend_stocks/ \
  -H "Content-Type: application/json" \
  -d '{"method": "min_vol", "budget": 10000}'

# Hierarchical Risk Parity (best diversification)
curl -X POST http://localhost:8000/recommend_stocks/ \
  -H "Content-Type: application/json" \
  -d '{"method": "hrp", "budget": 10000}'

# CVaR (tail risk optimization)
curl -X POST http://localhost:8000/recommend_stocks/ \
  -H "Content-Type: application/json" \
  -d '{"method": "cvar", "budget": 10000}'
```

### 3. Backtest a Portfolio
```bash
curl -X POST http://localhost:8000/backtest/ \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT", "GOOGL"],
    "weights": {"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25},
    "start_date": "2022-01-01"
  }'
```

### 4. Compare Methods
```bash
curl -X POST http://localhost:8000/compare_methods/ \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT", "GOOGL", "NVDA"],
    "methods": ["max_sharpe", "min_vol", "hrp"]
  }'
```

## Troubleshooting

### Module Not Found Error
```bash
# Reinstall packages
source .venv/bin/activate
uv pip install bt empyrical
```

### Port Already in Use
```bash
# Use different port
python app.py --port 8080
```

### Check Logs
```bash
# App logs show in terminal
# Or redirect to file
python app.py --port 8000 > app.log 2>&1 &
tail -f app.log
```

## Available Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `max_sharpe` | Maximum Sharpe ratio | Best risk-adjusted returns |
| `min_vol` | Minimum volatility | Lowest risk portfolio |
| `hrp` | Hierarchical Risk Parity | Best diversification |
| `cvar` | Conditional Value at Risk | Tail risk protection |
| `semivariance` | Downside risk | Downside protection |

## Next Steps

1. Import your watchlist (`POST /watchlist/import`)
2. Run portfolio optimization with different methods
3. Backtest historical performance
4. Check existing holdings for sell signals
5. Monitor portfolio risk (`POST /check_portfolio/`)
