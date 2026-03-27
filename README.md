# Introduction

## Description

![status](https://github.com/lejinvarghese/stock-screener/actions/workflows/pylint.yml/badge.svg)

A stock screener that can analyze your Wealthsimple (or custom) watchlist and recommend an optimized portfolio of stocks to a Telegram channel. It implements portfolio optimization methods, including classic General Efficient Frontier techniques, covariance shrinkage using Ledoit-Wolf etc. Future versions to include Black-Litterman Allocation, configurable objectives etc.

## Process

The selection process involves two key steps:

1. Pre-selection of watchlist stocks (based on Trading View 1 Week Interval technical analysis based recommendations) to assess an active Buy signal.
    1. Trading View uses an ensemble of Lagging Indicators (Moving Averages) and Leading Indicators (Oscillators) to summarize the final recommendation.
    2. [Sample: TSLA](https://www.tradingview.com/symbols/NASDAQ-TSLA/technicals/)
2. Portfolio Optimization (based on Efficient Frontier techniques) for an investment value.

![image](https://pyportfolioopt.readthedocs.io/en/latest/_images/efficient_frontier.png)

## Sample

Ticker Trends:
![Ticker Trend](docs/ohlc.png)

Efficient Frontier:
![Efficient Frontier Optimization](docs/pf_optimizer.png)

Covariance Matrix:
![Covariance Matrix](docs/pf_cov_matrix.png)

Covariance Matrix (Cluster Map):
![Covariance Cluster Map](docs/pf_cov_clusters.png)

## Quick Start (TL;DR)

```bash
# 1. Setup environment
curl -LsSf https://astral.sh/uv/install.sh | sh  # Install uv
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Configure (create .env from .env.example and add your credentials)
cp .env.example .env
nano .env  # or vim, code, etc.

# 3. Run
python app.py --port 5004

# 4. Open browser → http://localhost:5004
```

## Setup & Run

### Prerequisites
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended)
- Python 3.11+ (required for numpy 2.x and pandas 2.x)

### Installation
```sh
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with Python 3.11
uv venv --python 3.11

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Create .env file with required variables (see .env.example)
```

### Run


#### Terminal/CLI Usage
```sh
# Activate the virtual environment
source .venv/bin/activate

# Run on default port (5004)
python app.py

# Or specify a custom port
python app.py --port 8000

# Run in background
python app.py --port 5004 > app.log 2>&1 &

# View logs (if running in background)
tail -f app.log
```

Access the web interface at: **http://localhost:5004** (or your chosen port)

## Terminal/CLI Usage

### Basic Operations

```sh
# Start the app (runs on http://localhost:5004)
source .venv/bin/activate
python app.py

# Custom port
python app.py --port 8080

# Background mode
python app.py --port 5004 > app.log 2>&1 &

# Check if app is running
curl http://localhost:5004/

# View logs (if running in background)
tail -f app.log

# Stop background process
pkill -f "python.*app.py"
```

### API Testing with curl

#### Watchlist Management
```sh
# Get all symbols in watchlist
curl http://localhost:5004/watchlist/

# Add a stock symbol
curl -X POST http://localhost:5004/watchlist/add \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'

# Add multiple stocks
for symbol in MSFT GOOGL NVDA TSLA; do
  curl -X POST http://localhost:5004/watchlist/add \
    -H "Content-Type: application/json" \
    -d "{\"symbol\": \"$symbol\"}"
done

# Remove a stock
curl -X POST http://localhost:5004/watchlist/remove \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'

# Clear entire watchlist
curl -X POST http://localhost:5004/watchlist/clear
```

#### Portfolio Optimization
```sh
# Run portfolio analysis with default settings
curl -X POST http://localhost:5004/recommend_stocks/ \
  -H "Content-Type: application/json" \
  -d '{
    "threshold": 0.05,
    "budget": 10000,
    "method": "max_sharpe"
  }'

# Use semivariance method (downside risk optimization)
curl -X POST http://localhost:5004/recommend_stocks/ \
  -H "Content-Type: application/json" \
  -d '{
    "threshold": 0.01,
    "budget": 25000,
    "method": "semivariance"
  }'

# Save results to file
curl -X POST http://localhost:5004/recommend_stocks/ \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.05, "budget": 10000, "method": "max_sharpe"}' \
  -o portfolio_results.json

# View results
cat portfolio_results.json | python -m json.tool
```

#### Import/Export Watchlist
```sh
# Export current watchlist to CSV
curl http://localhost:5004/watchlist/export -o my_watchlist.csv

# Import watchlist from CSV
curl -X POST http://localhost:5004/watchlist/import \
  -H "Content-Type: application/json" \
  -d "{\"csv_content\": \"$(cat my_watchlist.csv)\"}"
```

### Telegram Bot Setup (Optional)
```sh
# Set webhook for Telegram bot
curl --request POST \
  --url https://api.telegram.org/bot$TELEGRAM_TOKEN/setWebhook \
  --header 'content-type: application/json' \
  --data '{"url": "https://your-domain.com/webhook"}'

# Check webhook info
curl https://api.telegram.org/bot$TELEGRAM_TOKEN/getWebhookInfo
```

## Environment Variables

Must have a `*.env` with the following variables for full functionality:

```sh
TELEGRAM_TOKEN=XXX
TELEGRAM_ID=XXX
WEALTHSIMPLE_USERNAME=XXX
WEALTHSIMPLE_PASSWORD=XXX
```

Note: Wealthsimple has an added layer of security with an OTP (One Time Password) that you'll need to enter in the terminal everytime you run the application.