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

## Setup & Run

### Prerequisites
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Installation
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.13

source .venv/bin/activate
uv pip install -r requirements.txt

# Create .env file with required variables
```

### Run
```sh
source .venv/bin/activate
python app.py
```

## Curl samples

```sh
curl --request POST --url https://api.telegram.org/bot$TELEGRAM_TOKEN/setWebhook --header 'content-type: application/json' --data '{"url": "https://2j48cpk83h.execute-api.us-east-1.amazonaws.com/dev"}'
#https://api.telegram.org/bot$TELEGRAM_TOKEN/getWebhookInfo
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