# Introduction

## Description

![status](https://github.com/lejinvarghese/stock-screener/actions/workflows/pylint.yml/badge.svg)

A stock screener that can analyze your  Wealthsimple (or custom) watchlist and recommend an optimized portfolio of stocks to a Telegram channel. It implements portfolio optimization methods, including classic General Efficient Frontier techniques, covariance shrinkage using Ledoit-Wolf etc. Future versions to include Black-Litterman Allocation, configurable objectives etc.

## Process

The selection process involves two key steps:

1. Pre-selection of watchlist stocks (based on both Technical Analysis and Fundamental Indicators) to assess an active Buy signal.
2. Portfolio Optimization (based on Efficient Frontier techniques) for an investment value.

## Sample

Ticker Trends: 
![Ticker Trend](docs/ohlc.png)

Efficient Frontier: 
![Efficient Frontier Optimization](docs/pf_optimizer.png)

Covariance Matrix: 
![Covariance Matrix](docs/pf_cov_matrix.png)

Covariance Matrix (Cluster Map): 
![Covariance Cluster Map](docs/pf_cov_clusters.png)

## Curl samples

```bash
curl --request POST --url https://api.telegram.org/bot$TELEGRAM_TOKEN/setWebhook --header 'content-type: application/json' --data '{"url": "https://2j48cpk83h.execute-api.us-east-1.amazonaws.com/dev"}'
#https://api.telegram.org/bot$TELEGRAM_TOKEN/getWebhookInfo
```
