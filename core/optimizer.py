#!/usr/bin/env python

"""
Optimize portfolio for stocks
"""

import os
from multiprocessing.pool import ThreadPool
from warnings import filterwarnings
import json
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import (
    risk_models,
    expected_returns,
    discrete_allocation,
    plotting,
    objective_functions,
)

try:
    from core.utils import get_data, get_prices, send_image, send_message
except:
    from utils import get_data, get_prices, send_image, send_message

filterwarnings("ignore")
load_dotenv()

DIRECTORY = "/media/starscream/wheeljack/projects/"
PROJECT = "stock-screener"
PATH = os.path.join(DIRECTORY, PROJECT)
N_TICKERS = 10

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_ID = os.getenv("TELEGRAM_ID")
TELEGRAM_BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# plotting parameters


def optimize(prices, value=1000):
    """
    Optimize portfolio
    """
    # expected returns and sample covariance
    hist_returns = expected_returns.ema_historical_return(prices)
    covariance_matrix = risk_models.exp_cov(prices, log_returns=True)

    # optimize portfolio for the objectives
    ef_optimizer = EfficientFrontier(hist_returns, covariance_matrix)
    try:
        plotting.plot_efficient_frontier(
            ef_optimizer,
            ef_param="return",
            show_assets=True,
            filename=f"{PATH}/data/outputs/pf_optimizer.png",
        )
        plt.close()
    except:
        pass

    ef_optimizer.add_objective(objective_functions.L2_reg, gamma=0.1)

    # objective to solve for
    ef_optimizer.max_sharpe()
    # ef_optimizer.min_volatility()
    # ef_optimizer.max_quadratic_utility()
    # ef_optimizer.efficient_risk(target_volatility=0.3)
    # ef_optimizer.efficient_return(target_return=1.0)

    cleaned_weights = ef_optimizer.clean_weights(cutoff=0.01)
    print(f"Cleaned Weights: {cleaned_weights}")
    print(f"Performance: {ef_optimizer.portfolio_performance(verbose=True)}")

    plotting.plot_covariance(
        risk_models.sample_cov(prices, log_returns=True),
        filename=f"{PATH}/data/outputs/pf_cov_matrix.png",
    )
    plt.close()

    latest_prices = discrete_allocation.get_latest_prices(prices)
    allocation, leftover = discrete_allocation.DiscreteAllocation(
        cleaned_weights, latest_prices, total_portfolio_value=value
    ).lp_portfolio()
    print(f"Recommended Allocation: {allocation}")
    print("Funds remaining: ${:.2f}".format(leftover))

    send_image(TELEGRAM_TOKEN, TELEGRAM_ID, f"{PATH}/data/outputs/pf_optimizer.png")
    send_image(TELEGRAM_TOKEN, TELEGRAM_ID, f"{PATH}/data/outputs/pf_cov_matrix.png")
    send_message(
        TELEGRAM_TOKEN,
        TELEGRAM_ID,
        f"Recommended Allocation: A portfolio {len(allocation)} stocks: {allocation}",
    )
    send_message(
        TELEGRAM_TOKEN,
        TELEGRAM_ID,
        "Performance: Expected Annual Return: {:.2%}, \
        Volatility: {:.2%}, Sharpe Ratio: {:.2f}".format(
            *list(ef_optimizer.portfolio_performance())
        ),
    )

    return allocation


def run(tickers):
    """
    Runs the Optimization pipeline
    1. Retrieves the data
    2. Reformats the data
    3. Runs the optimizer
    """
    send_message(TELEGRAM_TOKEN, TELEGRAM_ID, "Optimizing Portfolio: ")

    with ThreadPool() as t_pool:
        data = t_pool.map(get_data, tickers)
    data = list(filter(None.__ne__, data))
    prices = get_prices(data)

    return [*optimize(prices, value=2500).keys()]


if __name__ == "__main__":
    df_watchlist = pd.read_csv(f"{PATH}/data/inputs/my_watchlist.csv")
    tickers = list(df_watchlist.Symbol.unique())[:N_TICKERS]
    print(json.dumps(run(tickers)))
