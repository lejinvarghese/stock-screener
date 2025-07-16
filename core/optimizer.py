#!/usr/bin/env python

"""
Optimize portfolio for stocks
"""

import os
from multiprocessing.pool import ThreadPool
from warnings import filterwarnings
import json
import pandas as pd
import numpy as np

# Set matplotlib backend before importing pyplot to avoid GUI issues in threads
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend for thread safety
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import (
    risk_models,
    expected_returns,
    discrete_allocation,
    plotting,
    objective_functions,
)
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

# from cvxpy import norm

try:
    from core.utils import get_data, get_price_ticker_matrix, send_image, send_message
except:
    from utils import get_data, get_price_ticker_matrix, send_image, send_message

filterwarnings("ignore")
load_dotenv()

PATH = os.getcwd()
N_TICKERS = 10

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_ID = os.getenv("TELEGRAM_ID")
TELEGRAM_BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# plotting parameters


def optimize(prices, value, n_tickers=N_TICKERS):
    """
    Optimize portfolio
    """
    # expected returns and sample covariance
    hist_returns = expected_returns.ema_historical_return(prices)
    # covariance_matrix = risk_models.exp_cov(prices, log_returns=True)
    covariance_matrix = risk_models.CovarianceShrinkage(
        prices, log_returns=True
    ).ledoit_wolf()

    # Create output directory if it doesn't exist
    import os

    os.makedirs(f"{PATH}/data/outputs", exist_ok=True)

    console.print("[blue]Generating portfolio optimization charts...[/blue]")

    # Create first EfficientFrontier instance for plotting
    ef_plotter = EfficientFrontier(hist_returns, covariance_matrix)
    try:
        plotting.plot_efficient_frontier(
            ef_plotter,
            ef_param="return",
            show_assets=True,
            filename=f"{PATH}/data/outputs/pf_optimizer.png",
        )
        plt.close()
    except:
        pass

    # Create second EfficientFrontier instance for optimization
    ef_optimizer = EfficientFrontier(hist_returns, covariance_matrix)

    initial_weights = np.array([1 / n_tickers] * n_tickers)
    ef_optimizer.add_objective(objective_functions.L2_reg, gamma=1)
    ef_optimizer.add_objective(
        objective_functions.transaction_cost, w_prev=initial_weights, k=0.02
    )

    # objective to solve for
    ef_optimizer.max_sharpe()

    cleaned_weights = ef_optimizer.clean_weights(cutoff=0.01)
    # Display cleaned weights in a nice table
    weights_table = Table(
        title="Optimized Portfolio Weights", show_header=True, header_style="bold green"
    )
    weights_table.add_column("Stock", style="cyan")
    weights_table.add_column("Weight", style="yellow")
    weights_table.add_column("Percentage", style="green")

    for stock, weight in cleaned_weights.items():
        if weight > 0:
            weights_table.add_row(stock, f"{weight:.4f}", f"{weight*100:.2f}%")

    console.print(weights_table)
    # Display performance metrics
    perf = ef_optimizer.portfolio_performance(verbose=True)
    perf_table = Table(
        title="Portfolio Performance", show_header=True, header_style="bold blue"
    )
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="yellow")

    perf_table.add_row("Expected Annual Return", f"{perf[0]*100:.2f}%")
    perf_table.add_row("Annual Volatility", f"{perf[1]*100:.2f}%")
    perf_table.add_row("Sharpe Ratio", f"{perf[2]:.2f}")

    console.print(perf_table)

    plotting.plot_covariance(
        covariance_matrix,
        filename=f"{PATH}/data/outputs/pf_cov_matrix.png",
    )
    plt.close()
    with sns.plotting_context(context="paper", font_scale=0.75):
        sns.clustermap(covariance_matrix)
    plt.savefig(f"{PATH}/data/outputs/pf_cov_clusters.png")
    plt.close()

    latest_prices = discrete_allocation.get_latest_prices(prices)
    allocation, leftover = discrete_allocation.DiscreteAllocation(
        cleaned_weights, latest_prices, total_portfolio_value=value
    ).lp_portfolio()
    # Display allocation in a nice table
    allocation_table = Table(
        title="Recommended Stock Allocation",
        show_header=True,
        header_style="bold magenta",
    )
    allocation_table.add_column("Stock", style="cyan")
    allocation_table.add_column("Shares", style="yellow")

    for stock, shares in allocation.items():
        allocation_table.add_row(stock, str(shares))

    console.print(allocation_table)
    console.print(f"[green]Funds remaining: ${leftover:.2f}[/green]")

    send_image(TELEGRAM_TOKEN, TELEGRAM_ID, f"{PATH}/data/outputs/pf_optimizer.png")
    send_image(TELEGRAM_TOKEN, TELEGRAM_ID, f"{PATH}/data/outputs/pf_cov_clusters.png")
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

    return {
        "allocation": allocation,
        "weights": cleaned_weights,
        "performance": ef_optimizer.portfolio_performance(verbose=True),
    }


def run(tickers, value=1000):
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
    console.print(f"[blue]Received data for {len(data)} stocks[/blue]")
    prices = get_price_ticker_matrix(data)

    optimization_result = optimize(prices, value=value, n_tickers=len(data))
    return optimization_result


if __name__ == "__main__":
    df_watchlist = pd.read_csv(f"{PATH}/data/inputs/my_watchlist.csv")
    tickers = list(df_watchlist.Symbol.unique())[:N_TICKERS]
    result = run(tickers)
    console.print(
        Panel(
            f"[bold green]Optimization completed![/bold green]",
            title="Success",
            border_style="green",
        )
    )
    print(json.dumps(result))
