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
import click

import matplotlib

matplotlib.use("Agg")
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
    EfficientSemivariance,
)
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

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

os.makedirs(f"{PATH}/data/outputs", exist_ok=True)


def optimize(
    prices,
    budget,
    n_tickers=N_TICKERS,
    cutoff=0.01,
    fractional_threshold=0.05,
    send_telegram=True,
):
    """
    Optimize portfolio
    """
    mu = expected_returns.ema_historical_return(prices)

    gamma = 0.3  # shrinkage intensity
    mu = (1 - gamma) * mu + gamma * mu.mean()

    # covariance_matrix = risk_models.exp_cov(prices, log_returns=True)
    covariance_matrix = risk_models.CovarianceShrinkage(
        prices, log_returns=True
    ).ledoit_wolf()

    console.print("[blue]Generating portfolio optimization charts...[/blue]")
    ef_optimizer = EfficientFrontier(mu, covariance_matrix)

    # historical_returns = expected_returns.returns_from_prices(df)
    # ef_optimizer = EfficientSemivariance(mu, historical_returns)
    # ef_optimizer.efficient_return(0.20)

    initial_weights = np.array([1 / n_tickers] * n_tickers)
    ef_optimizer.add_objective(objective_functions.L2_reg, gamma=0.1)
    ef_optimizer.add_objective(
        objective_functions.transaction_cost, w_prev=initial_weights, k=0.001
    )
    ef_optimizer.max_sharpe()
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

    cleaned_weights = ef_optimizer.clean_weights(cutoff=cutoff)
    filtered_weights = {
        stock: weight for stock, weight in cleaned_weights.items() if weight > 0
    }
    total_weight = sum(filtered_weights.values())
    filtered_weights = {
        stock: weight / total_weight for stock, weight in filtered_weights.items()
    }
    sorted_weights = sorted(filtered_weights.items(), key=lambda x: x[1], reverse=True)
    weights_table = Table(
        title="Optimized Portfolio Weights", show_header=True, header_style="bold green"
    )
    weights_table.add_column("Stock", style="cyan")
    weights_table.add_column("Percentage", style="green")

    for stock, weight in sorted_weights:
        weights_table.add_row(stock, f"{weight*100:.2f}%")

    console.print(weights_table)

    perf = ef_optimizer.portfolio_performance(verbose=True)
    perf_table = Table(
        title="Portfolio Performance", show_header=True, header_style="bold blue"
    )
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Budget", style="yellow")

    perf_table.add_row("Expected Annual Return", f"{perf[0]*100:.2f}%")
    perf_table.add_row("Annual Volatility", f"{perf[1]*100:.2f}%")
    perf_table.add_row("Sharpe Ratio", f"{perf[2]:.2f}")

    console.print(perf_table)

    with sns.plotting_context(context="paper", font_scale=0.75):
        sns.clustermap(covariance_matrix, vmin=0.05, vmax=0.5)
    plt.savefig(f"{PATH}/data/outputs/pf_cov_clusters.png")
    plt.close()

    # Get prices only for stocks with non-zero weights
    filtered_prices = prices[list(filtered_weights.keys())]
    latest_prices = discrete_allocation.get_latest_prices(filtered_prices)

    # Start with discrete allocation
    allocation, leftover = discrete_allocation.DiscreteAllocation(
        filtered_weights, latest_prices, total_portfolio_value=budget
    ).lp_portfolio()

    # Add fractional shares for stocks above threshold that have 0 shares
    for stock, weight in filtered_weights.items():
        if weight >= fractional_threshold and allocation.get(stock, 0) == 0:
            # Calculate fractional shares for this stock
            target_value = weight * budget
            stock_price = latest_prices[stock]
            fractional_shares = target_value / stock_price
            allocation[stock] = fractional_shares
            console.print(
                f"[yellow]Added fractional shares for {stock}: {fractional_shares:.3f} shares (${target_value:.2f})[/yellow]"
            )
    allocation_table = Table(
        title="Recommended Stock Allocation",
        show_header=True,
        header_style="bold magenta",
    )
    allocation_table.add_column("Stock", style="cyan")
    allocation_table.add_column("Shares", style="yellow")

    sorted_allocation = sorted(allocation.items(), key=lambda x: x[1], reverse=True)

    for stock, shares in sorted_allocation:
        if shares == int(shares):
            allocation_table.add_row(stock, str(int(shares)))
        else:
            allocation_table.add_row(stock, f"{shares:.2f}")

    console.print(allocation_table)
    console.print(f"[green]Funds remaining: ${leftover:.2f}[/green]")

    if send_telegram:
        send_image(TELEGRAM_TOKEN, TELEGRAM_ID, f"{PATH}/data/outputs/pf_optimizer.png")
        send_image(
            TELEGRAM_TOKEN, TELEGRAM_ID, f"{PATH}/data/outputs/pf_cov_clusters.png"
        )
        send_image(
            TELEGRAM_TOKEN, TELEGRAM_ID, f"{PATH}/data/outputs/pf_cov_matrix.png"
        )
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
                *perf
            ),
        )

    return {
        "allocation": allocation,
        "weights": filtered_weights,
        "performance": perf,
    }


def run(
    tickers, budget=1000, cutoff=0.01, fractional_threshold=0.05, send_telegram=True
):
    """
    Runs the Optimization pipeline
    1. Retrieves the data
    2. Reformats the data
    3. Runs the optimizer
    """
    if send_telegram:
        send_message(TELEGRAM_TOKEN, TELEGRAM_ID, "Optimizing Portfolio: ")

    with ThreadPool() as t_pool:
        data = t_pool.map(get_data, tickers)
    data = list(filter(None.__ne__, data))
    console.print(f"[blue]Received data for {len(data)} stocks[/blue]")

    prices = get_price_ticker_matrix(data)

    optimization_result = optimize(
        prices,
        budget=budget,
        n_tickers=len(data),
        cutoff=cutoff,
        fractional_threshold=fractional_threshold,
        send_telegram=send_telegram,
    )
    return optimization_result


@click.command()
@click.option(
    "--threshold",
    default=0.1,
    help="Minimum weight threshold for cleaned weights (default: 0.1)",
)
@click.option(
    "--budget", default=1000, help="Portfolio value in dollars (default: 1000)"
)
@click.option("--send-telegram", is_flag=True, help="Send messages to Telegram")
def main(threshold, budget, send_telegram):
    """Portfolio optimization with sample stocks"""
    sample_tickers = [
        "AAPL",
        "GOOGL",
        "MSFT",
        "AMZN",
        "TSLA",
        "NVDA",
        "META",
        "NFLX",
        "ADBE",
        "CRM",
        "PYPL",
        "INTC",
        "AMD",
        "ORCL",
        "CSCO",
    ]

    console.print(
        f"[blue]Using threshold {threshold} for portfolio optimization[/blue]"
    )
    result = run(
        sample_tickers, budget=budget, cutoff=threshold, send_telegram=send_telegram
    )
    console.print(
        Panel(
            f"[bold green]Optimization completed![/bold green]",
            title="Success",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
