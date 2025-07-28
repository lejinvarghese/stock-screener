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
    method="max_sharpe",
):
    """
    Optimize portfolio using specified method
    Available methods:
    - max_sharpe: Maximum Sharpe ratio using EfficientFrontier
    - semivariance: EfficientSemivariance for downside risk optimization
    """
    console.print(f"[blue]Using optimization method: {method}[/blue]")
    
    mu = expected_returns.ema_historical_return(prices)
    gamma = 0.3  # shrinkage intensity
    mu = (1 - gamma) * mu + gamma * mu.mean()

    console.print("[blue]Generating portfolio optimization charts...[/blue]")
    
    # Always calculate covariance matrix for plotting
    covariance_matrix = risk_models.CovarianceShrinkage(
        prices, log_returns=True
    ).ledoit_wolf()
    
    if method == "semivariance":
        # Use EfficientSemivariance for downside risk optimization
        historical_returns = expected_returns.returns_from_prices(prices)
        ef_optimizer = EfficientSemivariance(mu, historical_returns)
        console.print("[blue]Using EfficientSemivariance for downside risk optimization[/blue]")
    else:
        # Default to EfficientFrontier with max Sharpe
        ef_optimizer = EfficientFrontier(mu, covariance_matrix)
        console.print("[blue]Using EfficientFrontier for maximum Sharpe ratio[/blue]")

    initial_weights = np.array([1 / n_tickers] * n_tickers)
    
    # Add objectives based on optimization method
    if method == "semivariance":
        # EfficientSemivariance supports fewer objectives
        ef_optimizer.add_objective(objective_functions.L2_reg, gamma=0.01)
    else:
        # EfficientFrontier supports more objectives
        ef_optimizer.add_objective(objective_functions.L2_reg, gamma=0.01)
        ef_optimizer.add_objective(
            objective_functions.transaction_cost, w_prev=initial_weights, k=0.001
        )
    
    # Try optimization with fallback solvers
    solvers_to_try = ['OSQP', 'ECOS', 'SCS']
    optimization_successful = False
    
    for solver in solvers_to_try:
        try:
            console.print(f"[dim]Attempting optimization with {solver} solver...[/dim]")
            ef_optimizer._solver = solver
            
            # Run optimization based on method
            if method == "semivariance":
                # For semivariance, target a reasonable return (20%)
                ef_optimizer.efficient_return(0.20)
            else:
                # Default max Sharpe optimization
                ef_optimizer.max_sharpe()
                
            console.print(f"[green]Optimization successful with {solver} solver[/green]")
            optimization_successful = True
            break
        except Exception as e:
            console.print(f"[yellow]Warning: {solver} solver failed: {str(e)}[/yellow]")
            # Reset the optimizer for next attempt
            if method == "semivariance":
                historical_returns = expected_returns.returns_from_prices(prices)
                ef_optimizer = EfficientSemivariance(mu, historical_returns)
                ef_optimizer.add_objective(objective_functions.L2_reg, gamma=0.1)
            else:
                ef_optimizer = EfficientFrontier(mu, covariance_matrix)
                ef_optimizer.add_objective(objective_functions.L2_reg, gamma=0.1)
                ef_optimizer.add_objective(
                    objective_functions.transaction_cost, w_prev=initial_weights, k=0.001
                )
            continue
    
    if not optimization_successful:
        console.print("[red]ERROR: All solvers failed. Trying simplified optimization...[/red]")
        # Try with no additional objectives as fallback
        if method == "semivariance":
            historical_returns = expected_returns.returns_from_prices(prices)
            ef_optimizer = EfficientSemivariance(mu, historical_returns)
            try:
                ef_optimizer.efficient_return(0.15)  # Lower target for fallback
                console.print("[green]Simplified semivariance optimization successful[/green]")
            except Exception as e:
                console.print(f"[red]ERROR: Even simplified semivariance optimization failed: {str(e)}[/red]")
                raise RuntimeError(f"Semivariance portfolio optimization failed with all solvers: {str(e)}")
        else:
            ef_optimizer = EfficientFrontier(mu, covariance_matrix)
            try:
                ef_optimizer.max_sharpe()
                console.print("[green]Simplified max Sharpe optimization successful[/green]")
            except Exception as e:
                console.print(f"[red]ERROR: Even simplified max Sharpe optimization failed: {str(e)}[/red]")
                raise RuntimeError(f"Max Sharpe portfolio optimization failed with all solvers: {str(e)}")
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
    
    # Check if all weights were filtered out
    if not filtered_weights:
        console.print(f"[yellow]WARNING: All weights below cutoff ({cutoff}). Using lower cutoff for {method} method...[/yellow]")
        # For semivariance, try with a much lower cutoff or no cutoff
        if method == "semivariance":
            cleaned_weights = ef_optimizer.clean_weights(cutoff=0.001)  # Very low cutoff
            filtered_weights = {
                stock: weight for stock, weight in cleaned_weights.items() if weight > 0
            }
            console.print(f"[blue]Found {len(filtered_weights)} stocks with cutoff 0.001[/blue]")
            
        # If still empty, use top weights without cutoff
        if not filtered_weights:
            console.print("[yellow]Still no weights found. Using top 5 weights without cutoff...[/yellow]")
            raw_weights = ef_optimizer.clean_weights(cutoff=0.0)
            # Get top 5 stocks by weight
            sorted_raw = sorted(raw_weights.items(), key=lambda x: x[1], reverse=True)[:5]
            filtered_weights = dict(sorted_raw)
            console.print(f"[blue]Selected top {len(filtered_weights)} stocks: {list(filtered_weights.keys())}[/blue]")
    
    # Normalize weights to sum to 1
    total_weight = sum(filtered_weights.values())
    if total_weight > 0:
        filtered_weights = {
            stock: weight / total_weight for stock, weight in filtered_weights.items()
        }
    else:
        console.print("[red]ERROR: No valid weights found even after fallback. Cannot proceed with optimization.[/red]")
        raise ValueError("Portfolio optimization failed - no valid weights found")
    
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
    tickers, budget=1000, cutoff=0.01, fractional_threshold=0.05, send_telegram=True, method="max_sharpe"
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

    # Filter out stocks with insufficient data (need at least 252 trading days for reliable optimization)
    MIN_HISTORY_DAYS = 252
    filtered_data = []
    for stock_data in data:
        if stock_data is not None and len(stock_data) >= MIN_HISTORY_DAYS:
            filtered_data.append(stock_data)
        else:
            ticker = stock_data.ticker.iloc[0] if stock_data is not None and 'ticker' in stock_data.columns else 'Unknown'
            console.print(f"[yellow]Filtering out {ticker}: insufficient data ({len(stock_data) if stock_data is not None else 0} days, need {MIN_HISTORY_DAYS})[/yellow]")
    
    if len(filtered_data) < 2:
        console.print(f"[red]ERROR: Need at least 2 stocks with sufficient data for optimization. Only have {len(filtered_data)}[/red]")
        raise ValueError(f"Insufficient stocks for optimization. Need at least 2 stocks with {MIN_HISTORY_DAYS}+ days of data")
    
    console.print(f"[green]Using {len(filtered_data)} stocks with sufficient data for optimization[/green]")
    data = filtered_data

    prices = get_price_ticker_matrix(data)

    optimization_result = optimize(
        prices,
        budget=budget,
        n_tickers=len(data),
        cutoff=cutoff,
        fractional_threshold=fractional_threshold,
        send_telegram=send_telegram,
        method=method,
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
@click.option(
    "--method",
    default="max_sharpe",
    type=click.Choice(['max_sharpe', 'semivariance']),
    help="Optimization method: max_sharpe or semivariance (default: max_sharpe)"
)
def main(threshold, budget, send_telegram, method):
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
        f"[blue]Using threshold {threshold} and method {method} for portfolio optimization[/blue]"
    )
    result = run(
        sample_tickers, budget=budget, cutoff=threshold, send_telegram=send_telegram, method=method
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
