#!/usr/bin/env python3
"""
Practical Portfolio Management Examples
Using PyPortfolioOpt + yfinance + existing libraries
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import (
    expected_returns,
    risk_models,
    EfficientFrontier,
    EfficientSemivariance,
    EfficientCVaR,
    EfficientCDaR,
    HRPOpt,
    CLA,
    BlackLittermanModel,
    objective_functions,
    DiscreteAllocation,
    get_latest_prices,
)

# ============================================================================
# EXAMPLE 1: Basic Max Sharpe Portfolio
# ============================================================================

def example_max_sharpe():
    """Standard maximum Sharpe ratio optimization"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Maximum Sharpe Ratio Portfolio")
    print("="*80)

    # Get data
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Adj Close']

    # Calculate expected returns and covariance
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.CovarianceShrinkage(data).ledoit_wolf()

    # Optimize
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    print("\nOptimal Weights:")
    for ticker, weight in cleaned_weights.items():
        if weight > 0:
            print(f"  {ticker}: {weight*100:.2f}%")

    # Performance
    perf = ef.portfolio_performance(verbose=True)
    print(f"\nExpected Return: {perf[0]*100:.2f}%")
    print(f"Volatility: {perf[1]*100:.2f}%")
    print(f"Sharpe Ratio: {perf[2]:.2f}")

    return cleaned_weights, data


# ============================================================================
# EXAMPLE 2: Minimum Volatility Portfolio
# ============================================================================

def example_min_volatility():
    """Conservative minimum volatility portfolio"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Minimum Volatility Portfolio")
    print("="*80)

    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Adj Close']

    mu = expected_returns.mean_historical_return(data)
    S = risk_models.CovarianceShrinkage(data).ledoit_wolf()

    ef = EfficientFrontier(mu, S)
    weights = ef.min_volatility()
    cleaned_weights = ef.clean_weights()

    print("\nOptimal Weights (Min Vol):")
    for ticker, weight in cleaned_weights.items():
        if weight > 0:
            print(f"  {ticker}: {weight*100:.2f}%")

    perf = ef.portfolio_performance(verbose=True)
    print(f"\nExpected Return: {perf[0]*100:.2f}%")
    print(f"Volatility: {perf[1]*100:.2f}%")
    print(f"Sharpe Ratio: {perf[2]:.2f}")


# ============================================================================
# EXAMPLE 3: Target Return Portfolio
# ============================================================================

def example_target_return():
    """Achieve specific return with minimum risk"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Target 20% Return Portfolio")
    print("="*80)

    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Adj Close']

    mu = expected_returns.mean_historical_return(data)
    S = risk_models.CovarianceShrinkage(data).ledoit_wolf()

    ef = EfficientFrontier(mu, S)
    weights = ef.efficient_return(target_return=0.20)  # 20% return
    cleaned_weights = ef.clean_weights()

    print("\nOptimal Weights (Target 20%):")
    for ticker, weight in cleaned_weights.items():
        if weight > 0:
            print(f"  {ticker}: {weight*100:.2f}%")

    perf = ef.portfolio_performance(verbose=True)
    print(f"\nExpected Return: {perf[0]*100:.2f}%")
    print(f"Volatility: {perf[1]*100:.2f}%")
    print(f"Sharpe Ratio: {perf[2]:.2f}")


# ============================================================================
# EXAMPLE 4: Downside Risk (Semivariance) Portfolio
# ============================================================================

def example_semivariance():
    """Focus on downside risk only"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Semivariance (Downside Risk) Portfolio")
    print("="*80)

    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Adj Close']

    mu = expected_returns.mean_historical_return(data)
    returns = expected_returns.returns_from_prices(data)

    # EfficientSemivariance uses returns, not covariance
    es = EfficientSemivariance(mu, returns)
    weights = es.efficient_return(target_return=0.20)
    cleaned_weights = es.clean_weights()

    print("\nOptimal Weights (Semivariance):")
    for ticker, weight in cleaned_weights.items():
        if weight > 0:
            print(f"  {ticker}: {weight*100:.2f}%")

    perf = es.portfolio_performance(verbose=True)
    print(f"\nExpected Return: {perf[0]*100:.2f}%")
    print(f"Semideviation: {perf[1]*100:.2f}%")
    print(f"Sortino Ratio: {perf[2]:.2f}")


# ============================================================================
# EXAMPLE 5: CVaR (Tail Risk) Portfolio
# ============================================================================

def example_cvar():
    """Minimize conditional value at risk (tail risk)"""
    print("\n" + "="*80)
    print("EXAMPLE 5: CVaR (Tail Risk) Portfolio")
    print("="*80)

    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Adj Close']

    mu = expected_returns.mean_historical_return(data)
    returns = expected_returns.returns_from_prices(data)

    # EfficientCVaR - focus on worst 5% of outcomes
    ecvar = EfficientCVaR(mu, returns, beta=0.95)
    weights = ecvar.efficient_return(target_return=0.20)
    cleaned_weights = ecvar.clean_weights()

    print("\nOptimal Weights (CVaR):")
    for ticker, weight in cleaned_weights.items():
        if weight > 0:
            print(f"  {ticker}: {weight*100:.2f}%")

    perf = ecvar.portfolio_performance(verbose=True)
    print(f"\nExpected Return: {perf[0]*100:.2f}%")
    print(f"CVaR: {perf[1]*100:.2f}%")


# ============================================================================
# EXAMPLE 6: Hierarchical Risk Parity (HRP)
# ============================================================================

def example_hrp():
    """Modern portfolio construction without covariance matrix issues"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Hierarchical Risk Parity (HRP)")
    print("="*80)

    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'JPM', 'BAC', 'XOM']
    data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Adj Close']

    returns = expected_returns.returns_from_prices(data)

    # HRP doesn't need expected returns or covariance
    hrp = HRPOpt(returns)
    weights = hrp.optimize()

    print("\nHRP Weights:")
    for ticker, weight in weights.items():
        if weight > 0.01:  # Show weights > 1%
            print(f"  {ticker}: {weight*100:.2f}%")

    perf = hrp.portfolio_performance(verbose=True)
    print(f"\nExpected Return: {perf[0]*100:.2f}%")
    print(f"Volatility: {perf[1]*100:.2f}%")
    print(f"Sharpe Ratio: {perf[2]:.2f}")


# ============================================================================
# EXAMPLE 7: Portfolio with Constraints
# ============================================================================

def example_constraints():
    """Add sector constraints and bounds"""
    print("\n" + "="*80)
    print("EXAMPLE 7: Portfolio with Sector Constraints")
    print("="*80)

    tickers = ['AAPL', 'GOOGL', 'MSFT', 'JPM', 'BAC', 'XOM', 'CVX']
    data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Adj Close']

    mu = expected_returns.mean_historical_return(data)
    S = risk_models.CovarianceShrinkage(data).ledoit_wolf()

    # Define sectors
    sector_mapper = {
        'AAPL': 'Tech',
        'GOOGL': 'Tech',
        'MSFT': 'Tech',
        'JPM': 'Finance',
        'BAC': 'Finance',
        'XOM': 'Energy',
        'CVX': 'Energy'
    }

    # Sector limits
    sector_lower = {'Tech': 0.2, 'Finance': 0.1}    # Min allocation
    sector_upper = {'Tech': 0.5, 'Finance': 0.3}    # Max allocation

    ef = EfficientFrontier(mu, S)
    ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)

    # Also add minimum weight per stock
    ef.add_constraint(lambda w: w >= 0.05)  # Min 5% per stock

    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    print("\nOptimal Weights (with constraints):")
    for ticker, weight in cleaned_weights.items():
        sector = sector_mapper[ticker]
        if weight > 0:
            print(f"  {ticker} ({sector}): {weight*100:.2f}%")

    perf = ef.portfolio_performance(verbose=True)
    print(f"\nExpected Return: {perf[0]*100:.2f}%")
    print(f"Volatility: {perf[1]*100:.2f}%")
    print(f"Sharpe Ratio: {perf[2]:.2f}")


# ============================================================================
# EXAMPLE 8: Portfolio with Transaction Costs
# ============================================================================

def example_transaction_costs():
    """Account for trading costs when rebalancing"""
    print("\n" + "="*80)
    print("EXAMPLE 8: Portfolio with Transaction Costs")
    print("="*80)

    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Adj Close']

    mu = expected_returns.mean_historical_return(data)
    S = risk_models.CovarianceShrinkage(data).ledoit_wolf()

    # Current portfolio (before rebalancing)
    current_weights = np.array([0.3, 0.2, 0.2, 0.2, 0.1])

    ef = EfficientFrontier(mu, S)

    # Add transaction cost objective (0.1% per trade)
    ef.add_objective(
        objective_functions.transaction_cost,
        w_prev=current_weights,
        k=0.001  # 0.1% cost
    )

    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    print("\nCurrent Portfolio:")
    for ticker, weight in zip(tickers, current_weights):
        print(f"  {ticker}: {weight*100:.2f}%")

    print("\nOptimal Weights (with transaction costs):")
    for ticker, weight in cleaned_weights.items():
        if weight > 0:
            print(f"  {ticker}: {weight*100:.2f}%")

    perf = ef.portfolio_performance(verbose=True)
    print(f"\nExpected Return: {perf[0]*100:.2f}%")
    print(f"Volatility: {perf[1]*100:.2f}%")
    print(f"Sharpe Ratio: {perf[2]:.2f}")


# ============================================================================
# EXAMPLE 9: Black-Litterman with Views
# ============================================================================

def example_black_litterman():
    """Incorporate market views into optimization"""
    print("\n" + "="*80)
    print("EXAMPLE 9: Black-Litterman with Market Views")
    print("="*80)

    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Adj Close']

    S = risk_models.CovarianceShrinkage(data).ledoit_wolf()

    # Market capitalizations (approximate)
    market_caps = {
        'AAPL': 3000e9,
        'GOOGL': 1500e9,
        'MSFT': 2500e9,
        'AMZN': 1600e9,
        'TSLA': 800e9
    }

    # Create Black-Litterman model
    bl = BlackLittermanModel(S, pi='market', market_caps=market_caps)

    # Add views: "I think AAPL will return 25%, TSLA will return -5%"
    viewdict = {
        'AAPL': 0.25,   # Bullish on Apple
        'TSLA': -0.05   # Bearish on Tesla
    }

    # Get Black-Litterman returns
    bl_returns = bl.bl_returns()
    bl_cov = bl.bl_cov()

    # Optimize with BL returns
    ef = EfficientFrontier(bl_returns, bl_cov)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    print("\nMarket Views:")
    for ticker, view in viewdict.items():
        print(f"  {ticker}: {view*100:+.1f}% expected return")

    print("\nOptimal Weights (Black-Litterman):")
    for ticker, weight in cleaned_weights.items():
        if weight > 0:
            print(f"  {ticker}: {weight*100:.2f}%")

    perf = ef.portfolio_performance(verbose=True)
    print(f"\nExpected Return: {perf[0]*100:.2f}%")
    print(f"Volatility: {perf[1]*100:.2f}%")
    print(f"Sharpe Ratio: {perf[2]:.2f}")


# ============================================================================
# EXAMPLE 10: Discrete Allocation (Whole Shares)
# ============================================================================

def example_discrete_allocation():
    """Convert weights to actual shares to buy"""
    print("\n" + "="*80)
    print("EXAMPLE 10: Discrete Allocation for $10,000 Portfolio")
    print("="*80)

    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Adj Close']

    mu = expected_returns.mean_historical_return(data)
    S = risk_models.CovarianceShrinkage(data).ledoit_wolf()

    # Optimize
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    print("\nOptimal Weights:")
    for ticker, weight in cleaned_weights.items():
        if weight > 0:
            print(f"  {ticker}: {weight*100:.2f}%")

    # Get latest prices
    latest_prices = get_latest_prices(data)

    # Allocate $10,000
    budget = 10000
    da = DiscreteAllocation(
        cleaned_weights,
        latest_prices,
        total_portfolio_value=budget
    )

    allocation, leftover = da.lp_portfolio()

    print(f"\nFor ${budget:,.0f} investment:")
    print("\nShares to Buy:")
    total_invested = 0
    for ticker, shares in allocation.items():
        cost = shares * latest_prices[ticker]
        total_invested += cost
        print(f"  {ticker}: {shares} shares @ ${latest_prices[ticker]:.2f} = ${cost:,.2f}")

    print(f"\nTotal Invested: ${total_invested:,.2f}")
    print(f"Cash Leftover: ${leftover:.2f}")


# ============================================================================
# EXAMPLE 11: Multiple Risk Measures Comparison
# ============================================================================

def example_compare_risk_measures():
    """Compare different risk measures on same portfolio"""
    print("\n" + "="*80)
    print("EXAMPLE 11: Comparing Different Risk Measures")
    print("="*80)

    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Adj Close']

    mu = expected_returns.mean_historical_return(data)
    S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
    returns = expected_returns.returns_from_prices(data)

    results = []

    # 1. Standard Mean-Variance
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe()
    w1 = ef.clean_weights()
    p1 = ef.portfolio_performance()
    results.append(('Mean-Variance', w1, p1))

    # 2. Semivariance (downside)
    es = EfficientSemivariance(mu, returns)
    es.efficient_return(target_return=0.20)
    w2 = es.clean_weights()
    p2 = es.portfolio_performance()
    results.append(('Semivariance', w2, p2))

    # 3. CVaR (tail risk)
    ecvar = EfficientCVaR(mu, returns, beta=0.95)
    ecvar.efficient_return(target_return=0.20)
    w3 = ecvar.clean_weights()
    p3 = ecvar.portfolio_performance()
    results.append(('CVaR', w3, p3))

    # 4. CDaR (drawdown risk)
    ecdar = EfficientCDaR(mu, returns, beta=0.95)
    ecdar.efficient_return(target_return=0.20)
    w4 = ecdar.clean_weights()
    p4 = ecdar.portfolio_performance()
    results.append(('CDaR', w4, p4))

    # 5. HRP (hierarchical)
    hrp = HRPOpt(returns)
    w5 = hrp.optimize()
    p5 = hrp.portfolio_performance()
    results.append(('HRP', w5, p5))

    # Print comparison
    print("\n{:<20} {:>12} {:>12} {:>12}".format("Method", "Return", "Risk", "Sharpe"))
    print("-" * 60)
    for name, weights, perf in results:
        print("{:<20} {:>11.2%} {:>11.2%} {:>12.2f}".format(
            name, perf[0], perf[1], perf[2]
        ))

    # Show weight differences
    print("\n\nWeight Allocation Comparison:")
    print("{:<10}".format("Ticker"), end="")
    for name, _, _ in results:
        print(f" {name:>12}", end="")
    print()
    print("-" * (10 + 13 * len(results)))

    for ticker in tickers:
        print(f"{ticker:<10}", end="")
        for name, weights, _ in results:
            weight = weights.get(ticker, 0)
            print(f" {weight*100:>11.1f}%", end="")
        print()


# ============================================================================
# EXAMPLE 12: Using Fundamental Data for Screening
# ============================================================================

def example_fundamental_screening():
    """Screen stocks using fundamental data before optimization"""
    print("\n" + "="*80)
    print("EXAMPLE 12: Fundamental Screening + Optimization")
    print("="*80)

    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD']

    print("\nScreening stocks by fundamentals...")
    selected_tickers = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Screening criteria
            pe = info.get('trailingPE', 999)
            roe = info.get('returnOnEquity', 0)
            debt_to_equity = info.get('debtToEquity', 999)

            # Filter: P/E < 50, ROE > 15%, Debt/Equity < 100
            if pe < 50 and roe > 0.15 and debt_to_equity < 100:
                selected_tickers.append(ticker)
                print(f"  ✓ {ticker}: PE={pe:.1f}, ROE={roe*100:.1f}%, D/E={debt_to_equity:.1f}")
            else:
                print(f"  ✗ {ticker}: PE={pe:.1f}, ROE={roe*100:.1f}%, D/E={debt_to_equity:.1f}")
        except:
            print(f"  ✗ {ticker}: Data unavailable")

    if len(selected_tickers) < 2:
        print("\nNot enough stocks passed screening")
        return

    print(f"\n{len(selected_tickers)} stocks passed screening")

    # Optimize selected stocks
    data = yf.download(selected_tickers, start='2020-01-01', end='2024-01-01')['Adj Close']

    mu = expected_returns.mean_historical_return(data)
    S = risk_models.CovarianceShrinkage(data).ledoit_wolf()

    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    print("\nOptimal Portfolio (from screened stocks):")
    for ticker, weight in cleaned_weights.items():
        if weight > 0:
            print(f"  {ticker}: {weight*100:.2f}%")

    perf = ef.portfolio_performance(verbose=True)
    print(f"\nExpected Return: {perf[0]*100:.2f}%")
    print(f"Volatility: {perf[1]*100:.2f}%")
    print(f"Sharpe Ratio: {perf[2]:.2f}")


# ============================================================================
# Main - Run Examples
# ============================================================================

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    # Choose which examples to run
    examples = {
        '1': ('Max Sharpe', example_max_sharpe),
        '2': ('Min Volatility', example_min_volatility),
        '3': ('Target Return', example_target_return),
        '4': ('Semivariance', example_semivariance),
        '5': ('CVaR', example_cvar),
        '6': ('HRP', example_hrp),
        '7': ('Constraints', example_constraints),
        '8': ('Transaction Costs', example_transaction_costs),
        '9': ('Black-Litterman', example_black_litterman),
        '10': ('Discrete Allocation', example_discrete_allocation),
        '11': ('Compare Risk Measures', example_compare_risk_measures),
        '12': ('Fundamental Screening', example_fundamental_screening),
    }

    print("\nAvailable Examples:")
    for num, (name, _) in examples.items():
        print(f"  {num}. {name}")
    print("\nRun with: python PORTFOLIO_EXAMPLES.py")
    print("Edit the script to choose which examples to run\n")

    # Run all examples (comment out ones you don't want)
    # example_max_sharpe()
    # example_min_volatility()
    # example_target_return()
    # example_semivariance()
    # example_cvar()
    # example_hrp()
    # example_constraints()
    # example_transaction_costs()
    # example_black_litterman()
    # example_discrete_allocation()
    # example_compare_risk_measures()
    # example_fundamental_screening()

    print("\nTo run examples, uncomment the function calls at the bottom of this file")
