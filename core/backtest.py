"""
Portfolio backtesting framework using bt library
Tests historical performance of allocation strategies
"""

import bt
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table

console = Console()


class PortfolioBacktester:
    """
    Backtest portfolio strategies using bt library

    Features:
    - Fixed-weight allocation backtesting
    - Walk-forward optimization
    - Performance metrics (Sharpe, Sortino, Calmar, max drawdown)
    """

    def __init__(self):
        pass

    def backtest_allocation(
        self,
        tickers: List[str],
        weights: Dict[str, float],
        start_date: str = None,
        end_date: str = None,
        initial_capital: float = 10000,
        rebalance_freq: str = "monthly",
    ) -> Dict:
        """
        Backtest a fixed-weight portfolio allocation

        Args:
            tickers: ['AAPL', 'MSFT', 'GOOGL']
            weights: {'AAPL': 0.40, 'MSFT': 0.35, 'GOOGL': 0.25}
            start_date: '2020-01-01' (default: 2 years ago)
            end_date: '2024-01-01' (default: today)
            initial_capital: Starting portfolio value
            rebalance_freq: 'monthly', 'quarterly', or 'daily'

        Returns:
            {
                'total_return': 0.453,
                'cagr': 0.118,
                'sharpe': 1.23,
                'sortino': 1.45,
                'max_drawdown': -0.18,
                'calmar': 0.66,
                'equity_curve': {...},
                'monthly_returns': {...}
            }
        """
        console.print(f"[blue]Backtesting {len(tickers)} stocks allocation...[/blue]")

        # Default dates
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

        try:
            # Download price data
            console.print(f"[dim]Fetching price data from {start_date} to {end_date}...[/dim]")
            prices = yf.download(tickers, start=start_date, end=end_date, progress=False)[
                "Adj Close"
            ]

            if prices.empty:
                return {"error": "No price data available for these tickers"}

            # Handle single ticker case
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=tickers[0])

            # Create bt strategy
            if rebalance_freq == "monthly":
                rebalance_algo = bt.algos.RunMonthly()
            elif rebalance_freq == "quarterly":
                rebalance_algo = bt.algos.RunQuarterly()
            else:
                rebalance_algo = bt.algos.RunDaily()

            strategy = bt.Strategy(
                "Portfolio",
                [
                    rebalance_algo,
                    bt.algos.SelectAll(),
                    bt.algos.WeighSpecified(**weights),
                    bt.algos.Rebalance(),
                ],
            )

            # Create backtest
            test = bt.Backtest(strategy, prices, initial_capital=initial_capital)
            result = bt.run(test)

            # Extract metrics
            stats = result.stats

            # Get equity curve
            equity_curve = result.prices["Portfolio"].to_dict()
            equity_dates = [d.strftime("%Y-%m-%d") for d in result.prices.index]

            console.print(f"[green]✓ Backtest complete[/green]")
            console.print(f"[cyan]Total Return: {stats.loc['total_return'][0]:.2%}[/cyan]")
            console.print(f"[cyan]Sharpe Ratio: {stats.loc['daily_sharpe'][0]:.2f}[/cyan]")
            console.print(f"[cyan]Max Drawdown: {stats.loc['max_drawdown'][0]:.2%}[/cyan]")

            return {
                "total_return": float(stats.loc["total_return"][0]),
                "cagr": float(stats.loc["cagr"][0]),
                "max_drawdown": float(stats.loc["max_drawdown"][0]),
                "sharpe": float(stats.loc["daily_sharpe"][0]),
                "sortino": float(stats.loc["daily_sortino"][0]),
                "calmar": float(stats.loc["calmar"][0]),
                "volatility": float(stats.loc["daily_vol"][0]),
                "best_day": float(stats.loc["best"][0]),
                "worst_day": float(stats.loc["worst"][0]),
                "equity_curve": {
                    "dates": equity_dates,
                    "values": [float(v) for v in result.prices["Portfolio"].values],
                },
                "start_date": start_date,
                "end_date": end_date,
                "initial_capital": initial_capital,
                "final_value": float(result.prices["Portfolio"].iloc[-1]),
            }

        except Exception as e:
            console.print(f"[red]✗ Backtest failed: {e}[/red]")
            return {"error": str(e)}

    def walk_forward_optimization(
        self,
        tickers: List[str],
        train_months: int = 12,
        test_months: int = 3,
        start_date: str = None,
        optimization_method: str = "max_sharpe",
    ) -> List[Dict]:
        """
        Walk-forward optimization

        Process:
        1. Train on N months, optimize weights
        2. Test on next M months (out-of-sample)
        3. Roll forward, repeat

        Args:
            tickers: List of stock symbols
            train_months: Training period (default: 12 months)
            test_months: Testing period (default: 3 months)
            start_date: Start date (default: 3 years ago)
            optimization_method: 'max_sharpe', 'min_vol', 'hrp'

        Returns:
            [
                {
                    'train_period': '2020-01 to 2021-01',
                    'test_period': '2021-01 to 2021-04',
                    'weights': {...},
                    'test_return': 0.15,
                    'test_sharpe': 1.2
                },
                ...
            ]
        """
        from core.optimizer import optimize as portfolio_optimize

        console.print("[blue]Starting walk-forward optimization...[/blue]")

        # Default start date
        if not start_date:
            start_date = (datetime.now() - timedelta(days=1095)).strftime("%Y-%m-%d")

        # Download all data
        prices = yf.download(tickers, start=start_date, progress=False)["Adj Close"]

        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])

        results = []
        train_days = train_months * 30
        test_days = test_months * 30

        start_idx = 0

        while start_idx + train_days + test_days < len(prices):
            # Training period
            train_end_idx = start_idx + train_days
            train_data = prices.iloc[start_idx:train_end_idx]

            console.print(
                f"[dim]Train: {train_data.index[0].date()} to {train_data.index[-1].date()}[/dim]"
            )

            try:
                # Optimize on training data
                optimized = portfolio_optimize(
                    train_data, budget=10000, method=optimization_method
                )
                weights = optimized["weights"]

                # Test period
                test_start_idx = train_end_idx
                test_end_idx = test_start_idx + test_days
                test_data = prices.iloc[test_start_idx:test_end_idx]

                console.print(
                    f"[dim]Test: {test_data.index[0].date()} to {test_data.index[-1].date()}[/dim]"
                )

                # Calculate test performance
                returns = test_data.pct_change().dropna()

                # Portfolio returns (weighted average)
                portfolio_returns = pd.Series(0.0, index=returns.index)
                for ticker, weight in weights.items():
                    if ticker in returns.columns:
                        portfolio_returns += returns[ticker] * weight

                # Performance metrics
                test_return = (1 + portfolio_returns).prod() - 1
                test_sharpe = (
                    portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
                )
                test_volatility = portfolio_returns.std() * np.sqrt(252)

                results.append(
                    {
                        "train_period": f"{train_data.index[0].date()} to {train_data.index[-1].date()}",
                        "test_period": f"{test_data.index[0].date()} to {test_data.index[-1].date()}",
                        "weights": weights,
                        "test_return": float(test_return),
                        "test_sharpe": float(test_sharpe),
                        "test_volatility": float(test_volatility),
                    }
                )

                console.print(
                    f"[green]✓ Period {len(results)}: Return={test_return:.2%}, Sharpe={test_sharpe:.2f}[/green]"
                )

            except Exception as e:
                console.print(f"[yellow]⚠ Period optimization failed: {e}[/yellow]")

            # Roll forward
            start_idx += test_days

        # Summary statistics
        if results:
            avg_return = np.mean([r["test_return"] for r in results])
            avg_sharpe = np.mean([r["test_sharpe"] for r in results])
            console.print(f"\n[bold green]Walk-Forward Summary:[/bold green]")
            console.print(f"[cyan]Periods: {len(results)}[/cyan]")
            console.print(f"[cyan]Avg Return: {avg_return:.2%}[/cyan]")
            console.print(f"[cyan]Avg Sharpe: {avg_sharpe:.2f}[/cyan]")

        return results

    def compare_methods(
        self,
        tickers: List[str],
        methods: List[str] = None,
        start_date: str = None,
        end_date: str = None,
    ) -> Dict:
        """
        Compare multiple optimization methods side-by-side

        Args:
            tickers: List of stock symbols
            methods: ['max_sharpe', 'min_vol', 'hrp'] (default: all)
            start_date: Start date for comparison
            end_date: End date for comparison

        Returns:
            {
                'max_sharpe': {...backtest results...},
                'min_vol': {...backtest results...},
                'hrp': {...backtest results...}
            }
        """
        from core.optimizer import optimize as portfolio_optimize

        if not methods:
            methods = ["max_sharpe", "min_vol", "hrp", "semivariance"]

        console.print(f"[blue]Comparing {len(methods)} optimization methods...[/blue]")

        # Default dates
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

        # Download price data for optimization
        prices = yf.download(tickers, start=start_date, end=end_date, progress=False)[
            "Adj Close"
        ]

        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])

        results = {}

        for method in methods:
            try:
                console.print(f"\n[cyan]Testing method: {method}[/cyan]")

                # Optimize portfolio
                optimized = portfolio_optimize(prices, budget=10000, method=method)
                weights = optimized["weights"]

                # Backtest the allocation
                backtest_result = self.backtest_allocation(
                    list(weights.keys()),
                    weights,
                    start_date=start_date,
                    end_date=end_date,
                )

                results[method] = {
                    "weights": weights,
                    "backtest": backtest_result,
                }

                console.print(
                    f"[green]✓ {method}: Return={backtest_result.get('total_return', 0):.2%}, "
                    f"Sharpe={backtest_result.get('sharpe', 0):.2f}[/green]"
                )

            except Exception as e:
                console.print(f"[red]✗ {method} failed: {e}[/red]")
                results[method] = {"error": str(e)}

        # Display comparison table
        self._display_comparison_table(results)

        return results

    def _display_comparison_table(self, results: Dict):
        """Display comparison table of methods"""
        table = Table(title="Method Comparison", show_header=True, header_style="bold magenta")

        table.add_column("Method", style="cyan")
        table.add_column("Return", style="green")
        table.add_column("Sharpe", style="blue")
        table.add_column("Max DD", style="red")
        table.add_column("Volatility", style="yellow")

        for method, data in results.items():
            if "error" in data:
                table.add_row(method, "ERROR", "-", "-", "-")
            else:
                backtest = data.get("backtest", {})
                table.add_row(
                    method,
                    f"{backtest.get('total_return', 0):.2%}",
                    f"{backtest.get('sharpe', 0):.2f}",
                    f"{backtest.get('max_drawdown', 0):.2%}",
                    f"{backtest.get('volatility', 0):.2%}",
                )

        console.print(table)
