"""
Flask application that serves as the API
"""

#!/usr/bin/env python
import logging
import json
import os
import argparse
import matplotlib

matplotlib.use("Agg")

from flask import Flask, render_template, request, jsonify, send_from_directory
from core.analyzer import run as analyze
from core.optimizer import run as optimize
from core.watchlist import WatchlistManager, get_custom_watchlist
from core.sell_analyzer import SellAnalyzer
from core.portfolio_manager import PortfolioManager
from core.backtest import PortfolioBacktester
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

app = Flask(__name__, static_url_path="", static_folder="static")


@app.route("/")
def template():
    """
    Returns base template
    """
    return render_template("index.html")


@app.route("/images/<filename>")
def serve_image(filename):
    """
    Serve generated images
    """
    return send_from_directory("data/outputs", filename)


@app.route("/trigger_otp/", methods=["POST"])
def trigger_otp():
    """
    Trigger OTP request from Wealthsimple
    """
    try:
        import core.wealthsimple as ws
        import threading

        # Clear any existing OTP
        ws.current_otp = None
        ws.otp_requested = False

        # Start OTP request in background thread
        def start_otp_request():
            try:
                ws.trigger_otp_request()
            except Exception as e:
                console.print(f"[red]OTP request failed: {e}[/red]")

        thread = threading.Thread(target=start_otp_request)
        thread.daemon = True
        thread.start()

        return jsonify(
            {
                "success": True,
                "message": "OTP request sent! Check your authenticator app.",
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/recommend_stocks/", methods=["POST"])
def recommend_stocks():
    """
    Recommends stocks using custom watchlist (no OTP required)
    """
    try:
        # Get parameters from request
        data = request.get_json() or {}
        threshold = data.get("threshold", 0.05)
        budget = data.get("budget", 1000)
        method = data.get("method", "max_sharpe")  # Default to max_sharpe

        console.print(
            f"[blue]Portfolio parameters: {threshold} threshold, ${budget} budget, {method} method[/blue]"
        )

        # Use custom watchlist instead of Wealthsimple
        pre_selected_stocks = analyze(send_telegram=False)
        console.print(
            Panel(
                f"[bold cyan]Pre-selected stocks:[/bold cyan] {', '.join(pre_selected_stocks)}",
                title="Analysis Results",
                border_style="cyan",
            )
        )
        optimized_stocks = optimize(
            pre_selected_stocks, budget=budget, cutoff=threshold, send_telegram=False, method=method
        )
        console.print(
            Panel(
                f"[bold green]Portfolio optimization completed![/bold green]",
                title="Success",
                border_style="green",
            )
        )

        # Prepare response with portfolio data and image paths
        import os
        import glob

        # Get all generated images
        image_dir = "data/outputs"
        images = {}

        # Portfolio optimization charts
        if os.path.exists(f"{image_dir}/pf_optimizer.png"):
            images["efficient_frontier"] = f"/images/pf_optimizer.png"
        if os.path.exists(f"{image_dir}/pf_cov_clusters.png"):
            images["covariance_clusters"] = f"/images/pf_cov_clusters.png"

        # Individual stock charts
        stock_charts = []
        for stock in pre_selected_stocks:
            chart_path = f"{image_dir}/ohlc_{stock}.png"
            if os.path.exists(chart_path):
                stock_charts.append(
                    {"symbol": stock, "chart": f"/images/ohlc_{stock}.png"}
                )

        # Get industry/sector information for portfolio stocks
        stock_info = {}
        if optimized_stocks and optimized_stocks.get("weights"):
            import yfinance as yf

            for symbol in optimized_stocks["weights"].keys():
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    stock_info[symbol] = {
                        "sector": info.get("sector", "N/A"),
                        "industry": info.get("industry", "N/A"),
                        "shortName": info.get("shortName", symbol),
                    }
                    console.print(
                        f"[blue]Got info for {symbol}: {stock_info[symbol]['sector']} - {stock_info[symbol]['industry']}[/blue]"
                    )
                except Exception as e:
                    console.print(
                        f"[yellow]Could not get info for {symbol}: {e}[/yellow]"
                    )
                    stock_info[symbol] = {
                        "sector": "N/A",
                        "industry": "N/A",
                        "shortName": symbol,
                    }

        response = {
            "portfolio": optimized_stocks,
            "selected_stocks": pre_selected_stocks,
            "images": images,
            "stock_charts": stock_charts,
            "stock_info": stock_info,
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/watchlist/", methods=["GET"])
def get_watchlist():
    """Get current watchlist symbols"""
    try:
        manager = WatchlistManager()
        symbols = manager.get_symbols("Default")
        return jsonify({"symbols": symbols})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/watchlist/add", methods=["POST"])
def add_to_watchlist():
    """Add a symbol to the watchlist"""
    try:
        symbol = request.json.get("symbol", "").strip().upper()
        if not symbol:
            return jsonify({"error": "Symbol is required"}), 400

        manager = WatchlistManager()
        success = manager.add_symbol("Default", symbol)

        if success:
            return jsonify({"message": f"Added {symbol} to watchlist"})
        else:
            return jsonify({"error": f"Failed to add {symbol} or already exists"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/watchlist/remove", methods=["POST"])
def remove_from_watchlist():
    """Remove a symbol from the watchlist"""
    try:
        symbol = request.json.get("symbol", "").strip().upper()
        if not symbol:
            return jsonify({"error": "Symbol is required"}), 400

        manager = WatchlistManager()
        success = manager.remove_symbol("Default", symbol)

        if success:
            return jsonify({"message": f"Removed {symbol} from watchlist"})
        else:
            return jsonify({"error": f"Symbol {symbol} not found in watchlist"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/watchlist/clear", methods=["POST"])
def clear_watchlist():
    """Clear all symbols from the watchlist"""
    try:
        manager = WatchlistManager()
        success = manager.clear_watchlist("Default")

        if success:
            return jsonify({"message": "Watchlist cleared"})
        else:
            return jsonify({"error": "Failed to clear watchlist"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/watchlist/import", methods=["POST"])
def import_watchlist():
    """Import symbols from CSV content"""
    try:
        csv_content = request.json.get("csv_content", "")
        if not csv_content:
            return jsonify({"error": "CSV content is required"}), 400

        console.print(f"[cyan]Received CSV import request[/cyan]")
        console.print(f"[cyan]CSV content length: {len(csv_content)}[/cyan]")

        manager = WatchlistManager()
        added_count, skipped_count = manager.import_from_csv("Default", csv_content)

        if added_count == 0 and skipped_count == 0:
            return (
                jsonify({"error": "No symbols could be imported. Check CSV format."}),
                400,
            )

        return jsonify(
            {
                "message": f"Import complete: {added_count} added, {skipped_count} skipped",
                "added": added_count,
                "skipped": skipped_count,
            }
        )
    except Exception as e:
        console.print(f"[red]Import error: {e}[/red]")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/watchlist/export", methods=["GET"])
def export_watchlist():
    """Export watchlist as CSV"""
    try:
        manager = WatchlistManager()
        csv_content = manager.export_to_csv("Default")

        from flask import Response

        return Response(
            csv_content,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=watchlist.csv"},
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_company_names", methods=["POST"])
def get_company_names():
    """Get company names for given symbols"""
    try:
        symbols = request.json.get("symbols", [])
        if not symbols:
            return jsonify({"error": "No symbols provided"}), 400

        import yfinance as yf

        company_names = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                company_names[symbol] = info.get(
                    "shortName", info.get("longName", symbol)
                )
            except Exception as e:
                console.print(f"[yellow]Could not get name for {symbol}: {e}[/yellow]")
                company_names[symbol] = symbol

        return jsonify({"company_names": company_names})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/check_sells/", methods=["POST"])
def check_sell_signals():
    """
    Check sell signals for portfolio holdings

    POST /check_sells/
    {
        "holdings": [
            {"symbol": "AAPL", "entry_price": 150, "entry_date": "2024-01-01", "shares": 10},
            {"symbol": "MSFT", "entry_price": 300, "shares": 5}
        ]
    }

    Returns:
        {
            "sell_recommendations": [
                {
                    "symbol": "AAPL",
                    "sell_signal": true,
                    "reasons": ["stop_loss"],
                    "current_price": 138.50,
                    "gain_loss": -0.077,
                    "recommendation": "SELL",
                    "priority": "HIGH"
                },
                ...
            ]
        }
    """
    try:
        data = request.get_json()
        holdings = data.get("holdings", [])

        if not holdings:
            return jsonify({"error": "No holdings provided"}), 400

        analyzer = SellAnalyzer()
        results = analyzer.batch_analyze(holdings)

        console.print(f"[green]Analyzed {len(results)} positions[/green]")
        sell_count = sum(1 for r in results if r.get("recommendation") == "SELL")
        if sell_count > 0:
            console.print(f"[red]Found {sell_count} SELL recommendations[/red]")

        return jsonify({"sell_recommendations": results})

    except Exception as e:
        console.print(f"[red]Error in check_sells: {e}[/red]")
        return jsonify({"error": str(e)}), 500


@app.route("/check_portfolio/", methods=["POST"])
def check_portfolio_risk():
    """
    Check portfolio-level risk metrics

    POST /check_portfolio/
    {
        "current_portfolio": {"AAPL": 0.35, "MSFT": 0.25, "GOOGL": 0.20, "TSLA": 0.20},
        "target_portfolio": {"AAPL": 0.30, "MSFT": 0.25, "GOOGL": 0.25, "TSLA": 0.20}
    }

    Returns:
        {
            "position_violations": [...],
            "concentration": {...},
            "rebalancing": {...},
            "overall_recommendation": "REBALANCE" | "TRIM" | "OK"
        }
    """
    try:
        data = request.get_json()
        current = data.get("current_portfolio", {})
        target = data.get("target_portfolio")

        if not current:
            return jsonify({"error": "current_portfolio required"}), 400

        manager = PortfolioManager()
        results = manager.comprehensive_check(current, target)

        console.print(f"[blue]Portfolio check: {results['overall_recommendation']}[/blue]")

        return jsonify(results)

    except Exception as e:
        console.print(f"[red]Error in check_portfolio: {e}[/red]")
        return jsonify({"error": str(e)}), 500


@app.route("/backtest/", methods=["POST"])
def run_backtest():
    """
    Backtest a portfolio allocation

    POST /backtest/
    {
        "tickers": ["AAPL", "MSFT", "GOOGL"],
        "weights": {"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25},
        "start_date": "2020-01-01",
        "end_date": "2024-01-01",
        "initial_capital": 10000,
        "rebalance_freq": "monthly"
    }

    Returns:
        {
            "total_return": 0.453,
            "cagr": 0.118,
            "sharpe": 1.23,
            "max_drawdown": -0.18,
            "equity_curve": {...}
        }
    """
    try:
        data = request.get_json()

        tickers = data.get("tickers", [])
        weights = data.get("weights", {})

        if not tickers or not weights:
            return jsonify({"error": "tickers and weights required"}), 400

        backtester = PortfolioBacktester()
        results = backtester.backtest_allocation(
            tickers,
            weights,
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            initial_capital=data.get("initial_capital", 10000),
            rebalance_freq=data.get("rebalance_freq", "monthly"),
        )

        if "error" in results:
            return jsonify(results), 400

        console.print(f"[green]Backtest complete: {results['total_return']:.2%} return[/green]")

        return jsonify(results)

    except Exception as e:
        console.print(f"[red]Error in backtest: {e}[/red]")
        return jsonify({"error": str(e)}), 500


@app.route("/compare_methods/", methods=["POST"])
def compare_optimization_methods():
    """
    Compare multiple optimization methods

    POST /compare_methods/
    {
        "tickers": ["AAPL", "MSFT", "GOOGL", "TSLA"],
        "methods": ["max_sharpe", "min_vol", "hrp"],
        "start_date": "2020-01-01",
        "end_date": "2024-01-01"
    }

    Returns:
        {
            "max_sharpe": {...},
            "min_vol": {...},
            "hrp": {...}
        }
    """
    try:
        data = request.get_json()

        tickers = data.get("tickers", [])
        methods = data.get("methods", ["max_sharpe", "min_vol", "hrp"])

        if not tickers:
            return jsonify({"error": "tickers required"}), 400

        backtester = PortfolioBacktester()
        results = backtester.compare_methods(
            tickers,
            methods=methods,
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
        )

        console.print(f"[green]Method comparison complete for {len(results)} methods[/green]")

        return jsonify(results)

    except Exception as e:
        console.print(f"[red]Error in compare_methods: {e}[/red]")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(500)
def server_error(error):
    """
    Log the error and stacktrace.
    """
    logging.exception("An error occurred during a request: %s", error)
    return "An internal error occurred.", 500


if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--port", type=int, default=5004, help="Port to run the server on"
        )
        args = parser.parse_args()
        port = args.port
        console.print(
            Panel(
                f"[bold green]Stock Screener Flask App Starting![/bold green]\n[blue]Server running on: http://0.0.0.0:{port}[/blue]\n[yellow]Ready to analyze portfolios![/yellow]",
                title="Stock Screener",
                border_style="green",
            )
        )
        app.run(host="0.0.0.0", port=port, debug=True)
