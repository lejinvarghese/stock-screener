#!/usr/bin/env python

"""
Run technical and fundamental analysis
"""

import os
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
from warnings import filterwarnings

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from ffn.core import GroupStats
import yfinance as yf

# Set matplotlib backend before importing pyplot to avoid GUI issues in threads
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend for thread safety
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance import original_flavor as of
from tradingview_ta import Interval, get_multiple_analysis
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

try:
    from core.utils import get_data, send_image, send_message, numeric_round
    from core.watchlist import get_custom_watchlist
except:
    from utils import get_data, send_image, send_message, numeric_round
    from watchlist import get_custom_watchlist

filterwarnings("ignore")
load_dotenv()

PATH = os.getcwd()
N_PROCESS = cpu_count() - 2

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_ID = os.getenv("TELEGRAM_ID")
TELEGRAM_BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# windows
W_MA_STRONG_SHORT = 50
W_MA_STRONG_LONG = 200
W_MA_EARLY_SHORT = 15
W_MA_EARLY_LONG = 50

# plotting parameters
plt.style.use("ggplot")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 12
plt.rcParams["savefig.dpi"] = 240
plt.rcParams["savefig.format"] = "png"
# plt.rcParams["savefig.jpeg_quality"] = 100  # Not available in newer matplotlib


def get_trading_view_buy_ratings(tickers):
    console.print(
        Panel(
            f"[bold cyan]Getting TradingView ratings for tickers:[/bold cyan] {', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''}",
            title="TradingView Analysis",
            border_style="cyan",
        )
    )

    if not tickers or len(tickers) == 0:
        console.print(
            "[yellow]WARNING: No tickers provided for TradingView analysis[/yellow]"
        )
        return []
    tickers = [ticker.split("-")[0] for ticker in tickers]
    exch_tickers = (
        [f"nyse:{ticker}" for ticker in tickers]
        + [f"nasdaq:{ticker}" for ticker in tickers]
        + [f"tsx:{ticker}" for ticker in tickers]
    )

    console.print(f"[dim]Exchange tickers: {exch_tickers[:5]}...[/dim]")

    try:
        tv_analysis = get_multiple_analysis(
            screener="america", interval=Interval.INTERVAL_1_DAY, symbols=exch_tickers
        )
        console.print(
            f"[green]TradingView analysis completed for {len(tv_analysis)} tickers[/green]"
        )
    except Exception as e:
        console.print(f"[red]ERROR: TradingView analysis failed: {e}[/red]")
        # Return a subset of the original tickers as fallback
        console.print("[yellow]Using fallback tickers...[/yellow]")
        return tickers[:5]  # Return first 5 tickers as fallback

    selected_stocks = []
    console.print("[blue]Processing TradingView recommendations:[/blue]")

    for ticker in tv_analysis:
        ticker_results = tv_analysis.get(str.upper(ticker))
        try:
            ticker_reco = ticker_results.summary.get("RECOMMENDATION")
            oscillators = ticker_results.oscillators
            osc_buy_ratio = oscillators["BUY"] / (
                oscillators["BUY"] + oscillators["SELL"]
            )
        except:
            ticker_reco = "NA"
            osc_buy_ratio = 0

        clean_ticker = ticker.split(":")[-1]

        if "BUY" in ticker_reco and osc_buy_ratio > 0.5:
            console.print(
                f"[green]✓ {clean_ticker}: {ticker_reco} (Osc: {osc_buy_ratio:.2%}) - SELECTED[/green]"
            )
            selected_stocks.append(clean_ticker)
        else:
            console.print(
                f"[dim]✗ {clean_ticker}: {ticker_reco} (Osc: {osc_buy_ratio:.2%}) - SKIPPED[/dim]"
            )

    if selected_stocks:
        table = Table(
            title="Stocks with BUY Rating", show_header=True, header_style="bold green"
        )
        table.add_column("Stock Symbol", style="cyan")
        for stock in selected_stocks:
            table.add_row(stock)
        console.print(table)
    else:
        console.print("[yellow]WARNING: No stocks with BUY rating found[/yellow]")

    # If no BUY recommendations found, return a sample of original tickers
    if not selected_stocks:
        console.print(
            "[yellow]No BUY recommendations found, returning sample of original tickers[/yellow]"
        )
        selected_stocks = tickers[:3]  # Return first 3 as fallback

    return selected_stocks


def get_metrics(data):
    """
    Get financial metrics for the ticker
    """
    ticker = data.ticker.unique()[0]
    try:
        info = yf.Ticker(ticker).info
        info = dict(filter(lambda item: item[1] is not None, info.items()))
    except:
        info = dict()
    t_stats = GroupStats(data["Close"]).stats.to_dict(orient="dict").get("Close")

    metrics = {}
    metrics["beta"] = beta = numeric_round(info.get("beta", "N/A"), 2)
    metrics["peg"] = peg = numeric_round(info.get("pegRatio", "N/A"), 2)
    metrics["ptb"] = ptb = numeric_round(info.get("priceToBook", "N/A"), 2)
    metrics["dividend_pt"] = dividend_pt = numeric_round(
        info.get("dividendRate", "N/A"), 2
    )
    metrics["payout"] = payout = numeric_round(info.get("payoutRatio", "N/A"), 2)
    metrics["calmar"] = calmar = numeric_round(t_stats.get("calmar", "N/A"), 2)
    metrics["cagr"] = cagr = numeric_round(t_stats.get("cagr", "N/A"), 2)
    metrics["monthly_sharpe"] = monthly_sharpe = numeric_round(
        t_stats.get("monthly_sharpe", "N/A"), 2
    )
    metrics["monthly_sortino"] = monthly_sortino = numeric_round(
        t_stats.get("monthly_sortino", "N/A"), 2
    )

    metrics_summary = f"""
    \u03b2: {beta},\n
    PEG Ratio: {peg},\n
    P/B Ratio: {ptb},\n
    Dividend rate: {dividend_pt},\n
    Payout Ratio: {payout}, \n
    Calmar Ratio: {calmar}, \n
    CAGR: {cagr}, \n
    Monthly Sharpe: {monthly_sharpe}, \n
    Monthly Sortino: {monthly_sortino}
    """
    return ticker, metrics_summary, metrics


def signals_ma(data):
    """
    Heikin-Ashi advanced ohlc indicators
    """
    
    # Check if we have enough data for analysis
    if len(data) < max(W_MA_STRONG_LONG, W_MA_STRONG_SHORT, W_MA_EARLY_LONG, W_MA_EARLY_SHORT):
        console.print(f"[yellow]WARNING: Insufficient data ({len(data)} rows) for {data.ticker.iloc[0] if 'ticker' in data.columns else 'unknown'}, skipping technical analysis[/yellow]")
        # Return minimal data structure with all required columns
        signals = pd.DataFrame(index=data.index)
        
        # Signal columns
        signals["signal_strong_rising"] = 0.0
        signals["signal_early_rising"] = 0.0
        signals["signal_early_warning"] = 0.0
        signals["signal_strong_warning"] = 0.0
        signals["positions_early_rising"] = 0.0
        signals["positions_early_warning"] = 0.0
        signals["positions_warning"] = 0.0
        signals["positions_strong"] = 0.0
        
        # Moving average columns (use Close price as fallback)
        signals["ma_early_short"] = data["Close"]
        signals["ma_strong_short"] = data["Close"]
        signals["ma_strong_long"] = data["Close"]
        signals["ma_early_long_high"] = data["High"] if "High" in data.columns else data["Close"]
        signals["ma_early_long_low"] = data["Low"] if "Low" in data.columns else data["Close"]
        
        # Volume
        signals["volume"] = data["Volume"] if "Volume" in data.columns else 0.0
        signals = signals.merge(data[["Close", "Volume"]], left_index=True, right_index=True, how='left')
        
        # Create simple OHLC data
        data.index = pd.to_datetime(data.index)
        data_values = data[["Open", "High", "Low", "Close"]].values.tolist()
        plot_dates = mdates.date2num(data.index)
        ohlc = [[plot_dates[i]] + data_values[i] for i in range(len(plot_dates))]
        
        return ohlc, signals

    # regular indicators are replaced with the advanced heikin-ashi indicators
    ha_close = 0.25 * (2 * data.Close + data.Open + data.Low)
    ha_high = data[["High", "Open", "Close"]].max(axis=1)
    ha_low = data[["Low", "Open", "Close"]].min(axis=1)
    ha_open = 0.5 * (data.Close.shift(1) + data.Open.shift(1))
    data["Close"] = ha_close
    data["High"] = ha_high
    data["Low"] = ha_low
    data["Open"] = ha_open
    del ha_close, ha_high, ha_low, ha_open

    data.index = pd.to_datetime(data.index)
    data_values = data[["Open", "High", "Low", "Close"]].values.tolist()
    plot_dates = mdates.date2num(data.index)
    ohlc = [[plot_dates[i]] + data_values[i] for i in range(len(plot_dates))]
    del data_values, plot_dates

    # Initialize the `signals` DataFrame with the `signal` column
    signals = pd.DataFrame(index=data.index)
    (
        signals["signal_strong_rising"],
        signals["signal_early_rising"],
        signals["signal_early_warning"],
        signals["signal_strong_warning"],
    ) = (0.0, 0.0, 0.0, 0.0)

    # Create moving average over  the windows
    signals["volume"] = data["Volume"]
    signals["ma_strong_short"] = (
        data["Close"].ewm(W_MA_STRONG_SHORT, adjust=False).mean()
    )
    signals["ma_strong_long"] = data["Close"].ewm(W_MA_STRONG_LONG, adjust=False).mean()

    signals["ma_early_short"] = data["Close"].ewm(W_MA_EARLY_SHORT, adjust=False).mean()
    signals["ma_early_long_high"] = (
        data["High"].rolling(window=W_MA_EARLY_LONG, min_periods=1, center=False).mean()
    )
    signals["ma_early_long_low"] = (
        data["Low"].rolling(window=W_MA_EARLY_LONG, min_periods=1, center=False).mean()
    )

    # Create signals
    signals.iloc[W_MA_EARLY_SHORT:, signals.columns.get_loc("signal_early_rising")] = np.where(
        signals.iloc[W_MA_EARLY_SHORT:, signals.columns.get_loc("ma_early_short")]
        > signals.iloc[W_MA_EARLY_SHORT:, signals.columns.get_loc("ma_early_long_high")],
        1.0,
        0.0,
    )
    signals["positions_early_rising"] = signals["signal_early_rising"].diff()

    signals.iloc[W_MA_EARLY_SHORT:, signals.columns.get_loc("signal_early_warning")] = np.where(
        signals.iloc[W_MA_EARLY_SHORT:, signals.columns.get_loc("ma_early_short")]
        < signals.iloc[W_MA_EARLY_SHORT:, signals.columns.get_loc("ma_early_long_low")],
        1.0,
        0.0,
    )
    signals["positions_early_warning"] = signals["signal_early_warning"].diff()

    signals.iloc[W_MA_EARLY_SHORT:, signals.columns.get_loc("signal_strong_warning")] = np.where(
        signals.iloc[W_MA_EARLY_SHORT:, signals.columns.get_loc("ma_early_short")]
        < signals.iloc[W_MA_EARLY_SHORT:, signals.columns.get_loc("ma_strong_short")],
        1.0,
        0.0,
    )
    signals["positions_warning"] = signals["signal_strong_warning"].diff()

    signals.iloc[W_MA_STRONG_SHORT:, signals.columns.get_loc("signal_strong_rising")] = np.where(
        signals.iloc[W_MA_STRONG_SHORT:, signals.columns.get_loc("ma_strong_short")]
        > signals.iloc[W_MA_STRONG_SHORT:, signals.columns.get_loc("ma_strong_long")],
        1.0,
        0.0,
    )
    signals["positions_strong"] = signals["signal_strong_rising"].diff()
    signals = signals.merge(
        data[["Close", "Volume"]], left_index=True, right_index=True
    )
    return ohlc, signals


def create_plot(signals, ohlc, metrics_summary, ticker):
    """
    Create the visualizations
    """
    fig, (ax_0, ax_1) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(24, 16),
        gridspec_kw={"height_ratios": [8, 1], "hspace": 0.02},
        dpi=120,
    )

    # trends
    ax_0.text(0.01, 0.8, metrics_summary, va="center", transform=ax_0.transAxes)
    of.candlestick_ohlc(
        ax_0, ohlc, colorup="#77d879", colordown="#db3f3f", width=1, alpha=0.8
    )
    ax_0.plot(
        signals["ma_early_short"],
        color="lightgreen",
        linewidth=2,
        label=f"Close, {W_MA_EARLY_SHORT}-Day EMA",
    )
    ax_0.plot(
        signals["ma_early_long_high"],
        color="palegreen",
        linestyle=":",
        linewidth=1,
        label=f"High, {W_MA_EARLY_LONG}-Day SMA",
    )
    ax_0.plot(
        signals["ma_early_long_low"],
        color="salmon",
        linestyle=":",
        linewidth=1,
        label=f"Low, {W_MA_EARLY_LONG}-Day SMA",
    )

    ax_0.plot(
        signals["ma_strong_short"],
        color="seagreen",
        linewidth=2,
        label=f"Close, {W_MA_STRONG_SHORT}-Day EMA",
    )
    ax_0.plot(
        signals["ma_strong_long"],
        color="red",
        linestyle=":",
        linewidth=1,
        label=f"Close, {W_MA_STRONG_LONG}-Day EMA",
    )

    # ticks
    ax_0.plot(
        signals.loc[signals.positions_early_rising == 1.0].index,
        signals.ma_early_short[signals.positions_early_rising == 1.0],
        "^",
        markersize=10,
        color="springgreen",
        label="early rising",
    )
    ax_0.plot(
        signals.loc[signals.positions_early_warning == 1.0].index,
        signals.ma_early_short[signals.positions_early_warning == 1.0],
        "v",
        markersize=10,
        color="gold",
        label="early warning",
    )
    ax_0.plot(
        signals.loc[signals.positions_warning == 1.0].index,
        signals.ma_early_short[signals.positions_warning == 1.0],
        "v",
        markersize=10,
        color="darkorange",
        label="strong warning",
    )
    ax_0.plot(
        signals.loc[signals.positions_strong == 1.0].index,
        signals.ma_strong_short[signals.positions_strong == 1.0],
        "^",
        markersize=10,
        color="forestgreen",
        label="strong rising",
    )
    ax_0.plot(
        signals.loc[signals.positions_strong == -1.0].index,
        signals.ma_strong_short[signals.positions_strong == -1.0],
        "v",
        markersize=10,
        color="red",
        label="strong decline",
    )

    ax_0.set_ylabel("Price ($)")
    ax_0.set_title(f"Trend for stock: {str.upper(ticker)}")
    ax_0.legend(loc="upper right", bbox_to_anchor=(1.14, 1.0))

    # secondary graph: volume
    ax_1.bar(
        x=signals.index,
        height=signals["volume"],
        color="crimson",
        align="center",
        alpha=0.8,
        label="Volume",
    )
    ax_1.axhline(signals["volume"].tail(180).median(), linestyle=":", linewidth=1)
    ax_1.set_xlabel("Date")
    ax_1.set_ylabel("Volume")
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    fig.autofmt_xdate()
    # Create output directory if it doesn't exist
    import os

    os.makedirs(f"{PATH}/data/outputs", exist_ok=True)
    plt.savefig(f"{PATH}/data/outputs/ohlc_{ticker}.png")


def analyze_ticker(data, send_telegram=True):
    """
    Analyzes all tickers
    """
    ticker, metrics_summary, _ = get_metrics(data)
    ohlc, signals = signals_ma(data)

    # curr_signal_strong_rising = signals.iloc[[-1]]["signal_strong_rising"].values[0]
    # curr_signal_warning = max(
    #     int(signals.iloc[[-1]]["signal_strong_warning"].values[0]), 0
    # )

    # if (
    #     ((curr_signal_strong_rising == 1))
    #     & (curr_signal_warning < 1)
    #     & (metrics.get("cagr", 0.0) >= 0.10)
    # ):
    create_plot(signals, ohlc, metrics_summary, ticker)
    if send_telegram:
        send_image(
            TELEGRAM_TOKEN, TELEGRAM_ID, f"{PATH}/data/outputs/ohlc_{ticker}.png"
        )
    plt.close("all")
    return ticker


def run(n_tickers=100, mode="wealthsimple", send_telegram=True):
    """
    Runs the analysis for all the provided tickers
    """

    if mode == "wealthsimple":
        watchlist = get_custom_watchlist()[:n_tickers]
        if watchlist:
            table = Table(
                title="Custom Watchlist", show_header=True, header_style="bold blue"
            )
            table.add_column("Stock Symbol", style="cyan")
            for stock in watchlist[:10]:  # Show first 10
                table.add_row(stock)
            if len(watchlist) > 10:
                table.add_row(f"... and {len(watchlist) - 10} more")
            console.print(table)
        console.print(
            f"[blue]Watchlist contains {len(watchlist) if watchlist else 0} stocks[/blue]"
        )

        # Fallback to sample stocks if watchlist is empty
        if not watchlist or len(watchlist) == 0:
            console.print(
                "[yellow]WARNING: Watchlist is empty, using sample stocks[/yellow]"
            )
            watchlist = [
                "AAPL",
                "GOOGL",
                "MSFT",
                "TSLA",
                "AMZN",
                "NVDA",
                "META",
                "NFLX",
            ][:n_tickers]

    elif mode == "local":
        watchlist = list(
            pd.read_csv(f"{PATH}/data/inputs/my_watchlist.csv").Symbol.unique()
        )[:n_tickers]

    console.print(
        Panel(
            f"[bold green]Final watchlist for analysis:[/bold green] {', '.join(watchlist)}",
            title="Analysis Target",
            border_style="green",
        )
    )

    if not watchlist or len(watchlist) == 0:
        console.print("[red]ERROR: No stocks available for analysis[/red]")
        raise ValueError("No stocks available for analysis")

    if send_telegram:
        send_message(TELEGRAM_TOKEN, TELEGRAM_ID, "Retrieving Trading View Ratings: ")

    selected_stocks = get_trading_view_buy_ratings(watchlist)

    if send_telegram:
        send_message(
            TELEGRAM_TOKEN,
            TELEGRAM_ID,
            f"You have {len(selected_stocks)} stocks with an active BUY rating: "
            + "; ".join(selected_stocks),
        )

    if send_telegram:
        send_message(TELEGRAM_TOKEN, TELEGRAM_ID, "Retrieving Performance History: ")

    with ThreadPool(N_PROCESS) as t_pool:
        data = t_pool.map_async(get_data, selected_stocks).get()
    data = list(filter(None.__ne__, data))
    if send_telegram:
        send_message(
            TELEGRAM_TOKEN, TELEGRAM_ID, "Obtaining Additional Technical Metrics: "
        )

    # Create a partial function to pass send_telegram to analyze_ticker
    from functools import partial

    analyze_ticker_with_telegram = partial(analyze_ticker, send_telegram=send_telegram)

    with Pool(N_PROCESS) as pool:
        _ = pool.map_async(analyze_ticker_with_telegram, data).get()

    return selected_stocks


if __name__ == "__main__":
    run(n_tickers=10)
