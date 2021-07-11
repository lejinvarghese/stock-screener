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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance import original_flavor as of

try:
    from core.utils import get_data, send_image, send_message, numeric_round
except:
    from utils import get_data, send_image, send_message, numeric_round

filterwarnings("ignore")
load_dotenv()

DIRECTORY = "/media/starscream/wheeljack/projects/"
PROJECT = "stock-screener"
PATH = os.path.join(DIRECTORY, PROJECT)
N_PROCESS = cpu_count() - 1

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
plt.rcParams["savefig.jpeg_quality"] = 100


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
    t_stats = (
        GroupStats(data["Adj Close"]).stats.to_dict(orient="dict").get("Adj Close")
    )

    beta = numeric_round(info.get("beta", "N/A"), 2)
    peg = numeric_round(info.get("pegRatio", "N/A"), 2)
    ptb = numeric_round(info.get("priceToBook", "N/A"), 2)
    dividend_pt = numeric_round(info.get("dividendRate", "N/A"), 2)
    payout = numeric_round(info.get("payoutRatio", "N/A"), 2)
    calmar = numeric_round(t_stats.get("calmar", "N/A"), 2)
    cagr = numeric_round(t_stats.get("cagr", "N/A"), 2)
    monthly_sharpe = numeric_round(t_stats.get("monthly_sharpe", "N/A"), 2)
    monthly_sortino = numeric_round(t_stats.get("monthly_sortino", "N/A"), 2)

    metrics = f"""
    \u03B2: {beta},\n
    PEG Ratio: {peg},\n
    P/B Ratio: {ptb},\n
    Dividend rate: {dividend_pt},\n
    Payout Ratio: {payout}, \n
    Calmar Ratio: {calmar}, \n
    CAGR: {cagr}, \n
    Monthly Sharpe: {monthly_sharpe}, \n
    Monthly Sortino: {monthly_sortino}
    """
    return ticker, metrics


def signals_ma(data):
    """
    Heikin-Ashi advanced ohlc indicators
    """
    ha_close = 0.25 * (2 * data.Close + data.Open + data.Low)
    ha_high = data[["High", "Open", "Close"]].max(axis=1)
    ha_low = data[["Low", "Open", "Close"]].min(axis=1)
    ha_open = 0.5 * (data.Close.shift(1) + data.Open.shift(1))
    data["Close"] = ha_close
    data["High"] = ha_high
    data["Low"] = ha_low
    data["Open"] = ha_open

    data.index = pd.to_datetime(data.index)
    dvalues = data[["Open", "High", "Low", "Close"]].values.tolist()
    pdates = mdates.date2num(data.index)
    ohlc = [[pdates[i]] + dvalues[i] for i in range(len(pdates))]

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
        data["Adj Close"].ewm(W_MA_STRONG_SHORT, adjust=False).mean()
    )
    signals["ma_strong_long"] = (
        data["Adj Close"].ewm(W_MA_STRONG_LONG, adjust=False).mean()
    )

    signals["ma_early_short"] = data["Close"].ewm(W_MA_EARLY_SHORT, adjust=False).mean()
    signals["ma_early_long_high"] = (
        data["High"].rolling(window=W_MA_EARLY_LONG, min_periods=1, center=False).mean()
    )
    signals["ma_early_long_low"] = (
        data["Low"].rolling(window=W_MA_EARLY_LONG, min_periods=1, center=False).mean()
    )

    # Create signals
    signals["signal_early_rising"][W_MA_EARLY_SHORT:] = np.where(
        signals["ma_early_short"][W_MA_EARLY_SHORT:]
        > signals["ma_early_long_high"][W_MA_EARLY_SHORT:],
        1.0,
        0.0,
    )
    signals["positions_early_rising"] = signals["signal_early_rising"].diff()

    signals["signal_early_warning"][W_MA_EARLY_SHORT:] = np.where(
        signals["ma_early_short"][W_MA_EARLY_SHORT:]
        < signals["ma_early_long_low"][W_MA_EARLY_SHORT:],
        1.0,
        0.0,
    )
    signals["positions_early_warning"] = signals["signal_early_warning"].diff()

    signals["signal_strong_warning"][W_MA_EARLY_SHORT:] = np.where(
        signals["ma_early_short"][W_MA_EARLY_SHORT:]
        < signals["ma_strong_short"][W_MA_EARLY_SHORT:],
        1.0,
        0.0,
    )
    signals["positions_warning"] = signals["signal_strong_warning"].diff()

    signals["signal_strong_rising"][W_MA_STRONG_SHORT:] = np.where(
        signals["ma_strong_short"][W_MA_STRONG_SHORT:]
        > signals["ma_strong_long"][W_MA_STRONG_SHORT:],
        1.0,
        0.0,
    )
    signals["positions_strong"] = signals["signal_strong_rising"].diff()
    signals = signals.merge(
        data[["Adj Close", "Volume"]], left_index=True, right_index=True
    )
    return ohlc, signals


def create_plot(signals, ohlc, metrics, ticker):
    """
    Create the visualizations
    """
    fig, (ax_0, ax_1) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(24, 16),
        gridspec_kw={"height_ratios": [8, 1], "hspace": 0.02},
        dpi=240,
    )

    # trends
    ax_0.text(0.01, 0.8, metrics, va="center", transform=ax_0.transAxes)
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
        label=f"Adj Close, {W_MA_STRONG_SHORT}-Day EMA",
    )
    ax_0.plot(
        signals["ma_strong_long"],
        color="red",
        linestyle=":",
        linewidth=1,
        label=f"Adj Close, {W_MA_STRONG_LONG}-Day EMA",
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
    plt.savefig(f"{PATH}/data/outputs/ohlc_{ticker}.png")


def analyze_ticker(data):
    """
    Analyzes all tickers
    """
    ticker, metrics = get_metrics(data)
    ohlc, signals = signals_ma(data)

    curr_signal_strong_rising = signals.iloc[[-1]]["signal_strong_rising"].values[0]
    curr_signal_warning = signals.iloc[[-1]]["signal_strong_warning"].values[0]

    if ((curr_signal_strong_rising == 1)) & (curr_signal_warning < 1):
        create_plot(signals, ohlc, metrics, ticker)
        send_image(
            TELEGRAM_TOKEN, TELEGRAM_ID, f"{PATH}/data/outputs/ohlc_{ticker}.png"
        )
        plt.close("all")
        return ticker
    else:
        return None


def run(n_tickers=100):
    """
    Runs the analysis for all the provided tickers
    """
    df_watchlist = pd.read_csv(f"{PATH}/data/inputs/my_watchlist.csv")
    watchlist = list(df_watchlist.Symbol.unique())[:n_tickers]
    send_message(TELEGRAM_TOKEN, TELEGRAM_ID, "Retrieving Performance History: ")

    with ThreadPool() as t_pool:
        data = t_pool.map(get_data, watchlist)
    data = list(filter(None.__ne__, data))
    send_message(TELEGRAM_TOKEN, TELEGRAM_ID, "Running Technical Analysis: ")
    with Pool(N_PROCESS) as pool:
        pre_selected_stocks = pool.map(analyze_ticker, data)

    pre_selected_stocks = list(filter(None.__ne__, pre_selected_stocks))
    send_message(
        TELEGRAM_TOKEN,
        TELEGRAM_ID,
        f"You have {len(pre_selected_stocks)} stocks with an active BUY rating: "
        + "; ".join(pre_selected_stocks),
    )

    return pre_selected_stocks


if __name__ == "__main__":
    run(n_tickers=10)
