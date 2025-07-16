"""
Utilities
"""

from time import mktime
import subprocess
from warnings import filterwarnings
from functools import reduce
from datetime import date
from dateutil.relativedelta import relativedelta
import pandas as pd
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

filterwarnings("ignore")
console = Console()


def get_data(ticker, period=36):
    """
    Get the relevant data for tickers
    """
    console.print(
        f"[bold cyan]Fetching data for:[/bold cyan] [yellow]{ticker}[/yellow]"
    )
    start_time = int(mktime((date.today() - relativedelta(months=+period)).timetuple()))
    end_time = int(mktime(date.today().timetuple()))
    try:
        # pylint: disable=line-too-long
        url = f"https://query2.finance.yahoo.com/v7/finance/download/{ticker}?symbol={ticker}&period1={start_time}&period2={end_time}&interval=1d&events=history&includeAdjustedClose=true"
        console.print(f"[dim]Connecting to Yahoo Finance for {ticker}...[/dim]")
        data = pd.read_csv(url)
        data["Date"] = pd.to_datetime(data["Date"])
        data.set_index("Date", inplace=True)
        # pylint: disable=unsupported-assignment-operation
        data["ticker"] = ticker
        console.print(
            f"[green]Successfully fetched {len(data)} rows for {ticker}[/green]"
        )
        return data
    except Exception as e:
        console.print(f"[red]ERROR: Failed to fetch data for {ticker}: {e}[/red]")

        # Try alternative method using yfinance
        try:
            import yfinance as yf

            console.print(f"[yellow]Trying yfinance fallback for {ticker}...[/yellow]")
            yf_ticker = yf.Ticker(ticker)
            data = yf_ticker.history(period="3y")  # 3 years of data
            if not data.empty:
                data["ticker"] = ticker
                console.print(
                    f"[green]Successfully fetched {len(data)} rows for {ticker} via yfinance[/green]"
                )
                return data
            else:
                console.print(
                    f"[yellow]WARNING: No data returned from yfinance for {ticker}[/yellow]"
                )
                return None
        except Exception as e2:
            console.print(f"[red]ERROR: yfinance also failed for {ticker}: {e2}[/red]")
            return None


def get_price_ticker_matrix(data, price_type="Close"):
    """
    Reformats the relevant prices from tickers
    """
    prices = []
    for data_t in data:
        ticker = data_t["ticker"].iloc[0]
        console.print(f"[blue]Processing ticker:[/blue] [bold]{ticker}[/bold]")

        # Extract the Close price data
        price_data = data_t[["Close"]].copy()
        price_data.columns = [ticker]
        prices.append(price_data)

    prices = reduce(
        lambda df_1, df_2: pd.merge(
            df_1, df_2, left_index=True, right_index=True, how="outer"
        ),
        prices,
    )
    return prices


def send_image(telegram_token, telegram_id, image):
    """
    Sends an image to telegram bot
    """
    command = (
        "curl -s -X POST https://api.telegram.org/bot"
        + telegram_token
        + "/sendPhoto -F chat_id="
        + telegram_id
        + " -F photo=@"
        + image
    )
    subprocess.call(command.split(" "))


def send_message(telegram_token, telegram_id, message=""):
    """
    Sends a text message to telegram bot
    """
    command = (
        "https://api.telegram.org/bot"
        + telegram_token
        + "/sendMessage?chat_id="
        + telegram_id
        + "&parse_mode=Markdown&text="
        + message
    )
    requests.get(command)


def numeric_round(variable, length=0):
    """
    Helps round values only when numeric
    """
    if isinstance(variable, (int, float, complex)) and not isinstance(variable, bool):
        return round(variable, length)
    else:
        return variable
