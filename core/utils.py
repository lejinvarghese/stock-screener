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

filterwarnings("ignore")


def get_data(ticker, period=36):
    """
    Get the relevant data for tickers
    """
    print(f"Ticker: {ticker}")
    start_time = int(mktime((date.today() - relativedelta(months=+period)).timetuple()))
    end_time = int(mktime(date.today().timetuple()))
    try:
        # pylint: disable=line-too-long
        url = f"https://query2.finance.yahoo.com/v7/finance/download/{ticker}?symbol={ticker}&period1={start_time}&period2={end_time}&interval=1d&events=history&includeAdjustedClose=true"
        data = pd.read_csv(url)
        data["Date"] = pd.to_datetime(data["Date"])
        data.set_index("Date", inplace=True)
        # pylint: disable=unsupported-assignment-operation
        data["ticker"] = ticker
        return data
    except Exception:
        pass


def get_price_ticker_matrix(data, price_type="Adj Close"):
    """
    Reformats the relevant prices from tickers
    """
    prices = []
    for data_t in data:
        ticker = data_t.ticker.iloc[[-1]]
        data_t = data_t[[price_type]]
        data_t.columns = [ticker]
        prices.append(data_t)

    print(prices[:2], len(prices))
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
