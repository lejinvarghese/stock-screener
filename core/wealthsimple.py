from wsimple.api import Wsimple
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()


def get_otp():
    return input("Enter One Time Password: \n>>>")


USERNAME = os.getenv("WEALTHSIMPLE_USERNAME")
PASSWORD = os.getenv("WEALTHSIMPLE_PASSWORD")

wealthsimple_client = Wsimple(USERNAME, PASSWORD, otp_callback=get_otp)


def get_wealthsimple_watchlist(wealthsimple_client):

    if wealthsimple_client.is_operational():
        watchlist = wealthsimple_client.get_watchlist()
        tickers = [ticker["stock"]["symbol"] for ticker in watchlist["securities"]]
        print(f"Total tickers: {len(tickers)}, sample: {tickers[0]}")
        return tickers
    else:
        return None


def add_to_wealthsimple_watchlist(wealthsimple_client, file_name, col="Symbol"):
    local_watchlist = pd.read_csv(file_name)[col].unique().tolist()
    for ticker in local_watchlist:
        try:
            wealthsimple_client.add_watchlist(wealthsimple_client.find_securities(ticker)["id"])
        except:
            continue