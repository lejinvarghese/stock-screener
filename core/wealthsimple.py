"""
A module to login to Wealthsimple and perform actions
"""

from wsimple.api import Wsimple
import os
from dotenv import load_dotenv

load_dotenv()

credentials = {}
credentials['user'] = os.getenv("WEALTHSIMPLE_USERNAME")
credentials['pass'] = os.getenv("WEALTHSIMPLE_PASSWORD")


def initialize_wealthsimple(credentials):
    """
    Initialize wealthsimple client and login
    """


    def get_otp():
        """
        Get One Time Password to be manually entered by user.
        """
        return input("Enter One Time Password: \n>>>")

    print(credentials.get("user"), credentials.get("pass"))
    wealthsimple_client = Wsimple(credentials.get("user"), credentials.get("pass"), otp_callback=get_otp)
    return wealthsimple_client


def get_wealthsimple_watchlist(credentials):
    """
    Gets the tickers in a wealthsimple watchlist
    """
    wealthsimple_client = initialize_wealthsimple(credentials)
    if wealthsimple_client.is_operational():
        watchlist = wealthsimple_client.get_watchlist()
        tickers = [ticker["stock"]["symbol"] for ticker in watchlist["securities"]]
        print(f"Total tickers: {len(tickers)}, sample: {tickers[0]}")
        return tickers
    else:
        return None


def add_to_wealthsimple_watchlist(credentials, file_name, col="Symbol"):
    """
    Adds tickers to wealthsimple watchlist from a dataframe of tickers
    """
    import pandas as pd
    wealthsimple_client = initialize_wealthsimple(credentials)

    local_watchlist = pd.read_csv(file_name)[col].unique().tolist()
    for ticker in local_watchlist:
        try:
            wealthsimple_client.add_watchlist(
                wealthsimple_client.find_securities(ticker)["id"]
            )
        except:
            continue

if __name__ == "__main__":

    print(get_wealthsimple_watchlist(credentials)[:3])
