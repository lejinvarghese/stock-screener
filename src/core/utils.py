from time import mktime
import pandas as pd
import requests
import subprocess
from dateutil.relativedelta import relativedelta
from datetime import date
from functools import reduce
from warnings import filterwarnings
filterwarnings("ignore")

def get_data(ticker, period = 36):
    print(f"Ticker: {ticker}")
    start_time = int(
        mktime((date.today() - relativedelta(months=+period)).timetuple()))
    end_time = int(mktime(date.today().timetuple()))
    try:
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_time}&period2={end_time}&interval=1d&events=history&includeAdjustedClose=true"
        data = pd.read_csv(url)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        data['ticker'] = ticker
        return data
    except:
        pass

def get_prices(data, price_type = "Adj Close"):
    prices = []
    for i in data:
        ticker = i.ticker.iloc[[-1]]
        i = i[[price_type]]
        i.columns = [ticker]
        prices.append(i)
    prices = reduce(lambda df_1,df_2: pd.merge(df_1,df_2, left_index=True, right_index=True, how="outer"), prices)
    return prices

def send_image(telegram_token, telegram_id, image):
    command = 'curl -s -X POST https://api.telegram.org/bot' + telegram_token + \
        '/sendPhoto -F chat_id=' + telegram_id + " -F photo=@" + image
    subprocess.call(command.split(' '))

def send_message(telegram_token, telegram_id, message=''):
    command = 'https://api.telegram.org/bot' + telegram_token + \
        '/sendMessage?chat_id=' + telegram_id + '&parse_mode=Markdown&text=' + message
    requests.get(command)
