import os
from datetime import date
from dateutil.relativedelta import relativedelta
import subprocess
import requests
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
from mplfinance import original_flavor as of
import matplotlib.dates as mdates
import mplfinance as mpf

DIRECTORY = '/media/starscream/wheeljack/projects/'
PROJECT = 'stocks-bot'

TELEGRAM_TOKEN =os.getenv("TELEGRAM_TOKEN")
TELEGRAM_ID =os.getenv("TELEGRAM_ID")
TELEGRAM_BASE_URL = "https://api.telegram.org/bot{}".format(TELEGRAM_TOKEN)

plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Adobe Clean', 'DejaVu Sans']
plt.rcParams['font.size'] = 12
plt.rcParams['savefig.dpi'] = 240
plt.rcParams["savefig.format"] = 'png'
plt.rcParams['savefig.jpeg_quality']=100

def get_data(ticker):
    return pdr.get_data_yahoo(ticker, 
                          start=date.today() - relativedelta(months=+24), 
                          end=date.today())

def upload_watchlist(csv):
    print('csv')

def recommend_starter_stocks(max_stock_price, industries, min_return):
    print('csv')

def get_portfolio(selected_stocks, budget):
    print('csv')

def back_testing(stock):
    print('csv')


def get_ma(data, sma_short = 50, sma_long = 200, ema_short = 15, hsma_long = 40, lsma_long = 40):
    
    data.index = pd.to_datetime(data.index)
    dvalues = data[['Open', 'High', 'Low', 'Close']].values.tolist()
    pdates = mdates.date2num(data.index)
    ohlc = [ [pdates[i]] + dvalues[i] for i in range(len(pdates)) ]

    # Initialize the `signals` DataFrame with the `signal` column
    signals = pd.DataFrame(index=data.index)
    signals['signal_sma'], signals['signal_ema_hsma'] , signals['signal_ema_lsma'] = 0.0, 0.0, 0.0

    # Create moving average over  the windows
    signals['sma_s'] =  data['Adj Close'].rolling(window=sma_short, min_periods=1, center=False).mean()
    signals['sma_l'] = data['Adj Close'].rolling(window=sma_long, min_periods=1, center=False).mean()
    
    signals['ema_s'] = data['Close'].ewm(ema_short, adjust=False).mean()
    signals['hsma_l'] = data['High'].rolling(window=hsma_long, min_periods=1, center=False).mean()
    signals['lsma_l'] = data['Low'].rolling(window=lsma_long, min_periods=1, center=False).mean()
    
    # Create signals
    signals['signal_ema_hsma'][ema_short:] = np.where(signals['ema_s'][ema_short:] > signals['hsma_l'][ema_short:], 1.0, 0.0)
    signals['positions_ema_hsma'] = signals['signal_ema_hsma'].diff()

    signals['signal_ema_lsma'][ema_short:] = np.where(signals['ema_s'][ema_short:] < signals['lsma_l'][ema_short:], 1.0, 0.0)
    signals['positions_ema_lsma'] = signals['signal_ema_lsma'].diff()

    signals['signal_sma'][sma_short:] = np.where(signals['sma_s'][sma_short:] > signals['sma_l'][sma_short:], 1.0, 0.0)
    signals['positions_sma'] = signals['signal_sma'].diff()
    signals = signals.merge(data[['Adj Close', 'Volume']], left_index = True, right_index = True)

    return ohlc, signals

def get_plot(ticker, plt_w = 365):
    # Initialize the plot figure
    fig, ax = plt.subplots(figsize = (10, 5))

    signals_ = signals.tail(plt_w)
    of.candlestick_ohlc(ax, ohlc[-plt_w:], colorup='#77d879', colordown='#db3f3f', width=0.8)
    ax.plot(signals_['hsma_l'], color = 'dimgrey', linestyle='--', linewidth = 1, label='High, 40-Day SMA')
    ax.plot(signals_['lsma_l'], color = 'dimgrey', linestyle='--', linewidth = 1, label='Low, 40-Day SMA')
    ax.plot(signals_['sma_l'], color = 'dimgrey', linestyle='--', linewidth = 1, label='Adj Close, 200-Day SMA')

    ax.plot(signals_['ema_s'], color = 'red', linewidth = 1, label='Close, 15-Day EMA')
    ax.plot(signals_['sma_s'], color = 'firebrick', linewidth = 1, label='Adj Close, 50-Day SMA')


    ax.plot(signals_.loc[signals_.positions_ema_hsma == 1.0].index, signals_.ema_s[signals_.positions_ema_hsma == 1.0], '^', markersize=10, color='springgreen')
    ax.plot(signals_.loc[signals_.positions_ema_lsma == 1.0].index, signals_.ema_s[signals_.positions_ema_lsma == 1.0], 'v', markersize=10, color='red')

    ax.plot(signals_.loc[signals_.positions_sma == 1.0].index, signals_.sma_s[signals_.positions_sma == 1.0], '^', markersize=10, color='forestgreen')
    ax.plot(signals_.loc[signals_.positions_sma == -1.0].index, signals_.sma_s[signals_.positions_sma == -1.0], 'v', markersize=10, color='firebrick')

    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title('Trend for stock: {}'.format(ticker))
    ax.legend()
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    fig.autofmt_xdate()
    plt.savefig('plot.png')

def send_image(telegram_token, telegram_id, image):
        command = 'curl -s -X POST https://api.telegram.org/bot' + telegram_token + '/sendPhoto -F chat_id=' + telegram_id + " -F photo=@" + image
        subprocess.call(command.split(' '))
    
def send_text(telegram_token, telegram_id, message):
    send_text = 'https://api.telegram.org/bot' + telegram_token + '/sendMessage?chat_id=' + telegram_id + '&parse_mode=Markdown&text=' + message
    requests.get(send_text)




if __name__=='__main__':

    df_wl = pd.read_csv('{}/data/my_watchlist.csv'.format(os.path.join(DIRECTORY, PROJECT)))
    watchlist = list(df_wl.Symbol.unique())[:7]
    selected_stocks = []

    send_text(TELEGRAM_TOKEN, TELEGRAM_ID, "Starting analysis")

    for _, ticker in enumerate(watchlist):
        data = get_data(ticker)
        ohlc, signals = get_ma(data)

        current_signal_sma = signals.iloc[[-1]]['signal_sma'].values[0]
        current_signal_ema_hsma = signals.iloc[[-1]]['signal_ema_hsma'].values[0]
        current_price = signals.iloc[[-1]]['Adj Close'].values[0]
        
        if ((current_signal_sma==1) | (current_signal_ema_hsma==1))& (current_price<60):
            get_plot(ticker)
            send_image(TELEGRAM_TOKEN, TELEGRAM_ID,  'plot.png')
            plt.close('all')
            selected_stocks.append(ticker)
        
    send_text(TELEGRAM_TOKEN, TELEGRAM_ID, 'Selected stocks: ' + '; '.join(selected_stocks))
    send_text(TELEGRAM_TOKEN, TELEGRAM_ID,  "Finished analysis")