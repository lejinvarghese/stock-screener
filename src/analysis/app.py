#!/usr/bin/env python

import os
import warnings
warnings.filterwarnings("ignore")
from datetime import date
from dateutil.relativedelta import relativedelta
import subprocess
import requests
from dotenv import load_dotenv
import multiprocessing as mp
from multiprocessing import Pool
load_dotenv()

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
from mplfinance import original_flavor as of
import matplotlib.dates as mdates
import mplfinance as mpf
import yfinance as yf
from ffn.core import GroupStats

DIRECTORY = '/media/starscream/wheeljack/projects/'
PROJECT = 'stock-screener'
PATH = os.path.join(DIRECTORY, PROJECT)

TELEGRAM_TOKEN =os.getenv("TELEGRAM_TOKEN")
TELEGRAM_ID =os.getenv("TELEGRAM_ID")
TELEGRAM_BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

#windows
W_MA_STRONG_SHORT = 50
W_MA_STRONG_LONG = 200
W_MA_EARLY_SHORT = 15
W_MA_EARLY_LONG = 50

#plotting parameters
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Adobe Clean', 'DejaVu Sans']
plt.rcParams['font.size'] = 12
plt.rcParams['savefig.dpi'] = 240
plt.rcParams["savefig.format"] = 'png'
plt.rcParams['savefig.jpeg_quality']=100

def n_round(x, n = 0):
    if isinstance(x, (int, float, complex)) and not isinstance(x, bool):
        return round(x, n)
    else:
        return x

def get_data(ticker):
    data = pdr.get_data_yahoo(ticker, 
                          start=date.today() - relativedelta(months=+48), 
                          end=date.today())
    try:
        info = yf.Ticker(ticker).info
        info = dict(filter(lambda item: item[1] is not None, info.items()))
    except:
        info = dict()
    t_stats = GroupStats(data['Adj Close']).stats.to_dict(orient='dict').get('Adj Close')

    beta = n_round(info.get('beta', 'N/A'), 2)
    peg = n_round(info.get('pegRatio', 'N/A'), 2)
    ptb = n_round(info.get('priceToBook', 'N/A'), 2)
    dividend_pt = n_round(info.get('dividendRate', 'N/A'), 2)
    payout = n_round(info.get('payoutRatio', 'N/A'), 2)
    calmar = n_round(t_stats.get('calmar', 'N/A'), 2)
    cagr = n_round(t_stats.get('cagr', 'N/A'), 2)
    monthly_sharpe = n_round(t_stats.get('monthly_sharpe', 'N/A'), 2)
    monthly_sortino = n_round(t_stats.get('monthly_sortino', 'N/A'), 2)

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
    return data, metrics
  
def upload_watchlist(csv):
    print('csv')

def recommend_starter_stocks(max_stock_price, industries, min_return):
    print('csv')

def get_portfolio(selected_stocks, budget):
    print('csv')

def back_testing(stock):
    print('csv')

def analyze_ma(data):
    
    data.index = pd.to_datetime(data.index)
    dvalues = data[['Open', 'High', 'Low', 'Close']].values.tolist()
    pdates = mdates.date2num(data.index)
    ohlc = [[pdates[i]] + dvalues[i] for i in range(len(pdates))]

    # Initialize the `signals` DataFrame with the `signal` column
    signals = pd.DataFrame(index=data.index)
    signals['signal_strong'], signals['signal_early_rising'], signals['signal_early_warning'], signals['signal_warning'] = 0.0, 0.0, 0.0, 0.0

    # Create moving average over  the windows
    signals['volume'] = data['Volume']
    signals['ma_strong_short'] =  data['Adj Close'].ewm(W_MA_STRONG_SHORT, adjust=False).mean()
    signals['ma_strong_long'] = data['Adj Close'].ewm(W_MA_STRONG_LONG, adjust=False).mean()
    
    signals['ma_early_short'] = data['Close'].ewm(W_MA_EARLY_SHORT, adjust=False).mean()
    signals['ma_early_long_high'] = data['High'].rolling(window=W_MA_EARLY_LONG, min_periods=1, center=False).mean()
    signals['ma_early_long_low'] = data['Low'].rolling(window=W_MA_EARLY_LONG, min_periods=1, center=False).mean()
    
    # Create signals
    signals['signal_early_rising'][W_MA_EARLY_SHORT:] = np.where(signals['ma_early_short'][W_MA_EARLY_SHORT:] > signals['ma_early_long_high'][W_MA_EARLY_SHORT:], 1.0, 0.0)
    signals['positions_early_rising'] = signals['signal_early_rising'].diff()

    signals['signal_early_warning'][W_MA_EARLY_SHORT:] = np.where(signals['ma_early_short'][W_MA_EARLY_SHORT:] < signals['ma_early_long_low'][W_MA_EARLY_SHORT:], 1.0, 0.0)
    signals['positions_early_warning'] = signals['signal_early_warning'].diff()

    signals['signal_warning'][W_MA_EARLY_SHORT:] = np.where(signals['ma_early_short'][W_MA_EARLY_SHORT:] < signals['ma_strong_short'][W_MA_EARLY_SHORT:], 1.0, 0.0)
    signals['positions_warning'] = signals['signal_warning'].diff()

    signals['signal_strong'][W_MA_STRONG_SHORT:] = np.where(signals['ma_strong_short'][W_MA_STRONG_SHORT:] > signals['ma_strong_long'][W_MA_STRONG_SHORT:], 1.0, 0.0)
    signals['positions_strong'] = signals['signal_strong'].diff()
    signals = signals.merge(data[['Adj Close', 'Volume']], left_index = True, right_index = True)

    return ohlc, signals

def create_plot(signals, ohlc, metrics, ticker):
    # Initialize the plot figure
    fig, (ax , ax1)= plt.subplots(nrows=2, ncols=1, figsize = (24, 16), gridspec_kw={'height_ratios': [8, 1], 'hspace': 0.02}, dpi=240)

    #trends
    ax.text(0.01, 0.8, metrics, va='center', transform=ax.transAxes)
    of.candlestick_ohlc(ax, ohlc, colorup='#77d879', colordown='#db3f3f', width=0.5, alpha=0.2)
    ax.plot(signals['ma_early_short'], color = 'lightgreen', linewidth = 2, label=f'Close, {W_MA_EARLY_SHORT}-Day EMA')
    ax.plot(signals['ma_early_long_high'], color = 'palegreen', linestyle=':', linewidth = 1, label=f'High, {W_MA_EARLY_LONG}-Day SMA')
    ax.plot(signals['ma_early_long_low'], color = 'salmon', linestyle=':', linewidth = 1,  label=f'Low, {W_MA_EARLY_LONG}-Day SMA')

    ax.plot(signals['ma_strong_short'], color = 'seagreen', linewidth = 2, label=f'Adj Close, {W_MA_STRONG_SHORT}-Day EMA')
    ax.plot(signals['ma_strong_long'], color = 'red', linestyle=':', linewidth = 1, label=f'Adj Close, {W_MA_STRONG_LONG}-Day EMA')

    #ticks
    ax.plot(signals.loc[signals.positions_early_rising == 1.0].index, signals.ma_early_short[signals.positions_early_rising == 1.0], '^', markersize=10, color='springgreen', label='early rising')
    ax.plot(signals.loc[signals.positions_early_warning == 1.0].index, signals.ma_early_short[signals.positions_early_warning == 1.0], 'v', markersize=10, color='gold', label='early warning')
    ax.plot(signals.loc[signals.positions_warning == 1.0].index, signals.ma_early_short[signals.positions_warning == 1.0], 'v', markersize=10, color='darkorange', label='warning')
    ax.plot(signals.loc[signals.positions_strong == 1.0].index, signals.ma_strong_short[signals.positions_strong == 1.0], '^', markersize=10, color='forestgreen', label='strong rising')
    ax.plot(signals.loc[signals.positions_strong == -1.0].index, signals.ma_strong_short[signals.positions_strong == -1.0], 'v', markersize=10, color='red',  label='strong decline')
    
    ax.set_ylabel('Price ($)')
    ax.set_title(f'Trend for stock: {str.upper(ticker)}')
    ax.legend(loc='upper right', bbox_to_anchor=(1.14, 1.0))
    
    #secondary graph: volume
    ax1.bar(x = signals.index, height = signals['volume'], color = 'slategray', align='center', alpha=0.5, label='Volume')
    ax1.axhline(signals['volume'].tail(180).median(), linestyle=':', linewidth = 1)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Volume')
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    fig.autofmt_xdate()
    plt.savefig(f'{PATH}/data/outputs/plot_{ticker}.png')

def send_image(telegram_token, telegram_id, image):
        command = 'curl -s -X POST https://api.telegram.org/bot' + telegram_token + '/sendPhoto -F chat_id=' + telegram_id + " -F photo=@" + image
        subprocess.call(command.split(' '))
    
def send_text(telegram_token, telegram_id, message):
    send_text = 'https://api.telegram.org/bot' + telegram_token + '/sendMessage?chat_id=' + telegram_id + '&parse_mode=Markdown&text=' + message
    requests.get(send_text)

def main(ticker):
    data, metrics = get_data(ticker)
    ohlc, signals = analyze_ma(data)

    current_signal_strong = signals.iloc[[-1]]['signal_strong'].values[0]
    current_signal_early_rising = signals.iloc[[-1]]['signal_early_rising'].values[0]
    current_price = signals.iloc[[-1]]['Adj Close'].values[0]

    if ((current_signal_strong==1) | (current_signal_early_rising==1)) & (current_price<90):
        create_plot(signals, ohlc, metrics, ticker)
        send_image(TELEGRAM_TOKEN, TELEGRAM_ID, f'{PATH}/data/outputs/plot_{ticker}.png')
        plt.close('all')
        return ticker

    # with open(f'{PATH}/data/outputs/selected_tickers.txt', 'w') as text_file:
    #     print(f"{ticker}", file=text_file)

if __name__=='__main__':

    df_wl = pd.read_csv(f'{PATH}/data/inputs/my_watchlist.csv')
    watchlist = list(df_wl.Symbol.unique())[:15]
    send_text(TELEGRAM_TOKEN, TELEGRAM_ID, "Starting Stock Spock: ")
    cores = mp.cpu_count() - 1
    with Pool(cores) as p:
        selected_stocks = p.map(main, watchlist)

    selected_stocks = list(filter(None.__ne__, selected_stocks))
    send_text(TELEGRAM_TOKEN, TELEGRAM_ID, 'It is only logical to consider: ' + '; '.join(selected_stocks))
    send_text(TELEGRAM_TOKEN, TELEGRAM_ID,  'Live long and prosper \U0001F596')