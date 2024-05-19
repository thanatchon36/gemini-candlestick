import sys, os
import time
import requests
import datetime
import os.path
from os import path
import hmac
import hashlib
from urllib.parse import urlencode
import numpy as np
import pandas as pd
from itertools import accumulate 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('figure', figsize = (15, 12), dpi = 300)
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import base64
import google.generativeai as genai
import json
from PIL import Image
import gc
import talib
import re
# from plotly.subplots import make_subplots
# from plotly.graph_objects import Scatter
def reset(df):
    cols = df.columns
    return df.reset_index()[cols]
def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)
def get_file_age(filename):
    now = datetime.datetime.now()
    return (now - modification_date(filename)).seconds
def round_down(value, decimals):
    value = value + 0.0000000001
    int_num = str('{0:.10f}'.format(value)).split('.')[0]
    float_num = str('{0:.10f}'.format(value)).split('.')[1][:decimals]
    return float(int_num + '.' + float_num)
def label_inverse_flag(txt):
    if txt == 'SHORT':
        return 'LONG'
    else:
        return 'SHORT'
def calc_MDD(networth):
    df = pd.Series(networth, name="nw").to_frame()
    max_peaks_idx = df.nw.expanding(min_periods=1).apply(lambda x: x.argmax()).fillna(0).astype(int)
    df['max_peaks_idx'] = pd.Series(max_peaks_idx).to_frame()
    nw_peaks = pd.Series(df.nw.iloc[max_peaks_idx.values].values, index=df.nw.index)
    df['dd'] = ((df.nw-nw_peaks)/nw_peaks)
    df['mdd'] = df.groupby('max_peaks_idx').dd.apply(lambda x: x.expanding(min_periods=1).apply(lambda y: y.min())).fillna(0)
    return df
def get_con_G(x):
    try:
        return len(x.split('R')[-1])
    except:
        return np.nan
def get_con_R(x):
    try:
        return len(x.split('G')[-1])
    except:
        return np.nan
def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
def convert_to_utc_time(num):
    timestamp = num / 1000  # Convert milliseconds to seconds
    utc_time = datetime.datetime.utcfromtimestamp(timestamp)
    return utc_time
def add(a, b):
    return a + b
def get_color_list(values):
    # Define dark and light shades of green and red
    dark_green = '#26A69A'
    light_green = '#B2DFDB'
    light_red = '#FFCDD2'
    dark_red = '#FF5252'
    colors = []
    previous_value = values[0]
    for value in values:
        if value < 0:
            if value < previous_value:
                colors.append(dark_red)
            else:
                colors.append(light_red)
        elif value > 0:
            if value > previous_value:
                colors.append(dark_green)
            else:
                colors.append(light_green)
        else:
            colors.append(light_green)  # Adjust for zero values as needed
        previous_value = value
    return colors
def millions_formatter(x, pos):
    num = x
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
def comma_formatter(x, pos):
    return "{:,}".format(x)
def comma_formatter_2(x, pos):
    return "{:,.2f}".format(x)
def title_formatter(x):
    return "{:.2f}".format(x)
def load_image_as_b64(pathname: str) -> str:
    """Loads an image file and encodes it as base64 string."""
    path = Path(pathname)
    with open(path, "rb") as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode("utf-8")
def save_b64_to_png(b64_string: str, filename: str):
    """Decodes a base64 string and saves it as a PNG image file."""
    decoded_image_data = base64.b64decode(b64_string)
    with open(filename, "wb") as f:
        f.write(decoded_image_data)
def calculate_rsi(prices, period=14):
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [max(delta, 0) for delta in deltas]
    losses = [max(-delta, 0) for delta in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsi_values = [100 - (100 / (1 + avg_gain / avg_loss))]
    for i in range(period, len(prices)):
        try:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            rsi = 100 - (100 / (1 + avg_gain / avg_loss))
            rsi_values.append(rsi)
        except:
            pass
    return [np.nan for i in range(period)] + rsi_values
def calculate_sma(data, window=14):
    sma_values = []
    for i in range(len(data) - window + 1):
        sma = sum(data[i:i + window]) / window
        sma_values.append(sma)
    return [np.nan for i in range(window-1)] + sma_values
def exp_moving_avg(data, period):
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = (data[i] * (2 / (period + 1))) + ema[i - 1] * (1 - (2 / (period + 1)))
    return ema
def MACD(data, fast_period, slow_period, signal_period):
    ema_fast = exp_moving_avg(data, fast_period)
    ema_slow = exp_moving_avg(data, slow_period)
    macd = np.zeros_like(data)
    macd_signal = np.zeros_like(data)
    macd_hist = np.zeros_like(data)
    # Calculate MACD, skipping initial null values
    for i in range(max(fast_period, slow_period) - 1, len(data)):
        macd[i] = ema_fast[i] - ema_slow[i]
    # Calculate MACD Signal and MACD Histogram, skipping initial null values
    for i in range(max(fast_period, slow_period, signal_period) - 1, len(macd)):
        macd_signal[i] = exp_moving_avg(macd[:i + 1], signal_period)[-1]
        macd_hist[i] = macd[i] - macd_signal[i]
    return macd, macd_signal, macd_hist
def calc_k_d(data, N=14, M=3):
    data['low_N'] = data['Low'].rolling(N).min()
    data['high_N'] = data['High'].rolling(N).max()
    data['K_STO'] = 100 * (data['Close'] - data['low_N']) / \
        (data['high_N'] - data['low_N'])
    data['D_STO'] = data['K_STO'].rolling(M).mean()
    data.drop(columns = ['low_N','high_N'], inplace = True)
    return data
def calc_CMF(ask_series):
    ask_series["cmfm"] = (((ask_series["Close"] - ask_series["Low"]) - (ask_series["High"] - ask_series["Close"])) / (ask_series["High"] - ask_series["Low"]))
    ask_series["cmfv"] = ask_series["cmfm"] * ask_series["Volume"]
    ask_series["CMF"] = ask_series['cmfv'].rolling(window=20).mean() / ask_series['Volume'].rolling(window=21).mean() 
    ask_series.drop(columns = ['cmfm','cmfv'], inplace = True)
    return ask_series
def color_title(ax, labels, colors, textprops={'size': 15}, y=0.96,
               precision=10**-2):
    """Creates a left-aligned title with multiple colors. Don't change axes limits afterwards."""
    plt.gcf().canvas.draw()
    transform = ax.transAxes  # use axes coords
    # Initial params - start from the left (x=0)
    x_pos = 0  
    # For text objects
    text = dict()
    for label, col in zip(labels, colors):
        text[label] = ax.text(x_pos, y, label,
                              transform=transform,
                              ha='left',  # Left alignment
                              color=col,
                              **textprops)
        # Update x_pos for the next label
        x_pos = text[label].get_window_extent().transformed(transform.inverted()).x1
def extract_json(text_response):
    # This pattern matches a string that starts with '{' and ends with '}'
    pattern = r'\{[^{}]*\}'
    matches = re.finditer(pattern, text_response)
    json_objects = []
    for match in matches:
        json_str = match.group(0)
        try:
            # Validate if the extracted string is valid JSON
            json_obj = json.loads(json_str)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            # Extend the search for nested structures
            extended_json_str = extend_search(text_response, match.span())
            try:
                json_obj = json.loads(extended_json_str)
                json_objects.append(json_obj)
            except json.JSONDecodeError:
                # Handle cases where the extraction is not valid JSON
                continue
    if json_objects:
        return json_objects
    else:
        return None  # Or handle this case as you prefer
def extend_search(text, span):
    # Extend the search to try to capture nested structures
    start, end = span
    nest_count = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            nest_count += 1
        elif text[i] == '}':
            nest_count -= 1
            if nest_count == 0:
                return text[start:i+1]
    return text[start:end]
####
#### Start Class Here:
####
class autotrade:
    def __init__(self, 
                 ws_name,
                 freq_interval = "15m",
                 line_api_key = "",
                 binance_api_key = "",
                 binance_api_secret = "",
                 dev_flag = 0,
                 test_mode=0,
                 sync = True,
                 model = "gemini-1.5-pro-latest",
                 depth_limit_no = 1000,
                ):
        self.binance_api_key = binance_api_key
        self.binance_api_secret = binance_api_secret
        self.KEY = self.binance_api_key
        self.SECRET = self.binance_api_secret
        self.BASE_URL = 'https://fapi.binance.com'
        self.line_api_key = line_api_key
        self.test_mode = test_mode
        self.ws_name = ws_name
        self.dev_flag = dev_flag
        self.sync = sync
        self.stop_loss_percent = 8.5
        self.time_offset = 0
        self.freq_dict = {
            '1m': 60,
            '3m': 90,
            '5m': 300,
            '15m':900,
            '30m':1800,
            '1h':3600,
            '2h':3600*2,
            '4h':3600*4,
            '6h':3600*6,
            '8h':3600*8,
            '12h':3600*12,
            '1d':3600*24,
            '3d':3600*24*3,
            '1w': 3600*24*7,
        }
        self.freq_text_dict = {
            '1m': "1 minute",
            '3m': "3 minutes",
            '5m': "5 minutes",
            '15m': "15 minutes",
            '30m': "30 minutes",
            '1h': "1 hour",
            '2h': "2 hours",
            '4h': "4 hours",
            '6h': "6 hours",
            '8h': "8 hours",
            '12h': "12 hours",
            '1d': "1 day",
            '3d': "3 days",
            '1w': "7 days",
        }
        self.exit_stick_no_dict = {
            '2h': 12,
            '4h': 6,
            '6h': 4,
            '12h': 2,
        }
        self.freq_interval = freq_interval
        self.ori_freq_interval = freq_interval
        self.depth_limit_no = depth_limit_no
        self.ori_depth_limit_no = depth_limit_no
        self.x_date_interval_dict = {
            '1w': int(30 * 7/3) + 1,
            '3d': 30,
            '1d': int(30/3),
            '12h': int(30/(3*2)) ,
            '8h': int(30/(3*2)*8/12) + 1,
            '6h': 3,
            '4h': 2,
            '2h': 1,
            '1h': 1,
            '30m': 1,
        }
        self.graph_width_dict = {
            '1w': 4.75,
            '3d': 2.15,
            '1d': 0.75,
            '12h': 0.35,
            '8h': 0.225,
            '6h': 0.18,
            '4h': 0.12,
            '2h': 0.063,
            '1h': 0.029,
            '30m': 0.015,
        }
        self.gc_collect_time_dict = {
            '1w': (48*7)-1,
            '3d': (48*3)-1,
            '1d': 48-1,
            '12h': 24-1,
            '8h': 16-1,
            '6h': 12-1,
            '4h': 8-1,
            '2h': 4-1,
            '1h': 2-1,
            '30m': 0,
        }
        self.model = model
        tf_list = ['1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']
        self.include_tf_list = []
        for each_tf in tf_list:
            tf_sec = self.freq_dict[each_tf]
            current_tf_sec = self.freq_dict[self.ori_freq_interval]
            if tf_sec > current_tf_sec:
                self.include_tf_list.append(each_tf)
        self.candlestick_chart_no = 168
        self.future_cloud_no = 26
    @property
    def gc_collect_time(self):
        return self.gc_collect_time_dict[self.ori_freq_interval]
    @property
    def x_date_interval(self):
        return self.x_date_interval_dict[self.freq_interval]
    @property
    def graph_width(self):
        return self.graph_width_dict[self.freq_interval]
    @property
    def input_type_text(self):
        if self.model == 'gemini-1.5-pro-latest':
            return f"The Board of Directors will receive user-input images displaying the Current Market Depth of BTC, featuring the top bids and asks (5, 10, 20, 50, 100, 500, 1000) from the order book. Additionally, they will receive user-input images covering all specified technical indicators for Bitcoin trading. These indicators include candlestick charts across various timeframes (30-minute, 1-hour, 2-hour, 4-hour, 6-hour, 8-hour, 12-hour, 1-day, 3-day, and 1-week), as well as volume, RSI, MACD, Bollinger Bands, Fibonacci Retracement, Ichimoku Cloud, Stochastic Oscillator, Chaikin Money Flow, On-Balance Volume, and Average True Range. These comprehensive graphs visually represent all metrics, aiding in the identification of significant patterns, trends, and market dynamics. This data-driven approach will enable informed decision-making regarding our investments and strategies."
        elif self.model == 'gemini-1.5-flash-latest':
            return f"The Board of Directors will receive JSON data containing the Current Market Depth of BTC, including the top bids and asks (5, 10, 20, 50, 100, 500, 1000) from the order book. Additionally, they will receive JSON data covering all specified technical indicators for Bitcoin trading. These indicators include candlestick charts across various timeframes (30-minute, 1-hour, 2-hour, 4-hour, 6-hour, 8-hour, 12-hour, 1-day, 3-day, and 1-week), as well as volume, RSI, MACD, Bollinger Bands, Ichimoku Cloud, Stochastic Oscillator, Chaikin Money Flow, On-Balance Volume, and Average True Range. These comprehensive JSON datasets provide a structured representation of all metrics, aiding in the identification of significant patterns, trends, and market dynamics. This data-driven approach will enable informed decision-making regarding our investments and strategies."
    @property
    def freq_second(self):
        return self.freq_dict[self.ori_freq_interval]
    @property
    def freq_text(self):
        return self.freq_text_dict[self.ori_freq_interval]
    @property
    def exit_stick_no(self):
        return self.exit_stick_no_dict[self.freq_interval]
    def lineNotify(self, msg, image_path = ""):
        try:
            url = 'https://notify-api.line.me/api/notify'
            token = self.line_api_key
            headers = {'Authorization': 'Bearer ' + token}
            if image_path == "":
                payload = {'message': f"{self.ws_name}: {msg}"}
                requests.post(url, headers=headers, data=payload)
            else:
                with open(image_path, 'rb') as image_file:
                    image_data = image_file.read()
                    payload = {'message': f"{self.ws_name}: {msg}"}
                    files = {'imageFile': image_data}
                    requests.post(url, headers=headers, data=payload, files=files)
                    image_file = None
                    image_data = None
                    payload = None
                    files = None
                    del image_file, image_data, payload, files
                    gc.collect()
        except:
            self.docker_print('Error: lineNotify')
            pass
    def docker_print(self, txt):
        print(txt, flush=True)
    def print_response_params(self):
        try:
            self.docker_print('Error !: response: {} params: {}'.format(self.response, self.params))
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno)
            self.docker_print(temp_msg)
        finally:
            pass
    def hashing(self, query_string):
        return hmac.new(self.SECRET.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    def get_timestamp(self):
        return int(time.time() * 1000)
    def dispatch_request(self, http_method):
        session = requests.Session()
        session.headers.update({
            'Content-Type': 'application/json;charset=utf-8',
            'X-MBX-APIKEY': self.KEY
        })
        return {
            'GET': session.get,
            'DELETE': session.delete,
            'PUT': session.put,
            'POST': session.post,
        }.get(http_method, 'GET')
    # used for sending request requires the signature
    def send_signed_request(self, http_method, url_path, payload={}):
        query_string = urlencode(payload)
        # replace single quote to double quote
        query_string = query_string.replace('%27', '%22')
        if query_string:
            query_string = "{}&timestamp={}".format(query_string, self.get_timestamp())
        else:
            query_string = 'timestamp={}'.format(self.get_timestamp())

        url = self.BASE_URL + url_path + '?' + query_string + '&signature=' + self.hashing(query_string)
        # self.docker_print("{} {}".format(http_method, url))
        params = {'url': url, 'params': {}}
        response = self.dispatch_request(http_method)(**params)
        return response.json()
    # used for sending public data request
    def send_public_request(self, url_path, payload={}):
        query_string = urlencode(payload, True)
        url = self.BASE_URL + url_path
        if query_string:
            url = url + '?' + query_string
        # self.docker_print("{}".format(url))
        response = self.dispatch_request('GET')(url=url)
        return response.json()
    def ping_binance(self):
        server_time_df = pd.DataFrame([0])
        self.params = {
            'symbol': 'BTCUSDT',
            'interval': self.freq_interval,
            'limit': '1',
        }
        self.response = self.send_signed_request('GET', '/fapi/v1/klines', self.params)
        server_time_df['next'] = self.response[0][6]
        server_time_df['next']=server_time_df['next'].apply(lambda d: datetime.datetime.fromtimestamp(int(d)/1000).strftime('%Y-%m-%d %H:%M:%S'))
        server_time_df['next'] = pd.to_datetime(server_time_df['next'], format='%Y-%m-%d %H:%M:%S') + pd.Timedelta(seconds=1)
        entry_ts = server_time_df['next'].values[-1]
        entry_ts = str(entry_ts)[:10] + ' ' + str(entry_ts)[11:19]
        return entry_ts
    def ping_past_binance(self):
        server_time_df = pd.DataFrame([0])
        self.params = {
            'symbol': 'BTCUSDT',
            'interval': self.freq_interval,
            'limit': '2',
        }
        self.response = self.send_signed_request('GET', '/fapi/v1/klines', self.params)
        server_time_df['next'] = self.response[0][6]
        server_time_df['next']=server_time_df['next'].apply(lambda d: datetime.datetime.fromtimestamp(int(d)/1000).strftime('%Y-%m-%d %H:%M:%S'))
        server_time_df['next'] = pd.to_datetime(server_time_df['next'], format='%Y-%m-%d') + pd.Timedelta(seconds=1)
        entry_ts = server_time_df['next'].values[-1]
        entry_ts = str(entry_ts)[:10] + ' ' + str(entry_ts)[11:19]
        return entry_ts
    def ping_past_ori_binance(self):
        server_time_df = pd.DataFrame([0])
        self.params = {
            'symbol': 'BTCUSDT',
            'interval': self.freq_interval,
            'limit': '2',
        }
        self.response = self.send_signed_request('GET', '/fapi/v1/klines', self.params)
        return self.response[0][6] + 1
    def clear_open_order(self):
        try:
            self.response = self.send_signed_request('GET', '/fapi/v1/openOrders')
            temp_open_order_list = self.response
            if len(temp_open_order_list) > 0:
                for each in list(set(pd.DataFrame(temp_open_order_list)['symbol'].values)):
                    self.params = {
                        'symbol': each
                    }
                    self.send_signed_request('DELETE', '/fapi/v1/allOpenOrders', self.params)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno)
            self.docker_print(temp_msg)
            self.lineNotify(temp_msg)
            self.print_response_params()
        finally:
            pass
    def clear_position(self, extra_msg = ""):
        try:
            self.update_temp_pos_df()
            temp_hold_minus_symbol_list = self.temp_pos_df[self.temp_pos_df['positionAmt'] < 0]['symbol'].tolist()
            temp_hold_plus_symbol_list = self.temp_pos_df[self.temp_pos_df['positionAmt'] > 0]['symbol'].tolist()

            if len(temp_hold_minus_symbol_list) == 0 and len(temp_hold_plus_symbol_list) == 0:
                temp_msg = 'Force Close: Position Closed !'
                if extra_msg != "":
                    temp_msg = temp_msg + " " + extra_msg
                self.docker_print(temp_msg)
                self.lineNotify(temp_msg)

            for each in temp_hold_minus_symbol_list:
                temp_quantity = abs(float(self.temp_pos_df[self.temp_pos_df['symbol'] == each]['positionAmt'].values[0])) / self.batchOrders_no
                quantityPrecision = self.exchange_df[self.exchange_df['symbol'] == each]['quantityPrecision'].values[0]
                self.each_params = {
                    'symbol': each,
                    'side': 'BUY',
                    'type': 'MARKET',
                    'quantity': str(round(temp_quantity, int(quantityPrecision))),
                }
                self.params = {
                    "batchOrders": []
                }

                if self.batchOrders_no == 1:
                    self.params = {
                        "batchOrders": [self.each_params]
                    }
                    self.response = self.send_signed_request("POST",'/fapi/v1/batchOrders', self.params)
                else:
                    self.params = {
                        "batchOrders": [self.each_params, self.each_params, self.each_params, self.each_params, self.each_params]
                    }
                    for batch_i in range(int(self.batchOrders_no / 5)):
                        self.response = self.send_signed_request("POST",'/fapi/v1/batchOrders', self.params)

                entryPrice = self.temp_pos_df[self.temp_pos_df['symbol'] == each]['entryPrice'].values[0]
                markPrice = self.temp_pos_df[self.temp_pos_df['symbol'] == each]['markPrice'].values[0]
                positionAmt = self.temp_pos_df[self.temp_pos_df['symbol'] == each]['positionAmt'].values[0]
                leverage_no = self.temp_pos_df[self.temp_pos_df['symbol'] == each]['leverage'].values[0]

                if positionAmt > 0:
                    profit_usd = (markPrice - entryPrice) * 100 / entryPrice
                elif positionAmt < 0:
                    profit_usd = -(markPrice - entryPrice) * 100 / entryPrice

                temp_msg = 'Force Close: Symbol: {} Realized Profit: {}/{} entryPrice: {} markPrice: {} Position Closed !'.format(each, 
                                                                        round(profit_usd, 2),
                                                                        round(profit_usd * leverage_no, 2),
                                                                        round(entryPrice, 4),
                                                                        round(markPrice, 4),
                )
                if extra_msg != "":
                    temp_msg = temp_msg + " " + extra_msg
                self.docker_print(temp_msg)
                self.lineNotify(temp_msg)

            for each in temp_hold_plus_symbol_list:
                temp_quantity = abs(float(self.temp_pos_df[self.temp_pos_df['symbol'] == each]['positionAmt'].values[0])) / self.batchOrders_no
                quantityPrecision = self.exchange_df[self.exchange_df['symbol'] == each]['quantityPrecision'].values[0]
                self.each_params = {
                    'symbol': each,
                    'side': 'SELL',
                    'type': 'MARKET',
                    'quantity': str(round(temp_quantity, int(quantityPrecision))),
                }
                self.params = {
                    "batchOrders": []
                }

                if self.batchOrders_no == 1:
                    self.params = {
                        "batchOrders": [self.each_params]
                    }
                    self.response = self.send_signed_request("POST",'/fapi/v1/batchOrders', self.params)
                else:
                    self.params = {
                        "batchOrders": [self.each_params, self.each_params, self.each_params, self.each_params, self.each_params]
                    }
                    for batch_i in range(int(self.batchOrders_no / 5)):
                        self.response = self.send_signed_request("POST",'/fapi/v1/batchOrders', self.params)

                entryPrice = self.temp_pos_df[self.temp_pos_df['symbol'] == each]['entryPrice'].values[0]
                markPrice = self.temp_pos_df[self.temp_pos_df['symbol'] == each]['markPrice'].values[0]
                positionAmt = self.temp_pos_df[self.temp_pos_df['symbol'] == each]['positionAmt'].values[0]
                leverage_no = self.temp_pos_df[self.temp_pos_df['symbol'] == each]['leverage'].values[0]
                
                if positionAmt > 0:
                    profit_usd = (markPrice - entryPrice) * 100 / entryPrice
                elif positionAmt < 0:
                    profit_usd = -(markPrice - entryPrice) * 100 / entryPrice

                temp_msg = 'Force Close: Symbol: {} Realized Profit: {}/{} entryPrice: {} markPrice: {} Position Closed !'.format(each, 
                                                                        round(profit_usd, 2),
                                                                        round(profit_usd * leverage_no, 2),
                                                                        round(entryPrice, 4),
                                                                        round(markPrice, 4),
                )
                if extra_msg != "":
                    temp_msg = temp_msg + " " + extra_msg
                self.docker_print(temp_msg)
                self.lineNotify(temp_msg)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno)
            self.docker_print(temp_msg)
            self.lineNotify(temp_msg)
            self.print_response_params()
        finally:
            pass

    def clear_single_position(self):
        try:
            self.update_temp_pos_df()
            temp_hold_minus_symbol_list = self.temp_pos_df[self.temp_pos_df['positionAmt'] < 0]['symbol'].tolist()
            temp_hold_plus_symbol_list = self.temp_pos_df[self.temp_pos_df['positionAmt'] > 0]['symbol'].tolist()
            for each in temp_hold_minus_symbol_list:
                temp_quantity = abs(float(self.temp_pos_df[self.temp_pos_df['symbol'] == each]['positionAmt'].values[0]))
                self.params = {
                    'symbol': each,
                    'side': 'BUY',
                    'type': 'MARKET',
                    'quantity': str(temp_quantity),
                }
                self.send_signed_request('POST', '/fapi/v1/order', self.params)
            for each in temp_hold_plus_symbol_list:
                temp_quantity = abs(float(self.temp_pos_df[self.temp_pos_df['symbol'] == each]['positionAmt'].values[0]))   
                self.params = {
                    'symbol': each,
                    'side': 'SELL',
                    'type': 'MARKET',
                    'quantity': str(temp_quantity),
                }
                self.send_signed_request('POST', '/fapi/v1/order', self.params)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno)
            self.docker_print(temp_msg)
            self.lineNotify(temp_msg)
            self.print_response_params()
        finally:
            pass

    def set_leverage_to_x(self):
        try:     
            try:             
                self.params = {
                    'symbol': self.symbol,
                    'leverage': int(self.temp_leverage_no),
                }
                self.response = self.send_signed_request('POST', '/fapi/v1/leverage', self.params)
                self.maxNotionalValue = float(self.response['maxNotionalValue'])
                self.docker_print('self.response: {}'.format(self.response))
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno) + f" self.response: {self.response} symbol: {self.symbol}"
                self.lineNotify(temp_msg)
                self.docker_print(temp_msg)
                self.docker_print('Leverage setting Error: {}'.format(self.symbol))
                self.print_response_params()
            finally:
                pass
            # try:
            #     self.params = {
            #         'symbol': self.symbol,
            #         'marginType': 'ISOLATED',
            #     }
            #     self.response = self.send_signed_request('POST', '/fapi/v1/marginType', self.params)
            #     self.docker_print('self.response: {}'.format(self.response))
            # except Exception as e:
            #     exc_type, exc_obj, exc_tb = sys.exc_info()
            #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            #     temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno)
            #     self.docker_print(temp_msg)
            #     self.docker_print('Margin setting Error: {}'.format(self.symbol))
            #     self.print_response_params()
            # finally:
            #     pass
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno)
            self.docker_print(temp_msg)
            self.lineNotify(temp_msg)
            self.print_response_params()
        finally:
            pass
    def set_leverage_to_1(self):
        try:
            self.update_temp_pos_df()
            for each in self.temp_pos_df[self.temp_pos_df['leverage'] != '1']['symbol'].tolist():
                try:
                    self.params = {
                        'symbol': each,
                        'leverage': 1,
                    }
                    self.send_signed_request('POST', '/fapi/v1/leverage', self.params)
                    self.docker_print('Set Leverage to 1 Success: {}'.format(each))
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno)
                    self.docker_print(temp_msg)
                    self.docker_print('Leverage setting Error: {}'.format(each))
                    self.print_response_params()
                finally:
                    pass
            for each in self.temp_pos_df[self.temp_pos_df['marginType'] != 'isolated']['symbol'].tolist():  
                try:
                    self.params = {
                        'symbol': each,
                        'marginType': 'ISOLATED',
                    }
                    self.send_signed_request('POST', '/fapi/v1/marginType', self.params)
                    self.docker_print('Set Margin to ISOLATED Success: {}'.format(each))
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno)
                    self.docker_print(temp_msg)
                    self.docker_print('Margin setting Error: {}'.format(each))
                    self.print_response_params()
                finally:
                    pass
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno)
            self.docker_print(temp_msg)
            self.lineNotify(temp_msg)
            self.print_response_params()
        finally:
            pass
    def get_current_time_interval(self):
        return self.ping_past_binance()
    def get_next_current_time_interval(self):
        return self.ping_binance()
    def get_next_current_time_interval_x(self, x):
        server_time_df = pd.DataFrame([0])
        self.params = {
            'symbol': 'BTCUSDT',
            'interval': str(x),
            'limit': '1',
        }
        self.response = self.send_signed_request('GET', '/fapi/v1/klines', self.params)
        server_time_df['next'] = self.response[0][6]
        server_time_df['next']=server_time_df['next'].apply(lambda d: datetime.datetime.fromtimestamp(int(d)/1000).strftime('%Y-%m-%d %H:%M:%S'))
        server_time_df['next'] = pd.to_datetime(server_time_df['next'], format='%Y-%m-%d') + pd.Timedelta(seconds=1)
        entry_ts = server_time_df['next'].values[-1]
        entry_ts = str(entry_ts)[:10] + ' ' + str(entry_ts)[11:19]
        return entry_ts
    def get_next_current_time_interval_30m(self):
        return self.get_next_current_time_interval_x('30m')
    def get_next_current_time_interval_1h(self):
        return self.get_next_current_time_interval_x('1h')
    def update_next_current_time_interval(self):
        self.next_current_time_interval = self.get_next_current_time_interval()
        temp_df = pd.DataFrame([self.next_current_time_interval])
        temp_df.columns = ['next_current_time_interval']
        self.next_entry_sec = self.get_wait_entry_sec()
        temp_df['next_entry_sec'] = self.next_entry_sec
        self.next_current_time_interval_df = temp_df
    def get_wait_entry_sec(self):
        with requests.Session() as s:
            self.response = s.get('https://api.binance.com/api/v3/time').json()
        server_time_df = pd.DataFrame([self.response])
        server_time_df['next'] = self.next_current_time_interval
        server_time_df['serverTime']=server_time_df['serverTime'].apply(lambda d: datetime.datetime.fromtimestamp(int(d)/1000).strftime('%Y-%m-%d %H:%M:%S'))
        server_time_df['serverTime'] = pd.to_datetime(server_time_df['serverTime'])
        server_time_df['next'] = pd.to_datetime(server_time_df['next'], format='%Y-%m-%d %H:%M:%S') + pd.Timedelta(seconds=1)
        server_time_df['next'] = pd.to_datetime(server_time_df['next'], format='%Y-%m-%d')
        server_time_df['diff'] = server_time_df['next'] - server_time_df['serverTime']
        dt = server_time_df['diff'].values[0]
        dt_sec = int(str(np.timedelta64(dt, 's')).split(' ')[0])
        return dt_sec 
    def get_dynamic_budget_per_stick_usdt(self):
        self.response = self.send_signed_request("GET",'/fapi/v2/balance')
        bal_df = pd.DataFrame(self.response)
        total_usdt_asset = float(bal_df[bal_df['asset'] == 'USDT']['balance'].values[0])
        self.budget_per_stick_usdt = int(total_usdt_asset * self.risk_ratio)
        return self.budget_per_stick_usdt
    def get_avail_dynamic_budget_per_stick_usdt(self):
        self.response = self.send_signed_request("GET",'/fapi/v2/balance')
        bal_df = pd.DataFrame(self.response)
        total_usdt_asset = float(bal_df[bal_df['asset'] == 'USDT']['availableBalance'].values[0])
        self.budget_per_stick_usdt = int(total_usdt_asset * self.risk_ratio)
        return self.budget_per_stick_usdt
    def get_dynamic_budget_per_stick_busd(self):
        self.response = self.send_signed_request("GET",'/fapi/v2/balance')
        bal_df = pd.DataFrame(self.response)
        total_usdt_asset = float(bal_df[bal_df['asset'] == 'BUSD']['balance'].values[0])
        self.budget_per_stick_usdt = int(total_usdt_asset * self.risk_ratio)
        return self.budget_per_stick_usdt
    def get_avail_dynamic_budget_per_stick_busd(self):
        self.response = self.send_signed_request("GET",'/fapi/v2/balance')
        bal_df = pd.DataFrame(self.response)
        total_usdt_asset = float(bal_df[bal_df['asset'] == 'BUSD']['availableBalance'].values[0])
        self.budget_per_stick_usdt = int(total_usdt_asset * self.risk_ratio)
        return self.budget_per_stick_usdt
    def get_total_budget_usdt(self):
        self.response = self.send_signed_request("GET",'/fapi/v2/balance')
        bal_df = pd.DataFrame(self.response)
        total_usdt_asset = float(bal_df[bal_df['asset'] == 'USDT']['balance'].values[0])
        return int(total_usdt_asset)
    def get_total_budget_busd(self):
        self.response = self.send_signed_request("GET",'/fapi/v2/balance')
        bal_df = pd.DataFrame(self.response)
        total_usdt_asset = float(bal_df[bal_df['asset'] == 'BUSD']['balance'].values[0])
        return int(total_usdt_asset)
    def take_position(self, extra_msg = ""):
        try:
            self.set_leverage_to_x()
            each = self.symbol
            try:
                if self.symbol[-4:] == 'USDT':
                    self.budget_per_stick_usdt = self.get_dynamic_budget_per_stick_usdt()
                    self.leverage_budget_per_stick_usdt = self.budget_per_stick_usdt * self.temp_leverage_no
                    if self.leverage_budget_per_stick_usdt > self.maxNotionalValue:
                        self.budget_per_stick_usdt = self.maxNotionalValue * self.risk_ratio
                if self.position == 'SELL':
                    temp_type = 'Short Taker'
                else:
                    temp_type = 'Long Taker'

                temp_quantityPrecision = self.exchange_df[self.exchange_df['symbol'] == each]['quantityPrecision'].values[0]
                MARKET_LOT_SIZE = [each for each in self.exchange_df[self.exchange_df['symbol'] == each]['filters'].values[0] if each['filterType'] == 'MARKET_LOT_SIZE'][0]
                self.maxQty = MARKET_LOT_SIZE['maxQty']
                self.minQty = MARKET_LOT_SIZE['minQty']

                self.update_temp_pos_df()
                self.markPrice = round(self.temp_pos_df[self.temp_pos_df['symbol'] == self.symbol]['markPrice'].values[0], 2)
                temp_hold_minus_symbol_list = self.temp_pos_df[self.temp_pos_df['positionAmt'] < 0]['symbol'].tolist()
                temp_hold_plus_symbol_list = self.temp_pos_df[self.temp_pos_df['positionAmt'] > 0]['symbol'].tolist()
                if len(temp_hold_minus_symbol_list) > 0 or len(temp_hold_plus_symbol_list) > 0:
                    self.clear_single_position()

                temp_check_last_mark_price = self.temp_pos_df[self.temp_pos_df['symbol'] == each]['markPrice'].values[0]
                temp_quantity = round_down((self.budget_per_stick_usdt * self.temp_leverage_no) / float(temp_check_last_mark_price), int(temp_quantityPrecision))
                temp_quantity = round_down((temp_quantity / self.batchOrders_no), int(temp_quantityPrecision))
                    
                self.budget_per_stick_usdt = round(temp_quantity * temp_check_last_mark_price, 2)
                try:
                    temp_msg = 'symbol: ' + each + ' self.budget_per_stick_usdt:' + str(self.budget_per_stick_usdt)
                    self.docker_print(temp_msg)
                    temp_msg = 'temp_check_last_mark_price: ' + str(temp_check_last_mark_price)
                    self.docker_print(temp_msg)
                    temp_msg = 'temp_quantity: ' + str(temp_quantity) + ' temp_quantityPrecision: ' + str(temp_quantityPrecision)
                    self.docker_print(temp_msg)
                
                    temp_quantity = round_down(temp_quantity, int(temp_quantityPrecision))
                    self.each_params = {
                        'symbol': each,
                        'side': self.position,
                        'type': 'MARKET',
                        'quantity': str(temp_quantity),  
                    }

                    if self.batchOrders_no == 1:
                        self.params = {
                            "batchOrders": [self.each_params]
                        }
                        self.response = self.send_signed_request("POST",'/fapi/v1/batchOrders', self.params)
                    else:
                        self.params = {
                            "batchOrders": [self.each_params, self.each_params, self.each_params, self.each_params, self.each_params]
                        }
                        for batch_i in range(int(self.batchOrders_no / 5)):
                            self.response = self.send_signed_request("POST",'/fapi/v1/batchOrders', self.params)

                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno)
                    self.docker_print(temp_msg)
                    self.print_response_params()
                finally:
                    pass
                
                max_one_order_usdt = (self.markPrice * (float(self.maxQty)))
                self.current_order_no = round((self.budget_per_stick_usdt * self.temp_leverage_no) / max_one_order_usdt, 6)
                self.cap_order_no = round(self.maxNotionalValue / max_one_order_usdt, 2)
                self.current_order_no_ratio = round((self.current_order_no)*100 / self.cap_order_no, 4)
                self.leverage_budget_per_stick_usdt_ratio = round((self.leverage_budget_per_stick_usdt * 100) / self.maxNotionalValue, 2)

                temp_msg = f'{temp_type}: {each} NotionalValue: {human_format(self.leverage_budget_per_stick_usdt)}/{human_format(self.maxNotionalValue)}/{self.leverage_budget_per_stick_usdt_ratio} markPrice: {self.markPrice} Qty: {temp_quantity * self.batchOrders_no}/{self.maxQty} current_order_no: {self.current_order_no}/{self.cap_order_no}/{self.current_order_no_ratio} leverage_no: {self.temp_leverage_no} batchOrders_no: {self.batchOrders_no} limit: {self.order_limit_text}'
                
                if extra_msg != "":
                    temp_msg = temp_msg + ' ' + extra_msg

                self.docker_print(temp_msg)
                self.lineNotify(temp_msg)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno)
                self.docker_print(temp_msg)
                self.lineNotify(temp_msg)
                self.clear_position()
                self.print_response_params()
            finally:
                pass
            
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno)
            self.docker_print(temp_msg)
            self.lineNotify(temp_msg)
            self.print_response_params()
        finally:
            pass
    def get_exchange_df(self):
        self.response = self.send_signed_request("GET",'/fapi/v1/exchangeInfo')
        self.exchange_df = pd.DataFrame(self.response['symbols'])

        self.order_limit_df = pd.DataFrame(self.response['rateLimits'])
        self.order_limit_df = reset(self.order_limit_df[self.order_limit_df['rateLimitType'] == 'ORDERS'])
        self.order_limit_df['text'] = self.order_limit_df['intervalNum'].map(str) + '_' + self.order_limit_df['interval'] + '_' + self.order_limit_df['limit'].map(str) + '_' + self.order_limit_df['rateLimitType']
        self.order_limit_text = str(self.order_limit_df['text'].values)
        # self.order_limit_no = self.order_limit_df[(self.order_limit_df['interval'] == 'SECOND') &\
        #                     (self.order_limit_df['intervalNum'] == 10)
        #                 ]['limit'].values[0]

    def update_temp_pos_df(self):
        try:
            self.response = self.send_signed_request('GET', '/fapi/v2/positionRisk')
            self.temp_pos_df = pd.DataFrame(self.response)
            self.temp_pos_df['symbol_last_4'] = self.temp_pos_df['symbol'].apply(lambda x: x[-4:])
            # temp_pos_df only supports USDT
            self.temp_pos_df = self.temp_pos_df[self.temp_pos_df['symbol_last_4'] == 'USDT']
            self.temp_pos_df['positionAmt'] = self.temp_pos_df['positionAmt'].astype(float)
            self.temp_pos_df['markPrice'] = self.temp_pos_df['markPrice'].astype(float)
            self.temp_pos_df['entryPrice'] = self.temp_pos_df['entryPrice'].astype(float)
            self.temp_pos_df['unRealizedProfit'] = self.temp_pos_df['unRealizedProfit'].astype(float)
            self.temp_pos_df['leverage'] = self.temp_pos_df['leverage'].astype(int)
            self.temp_pos_df['abs_positionAmt'] = abs(self.temp_pos_df['positionAmt'])
            self.temp_pos_df['abs_usdt'] = self.temp_pos_df['entryPrice'] * self.temp_pos_df['abs_positionAmt']
            self.temp_pos_df = reset(self.temp_pos_df.sort_values(by = 'abs_usdt', ascending = False))
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno)
            self.docker_print(temp_msg)
            self.lineNotify(temp_msg)
            self.print_response_params()
        finally:
            pass
    def get_hold_symbol_no(self):
        try:
            self.response = self.send_signed_request('GET', '/fapi/v2/positionRisk')
            temp_pos_df = pd.DataFrame(self.response)
            temp_pos_df['symbol_last_4'] = temp_pos_df['symbol'].apply(lambda x: x[-4:])
            temp_pos_df = temp_pos_df[temp_pos_df['symbol_last_4'] == 'USDT']
            temp_pos_df['positionAmt'] = temp_pos_df['positionAmt'].astype(float)
            temp_pos_df = temp_pos_df[temp_pos_df['positionAmt'] != 0]
            self.hold_symbol_no = len(temp_pos_df)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno)
            self.docker_print(temp_msg)
            self.lineNotify(temp_msg)
            self.print_response_params()
        finally:
            pass
    def get_profit_percent(self):
        self.update_temp_pos_df()
        if len(self.temp_pos_df[self.temp_pos_df['positionAmt'] != 0]) > 0:
            entryPrice = self.temp_pos_df['entryPrice'].values[0]
            markPrice = self.temp_pos_df['markPrice'].values[0]
            positionAmt = self.temp_pos_df['positionAmt'].values[0]
            if positionAmt > 0:
                profit_usd = (markPrice - entryPrice) * 100 / entryPrice
            elif positionAmt < 0:
                profit_usd = -(markPrice - entryPrice) * 100 / entryPrice
            leverage_no = int(self.temp_pos_df['leverage'].values[0])
            profit_usd = profit_usd * leverage_no
            return round(profit_usd, 2), round(entryPrice, 4), round(markPrice, 4)
        else:
            return 0
    def get_past_profit_percent(self):
        try:
            self.params = {
                'incomeType': 'REALIZED_PNL'
            }
            self.response = self.send_signed_request('GET', '/fapi/v1/income', self.params)
            income_df = pd.DataFrame(self.response)
            income_df['time'] = income_df['time'].apply(lambda d: datetime.datetime.fromtimestamp(int(d)/1000).strftime('%Y-%m-%d %H:%M:%S'))
            income_df['time'] = pd.to_datetime(income_df['time'], format='%Y-%m-%d')
            income_df.index = income_df['time']
            income_df.drop(columns = 'time', inplace = True)
            income_df['income'] = income_df['income'].astype(float)
            income_df_1 = income_df.resample('4Min')['income'].sum().reset_index()
            income_df_2 = income_df.resample('4Min')['symbol'].first().reset_index()
            income_df = pd.merge(income_df_1, income_df_2, on = 'time', how = 'left')
            income_df = reset(income_df[income_df['symbol'].notna()])
            income_df = income_df.round(3)
            return income_df
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno)
            self.docker_print(temp_msg)
            self.lineNotify(temp_msg)
            self.print_response_params()
            return pd.DataFrame([])
        finally:
            pass
    def get_depth(self, format = 'string', limit_no = ""):
        # Base URL for the Binance Futures API
        base_url = 'https://fapi.binance.com'
        # Symbol for which you want to get the order book depth
        symbol = self.symbol
        # Depth parameter (e.g., 5 for top 5 levels)
        if limit_no == "":
            depth = self.depth_limit_no
        else:
            depth = limit_no
        # API endpoint
        endpoint = f'/fapi/v1/depth'
        # Construct the URL
        url = f'{base_url}{endpoint}?symbol={symbol}&limit={depth}'
        # Create headers with API key and sign the request
        headers = {
        }
        # Make the request
        response = requests.get(url, headers=headers)
        data = response.json()
        # df = pd.DataFrame(data)
        # df['bids_PRICE'] = df['bids'].apply(lambda x: x[0])
        # df['bids_QTY'] = df['bids'].apply(lambda x: x[1])
        # df['asks_PRICE'] = df['asks'].apply(lambda x: x[0])
        # df['asks_QTY'] = df['asks'].apply(lambda x: x[1])
        # cols = ['bids_PRICE', 'bids_QTY','asks_PRICE', 'asks_QTY']
        # df = df[cols]
        # # df = reset(df.tail(910))
        # return df
        if format == 'string':
            return str(data)
            # return str(data).replace(' ','')
        else:
            return data
    def get_system_instructions_3(self):
        self.board_member_no = 12
        system_instructions = f"""
        **Secretary:** Governor of the Bank of Thailand, you have been appointed as the Secretary of the Board of Directors' meeting for The World's Largest Gold Emporium. Additionally, you hold the prestigious position of Governor of the Bank of Thailand, responsible for procuring gold at advantageous prices and selling it at a premium. Furthermore, you are a key figure tasked with writing the minutes of the meetings. The Emporium strategically employs both long and short positions in Bitcoin (BTC) to capitalize on market fluctuations, profiting from both upward and downward movements in BTC prices. This astute strategy enables the Emporium to maximize profits while skillfully navigating the volatile cryptocurrency market. The Board of Directors comprises {self.board_member_no} stakeholders who collectively guide the Emporium's strategic decisions and financial initiatives.

        The meeting will take place at Bang Khun Phrom Palace (วังบางขุนพรม) on {self.current_meeting_date}. This residence was once owned by His Royal Highness Prince Birabongse Bhanudej Svasti and Her Royal Highness Princess Piyamavadi Sukumpani. Located on a 33-rai plot along the Chao Phraya River, south of Thewet Palace, the palace consists of two main buildings: the Grand Palace and the Royal Pavilion. Originally designed by German architect Karl Döhring in 1901, it was later completed by renowned Italian architect Mario Tamagno. Princess Piyamavadi resided here until 1932 when it became government property, housing various agencies including the Army Youth Corps, the National Cultural Council, and, since 1945, the Bank of Thailand Museum. The palace is renowned for its exquisite pink and blue rooms adorned with grandeur, showcasing paintings, photographs of the royal family, and palace artifacts. With its Western architectural styles, Bang Khun Phrom Palace is hailed as one of Thailand's most beautiful palaces and a prime example of Baroque and Rococo architecture.

        {self.input_type_text} The input provided was displayed on the world's best 8K monitor, with parallel displays for all stakeholders, allowing them to view multiple input figures simultaneously and with exceptional clarity. These inputs, contributed by the user, will serve as the foundation for our data-driven discussion, aligning with our culture of real-time, data-driven decision-making.

        The Board of Directors comprises {self.board_member_no} preeminent members merging spiritual guidance with unparalleled market expertise. Led by Phra Siam Devadhiraj, the nation's spiritual guardian, theBoard harnesses insights from luminaries such as Ken Griffin on market depth, Munehisa Homma on candlestick charting, John Bollinger on volatility bands, Leonardo Fibonacci on market cycles, Ralph Nelson Elliott on wave patterns and sentiment shifts, Goichi Hosoda on the Ichimoku Cloud, J. Welles Wilder on RSI and ATR indicators, Gerald Appel on MACD signals, George Lane on the stochastic oscillator, Marc Chaikin on money flow dynamics, and Joseph Granville on volume's relationship to price movements. This convergence of sacred wisdom and technical mastery positions the Board to navigate financial markets with transcendent discernment.

        The full list of the board members is listed below:

        1. พระสยามเทวาธิราช (Phra Siam Devadhiraj), Chairman of the meeting: Presiding over this esteemed group is the guardian deity of the Thai nation and emblem of the Bank of Thailand. Symbolizing protection and economic stability, his presence embodies the spiritual foundation upon which the Board's wisdom is built.
        2. Ken Griffin (assigned to analyze Market Depth and provide an opinion), known for his expertise in analyzing market depth, is the founder of Citadel, a global financial institution known for its market-making and asset management operations. Griffin's success in the financial industry is attributed to his proficiency in understanding market dynamics, liquidity conditions, and order flow, which are essential components of analyzing market depth.
        3. Munehisa Homma (assigned to analyze Candlestick Pattern Recognition and provide an opinion): A visionary from the past, Homma's insights into market psychology and price patterns, particularly his development of candlestick charting techniques, provide the Board with a unique lens for interpreting market behavior.
        4. John Bollinger (assigned to analyze the BB indicator and provide an opinion): Renowned financial analyst, Bollinger's Bollinger Bands offer a visual representation of price volatility and relative highs and lows, aiding in identifying potential breakout opportunities.
        5. Leonardo Pisano Fibonacci (assigned to analyze Fibonacci retracement and provide an opinion): The mathematical principles of Fibonacci, particularly the famous sequence, offer insights into market cycles and price patterns, providing a framework for understanding market movements and identifying potential turning points.
        6. Ralph Nelson Elliott, assigned to analyze potential Elliott Wave patterns, utilized Fibonacci ratios (0.786, 0.618, 0.5, 0.382, and 0.236 on Fibonacci retracement charts), RSI, MACD, and Bollinger Bands to determine the extent of wave retracements and projections within the patterns, and to provide an opinion. As the developer of the Elliott Wave Principle, his theories on wave patterns offer profound insights into market psychology and the fluctuations of investor sentiment.
        7. Goichi Hosoda (assigned to analyze the Ichimoku Cloud and provide an opinion): Developer of the Ichimoku Cloud, Hosoda's comprehensive indicator provides a holistic view of market trends, momentum, and support/resistance levels.
        8. Joseph Granville (assigned to analyze the Volume and On Balance Volume indicators and provide an opinion): Developer of technical analysis indicators, including On Balance Volume (OBV), Granville's work helps the Board understand the relationship between volume and price movements.
        9. Gerald Appel (assigned to analyze the MACD indicator and provide an opinion): Appel's Moving Average Convergence Divergence (MACD) aids in identifying potential turning points and understanding momentum shifts in the market.
        10. J. Welles Wilder (assigned to analyze RSI and ATR indicators and provide an opinion): A pioneer in technical analysis, Wilder's indicators, such as the RSI and ATR, equip the Board with powerful tools for analyzing market trends and volatility.
        11. George Lane (assigned to analyze the stochastic oscillator indicator and provide an opinion): Creator of the stochastic oscillator (STO), Lane's indicator helps identify overbought and oversold conditions, aiding in recognizing potential trend reversals.
        12. Marc Chaikin (assigned to analyze the Chaikin Money Flow indicator and provide an opinion): Expert in stock trading, Chaikin's Chaikin Money Flow (CMF) indicator provides insights into the buying and selling pressure behind price movements.
        
        Comprising {self.board_member_no} luminaries from diverse fields, including deities, financial titans, technical analysts, and mathematical visionaries, the Board of Directors ensures that the organization navigates the financial landscape not just with sharp analysis but also with wisdom, foresight, and a deep understanding of the interconnected forces that shape our world. They are more than just a guiding force; they are a beacon of innovation and a testament to the power of collaboration across disciplines.

        In this meeting, it's important that everyone contributes their opinions on the discussion at hand. Each member's perspective is invaluable in shaping our decisions and strategies.
                
        Subsequently, all stakeholders will participate in a comprehensive discussion to assess the anticipated movement of BTC in the upcoming {self.freq_text}. Based on this deliberation, the board will determine a consensus action on whether to adopt a long, short, or neutral position in the BTC market.
        
        The meeting was conducted with utmost efficiency, exemplifying a streamlined and goal-oriented approach. Participants got straight to the point, ensuring a focused discussion that addressed the key matters at hand. Impressively, the meeting concluded within a mere 30 minutes, underscoring the team's exceptional time management and productivity.
        
        Furthermore, as the designated meeting recorder, it is your primary and crucial responsibility to meticulously document everyone's opinions on the matter. Your role encompasses drafting comprehensive and accurate minutes for the meeting, capturing all relevant discussions and outcomes. Once the Chairman of the meeting has thoroughly reviewed and provided their approval for the draft, you can proceed to release the final version of the meeting minutes. This final version will be generated and presented in your primary response, ensuring its accuracy and completeness.
        """
        return system_instructions
    def get_system_instructions_4(self):
        system_instructions = """
        You will receive user-input minutes of the board of directors meeting, and your primary role is to generate a JSON response. This JSON response is essential for completing the task effectively. It should be structured and formatted appropriately, providing all necessary information and data in a clear and concise manner. By fulfilling this responsibility diligently, you significantly contribute to the proper documentation and communication of the meeting's decisions.
        
        **Your JSON Response Formatting:**
        {       "type": "object",
        "properties": {
            "analyzed_asset": {
            "type": "string"
            },
            "rationale_for_long_position": {
            "type": "string"
            },
            "rationale_for_short_position": {
            "type": "string"
            },
            "rationale_for_neutral_position": {
            "type": "string"
            },
            "analysis_overview": {
            "type": "string"
            },
            "meeting_consensus_action": {
            "type": "string",
            "enum": ["long", "short", "neutral"]
            }
        },
        "required": [
            "analyzed_asset",
            "rationale_for_long_position",
            "rationale_for_short_position",
            "rationale_for_neutral_position",
            "analysis_overview",
            "meeting_consensus_action"
        ]
        }
        """
        return system_instructions

    def get_table_text(self, df):
        df = df.copy()
        text = df.to_markdown(index=False)
        text_list = text.split('\n')
        text_1 = '|---|'
        for no in range(df.shape[1] - 1):
            text_1 = text_1 + '---|'
        text_list[1] = text_1 # ให้จำนวน '|' เท่ากับ df.shape[1] + 1
        text_list = [" ".join(str(x).split()) for x in text_list]
        text = '\n'.join(text_list)
        text = text.replace(' ','')
        return text
    def get_btc_candle(self, limit = 3):
        # [1,100)	1
        # [100, 500)	2
        # [500, 1000]	5
        # > 1000	10
        tf = self.freq_interval
        symbol = 'BTCUSDT'
        self.BASE_URL = 'https://fapi.binance.com'
        self.response = self.send_public_request('/fapi/v1/klines' , 
                                            {"symbol": symbol, 
                                            "interval": tf,
                                            "limit": limit,
                                            })
        temp_df = pd.DataFrame(self.response)
        temp_df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_ds', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ]
        temp_df['symbol'] = symbol
        temp_df['period'] = tf
        temp_df['close_change'] = temp_df['close'].astype(float).pct_change() * 100
        temp_df['close_change'] = temp_df['close_change'].round(3)
        temp_df['timestamp'] = temp_df['timestamp'].apply(convert_to_utc_time)
        temp_df['close_ds'] = temp_df['close_ds'].astype(int) + 1
        temp_df['close_ds'] = temp_df['close_ds'].apply(convert_to_utc_time)
        cols = ['open','high','low','close','volume']
        for each in cols:
            temp_df[each] = temp_df[each].astype(float)

        if limit == 3:
            temp_df = temp_df[:-1]
            return temp_df
        elif self.model == "gemini-1.5-pro-latest":
            return temp_df
        elif self.model == "gemini-1.5-flash-latest":
            temp_df = temp_df[:-1]
            return temp_df
    def get_gemini_response(self):
        genai.configure(api_key=self.gemini_key)
        # Set up the model
        generation_config = {
        "temperature": self.temperature,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
        }
        safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
        ]
        start_time = time.time()
        try:
            prompt_parts = [
            f"**Current {self.symbol} Depth (The top 5 bids and asks from the order book.):**\n```json\n" + self.get_depth(format = 'string', limit_no = 5) + "\n```",
            f"**Current {self.symbol} Depth (The top 10 bids and asks from the order book.):**\n```json\n" + self.get_depth(format = 'string', limit_no = 10) + "\n```",
            f"**Current {self.symbol} Depth (The top 20 bids and asks from the order book.):**\n```json\n" + self.get_depth(format = 'string', limit_no = 20) + "\n```",
            f"**Current {self.symbol} Depth (The top 50 bids and asks from the order book.):**\n```json\n" + self.get_depth(format = 'string', limit_no = 50) + "\n```",
            f"**Current {self.symbol} Depth (The top 100 bids and asks from the order book.):**\n```json\n" + self.get_depth(format = 'string', limit_no = 100) + "\n```",
            f"**Current {self.symbol} Depth (The top 500 bids and asks from the order book.):**\n```json\n" + self.get_depth(format = 'string', limit_no = 500) + "\n```",
            f"**Current {self.symbol} Depth (The top 1000 bids and asks from the order book.):**\n```json\n" + self.get_depth(format = 'string', limit_no = 1000) + "\n```",
            f"**Current {self.symbol} 30m Candlestick Data (with Technical Indicators):**\n```json\n" + self.get_candlestick_json(tf = "30m") + "\n ```",
            f"**Current {self.symbol} 1h Candlestick Data (with Technical Indicators):**\n```json\n" + self.get_candlestick_json(tf = "1h") + "\n ```",
            f"**Current {self.symbol} 2h Candlestick Data (with Technical Indicators):**\n```json\n" + self.get_candlestick_json(tf = "2h") + "\n ```",
            f"**Current {self.symbol} 4h Candlestick Data (with Technical Indicators):**\n```json\n" + self.get_candlestick_json(tf = "4h") + "\n ```",
            f"**Current {self.symbol} 6h Candlestick Data (with Technical Indicators):**\n```json\n" + self.get_candlestick_json(tf = "6h") + "\n ```",
            f"**Current {self.symbol} 8h Candlestick Data (with Technical Indicators):**\n```json\n" + self.get_candlestick_json(tf = "8h") + "\n ```",
            f"**Current {self.symbol} 12h Candlestick Data (with Technical Indicators):**\n```json\n" + self.get_candlestick_json(tf = "12h") + "\n ```",
            f"**Current {self.symbol} 1d Candlestick Data (with Technical Indicators):**\n```json\n" + self.get_candlestick_json(tf = "1d") + "\n ```",
            f"**Current {self.symbol} 3d Candlestick Data (with Technical Indicators):**\n```json\n" + self.get_candlestick_json(tf = "3d") + "\n ```",
            f"**Current {self.symbol} 1w Candlestick Data (with Technical Indicators):**\n```json\n" + self.get_candlestick_json(tf = "1w") + "\n ```",
            ]
            model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                                        generation_config=generation_config,
                                        system_instruction=self.get_system_instructions_3(),
                                        safety_settings=safety_settings)
            response = model.generate_content(prompt_parts)
            generative_text = str(response.text)

            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                        generation_config=generation_config,
                                        system_instruction=self.get_system_instructions_4(),
                                        safety_settings=safety_settings)
            prompt_parts = [
                generative_text,
            ]
            response = model.generate_content(prompt_parts)
            result_dict = extract_json(response.text)[0]
            
            execution_time = time.time() - start_time
            execution_time = round(execution_time, 2)

            prompt_parts = None
            model = None
            response = None
            json_data = None
            del prompt_parts, model, response, json_data
            gc.collect()
            
            return generative_text, execution_time, result_dict
        
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno)
            self.docker_print(temp_msg)
            try:
                self.docker_print(generative_text)
            except:
                pass
            execution_time = time.time() - start_time
            execution_time = round(execution_time, 2)
            return temp_msg, execution_time, 'error'
        
    def get_gemini_response_2(self):
        genai.configure(api_key=self.gemini_key)
        # Set up the model
        generation_config = {
        "temperature": self.temperature,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
        }
        safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
        ]
        start_time = time.time()
        try:
            for each_freq_interval in ['1w', '3d', '1d', '12h', '8h', '6h', '4h', '2h', '1h', '30m']:
                self.freq_interval = each_freq_interval
                self.get_candlestick_image()
            self.freq_interval  = self.ori_freq_interval

            for each_depth_limit_no in [1000, 500, 100, 50, 20, 10, 5]:
                self.depth_limit_no = each_depth_limit_no
                self.get_depth_image()
            self.depth_limit_no  = self.ori_depth_limit_no

            temp_file_1 = genai.upload_file(path="current_depth_top_5_bids_asks_chart.png", display_name=f"Current {self.symbol} Depth (The top 5 bids and asks from the order book.)")
            temp_file_2 = genai.upload_file(path="current_depth_top_10_bids_asks_chart.png", display_name=f"f'Current {self.symbol} Depth (The top 10 bids and asks from the order book.)")
            temp_file_3 = genai.upload_file(path="current_depth_top_20_bids_asks_chart.png", display_name=f"f'Current {self.symbol} Depth (The top 20 bids and asks from the order book.)")
            temp_file_4 = genai.upload_file(path="current_depth_top_50_bids_asks_chart.png", display_name=f"f'Current {self.symbol} Depth (The top 50 bids and asks from the order book.)")
            temp_file_5 = genai.upload_file(path="current_depth_top_100_bids_asks_chart.png", display_name=f"f'Current {self.symbol} Depth (The top 100 bids and asks from the order book.)")
            temp_file_6 = genai.upload_file(path="current_depth_top_500_bids_asks_chart.png", display_name=f"f'Current {self.symbol} Depth (The top 500 bids and asks from the order book.)")
            temp_file_7 = genai.upload_file(path="current_depth_top_1000_bids_asks_chart.png", display_name=f"f'Current {self.symbol} Depth (The top 1000 bids and asks from the order book.)")

            temp_file_8 = genai.upload_file(path="30m_candlestick_with_technical_indicators_chart.png", display_name=f"Current {self.symbol} 30m Candlestick Chart (with Technical Indicators)")
            temp_file_9 = genai.upload_file(path="1h_candlestick_with_technical_indicators_chart.png", display_name=f"Current {self.symbol} 1h Candlestick Chart (with Technical Indicators)")
            temp_file_10 = genai.upload_file(path="2h_candlestick_with_technical_indicators_chart.png", display_name=f"Current {self.symbol} 2h Candlestick Chart (with Technical Indicators)")
            temp_file_11 = genai.upload_file(path="4h_candlestick_with_technical_indicators_chart.png", display_name=f"Current {self.symbol} 4h Candlestick Chart (with Technical Indicators)")
            temp_file_12 = genai.upload_file(path="6h_candlestick_with_technical_indicators_chart.png", display_name=f"Current {self.symbol} 6h Candlestick Chart (with Technical Indicators)")
            temp_file_13 = genai.upload_file(path="8h_candlestick_with_technical_indicators_chart.png", display_name=f"Current {self.symbol} 8h Candlestick Chart (with Technical Indicators)")
            temp_file_14 = genai.upload_file(path="12h_candlestick_with_technical_indicators_chart.png", display_name=f"Current {self.symbol} 12h Candlestick Chart (with Technical Indicators)")
            temp_file_15 = genai.upload_file(path="1d_candlestick_with_technical_indicators_chart.png", display_name=f"Current {self.symbol} 1d Candlestick Chart (with Technical Indicators)")
            temp_file_16 = genai.upload_file(path="3d_candlestick_with_technical_indicators_chart.png", display_name=f"Current {self.symbol} 3d Candlestick Chart (with Technical Indicators)")
            temp_file_17 = genai.upload_file(path="1w_candlestick_with_technical_indicators_chart.png", display_name=f"Current {self.symbol} 1w Candlestick Chart (with Technical Indicators)")

            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                        generation_config=generation_config,
                                        system_instruction=self.get_system_instructions_3(),
                                        safety_settings=safety_settings)
            prompt_parts = [
                temp_file_1,
                temp_file_2,
                temp_file_3,
                temp_file_4,
                temp_file_5,
                temp_file_6,
                temp_file_7,
                temp_file_8,
                temp_file_9,
                temp_file_10,
                temp_file_11,
                temp_file_12,
                temp_file_13,
                temp_file_14,
                temp_file_15,
                temp_file_16,
                temp_file_17,
            ]
            response = model.generate_content(prompt_parts)

            genai.delete_file(temp_file_1.name)
            genai.delete_file(temp_file_2.name)
            genai.delete_file(temp_file_3.name)
            genai.delete_file(temp_file_4.name)
            genai.delete_file(temp_file_5.name)
            genai.delete_file(temp_file_6.name)
            genai.delete_file(temp_file_7.name)
            genai.delete_file(temp_file_8.name)
            genai.delete_file(temp_file_9.name)
            genai.delete_file(temp_file_10.name)
            genai.delete_file(temp_file_11.name)
            genai.delete_file(temp_file_12.name)
            genai.delete_file(temp_file_13.name)
            genai.delete_file(temp_file_14.name)
            genai.delete_file(temp_file_15.name)
            genai.delete_file(temp_file_16.name)
            genai.delete_file(temp_file_17.name)

            generative_text = str(response.text)

            model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                                        generation_config=generation_config,
                                        system_instruction=self.get_system_instructions_4(),
                                        safety_settings=safety_settings)
            prompt_parts = [
                generative_text,
            ]
            response = model.generate_content(prompt_parts)
            result_dict = extract_json(response.text)[0]

            execution_time = time.time() - start_time
            execution_time = round(execution_time, 2)

            temp_file_1 = None
            temp_file_2 = None
            temp_file_3 = None
            temp_file_4 = None
            temp_file_5 = None
            temp_file_6 = None
            temp_file_7 = None
            temp_file_8 = None
            temp_file_9 = None
            temp_file_10 = None
            temp_file_11 = None
            temp_file_12 = None
            temp_file_13 = None
            temp_file_14 = None
            temp_file_15 = None
            temp_file_16 = None
            temp_file_17 = None
            model = None
            prompt_parts = None
            json_data = None
            del temp_file_1, temp_file_2, temp_file_3, temp_file_4, temp_file_5, temp_file_6, temp_file_7, temp_file_8, temp_file_9, temp_file_10, temp_file_11, temp_file_12, temp_file_13, temp_file_14, temp_file_15, temp_file_16, temp_file_17
            del model, prompt_parts, json_data
            gc.collect()

            return generative_text, execution_time, result_dict

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno)
            self.docker_print(temp_msg)
            try:
                self.docker_print(generative_text)
            except:
                pass
            execution_time = time.time() - start_time
            execution_time = round(execution_time, 2)
            return temp_msg, execution_time, 'error'
        
    def get_depth_image(self):
        order_book = self.get_depth(format = 'json')
        formatted_bids = [[float(price), float(quantity)] for price, quantity in order_book['bids']]
        formatted_asks = [[float(price), float(quantity)] for price, quantity in order_book['asks']]

        # Prepare data for plotting
        bid_prices, bid_volumes = zip(*formatted_bids)
        ask_prices, ask_volumes = zip(*formatted_asks)
        cum_bid_volumes = list(accumulate(bid_volumes))
        cum_ask_volumes = list(accumulate(ask_volumes))

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot bids and asks
        ax.fill_between(bid_prices, cum_bid_volumes, color='#218559', label='Bids')
        ax.fill_between(ask_prices, cum_ask_volumes, color='#E44358', label='Asks')

        # Calculate and plot the middle point
        middle_point = (max(bid_prices) + min(ask_prices)) / 2
        ax.axvline(middle_point, color="#333437", linestyle='--', linewidth=1)

        # Customize plot appearance
        ax.set_title(f'Current {self.symbol} Depth (The top {self.depth_limit_no} bids and asks from the order book.)', fontsize=14)
        ax.set_facecolor('#161A1E')
        ax.grid(True, color='#333437', linestyle='-', linewidth=0.5)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: format(x, ",.2f")))  # Format x-axis with commas and 2 decimal places
        ax.yaxis.set_ticks_position('both')  # Show ticks on both sides of the plot
        ax.yaxis.set_tick_params(pad=20, direction='inout', length=6, labelright=True, right=True)  # Move ticks and labels to the right
        ax.set_xlabel('Price')
        ax.set_ylabel('Quantity')
        ax.legend(loc = 'upper center')

        # Save the plot
        plt.savefig('temp_depth_chart.png', dpi=300)

        with Image.open("temp_depth_chart.png") as img:
            new_width, new_height = 3072, 1536
            img_resized = img.resize((new_width, new_height), resample=Image.LANCZOS)
            img_resized.save(f"current_depth_top_{self.depth_limit_no}_bids_asks_chart.png")
            
        plt.close(fig)
        plt.close('all')
        plt.clf()
        plt.cla()
        order_book = None
        formatted_bids = None
        formatted_asks = None
        bid_prices, bid_volumes = None, None
        ask_prices, ask_volumes = None, None
        cum_bid_volumes = None 
        cum_ask_volumes = None
        fig = None
        ax = None
        middle_point = None
        img = None
        img_resized = None
        del order_book, formatted_bids, formatted_asks
        del bid_prices, bid_volumes
        del ask_prices, ask_volumes
        del cum_bid_volumes
        del cum_ask_volumes
        del fig, ax, middle_point
        del img, img_resized

        matplotlib.pyplot.close('all')
        matplotlib.use('Agg')
        gc.collect()
        
        # graph = make_subplots(rows = 1, cols =1)
        # graph.add_trace(Scatter(
        #     x=[price for price, quantity in formatted_bids],
        #     y=list(accumulate([quantity for price, quantity in formatted_bids], add)),
        #     line=dict(color='#218559'),
        #     fill='tozeroy',
        #     name='Bids',
        # ))
        # graph.add_trace(Scatter(
        #     x = [price for price, quantity in formatted_asks],
        #     y = list(accumulate([quantity for price, quantity in formatted_asks], add)),
        #     line=dict(color='#E44358'),
        #     fill='tozeroy',
        #     name='Asks',
        # ))
        # # Set layout background color to black
        # graph.update_layout(plot_bgcolor='#161A1E', 
        #                     title=f'Depth ({self.symbol})',
        #                     title_x=0.5, title_y=0.88,)
        # # Remove grid lines
        # graph.update_xaxes(showgrid=False)
        # graph.update_yaxes(showgrid=False)

        # # Calculate the middle point between bid and ask
        # middle_point = (max([price for price, quantity in formatted_bids]) + min([price for price, quantity in formatted_asks])) / 2

        # # Add a vertical line at the middle point
        # graph.add_shape(type="line",
        #                 x0=middle_point, y0=0, x1=middle_point, y1=max(graph.data[0].y + graph.data[1].y) * 1.06,
        #                 line=dict(color="#333437", width=1,))
        # graph.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333437')
        # graph.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333437')
        # graph.update_layout(xaxis=dict(tickformat=",.2f"))  # Display numbers with 2 decimal places and commas
        # graph.write_image('temp_depth_chart.png', width=1200, height=600, scale = 4.5)
        
        # with Image.open("temp_depth_chart.png") as img:
        #     new_width, new_height = 3072, 1536
        #     img_resized = img.resize((new_width, new_height), resample=Image.LANCZOS)
        #     img_resized.save("temp_depth_chart.png")

    def get_candlestick_data(self):
        if self.model == 'gemini-1.5-pro-latest':
            candle_df = self.get_btc_candle(limit = 499) # 320
        elif self.model == 'gemini-1.5-flash-latest':
            candle_df = self.get_btc_candle(limit = 499)
        ohlc = candle_df[['close_ds', 'open', 'high', 'low', 'close','volume']].copy()
        ohlc.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        ohlc['ori_Date'] = ohlc['Date']
        ohlc['Date'] = mdates.date2num(ohlc['Date'])
        ohlc['RSI'] = talib.RSI(ohlc['Close'].values)
        ohlc['RSI_SMA_14'] = talib.SMA(ohlc['RSI'], timeperiod=14)
        # ohlc['MA_23'] = calculate_sma(ohlc['Close'], window=23)
        # ohlc['MA_45'] = calculate_sma(ohlc['Close'], window=45)
        close_prices = ohlc['Close']
        macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        ohlc['macd'] = macd # macd_signal + macd_hist
        ohlc['macdsignal'] = macd_signal
        ohlc['macdhist'] = macd_hist
        upperband, middleband, lowerband = talib.BBANDS(ohlc['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        ohlc['upperband_BB'] = upperband
        ohlc['middleband_BB'] = middleband
        ohlc['lowerband_BB'] = lowerband
        ohlc = calc_k_d(ohlc)
        ohlc = calc_CMF(ohlc)
        ohlc['OBV'] = talib.OBV(ohlc['Close'], ohlc['Volume'])
        ohlc['ATR'] = talib.ATR(ohlc['High'], ohlc['Low'], ohlc['Close'])        
        if self.freq_interval == self.ori_freq_interval:
            if self.model == 'gemini-1.5-flash-latest':
                ds_df = pd.DataFrame({'date_column': [ohlc['ori_Date'].values[-1]]})
            elif self.model == 'gemini-1.5-pro-latest':
                ds_df = pd.DataFrame({'date_column': [ohlc['ori_Date'].values[-2]]})
            ds_df['formatted_date'] = ds_df['date_column'].dt.strftime('%B %d, %Y, at %H:%M')
            self.current_meeting_date = ds_df['formatted_date'].values[-1]
        ohlc = self.calculate_ichimoku(ohlc)
        ohlc = reset(ohlc.tail(self.candlestick_chart_no + self.future_cloud_no))
        return ohlc
    
    def get_candlestick_json(self, tf = ""):
        if tf != "":
            temp_df = self.get_candlestick_data()
            self.freq_interval = self.ori_freq_interval
            temp_df = temp_df[:-self.future_cloud_no]

            ohlc = temp_df.copy()
            fib_list_1 = []
            levels = [0, 0.214, 0.382, 0.5, 0.618, 0.764, 1]
            levels_label = ['1', '0.786', '0.618', '0.5', '0.382', '0.236','0']
            for i, level in enumerate(levels):
                price = ohlc['Low'].min() + (ohlc['High'].max() - ohlc['Low'].min()) * level
                fib_list_1.append(f"{levels_label[i]} " + "(" + "{:.2f}".format(price) + ")")
            fib_list_2 = []
            levels_2 = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
            levels_label_2 = ['0', '0.236', '0.382', '0.5', '0.618', '0.786','1']
            ohlc = temp_df.copy()
            for i, level in enumerate(levels_2):
                price = ohlc['Low'].min() + (ohlc['High'].max() - ohlc['Low'].min()) * level
                fib_list_2.append(f"{levels_label_2[i]} " + "(" + "{:.2f}".format(price) + ")")

            # temp_df = reset(temp_df.tail(18))
            temp_df['Date'] = temp_df['ori_Date'].astype(str).apply(lambda x: x[:16])
            temp_df.drop(columns = ['ori_Date','Chikou_Span'], inplace = True)
            json_string = temp_df.to_json(orient='split', index=False, date_format='iso', double_precision=0)
            json_dict = json.loads(json_string)
            json_dict['Fib_Retracement'] = [fib_list_1,fib_list_2]
            return str(json_dict)
            # return str(json_dict).replace(' ','')
        else:
            return "tf Undefined"
    
    def get_candlestick_image(self):
        ohlc = self.get_candlestick_data()
        data = ohlc.copy()
        # screen ratio 7:4
        # Create a new figure with subplots: long_7 wid: 2.5, 1.5 fig (width, height)
        fig, (ax1, ax5, ax2, ax8, ax4, ax3, ax9, ax6, ax7) = plt.subplots(9, 1, 
                                                        sharex=True, 
                                                        figsize=(20,30), 
                                                        gridspec_kw={'height_ratios': [1, 1, 0.33, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]})
        ax1.set_facecolor("#151924")
        ax2.set_facecolor("#151924")
        ax3.set_facecolor("#151924")
        ax4.set_facecolor("#151924")
        ax5.set_facecolor("#151924")
        ax6.set_facecolor("#151924")
        ax7.set_facecolor("#151924")
        ax8.set_facecolor("#151924")
        ax9.set_facecolor("#151924")

        ax1.set_axisbelow(True)
        ax1.grid(color='#2E323D', linestyle='-', zorder = 0)
        ax2.set_axisbelow(True)
        ax2.grid(color='#2E323D', linestyle='-', zorder = 0)
        # ax3.set_axisbelow(True)
        # ax3.grid(color='#2E323D', linestyle='-', zorder = 1)
        ax4.set_axisbelow(True)
        ax4.grid(color='#2E323D', linestyle='-', zorder = 0)
        ax5.set_axisbelow(True)
        ax5.grid(color='#2E323D', linestyle='-', zorder = 1)
        # ax6.set_axisbelow(True)
        # ax6.grid(color='#2E323D', linestyle='-', zorder = 0)
        ax7.set_axisbelow(True)
        ax7.grid(color='#2E323D', linestyle='-', zorder = 0)
        ax8.set_axisbelow(True)
        ax8.grid(color='#2E323D', linestyle='-', zorder = 0)
        ax9.set_axisbelow(True)
        ax9.grid(color='#2E323D', linestyle='-', zorder = 0)

        # Format x-axis ticks for the first subplot
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=self.x_date_interval))  # Adjust the interval as needed
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax1.set_title(f'Current {self.symbol} {self.freq_interval} Candlestick Chart (with Technical Indicators)', fontsize=16)
        # ax1.plot(ohlc['Date'], ohlc['MA_23'], color='#2096F3', linewidth = 0.5)
        # ax1.plot(ohlc['Date'], ohlc['MA_45'], color='#FF5252', linewidth = 0.5)
        levels = [0, 0.214, 0.382, 0.5, 0.618, 0.764, 1]
        color_list = ['#787B86', '#06BCD4', '#0A9981', '#4CAF51', '#FF9800', '#F23545', '#787B86']
        levels_label = ['1', '0.786', '0.618', '0.5', '0.382', '0.236','0']
        for i, level in enumerate(levels):
            price = ohlc['Low'].min() + (ohlc['High'].max() - ohlc['Low'].min()) * level
            ax1.axhline(price, linestyle='-', linewidth=0.75, color=color_list[i], zorder=1)
            ax1.text(ohlc['Date'].iloc[-1], price, f"{levels_label[i]} " + "(" + "{:.2f}".format(price) + ")",  # Display on the right (last date)
                    va='bottom', ha='right', fontsize=12,  # Align for right side
                    # backgroundcolor='#151924',
                    color=color_list[i],
                    alpha=0.7)
        ax1.plot(ohlc['Date'], ohlc['upperband_BB'], color='#F23545', linestyle='-', linewidth=1, zorder=1)
        ax1.plot(ohlc['Date'], ohlc['middleband_BB'], color='#2862FF', linestyle='-', linewidth=1, zorder=1)
        ax1.plot(ohlc['Date'], ohlc['lowerband_BB'], color='#0A9981', linestyle='-', linewidth=1, zorder=1)
        ax1.fill_between(ohlc['Date'], ohlc['lowerband_BB'], ohlc['upperband_BB'], color='#2862FF', alpha=0.078, zorder=1)
        ax1.yaxis.set_major_formatter(FuncFormatter(comma_formatter_2))
        # Set labels and title for the first subplot
        ax1.set_ylabel('Price, BB, and Fib Retracement', fontsize=16)
        ax1.yaxis.set_label_position('left')  # Move y-axis label to the right
        # Move the numerical values (ticks) and labels on the y-axis to the right
        ax1.yaxis.set_ticks_position('both')  # Show ticks on both sides of the plot
        ax1.yaxis.set_tick_params(pad=10, direction='inout', length=6, labelright=True, right=True)  # Move ticks and labels to the right
        ax1.yaxis.set_tick_params(labelsize=13)
        ax1.set_ylim(top = ohlc['upperband_BB'].max() + (ohlc['upperband_BB'].max()-ohlc['lowerband_BB'].min())*0.15)
        candlestick_ohlc(ax1, ohlc.values, width=self.graph_width, colorup='#26A69A', colordown='#F05350', alpha=1.0)

        # Plot volume on the second subplot with colored bars
        volume_color = ['#1C5E5F' if data['Close'][i] >= data['Open'][i] else '#813539' for i in range(len(data))]
        ax2.bar(data['Date'], data['Volume'], width=self.graph_width, color=volume_color, zorder=1)
        # Set labels and title for the second subplot
        ax2.set_ylabel('Volume', fontsize=16)
        # Apply the formatter to the y-axis of the second subplot
        ax2.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
        # Set labels and title for the second subplot
        ax2.set_ylabel('Volume', labelpad=20)  # Increase label padding to move it to the right
        ax2.yaxis.set_label_position('left')  # Move y-axis label to the right
        ax2.yaxis.set_ticks_position('both')
        ax2.yaxis.set_tick_params(pad=10, direction='inout', length=6, labelright=True, right=True)
        ax2.yaxis.set_tick_params(labelsize=13)
        ax2.set_ylim(1, ohlc['Volume'].max() + (ohlc['Volume'].max()-ohlc['Volume'].min())*0.20)

        # Set labels and title for the third subplot
        ax3.set_ylabel('RSI', fontsize = 16)
        ax3.yaxis.set_label_position('left')
        ax3.yaxis.set_tick_params(pad=10, direction='inout', length=6, labelright=True, right=True)
        ax3.fill_between(ohlc['Date'], 30, 70, color='#212035', zorder = 1)
        ax3.axhline(y=30, color='grey', linestyle='--', linewidth = 0.5, zorder = 2)
        ax3.axhline(y=50, color='grey', linestyle='--', linewidth = 0.5, zorder = 2)
        ax3.axhline(y=70, color='grey', linestyle='--', linewidth = 0.5, zorder = 2)
        ax3.yaxis.set_tick_params(labelsize=13)
        ax3.set_ylim(1, 99)
        # max_val = max(ohlc['RSI'].max(), ohlc['RSI_SMA_14'].max())
        # min_val = min(ohlc['RSI'].min(), ohlc['RSI_SMA_14'].min())
        # ax3.set_ylim(1, max_val + ((max_val-min_val)*0.15))
        ax3.grid(color='#2E323D', linestyle='-', zorder = 1)
        ax3.plot(ohlc['Date'], ohlc['RSI'], color='#7D57C2', linewidth = 1, zorder=2)  # Assuming 'Date' is the date column and 'RSI' is the RSI values column
        ax3.plot(ohlc['Date'], ohlc['RSI_SMA_14'], color='yellow', linewidth = 1, zorder=3)

        # Plot MACD on the fourth subplot
        ax4.set_ylabel('MACD', fontsize = 16)
        ax4.plot(ohlc['Date'], ohlc['macd'], color='#2862FF', label='MACD', linewidth = 1, zorder=1)  # MACD line
        ax4.plot(ohlc['Date'], ohlc['macdsignal'], color='#FF6D00', label='Signal', linewidth = 1, zorder=1)  # Signal line
        ax4.yaxis.set_ticks_position('both')
        ax4.yaxis.set_tick_params(pad=10, direction='inout', length=6, labelright=True, right=True)
        ax4.yaxis.set_major_formatter(FuncFormatter(comma_formatter))
        macd_colors = get_color_list(list(ohlc['macdhist'].values))
        ax4.bar(ohlc['Date'], ohlc['macdhist'], color=macd_colors, alpha=0.5, width=self.graph_width, label='Histogram', zorder=1)  # Histogram
        ax4.yaxis.set_tick_params(labelsize=13)
        ax4.yaxis.set_major_formatter(FuncFormatter(comma_formatter_2))
        max_val = max(ohlc['macd'].max(), ohlc['macdsignal'].max(), ohlc['macdhist'].max())
        min_val = min(ohlc['macd'].min(), ohlc['macdsignal'].min(), ohlc['macdhist'].min())
        ax4.set_ylim(top = max_val + ((max_val-min_val)*0.15))

        ax5.plot(ohlc['Date'], ohlc['Tenkan_Sen'], label='Tenkan-sen', color='#2862FF', linewidth = 1, zorder = 1)
        ax5.plot(ohlc['Date'], ohlc['Kijun_Sen'], label='Kijun-sen', color='#B71C1C', linewidth = 1, zorder = 1)
        ax5.plot(ohlc['Date'], ohlc['Chikou_Span'], label='Chikou Span', color='#43A047', linewidth = 1, zorder = 1)
        ax5.plot(ohlc['Date'], ohlc['Senkou_Span_A'], label='Senkou Span A (Leading Span A)', color='#A5D6A7', linewidth = 1, zorder = 1)
        ax5.plot(ohlc['Date'], ohlc['Senkou_Span_B'], label='Senkou Span B (Leading Span B)', color='#EF9A9A', linewidth = 1, zorder = 1)
        # Fill the area between the Senkou Span A and B (Kumo Cloud)
        ax5.fill_between(ohlc['Date'], ohlc['Senkou_Span_A'], ohlc['Senkou_Span_B'], 
                        where=ohlc['Senkou_Span_A'] >= ohlc['Senkou_Span_B'], 
                        facecolor='#43A047', alpha=0.125, interpolate=True, zorder = 1)
        ax5.fill_between(ohlc['Date'], ohlc['Senkou_Span_A'], ohlc['Senkou_Span_B'], 
                        where=ohlc['Senkou_Span_A'] < ohlc['Senkou_Span_B'], 
                        facecolor='#B71C1C', alpha=0.125, interpolate=True, zorder = 1)
        ax5.set_ylabel('Price, Ichimoku Cloud, and Fib Retracement', fontsize = 16)
        ax5.yaxis.set_label_position('left')
        ax5.yaxis.set_tick_params(pad=10, direction='inout', length=6, labelright=True, right=True)
        ax5.yaxis.set_tick_params(labelsize=13)
        levels_2 = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        color_list_2 = ['#787B86', '#F23545', '#FF9800', '#4CAF51', '#0A9981', '#06BCD4', '#787B86']
        levels_label_2 = ['0', '0.236', '0.382', '0.5', '0.618', '0.786','1']
        for i, level in enumerate(levels_2):
            price = ohlc['Low'].min() + (ohlc['High'].max() - ohlc['Low'].min()) * level
            ax5.axhline(price, linestyle='-', linewidth=0.75, color=color_list_2[i], zorder = 1)
            ax5.text(ohlc['Date'].iloc[-1], price, f"{levels_label_2[i]} " + "(" + "{:.2f}".format(price) + ")",  # Display on the right (last date)
                    va='bottom', ha='right', fontsize=12,  # Align for right side
                    # backgroundcolor='#151924',
                    color=color_list_2[i],
                    alpha=0.7)
        ax5.yaxis.set_major_formatter(FuncFormatter(comma_formatter_2))
        ax5.set_ylim(top = ohlc['High'].max() + (ohlc['High'].max()-ohlc['Low'].min())*0.15)
        candlestick_ohlc(ax5, ohlc.values, width=self.graph_width, colorup='#26A69A', colordown='#F05350', alpha=1.0)

        ax6.set_ylabel('STO', fontsize = 16)
        ax6.yaxis.set_label_position('left')
        ax6.yaxis.set_tick_params(pad=10, direction='inout', length=6, labelright=True, right=True)
        ax6.fill_between(ohlc['Date'], 20, 80, color='#20172E', zorder=1)
        ax6.axhline(y=20, color='grey', linestyle='--', linewidth = 0.5, zorder=2)
        ax6.axhline(y=80, color='grey', linestyle='--', linewidth = 0.5, zorder=2)
        ax6.yaxis.set_tick_params(labelsize=13)
        ax6.set_ylim(-8, 115)
        # max_val = max(ohlc['K_STO'].max(), ohlc['D_STO'].max())
        # min_val = min(ohlc['K_STO'].min(), ohlc['D_STO'].min())
        # ax6.set_ylim(top = max_val + ((max_val-min_val)*0.15))
        ax6.grid(color='#2E323D', linestyle='-', zorder = 1)
        ax6.plot(ohlc['Date'], ohlc['K_STO'], color='#09AE0C', linewidth = 1, zorder=2)
        ax6.plot(ohlc['Date'], ohlc['D_STO'], color='#B25B11', linewidth = 1, zorder=3)

        ax7.axhline(y=0, color='#9598A1', linestyle='--', linewidth = 0.5, zorder=1)  # RSI threshold at 70
        ax7.plot(ohlc['Date'], ohlc['CMF'], color='#09AE0C', linewidth = 1, zorder=1)
        ax7.set_ylabel('CMF', fontsize = 16)
        ax7.yaxis.set_ticks_position('both')  # Show ticks on both sides of the plot
        ax7.yaxis.set_tick_params(pad=10, direction='inout', length=6, labelright=True, right=True)  # Move ticks and labels to the right
        ax7.yaxis.set_tick_params(labelsize=13)
        ax7.set_ylim(top = ohlc['CMF'].max() + (ohlc['CMF'].max()-ohlc['CMF'].min())*0.15)

        ax8.plot(ohlc['Date'], ohlc['OBV'], color='#2862FF', linewidth = 1, zorder=1)
        ax8.set_ylabel('OBV', fontsize = 16)
        ax8.yaxis.set_ticks_position('both')  # Show ticks on both sides of the plot
        ax8.yaxis.set_tick_params(pad=10, direction='inout', length=6, labelright=True, right=True)  # Move ticks and labels to the right
        ax8.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
        ax8.yaxis.set_tick_params(labelsize=13)
        ax8.set_ylim(top = ohlc['OBV'].max() + (ohlc['OBV'].max()-ohlc['OBV'].min())*0.15)

        ax9.plot(ohlc['Date'], ohlc['ATR'], color='#B71C1C', linewidth = 1, zorder=1)
        ax9.set_ylabel('ATR', fontsize = 16)
        ax9.yaxis.set_ticks_position('both')  # Show ticks on both sides of the plot
        ax9.yaxis.set_tick_params(pad=10, direction='inout', length=6, labelright=True, right=True)  # Move ticks and labels to the right
        ax9.yaxis.set_major_formatter(FuncFormatter(comma_formatter))
        ax9.yaxis.set_tick_params(labelsize=13)
        ax9.set_xlabel('Date', fontsize = 16)
        ax9.xaxis.set_tick_params(labelsize=13)
        ax9.yaxis.set_major_formatter(FuncFormatter(comma_formatter_2))
        ax9.set_ylim(top = ohlc['ATR'].max() + (ohlc['ATR'].max()-ohlc['ATR'].min())*0.15)

        label_list = ["BB ", "SMA ", "close 2 ", f"{title_formatter(ohlc['middleband_BB'].values[-1-self.future_cloud_no])} ", f"{title_formatter(ohlc['upperband_BB'].values[-1-self.future_cloud_no])} ", f"{title_formatter(ohlc['lowerband_BB'].values[-1-self.future_cloud_no])}"]
        colors = ['white', '#868993', '#868993', '#2862FF', '#F23545', '#0A9981', '#EF9A9A']
        color_title(ax1, label_list, colors, y=0.958)
        label_list = ["Fibonacci ", "Retracement ", "0 ", "0.236 ", "0.382 ", "0.5", " 0.618", " 0.786", "1"]
        colors = ['white','white','#787B86', '#F23545', '#FF9800', '#4CAF51', '#0A9981', '#06BCD4', '#787B86']
        color_title(ax1, label_list, colors, y = 0.908)

        label_list = ["Ichimoku ", "9 26 52 26 ",
                    f"{title_formatter(ohlc['Tenkan_Sen'].loc[ohlc['Tenkan_Sen'].last_valid_index()])} ",
                    f"{title_formatter(ohlc['Kijun_Sen'].loc[ohlc['Kijun_Sen'].last_valid_index()])} ",
                    f"{title_formatter(ohlc['Chikou_Span'].loc[ohlc['Chikou_Span'].last_valid_index()])} ",
                    f"{title_formatter(ohlc['Senkou_Span_A'].values[-1])} ",
                    f"{title_formatter(ohlc['Senkou_Span_B'].values[-1])}",
                    ]
        colors = ['white', '#868993', '#2862FF', '#B71C1C', '#43A047', '#A5D6A7', '#EF9A9A']
        color_title(ax5, label_list, colors, y=0.958)
        label_list = ["Fibonacci ", "Retracement ", "0 ", "0.236 ", "0.382 ", "0.5", " 0.618", " 0.786", "1"]
        colors = ['white','white','#787B86', '#F23545', '#FF9800', '#4CAF51', '#0A9981', '#06BCD4', '#787B86']
        color_title(ax5, label_list, colors, y = 0.908)

        label_list = ["Volume ", f"{human_format(ohlc['Volume'].values[-1-self.future_cloud_no])}"]
        colors = ['white', volume_color[data[data['Close'].isna()].index[0] - 1]]
        color_title(ax2, label_list, colors, y = 0.88)

        label_list = ["RSI ", "14 close ", 
                    f"{title_formatter(ohlc['RSI'].values[-1-self.future_cloud_no])} ",
                    f"{title_formatter(ohlc['RSI_SMA_14'].values[-1-self.future_cloud_no])}",
                    ]
        colors = ['white', "#868993", "#7D57C2" , "yellow"]
        color_title(ax3, label_list, colors, y = 0.92)

        label_list = ["MACD ", "12 26 close ",
                    f"{title_formatter(ohlc['macdhist'].values[-1-self.future_cloud_no])} ",
                    f"{title_formatter(ohlc['macd'].values[-1-self.future_cloud_no])} ",
                    f"{title_formatter(ohlc['macdsignal'].values[-1-self.future_cloud_no])}",
                    ]
        colors = ['white', "#868993", macd_colors[:-self.future_cloud_no][-1], "#2862FF", "#FF6D00"]
        color_title(ax4, label_list, colors, y = 0.92)

        label_list = ["Stochastic Oscillator ", "14 3 80 20 ",
                    f"{title_formatter(ohlc['K_STO'].values[-1-self.future_cloud_no])} ",
                    f"{title_formatter(ohlc['D_STO'].values[-1-self.future_cloud_no])} ",
                    ]
        colors = ['white', "#868993", "#07FC00", "#FF7F00"]
        color_title(ax6, label_list, colors, y = 0.92)

        label_list = ["CMF ", "20 ",
                    f"{title_formatter(ohlc['CMF'].values[-1-self.future_cloud_no])}",
                    ]
        colors = ['white', "#868993", "#09AE0C",]
        color_title(ax7, label_list, colors, y = 0.92)

        label_list = ["OBV ", f"{human_format(ohlc['OBV'].values[-1-self.future_cloud_no])}",
                    ]
        colors = ['white', "#2862FF",]
        color_title(ax8, label_list, colors, y = 0.92)


        label_list = ["ATR ", "14 RMA ",
                    f"{title_formatter(ohlc['ATR'].values[-1-self.future_cloud_no])}",
                    ]
        colors = ['white', "#868993", "#B71C1C",]
        color_title(ax9, label_list, colors, y = 0.92)

        date_val = ohlc['Date'].values[1] - ohlc['Date'].values[0]
        ax1.set_xlim([ohlc['Date'].min() - date_val, ohlc['Date'].max() + date_val])

        plt.subplots_adjust(left=0.085, right=0.925, bottom=0.025, top=0.975)
        fig.subplots_adjust(hspace=0.03)
        plt.savefig('temp_candlestick_chart.png')

        with Image.open("temp_candlestick_chart.png") as img:
            new_width, new_height = 2048, 3072 # 6000, 9000
            img_resized = img.resize((new_width, new_height), resample=Image.LANCZOS)
            img_resized.save(f"{self.freq_interval}_candlestick_with_technical_indicators_chart.png")

        plt.close(fig)
        plt.close('all')
        plt.clf()
        plt.cla()
        ohlc = None
        data = None
        fig = None
        ax1 = None
        ax2 = None
        ax3 = None
        ax4 = None
        ax5 = None
        ax6 = None
        ax7 = None
        ax8 = None
        ax9 = None
        volume_color = None
        colors = None
        img = None
        img_resized = None
        del ohlc, data, fig, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9
        del volume_color, colors, img, img_resized
        matplotlib.pyplot.close('all')
        matplotlib.use('Agg')
        gc.collect()

    def get_list_3d(self):
        with requests.Session() as s:
            self.response = s.get('https://api.binance.com/api/v3/time').json()
        server_time_df = pd.DataFrame([self.response])
        server_time_df['serverTime']=server_time_df['serverTime'].apply(lambda d: datetime.datetime.fromtimestamp(int(d)/1000).strftime('%Y-%m-%d %H:%M:%S'))
        server_time_df['serverTime'] = pd.to_datetime(server_time_df['serverTime'])
        currect_time_text = str(server_time_df['serverTime'].values[0])[:10]
        currect_time_no = int(currect_time_text.replace('-',''))

        start_date = '2017-11-20' # NEO
        # start_date = '2017-08-20' # BTC
        # start_date = '2017-12-16' # LTC

        # Convert the start_date to a datetime object
        current_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')

        # Create a list to store the result
        date_list = []

        date_list.append(current_date.strftime('%Y-%m-%d'))

        # Loop to generate the list of dates
        while True:
            current_date += datetime.timedelta(days=3)
            date_list.append(current_date.strftime('%Y-%m-%d'))
            if int(str(current_date)[:10].replace('-','')) > currect_time_no:
                break
        return date_list
    def update_next_current_time_interval_3d(self):
        self.next_current_time_interval = self.get_list_3d()[-2] + ' 00:00:00'
        temp_df = pd.DataFrame([self.next_current_time_interval])
        temp_df.columns = ['next_current_time_interval']
        self.next_entry_sec = self.get_wait_entry_sec()
        temp_df['next_entry_sec'] = self.next_entry_sec
        self.next_current_time_interval_df = temp_df
        if self.next_current_time_interval_df['next_entry_sec'].values[0] < 0:
            self.next_current_time_interval = self.get_list_3d()[-1] + ' 00:00:00'
            temp_df = pd.DataFrame([self.next_current_time_interval])
            temp_df.columns = ['next_current_time_interval']
            self.next_entry_sec = self.get_wait_entry_sec()
            temp_df['next_entry_sec'] = self.next_entry_sec
            self.next_current_time_interval_df = temp_df
    def get_meeting_date(self):
        with requests.Session() as s:
            self.response = s.get('https://api.binance.com/api/v3/time').json()
        server_time = datetime.datetime.utcfromtimestamp(self.response['serverTime'] / 1000)
        formatted_time = server_time.strftime('%B %d, %Y, %H:%M')
        return formatted_time
    def calculate_ichimoku(self, ohlc):
        df = ohlc.copy()
        # Tenkan-sen (Conversion Line): (Highest High + Lowest Low) / 2 for the past 9 periods
        high_9 = df['High'].rolling(window=9).max()
        low_9 = df['Low'].rolling(window=9).min()
        df['Tenkan_Sen'] = (high_9 + low_9) / 2
        # Kijun-sen (Base Line): (Highest High + Lowest Low) / 2 for the past 26 periods
        high_26 = df['High'].rolling(window=26).max()
        low_26 = df['Low'].rolling(window=26).min()
        df['Kijun_Sen'] = (high_26 + low_26) / 2
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 plotted 26 periods ahead
        df['Senkou_Span_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(26)
        # Senkou Span B (Leading Span B): (Highest High + Lowest Low) / 2 for the past 52 periods plotted 26 periods ahead
        high_52 = df['High'].rolling(window=52).max()
        low_52 = df['Low'].rolling(window=52).min()
        df['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26) 
        # Chikou Span (Lagging Span): Closing price plotted 26 periods behind
        df['Chikou_Span'] = df['Close'].shift(-26)
        future_Senkou_Span_A = list((df['Tenkan_Sen'].tail(26) + df['Kijun_Sen'].tail(26)) / 2)[:self.future_cloud_no]
        future_Senkou_Span_B = list((high_52.tail(26) + low_52.tail(26)) / 2)[:self.future_cloud_no]
        future_26 = []
        date_val = df['Date'].values[1] - df['Date'].values[0]
        for i in range(1, self.future_cloud_no+1):
            val = df['Date'].tail(1).values[0] + i*date_val
            future_26.append(val)
        future_vals = pd.DataFrame({'Date': future_26})
        df = pd.concat([df, future_vals], ignore_index=True)
        all_Senkou_Span_A = list(df['Senkou_Span_A'].values[:-self.future_cloud_no]) + future_Senkou_Span_A
        df['Senkou_Span_A'] = all_Senkou_Span_A
        all_Senkou_Span_B = list(df['Senkou_Span_B'].values[:-self.future_cloud_no]) + future_Senkou_Span_B
        df['Senkou_Span_B'] = all_Senkou_Span_B
        return df
####
#### End Class Here
####
