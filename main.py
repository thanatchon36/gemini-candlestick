#!/usr/bin/env python
from autotrade import autotrade
import os, sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
import gc
def reset(df):
    cols = df.columns
    return df.reset_index()[cols]
def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
def get_cumsum_bal(return_list, leverage_no = 1, fee_rate = 0, stick_no = 1):
    init_bal = 100
    bal_list = [init_bal]
    
    for each in return_list:
        each = each * leverage_no
        stick_budget = init_bal / stick_no
        fee = (stick_budget * fee_rate / 100) * 2 * leverage_no
        if each != 0:
            stick_return = stick_budget * (each) / 100
            
            if each < -80:
                stick_return = -stick_budget
            
            init_bal = init_bal + stick_return - fee
            bal_list.append(init_bal)
    bal_df = pd.DataFrame(bal_list)
    bal_df.columns = ['final_bal']
    bal_df = reset(bal_df[bal_df['final_bal'].notna()])
    return bal_df
def convert_ts(timestamp):
    return datetime.strptime(timestamp, '%Y-%m-%d %H:%M')
print('OK !', flush=True)

recommendation_list = []
future_close_list = []
timestamp_list = []

def main():
    ws_name = os.getenv('ws_name')
    line_api_key = os.getenv('line_api_key')
    gemini_key = os.getenv('gemini_key')
    timeframe = os.getenv('timeframe')
    depth_limit_no = int(os.getenv('depth_limit_no'))
    temperature = float(os.getenv('temperature'))
    model = os.getenv('model')
    entry_time_minute = int(os.getenv('entry_time_minute'))

    self = autotrade(   binance_api_key = "",
                        binance_api_secret = "",
                        line_api_key = line_api_key,
                        ws_name = ws_name,
                        freq_interval = timeframe,
                        depth_limit_no = depth_limit_no,
    )
    self.temp_leverage_no = 1
    self.risk_ratio = 0.95
    self.batchOrders_no = 1
    self.gemini_key = gemini_key
    self.temperature = temperature
    self.symbol = 'BTCUSDT'
    self.model = model
    self.error_count = 0
    
    # Before Entry
    time.sleep(60 * entry_time_minute)
    try:
        if self.model == 'gemini-1.5-pro-latest':
            generative_response, execution_time, result_dict = self.get_gemini_response_2()
        elif self.model == 'gemini-1.5-flash-latest':
            generative_response, execution_time, result_dict = self.get_gemini_response()
        temp_msg_1 = 'Test: ' + generative_response
        temp_msg_2 = f'execution_time: {execution_time} {result_dict}'
        temp_msg_3 = f'Reason: Close Position Before Entry leverage_no: {self.temp_leverage_no} risk_ratio: {self.risk_ratio} batchOrders_no: {self.batchOrders_no} timeframe: {timeframe} depth_limit_no: {depth_limit_no} temperature: {temperature} model: {model}'
        self.lineNotify(temp_msg_1)
        self.lineNotify(temp_msg_2)
        if result_dict != 'error':
            if self.model == 'gemini-1.5-pro-latest':
                self.lineNotify(f"Current {self.symbol} Depth (The top {self.depth_limit_no} bids and asks from the order book.)", image_path = f"current_depth_top_{self.depth_limit_no}_bids_asks_chart.png")
                self.lineNotify(f"Current {self.symbol} {self.freq_interval} Candlestick Chart (with Technical Indicators)", image_path = f"{self.freq_interval}_candlestick_with_technical_indicators_chart.png")
    
                # self.lineNotify(f"Current {self.symbol} Depth (The top 5 bids and asks from the order book.)", image_path = "current_depth_top_5_bids_asks_chart.png")
                # self.lineNotify(f"Current {self.symbol} Depth (The top 10 bids and asks from the order book.)", image_path = "current_depth_top_10_bids_asks_chart.png")
                # self.lineNotify(f"Current {self.symbol} Depth (The top 20 bids and asks from the order book.)", image_path = "current_depth_top_20_bids_asks_chart.png")
                # self.lineNotify(f"Current {self.symbol} Depth (The top 50 bids and asks from the order book.)", image_path = "current_depth_top_50_bids_asks_chart.png")
                # self.lineNotify(f"Current {self.symbol} Depth (The top 100 bids and asks from the order book.)", image_path = "current_depth_top_100_bids_asks_chart.png")
                # self.lineNotify(f"Current {self.symbol} Depth (The top 500 bids and asks from the order book.)", image_path = "current_depth_top_500_bids_asks_chart.png")
                # self.lineNotify(f"Current {self.symbol} Depth (The top 1000 bids and asks from the order book.)", image_path = "current_depth_top_1000_bids_asks_chart.png")

                # self.lineNotify(f"Current {self.symbol} 30m Candlestick Chart (with Technical Indicators)", image_path = "30m_candlestick_with_technical_indicators_chart.png")
                # self.lineNotify(f"Current {self.symbol} 1h Candlestick Chart (with Technical Indicators)", image_path = "1h_candlestick_with_technical_indicators_chart.png")
                # self.lineNotify(f"Current {self.symbol} 2h Candlestick Chart (with Technical Indicators)", image_path = "2h_candlestick_with_technical_indicators_chart.png")
                # self.lineNotify(f"Current {self.symbol} 4h Candlestick Chart (with Technical Indicators)", image_path = "4h_candlestick_with_technical_indicators_chart.png")
                # self.lineNotify(f"Current {self.symbol} 6h Candlestick Chart (with Technical Indicators)", image_path = "6h_candlestick_with_technical_indicators_chart.png")
                # self.lineNotify(f"Current {self.symbol} 8h Candlestick Chart (with Technical Indicators)", image_path = "8h_candlestick_with_technical_indicators_chart.png")
                # self.lineNotify(f"Current {self.symbol} 12h Candlestick Chart (with Technical Indicators)", image_path = "12h_candlestick_with_technical_indicators_chart.png")
                # self.lineNotify(f"Current {self.symbol} 1d Candlestick Chart (with Technical Indicators)", image_path = "1d_candlestick_with_technical_indicators_chart.png")
                # self.lineNotify(f"Current {self.symbol} 3d Candlestick Chart (with Technical Indicators)", image_path = "3d_candlestick_with_technical_indicators_chart.png")
                # self.lineNotify(f"Current {self.symbol} 1w Candlestick Chart (with Technical Indicators)", image_path = "1w_candlestick_with_technical_indicators_chart.png")
                
            recommendation_key_name = list(result_dict.keys())[-1]
            best_recommendation = result_dict[recommendation_key_name]
            self.lineNotify(f"{recommendation_key_name}: {best_recommendation}")
        self.lineNotify(temp_msg_3)
        
        # Garbage collect to free up memory
        gc.collect()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno)
        self.lineNotify(temp_msg)

     # Get wait sec before entry & wait before entry
    self.update_next_current_time_interval()
    
    wait_entry_sec = self.next_current_time_interval_df['next_entry_sec'].values[0]
    self.lineNotify(f'wait_entry_sec: {wait_entry_sec} wait_entry_min: {round(wait_entry_sec/60, 2)} wait_entry_h: {round(wait_entry_sec/(60*60), 2)}')
    time.sleep(wait_entry_sec)
    # Clock: entry time
    starttime_tf = time.time()
    i = 0
    while True:
        try:
            # Garbage collect to free up memory
            gc.collect()

            # At entry_time
            time.sleep(60 * entry_time_minute)
            while True:
                if self.model == 'gemini-1.5-pro-latest':
                    generative_response, execution_time, result_dict = self.get_gemini_response_2()
                elif self.model == 'gemini-1.5-flash-latest':
                    generative_response, execution_time, result_dict = self.get_gemini_response()
                temp_msg_1 = generative_response
                temp_msg_2 = f'execution_time: {execution_time} {result_dict}'
                self.lineNotify(temp_msg_1)
                self.lineNotify(temp_msg_2)
                if result_dict != 'error':
                    if self.model == 'gemini-1.5-pro-latest':
                        self.lineNotify(f"Current {self.symbol} Depth (The top {self.depth_limit_no} bids and asks from the order book.)", image_path = f"current_depth_top_{self.depth_limit_no}_bids_asks_chart.png")
                        self.lineNotify(f"Current {self.symbol} {self.freq_interval} Candlestick Chart (with Technical Indicators)", image_path = f"{self.freq_interval}_candlestick_with_technical_indicators_chart.png")
                        
                        # self.lineNotify(f"Current {self.symbol} Depth (The top 5 bids and asks from the order book.)", image_path = "current_depth_top_5_bids_asks_chart.png")
                        # self.lineNotify(f"Current {self.symbol} Depth (The top 10 bids and asks from the order book.)", image_path = "current_depth_top_10_bids_asks_chart.png")
                        # self.lineNotify(f"Current {self.symbol} Depth (The top 20 bids and asks from the order book.)", image_path = "current_depth_top_20_bids_asks_chart.png")
                        # self.lineNotify(f"Current {self.symbol} Depth (The top 50 bids and asks from the order book.)", image_path = "current_depth_top_50_bids_asks_chart.png")
                        # self.lineNotify(f"Current {self.symbol} Depth (The top 100 bids and asks from the order book.)", image_path = "current_depth_top_100_bids_asks_chart.png")
                        # self.lineNotify(f"Current {self.symbol} Depth (The top 500 bids and asks from the order book.)", image_path = "current_depth_top_500_bids_asks_chart.png")
                        # self.lineNotify(f"Current {self.symbol} Depth (The top 1000 bids and asks from the order book.)", image_path = "current_depth_top_1000_bids_asks_chart.png")

                        # self.lineNotify(f"Current {self.symbol} 30m Candlestick Chart (with Technical Indicators)", image_path = "30m_candlestick_with_technical_indicators_chart.png")
                        # self.lineNotify(f"Current {self.symbol} 1h Candlestick Chart (with Technical Indicators)", image_path = "1h_candlestick_with_technical_indicators_chart.png")
                        # self.lineNotify(f"Current {self.symbol} 2h Candlestick Chart (with Technical Indicators)", image_path = "2h_candlestick_with_technical_indicators_chart.png")
                        # self.lineNotify(f"Current {self.symbol} 4h Candlestick Chart (with Technical Indicators)", image_path = "4h_candlestick_with_technical_indicators_chart.png")
                        # self.lineNotify(f"Current {self.symbol} 6h Candlestick Chart (with Technical Indicators)", image_path = "6h_candlestick_with_technical_indicators_chart.png")
                        # self.lineNotify(f"Current {self.symbol} 8h Candlestick Chart (with Technical Indicators)", image_path = "8h_candlestick_with_technical_indicators_chart.png")
                        # self.lineNotify(f"Current {self.symbol} 12h Candlestick Chart (with Technical Indicators)", image_path = "12h_candlestick_with_technical_indicators_chart.png")
                        # self.lineNotify(f"Current {self.symbol} 1d Candlestick Chart (with Technical Indicators)", image_path = "1d_candlestick_with_technical_indicators_chart.png")
                        # self.lineNotify(f"Current {self.symbol} 3d Candlestick Chart (with Technical Indicators)", image_path = "3d_candlestick_with_technical_indicators_chart.png")
                        # self.lineNotify(f"Current {self.symbol} 1w Candlestick Chart (with Technical Indicators)", image_path = "1w_candlestick_with_technical_indicators_chart.png")
                    recommendation_key_name = list(result_dict.keys())[-1]
                    best_recommendation = result_dict[recommendation_key_name]
                    self.lineNotify(f"{recommendation_key_name}: {best_recommendation}")
                else:
                    best_recommendation = 'error'
                if result_dict == 'error':
                    self.error_count = self.error_count + 1
                    if self.error_count == 10:
                        break
                    time.sleep(10)
                    continue
                else:
                    btc_df = self.get_btc_candle()
                    btc_df = btc_df.tail(1)
                    btc_df = reset(btc_df)
                    timestamp = str(btc_df['timestamp'].values[0])[:16].replace('T',' ')
                    symbol = btc_df['symbol'].values[0]
                    timeframe = btc_df['period'].values[0]
                    close = btc_df['close'].values[0]
                    close_change = btc_df['close_change'].values[0]
                    temp_msg = f'timestamp: {timestamp} symbol: {symbol.upper()} timeframe: {timeframe} close: {close} close_change: {close_change}'
                    self.lineNotify(temp_msg)
                    recommendation_list.append(best_recommendation)
                    future_close_list.append(close_change)
                    timestamp_list.append(timestamp)
                    self.error_count = 0
                    break
                
            if i > 0:
                df_dict = {
                    'recommendation': recommendation_list,
                    'close': future_close_list,
                    'timestamp': timestamp_list,
                }
                return_df = pd.DataFrame(df_dict)
                return_df['future_close'] = return_df['close'].shift(-1)
                return_df['future_timestamp'] = return_df['timestamp'].shift(-1)
                return_df = reset(return_df[return_df['future_close'].notna()])
                return_df['timestamp'] = return_df['timestamp'].apply(convert_ts)
                return_df['future_timestamp'] = return_df['future_timestamp'].apply(convert_ts)
                return_df['future_timestamp_sec'] =  return_df['future_timestamp'] - return_df['timestamp']
                return_df['future_timestamp_sec'] = return_df['future_timestamp_sec'] / np.timedelta64(1, 's')
                return_df.loc[return_df['future_timestamp_sec'] != self.freq_second, 'recommendation'] = 'ts_error'

                return_df['future_close'] = return_df['future_close'].astype(float)
                return_df['long_win_no'] = 0
                return_df.loc[(return_df['recommendation'] == 'long') & (return_df['future_close'] > 0), 'long_win_no'] = 1
                return_df['short_win_no'] = 0
                return_df.loc[(return_df['recommendation'] == 'short') & (return_df['future_close'] < 0), 'short_win_no'] = 1
                return_df['long_short_win_no'] = return_df['long_win_no'] + return_df['short_win_no'] 
                long_win_no = return_df['long_win_no'].sum()
                long_total_no = return_df[return_df['recommendation'] == 'long'].shape[0]
                short_win_no = return_df['short_win_no'].sum()
                short_total_no = return_df[return_df['recommendation'] == 'short'].shape[0]
                total_stick_no = return_df[return_df['recommendation'].isin(['long','short'])].shape[0]
                long_short_win_no_text = f'long_short_total: {round(long_win_no/long_total_no, 2)}/{round(short_win_no/short_total_no,2)}/{round((long_win_no+short_win_no)/total_stick_no,2)} long_short_total: {long_win_no}/{long_total_no}|{short_win_no}/{short_total_no}|{long_win_no+short_win_no}/{total_stick_no}'
                take_pos_text_1 = str(return_df['recommendation'].value_counts()).replace('Name: count, dtype: int64','')
                take_pos_text_2 = str(return_df['recommendation'].value_counts(normalize = True).round(2)).replace('Name: proportion, dtype: float64','')
                take_pos_text = take_pos_text_1 + take_pos_text_2
                return_df['return'] = -abs(return_df['future_close'])
                return_df.loc[return_df['long_win_no'] == 1, 'return'] = return_df['future_close']
                return_df.loc[return_df['short_win_no'] == 1, 'return'] = -return_df['future_close']
                return_df['return'] = return_df['return'] - 0.1
                return_df.loc[~return_df['recommendation'].isin(['long','short']), 'return'] = 0
                final_bal = round(get_cumsum_bal(list(return_df['return'].values))['final_bal'].values[-1], 2)
                take_pos_text = take_pos_text.replace('recommendation','')
                self.lineNotify(take_pos_text)
                self.lineNotify(long_short_win_no_text + '\n' + f'timeframe: {self.freq_interval} depth_limit_no: {self.depth_limit_no} temperature: {self.temperature}' + '\n' + f'final_bal: {str(final_bal)}')
                
            # Garbage collect to free up memory
            gc.collect()
            for _ in range(self.gc_collect_time):
                time.sleep(60 * 30)
                gc.collect()

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno)
            try:
                self.lineNotify(temp_msg)
            except:
                print(temp_msg, flush=True)
            finally:
                pass
        finally:
            pass
            i = i + 1
            time.sleep(self.freq_second - ((time.time() - starttime_tf) % self.freq_second))
            
if __name__ == '__main__':
    main()