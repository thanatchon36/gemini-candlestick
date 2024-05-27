import os
import sys
import time
import json
import hmac
import hashlib
import base64
import random
import datetime
import requests
import numpy as np
import pandas as pd
import talib
import re
import yfinance as yf
import gc
from urllib.parse import urlencode
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from pytickersymbols import PyTickerSymbols
import google.generativeai as genai

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('figure', figsize = (15, 12), dpi = 300)
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
from matplotlib.ticker import FixedLocator
from matplotlib.dates import DayLocator, date2num, num2date
import pdfkit
from pdf2image import convert_from_path
from img2pdf import convert
from PyPDF2 import PdfMerger

def reset(df):
    cols = df.columns
    return df.reset_index()[cols]

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
        self.candlestick_chart_no = 168
        self.future_cloud_no = 26
    @property
    def example_analysis_text(self):
        self.example_analysis_text_list = [f'John Bollinger, {self.analysis_verb} the Bollinger Bands of **{self.company_ticker_list[0]}**,...',
                                    f'Goichi Hosoda, {self.analysis_verb} the Ichimoku Cloud for **{self.company_ticker_list[0]}**,...',
                                    f'Marc Chaikin, {self.analysis_verb} **{self.company_ticker_list[0]}**,...',
                                    f'J. Welles Wilder, {self.analysis_verb} the RSI of **{self.company_ticker_list[0]}**,...',
                                    f'Ralph Nelson Elliott, {self.analysis_verb} the price action of **{self.company_ticker_list[0]}**,...',
                                    f'Munehisa Homma, {self.analysis_verb} the candlestick patterns of **{self.company_ticker_list[0]}**,...',
                                    f'Leonardo Pisano Fibonacci, {self.analysis_verb} **{self.company_ticker_list[0]}**,...',
                                    f'Gerald Appel, {self.analysis_verb} the MACD of **{self.company_ticker_list[0]}**,...',
                                    f'George Lane, {self.analysis_verb} the stochastic oscillator for **{self.company_ticker_list[0]}**,...',
                                    f'Joseph Granville, {self.analysis_verb} **{self.company_ticker_list[0]}**,...',
                                    ]
        random.shuffle(self.example_analysis_text_list)
        return self.example_analysis_text_list[0]
    @property
    def analysis_verb(self):
        self.analysis_verb_list = ['examining',
                                    'reviewing',
                                    'analyzing',
                                    'assessing',
                                    'evaluating',
                                    'scrutinizing',
                                    'observing',
                                    ]
        random.shuffle(self.analysis_verb_list)
        return self.analysis_verb_list[0]
    @property
    def company_ticker_text(self):
        company_ticker_text = ', '.join(self.company_ticker_list)
        company_ticker_text = company_ticker_text.strip()
        company_ticker_text = " ".join(company_ticker_text.split())
        return company_ticker_text
    @property
    def company_ticker_list(self):
        company_ticker_list = []
        for index, row in self.sp100_nasdaq100_df.iterrows():
            company_ticker_list.append(f"{row['name']} ({row['symbol']})")
        random.shuffle(company_ticker_list)
        return company_ticker_list
    @property
    def ticker_company(self):
        return self.ticker_company_dict[self.ticker]
    @property
    def ticker_sector(self):
        return self.ticker_sector_dict[self.ticker]
    @property
    def ticker_list(self):
        return list(self.sp100_nasdaq100_df['symbol'].values)
    @property
    def gc_collect_time(self):
        return self.gc_collect_time_dict[self.ori_freq_interval]
    @property
    def graph_width(self):
        return self.graph_width_dict[self.freq_interval]
    def prep_sp100_nasdaq100_dataset(self):
        stock_data = PyTickerSymbols()
        sp100_df = pd.DataFrame(list(stock_data.get_stocks_by_index('S&P 100')))
        nasdaq100_df = pd.DataFrame(list(stock_data.get_stocks_by_index('NASDAQ 100')))
        sp100_nasdaq100_df = reset(pd.concat([sp100_df, nasdaq100_df]))
        sp100_nasdaq100_df = sp100_nasdaq100_df.groupby('symbol').first().reset_index()
        candlestick_df = yf.download(list(sp100_nasdaq100_df['symbol'].values), period=f'{self.candlestick_chart_no*2}d', interval="1d")
        date_list = [str(each)[:10] for each in list(candlestick_df['Close'].index)]
        self.sp100_nasdaq100_df_dict = {}
        for each_ticker in tqdm(sp100_nasdaq100_df['symbol'].values):
            temp_dict = {'Date': [],
                        'Open': [],
                        'High': [],
                        'Low': [],
                        'Close': [],
                        'Volume': [],
                        }
            temp_dict['Date'].extend(date_list)
            temp_dict['Open'].extend(list(candlestick_df['Open'][each_ticker].values))
            temp_dict['High'].extend(list(candlestick_df['High'][each_ticker].values))
            temp_dict['Low'].extend(list(candlestick_df['Low'][each_ticker].values))
            temp_dict['Close'].extend(list(candlestick_df['Close'][each_ticker].values))
            temp_dict['Volume'].extend(list(candlestick_df['Volume'][each_ticker].values))
            temp_df = pd.DataFrame(temp_dict)
            if pd.notna(temp_df['Close'].values[0]):
                self.sp100_nasdaq100_df_dict[each_ticker] = temp_df
        sp100_nasdaq100_df = sp100_nasdaq100_df[sp100_nasdaq100_df['symbol'].isin(list(self.sp100_nasdaq100_df_dict.keys()))]   
        self.sp100_nasdaq100_df = reset(sp100_nasdaq100_df)
        self.sp100_nasdaq100_df['indices'] = self.sp100_nasdaq100_df['indices'].apply(lambda x: ';'.join(x))
        self.sp100_nasdaq100_df['industries'] = self.sp100_nasdaq100_df['industries'].apply(lambda x: ';'.join(x))
        self.sp100_nasdaq100_df.to_csv('data/csv/sp100_nasdaq100.csv', index = False)
        self.ticker_sector_dict = dict(zip(self.sp100_nasdaq100_df['symbol'], self.sp100_nasdaq100_df['industries']))
        self.ticker_company_dict = dict(zip(self.sp100_nasdaq100_df['symbol'], self.sp100_nasdaq100_df['name']))
    # self.docker_print('Error !')
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
    def update_next_current_time_interval(self):
        self.next_current_time_interval = self.ping_binance()
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
    def get_judge_instructions(self):
        system_instructions = f"""
        ## JUDGING ROUND

        Sponsor will evaluate each Entrant and their Submission. Your Submission, including Your video and code will be evaluated based on following judging criteria (the “Judging Criteria”), weighted equally:

        * Submissions will be evaluated by Google judges who excel in the following five (5) categories as they relate to this challenge: impact, remarkability, creativity, usefulness, and execution. Each criteria will be scored on a scale of 1 (strongly disagree) to 5 (strongly agree). 

        The judging criteria is as follows:

        **Category 1: Impact**

        * Is the solution easy and enjoyable to use for everyone, including people with disabilities? (maximum 5 points)
        * Does this solution have potential to contribute meaningfully to environmental sustainability? (maximum 5 points)
        * Does this solution have potential to contribute meaningfully to improving people's lives? (maximum 5 points)

        **Category 2: Remarkability**

        * Is the submission surprising to those that are well-versed in Large Language Models (“LLM”)? (maximum 5 points)
        * Is the submission surprising to those that are not well-versed in LLM? (maximum 5 points)

        **Category 3: Creativity**

        * Does the submission differ from existing, well known, applications in functionality? (maximum 5 points)
        * Does the submission differ from existing, well known, applications in user experience? (maximum 5 points)
        * Is the submission implemented through the use of creative problem-solving approaches? (maximum 5 points)

        **Category 4: Usefulness**

        * Does the submission include a well-defined target user persona/segmentation? (maximum 5 points)
        * Does the submission identify how the solution addresses specific user needs? (maximum 5 points)
        * How well does the solution, as implemented, help users meet these needs? (maximum 5 points)

        **Category 5: Execution**

        * Is the solution well-designed and adhere to software engineering practices? (maximum 5 points)
        * Is the LLM component of the solution well-designed and adhere to Machine Learning (ML)/LLM best practices? (maximum 5 points)

        **Maximum score: 65**

        *Best Overall Submission will be determined by the Entrant who has the highest score in the combined categories of Impact, Creativity and Usefulness. In the event of a tie, Sponsor will determine the Best Overall Submission Prize. Sponsor’s decision is final and binding.* 
        """
        return system_instructions
    def get_sector_text(self):
        sector_list = ['Information Technology',
                       'Health Care',
                       'Consumer Discretionary',
                       'Communication Services',
                       'Industrials',  
                       'Consumer Staples',
                       'Utilities',
                       'Renewable Energy',
                       'Real Estate',
                       'Materials',
                       'Financials',]
        random.shuffle(sector_list)
        return  "- " + "\n- ".join(sector_list)
    def get_system_instructions_1(self):
        self.board_member_no = 10
        # self.current_meeting_date
        # self.company_ticker_text
        # self.company_ticker_list[0]
        # len(self.company_ticker_list)
        system_instructions = f"""
        {self.get_judge_instructions()}

        **Google Gemini:** The Gemini Candlestick Investment Fund employs a dynamic investment strategy that capitalizes on both market growth and volatility. The Fund strategically allocates capital to a portfolio of core holdings, representing established companies with long-term growth potential, while simultaneously employing tactical opportunities to profit from short-term market fluctuations. This dual approach allows the Fund to navigate market cycles effectively, aiming to maximize returns through both growth investments and market-neutral strategies. This strategy, guided by the collective wisdom of the ten-member Board of Directors, ensures a balanced and diversified approach to investment, seeking to generate consistent and sustainable profitability for its stakeholders. The meeting minutes will primarily focus on the judging round mentioned earlier and aim to maximize the score closely to 65 as much as possible.

        **Target User Persona:** This application is designed for financial analysts, portfolio managers, and individuals interested in understanding how technical analysis and expert opinions can be integrated to make investment decisions. 

        **User Needs:**  Users need a clear and concise summary of expert opinions on various stocks and overall market sectors based on technical indicators and market trends. This information aids in making informed investment decisions. 

        **Meeting Details:**

        * **Date:** {self.current_meeting_date}
        * **Time:** 00:00 - 01:00
        * **Location:** Google Meet

        **Stock Selection:**

        The Board of Directors will be analyzing the following {len(self.company_ticker_list)} stocks:

        {self.company_ticker_text}.

        **Data:**

        The Board will have access to hypothetical candlestick charts and technical indicator data (Volume, RSI, MACD, Bollinger Bands, Fibonacci Retracement, Ichimoku Cloud, Stochastic Oscillator, Chaikin Money Flow, On-Balance Volume, and Average True Range) for each of these stocks, reflecting recent market activity. 

        **Board Members:**

        The Board of Directors comprises 10 preeminent members merging technical expertise with a deep understanding of market sectors and trends:

        1. Munehisa Homma, Chairman of the meeting:(assigned to analyze Candlestick Pattern Recognition and provide an opinion)
        2. John Bollinger (assigned to analyze the BB indicator and provide an opinion)
        3. Leonardo Pisano Fibonacci (assigned to analyze Fibonacci retracement and provide an opinion)
        4. Ralph Nelson Elliott, assigned to analyze potential Elliott Wave patterns, utilized Fibonacci ratios (0.786, 0.618, 0.5, 0.382, and 0.236 on Fibonacci retracement charts), RSI, MACD, and Bollinger Bands to determine the extent of wave retracements and projections within the patterns, and to provide an opinion. 
        5. Goichi Hosoda (assigned to analyze the Ichimoku Cloud and provide an opinion) 
        6. Joseph Granville (assigned to analyze the Volume and On Balance Volume indicators and provide an opinion)
        7. Gerald Appel (assigned to analyze the MACD indicator and provide an opinion)
        8. J. Welles Wilder (assigned to analyze RSI and ATR indicators and provide an opinion)
        9. George Lane (assigned to analyze the stochastic oscillator indicator and provide an opinion)
        10. Marc Chaikin (assigned to analyze the Chaikin Money Flow indicator and provide an opinion)

        **Instructions:**

        1. **Broad Market Analysis:**
            * Begin the meeting minutes with a concise analysis of the following sectors:
            {self.get_sector_text()}
            *  Synthesize the Board's collective opinion on each sector, including relevant technical indicator observations and potential trends. 
            *  Utilize a more professional and business-oriented language, incorporating critical analysis that establishes connections among significant sectors.
            *  Link between sectors in Broad Market Analysis and generate critical thoughts on these links.

        2. **Discussion:**
            * Following the broad market analysis, describe a hypothetical discussion flow focusing on specific stock analysis examples.
            * **Example:** "{self.example_analysis_text}"
            * Continue creating a narrative for the meeting, incorporating various opinions and disagreements, connecting board member expertise with specific ticker symbols from the list.
            * Highlight at least one stock from each market sector.

        3. **Consensus & Action:**

        * **Market Sentiment:**  
            * The Board assessed the overall market sentiment for relevant sectors, considering: 
                * [List specific factors analyzed, e.g., economic indicators, sector trends, investor sentiment surveys].
            * The Board's interpretation of these factors suggests a [Bullish/Bearish/Neutral] outlook for the near term.
        * **Fund Position:**

        **Based on an analysis of market sentiment and identified opportunities, the Board has determined the following strategy:**
            * **Adopting bullish positions:** To capitalize on potential market growth.
            * **Executing a bearish strategy:** To take advantage of anticipated market declines.
            * **Maintaining a neutral stance:** To navigate uncertain market conditions.
            * **Focusing on specific sectors:** To maximize potential returns.
            * **Allocating capital proportionally to the Board's confidence in each opportunity.**
            **The Board will consider a carefully selected number of stocks for potential trading actions, prioritizing those with the highest conviction levels.**
            **Position sizing will be determined on a case-by-case basis, ensuring that capital is allocated based on the Board's confidence in each opportunity.** 

        * **Alternative Strategies Considered:**
            * [Briefly list alternative approaches that were discussed and the rationale for not selecting them at this time. This demonstrates thoroughness.]
        * **Risk Management:**
            * The Board acknowledges the inherent risks associated with the chosen strategy and has implemented appropriate measures to mitigate potential losses, such as:
                * [List risk management techniques, e.g., stop-loss orders, diversification across sectors, hedging strategies, position limits]. 
        * **Monitoring & Review:**
            * The Board will actively monitor market conditions and the performance of selected positions.
            * The Board will reconvene in [timeframe, e.g., one week, two weeks] to review the fund's position and make any necessary adjustments based on evolving market dynamics and new information.

        4. **Ticker Symbols of Interest:**
            * Based on the discussion, list a few of the ticker symbols that were highlighted and provide reasons for their attention. These reasons should directly relate to the analysis conducted by the board members.

        5. **Further Action:**
            * The Board instructed the Fund's management team to execute the agreed-upon market position and further investigate the highlighted ticker symbols for potential investment actions aligned with the Fund's overall strategy.

        6. **Meeting Adjourned:** 01:30

        7. **Approved by:**
            * Munehisa Homma, Chairman
        """
        return system_instructions
    def get_system_instructions_2(self):
        system_instructions = f"""
        {self.get_judge_instructions()}
        
        ## Gemini Candlestick Investment Fund Meeting Summary - {self.current_meeting_date}

        You are provided with detailed minutes from the latest Gemini Candlestick Investment Fund meeting.  Your task is to analyze this information and create a concise and insightful summary for the fund's managers. 

        **Meeting Minutes:** 
        [Insert the transcript of the full meeting recorded in the user's initial input]

        **Objective:**  Summarize the provided meeting minutes, focusing on key decisions, market observations, and interconnections between sectors. Extract impactful insights relevant to the Gemini Candlestick Investment Fund's investment strategy.

        **Instructions:**

        1. **Concise Summary:**  Provide a clear and concise summary of the meeting's key discussions and decisions. This should include:
            * Overall market sentiment and the rationale behind it.
            * The fund's investment strategy and its justification.
            * Risk management measures being implemented. 

        2. **Sector Analysis:**  Analyze the discussion around each sector. Highlight:
            * The perceived strength or weakness of the sector.
            * Key factors and technical indicators driving the assessment.
            * Specific companies mentioned and the rationale for their inclusion.

        3. **Interconnected Insights:**  Identify and elaborate on the key interconnections observed between different sectors. Explain how these connections influence the fund's decisions.

        4. **Actionable Takeaways:**  Based on the meeting discussions, extract actionable takeaways and insights relevant to the fund's investment strategy. This might include:
            * Emerging investment opportunities.
            * Potential risks to be aware of.
            * Key trends influencing market dynamics.

        5. **Language:**  Use clear, professional, and business-oriented language. Avoid technical jargon where possible and explain any necessary technical terms in a way that is easy to understand.

        **Example Insights:**

        * "The meeting highlighted a growing divergence between traditional utilities and renewable energy, indicating a potential shift in investor sentiment towards sustainable investments."
        * "The strong performance of [INSERT_TICKET] suggests continued growth in the technology sector, potentially creating ripple effects in related industries like consumer electronics and e-commerce."
        * "The fund's cautious optimism reflects a balanced approach, capitalizing on growth opportunities while acknowledging potential market risks." 
        """
        return system_instructions
    def get_system_instructions_3(self):
        system_instructions = """
        Your role is to extract ticker symbols of "**4. Ticker Symbols of Interest:**" into JSON object format.
        """
        return system_instructions
    def get_system_instructions_4(self):
        system_instructions = """
        Your role is to translate the user's meeting minutes into HTML format.
        """
        return system_instructions    
    def get_gemini_response(self):
        genai.configure(api_key=self.gemini_key)
        # Set up the model
        generation_config = {
        "temperature": self.temperature,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 500000,
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
            # for each_ticker in tqdm(self.sp100_nasdaq100_df['symbol'].values[:202]):
            #     self.ticker = each_ticker
            #     self.get_candlestick_image()

            self.prompt_parts = []
            for each_ticker in tqdm(self.ticker_list[:202]):
                self.ticker = each_ticker
                temp_file = genai.upload_file(path=f"data/png/{each_ticker}.png", display_name=f'{self.ticker_company} ({each_ticker}): 1d Candlestick Chart (with Technical Indicators)')
                self.prompt_parts.append(temp_file)
            random.shuffle(self.prompt_parts)
            system_instruction_1 = self.get_system_instructions_1()
            print(system_instruction_1)
            minutes_text = str(genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                        generation_config=generation_config,
                                        system_instruction=system_instruction_1,
                                        safety_settings=safety_settings).generate_content(self.prompt_parts,
                                            request_options={"timeout": 1000}).text)
            print(execution_time, minutes_text)

            for each_prompt_part in tqdm(self.prompt_parts):
                genai.delete_file(each_prompt_part.name)

            minutes_html = str(genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                            generation_config=generation_config,
                            system_instruction=self.get_system_instructions_4(),
                            safety_settings=safety_settings).generate_content(minutes_text).text)
            minutes_html = minutes_html.replace('```html','')
            minutes_html = minutes_html.replace('```','')
            pdfkit.from_string(minutes_html, 'data/pdf/minutes.pdf')
            self.make_pdf_uncroppable('data/pdf/minutes.pdf','data/pdf/minutes.pdf')
            
            summary_text = str(genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                        generation_config=generation_config,
                                        system_instruction=self.get_system_instructions_2(),
                                        safety_settings=safety_settings).generate_content(minutes_text).text)
            summary_html = str(genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                                        generation_config=generation_config,
                                        system_instruction=self.get_system_instructions_4(),
                                        safety_settings=safety_settings).generate_content(summary_text).text)
            summary_html = summary_html.replace('```html','')
            summary_html = summary_html.replace('```','')
            pdfkit.from_string(summary_html, 'data/pdf/summary.pdf')
            self.make_pdf_uncroppable('data/pdf/summary.pdf','data/pdf/summary.pdf')

            for _ in range(6):
                try:
                    interest_ticker_list = str(genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                                                generation_config=generation_config,
                                                system_instruction="""Your role is to extract ticker symbols of "**4. Ticker Symbols of Interest:**" into JSON object format.""",
                                                safety_settings=safety_settings).generate_content(minutes_text).text)
                    key = list(self.extract_json(interest_ticker_list)[0].keys())[0]
                    interest_ticker_list = self.extract_json(interest_ticker_list)[0][key]
                    match_no = 0
                    for each_ticket in ticket_list:
                        if each_ticket in self.ticker_list:
                            match_no = match_no + 1
                    if match_no == len(ticket_list):
                        ticket_list = ticket_list[:12]
                        # Convert the PNG files to PDF
                        png_files = [f'data/png/{each}.png' for each in interest_ticker_list]
                        with open("data/pdf/png.pdf", "wb") as pdf_file:
                            pdf_bytes = convert(png_files)
                            pdf_file.write(pdf_bytes)
                        pdf_paths = ["data/pdf/minutes.pdf", "data/pdf/png.pdf"]
                        self.merge_pdfs(pdf_paths, pdf_paths[0])
                        break
                except:
                    pass

            

            # model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
            #                             generation_config=generation_config,
            #                             system_instruction=self.get_system_instructions_4(),
            #                             safety_settings=safety_settings)
            # prompt_parts = [
            #     generative_text,
            # ]
            # response = model.generate_content(prompt_parts)
            # result_dict = extract_json(response.text)[0]

            execution_time = time.time() - start_time
            execution_time = round(execution_time, 2)

            for each_prompt_part in self.prompt_parts:
                each_prompt_part = None
                del each_prompt_part
            model = None
            prompt_parts = None
            json_data = None
            del model, prompt_parts, json_data
            gc.collect()

            # return generative_text, execution_time, result_dict
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno)
            self.docker_print(temp_msg)
            try:
                self.docker_print(generative_text)
            except:
                pass
            try:
                for each_prompt_part in tqdm(self.prompt_parts):
                    genai.delete_file(each_prompt_part.name)
            except:
                pass
            execution_time = time.time() - start_time
            execution_time = round(execution_time, 2)
            return temp_msg, execution_time, 'error'      
    def get_candlestick_data(self):
        ohlc = self.sp100_nasdaq100_df_dict[self.ticker].copy()
        ohlc['ori_Date'] = ohlc['Date']
        ohlc['Date'] = mdates.date2num(ohlc['Date'])
        ohlc['RSI'] = talib.RSI(ohlc['Close'].values)
        ohlc['RSI_SMA_14'] = talib.SMA(ohlc['RSI'], timeperiod=14)
        close_prices = ohlc['Close']
        macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        ohlc['macd'] = macd
        ohlc['macdsignal'] = macd_signal
        ohlc['macdhist'] = macd_hist
        upperband, middleband, lowerband = talib.BBANDS(ohlc['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        ohlc['upperband_BB'] = upperband
        ohlc['middleband_BB'] = middleband
        ohlc['lowerband_BB'] = lowerband
        ohlc = self.calc_k_d(ohlc)
        ohlc = self.calc_CMF(ohlc)
        ohlc['OBV'] = talib.OBV(ohlc['Close'], ohlc['Volume'])
        ohlc['ATR'] = talib.ATR(ohlc['High'], ohlc['Low'], ohlc['Close'])  
        ds_df = pd.DataFrame({'date_column': [ohlc['ori_Date'].values[-1]]})
        ds_df['date_column'] = pd.to_datetime(ds_df['date_column'])
        ds_df['formatted_date'] = ds_df['date_column'].dt.strftime('%B %d, %Y, at %H:%M')
        self.current_meeting_date = ds_df['formatted_date'].values[-1]
        ohlc = self.calculate_ichimoku(ohlc)
        ohlc = reset(ohlc.tail(self.candlestick_chart_no + self.future_cloud_no))
        concat_date_list = []
        for i in range(self.candlestick_chart_no + self.future_cloud_no):
            concat_date_list.append(ohlc['Date'].max() - (i+1))
        ohlc['Date'] = concat_date_list[::-1]
        return ohlc    
    def get_candlestick_image(self):
        ohlc = self.get_candlestick_data()
        data = ohlc.copy()
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
        ax4.set_axisbelow(True)
        ax4.grid(color='#2E323D', linestyle='-', zorder = 0)
        ax5.set_axisbelow(True)
        ax5.grid(color='#2E323D', linestyle='-', zorder = 1)
        ax7.set_axisbelow(True)
        ax7.grid(color='#2E323D', linestyle='-', zorder = 0)
        ax8.set_axisbelow(True)
        ax8.grid(color='#2E323D', linestyle='-', zorder = 0)
        ax9.set_axisbelow(True)
        ax9.grid(color='#2E323D', linestyle='-', zorder = 0)

        ax1.set_title(f'{self.ticker_company} ({self.ticker}): 1d Candlestick Chart (with Technical Indicators)', fontsize=16)
        levels = [0, 0.214, 0.382, 0.5, 0.618, 0.764, 1]
        color_list = ['#787B86', '#06BCD4', '#0A9981', '#4CAF51', '#FF9800', '#F23545', '#787B86']
        levels_label = ['1', '0.786', '0.618', '0.5', '0.382', '0.236','0']
        for i, level in enumerate(levels):
            price = ohlc['Low'].min() + (ohlc['High'].max() - ohlc['Low'].min()) * level
            ax1.axhline(price, linestyle='-', linewidth=0.75, color=color_list[i], zorder=1)
            ax1.text(ohlc['Date'].iloc[-1], price, f"{levels_label[i]} " + "(" + "{:.2f}".format(price) + ")",  # Display on the right (last date)
                    va='bottom', ha='right', fontsize=12,  # Align for right side
                    color=color_list[i],
                    alpha=0.7)
        ax1.plot(ohlc['Date'], ohlc['upperband_BB'], color='#F23545', linestyle='-', linewidth=1, zorder=1)
        ax1.plot(ohlc['Date'], ohlc['middleband_BB'], color='#2862FF', linestyle='-', linewidth=1, zorder=1)
        ax1.plot(ohlc['Date'], ohlc['lowerband_BB'], color='#0A9981', linestyle='-', linewidth=1, zorder=1)
        ax1.fill_between(ohlc['Date'], ohlc['lowerband_BB'], ohlc['upperband_BB'], color='#2862FF', alpha=0.078, zorder=1)
        ax1.yaxis.set_major_formatter(FuncFormatter(self.comma_formatter_2))
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
        ax2.yaxis.set_major_formatter(FuncFormatter(self.millions_formatter))
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
        ax3.grid(color='#2E323D', linestyle='-', zorder = 1)
        ax3.plot(ohlc['Date'], ohlc['RSI'], color='#7D57C2', linewidth = 1, zorder=2)  # Assuming 'Date' is the date column and 'RSI' is the RSI values column
        ax3.plot(ohlc['Date'], ohlc['RSI_SMA_14'], color='yellow', linewidth = 1, zorder=3)

        # Plot MACD on the fourth subplot
        ax4.set_ylabel('MACD', fontsize = 16)
        ax4.plot(ohlc['Date'], ohlc['macd'], color='#2862FF', label='MACD', linewidth = 1, zorder=1)  # MACD line
        ax4.plot(ohlc['Date'], ohlc['macdsignal'], color='#FF6D00', label='Signal', linewidth = 1, zorder=1)  # Signal line
        ax4.yaxis.set_ticks_position('both')
        ax4.yaxis.set_tick_params(pad=10, direction='inout', length=6, labelright=True, right=True)
        ax4.yaxis.set_major_formatter(FuncFormatter(self.comma_formatter))
        macd_colors = self.get_color_list(list(ohlc['macdhist'].values))
        ax4.bar(ohlc['Date'], ohlc['macdhist'], color=macd_colors, alpha=0.5, width=self.graph_width, label='Histogram', zorder=1)  # Histogram
        ax4.yaxis.set_tick_params(labelsize=13)
        ax4.yaxis.set_major_formatter(FuncFormatter(self.comma_formatter_2))
        max_val = max(ohlc['macd'].max(), ohlc['macdsignal'].max(), ohlc['macdhist'].max())
        min_val = min(ohlc['macd'].min(), ohlc['macdsignal'].min(), ohlc['macdhist'].min())
        ax4.set_ylim(top = max_val + ((max_val-min_val)*0.15))

        ax5.plot(ohlc['Date'], ohlc['Tenkan_Sen'], label='Tenkan-sen', color='#2862FF', linewidth = 1, zorder = 1)
        ax5.plot(ohlc['Date'], ohlc['Kijun_Sen'], label='Kijun-sen', color='#B71C1C', linewidth = 1, zorder = 1)
        ax5.plot(ohlc['Date'], ohlc['Chikou_Span'], label='Chikou Span', color='#43A047', linewidth = 1, zorder = 1)
        ax5.plot(ohlc['Date'], ohlc['Senkou_Span_A'], label='Senkou Span A (Leading Span A)', color='#A5D6A7', linewidth = 1, zorder = 1)
        ax5.plot(ohlc['Date'], ohlc['Senkou_Span_B'], label='Senkou Span B (Leading Span B)', color='#EF9A9A', linewidth = 1, zorder = 1)
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
                    color=color_list_2[i],
                    alpha=0.7)
        ax5.yaxis.set_major_formatter(FuncFormatter(self.comma_formatter_2))
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
        ax8.yaxis.set_major_formatter(FuncFormatter(self.millions_formatter))
        ax8.yaxis.set_tick_params(labelsize=13)
        ax8.set_ylim(top = ohlc['OBV'].max() + (ohlc['OBV'].max()-ohlc['OBV'].min())*0.15)

        ax9.plot(ohlc['Date'], ohlc['ATR'], color='#B71C1C', linewidth = 1, zorder=1)
        ax9.set_ylabel('ATR', fontsize = 16)
        ax9.yaxis.set_ticks_position('both')  # Show ticks on both sides of the plot
        ax9.yaxis.set_tick_params(pad=10, direction='inout', length=6, labelright=True, right=True)  # Move ticks and labels to the right
        ax9.yaxis.set_major_formatter(FuncFormatter(self.comma_formatter))
        ax9.yaxis.set_tick_params(labelsize=13)
        ax9.set_xlabel('Date', fontsize = 16)
        ax9.xaxis.set_tick_params(labelsize=13)
        ax9.yaxis.set_major_formatter(FuncFormatter(self.comma_formatter_2))
        ax9.set_ylim(top = ohlc['ATR'].max() + (ohlc['ATR'].max()-ohlc['ATR'].min())*0.15)

        label_list = ["BB ", "SMA ", "close 2 ", f"{self.title_formatter(ohlc['middleband_BB'].values[-1-self.future_cloud_no])} ", f"{self.title_formatter(ohlc['upperband_BB'].values[-1-self.future_cloud_no])} ", f"{self.title_formatter(ohlc['lowerband_BB'].values[-1-self.future_cloud_no])}"]
        colors = ['white', '#868993', '#868993', '#2862FF', '#F23545', '#0A9981', '#EF9A9A']
        self.color_title(ax1, label_list, colors, y=0.958)
        label_list = ["Fibonacci ", "Retracement ", "0 ", "0.236 ", "0.382 ", "0.5", " 0.618", " 0.786", "1"]
        colors = ['white','white','#787B86', '#F23545', '#FF9800', '#4CAF51', '#0A9981', '#06BCD4', '#787B86']
        self.color_title(ax1, label_list, colors, y = 0.908)

        label_list = ["Ichimoku ", "9 26 52 26 ",
                    f"{self.title_formatter(ohlc['Tenkan_Sen'].loc[ohlc['Tenkan_Sen'].last_valid_index()])} ",
                    f"{self.title_formatter(ohlc['Kijun_Sen'].loc[ohlc['Kijun_Sen'].last_valid_index()])} ",
                    f"{self.title_formatter(ohlc['Chikou_Span'].loc[ohlc['Chikou_Span'].last_valid_index()])} ",
                    f"{self.title_formatter(ohlc['Senkou_Span_A'].values[-1])} ",
                    f"{self.title_formatter(ohlc['Senkou_Span_B'].values[-1])}",
                    ]
        colors = ['white', '#868993', '#2862FF', '#B71C1C', '#43A047', '#A5D6A7', '#EF9A9A']
        self.color_title(ax5, label_list, colors, y=0.958)
        label_list = ["Fibonacci ", "Retracement ", "0 ", "0.236 ", "0.382 ", "0.5", " 0.618", " 0.786", "1"]
        colors = ['white','white','#787B86', '#F23545', '#FF9800', '#4CAF51', '#0A9981', '#06BCD4', '#787B86']
        self.color_title(ax5, label_list, colors, y = 0.908)

        label_list = ["Volume ", f"{self.human_format(ohlc['Volume'].values[-1-self.future_cloud_no])}"]
        colors = ['white', volume_color[data[data['Close'].isna()].index[0] - 1]]
        self.color_title(ax2, label_list, colors, y = 0.88)

        label_list = ["RSI ", "14 close ", 
                    f"{self.title_formatter(ohlc['RSI'].values[-1-self.future_cloud_no])} ",
                    f"{self.title_formatter(ohlc['RSI_SMA_14'].values[-1-self.future_cloud_no])}",
                    ]
        colors = ['white', "#868993", "#7D57C2" , "yellow"]
        self.color_title(ax3, label_list, colors, y = 0.92)

        label_list = ["MACD ", "12 26 close ",
                    f"{self.title_formatter(ohlc['macdhist'].values[-1-self.future_cloud_no])} ",
                    f"{self.title_formatter(ohlc['macd'].values[-1-self.future_cloud_no])} ",
                    f"{self.title_formatter(ohlc['macdsignal'].values[-1-self.future_cloud_no])}",
                    ]
        colors = ['white', "#868993", macd_colors[:-self.future_cloud_no][-1], "#2862FF", "#FF6D00"]
        self.color_title(ax4, label_list, colors, y = 0.92)

        label_list = ["Stochastic Oscillator ", "14 3 80 20 ",
                    f"{self.title_formatter(ohlc['K_STO'].values[-1-self.future_cloud_no])} ",
                    f"{self.title_formatter(ohlc['D_STO'].values[-1-self.future_cloud_no])} ",
                    ]
        colors = ['white', "#868993", "#07FC00", "#FF7F00"]
        self.color_title(ax6, label_list, colors, y = 0.92)

        label_list = ["CMF ", "20 ",
                    f"{self.title_formatter(ohlc['CMF'].values[-1-self.future_cloud_no])}",
                    ]
        colors = ['white', "#868993", "#09AE0C",]
        self.color_title(ax7, label_list, colors, y = 0.92)

        label_list = ["OBV ", f"{self.human_format(ohlc['OBV'].values[-1-self.future_cloud_no])}",
                    ]
        colors = ['white', "#2862FF",]
        self.color_title(ax8, label_list, colors, y = 0.92)

        label_list = ["ATR ", "14 RMA ",
                    f"{self.title_formatter(ohlc['ATR'].values[-1-self.future_cloud_no])}",
                    ]
        colors = ['white', "#868993", "#B71C1C",]
        self.color_title(ax9, label_list, colors, y = 0.92)

        date_val = ohlc['Date'].values[1] - ohlc['Date'].values[0]
        ax1.set_xlim([ohlc['Date'].min() - date_val, ohlc['Date'].max() + date_val])

        xticklabels_list = []
        start_i = 19 - 1 
        xticklabels_list.append(self.get_date_month_text(ohlc['ori_Date'].values[start_i - 1]))
        next_i = start_i + 24
        xticklabels_list.append(self.get_date_month_text(ohlc['ori_Date'].values[next_i]))
        for _ in range(5):
            next_i = next_i + 24
            xticklabels_list.append(self.get_date_month_text(ohlc['ori_Date'].values[next_i]))
        xticklabels_list_2 = [xticklabels_list[-1]]
        dates = [datetime.datetime.strptime(date, '%d %b') for date in xticklabels_list_2]
        last_date = dates[-1]
        next_days = []
        # Add days until we have 24 additional days
        days_to_add = 24
        while len(next_days) < days_to_add:
            last_date += datetime.timedelta(days=1)
            if last_date.weekday() < 5:  # Monday to Friday are 0-4
                next_days.append(last_date)
        next_days_str = [date.strftime('%d %b') for date in next_days]
        xticklabels_list.extend([next_days_str[-1]])
        # Format x-axis ticks for the first subplot
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=24))  # Adjust the interval as needed
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax1.set_xticklabels(xticklabels_list)

        plt.subplots_adjust(left=0.085, right=0.925, bottom=0.025, top=0.975)
        fig.subplots_adjust(hspace=0.03)
        plt.savefig('data/png/temp.png')

        # Google Gemini accepts images with a resolution of 3072x3072 pixels.
        with Image.open("data/png/temp.png") as img:
            new_width, new_height = 2048, 3072
            img_resized = img.resize((new_width, new_height), resample=Image.LANCZOS)
            img_resized.save(f"data/png/{self.ticker}.png")
        os.remove('data/png/temp.png')
        
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
    def calc_k_d(self, data, N=14, M=3):
        data['low_N'] = data['Low'].rolling(N).min()
        data['high_N'] = data['High'].rolling(N).max()
        data['K_STO'] = 100 * (data['Close'] - data['low_N']) / \
            (data['high_N'] - data['low_N'])
        data['D_STO'] = data['K_STO'].rolling(M).mean()
        data.drop(columns = ['low_N','high_N'], inplace = True)
        return data
    def calc_CMF(self, ask_series):
        ask_series["cmfm"] = (((ask_series["Close"] - ask_series["Low"]) - (ask_series["High"] - ask_series["Close"])) / (ask_series["High"] - ask_series["Low"]))
        ask_series["cmfv"] = ask_series["cmfm"] * ask_series["Volume"]
        ask_series["CMF"] = ask_series['cmfv'].rolling(window=20).mean() / ask_series['Volume'].rolling(window=21).mean() 
        ask_series.drop(columns = ['cmfm','cmfv'], inplace = True)
        return ask_series
    def human_format(self, num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        # add more suffixes if you need them
        return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
    def get_color_list(self, values):
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
    def millions_formatter(self, x, pos):
        num = x
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        # add more suffixes if you need them
        return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
    def comma_formatter(self, x, pos):
        return "{:,}".format(x)
    def comma_formatter_2(self, x, pos):
        return "{:,.2f}".format(x)
    def title_formatter(self, x):
        return "{:.2f}".format(x)
    def color_title(self, ax, labels, colors, textprops={'size': 15}, y=0.96):
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
    def extract_json(self, text_response):
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
                extended_json_str = self.extend_search(text_response, match.span())
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
    def extend_search(self, text, span):
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
    def get_date_month_text(self, date_str):
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return date_obj.strftime("%d %b")
    
    def make_pdf_uncroppable(self, input_pdf_path, output_pdf_path):
        """Converts a PDF to a sequence of images, creating an 'uncroppable' PDF.

        Args:
            input_pdf_path (str): The path to the input PDF file.
            output_pdf_path (str): The path where the output PDF will be saved.
        """

        images = convert_from_path(input_pdf_path)

        # If your PDF has multiple pages:
        if len(images) > 1:
            images[0].save(
                output_pdf_path,
                "PDF",
                resolution=100.0,
                save_all=True,
                append_images=images[1:],
            )
        else:
            images[0].save(output_pdf_path, "PDF", resolution=100.0)
    def merge_pdfs(self, paths, output_filename):
        merger = PdfMerger()
        for path in paths:
            merger.append(path)
        merger.write(output_filename)
        merger.close()
####
#### End Class Here
####
