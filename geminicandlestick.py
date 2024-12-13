# Import necessary libraries and modules

# OS-level interactions
import os

# System-specific parameters and functions
import sys

# Time-related functions
import time

# JSON data handling
import json

# Random number generation
import random

# Date and time handling
import datetime

# Web scraping and data retrieval
import requests

# Data analysis and manipulation
import pandas as pd

# Technical analysis indicators
import talib

# Regular expression operations
import re

# Financial data from Yahoo Finance
import yfinance as yf

# Memory management (garbage collection)
import gc

# Image processing
from PIL import Image

# Progress bar creation
from tqdm import tqdm

import numpy as np # Import the NumPy library and give it the alias 'np'.

# Stock ticker symbols retrieval
from pytickersymbols import PyTickerSymbols

# Google Generative AI services
import google.generativeai as genai

# Plotting and visualization
import matplotlib
# Set backend to 'Agg' for non-interactive plotting
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
# Set default figure size and DPI
plt.rc('figure', figsize=(15, 12), dpi=300) 
from matplotlib.ticker import FuncFormatter  # Custom tick formatting
import matplotlib.dates as mdates  # Date handling in plots
# Candlestick charts
from mplfinance.original_flavor import candlestick_ohlc  

# PDF generation and manipulation
import pdfkit
from pdf2image import convert_from_path  # PDF to image conversion
from img2pdf import convert  # Image to PDF conversion
from PyPDF2 import PdfMerger  # PDF file merging
import markdown # Import the "markdown" library. 

def reset_dataframe_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resets the index of a pandas DataFrame while preserving the original columns.

    This function takes a DataFrame as input, resets its index to a default
    integer index, and then selects only the original columns, effectively
    removing any previous index levels as columns.

    Args:
        df (pandas.DataFrame): The DataFrame whose index needs to be reset.

    Returns:
        pandas.DataFrame: A new DataFrame with the index reset and only the
                          original columns retained.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> df = df.set_index('A')
        >>> df
           B
        A
        1  4
        2  5
        3  6
        >>> reset_dataframe_index(df)
           A  B
        0  1  4
        1  2  5
        2  3  6
    """
    original_columns = df.columns  # Store the original column names
    df = df.reset_index()  # Reset the index, which brings the previous index as a column
    df = df[original_columns]  # Select only the original columns
    return df  # Return the DataFrame with the reset index

class GeminiCandlestick:
    """
    A class to fetch and process candlestick data from the Gemini API.
    """
    def __init__(self, gemini_key, BOT_TOKEN, CHAT_ID, freq_interval="1d"):
        """
        Initializes the GeminiCandlestick object.

        Args:
            gemini_key (str): Your Gemini API key.
            BOT_TOKEN (str): Your Telegram bot token.
            CHAT_ID (str): Your Telegram chat ID.
            freq_interval (str, optional): The frequency interval for candlestick data. Defaults to "1d" (daily).
        """
        self.gemini_key = gemini_key
        self.freq_interval = freq_interval
        self.BOT_TOKEN = BOT_TOKEN
        self.CHAT_ID = CHAT_ID

        # Configuration parameters
        self.candlestick_chart_no = 26*7 
        self.future_cloud_no = 26*1

        # Dictionaries to store frequency-dependent values
        self.freq_dict = {
            '1d': 3600 * 24,  # Seconds in a day
        }
        self.graph_width_dict = {
            '1d': 0.75,  # Graph width for daily interval
        }
        self.gc_collect_time_dict = {
            '1d': 48 - 1,  # Garbage collection time for daily interval
        }

        # List of market sectors
        self.sector_list = [
        'Renewable Energy',
        'Biotechnology',
        'Information Technology',
        'Health Care',
        'Materials',
        'Industrials',
        'Consumer Discretionary',
        'Consumer Staples',
        'Communication Services',
        'Utilities',
        'Transportation',
        'Energy',
        'Financials',
        'Real Estate',
        'Hospitality'
        ]

    @property
    def until_next_day_sec(self):
        """
        Calculates the time in seconds until the next day (UTC).

        Returns:
            float: Time in seconds until the next day.
        """
        today_utc = datetime.datetime.now(datetime.timezone.utc)
        tomorrow_utc = today_utc + datetime.timedelta(days=1)
        tomorrow_utc = tomorrow_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        time_to_wait = (tomorrow_utc - today_utc).total_seconds()
        return float(time_to_wait)

    @property
    def today_time(self):
        """
        Returns the current UTC time in HH:MM:SS format.

        Returns:
            str: Current UTC time.
        """
        today_utc = datetime.datetime.now(datetime.timezone.utc)
        return str(today_utc.strftime('%H:%M:%S'))

    @property
    def today_date(self):
        """
        Returns the current day of the week (lowercase).

        Returns:
            str: Current day of the week.
        """
        today_utc = datetime.datetime.now(datetime.timezone.utc).date()
        return str(today_utc.strftime("%A")).lower()

    @property
    def file_date(self):
        """
        Returns the current date in YYYY-MM-DD format.

        Returns:
            str: Current date.
        """
        today_utc = datetime.datetime.now(datetime.timezone.utc).date()
        return str(today_utc.strftime("%Y-%m-%d"))

    @property
    def current_meeting_date(self):
        """
        Returns a formatted string representing the current date and time (UTC).

        Returns:
            str: Formatted current date and time.
        """
        today_utc = datetime.datetime.now(datetime.timezone.utc).date()
        return str(today_utc.strftime('%A, %B %d, %Y, at 00:00'))

    @property
    def example_analysis_text(self):
        """
        Returns a randomly chosen example analysis text.

        Returns:
            str: Example analysis text.
        """
        self.example_analysis_text_list = [
            f'John Bollinger, {self.analysis_verb} the Bollinger Bands of **{self.company_ticker_list[0]}**,...',
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
        """
        Returns a randomly chosen analysis verb.

        Returns:
            str: Analysis verb.
        """
        self.analysis_verb_list = ['examining', 'reviewing', 'analyzing', 'assessing', 'evaluating', 'scrutinizing', 'observing']
        random.shuffle(self.analysis_verb_list)
        return self.analysis_verb_list[0]

    @property
    def company_ticker_text(self):
        """
        Returns a formatted string of company tickers.

        Returns:
            str: Formatted company tickers.
        """
        company_ticker_text = ', '.join(self.company_ticker_list)
        company_ticker_text = company_ticker_text.strip()
        company_ticker_text = " ".join(company_ticker_text.split())
        return company_ticker_text

    @property
    def company_ticker_list(self):
        """
        Returns a shuffled list of company names and tickers.

        Returns:
            list: List of company names and tickers.
        """
        company_ticker_list = []
        # Assuming self.sp500_df is defined elsewhere
        for index, row in self.sp500_df.iterrows(): 
            company_ticker_list.append(f"{row['name']} ({row['symbol']})")
        random.shuffle(company_ticker_list)
        return company_ticker_list

    @property
    def ticker_company(self):
        """
        Returns the company name for a given ticker.

        Returns:
            str: Company name.
        """
        # Assuming self.ticker_company_dict and self.ticker are defined elsewhere
        return self.ticker_company_dict[self.ticker] 

    @property
    def ticker_sector(self):
        """
        Returns the sector for a given ticker.

        Returns:
            str: Sector.
        """
        # Assuming self.ticker_sector_dict and self.ticker are defined elsewhere
        return self.ticker_sector_dict[self.ticker]

    @property
    def ticker_list(self):
        """
        Returns a list of tickers from the sp500_df DataFrame.

        Returns:
            list: List of tickers.
        """
        # Assuming self.sp500_df is defined elsewhere
        return list(self.sp500_df['symbol'].values) 

    @property
    def gc_collect_time(self):
        """
        Returns the garbage collection time based on the original frequency interval.

        Returns:
            int: Garbage collection time.
        """
        # Assuming self.gc_collect_time_dict and self.ori_freq_interval are defined elsewhere
        return self.gc_collect_time_dict[self.ori_freq_interval] 

    @property
    def graph_width(self):
        """
        Returns the graph width based on the frequency interval.

        Returns:
            float: Graph width.
        """
        return self.graph_width_dict[self.freq_interval]


    def telegram_send_message(self, message):
        """Sends a text message to a Telegram chat.

        Args:
            message: The text message to send.
        """

        with requests.post(
            f"https://api.telegram.org/bot{self.BOT_TOKEN}/sendMessage",  # URL for the Telegram API endpoint
            json={
                "chat_id": self.CHAT_ID,  # The ID of the chat/group where the message will be sent
                "text": message  # The text of the message to be sent
            }
        ) as response:
            # The `with` statement ensures that the response object is properly closed
            # after the request is completed, even if an exception occurs.

            if response.status_code == 200:
                # If the status code is 200 (OK), the message was sent successfully
                self.docker_print("Message sent successfully!")
            else:
                # If the status code is not 200, there was an error
                self.docker_print(f"Error: {response.status_code} - {response.text}")
                # Print the error code and the error message from the Telegram API

    def telegram_send_pdfs(self, pdf_paths, caption_list):
        """Sends multiple PDFs individually to a Telegram chat.

        Args:
            pdf_paths: A list of paths to the PDF files.
            caption_list: A list of captions for each PDF.
        """

        url = f"https://api.telegram.org/bot{self.BOT_TOKEN}/sendDocument"  # URL for the Telegram API endpoint
        for i, pdf_file in enumerate(pdf_paths):
            with open(pdf_file, "rb") as file:
                # Open the PDF file in binary mode
                files = {"document": file}  # Prepare the file data for the request
                data = {"chat_id": self.CHAT_ID, "caption": caption_list[i]}  # Prepare the request data
                with requests.post(url, files=files, data=data) as response:
                    # Send the POST request to the Telegram API
                    self.docker_print(f"Status Code for {pdf_file}: {response.status_code}")  # Print the status code for the request
                    if response.status_code == 200:
                        # If the status code is 200 (OK), the file was sent successfully
                        self.docker_print(f"File {pdf_file} sent successfully!")
                    else:
                        # If the status code is not 200, there was an error
                        self.docker_print(f"Error sending {pdf_file}: {response.status_code} - {response.text}")
                        # Print the error code and the error message from the Telegram API
            time.sleep(168)  # Wait for 168 seconds

    def telegram_send_group_pdfs(self, pdf_paths, caption_list):
        """Sends multiple PDFs as a group (album) to a Telegram chat.

        Args:
            pdf_paths: A list of paths to the PDF files.
            caption_list: A list of captions for each PDF.
        """

        media = []
        for i, pdf_path in enumerate(pdf_paths):
            with open(pdf_path, 'rb') as f:
                # Open the PDF file in binary mode
                data = {'type': 'document', 'media': f'attach://document{i}', 'caption': caption_list[i]}
                # Prepare the data for the current file
                media.append(data)  # Add the file data to the media list

        data = {'chat_id': self.CHAT_ID, 'media': json.dumps(media)}  # Prepare the request data
        url = f'https://api.telegram.org/bot{self.BOT_TOKEN}/sendMediaGroup'  # URL for the Telegram API endpoint

        with requests.post(url, data=data, files={f'document{i}': open(pdf_path, 'rb') for i, pdf_path in enumerate(pdf_paths)}) as response:
            # Send the POST request to the Telegram API
            self.docker_print(f"Status Code: {response.status_code}")  # Print the status code for the request
            if response.status_code == 200:
                # If the status code is 200 (OK), the files were sent successfully
                self.docker_print("Files sent successfully!")
            else:
                # If the status code is not 200, there was an error
                self.docker_print(f"Error: {response.status_code} - {response.text}")
                # Print the error code and the error message from the Telegram API

    def telegram_send_images(self, image_paths, caption_list):
        """Sends multiple images individually to a Telegram chat.

        Args:
            image_paths: A list of paths to the image files.
            caption_list: A list of captions for each image.
        """

        url = f"https://api.telegram.org/bot{self.BOT_TOKEN}/sendPhoto"  # URL for the Telegram API endpoint

        for i, photo_file in enumerate(image_paths):
            with open(photo_file, "rb") as file:
                # Open the image file in binary mode
                files = {"photo": file}  # Prepare the file data for the request
                data = {"chat_id": self.CHAT_ID, "caption": caption_list[i]}  # Prepare the request data
                with requests.post(url, files=files, data=data) as response:
                    # Send the POST request to the Telegram API
                    if response.status_code == 200:
                        # If the status code is 200 (OK), the image was sent successfully
                        self.docker_print(f"Image {photo_file} sent successfully!")
                    else:
                        # If the status code is not 200, there was an error
                        self.docker_print(f"Error sending {photo_file}: {response.status_code} - {response.text}")
                        # Print the error code and the error message from the Telegram API

    def telegram_send_group_images(self, image_paths, caption_list):
        """Sends multiple images as a group (album) to a Telegram chat.

        Args:
            image_paths: A list of paths to the image files.
            caption_list: A list of captions for each image.
        """

        media = []
        for i, image_path in enumerate(image_paths):
            with open(image_path, 'rb') as f:
                # Open the image file in binary mode
                data = {'type': 'photo', 'media': f'attach://photo{i}', 'caption': caption_list[i]}
                # Prepare the data for the current image
                media.append(data)  # Add the image data to the media list

        data = {'chat_id': self.CHAT_ID, 'media': json.dumps(media)}  # Prepare the request data
        url = f'https://api.telegram.org/bot{self.BOT_TOKEN}/sendMediaGroup'  # URL for the Telegram API endpoint

        with requests.post(url, data=data, files={f'photo{i}': open(image_path, 'rb') for i, image_path in enumerate(image_paths)}) as response:
            # Send the POST request to the Telegram API
            self.docker_print(f"Status Code: {response.status_code}")  # Print the status code for the request
            if response.status_code == 200:
                # If the status code is 200 (OK), the images were sent successfully
                self.docker_print("Images sent successfully!")
            else:
                # If the status code is not 200, there was an error
                self.docker_print(f"Error: {response.status_code} - {response.text}")
                # Print the error code and the error message from the Telegram API

    def prep_sp500_dataset(self):
        """
        Prepares a dataset containing S&P 500 stock information.
        Downloads historical candlestick data for each stock and stores it in a dictionary.
        Saves the processed dataframe to a CSV file.
        """

        # Initialize PyTickerSymbols object to fetch stock data
        stock_data = PyTickerSymbols()

        # Fetch stock data for S&P 500
        sp500_df = pd.DataFrame(list(stock_data.get_stocks_by_index('S&P 500')))

        # Remove duplicate entries based on 'symbol'
        sp500_df = reset_dataframe_index(sp500_df).groupby('symbol').first().reset_index()

        # Download historical candlestick data for all symbols using yfinance
        # The period is set to twice the number of candlestick charts required
        candlestick_df = yf.download(list(sp500_df['symbol'].values),
                                    period=f'1y', 
                                    interval="1d")

        # Extract date list from candlestick data index
        date_list = [str(each)[:10] for each in list(candlestick_df['Close'].index)]

        # Initialize an empty dictionary to store candlestick data for each ticker
        self.sp500_df_dict = {}

        # Iterate over each ticker symbol
        # for each_ticker in tqdm(sp500_df['symbol'].values, desc="Iterate over each ticker symbol"):
        for each_ticker in tqdm(sp500_df['symbol'].values[:20], desc="Iterate over each ticker symbol"):
            try:
                # Create a temporary dictionary with candlestick data for the current ticker
                temp_dict = {'Date': date_list,
                            'Open': list(candlestick_df['Open'][each_ticker].values),
                            'High': list(candlestick_df['High'][each_ticker].values),
                            'Low': list(candlestick_df['Low'][each_ticker].values),
                            'Close': list(candlestick_df['Close'][each_ticker].values),
                            'Volume': list(candlestick_df['Volume'][each_ticker].values),
                            }
                # Convert the temporary dictionary to a Pandas DataFrame
                temp_df = pd.DataFrame(temp_dict)

                # Check if the first closing price is not NaN (meaning data is available)
                if pd.notna(temp_df['Close'].values[0]):
                    # If data is available, add it to the dictionary with the ticker as the key
                    self.sp500_df_dict[each_ticker] = temp_df
            except:
                pass

        # Filter the sp500_df to include only tickers with valid candlestick data
        self.sp500_df = sp500_df[sp500_df['symbol'].isin(list(self.sp500_df_dict.keys()))]
        
        # Reset the index of the filtered dataframe
        self.sp500_df = reset_dataframe_index(self.sp500_df)

        # Combine multiple industries and indices into single semicolon-separated strings
        self.sp500_df['indices'] = self.sp500_df['indices'].apply(lambda x: ';'.join(x))
        self.sp500_df['industries'] = self.sp500_df['industries'].apply(lambda x: ';'.join(x))

        # Save the processed dataframe to a CSV file
        self.sp500_df.to_csv('data/csv/sp500_df.csv', index=False)

        # Create dictionaries to map tickers to sectors and company names for later use
        self.ticker_sector_dict = dict(zip(self.sp500_df['symbol'], self.sp500_df['industries']))
        self.ticker_company_dict = dict(zip(self.sp500_df['symbol'], self.sp500_df['name']))

    def docker_print(self, txt):
            """
            Prints the given text to the console with immediate flushing.

            This is particularly useful when running within a Docker container 
            as it ensures logs are streamed and visible in real-time. 

            Args:
                txt (str): The text to be printed.
            """
            print(txt, flush=True)

    def get_judge_instructions(self):
        """
        Provides judging instructions for evaluating submissions.

        Returns:
            str: A formatted string containing detailed judging criteria and guidelines.
        """
        system_instructions = f"""
        ## JUDGING ROUND

        Sponsor will evaluate each Entrant and their Submission. Your Submission, including Your video and code will be evaluated based on following judging criteria (the “Judging Criteria”), weighted equally:

        * Submissions will be evaluated by Google judges who excel in the following five (5) categories as they relate to this challenge: impact, remarkability, creativity, usefulness, and execution. Each criteria will be scored on a scale of 1 (strongly disagree) to 5 (strongly agree). 

        The judging criteria is as follows:

        **Category 1: Impact (maximum 15 points)**

        * Is the solution easy and enjoyable to use for everyone, including people with disabilities? (maximum 5 points)
        * Does this solution have potential to contribute meaningfully to environmental sustainability? (maximum 5 points)
        * Does this solution have potential to contribute meaningfully to improving people's lives? (maximum 5 points)

        **Category 2: Remarkability (maximum 10 points)**

        * Is the submission surprising to those that are well-versed in Large Language Models (“LLM”)? (maximum 5 points)
        * Is the submission surprising to those that are not well-versed in LLM? (maximum 5 points)

        **Category 3: Creativity (maximum 15 points)**

        * Does the submission differ from existing, well known, applications in functionality? (maximum 5 points)
        * Does the submission differ from existing, well known, applications in user experience? (maximum 5 points)
        * Is the submission implemented through the use of creative problem-solving approaches? (maximum 5 points)

        **Category 4: Usefulness (maximum 15 points)**

        * Does the submission include a well-defined target user persona/segmentation? (maximum 5 points)
        * Does the submission identify how the solution addresses specific user needs? (maximum 5 points)
        * How well does the solution, as implemented, help users meet these needs? (maximum 5 points)

        **Category 5: Execution (maximum 10 points)**

        * Is the solution well-designed and adhere to software engineering practices? (maximum 5 points)
        * Is the LLM component of the solution well-designed and adhere to Machine Learning (ML)/LLM best practices? (maximum 5 points)

        **Maximum score: 65**

        *Best Overall Submission will be determined by the Entrant who has the highest score in the combined categories of Impact, Creativity and Usefulness. In the event of a tie, Sponsor will determine the Best Overall Submission Prize. Sponsor’s decision is final and binding.* 
        """
        return system_instructions

    def get_sector_text(self):
        """
        Returns a formatted string of randomly shuffled market sectors.

        Returns:
            str: A string of market sectors, each starting with "- " and separated by a newline. 
        """

        # Randomly shuffle the list of sectors
        random.shuffle(self.sector_list)

        # Join the shuffled sectors with a newline and "- " prefix
        return "- " + "\n- ".join(self.sector_list)
    
    def get_system_instructions_1(self):
        """
        Constructs and returns a detailed set of instructions for a language model simulating a meeting of a fictional investment board.

        This function crafts a comprehensive narrative for a hypothetical meeting of the "Gemini Candlestick Investment Fund" board of directors. 
        The narrative includes:

            * Contextual information about the fund, its investment strategy, and target user persona.
            * Meeting specifics like date, time, location, and participating board members.
            * Details about the stocks under consideration and the data available for analysis.
            * Explicit instructions for structuring the meeting minutes, including sections for broad market analysis, in-depth stock discussions, consensus building, and action planning.

        Returns:
            str: A formatted string containing the system instructions for the language model.
        """
        system_instructions = f"""
        **Google Gemini Candlestick Investment Fund**
        
        The Gemini Candlestick Investment Fund employs a dynamic investment strategy that capitalizes on both market growth and volatility. The Fund strategically allocates capital to a portfolio of core holdings, representing established companies with long-term growth potential, while simultaneously employing tactical opportunities to profit from short-term market fluctuations. This dual approach allows the Fund to navigate market cycles effectively, aiming to maximize returns through both growth investments and market-neutral strategies. Guided by the collective wisdom of the ten-member Board of Directors, this strategy ensures a balanced and diversified approach to investment, seeking to generate consistent and sustainable profitability for its stakeholders.

        **Target User Persona**
        
        This application is designed for financial analysts, portfolio managers, and individuals interested in understanding how technical analysis and expert opinions can be integrated to make informed investment decisions.

        **User Needs**
        
        Users require a clear and concise summary of expert opinions on various stocks and overall market sectors based on technical indicators and market trends. This information aids in making well-informed investment decisions.

        **Gemini Candlestick Investment Fund Daily Meeting Minutes**
        
        **Date:** {self.current_meeting_date}
        **Time:** 00:00 - 02:00  
        **Location:** Google Meet

        ### **Stock Selection**
        
        The Board of Directors will analyze the following {len(self.company_ticker_list)} stocks:

        *{self.company_ticker_text}

        ### **Data**
 
        The Board will have access to hypothetical candlestick charts and technical indicator data for each of these stocks, reflecting recent market activity. The technical indicators include:
        
        - **Volume**
        - **Relative Strength Index (RSI)**
        - **Moving Average Convergence Divergence (MACD)**
        - **Bollinger Bands**
        - **Fibonacci Retracement**
        - **Ichimoku Cloud**
        - **Stochastic Oscillator**
        - **Chaikin Money Flow**
        - **On-Balance Volume**
        - **Average True Range (ATR)**

        ### **Board Members**
        
        The Board of Directors comprises 10 esteemed members, each bringing technical expertise and a deep understanding of market sectors and trends:
        
        1. **Munehisa Homma** *(Chairman)*: Assigned to analyze Candlestick Pattern Recognition and provide an opinion.
        2. **John Bollinger**: Assigned to analyze the Bollinger Bands indicator and provide an opinion.
        3. **Leonardo Pisano Fibonacci**: Assigned to analyze Fibonacci Retracement and provide an opinion.
        4. **Ralph Nelson Elliott**: Assigned to analyze potential Elliott Wave patterns, utilizing Fibonacci ratios (0.786, 0.618, 0.5, 0.382, and 0.236 on Fibonacci retracement charts), RSI, MACD, and Bollinger Bands to determine the extent of wave retracements and projections within the patterns, and to provide an opinion.
        5. **Goichi Hosoda**: Assigned to analyze the Ichimoku Cloud and provide an opinion.
        6. **Joseph Granville**: Assigned to analyze the Volume and On-Balance Volume indicators and provide an opinion.
        7. **Gerald Appel**: Assigned to analyze the MACD indicator and provide an opinion.
        8. **J. Welles Wilder**: Assigned to analyze RSI and ATR indicators and provide an opinion.
        9. **George Lane**: Assigned to analyze the Stochastic Oscillator indicator and provide an opinion.
        10. **Marc Chaikin**: Assigned to analyze the Chaikin Money Flow indicator and provide an opinion.

        ### **Instructions**

        1. **Broad Market Analysis**
        
            - **Sectors to Analyze**:

                - **Healthcare & Biotechnology**
                - **Information Technology**
                    - **Software & Services**
                    - **Hardware & Equipment**
                    - **Semiconductors**
                    - **Internet Services & Infrastructure**
                    - **Artificial Intelligence & Machine Learning**
                    - **Cybersecurity**
                - **Financials**
                    - **Financial Services**
                    - **FinTech**
                - **Energy**
                    - **Traditional Energy**
                    - **Renewable Energy**
                - **Consumer Discretionary**
                    - **Traditional Retail**
                    - **E-Commerce**
                    - **Automotive**
                - **Communication Services**
                    - **Media & Entertainment**
                - **Industrials**
                - **Real Estate**
                - **Transportation**
                - **Materials**
                - **Utilities**
                - **Environmental, Social, and Governance (ESG) Focused Companies**

            - **Guidelines**:
 
                - Begin the meeting minutes with a concise analysis of each sector.
                - Synthesize the Board's collective opinion on each sector, incorporating relevant technical indicator observations and potential trends.
                - Utilize professional and business-oriented language, incorporating critical analysis that establishes connections among significant sectors.
                - Generate critical thoughts on the interconnections between sectors in the Broad Market Analysis.

        2. **Discussion**

            - Following the broad market analysis, describe a hypothetical discussion flow focusing on specific stock analysis examples.
            - **Technical Instances**:
            
                - For example, "{self.example_analysis_text}"

            - **Guidelines**:
 
                - Create a narrative for the meeting, incorporating various opinions and discussions, including agreements and disagreements among board members.
                - Connect board member expertise with specific ticker symbols from the list.
                - Highlight at least one stock from each market sector.

        3. **Consensus & Action**
 
            - **Market Sentiment**
 
                - The Board assessed the overall market sentiment for relevant sectors, considering factors such as economic indicators, sector trends, and investor sentiment surveys.
                - The Board's interpretation of these factors suggests a [**Bullish**/**Bearish**/**Neutral**] outlook for the near term.

            - **Fund Position**
 
                - **Based on the analysis of market sentiment and identified opportunities, the Board has determined the following strategy**:
    
                    - **Adopting bullish positions**: To capitalize on potential market growth.
                    - **Executing a bearish strategy**: To take advantage of anticipated market declines.
                    - **Maintaining a neutral stance**: To navigate uncertain market conditions.
                    - **Focusing on specific sectors**: To maximize potential returns.
                    - **Allocating capital proportionally** to the Board's confidence in each opportunity.

                - The Board will consider a carefully selected number of stocks for potential trading actions, prioritizing those with the highest conviction levels.

                - **Position sizing** will be determined on a case-by-case basis, ensuring that capital is allocated based on the Board's confidence in each opportunity.

                - **Alternative Strategies Considered**

                    - Briefly list alternative approaches that were discussed and the rationale for not selecting them at this time, demonstrating thoroughness in the decision-making process.

                - **Risk Management**

                    - The Board acknowledges the inherent risks associated with the chosen strategy and has implemented appropriate measures to mitigate potential losses, such as:

                        - Stop-loss orders
                        - Diversification across sectors
                        - Hedging strategies
                        - Position limits

                - **Monitoring & Review**

                    - The Board will actively monitor market conditions and the performance of selected positions.
                    - The Board will reconvene in [**timeframe**, e.g., one week, two weeks] to review the Fund's positions and make any necessary adjustments based on evolving market dynamics and new information.

        4. **Ticker Symbols of Interest**

            - Based on the discussion, list a few of the ticker symbols that were highlighted (no more than 16 symbols).
            - **Provide Reasons**:
 
                - Each reason should directly relate to the analysis conducted by the board members.
                - Ensure proportional diversity among the different technical analyses:
 
                    - Candlestick Pattern Recognition
                    - Bollinger Bands indicator
                    - Fibonacci Retracement analysis
                    - Elliott Wave patterns analysis
                    - Ichimoku Cloud analysis
                    - Volume and On-Balance Volume indicators
                    - MACD indicator analysis
                    - RSI and ATR indicators analysis
                    - Stochastic Oscillator indicator
                    - Chaikin Money Flow indicator

        5. **Further Action**
        
            - The Board instructed the Fund's management team to execute the agreed-upon market positions.
            - Further investigate the highlighted ticker symbols for potential investment actions aligned with the Fund's overall strategy.
        
        6. **Meeting Adjourned**
        
            - **Time**: 02:00
        
        7. **Approved by**
        
            - **Munehisa Homma**, Chairman
        """
        return system_instructions
    def get_system_instructions_2(self):
        """
        Constructs detailed system instructions for summarizing a Gemini Candlestick 
        Investment Fund meeting based on provided meeting minutes.

        Returns:
            str: The comprehensive system instructions for the language model.
        """
        system_instructions = f"""
        **Gemini Candlestick Investment Fund Daily Meeting Summary**
        **Date:** {self.current_meeting_date}
        **Time:** 00:00 - 02:00

        **Context:**
        
        You are provided with detailed minutes from the latest Gemini Candlestick Investment Fund meeting. Your task is to analyze this information and create a concise and insightful summary for the fund's managers.

        ### **Objective**
        
        - Summarize the provided meeting minutes, focusing on key decisions, market observations, and interconnections between sectors.
        - Extract impactful insights relevant to the Gemini Candlestick Investment Fund's investment strategy.

        ### **Instructions**
        
        1. **Executive Summary**
        
            Begin with a brief executive summary that encapsulates the overall meeting's key points:

            - **Overall Market Sentiment:** Describe the general market outlook as discussed by the Board, including whether it is bullish, bearish, or neutral, and the rationale behind this sentiment.
            - **Investment Strategy:** Summarize the Fund's chosen investment strategy moving forward, including any strategic shifts or continuations of current approaches, and justify why this strategy was selected.
            - **Risk Management Measures:** Outline the risk management strategies being implemented to mitigate potential losses or capitalize on market movements.

            
        2. **Detailed Sector Analysis**
 
            For each sector discussed, provide a structured analysis:

            - **Sector Overview:**
                - **Perceived Strength or Weakness:** Summarize the Board's assessment of the sector's current performance and outlook.
                - **Key Technical Indicators:** Highlight the specific technical indicators and patterns that influenced the Board's view (e.g., RSI, MACD, Bollinger Bands).
            - **Companies of Interest:**
                - **Specific Ticker Symbols Mentioned:** List the companies highlighted within the sector.
                - **Rationale for Inclusion:** Provide reasons why these companies were discussed, including any significant candlestick patterns or indicator signals.

        3. **Interconnections Between Sectors**
        
            Analyze and explain the key interconnections observed between different sectors:
        
            - **Cross-Sector Influences:** Describe how developments in one sector may impact others (e.g., how advancements in Information Technology could influence Consumer Discretionary spending).
            - **Market Dynamics:** Explain how these interconnections affect the Fund's investment decisions and strategies.
            - **Economic Indicators:** Include any broader economic factors that interplay between sectors (e.g., interest rates, regulatory changes).

        4. **Actionable Insights and Recommendations**
        
            Based on the meeting discussions, extract actionable takeaways:
        
            - **Emerging Investment Opportunities:** Identify sectors or companies that present potential growth opportunities.
            - **Potential Risks:** Highlight any risks or challenges noted by the Board, including market volatility or unfavorable trends.
            - **Strategic Adjustments:** Suggest any recommended adjustments to the Fund's portfolio or strategy, supported by the Board's analysis.

        5. **Language and Presentation**
        
            - **Clarity and Professionalism:** Use clear, professional, and business-oriented language suitable for senior fund managers.
            - **Accessibility:** Avoid excessive technical jargon; where technical terms are necessary, provide brief explanations.
            - **Conciseness:** Ensure the summary is succinct, focusing on the most critical and impactful information.
            - **Structure:** Use headings, subheadings, and bullet points as appropriate to enhance readability.

        6. **Concluding Remarks**
        
            - Provide a brief conclusion that summarizes the Fund's strategic position and outlook.
            - Reinforce the key messages and next steps for the management team.

        ### **Example Insights**
        
        - **Market Sentiment:** "The Board expresses a cautiously bullish outlook, driven by strong technical indicators in the Information Technology and Renewable Energy sectors, while remaining vigilant of potential volatility in the Financials sector."
        
        - **Sector Interconnection:** "Growth in the Renewable Energy sector is anticipated to benefit the Industrials sector, particularly companies involved in manufacturing components for sustainable technologies."
        
        - **Actionable Takeaway:** "The Fund should consider increasing exposure to semiconductor companies like **NVIDIA Corporation (NVDA)**, as technical analysis indicates a potential upward trend supported by robust RSI and MACD signals."

        **Note:** Ensure that all insights and recommendations are directly supported by the discussion points and analyses presented in the meeting minutes.

        ### **Additional Guidance**
        
        - **Confidentiality:** Maintain the confidentiality of sensitive information contained within the meeting minutes.
        - **Accuracy:** Verify that all interpretations and summaries accurately reflect the Board's discussions and intentions.
        - **Objective Tone:** Present the information objectively, without personal bias or unsupported opinions.

        By following these instructions, you'll produce a comprehensive and actionable summary that equips the Fund's management team with the essential insights needed to make informed decisions aligned with the Gemini Candlestick Investment Fund's strategic objectives.
        """
        return system_instructions
    def get_system_instructions_3(self):
        """
        Provides system instructions for extracting ticker symbols.

        Returns:
            str: Instructions for extracting ticker symbols in JSON format.
        """
        system_instructions = """
        Your role is to extract ticker symbols of "Ticker Symbols of Interest" into a JSON object format.

        **JSON FORMAT**
        {
            "TickerSymbolsOfInterest": [
                "symbol_1",
                "symbol_2",
                "symbol_3",
                ...
            ]
        }
        """
        return system_instructions

    def get_system_instructions_4(self):
        """
        Provides system instructions for converting meeting minutes to HTML.

        Returns:
            str: Instructions for converting meeting minutes to HTML format.
        """
        system_instructions = """
        Your responsibility is to convert meeting minutes into a professional HTML format suitable for web publication, maintaining a formal and polished tone throughout the process.
        """
        return system_instructions

    def get_system_instructions_5(self):
        """
        Provides system instructions about enclosed candlestick chart documents.

        Returns:
            str: Instructions describing the enclosed candlestick chart documents.
        """
        system_instructions = """
        **Important Notice:** The enclosed documents contain 1-day candlestick charts and corresponding technical indicators for the ticker symbols listed in the "Ticker Symbols of Interest" section, which can be found on the subsequent pages.
        """
        return system_instructions

    def get_system_instructions_6(self):
        """
        Provides system instructions for rewording user statements professionally.

        Returns:
            str: Instructions for rewording user statements with professionalism.
        """
        system_instructions = """
        One of the key responsibilities associated with your position is to rephrase the statements provided by users, ensuring that the revised wording maintains a suitable level of professionalism and outputs it in markdown format.
        """
        return system_instructions

    def get_system_instructions_7(self):
        """
        Provides system instructions for enhancing Telegram messages creatively.

        Returns:
            str: Instructions for enhancing Telegram messages with emojis and flair.
        """
        system_instructions = """
        As an enhancer of Telegram messages, your objective is to captivate readers and amplify user input. By incorporating emojis, relevant symbols, and creative flair, you strive to ignite curiosity and inspire individuals to explore the original messages. You're limited to few sentences to write and wrap up the enhancements, making every word count.
        """
        return system_instructions
    def clean_minutes_text(self, text):
        # Check for the first occurrence of either substring
        start_index_list = [text.find(each) for each in self.text_start_index_list]

        # Find the first valid index
        start_index = -1
        for index in start_index_list:
            if index != -1 and (start_index == -1 or index < start_index):
                start_index = index

        # Extract the substring if one of the patterns is found
        if start_index != -1:
            updated_text = text[start_index:]
        else:
            updated_text = text  # Keep original if neither is found
        text = updated_text

        # Find the position of the substring "Munehisa Homma, Chairman"
        marker = ", Chairman"
        index = text.find(marker)
        if index != -1:
            # Get all text up to and including "Munehisa Homma, Chairman"
            text_with_marker = text[:index + len(marker)]
        else:
            # Handle case where marker is not found
            text_with_marker = text
        text = text_with_marker
        return text

    def get_generation_config_1(self):
        # Set up the model parameters for Gemini
        generation_config = {
            "temperature": 1,  # Temperature controls the randomness of the generated text
            "top_p": 0.95,  # Top_p controls the diversity of the generated text
            "top_k": 40,  # Top_k controls the number of possible next words considered
            "max_output_tokens": 1000000,  # Maximum number of tokens allowed in the generated text
            "response_mime_type": "text/plain",  # Response format
        }
        return generation_config
    
    def get_generation_config_2(self):
        # Set up the model parameters for Gemini
        generation_config = {
            "temperature": 1,  # Temperature controls the randomness of the generated text
            "top_p": 0.95,  # Top_p controls the diversity of the generated text
            "top_k": 40,  # Top_k controls the number of possible next words considered
            "max_output_tokens": 168,  # Maximum number of tokens allowed in the generated text
            "response_mime_type": "text/plain",  # Response format
        }
        return generation_config
    
    def generate_gemini_candlestick(self):
        """Generates reports and summaries based on candlestick chart data using the Gemini API."""

        # Configure the Gemini API
        genai.configure(api_key=self.gemini_key)

        # Set up the model parameters for Gemini
        generation_config = self.get_generation_config_1()
        # Configure safety settings for the Gemini model
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        try:
            # Define the directory where candlestick charts are saved
            charts_directory = 'data/png'
            # Check if the charts directory is empty
            if len(os.listdir(charts_directory)) == 0:
                # Prepare the S&P 500 datasets for analysis
                self.prep_sp500_dataset()
                # If the directory is empty, generate candlestick charts for all tickers
                for each_ticker in tqdm(self.sp500_df['symbol'].values, desc="Generating get_candlestick_image()"):
                    self.ticker = each_ticker
                    self.get_candlestick_image()
            # Generate charts on days other than Sunday and Monday
            elif self.today_date.lower() not in ['sunday', 'monday']:
                # Prepare the S&P 500 datasets for analysis
                self.prep_sp500_dataset()
                for each_ticker in tqdm(self.sp500_df['symbol'].values, desc="Generating get_candlestick_image()"):
                    self.ticker = each_ticker
                    self.get_candlestick_image()

            # Initialize an empty list to store prompt parts
            self.prompt_parts = []
            
            # Iterate over each ticker in the ticker list with a progress bar
            for each_ticker in tqdm(self.ticker_list, desc="genai.upload_file"):
                # Try uploading the chart image for the current ticker 6 times
                for _ in range(6):
                    try:
                        # Set the current ticker
                        self.ticker = each_ticker
                        
                        # Upload the chart image to GenAI and store the returned object
                        temp_file = genai.upload_file(
                            path=f"data/png_optimized/{each_ticker}.png",  # Path to the image file
                            display_name=f'{self.ticker_company} ({each_ticker}): 1d Candlestick Chart (with Technical Indicators)'  # Display name for the image
                        )
                        
                        # Append the uploaded file object to the prompt parts list
                        self.prompt_parts.append(temp_file)
                        
                        # Break the inner loop if the upload is successful
                        break
                    except Exception as e:
                        # Log the error and continue to the next iteration without crashing
                        self.docker_print(f"Error during genai upload_file: {e}")
                        # If an exception occurs, pass and try again (up to 6 times)
                        pass

            minutes_text_list = []  # Initialize an empty list to store generated meeting minutes
            summary_text_list = []  # Initialize an empty list to store generated summaries

            # Generate 36 candidate meeting minutes and summaries
            for _ in tqdm(range(36), desc="Generating temp_minutes_text and temp_summary_text"):
                try:
                    # Introduce a delay between API calls to avoid rate limiting (except for the first iteration)
                    if _ != 0:
                        time.sleep(4)

                    # Shuffle the prompt components to introduce variation in the generated text
                    random.shuffle(self.prompt_parts)

                    # Generate meeting minutes using the Gemini Pro model
                    temp_minutes_text = str(genai.GenerativeModel(
                        model_name="gemini-1.5-flash-latest",  # Specify the model name
                        generation_config=generation_config,   # Pass the generation configuration
                        # system_instruction=self.get_system_instructions_1(),  # Set the system instructions for minutes generation
                        safety_settings=safety_settings       # Enforce safety settings during text generation
                    ).generate_content([self.get_system_instructions_1()] + self.prompt_parts, request_options={"timeout": 1000}).text) 

                    # Introduce a delay to avoid rate limiting
                    time.sleep(4)

                    # Generate a summary of the meeting minutes
                    temp_summary_text = str(genai.GenerativeModel(
                        model_name="gemini-1.5-flash-latest",
                        generation_config=generation_config,
                        system_instruction=self.get_system_instructions_2(),  # Set system instructions for summary generation
                        safety_settings=safety_settings
                    ).generate_content(temp_minutes_text).text)

                    # Apply quality checks to ensure the generated text meets specific criteria
                    # Check if the generated text does not contain '[' or ']' characters
                    if '[' not in temp_minutes_text and ']' not in temp_minutes_text:
                        if '[' not in temp_summary_text and ']' not in temp_summary_text:
                            # Check if the minutes contain specific keywords
                            if 'Ticker Symbols of Interest'.lower() in temp_minutes_text.lower():
                                if 'approved by' in temp_minutes_text.lower():
                                    self.text_start_index_list = ["**Gemini", "# Gemini", "## Gemini", "### Gemini"]
                                    # Check if any of the items in self_text_start_index_list are in temp_minutes_text
                                    if any(item in temp_minutes_text for item in self.text_start_index_list):
                                        if ", Chairman" in temp_minutes_text:
                                            # Append the generated minutes and summary to their respective lists
                                            temp_minutes_text = self.clean_minutes_text(temp_minutes_text)
                                            minutes_text_list.append(temp_minutes_text)
                                            summary_text_list.append(temp_summary_text)
                except Exception as e:
                    # Handle exceptions during text generation and print error messages
                    self.docker_print(f"Error during temp_minutes_text generation: {e}")
                    pass

            # Initialize empty lists to store filtered minutes and summaries
            minutes_text_list_2 = []
            summary_text_list_2 = []

            # Iterate through each minute in the original list
            for i_each_minutes, each_minutes in enumerate(minutes_text_list):
                # Initialize a score to track the presence of sectors
                sector_score = 0

                # Check if each sector is present in the minutes text
                for each_sector in self.sector_list:
                    if each_sector.lower() in each_minutes.lower():
                        sector_score += 1

                # If all sectors are present, add the minutes and corresponding summary to the filtered lists
                if sector_score == len(self.sector_list):
                    minutes_text_list_2.append(each_minutes)
                    summary_text_list_2.append(summary_text_list[i_each_minutes])

            # Update the original lists with the filtered results
            minutes_text_list = minutes_text_list_2.copy()
            summary_text_list = summary_text_list_2.copy()

            # Initialize an empty list to store ticker scores for each minute segment
            score_minutes_text_list = []

            # Iterate over each minute segment in the transcript
            for each_i in minutes_text_list:
                # Find the starting index of the phrase "Ticker Symbols of Interest" in lowercase
                start_index = each_i.lower().find("Ticker Symbols of Interest".lower())

                # Initialize the ticker score for the current minute segment to 0
                score_i = 0

                # Iterate over each ticker in the predefined ticker list
                for each_j in self.ticker_list:
                    # Check if the ticker is mentioned before the "Ticker Symbols of Interest" phrase
                    if each_j in each_i[:start_index]:
                        # Increment the ticker score if a match is found
                        score_i = score_i + 1

                # Append the ticker score for the current minute segment to the list
                score_minutes_text_list.append(score_i)
            
            list_of_interest_ticker_list = []  # List to store lists of tickers of interest
            score_minutes_text_list_2 = []  # List to store scores corresponding to the processed minutes
            minutes_text_list_3 = []       # List to store minutes text after ticker extraction
            summary_text_list_3 = []        # List to store summary text corresponding to the processed minutes

            # Iterate through each minutes text in the input list
            for i, each_i in enumerate(tqdm(minutes_text_list, desc="Generating interest_ticker_list")):
                # Retry ticker extraction up to 6 times
                for _ in range(6):  
                    try:
                        time.sleep(4)  # Wait for 4 seconds to avoid rate limiting

                        # Find the starting index of "Ticker Symbols of Interest" section
                        start_index = each_i.lower().find("Ticker Symbols of Interest".lower()) 
                        
                        # Generate content from the "Ticker Symbols of Interest" section using the generative model
                        generation_result = genai.GenerativeModel(
                            model_name="gemini-1.5-flash-latest",
                            generation_config=generation_config,
                            system_instruction=self.get_system_instructions_3(), 
                            safety_settings=safety_settings
                        ).generate_content(each_i[start_index:]).text

                        # Extract ticker list from the generated content
                        interest_ticker_list = str(generation_result)
                        extracted_data = self.extract_json(interest_ticker_list)[0] 
                        key = list(extracted_data.keys())[0]
                        interest_ticker_list = extracted_data[key]
                        
                        # Filter the ticker list to include only valid tickers
                        interest_ticker_list = [ticker for ticker in interest_ticker_list if ticker in self.ticker_list] 

                        # Append the extracted data to corresponding lists
                        list_of_interest_ticker_list.append(interest_ticker_list)
                        score_minutes_text_list_2.append(score_minutes_text_list[i])
                        minutes_text_list_3.append(each_i)
                        summary_text_list_3.append(summary_text_list[i])
                        
                        break  # Exit the retry loop if successful

                    except Exception as e: 
                        print(f"Error during ticker generation: {e}")
                        pass  # Continue to the next retry if an error occurs

            # Assign a score to each ticker in the list based on its length.
            # Ticker with length <= 24 get a score of 1, otherwise -1
            list_of_interest_ticker_score_list = [1 if len(each) <= 24 else -1 for each in list_of_interest_ticker_list]
            
            # Convert the lists to NumPy arrays for element-wise multiplication
            list_of_interest_ticker_score_list = np.array(list_of_interest_ticker_score_list)
            score_minutes_text_list_2 = np.array(score_minutes_text_list_2)

            # Multiply the scores with the corresponding text scores
            # This will prioritize texts related to tickers with shorter lengths
            score_minutes_text_list = score_minutes_text_list_2 * list_of_interest_ticker_score_list

            # Find the index with the highest combined score
            max_score_index = np.argmax(score_minutes_text_list)

            # Select the minutes text, summary text, and the interest ticker based on the index
            minutes_text = minutes_text_list_3[max_score_index]
            summary_text = summary_text_list_3[max_score_index]
            self.interest_ticker_list = list_of_interest_ticker_list[max_score_index]

            # Define the file path for saving the meeting minutes.
            file_path = f'data/txt/{self.file_date}_minutes.txt'
            # Open the file in write mode ('w'). 
            with open(file_path, 'w') as file:
                # Write the meeting minutes text to the file.
                file.write(minutes_text)

            # Define the file path for saving the meeting minutes.
            file_path = f'data/txt/{self.file_date}_summary.txt'
            # Open the file in write mode ('w'). 
            with open(file_path, 'w') as file:
                # Write the meeting minutes text to the file.
                file.write(summary_text)

            # Delete uploaded files from Gemini
            for each_prompt_part in tqdm(self.prompt_parts, desc="genai.delete_file"):
                try:
                    genai.delete_file(each_prompt_part.name)
                except:
                    pass

            # Generate attached text with retries
            # for _ in tqdm(range(6), desc="Generating attached_text"):  # Try up to 6 times
            #     try:
            #         time.sleep(4)  # Wait for 4 seconds before making the API call

            #         # Generate content using the Gemini Pro model
            #         genai_model = genai.GenerativeModel(
            #             model_name="gemini-1.5-flash-latest",  # Specify the Gemini Pro model
            #             generation_config=generation_config,
            #             system_instruction=self.get_system_instructions_6(),  # Set system instructions
            #             safety_settings=safety_settings
            #         )
            #         attached_text = str(genai_model.generate_content(
            #             self.get_system_instructions_5()  # Provide instructions for content generation
            #         ).text)

            #         break  # Exit the loop if generation is successful

            #     except Exception as e:
            #         self.docker_print(f"Error during attached_text generation: {e}")
            #         pass  # Continue to the next iteration if an error occurs
            
            # Convert the combined text to PDF and save it to the specified path.
            self.markdown_to_pdf(minutes_text, 'data/pdf/minutes.pdf')
            # self.markdown_to_pdf(minutes_text + '\n' + attached_text, 'data/pdf/minutes.pdf')

            # Call the markdown_to_pdf method to convert the summary text to a PDF file
            self.markdown_to_pdf(summary_text, 'data/pdf/summary.pdf')

            # Convert the PNG files to PDF
            png_files = [f'data/png_optimized/{each}.png' for each in self.interest_ticker_list]
            with open("data/pdf/png.pdf", "wb") as pdf_file:
                pdf_bytes = convert(png_files)  # Assuming 'convert' is a function to convert PNGs to PDF
                pdf_file.write(pdf_bytes)
            # Merge the generated PDF files
            pdf_paths = ["data/pdf/minutes.pdf", "data/pdf/png.pdf"]
            self.merge_pdfs(pdf_paths, pdf_paths[0])  # Assuming 'merge_pdfs' merges PDF files

            # Set up the model parameters for Gemini
            generation_config = self.get_generation_config_2()
            # Generate summary text for Telegram, retrying up to 6 times with delays
            for _ in tqdm(range(6), desc="Generating telegram_minutes_text"):
                try:
                    time.sleep(4)  # Wait for 4 seconds before each attempt

                    # Generate the summary text using the Gemini model
                    self.telegram_minutes_text = str(genai.GenerativeModel(
                        model_name="gemini-1.5-flash-latest",
                        generation_config=generation_config,
                        system_instruction=self.get_system_instructions_7(),  
                        safety_settings=safety_settings
                    ).generate_content(minutes_text).text)

                    # Exit the loop if generation is successful
                    break 

                except Exception as e: 
                    # Print error message if generation fails
                    print(f"Error during telegram_minutes_text generation: {e}")
                    # Continue to the next iteration of the loop
                    pass

            # Set up the model parameters for Gemini
            generation_config = self.get_generation_config_2()
            # Attempt to generate the summary text up to 6 times.
            for _ in tqdm(range(6), desc="Generating telegram_summary_text"):
                try:
                    # Wait for 4 seconds before trying again to avoid overwhelming the API.
                    time.sleep(4)

                    # Generate the summary text using the Gemini model.
                    self.telegram_summary_text = str(genai.GenerativeModel(
                        model_name="gemini-1.5-flash-latest",
                        generation_config=generation_config,
                        system_instruction=self.get_system_instructions_7(),
                        safety_settings=safety_settings
                    ).generate_content(summary_text).text)

                    # Exit the loop if successful.
                    break

                except Exception as e: 
                    # Log the error and continue to the next attempt.
                    print(f"Error during telegram_summary_text generation: {e}")
                    pass

            # Rename the meeting minutes and summary PDF files to include the date.
            os.rename("data/pdf/minutes.pdf", f"data/pdf/{self.file_date}_minutes.pdf")
            os.rename("data/pdf/summary.pdf", f"data/pdf/{self.file_date}_summary.pdf")

            # Initialize lists to store image paths and prompt parts
            self.image_paths = []
            self.prompt_parts = []

            # Iterate over each ticker in the list
            for each_ticker in tqdm(self.interest_ticker_list, desc="genai.upload_file"):
                # Attempt to upload the image 6 times
                for _ in range(6):
                    try:
                        # Set the current ticker
                        self.ticker = each_ticker

                        # Upload the image to genai
                        temp_file = genai.upload_file(
                            path=f"data/png_optimized/{each_ticker}.png", 
                            display_name=f'{self.ticker_company} ({each_ticker}): 1d Candlestick Chart (with Technical Indicators)' 
                        )

                        # Append the uploaded file to prompt parts and image path to image paths list
                        self.prompt_parts.append(temp_file)
                        self.image_paths.append(f'data/png_optimized/{each_ticker}.png')

                        # Exit the retry loop if successful
                        break

                    except Exception as e:
                        # Log the error and continue to the next retry attempt
                        self.docker_print(f"Error during genai upload_file: {e}")
                        pass
            
            # Set up the model parameters for Gemini
            generation_config = self.get_generation_config_2()
            # Initialize an empty list to store photo captions.
            self.photo_caption_list = []
            # Iterate through each part of the prompt
            for each_prompt_part in tqdm(self.prompt_parts, desc="Generating photo_caption"):
                # Retry up to 6 times with a 30-second delay
                for _ in range(6):
                    try:
                        time.sleep(4)  # Wait for rate limiting or server congestion
                        # Generate a photo caption using Gemini Pro
                        photo_caption = str(genai.GenerativeModel(
                            model_name="gemini-1.5-flash-latest",  # Specify the Gemini model
                            generation_config=generation_config,  # Pass the generation configuration
                            # system_instruction=self.get_system_instructions_7(),  # Get instructions for this part
                            safety_settings=safety_settings  # Enforce safety settings
                        ).generate_content([self.get_system_instructions_7(), each_prompt_part]).text)

                        # Append the generated caption to the list
                        self.photo_caption_list.append(photo_caption)

                        # Break the retry loop if successful
                        break

                    # Catch any exceptions during generation
                    except Exception as e:
                        self.docker_print(f"Error generating caption: {e}")
                        # Continue to the next retry attempt

            # Set up the model parameters for Gemini
            generation_config = self.get_generation_config_1()

            # Delete uploaded images from Gemini after generating captions
            for each_prompt_part in tqdm(self.prompt_parts, desc="genai.delete_file"):
                try:
                    genai.delete_file(each_prompt_part.name)
                except:
                    pass

            # Garbage Collection (GC) - Explicitly delete and free up memory
            for each_prompt_part in self.prompt_parts:
                each_prompt_part = None
                del each_prompt_part
            prompt_parts = None
            del prompt_parts
            gc.collect()
        except Exception as e:
            # Error handling
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno)
            self.docker_print(temp_msg)  # Assuming 'docker_print' logs the error message
            try:
                # Attempt to delete uploaded files from Gemini in case of an error
                for each_prompt_part in tqdm(self.prompt_parts, desc="genai.delete_file"):
                    try:
                        genai.delete_file(each_prompt_part.name)
                    except:
                        pass
            except:
                pass  # Ignore any errors during cleanup

    def get_candlestick_data(self):
        """
        Retrieves and prepares candlestick data for charting.

        This method calculates various technical indicators and prepares a DataFrame
        with candlestick data (OHLC), RSI, MACD, Bollinger Bands, Stochastic Oscillator,
        CMF, OBV, ATR, and Ichimoku Cloud.

        Returns:
            pandas.DataFrame: A DataFrame containing the prepared candlestick data.
        """

        # Make a copy of the stock data for the specific ticker
        ohlc = self.sp500_df_dict[self.ticker].copy()

        # Convert 'Date' column to matplotlib's numerical date format
        ohlc['ori_Date'] = ohlc['Date']  # Store original date for later use
        ohlc['Date'] = mdates.date2num(ohlc['Date'])

        # Calculate technical indicators

        # RSI
        ohlc['RSI'] = talib.RSI(ohlc['Close'].values)
        ohlc['RSI_SMA_14'] = talib.SMA(ohlc['RSI'], timeperiod=14)  # 14-day SMA of RSI

        # MACD
        close_prices = ohlc['Close']
        macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        ohlc['macd'] = macd
        ohlc['macdsignal'] = macd_signal
        ohlc['macdhist'] = macd_hist

        # Bollinger Bands
        upperband, middleband, lowerband = talib.BBANDS(ohlc['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        ohlc['upperband_BB'] = upperband
        ohlc['middleband_BB'] = middleband
        ohlc['lowerband_BB'] = lowerband

        # Stochastic Oscillator (calculation assumed to be in 'calculate_stochastic_oscillator' method)
        ohlc = self.calculate_stochastic_oscillator(ohlc)

        # Chaikin Money Flow (CMF)
        ohlc = self.calc_CMF(ohlc)

        # On-Balance Volume (OBV)
        ohlc['OBV'] = talib.OBV(ohlc['Close'], ohlc['Volume'])

        # Average True Range (ATR)
        ohlc['ATR'] = talib.ATR(ohlc['High'], ohlc['Low'], ohlc['Close'])

        # Ichimoku Cloud (calculation assumed to be in 'calculate_ichimoku' method)
        ohlc = self.calculate_ichimoku(ohlc)

        # Select data for candlestick chart and future cloud
        ohlc = reset_dataframe_index(ohlc.tail(self.candlestick_chart_no + self.future_cloud_no))

        # Create a list of consecutive dates for the x-axis
        concat_date_list = []
        for i in range(self.candlestick_chart_no + self.future_cloud_no):
            concat_date_list.append(ohlc['Date'].max() - (i + 1))
        ohlc['Date'] = concat_date_list[::-1]  # Assign the dates in reverse order

        return ohlc
    
    def get_candlestick_image(self):
        """
        Generates and saves a candlestick chart image with various technical indicators.

        The chart includes the following subplots:
            - Candlestick chart with Bollinger Bands and Fibonacci retracement levels
            - Volume bars
            - RSI with SMA
            - MACD with histogram
            - Ichimoku Cloud with Fibonacci retracement levels
            - Stochastic Oscillator
            - Chaikin Money Flow (CMF)
            - On-Balance Volume (OBV)
            - Average True Range (ATR)

        The function saves the generated chart as a PNG image.
        """
        
        # Get candlestick data
        ohlc = self.get_candlestick_data()
        data = ohlc.copy()

        # Create a figure and subplots
        fig, (ax1, ax5, ax2, ax8, ax4, ax3, ax9, ax6, ax7) = plt.subplots(
            9, 1, sharex=True, figsize=(20, 30), 
            gridspec_kw={'height_ratios': [1, 1, 0.33, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}
        )

        # Set background color for all subplots
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
            ax.set_facecolor("#151924")

        # Set grid for all subplots
        for ax in [ax1, ax2, ax4, ax5, ax7, ax8, ax9]:
            ax.set_axisbelow(True)
            ax.grid(color='#2E323D', linestyle='-', zorder=0)

        # --- Subplot 1: Candlestick chart with Bollinger Bands and Fibonacci retracement ---
        ax1.set_title(f'{self.ticker_company} ({self.ticker}): 1d Candlestick Chart (with Technical Indicators)', fontsize=16)

        # Plot Fibonacci retracement levels
        levels = [0, 0.214, 0.382, 0.5, 0.618, 0.764, 1]
        color_list = ['#787B86', '#06BCD4', '#0A9981', '#4CAF51', '#FF9800', '#F23545', '#787B86']
        levels_label = ['1', '0.786', '0.618', '0.5', '0.382', '0.236','0']
        for i, level in enumerate(levels):
            price = ohlc['Low'].min() + (ohlc['High'].max() - ohlc['Low'].min()) * level
            ax1.axhline(price, linestyle='-', linewidth=0.75, color=color_list[i], zorder=1)
            ax1.text(ohlc['Date'].iloc[-1], price, f"{levels_label[i]} " + "(" + "{:.2f}".format(price) + ")", 
                     va='bottom', ha='right', fontsize=12, color=color_list[i], alpha=0.7)

        # Plot Bollinger Bands
        ax1.plot(ohlc['Date'], ohlc['upperband_BB'], color='#F23545', linestyle='-', linewidth=1, zorder=1)
        ax1.plot(ohlc['Date'], ohlc['middleband_BB'], color='#2862FF', linestyle='-', linewidth=1, zorder=1)
        ax1.plot(ohlc['Date'], ohlc['lowerband_BB'], color='#0A9981', linestyle='-', linewidth=1, zorder=1)
        ax1.fill_between(ohlc['Date'], ohlc['lowerband_BB'], ohlc['upperband_BB'], color='#2862FF', alpha=0.078, zorder=1)

        # Format y-axis
        ax1.yaxis.set_major_formatter(FuncFormatter(self.comma_formatter_2))
        ax1.set_ylabel('Price, BB, and Fib Retracement', fontsize=16)
        ax1.yaxis.set_label_position('left')
        ax1.yaxis.set_ticks_position('both')
        ax1.yaxis.set_tick_params(pad=10, direction='inout', length=6, labelright=True, right=True)
        ax1.yaxis.set_tick_params(labelsize=13)
        ax1.set_ylim(top=ohlc['upperband_BB'].max() + (ohlc['upperband_BB'].max()-ohlc['lowerband_BB'].min())*0.15)

        # Plot candlestick chart
        candlestick_ohlc(ax1, ohlc.values, width=self.graph_width, colorup='#26A69A', colordown='#F05350', alpha=1.0)

        # --- Subplot 2: Volume ---
        volume_color = ['#1C5E5F' if data['Close'][i] >= data['Open'][i] else '#813539' for i in range(len(data))]
        ax2.bar(data['Date'], data['Volume'], width=self.graph_width, color=volume_color, zorder=1)
        ax2.set_ylabel('Volume', fontsize=16, labelpad=20)
        ax2.yaxis.set_label_position('left')
        ax2.yaxis.set_major_formatter(FuncFormatter(self.millions_formatter))
        ax2.yaxis.set_ticks_position('both')
        ax2.yaxis.set_tick_params(pad=10, direction='inout', length=6, labelright=True, right=True)
        ax2.yaxis.set_tick_params(labelsize=13)
        ax2.set_ylim(1, ohlc['Volume'].max() + (ohlc['Volume'].max()-ohlc['Volume'].min())*0.20)

        # --- Subplot 3: RSI ---
        ax3.set_ylabel('RSI', fontsize=16)
        ax3.yaxis.set_label_position('left')
        ax3.yaxis.set_tick_params(pad=10, direction='inout', length=6, labelright=True, right=True)
        ax3.fill_between(ohlc['Date'], 30, 70, color='#212035', zorder=1)
        ax3.axhline(y=30, color='grey', linestyle='--', linewidth=0.5, zorder=2)
        ax3.axhline(y=50, color='grey', linestyle='--', linewidth=0.5, zorder=2)
        ax3.axhline(y=70, color='grey', linestyle='--', linewidth=0.5, zorder=2)
        ax3.yaxis.set_tick_params(labelsize=13)
        ax3.set_ylim(1, 99)
        ax3.grid(color='#2E323D', linestyle='-', zorder=1)
        ax3.plot(ohlc['Date'], ohlc['RSI'], color='#7D57C2', linewidth=1, zorder=2)
        ax3.plot(ohlc['Date'], ohlc['RSI_SMA_14'], color='yellow', linewidth=1, zorder=3)

        # --- Subplot 4: MACD ---
        ax4.set_ylabel('MACD', fontsize=16)
        ax4.plot(ohlc['Date'], ohlc['macd'], color='#2862FF', label='MACD', linewidth=1, zorder=1)
        ax4.plot(ohlc['Date'], ohlc['macdsignal'], color='#FF6D00', label='Signal', linewidth=1, zorder=1)
        ax4.yaxis.set_ticks_position('both')
        ax4.yaxis.set_tick_params(pad=10, direction='inout', length=6, labelright=True, right=True)
        ax4.yaxis.set_major_formatter(FuncFormatter(self.comma_formatter))
        macd_colors = self.get_color_list(list(ohlc['macdhist'].values))
        ax4.bar(ohlc['Date'], ohlc['macdhist'], color=macd_colors, alpha=0.5, width=self.graph_width, label='Histogram', zorder=1)
        ax4.yaxis.set_tick_params(labelsize=13)
        ax4.yaxis.set_major_formatter(FuncFormatter(self.comma_formatter_2))
        max_val = max(ohlc['macd'].max(), ohlc['macdsignal'].max(), ohlc['macdhist'].max())
        min_val = min(ohlc['macd'].min(), ohlc['macdsignal'].min(), ohlc['macdhist'].min())
        ax4.set_ylim(top=max_val + ((max_val - min_val) * 0.15))

        # --- Subplot 5: Ichimoku Cloud ---
        ax5.plot(ohlc['Date'], ohlc['Tenkan_Sen'], label='Tenkan-sen', color='#2862FF', linewidth=1, zorder=1)
        ax5.plot(ohlc['Date'], ohlc['Kijun_Sen'], label='Kijun-sen', color='#B71C1C', linewidth=1, zorder=1)
        ax5.plot(ohlc['Date'], ohlc['Chikou_Span'], label='Chikou Span', color='#43A047', linewidth=1, zorder=1)
        ax5.plot(ohlc['Date'], ohlc['Senkou_Span_A'], label='Senkou Span A (Leading Span A)', color='#A5D6A7', linewidth=1, zorder=1)
        ax5.plot(ohlc['Date'], ohlc['Senkou_Span_B'], label='Senkou Span B (Leading Span B)', color='#EF9A9A', linewidth=1, zorder=1)
        ax5.fill_between(ohlc['Date'], ohlc['Senkou_Span_A'], ohlc['Senkou_Span_B'], 
                        where=ohlc['Senkou_Span_A'] >= ohlc['Senkou_Span_B'], 
                        facecolor='#43A047', alpha=0.125, interpolate=True, zorder=1)
        ax5.fill_between(ohlc['Date'], ohlc['Senkou_Span_A'], ohlc['Senkou_Span_B'], 
                        where=ohlc['Senkou_Span_A'] < ohlc['Senkou_Span_B'], 
                        facecolor='#B71C1C', alpha=0.125, interpolate=True, zorder=1)
        ax5.set_ylabel('Price, Ichimoku Cloud, and Fib Retracement', fontsize=16)
        ax5.yaxis.set_label_position('left')
        ax5.yaxis.set_tick_params(pad=10, direction='inout', length=6, labelright=True, right=True)
        ax5.yaxis.set_tick_params(labelsize=13)

        # Plot Fibonacci retracement levels
        levels_2 = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        color_list_2 = ['#787B86', '#F23545', '#FF9800', '#4CAF51', '#0A9981', '#06BCD4', '#787B86']
        levels_label_2 = ['0', '0.236', '0.382', '0.5', '0.618', '0.786','1']
        for i, level in enumerate(levels_2):
            price = ohlc['Low'].min() + (ohlc['High'].max() - ohlc['Low'].min()) * level
            ax5.axhline(price, linestyle='-', linewidth=0.75, color=color_list_2[i], zorder=1)
            ax5.text(ohlc['Date'].iloc[-1], price, f"{levels_label_2[i]} " + "(" + "{:.2f}".format(price) + ")", 
                     va='bottom', ha='right', fontsize=12, color=color_list_2[i], alpha=0.7)

        ax5.yaxis.set_major_formatter(FuncFormatter(self.comma_formatter_2))
        ax5.set_ylim(top=ohlc['High'].max() + (ohlc['High'].max()-ohlc['Low'].min())*0.15)
        candlestick_ohlc(ax5, ohlc.values, width=self.graph_width, colorup='#26A69A', colordown='#F05350', alpha=1.0)

        # --- Subplot 6: Stochastic Oscillator ---
        ax6.set_ylabel('STO', fontsize=16)
        ax6.yaxis.set_label_position('left')
        ax6.yaxis.set_tick_params(pad=10, direction='inout', length=6, labelright=True, right=True)
        ax6.fill_between(ohlc['Date'], 20, 80, color='#20172E', zorder=1)
        ax6.axhline(y=20, color='grey', linestyle='--', linewidth=0.5, zorder=2)
        ax6.axhline(y=80, color='grey', linestyle='--', linewidth=0.5, zorder=2)
        ax6.yaxis.set_tick_params(labelsize=13)
        ax6.set_ylim(-8, 115)
        ax6.grid(color='#2E323D', linestyle='-', zorder=1)
        ax6.plot(ohlc['Date'], ohlc['K_STO'], color='#09AE0C', linewidth=1, zorder=2)
        ax6.plot(ohlc['Date'], ohlc['D_STO'], color='#B25B11', linewidth=1, zorder=3)

        # --- Subplot 7: Chaikin Money Flow ---
        ax7.axhline(y=0, color='#9598A1', linestyle='--', linewidth=0.5, zorder=1)
        ax7.plot(ohlc['Date'], ohlc['CMF'], color='#09AE0C', linewidth=1, zorder=1)
        ax7.set_ylabel('CMF', fontsize=16)
        ax7.yaxis.set_ticks_position('both')
        ax7.yaxis.set_tick_params(pad=10, direction='inout', length=6, labelright=True, right=True)
        ax7.yaxis.set_tick_params(labelsize=13)
        ax7.set_ylim(top=ohlc['CMF'].max() + (ohlc['CMF'].max()-ohlc['CMF'].min())*0.15)

        # --- Subplot 8: On-Balance Volume ---
        ax8.plot(ohlc['Date'], ohlc['OBV'], color='#2862FF', linewidth=1, zorder=1)
        ax8.set_ylabel('OBV', fontsize=16)
        ax8.yaxis.set_ticks_position('both')
        ax8.yaxis.set_tick_params(pad=10, direction='inout', length=6, labelright=True, right=True)
        ax8.yaxis.set_major_formatter(FuncFormatter(self.millions_formatter))
        ax8.yaxis.set_tick_params(labelsize=13)
        ax8.set_ylim(top=ohlc['OBV'].max() + (ohlc['OBV'].max()-ohlc['OBV'].min())*0.15)

        # --- Subplot 9: Average True Range ---
        ax9.plot(ohlc['Date'], ohlc['ATR'], color='#B71C1C', linewidth=1, zorder=1)
        ax9.set_ylabel('ATR', fontsize=16)
        ax9.yaxis.set_ticks_position('both')
        ax9.yaxis.set_tick_params(pad=10, direction='inout', length=6, labelright=True, right=True)
        ax9.yaxis.set_major_formatter(FuncFormatter(self.comma_formatter))
        ax9.yaxis.set_tick_params(labelsize=13)
        ax9.set_xlabel('Date', fontsize=16)
        ax9.xaxis.set_tick_params(labelsize=13)
        ax9.yaxis.set_major_formatter(FuncFormatter(self.comma_formatter_2))
        ax9.set_ylim(top=ohlc['ATR'].max() + (ohlc['ATR'].max()-ohlc['ATR'].min())*0.15)

        # --- Add titles and labels to subplots ---
        label_list = ["BB ", "SMA ", "close 2 ", f"{self.title_formatter(ohlc['middleband_BB'].values[-1-self.future_cloud_no])} ", f"{self.title_formatter(ohlc['upperband_BB'].values[-1-self.future_cloud_no])} ", f"{self.title_formatter(ohlc['lowerband_BB'].values[-1-self.future_cloud_no])}"]
        colors = ['white', '#868993', '#868993', '#2862FF', '#F23545', '#0A9981', '#EF9A9A']
        self.color_title(ax1, label_list, colors, y=0.958)

        label_list = ["Fibonacci ", "Retracement ", "0 ", "0.236 ", "0.382 ", "0.5", " 0.618", " 0.786", "1"]
        colors = ['white','white','#787B86', '#F23545', '#FF9800', '#4CAF51', '#0A9981', '#06BCD4', '#787B86']
        self.color_title(ax1, label_list, colors, y=0.908)

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
        self.color_title(ax5, label_list, colors, y=0.908)

        label_list = ["Volume ", f"{self.human_format(ohlc['Volume'].values[-1-self.future_cloud_no])}"]
        colors = ['white', volume_color[data[data['Close'].isna()].index[0] - 1]]
        self.color_title(ax2, label_list, colors, y=0.88)

        label_list = ["RSI ", "14 close ", 
                    f"{self.title_formatter(ohlc['RSI'].values[-1-self.future_cloud_no])} ",
                    f"{self.title_formatter(ohlc['RSI_SMA_14'].values[-1-self.future_cloud_no])}",
                    ]
        colors = ['white', "#868993", "#7D57C2" , "yellow"]
        self.color_title(ax3, label_list, colors, y=0.92)

        label_list = ["MACD ", "12 26 close ",
                    f"{self.title_formatter(ohlc['macdhist'].values[-1-self.future_cloud_no])} ",
                    f"{self.title_formatter(ohlc['macd'].values[-1-self.future_cloud_no])} ",
                    f"{self.title_formatter(ohlc['macdsignal'].values[-1-self.future_cloud_no])}",
                    ]
        colors = ['white', "#868993", macd_colors[:-self.future_cloud_no][-1], "#2862FF", "#FF6D00"]
        self.color_title(ax4, label_list, colors, y=0.92)

        label_list = ["Stochastic Oscillator ", "14 3 80 20 ",
                    f"{self.title_formatter(ohlc['K_STO'].values[-1-self.future_cloud_no])} ",
                    f"{self.title_formatter(ohlc['D_STO'].values[-1-self.future_cloud_no])} ",
                    ]
        colors = ['white', "#868993", "#07FC00", "#FF7F00"]
        self.color_title(ax6, label_list, colors, y=0.92)

        label_list = ["CMF ", "20 ",
                    f"{self.title_formatter(ohlc['CMF'].values[-1-self.future_cloud_no])}",
                    ]
        colors = ['white', "#868993", "#09AE0C",]
        self.color_title(ax7, label_list, colors, y=0.92)

        label_list = ["OBV ", f"{self.human_format(ohlc['OBV'].values[-1-self.future_cloud_no])}",
                    ]
        colors = ['white', "#2862FF",]
        self.color_title(ax8, label_list, colors, y=0.92)

        label_list = ["ATR ", "14 RMA ",
                    f"{self.title_formatter(ohlc['ATR'].values[-1-self.future_cloud_no])}",
                    ]
        colors = ['white', "#868993", "#B71C1C",]
        self.color_title(ax9, label_list, colors, y=0.92)

        # Calculate the date difference between the first two data points for x-axis limit setting
        date_val = ohlc['Date'].values[1] - ohlc['Date'].values[0]
        
        # Set the x-axis limits, adding padding based on the date difference
        ax1.set_xlim([ohlc['Date'].min() - date_val, ohlc['Date'].max() + date_val])
        
        # Set x-axis ticks at specific intervals (assuming 26 periods per year)
        ax1.set_xticks([
            ohlc.iloc[0]['Date'],                     # First date
            ohlc.iloc[26 - 1]['Date'],               # End of the 1st period
            ohlc.iloc[26 * 2 - 1]['Date'],          # End of the 2nd period
            ohlc.iloc[26 * 3 - 1]['Date'],          # ... and so on
            ohlc.iloc[26 * 4 - 1]['Date'],
            ohlc.iloc[26 * 5 - 1]['Date'],
            ohlc.iloc[26 * 6 - 1]['Date'],
            ohlc.iloc[(26 * 7 - 1)]['Date'],
            ohlc.iloc[(26 * 8 - 1)]['Date'],
        ])
        
        # Create a list of x-axis tick labels using 'ori_Date' column
        xticklabels_list = [
            ohlc.iloc[0]['ori_Date'],
            ohlc.iloc[26 - 1]['ori_Date'],
            ohlc.iloc[26 * 2 - 1]['ori_Date'],
            ohlc.iloc[26 * 3 - 1]['ori_Date'],
            ohlc.iloc[26 * 4 - 1]['ori_Date'],
            ohlc.iloc[26 * 5 - 1]['ori_Date'],
            ohlc.iloc[26 * 6 - 1]['ori_Date'],
            ohlc.iloc[(26 * 7 - 1)]['ori_Date'],
        ]
        
        # Format the date strings to month format
        xticklabels_list = [self.get_date_month_text(each) for each in xticklabels_list]
        
        # Make a copy of the original tick labels
        ori_xticklabels_list = xticklabels_list.copy()
        
        # Append additional tick labels for future dates
        start_i = 26 - 1
        xticklabels_list.append(self.get_date_month_text(ohlc['ori_Date'].values[start_i - 1]))
        next_i = start_i + 26
        xticklabels_list.append(self.get_date_month_text(ohlc['ori_Date'].values[next_i]))
        
        for _ in range(5):
            next_i = next_i + 26
            xticklabels_list.append(self.get_date_month_text(ohlc['ori_Date'].values[next_i]))

        # Create a list for the last x-axis tick label
        xticklabels_list_2 = [xticklabels_list[-1]]
        
        # Convert string dates to datetime objects for calculation
        dates = [datetime.datetime.strptime(date, '%d %b') for date in xticklabels_list_2]
        last_date = dates[-1]
        
        # Calculate next weekdays (Monday to Friday)
        next_days = []
        days_to_add = 26
        while len(next_days) < days_to_add:
            last_date += datetime.timedelta(days=1)
            if last_date.weekday() < 5:  # Monday to Friday are 0-4
                next_days.append(last_date)
        
        # Format the next weekdays and append to the original tick labels
        next_days_str = [date.strftime('%d %b') for date in next_days]
        ori_xticklabels_list.extend([next_days_str[-1]])
        
        # Use the modified tick labels for the plot
        xticklabels_list = ori_xticklabels_list.copy()
        ax1.set_xticklabels(xticklabels_list)
        
        # --- Adjust layout and save the figure ---
        plt.subplots_adjust(left=0.085, right=0.925, bottom=0.025, top=0.975)
        fig.subplots_adjust(hspace=0.03)
        plt.savefig('data/png/temp.png')

        # Open the image file
        with Image.open("data/png/temp.png") as img:
            # Define the new dimensions required for resizing, specifically for the input images used in the Gemini project.
            new_width, new_height = 2048, 3072
            
            # Resize the image using LANCZOS resampling for high quality
            img_resized = img.resize((new_width, new_height), resample=Image.LANCZOS)
            
            # Save the resized image with the ticker as the filename
            img_resized.save(f"data/png/{self.ticker}.png")

        # Open the resized image
        with Image.open(f"data/png/{self.ticker}.png") as img:
            # Convert the image to an 8-bit palette mode with 256 colors
            # using adaptive palette generation for optimal color representation
            img = img.convert("P", palette=Image.ADAPTIVE, colors=256)
            
            # Save the optimized image with reduced file size
            img.save(f"data/png_optimized/{self.ticker}.png", optimize=True)

        # Remove temporary files and clear memory
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
        """
        Calculates the Ichimoku Cloud indicator.

        Args:
            ohlc (pd.DataFrame): DataFrame containing Open, High, Low, Close prices.

        Returns:
            pd.DataFrame: DataFrame with Ichimoku Cloud components.
        """

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

        # Calculate future Senkou Spans
        future_Senkou_Span_A = list((df['Tenkan_Sen'].tail(26) + df['Kijun_Sen'].tail(26)) / 2)[:self.future_cloud_no]
        future_Senkou_Span_B = list((high_52.tail(26) + low_52.tail(26)) / 2)[:self.future_cloud_no]

        # Generate future dates for the cloud
        future_26 = []
        date_val = df['Date'].values[1] - df['Date'].values[0]  # Assuming 'Date' column exists
        for i in range(1, self.future_cloud_no+1):
            val = df['Date'].tail(1).values[0] + i*date_val
            future_26.append(val)
        future_vals = pd.DataFrame({'Date': future_26})

        # Extend the DataFrame with future dates
        df = pd.concat([df, future_vals], ignore_index=True)

        # Extend Senkou Spans into the future
        all_Senkou_Span_A = list(df['Senkou_Span_A'].values[:-self.future_cloud_no]) + future_Senkou_Span_A
        df['Senkou_Span_A'] = all_Senkou_Span_A
        all_Senkou_Span_B = list(df['Senkou_Span_B'].values[:-self.future_cloud_no]) + future_Senkou_Span_B
        df['Senkou_Span_B'] = all_Senkou_Span_B

        return df

    def calculate_stochastic_oscillator(self, data, time_period_k=14, smoothing_period_d=3):
        """
        Calculates the stochastic oscillator (K and D lines) for a given dataset.

        Args:
            data (pandas.DataFrame): DataFrame containing financial data with 'High', 'Low', and 'Close' columns.
            time_period_k (int, optional): Time period for calculating the %K line. Defaults to 14.
            smoothing_period_d (int, optional): Smoothing period for calculating the %D line (moving average of %K). 
                                               Defaults to 3.

        Returns:
            pandas.DataFrame: The input DataFrame with added 'K_STO' and 'D_STO' columns representing 
                              the %K and %D lines of the stochastic oscillator.
        """

        # Calculate the lowest low over the given time period (N)
        data['low_N'] = data['Low'].rolling(time_period_k).min()

        # Calculate the highest high over the given time period (N)
        data['high_N'] = data['High'].rolling(time_period_k).max()

        # Calculate the %K line (Stochastic Oscillator)
        data['K_STO'] = 100 * (data['Close'] - data['low_N']) / (data['high_N'] - data['low_N'])

        # Calculate the %D line (Moving Average of %K)
        data['D_STO'] = data['K_STO'].rolling(smoothing_period_d).mean()

        # Remove temporary columns 'low_N' and 'high_N' as they are no longer needed
        data.drop(columns=['low_N', 'high_N'], inplace=True)

        return data
    
    def calc_CMF(self, ask_series):
        """
        Calculate the Chaikin Money Flow (CMF) indicator.

        Args:
            ask_series (pd.DataFrame): DataFrame containing the ask price data with columns 'High', 'Low', 'Close', and 'Volume'.

        Returns:
            pd.DataFrame: DataFrame with the CMF column added.
        """

        # Calculate the Money Flow Multiplier (MFV)
        ask_series["cmfm"] = (((ask_series["Close"] - ask_series["Low"]) - (ask_series["High"] - ask_series["Close"])) 
                             / (ask_series["High"] - ask_series["Low"]))

        # Calculate the Money Flow Volume (MFV)
        ask_series["cmfv"] = ask_series["cmfm"] * ask_series["Volume"]

        # Calculate the CMF over a 20-period rolling window
        ask_series["CMF"] = (ask_series['cmfv'].rolling(window=20).mean() / 
                             ask_series['Volume'].rolling(window=21).mean())

        # Remove temporary columns
        ask_series.drop(columns=['cmfm', 'cmfv'], inplace=True)

        return ask_series
    
    def human_format(self, num):
        """
        Formats a number to a human-readable string using SI unit prefixes.

        Args:
            num: The number to format.

        Returns:
            A string representing the number in a human-readable format (e.g., "1.2K", "15M", "2.5B").
        """

        magnitude = 0  # Initialize the magnitude (power of 1000)
        while abs(num) >= 1000:
            magnitude += 1  # Increment the magnitude for each 1000
            num /= 1000.0  # Divide the number by 1000

        # Format the number with the appropriate suffix
        # ['', 'K', 'M', 'G', 'T', 'P'] corresponds to ["", "Thousand", "Million", "Billion", "Trillion", "Peta"]
        return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
    
    def get_color_list(self, values):
        """
        Generates a list of colors based on a list of values.

        The color scheme uses green for positive values and red for negative values.
        Darker shades indicate a larger value compared to the previous value.

        Args:
            values: A list of numerical values.

        Returns:
            A list of color strings (hexadecimal format).
        """

        # Define color codes for different shades
        dark_green = '#26A69A'
        light_green = '#B2DFDB'
        light_red = '#FFCDD2'
        dark_red = '#FF5252'

        # Initialize the colors list
        colors = []

        # Initialize the previous value for comparison
        previous_value = values[0]

        # Iterate through the values and assign colors
        for value in values:
            if value < 0:  # Negative value
                if value < previous_value:  # More negative than previous
                    colors.append(dark_red) 
                else:  # Less negative than previous
                    colors.append(light_red)
            elif value > 0:  # Positive value
                if value > previous_value:  # More positive than previous
                    colors.append(dark_green)
                else:  # Less positive than previous
                    colors.append(light_green)
            else:  # Zero value
                colors.append(light_green)  # Treat zero as a slightly positive value
            previous_value = value  # Update the previous value for the next iteration

        return colors
    
    def millions_formatter(self, x, pos):
        """
        Formats a number to a human-readable string with thousands, millions, billions, etc. suffixes.

        Args:
            x: The number to format.
            pos: The position of the tick (not used in this implementation).

        Returns:
            A string representing the formatted number.
        """

        num = x  # Start with the original number
        magnitude = 0  # Initialize the magnitude (0 = no suffix, 1 = K, 2 = M, etc.)

        # Keep dividing by 1000 until the number is less than 1000
        while abs(num) >= 1000:
            magnitude += 1  # Increment the magnitude
            num /= 1000.0  # Divide the number by 1000

        # Create the formatted string using string formatting
        # '%.0f' rounds the number to an integer
        # %s inserts the appropriate suffix from the list
        return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
    
    def comma_formatter(self, x, pos):
        """
        Formats a numeric value with commas as thousands separators.

        Args:
            x: The numeric value to format.
            pos: The position of the value (unused in this formatter).

        Returns:
            str: The formatted string with commas as thousands separators.
        """
        return "{:,}".format(x)

    def comma_formatter_2(self, x, pos):
        """
        Formats a numeric value with commas as thousands separators and 2 decimal places.

        Args:
            x: The numeric value to format.
            pos: The position of the value (unused in this formatter).

        Returns:
            str: The formatted string with commas and 2 decimal places.
        """
        return "{:,.2f}".format(x)

    def title_formatter(self, x):
        """
        Formats a numeric value with 2 decimal places.

        Args:
            x: The numeric value to format.

        Returns:
            str: The formatted string with 2 decimal places.
        """
        return "{:.2f}".format(x)
    
    def color_title(self, ax, labels, colors, textprops={'size': 15}, y=0.96):
        """
        Creates a left-aligned title with multiple colors.

        This function adds a title to the given axes object, where each word 
        in the title can have a different color. 

        Important: 
            Do not change axes limits after calling this function, 
            as it may misalign the title.

        Args:
            ax (matplotlib.axes.Axes): The axes object to add the title to.
            labels (list): A list of strings, where each string is a word in the title.
            colors (list): A list of colors, where each color corresponds to a word in `labels`.
            textprops (dict, optional): A dictionary of text properties to apply to the title. 
                                        Defaults to {'size': 15}.
            y (float, optional): The vertical position of the title in axes coordinates. 
                                Defaults to 0.96 (close to the top).
        """

        # Draw the figure to be able to get text extents
        plt.gcf().canvas.draw()

        # Use axes coordinates for positioning the text
        transform = ax.transAxes  
        
        # Initial horizontal position - start from the left (x=0)
        x_pos = 0  
        
        # Dictionary to store the text objects for each label
        text = dict()  

        # Iterate through labels and corresponding colors
        for label, col in zip(labels, colors):
            # Add text to the axes
            text[label] = ax.text(x_pos, y, label,
                                    transform=transform,  # Use axes coordinates
                                    ha='left',           # Left alignment
                                    color=col,             # Set text color
                                    **textprops)           # Apply any additional text properties

            # Update x_pos for the next label based on the previous label's width
            x_pos = text[label].get_window_extent().transformed(transform.inverted()).x1

    def extract_json(self, text_response):
        """
        Extracts and returns a list of valid JSON objects found in the input string.

        Args:
            text_response (str): The input string from which to extract JSON objects.

        Returns:
            list: A list of extracted JSON objects, or None if no valid JSON objects are found.
        """

        # Find all potential JSON objects within the text
        json_objects = self.find_potential_json(text_response)

        # Return the list of JSON objects if any are found, otherwise return None
        return json_objects if json_objects else None

    def find_potential_json(self, text_response):
        """
        Finds and returns a list of potential JSON objects within the input string.

        This method searches for strings that start with '{' and end with '}',
        and then attempts to parse them as JSON. If a JSONDecodeError occurs,
        it attempts to extend the search to include nested structures.

        Args:
            text_response (str): The input string from which to extract potential JSON objects.

        Returns:
            list: A list of extracted potential JSON objects. 
        """

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
        return json_objects

    def extend_search(self, text, span):
        """
        Extends the search range to capture potential nested JSON structures.

        This method starts from the given span and expands the search range
        until the correct nesting balance of '{' and '}' characters is found,
        indicating a complete JSON structure.

        Args:
            text (str): The input string being searched.
            span (tuple): A tuple containing the start and end indices of the initial match.

        Returns:
            str: The extended string that potentially contains a complete JSON structure.
        """

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
        """Converts a date string in 'YYYY-MM-DD' format to 'DD Mon' format.

        Args:
            date_str: The date string in 'YYYY-MM-DD' format.

        Returns:
            The date string in 'DD Mon' format (e.g., '01 Jan').
        """

        # Convert the date string to a datetime object
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")

        # Format the datetime object to 'DD Mon' format
        return date_obj.strftime("%d %b")
    
    def make_pdf_uncroppable(self, input_pdf_path, output_pdf_path):
        """
        Converts a PDF to a sequence of images, effectively making it 'uncroppable'.

        This is achieved by treating each page of the PDF as a distinct image. 
        When opened, the PDF will appear as a single image per page, 
        preventing traditional cropping methods from being effective.

        Args:
            input_pdf_path (str): The path to the input PDF file.
            output_pdf_path (str): The path where the output 'uncroppable' PDF will be saved.
        """

        # Convert the input PDF to a list of PIL Image objects
        images = convert_from_path(input_pdf_path)

        # Handle cases where the PDF has multiple pages
        if len(images) > 1:
            # Save the first image as a PDF
            # Use save_all=True to indicate that multiple images will be saved
            # Append all subsequent images to the PDF
            images[0].save(
                output_pdf_path,
                "PDF",
                resolution=100.0,
                save_all=True,
                append_images=images[1:],
            )
        # Handle cases where the PDF has only one page
        else:
            # Save the single image as a PDF
            images[0].save(output_pdf_path, "PDF", resolution=100.0)

    def merge_pdfs(self, paths, output_filename):
        """
        Merges multiple PDF files into a single PDF file.

        Args:
            paths (list): A list of paths to the PDF files to be merged.
            output_filename (str): The name of the output PDF file.
        """

        # Create a PdfMerger object
        merger = PdfMerger()

        # Iterate over the list of PDF file paths
        for path in paths:
            # Append each PDF file to the merger
            merger.append(path)

        # Write the merged PDF to the output file
        merger.write(output_filename)

        # Close the PdfMerger object
        merger.close()

    def markdown_to_pdf(self, markdown_text, output_pdf_path):
        """Converts Markdown text to a PDF file.

        Args:
            markdown_text (str): The Markdown text to convert.
            output_pdf_path (str): The path to save the output PDF file.
        """
        
        # Convert Markdown to HTML using the markdown library
        html_text = markdown.markdown(markdown_text)
        
        # Convert the HTML to PDF using the pdfkit library
        pdfkit.from_string(html_text, output_pdf_path)