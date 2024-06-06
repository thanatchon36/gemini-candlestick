# Gemini Candlestick Investment Fund Simulation

This repository hosts a Python application that simulates the investment decision-making process of the "Gemini Candlestick Investment Fund," a fictional investment board. The simulation employs candlestick chart analysis, technical indicators, and Google's Gemini Pro large language model to inform investment strategies.

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Use Cases](#use-cases)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Structure](#structure)
- [License](#license)
- [Hardware Requirements](#hardware-requirements)

## Introduction

The Gemini Candlestick Investment Fund application represents a sophisticated solution meticulously crafted to replicate the intricate dynamics of a board of directors' meeting. By harnessing advanced algorithms, it meticulously analyzes stock market data to formulate hypothetical investment strategies. Leveraging candlestick charts, technical indicators, and the computational prowess of Google's Gemini Pro large language model, this application offers a distinctive capability: reviving esteemed figures of the financial realm to engage in comprehensive discussions. This immersive feature enables these luminaries to convene and deliberate as if in a real-world setting. As a result, users are provided with an interactive environment that mirrors authentic financial analysis and decision-making processes, fostering a deeper understanding of investment strategies.

### Board of Directors

The esteemed board of directors comprises notable figures in the realm of financial analysis, each tasked with analyzing specific indicators and offering their expert insights:

1. **Munehisa Homma**, Chairman of the Board: Responsible for scrutinizing Candlestick Pattern Recognition and delivering informed perspectives.
2. **John Bollinger**: Tasked with assessing the Bollinger Bands (BB) indicator and providing analytical commentary.
3. **Leonardo Pisano Fibonacci**: Assigned to evaluate Fibonacci retracement levels and deliver insightful opinions.
4. **Ralph Nelson Elliott**: Engaged in analyzing potential Elliott Wave patterns employing Fibonacci ratios (0.786, 0.618, 0.5, 0.382, and 0.236 on Fibonacci retracement charts), RSI, MACD, and Bollinger Bands to ascertain wave retracements and projections within the patterns, and furnishing comprehensive opinions.
5. **Goichi Hosoda**: Entrusted with the evaluation of the Ichimoku Cloud and furnishing expert opinions.
6. **Joseph Granville**: Responsible for analyzing Volume and On-Balance Volume (OBV) indicators and delivering expert opinions.
7. **Gerald Appel**: Tasked with analyzing the MACD indicator and providing discerning viewpoints.
8. **J. Welles Wilder**: Charged with analyzing RSI and ATR indicators and offering well-founded opinions.
9. **George Lane**: Assigned to evaluate the Stochastic Oscillator indicator and provide expert insights.
10. **Marc Chaikin**: Entrusted with the analysis of the Chaikin Money Flow indicator and delivering comprehensive opinions.

## Key Features

- **Data Acquisition:** Efficiently retrieves historical candlestick data for selected stocks from the S&P 500 index utilizing the `yfinance` library.
- **Technical Analysis:** Conducts a comprehensive analysis by computing various technical indicators, including:
    - Relative Strength Index (RSI)
    - Moving Average Convergence Divergence (MACD)
    - Bollinger Bands
    - Fibonacci Retracement Levels
    - Ichimoku Cloud
    - Stochastic Oscillator
    - Chaikin Money Flow
    - On-Balance Volume (OBV)
    - Average True Range (ATR)
- **Candlestick Charting:** Generates detailed and visually appealing candlestick charts overlaid with technical indicators using `matplotlib` and `mplfinance`.
- **Gemini Pro Integration:** Utilizes Google's Gemini Pro large language model to:
    - Simulate discussions and analyses resembling those of board members.
    - Generate comprehensive meeting minutes covering market observations, individual stock analyses, and investment rationale.
    - Summarize actionable insights, key takeaways, and the fund's overall market strategy.
    - Create engaging captions for charts and reports.
- **Telegram Integration:** Automatically distributes daily reports, summaries, and highlighted candlestick charts to a designated Telegram chat via the Telegram Bot API.

## Use Cases

- **Educational Tool:** Serves as a comprehensive resource for understanding the application of candlestick charts and technical indicators in financial analysis, providing concrete examples and hands-on learning opportunities.
- **Simulation Environment:** Creates a secure and controlled platform for experimenting with various investment strategies, allowing users to observe potential outcomes without real financial risk.
- **Algorithmic Trading Inspiration:** Acts as a catalyst for the development of automated trading strategies, utilizing technical indicators and sentiment analysis from language models.
- **Financial Education:** Facilitates learning about different technical indicators, candlestick patterns, and their interpretations through an immersive and interactive simulated environment.

## Getting Started

To get the application up and running, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/gemini-candlestick.git
   ```

2. **Navigate to the Project Directory:**
   ```bash
   cd gemini-candlestick
   ```

3. **Start the Application:**
   ```bash
   docker-compose up -d 
   ```

## Configuration

Before running the application, you need to configure the following environment variables in the `docker-compose.yml` file:

- **`GEMINI_KEY`:** Your Google Gemini API key. Replace `YOUR_GEMINI_API_KEY` with your actual key.
- **`BOT_TOKEN`:** Your Telegram bot token. Replace `YOUR_TELEGRAM_BOT_TOKEN` with your bot's token.
- **`CHAT_ID`:** The ID of your Telegram chat. Replace `YOUR_TELEGRAM_CHAT_ID` with your chat's ID.

**Make sure to replace the placeholders with your actual credentials.**

## Structure

- **`docker-compose.yml`:** Defines the Docker container for running the application.
- **`Dockerfile.custom`:** Specifies instructions for building the Docker image, including installing dependencies and setting up the environment.
- **`geminicandlestick.py`:** Contains the main class, `GeminiCandlestick`, which handles data fetching, processing, charting, and Telegram interactions.
- **`main.py`:** The entry point of the application. Manages the overall workflow of data generation, charting, and Telegram notifications.

## License

This project is submitted to The Gemini API Developer Competition, sponsored by Google LLC. The submission is subject to the [Official Rules of the competition](https://ai.google.dev/competition), including the intellectual property provisions. 

## Hardware Requirements

- **Minimum & Recommended:** CPU: 2 cores (ARM64 architecture), RAM: 2GB, Storage: 16 GB

This application has been developed and tested on a MacBook M1, which comfortably meets these requirements.