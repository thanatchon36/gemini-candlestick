# Gemini Candlestick Investment Fund Simulation

This repository hosts a Python application that simulates the investment decision-making process of the "Gemini Candlestick Investment Fund," a fictional investment board. The simulation employs candlestick chart analysis, technical indicators, and Google's Gemini Pro large language model to inform investment strategies.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Use Cases](#use-cases)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Structure](#structure)
- [License](#license)
- [Hardware Requirements](#hardware-requirements)

## Introduction

The Gemini Candlestick Investment Fund application is a sophisticated tool designed to simulate the dynamics of a board of directors' meeting. It analyzes stock market data and formulates hypothetical investment strategies by leveraging candlestick charts and technical indicators. Utilizing the capabilities of Google Gemini Pro, the application mirrors real-world financial analysis and decision-making processes, providing a comprehensive and immersive experience.

### Board Members

The fictional board of directors includes renowned figures in the field of financial analysis, each assigned to analyze specific indicators and provide their expert opinions:

1. **Munehisa Homma**, Chairman of the Meeting: Analyzes Candlestick Pattern Recognition and provides opinions.
2. **John Bollinger**: Analyzes the Bollinger Bands (BB) indicator and provides opinions.
3. **Leonardo Pisano Fibonacci**: Analyzes Fibonacci retracement levels and provides opinions.
4. **Ralph Nelson Elliott**: Analyzes potential Elliott Wave patterns utilizing Fibonacci ratios (0.786, 0.618, 0.5, 0.382, and 0.236 on Fibonacci retracement charts), RSI, MACD, and Bollinger Bands to determine the extent of wave retracements and projections within the patterns, and provides opinions.
5. **Goichi Hosoda**: Analyzes the Ichimoku Cloud and provides opinions.
6. **Joseph Granville**: Analyzes Volume and On-Balance Volume (OBV) indicators and provides opinions.
7. **Gerald Appel**: Analyzes the MACD indicator and provides opinions.
8. **J. Welles Wilder**: Analyzes RSI and ATR indicators and provides opinions.
9. **George Lane**: Analyzes the Stochastic Oscillator indicator and provides opinions.
10. **Marc Chaikin**: Analyzes the Chaikin Money Flow indicator and provides opinions.

## Features

- **Data Acquisition:** Efficiently downloads historical candlestick data for selected stocks from the S&P 100 and Nasdaq 100 indices using the `yfinance` library.
- **Technical Analysis:** Computes a comprehensive suite of technical indicators, including:
    - Relative Strength Index (RSI)
    - Moving Average Convergence Divergence (MACD)
    - Bollinger Bands
    - Fibonacci Retracement Levels
    - Ichimoku Cloud
    - Stochastic Oscillator
    - Chaikin Money Flow
    - On-Balance Volume (OBV)
    - Average True Range (ATR)
- **Candlestick Charting:** Produces detailed and visually appealing candlestick charts with technical indicators overlaid, utilizing `matplotlib` and `mplfinance`.
- **Gemini Pro Integration:** Leverages Google's Gemini Pro large language model to:
    - Simulate the discussions and analyses of fictional board members.
    - Generate detailed meeting minutes, encompassing market observations, individual stock analyses, and the rationale behind investment decisions.
    - Summarize key takeaways, actionable insights, and the fund's overall market strategy.
    - Create engaging captions for generated charts and reports.
- **Telegram Integration:** Automatically delivers daily reports, summaries, and highlighted candlestick charts to a designated Telegram chat using the Telegram Bot API.

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