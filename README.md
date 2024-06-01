# Gemini Candlestick Investment Fund Simulation

This repository contains a Python application that simulates the investment decision-making process of a fictional investment board, the "Gemini Candlestick Investment Fund," using candlestick chart analysis, technical indicators, and Google's Gemini Pro large language model.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Use Cases](#use-cases)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Structure](#structure)
- [Contributing](#contributing)
- [License](#license)
- [Hardware Requirements](#hardware-requirements)

## Introduction

The Gemini Candlestick Investment Fund application simulates a board of directors' meeting, analyzing stock market data and generating hypothetical investment strategies. It leverages candlestick charts, technical indicators, and the power of Google Gemini Pro to simulate real-world financial analysis and decision-making processes.

## Features

- **Data Acquisition:** Downloads historical candlestick data for a curated list of stocks from the S&P 100 and Nasdaq 100 indices using the `yfinance` library.
- **Technical Analysis:** Calculates a variety of technical indicators, including:
    - RSI
    - MACD
    - Bollinger Bands
    - Fibonacci Retracement
    - Ichimoku Cloud
    - Stochastic Oscillator
    - Chaikin Money Flow
    - On-Balance Volume
    - Average True Range
- **Candlestick Charting:** Generates visually appealing candlestick charts with technical indicators overlaid using the `matplotlib` and `mplfinance` libraries.
- **Gemini Pro Integration:** Utilizes Google's Gemini Pro large language model for:
    - Simulating the discussions and analysis of the fictional board members.
    - Generating comprehensive meeting minutes, including market observations, individual stock analyses, and the rationale behind investment decisions.
    - Summarizing key takeaways, actionable insights, and the fund's overall market position.
    - Crafting engaging captions for generated charts and reports.
- **Telegram Integration:** Sends daily reports, summaries, and highlighted candlestick charts directly to a designated Telegram chat using the Telegram Bot API.

## Use Cases

- **Educational Tool:** Provides a practical example of how candlestick charts and technical indicators are used in financial analysis.
- **Simulation Environment:** Offers a safe and controlled setting to experiment with different investment strategies and observe their hypothetical outcomes.
- **Algorithmic Trading Inspiration:** The project can inspire the development of automated trading strategies based on technical indicators and sentiment analysis derived from LLMs.
- **Financial Education:** Can be used to learn about different technical indicators, candlestick patterns, and their interpretations in a simulated environment.

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

- **`gemini_key`:** Your Google Gemini API key. Replace `YOUR_GEMINI_API_KEY` with your actual key.
- **`BOT_TOKEN`:** Your Telegram bot token. Replace `YOUR_TELEGRAM_BOT_TOKEN` with your bot's token.
- **`CHAT_ID`:** The ID of your Telegram chat. Replace `YOUR_TELEGRAM_CHAT_ID` with your chat's ID.

**Make sure to replace the placeholders with your actual credentials.**

## Structure

- **`docker-compose.yml`:** Defines the Docker container for running the application.
- **`Dockerfile.custom`:** Specifies instructions for building the Docker image, including installing dependencies and setting up the environment.
- **`geminicandlestick.py`:** Contains the main class, `GeminiCandlestick`, which handles data fetching, processing, charting, and Telegram interactions.
- **`main.py`:** The entry point of the application. Manages the overall workflow of data generation, charting, and Telegram notifications.

## Contributing

Contributions are welcome! 

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a new Pull Request.

## License

This project is submitted to The Gemini API Developer Competition, sponsored by Google LLC. The submission is subject to the [Official Rules of the competition](https://ai.google.dev/competition), including the intellectual property provisions. 

## Hardware Requirements

- **Minimum & Recommended:** CPU: 2 cores (ARM64 architecture), RAM: 2GB, Storage: 16 GB

This application has been developed and tested on a MacBook M1, which comfortably meets these requirements.