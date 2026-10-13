#!/usr/bin/env python
"""Entry point: runs the Gemini Candlestick Investment Fund on a daily schedule (UTC)."""
import os
import sys
import time
import logging

from candlestick import clock
from candlestick.config import DATA_DIRECTORIES, DEFAULT_GEMINI_MODEL, MAX_MEDIA_GROUP_SIZE
from candlestick.fund import GeminiCandlestick
from candlestick.telegram import TelegramBot

# Configuration of Logger
logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Print a message to indicate the script has started
logger.info(msg='OK !')

def send_reports(fund, telegram, batch_wait_sec):
    """
    Sends the latest PDF reports, then the highlighted charts in batches.

    Args:
        fund (GeminiCandlestick): The fund after `generate_gemini_candlestick()` has run.
        telegram (TelegramBot): Where to send the reports.
        batch_wait_sec (int): Seconds to wait before each batch of charts (avoids Telegram rate limits).
    """
    # Send the PDF reports only if both captions were generated
    if fund.telegram_minutes_text != "" and fund.telegram_summary_text != "":
        telegram.send_pdfs(
            [   # List of PDF file paths to send
                f"data/pdf/{clock.file_date()}_minutes.pdf",
                f"data/pdf/{clock.file_date()}_summary.pdf"
            ],
            [   # List of captions for the PDFs
                fund.telegram_minutes_text,
                fund.telegram_summary_text
            ]
        )

    # Send the charts as Telegram albums of up to MAX_MEDIA_GROUP_SIZE images
    for i in range(0, len(fund.image_paths), MAX_MEDIA_GROUP_SIZE):
        time.sleep(batch_wait_sec)
        batch_images = fund.image_paths[i:i + MAX_MEDIA_GROUP_SIZE]
        batch_captions = fund.photo_caption_list[i:i + MAX_MEDIA_GROUP_SIZE]
        telegram.send_group_images(batch_images, batch_captions)


def main():
    """
    Runs the fund continuously:

    1. On start, generates a report and sends it straight away.
    2. Every day at 00:00 UTC, generates a new report (charts, board meeting, PDFs).
    3. Sends the report around 02:00 UTC via Telegram.
    """

    # Create the data directories if they don't exist
    for dir_path in DATA_DIRECTORIES:
        os.makedirs(dir_path, exist_ok=True)

    # Retrieve environment variables for API keys and configurations
    gemini_key = os.getenv('GEMINI_API_KEY') or os.getenv('gemini_key')  # Gemini API key ('gemini_key' kept for old setups)
    BOT_TOKEN = os.getenv('BOT_TOKEN')    # Telegram Bot Token
    CHAT_ID = os.getenv('CHAT_ID')        # Telegram Chat ID
    gemini_model = os.getenv('GEMINI_MODEL') or DEFAULT_GEMINI_MODEL  # Gemini model name

    # Stop early with a clear message if any required setting is missing
    missing = [name for name, value in [('GEMINI_API_KEY', gemini_key), ('BOT_TOKEN', BOT_TOKEN), ('CHAT_ID', CHAT_ID)] if not value]
    if missing:
        sys.exit(f"Missing environment variables: {', '.join(missing)}. Copy .env.example to .env and fill them in.")

    fund = GeminiCandlestick(
        gemini_key=gemini_key,
        freq_interval='1d',  # Set data frequency to daily ('1d')
        gemini_model=gemini_model
    )
    telegram = TelegramBot(BOT_TOKEN, CHAT_ID)

    # First run: generate and send immediately
    logger.info(msg=clock.today_time())
    fund.generate_gemini_candlestick()
    send_reports(fund, telegram, batch_wait_sec=168)
    logger.info(msg=clock.today_time())

    # Wait until the next day at 00:00 before starting the main loop
    time.sleep(clock.until_next_day_sec())

    # Main loop to continuously generate data, create charts, and send notifications
    while True:
        try:
            # Record the start time for candlestick generation
            start_time = time.time()

            # Wait 4 minutes before starting the process
            time.sleep(60 * 4)

            # Generate the Gemini candlestick data and charts
            fund.generate_gemini_candlestick()

            # Log the current date and time
            logger.info(msg=clock.today_time())

            # Sleep until 02:00, so reports are sent at the same time every day
            generate_candlestick_runtime = time.time() - start_time
            target_time_0X00 = 2.0 * 60 * 60 - generate_candlestick_runtime
            if target_time_0X00 > 0:
                time.sleep(target_time_0X00)

            send_reports(fund, telegram, batch_wait_sec=180)

        except Exception as e:
            # Error handling: Get exception information
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

            # Log the formatted error message
            logger.error(exc_info=True ,msg=f"Error !: {e}, {exc_type}, {fname}, {exc_tb.tb_lineno}")

        # Reset the outputs for the next iteration
        fund.telegram_minutes_text = ""
        fund.telegram_summary_text = ""
        fund.image_paths = []
        fund.photo_caption_list = []

        # Sleep until the next day at 00:00 before generating new data
        time.sleep(clock.until_next_day_sec())


# Run the main function if the script is executed
if __name__ == '__main__':
    main()
