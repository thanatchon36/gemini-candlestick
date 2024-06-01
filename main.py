#!/usr/bin/env python
from geminicandlestick import GeminiCandlestick
import os
import sys
import time

# Print a message to indicate the script has started
print('OK !', flush=True)

def main():
    """
    Main function to orchestrate the generation of candlestick data, charting, and Telegram notification.

    This script is designed to run continuously, performing the following tasks:
    1. Generates daily candlestick data for various assets.
    2. Creates charts from the generated data.
    3. Sends the generated charts and reports via Telegram to a specified chat ID.
    4. Sleeps until specific times for data generation and sending to maintain a daily schedule.

    The script leverages the `GeminiCandlestick` class to handle data fetching, processing, and Telegram interactions.
    """

    # Define directories for storing different data types (PDF, CSV, PNG)
    data_directories = [
        'data/pdf',  # Directory for storing PDF reports
        'data/csv',  # Directory for storing CSV data files
        'data/png'   # Directory for storing PNG image charts
    ]

    # Create the data directories if they don't exist
    for dir_path in data_directories:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Retrieve environment variables for API keys and configurations
    gemini_key = str(os.getenv('gemini_key'))  # Gemini API key
    BOT_TOKEN = str(os.getenv('BOT_TOKEN'))    # Telegram Bot Token
    CHAT_ID = str(os.getenv('CHAT_ID'))        # Telegram Chat ID

    # Create an instance of the GeminiCandlestick class 
    # This handles data and Telegram operations
    gemini_instance = GeminiCandlestick(
        gemini_key=gemini_key,
        BOT_TOKEN=BOT_TOKEN,
        CHAT_ID=CHAT_ID,
        freq_interval='1d'  # Set data frequency to daily ('1d')
    )

    gemini_instance.docker_print(gemini_instance.today_time)
    gemini_instance.prep_sp100_nasdaq100_dataset()
    gemini_instance.generate_gemini_candlestick()
    gemini_instance.docker_print(gemini_instance.today_time)
    # Send PDF reports (minutes and summary)
    gemini_instance.telegram_send_group_pdfs(
        [   # List of PDF file paths to send
            f"data/pdf/{gemini_instance.file_date}_minutes.pdf", 
            f"data/pdf/{gemini_instance.file_date}_summary.pdf"
        ],
        [   # List of captions for the PDFs
            gemini_instance.telegram_minutes_text, 
            gemini_instance.telegram_summary_text
        ]
    )

    # Send generated images
    gemini_instance.telegram_send_group_images(
        gemini_instance.image_paths,        # List of image paths to send
        gemini_instance.photo_caption_list   # List of captions for the images
    )


    # Wait until the next day at 00:00 before starting the main loop
    # This ensures the script starts generating data at the beginning of each day
    time.sleep(gemini_instance.until_next_day_sec)

    # Main loop to continuously generate data, create charts, and send notifications
    while True:
        try:
            # Record the start time for candlestick generation
            start_time = time.time()

            # Wait 8 minutes before starting the process
            time.sleep(60 * 8)

            # Prepare the S&P 100 and Nasdaq 100 datasets for analysis
            gemini_instance.prep_sp100_nasdaq100_dataset()

            # Generate the Gemini candlestick data and charts
            gemini_instance.generate_gemini_candlestick()

            # Log the current date and time
            gemini_instance.docker_print(gemini_instance.today_time)

            # Record the end time and calculate the runtime for data generation
            end_time = time.time()
            generate_candlestick_runtime = end_time - start_time

            # Calculate the sleep time until 01:30 AM
            # The script aims to send reports and charts at this time
            target_time_0130 = 1.5 * 60 * 60  # 01:30 AM in seconds (1.5 hours)
            sleep_until_0130 = target_time_0130 - generate_candlestick_runtime

            # Sleep until 01:30 AM if there's time left after data generation
            if sleep_until_0130 > 0:
                time.sleep(sleep_until_0130)

            # Send the generated reports and charts via Telegram at 01:30 AM

            # Send PDF reports (minutes and summary)
            gemini_instance.telegram_send_group_pdfs(
                [   # List of PDF file paths to send
                    f"data/pdf/{gemini_instance.file_date}_minutes.pdf", 
                    f"data/pdf/{gemini_instance.file_date}_summary.pdf"
                ],
                [   # List of captions for the PDFs
                    gemini_instance.telegram_minutes_text, 
                    gemini_instance.telegram_summary_text
                ]
            )

            # Send generated images
            gemini_instance.telegram_send_group_images(
                gemini_instance.image_paths,        # List of image paths to send
                gemini_instance.photo_caption_list   # List of captions for the images
            )

        except Exception as e:
            # Error handling: Get exception information
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

            # Format the error message
            temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno)

            # Log the error message (assuming docker_print is a logging method)
            gemini_instance.docker_print(temp_msg)

        # Sleep until the next day at 00:00 before generating new data
        time.sleep(gemini_instance.until_next_day_sec)

# Run the main function if the script is executed
if __name__ == '__main__':
    main()