#!/usr/bin/env python
from geminicandlestick import GeminiCandlestick
import os
import sys
import time

# Print a message to indicate successful initialization
print('OK !', flush=True)

def main():
    """
    Main function to run the candlestick generation and Telegram notification process.
    """

    # Define the directories to store different data types
    data_directories = [
        'data/pdf',
        'data/csv',
        'data/png'
    ]

    # Loop through each directory and create it if it doesn't exist
    for dir_path in data_directories:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Get environment variables
    gemini_key = str(os.getenv('gemini_key'))
    BOT_TOKEN = str(os.getenv('BOT_TOKEN'))
    CHAT_ID = str(os.getenv('CHAT_ID'))

    # Create an instance of the geminicandlestick class
    gemini_instance = GeminiCandlestick(gemini_key=gemini_key,
                                        BOT_TOKEN=BOT_TOKEN,
                                        CHAT_ID=CHAT_ID,
                                        freq_interval='1d')
    
    # Wait until the next day at 00:00
    time.sleep(gemini_instance.until_next_day_sec)

    while True:
        try:
            # Record the start time for candlestick generation
            start_time = time.time()

            # Generate Gemini candlestick data and charts
            gemini_instance.generate_gemini_candlestick()

            # Record the end time and calculate runtime
            end_time = time.time()
            generate_candlestick_runtime = end_time - start_time

            # Calculate sleep time until 01:30 AM
            target_time_0130 = 1.5 * 60 * 60  # 01:30 AM in seconds (1.5 hours)
            sleep_until_0130 = target_time_0130 - generate_candlestick_runtime
            
            # Sleep until 01:30 AM if there is time left
            if sleep_until_0130 > 0:
                time.sleep(sleep_until_0130)

            # At 01:30 AM:
            
            # Send PDF reports through Telegram
            gemini_instance.telegram_send_group_pdfs(
                [f"data/pdf/{gemini_instance.file_date}_minutes.pdf", f"data/pdf/{gemini_instance.file_date}_summary.pdf"],
                [gemini_instance.telegram_minutes_text, gemini_instance.telegram_summary_text]
            )

            # Send generated images through Telegram
            gemini_instance.telegram_send_group_images(gemini_instance.image_paths, gemini_instance.photo_caption_list)

            # Sleep until the next day at 00:00
            time.sleep(gemini_instance.until_next_day_sec)

        except Exception as e:
            # Error handling: Get exception information and print an error message
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno)
            gemini_instance.docker_print(temp_msg)

if __name__ == '__main__':
    main()
