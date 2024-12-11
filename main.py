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
        'data/png',   # Directory for storing PNG image charts
        'data/png_optimized', # Directory for storing optimized PNG image charts
        'data/txt',   # Directory for storing text files
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

    # Debug
    gemini_instance.docker_print(gemini_instance.today_time)
    gemini_instance.generate_gemini_candlestick()
    gemini_instance.telegram_send_pdfs(
        [   # List of PDF file paths to send
            f"data/pdf/{gemini_instance.file_date}_minutes.pdf", 
            f"data/pdf/{gemini_instance.file_date}_summary.pdf"
        ],
        [   # List of captions for the PDFs
            gemini_instance.telegram_minutes_text,
            gemini_instance.telegram_summary_text
        ]
    )
    max_batch_size = 8
    image_paths = gemini_instance.image_paths
    photo_caption_list = gemini_instance.photo_caption_list
    for i in range(0, len(image_paths), max_batch_size):
        time.sleep(168)  # Wait for 168 seconds
        end_index = min(i + max_batch_size, len(image_paths))
        batch_images = image_paths[i:end_index]
        batch_captions = photo_caption_list[i:end_index]
        gemini_instance.telegram_send_group_images(batch_images, batch_captions)
    gemini_instance.docker_print(gemini_instance.today_time)

    # Wait until the next day at 00:00 before starting the main loop
    # This ensures the script starts generating data at the beginning of each day
    time.sleep(gemini_instance.until_next_day_sec)

    # Main loop to continuously generate data, create charts, and send notifications
    while True:
        try:
            # Record the start time for candlestick generation
            start_time = time.time()

            # Wait 4 minutes before starting the process
            time.sleep(60 * 4)

            # Generate the Gemini candlestick data and charts
            gemini_instance.generate_gemini_candlestick()

            # Log the current date and time
            gemini_instance.docker_print(gemini_instance.today_time)

            # Record the end time and calculate the runtime for data generation
            end_time = time.time()
            generate_candlestick_runtime = end_time - start_time

            # Calculate the sleep time until 0X:00 AM
            # The script aims to send reports and charts at this time
            target_time_0X00 = 2.0 * 60 * 60  # 0X:00 AM in seconds (X.0 hours)
            target_time_0X00 = target_time_0X00 - generate_candlestick_runtime

            # Sleep until 0X:00 AM if there's time left after data generation
            if target_time_0X00 > 0:
                time.sleep(target_time_0X00)

            # Send the generated reports and charts via Telegram at 0X:00 AM
            
            # Check if there's text content for both minutes and summary reports
            if gemini_instance.telegram_minutes_text != "" and gemini_instance.telegram_summary_text != "":
                # Send the PDF reports with their respective captions
                gemini_instance.telegram_send_pdfs(
                    [   # List of PDF file paths to send
                        f"data/pdf/{gemini_instance.file_date}_minutes.pdf", 
                        f"data/pdf/{gemini_instance.file_date}_summary.pdf"
                    ],
                    [   # List of captions for the PDFs
                        gemini_instance.telegram_minutes_text, 
                        gemini_instance.telegram_summary_text
                    ]
                )

            # Set the maximum batch size for processing images.
            max_batch_size = 8

            # Get the list of image paths and their corresponding captions.
            image_paths = gemini_instance.image_paths
            photo_caption_list = gemini_instance.photo_caption_list

            # Iterate over the image paths in batches.
            for i in range(0, len(image_paths), max_batch_size):
                time.sleep(180)  # Wait for 180 seconds
                # Calculate the end index for the current batch.
                end_index = min(i + max_batch_size, len(image_paths))  # Ensure end_index doesn't go out of bounds

                # Extract the batch of images and captions.
                batch_images = image_paths[i:end_index]
                batch_captions = photo_caption_list[i:end_index]

                # Send the batch of images and captions using the gemini instance.
                gemini_instance.telegram_send_group_images(batch_images, batch_captions)
                
        except Exception as e:
            # Error handling: Get exception information
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

            # Format the error message
            temp_msg = 'Error !: {} {} {} {}'.format(e, exc_type, fname, exc_tb.tb_lineno)

            # Log the error message (assuming docker_print is a logging method)
            gemini_instance.docker_print(temp_msg)

        # Reset Gemini instance variables for the next iteration or request.
        gemini_instance.telegram_minutes_text = ""  # Clear the text for the minutes summary.
        gemini_instance.telegram_summary_text = ""  # Clear the text for the overall summary.
        gemini_instance.image_paths = []  # Reset the list of image paths.
        gemini_instance.photo_caption_list = []  # Reset the list of captions for the images.

        # Sleep until the next day at 00:00 before generating new data
        time.sleep(gemini_instance.until_next_day_sec)

# Run the main function if the script is executed
if __name__ == '__main__':
    main()