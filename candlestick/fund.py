"""The Gemini Candlestick Investment Fund: charts the S&P 500 and holds a daily AI board meeting."""
import gc
import logging
import os
import random
import sys
import time

import numpy as np
from tqdm import tqdm

from candlestick import clock, prompts
from candlestick.charts import draw_candlestick_chart
from candlestick.config import CANDLESTICK_CHART_NO, DEFAULT_GEMINI_MODEL, FUTURE_CLOUD_NO, GRAPH_WIDTH
from candlestick.gemini import LONG_OUTPUT_CONFIG, SHORT_OUTPUT_CONFIG, Gemini
from candlestick.indicators import add_indicators
from candlestick.market_data import load_sp500
from candlestick.parsing import MINUTES_START_MARKERS, clean_minutes_text, extract_json
from candlestick.reports import images_to_pdf, markdown_to_pdf, merge_pdfs

logger = logging.getLogger(__name__)

class GeminiCandlestick:
    """
    Runs one daily cycle: download prices, draw charts, generate the board meeting
    with Gemini, and build the PDF reports and Telegram captions.

    After `generate_gemini_candlestick()`, the results are in `telegram_minutes_text`,
    `telegram_summary_text`, `image_paths`, and `photo_caption_list`.
    """

    def __init__(self, gemini_key, freq_interval="1d", gemini_model=DEFAULT_GEMINI_MODEL):
        """
        Initializes the GeminiCandlestick object.

        Args:
            gemini_key (str): Your Gemini API key.
            freq_interval (str, optional): The frequency interval for candlestick data. Defaults to "1d" (daily).
            gemini_model (str, optional): The Gemini model name.
        """
        self.gemini_key = gemini_key
        self.gemini_model = gemini_model
        self.freq_interval = freq_interval

        # Outputs of the latest run
        self.telegram_minutes_text = ""
        self.telegram_summary_text = ""
        self.image_paths = []
        self.photo_caption_list = []

    @property
    def ticker_company(self):
        """Returns the company name for the current ticker."""
        return self.ticker_company_dict[self.ticker]

    @property
    def ticker_list(self):
        """Returns the list of tickers with price data."""
        return list(self.sp500_df['symbol'].values)

    def prep_sp500_dataset(self):
        """Downloads the S&P 500 list and daily prices for every ticker."""
        self.sp500_df, self.sp500_df_dict, self.ticker_company_dict = load_sp500()

    def get_candlestick_image(self):
        """Draws and saves the candlestick chart for the current ticker."""
        ohlc = add_indicators(self.sp500_df_dict[self.ticker], CANDLESTICK_CHART_NO, FUTURE_CLOUD_NO)
        draw_candlestick_chart(ohlc, self.ticker, self.ticker_company, GRAPH_WIDTH[self.freq_interval], FUTURE_CLOUD_NO)

    def generate_gemini_candlestick(self):
        """Generates reports and summaries based on candlestick chart data using the Gemini API."""

        # Configure the Gemini API
        self.gemini = Gemini(self.gemini_key, self.gemini_model)

        # Set up the model parameters for Gemini
        generation_config = LONG_OUTPUT_CONFIG

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
            elif clock.today_date().lower() not in ['sunday', 'monday']:
                # Prepare the S&P 500 datasets for analysis
                self.prep_sp500_dataset()
                for each_ticker in tqdm(self.sp500_df['symbol'].values, desc="Generating get_candlestick_image()"):
                    self.ticker = each_ticker
                    self.get_candlestick_image()

            # Initialize an empty list to store prompt parts
            self.prompt_parts = []
            
            # Iterate over each ticker in the ticker list with a progress bar
            for each_ticker in tqdm(self.ticker_list, desc="gemini.upload"):
                # Try uploading the chart image for the current ticker 6 times
                for _ in range(6):
                    try:
                        # Set the current ticker
                        self.ticker = each_ticker
                        
                        # Upload the chart image to GenAI and store the returned object
                        temp_file = self.gemini.upload(
                            f"data/png_optimized/{each_ticker}.png",  # Path to the image file
                            f'{self.ticker_company} ({each_ticker}): 1d Candlestick Chart (with Technical Indicators)'  # Display name for the image
                        )
                        
                        # Append the uploaded file object to the prompt parts list
                        self.prompt_parts.append(temp_file)
                        
                        # Break the inner loop if the upload is successful
                        break
                    except Exception as e:
                        # Log the error and continue to the next iteration without crashing
                        logger.error(exc_info=True, msg=f"Error during genai upload_file: {e}")
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
                    temp_minutes_text = self.gemini.generate(
                        [prompts.minutes_instructions(self.sp500_df)] + self.prompt_parts,
                        generation_config,  # Pass the generation configuration
                    )

                    # Introduce a delay to avoid rate limiting
                    time.sleep(4)

                    # Generate a summary of the meeting minutes
                    temp_summary_text = self.gemini.generate(
                        temp_minutes_text,
                        generation_config,
                        system_instruction=prompts.summary_instructions()  # Set system instructions for summary generation
                    )

                    # Apply quality checks to ensure the generated text meets specific criteria
                    # Check if the generated text does not contain '[' or ']' characters
                    if '[' not in temp_minutes_text and ']' not in temp_minutes_text:
                        if '[' not in temp_summary_text and ']' not in temp_summary_text:
                            # Check if the minutes contain specific keywords
                            if 'Ticker Symbols of Interest'.lower() in temp_minutes_text.lower():
                                if 'approved by' in temp_minutes_text.lower():
                                    # Check if any of the start markers are in temp_minutes_text
                                    if any(item in temp_minutes_text for item in MINUTES_START_MARKERS):
                                        if ", Chairman" in temp_minutes_text:
                                            if "|" not in temp_minutes_text and "|" not in temp_summary_text:
                                                # Append the generated minutes and summary to their respective lists
                                                temp_minutes_text = clean_minutes_text(temp_minutes_text)
                                                minutes_text_list.append(temp_minutes_text)
                                                summary_text_list.append(temp_summary_text)
                except Exception as e:
                    # Handle exceptions during text generation and print error messages
                    logger.error(exc_info=True, msg=f"Error during temp_minutes_text generation: {e}")
                    pass

            # Zip the lists together, sort by the length of the text in minutes_text_list, and unzip
            sorted_minutes_summary = sorted(zip(minutes_text_list, summary_text_list), key=lambda x: len(x[0]), reverse=True)
            sorted_minutes_text_list, sorted_summary_text_list = zip(*sorted_minutes_summary)

            # Convert the tuples back to lists (if needed)
            sorted_minutes_text_list = list(sorted_minutes_text_list)
            sorted_summary_text_list = list(sorted_summary_text_list)

            minutes_text_list = sorted_minutes_text_list.copy()
            summary_text_list = sorted_summary_text_list.copy()

            # Initialize empty lists to store filtered minutes and summaries
            # minutes_text_list_2 = []
            # summary_text_list_2 = []

            # Iterate through each minute in the original list
            # for i_each_minutes, each_minutes in enumerate(minutes_text_list):
            #     # Initialize a score to track the presence of sectors
            #     sector_score = 0

            #     # Check if each sector is present in the minutes text
            #     for each_sector in self.sector_list:
            #         if each_sector.lower() in each_minutes.lower():
            #             sector_score += 1

            #     # If all sectors are present, add the minutes and corresponding summary to the filtered lists
            #     if sector_score == len(self.sector_list):
            #         minutes_text_list_2.append(each_minutes)
            #         summary_text_list_2.append(summary_text_list[i_each_minutes])

            # # Update the original lists with the filtered results
            # minutes_text_list = minutes_text_list_2.copy()
            # summary_text_list = summary_text_list_2.copy()

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
                        generation_result = self.gemini.generate(
                            each_i[start_index:],
                            generation_config,
                            system_instruction=prompts.ticker_extraction_instructions()
                        )

                        # Extract ticker list from the generated content
                        interest_ticker_list = str(generation_result)
                        extracted_data = extract_json(interest_ticker_list)[0] 
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
                        logger.error(exc_info=True, msg=f"Error during ticker generation: {e}")
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
            file_path = f'data/txt/{clock.file_date()}_minutes.txt'
            # Open the file in write mode ('w'). 
            with open(file_path, 'w') as file:
                # Write the meeting minutes text to the file.
                file.write(minutes_text)

            # Define the file path for saving the meeting minutes.
            file_path = f'data/txt/{clock.file_date()}_summary.txt'
            # Open the file in write mode ('w'). 
            with open(file_path, 'w') as file:
                # Write the meeting minutes text to the file.
                file.write(summary_text)

            # Delete uploaded files from Gemini
            for each_prompt_part in tqdm(self.prompt_parts, desc="gemini.delete"):
                try:
                    self.gemini.delete(each_prompt_part.name)
                except Exception:
                    logger.error(exc_info=True, msg=f"Unable to delete file {each_prompt_part.name}")

            # Generate attached text with retries
            # for _ in tqdm(range(6), desc="Generating attached_text"):  # Try up to 6 times
            #     try:
            #         time.sleep(4)  # Wait for 4 seconds before making the API call

            #         # Generate content using the Gemini Pro model
            #         genai_model = genai.GenerativeModel(
            #             model_name="gemini-1.5-flash-latest",  # Specify the Gemini Pro model
            #             generation_config=generation_config,
            #             system_instruction=prompts.rephrase_instructions(),  # Set system instructions
            #             safety_settings=safety_settings
            #         )
            #         attached_text = str(genai_model.generate_content(
            #             prompts.enclosed_charts_notice()  # Provide instructions for content generation
            #         ).text)

            #         break  # Exit the loop if generation is successful

            #     except Exception as e:
            #         logger.error(exc_info=True, msg=f"Error during attached_text generation: {e}")
            #         pass  # Continue to the next iteration if an error occurs
            
            # Convert the combined text to PDF and save it to the specified path.
            markdown_to_pdf(minutes_text, 'data/pdf/minutes.pdf')
            # markdown_to_pdf(minutes_text + '\n' + attached_text, 'data/pdf/minutes.pdf')

            # Call the markdown_to_pdf method to convert the summary text to a PDF file
            markdown_to_pdf(summary_text, 'data/pdf/summary.pdf')

            # Convert the PNG files to PDF
            png_files = [f'data/png_optimized/{each}.png' for each in self.interest_ticker_list]
            images_to_pdf(png_files, "data/pdf/png.pdf")
            # Merge the generated PDF files
            pdf_paths = ["data/pdf/minutes.pdf", "data/pdf/png.pdf"]
            merge_pdfs(pdf_paths, pdf_paths[0])

            # Set up the model parameters for Gemini
            generation_config = SHORT_OUTPUT_CONFIG
            # Generate summary text for Telegram, retrying up to 6 times with delays
            for _ in tqdm(range(6), desc="Generating telegram_minutes_text"):
                try:
                    time.sleep(4)  # Wait for 4 seconds before each attempt

                    # Generate the summary text using the Gemini model
                    self.telegram_minutes_text = self.gemini.generate(
                        minutes_text,
                        generation_config,
                        system_instruction=prompts.telegram_caption_instructions()
                    )[:900]  # Stay under Telegram's 1024-character caption limit (emoji count double)

                    # Exit the loop if generation is successful
                    break 

                except Exception as e: 
                    # Print error message if generation fails
                    logger.error(exc_info=True, msg=f"Error during telegram_minutes_text generation: {e}")
                    # Continue to the next iteration of the loop
                    pass

            # Set up the model parameters for Gemini
            generation_config = SHORT_OUTPUT_CONFIG
            # Attempt to generate the summary text up to 6 times.
            for _ in tqdm(range(6), desc="Generating telegram_summary_text"):
                try:
                    # Wait for 4 seconds before trying again to avoid overwhelming the API.
                    time.sleep(4)

                    # Generate the summary text using the Gemini model.
                    self.telegram_summary_text = self.gemini.generate(
                        summary_text,
                        generation_config,
                        system_instruction=prompts.telegram_caption_instructions()
                    )[:900]  # Stay under Telegram's 1024-character caption limit (emoji count double)

                    # Exit the loop if successful.
                    break

                except Exception as e: 
                    # Log the error and continue to the next attempt.
                    logger.error(exc_info=True, msg=f"Error during telegram_summary_text generation: {e}")
                    pass

            # Rename the meeting minutes and summary PDF files to include the date.
            os.rename("data/pdf/minutes.pdf", f"data/pdf/{clock.file_date()}_minutes.pdf")
            os.rename("data/pdf/summary.pdf", f"data/pdf/{clock.file_date()}_summary.pdf")

            # Initialize lists to store image paths and prompt parts
            self.image_paths = []
            self.prompt_parts = []

            # Iterate over each ticker in the list
            for each_ticker in tqdm(self.interest_ticker_list, desc="gemini.upload"):
                # Attempt to upload the image 6 times
                for _ in range(6):
                    try:
                        # Set the current ticker
                        self.ticker = each_ticker

                        # Upload the image to genai
                        temp_file = self.gemini.upload(
                            f"data/png_optimized/{each_ticker}.png",  # Path to the image file
                            f'{self.ticker_company} ({each_ticker}): 1d Candlestick Chart (with Technical Indicators)'  # Display name for the image
                        )

                        # Append the uploaded file to prompt parts and image path to image paths list
                        self.prompt_parts.append(temp_file)
                        self.image_paths.append(f'data/png_optimized/{each_ticker}.png')

                        # Exit the retry loop if successful
                        break

                    except Exception as e:
                        # Log the error and continue to the next retry attempt
                        logger.error(exc_info=True, msg=f"Error during genai upload_file: {e}")
                        pass
            
            # Set up the model parameters for Gemini
            generation_config = SHORT_OUTPUT_CONFIG
            # Initialize an empty list to store photo captions.
            self.photo_caption_list = []
            # Iterate through each part of the prompt
            for each_prompt_part in tqdm(self.prompt_parts, desc="Generating photo_caption"):
                # Retry up to 6 times with a 30-second delay
                for _ in range(6):
                    try:
                        time.sleep(4)  # Wait for rate limiting or server congestion
                        # Generate a photo caption using Gemini Pro
                        photo_caption = self.gemini.generate(
                            [prompts.telegram_caption_instructions(), each_prompt_part],
                            generation_config,  # Pass the generation configuration
                        )[:900]  # Stay under Telegram's 1024-character caption limit (emoji count double)

                        # Append the generated caption to the list
                        self.photo_caption_list.append(photo_caption)

                        # Break the retry loop if successful
                        break

                    # Catch any exceptions during generation
                    except Exception as e:
                        logger.error(exc_info=True, msg=f"Error generating caption: {e}")
                        # Continue to the next retry attempt

            # Set up the model parameters for Gemini
            generation_config = LONG_OUTPUT_CONFIG

            # Delete uploaded images from Gemini after generating captions
            for each_prompt_part in tqdm(self.prompt_parts, desc="gemini.delete"):
                try:
                    self.gemini.delete(each_prompt_part.name)
                except Exception:
                    logger.error(exc_info=True, msg=f"Unable to delete image {each_prompt_part.name}")

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
            logger.error(exc_info=True, msg=temp_msg)
            try:
                # Attempt to delete uploaded files from Gemini in case of an error
                for each_prompt_part in tqdm(self.prompt_parts, desc="gemini.delete"):
                    try:
                        self.gemini.delete(each_prompt_part.name)
                    except Exception:
                        logger.error(exc_info=True, msg=f"Unable to delete file {each_prompt_part.name}")
            except Exception as e:
                logger.error(exc_info=True, msg=e)