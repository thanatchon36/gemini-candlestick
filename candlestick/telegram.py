"""Sending messages, PDFs, and images to a Telegram chat via the Bot API."""
import json
import time
import logging
import requests

logger = logging.getLogger(__name__)

class TelegramBot:
    """Sends content to one Telegram chat."""

    def __init__(self, BOT_TOKEN, CHAT_ID):
        """
        Args:
            BOT_TOKEN (str): Your Telegram bot token.
            CHAT_ID (str): Your Telegram chat ID.
        """
        self.BOT_TOKEN = BOT_TOKEN
        self.CHAT_ID = CHAT_ID

    def send_message(self, message):
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
                logger.info(msg="Message sent successfully!")
            else:
                # If the status code is not 200, there was an error
                logger.error(exc_info=True ,msg=f"Error: {response.status_code} - {response.text}")
                # Print the error code and the error message from the Telegram API

    def send_pdfs(self, pdf_paths, caption_list):
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
                    logger.info(f"Status Code for {pdf_file}: {response.status_code}")  # Print the status code for the request
                    if response.status_code == 200:
                        # If the status code is 200 (OK), the file was sent successfully
                        logger.info(f"File {pdf_file} sent successfully!")
                    else:
                        # If the status code is not 200, there was an error
                        logger.error(exc_info=True, msg=f"Error sending {pdf_file}: {response.status_code} - {response.text}")
                        # Print the error code and the error message from the Telegram API
            time.sleep(168)  # Wait for 168 seconds

    def send_group_pdfs(self, pdf_paths, caption_list):
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
            logger.info(f"Status Code: {response.status_code}")  # Print the status code for the request
            if response.status_code == 200:
                # If the status code is 200 (OK), the files were sent successfully
                logger.info("Files sent successfully!")
            else:
                # If the status code is not 200, there was an error
                logger.error(exc_info=True, msg=f"Error: {response.status_code} - {response.text}")
                # Print the error code and the error message from the Telegram API

    def send_images(self, image_paths, caption_list):
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
                        logger.info(f"Image {photo_file} sent successfully!")
                    else:
                        # If the status code is not 200, there was an error
                        logger.error(exc_info=True, msg=f"Error sending {photo_file}: {response.status_code} - {response.text}")
                        # Print the error code and the error message from the Telegram API

    def send_group_images(self, image_paths, caption_list):
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
            logger.info(f"Status Code: {response.status_code}")  # Print the status code for the request
            if response.status_code == 200:
                # If the status code is 200 (OK), the images were sent successfully
                logger.info("Images sent successfully!")
            else:
                # If the status code is not 200, there was an error
                logger.error(exc_info=True, msg=f"Error: {response.status_code} - {response.text}")
                # Print the error code and the error message from the Telegram API
