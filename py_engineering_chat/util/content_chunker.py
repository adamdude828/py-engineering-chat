import openai 
import json 
from bs4 import BeautifulSoup
from pydantic import BaseModel

class ContentChunk(BaseModel):
    chunk: str

class ContentChunkerResponse(BaseModel):
    chunks: list[ContentChunk]


class ContentChunker:
    def __init__(self, openai_api_key):
        """
        Initialize the ContentChunker with the OpenAI API key.
        """
        openai.api_key = openai_api_key

    def process_html(self, html_string):
        """
        Process the HTML string to extract text, split into chunks, and generate keywords
        using OpenAI's function calling with a specified JSON schema.

        :param html_string: The HTML content as a string.
        :return: A list of dictionaries containing chunks and their keywords.
        """
        # Use BeautifulSoup to parse the HTML content and extract text
        soup = BeautifulSoup(html_string, 'html.parser')
        extracted_text = soup.get_text(separator="\n")

        # Prepare the messages for the chat completion
        messages = [
            {"role": "system", "content": "You are an assistant that processes text and extracts chunks with keywords."},
            {"role": "user", "content": f"""
        Process the following text by splitting it into logical chunks (one or two paragraphs each) and provide keywords for
              each chunk:\n\n Here is the contet:\n\n{extracted_text}
              """}
        ]

        # Call the OpenAI ChatCompletion API with function calling
        response = openai.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",  # Ensure the model supports function calling
            messages=messages,
            response_format=ContentChunkerResponse,
        )
        if (response.choices[0].message.parsed):
            return response.choices[0].message.parsed.chunks
        else:
            return response.choices[0].message.refusal

# Usage Example:
