from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
from py_engineering_chat.util.logger_util import get_configured_logger
import json
from typing import List  # Import List from typing

class TextChunk(BaseModel):
    chunks: List[str] = Field(..., description="A list of text chunks extracted from the content.")  # Use List[str]

class TextSummarizer:
    def __init__(self):
        self.logger = get_configured_logger(__name__)
        self.llm = ChatOpenAI(temperature=0, model="gpt-4", verbose=True)  # Corrected model name
        
        # Define the output parser with the Pydantic model
        self.output_parser = PydanticOutputParser(pydantic_object=TextChunk)
        
        # Get the format instructions for the model to produce the output in the desired schema
        format_instructions = self.output_parser.get_format_instructions()
        
        self.summary_prompt = ChatPromptTemplate.from_template(
            "You are an AI assistant that processes text by dividing it into natural chunks. "
            "Given the following text, extract all textual content (ignoring any HTML tags) and divide it into coherent, self-contained chunks. "
            "Ensure that the output is in JSON format as specified below.\n\n"
            f"{format_instructions}\n\n"
            "Text:\n{content}\n\n"
            "Output the result in JSON format."
        )
        self.summary_chain = self.llm | self.output_parser

    def summarize(self, content: str) -> TextChunk:
        self.logger.debug("Starting summarization process.")
        
        # Strip HTML tags from the content
        soup = BeautifulSoup(content, 'html.parser')
        text_content = soup.get_text(separator='\n')  # Retain line breaks where appropriate
        self.logger.debug("Extracted text content from HTML.")

        # Prepare the prompt with the format instructions
        prompt_value = self.summary_prompt.format_prompt(
            content=text_content
        )
        
        # Invoke the LLM with the cleaned text and get the parsed output
        try:
            raw_result = self.summary_chain.invoke(text_content)
            self.logger.debug("Successfully obtained result from LLM.")
            
            # Validate and parse the JSON output
            json_result = json.loads(raw_result)
            result = self.output_parser.parse_obj(json_result)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decoding error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"An error occurred during summarization: {e}")
            raise
        
        print(result)
        
        return result  # This will be an instance of TextChunk containing chunks

    def __call__(self, content: str) -> TextChunk:
        return self.summarize(content)