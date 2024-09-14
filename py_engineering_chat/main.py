from dotenv import load_dotenv
import os
import sys
import click
import warnings
from agents.docs_agent import chat_with_docs_agent
from agents.planning_agent import run_conversation_planning_agent
from py_engineering_chat.research.research import crawl_and_store
from py_engineering_chat.research.list_collections import list_collections, list_collection_content
from py_engineering_chat.util.add_codebase import add_codebase
from py_engineering_chat.research.scan_codebase import scan_codebase
from py_engineering_chat.agents.general_agent import run_continuous_conversation  # Import the new function
from py_engineering_chat.util.logger_util import get_configured_logger  # Import the logger utility
import requests
from py_engineering_chat.agents.text_summarizer import TextSummarizer
from py_engineering_chat.util.content_chunker import ContentChunker  # Import the ContentChunker class

# Suppress the specific warning
warnings.filterwarnings("ignore", message=".*`clean_up_tokenization_spaces` was not set.*")

# Load environment variables from .env file
load_dotenv()

@click.group()
def cli():
    pass

@cli.command()
def chat_general():
    """Chat with the tools agent."""
    run_continuous_conversation()  # Call the new function

@cli.command()
def chat_docs():
    """Chat with the docs agent."""
    chat_with_docs_agent()

@cli.command()
def chat_planning():
    """Chat with the planning agent."""
    run_conversation_planning_agent()

@cli.command()
@click.argument('url')
@click.option('--depth', default=1, help='Crawl depth')
@click.option('--partition', required=True, help='Milvus partition name')
@click.option('--debug', is_flag=True, default=False, help='Enable debug output')
@click.option('--suppress-output', is_flag=True, default=True, help='Suppress output')
@click.option('--max-urls', type=int, default=None, help='Maximum number of URLs to crawl')  # New option
def research(url, depth, partition, debug, suppress_output, max_urls):
    """Crawl a URL and store the results in Milvus."""
    crawl_and_store(url, depth, partition, debug, suppress_output, max_urls)

@cli.command()
def list_chroma_collections():
    """List available collections in Chroma."""
    list_collections()

@cli.command()
@click.argument('collection_name')
def list_content(collection_name):
    """List content of a specific collection in Chroma."""
    list_collection_content(collection_name)

@cli.command()
@click.argument('project_name')
@click.argument('directory')
@click.argument('github_origin')
def add_codebase_command(project_name, directory, github_origin):
    """Add a codebase to the project."""
    add_codebase(project_name, directory, github_origin)

@cli.command()
@click.argument('project_name')
@click.option('--skip-summarization', is_flag=False, default=False, help='Skip model summarization')
@click.option('--max-file-count', type=int, default=-1, help='Maximum number of files to process')
def scan_project(project_name, skip_summarization, max_file_count):
    """Scan a project's codebase and store in Chroma."""
    scan_codebase(project_name, skip_summarization, max_file_count)

@cli.command()
@click.argument('url')
def summarize_url(url):
    """Fetch content from a URL, summarize it, and output the result."""
    logger = get_configured_logger('summarize_url')
    try:
        # Fetch content from the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        html_content = response.text

        # Initialize the ContentChunker with your OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")  # Ensure your .env file has this key
        chunker = ContentChunker(openai_api_key=openai_api_key)

        # Process the HTML content
        plain_text_chunks = chunker.process_html(html_content)

        # Output the plain text with added spaces and color
        logger.info(f"Processed {len(plain_text_chunks)} chunks from URL: {url}")
        print("Processed Text:")
        colors = ["\033[94m", "\033[92m", "\033[91m"]  # Blue, Green, and Red colors
        for i, chunk in enumerate(plain_text_chunks):
            color = colors[i % len(colors)]  # Alternate between colors
            print(color + chunk.chunk + "\033[0m\n")  # Apply color and a newline for spacing
    except requests.RequestException as e:
        logger.error(f"Failed to fetch content from URL: {e}", exc_info=True)
        print(f"Failed to fetch content from URL: {e}", file=sys.stderr)



if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)