from dotenv import load_dotenv
import os
import sys
import click
from agents.basic_agent import chat_with_basic_agent
from agents.tools_agent import chat_with_tools_agent
from py_engineering_chat.research.research import crawl_and_store  # Updated import statement

# Load environment variables from .env file
load_dotenv()

@click.group()
def cli():
    pass

@cli.command()
def chat_basic():
    """Chat with the basic agent."""
    chat_with_basic_agent()

@cli.command()
def chat_tools():
    """Chat with the tools agent."""
    chat_with_tools_agent()

@cli.command()
@click.argument('url')
@click.option('--depth', default=1, help='Crawl depth')
@click.option('--partition', required=True, help='Milvus partition name')
def research(url, depth, partition):
    """Crawl a URL and store the results in Milvus."""
    crawl_and_store(url, depth, partition)

if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)