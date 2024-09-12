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
@click.option('--keep-full-content', is_flag=True, default=True, help='Keep full content in addition to summary')
@click.option('--debug', is_flag=True, default=False, help='Enable debug output')
@click.option('--suppress-output', is_flag=True, default=True, help='Suppress output')
def research(url, depth, partition, keep_full_content, debug, suppress_output):
    """Crawl a URL and store the results in Milvus."""
    crawl_and_store(url, depth, partition, keep_full_content, debug, suppress_output)

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

if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)