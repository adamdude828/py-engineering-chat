import click
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import chromadb
from chromadb.config import Settings
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from py_engineering_chat.agents.text_summarizer import TextSummarizer
from py_engineering_chat.util.chat_settings_manager import ChatSettingsManager
from py_engineering_chat.util.logger_util import get_configured_logger
import sys
from contextlib import contextmanager

from py_engineering_chat.research.web_crawler import WebCrawler  # Import the WebCrawler class

@contextmanager
def suppress_stdout_stderr():
    """A context manager to suppress stdout and stderr."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def annotate_docs(doc_annotation):
    manager = ChatSettingsManager()
    path = "docs"  # Directly use "docs" at the root level
    manager.append_to_collection(path, doc_annotation)
    print(f"Annotated '{doc_annotation}' to the docs collection")

def crawl():
    data = request.json
    url = data.get('url')
    depth = data.get('depth')
    collection_name = data.get('collection_name')
    
    if not url or not depth or not collection_name:
        return jsonify({"error": "Missing parameters"}), 400

    crawl_and_store(url, depth, collection_name)
    return jsonify({"message": "Crawling started"}), 200

def crawl_and_store(url, depth, collection_name, debug=False, suppress_output=True, max_urls=None):
    # Initialize logger
    logger = get_configured_logger('crawler')

    # Print the input parameters
    print(f"URL: {url}")
    print(f"Depth: {depth}")
    print(f"Collection: {collection_name}")

    # Load environment variables
    load_dotenv(dotenv_path='../.env')

    # Get AI_SHADOW_DIRECTORY from environment variables
    ai_shadow_directory = os.getenv('AI_SHADOW_DIRECTORY')
    if not ai_shadow_directory:
        raise ValueError("AI_SHADOW_DIRECTORY environment variable is not set")

    # Initialize Chroma client
    chroma_db_path = os.path.join(ai_shadow_directory, '.chroma_db')
    logger.debug(f"Chroma DB Path: {chroma_db_path}")
    client = chromadb.PersistentClient(path=chroma_db_path)

    # Check if the collection exists before deleting
    collections = client.list_collections()
    if collection_name in [col.name for col in collections]:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection '{collection_name}'.")
    else:
        print(f"Collection '{collection_name}' does not exist.")

    # Create a new collection
    collection = client.create_collection(name=collection_name)

    # Initialize SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Initialize TextSummarizer
    summarizer = TextSummarizer()

    # Initialize WebCrawler
    crawler = WebCrawler(starting_url=url, max_depth=depth, max_urls=max_urls or 100)

    # Crawl the web and collect items
    crawled_items = []
    for crawled_url, response in crawler.crawl():
        #summary = summarizer.summarize(response.text)
        #print(f"Summary: {summary}")
        crawled_items.append({
            'url': crawled_url,
            'content': response.text,
        })
        # Log each link scanned
        logger.info(f"Scanned link: {crawled_url}")

    # Use the collected items instead of accessing spider.items
    results = len(crawled_items)

    # Modify the data preparation for Chroma
    ids = [str(i) for i in range(results)]
    documents = [item['summary'] for item in crawled_items]

    if not documents:
        print("No documents to process. Exiting.")
        return

    metadatas = [{"url": item['url']} for item in crawled_items]
    
    # Calculate embeddings
    embeddings = model.encode(documents)

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    if debug:
        print(f"Crawled, summarized, calculated embeddings, and stored {results} pages (summaries only) in collection '{collection_name}'")

    # Example usage of annotate_docs
    annotate_docs(collection_name)

if __name__ == '__main__':
    app.run(port=3000)  # Run Flask server on port 3000