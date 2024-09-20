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
from py_engineering_chat.util.content_chunker import ContentChunker
from py_engineering_chat.research.web_crawler import WebCrawler

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
    try:
        manager = ChatSettingsManager()
        path = "docs"
        manager.append_to_collection(path, doc_annotation)
        print(f"Annotated '{doc_annotation}' to the docs collection")
    except Exception as e:
        print(f"Error annotating docs: {e}")

def crawl_and_store(url, depth, collection_name, debug=False, suppress_output=True, max_urls=None):
    logger = get_configured_logger('crawler')
    
    try:
        print(f"URL: {url}")
        print(f"Depth: {depth}")
        print(f"Collection: {collection_name}")

        load_dotenv(dotenv_path='../.env')

        ai_shadow_directory = os.getenv('AI_SHADOW_DIRECTORY')
        if not ai_shadow_directory:
            raise ValueError("AI_SHADOW_DIRECTORY environment variable is not set")

        chroma_db_path = os.path.join(ai_shadow_directory, '.chroma_db')
        logger.debug(f"Chroma DB Path: {chroma_db_path}")
        
        try:
            client = chromadb.PersistentClient(path=chroma_db_path)
        except Exception as e:
            logger.error(f"Error initializing Chroma client: {e}")
            return

        try:
            collections = client.list_collections()
            if collection_name in [col.name for col in collections]:
                client.delete_collection(name=collection_name)
                print(f"Deleted existing collection '{collection_name}'.")
            else:
                print(f"Collection '{collection_name}' does not exist.")

            collection = client.create_collection(name=collection_name)
        except Exception as e:
            logger.error(f"Error managing collections: {e}")
            return

        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Error initializing SentenceTransformer: {e}")
            return

        summarizer = TextSummarizer()
        crawler = WebCrawler(starting_url=url, max_depth=depth, max_urls=max_urls or 100)

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY environment variable is not set")
            return
        chunker = ContentChunker(openai_api_key=openai_api_key)

        crawled_items = []
        for result in crawler.crawl():
            try:
                crawled_url, html_content = result
                logger.info(f"Scanned link: {crawled_url}")

                plain_text_chunks = chunker.process_html(html_content)

                for chunk in plain_text_chunks:
                    crawled_items.append({
                        'url': crawled_url,
                        'content': chunk.chunk,
                    })
            except Exception as e:
                logger.error(f"Error processing crawled item: {e}")
                continue

        results = len(crawled_items)

        if results == 0:
            logger.warning("No documents to process. Exiting.")
            return

        try:
            ids = [str(i) for i in range(results)]
            documents = [item['content'] for item in crawled_items]
            metadatas = [{"url": item['url']} for item in crawled_items]
            
            embeddings = model.encode(documents)

            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings.tolist(),
                metadatas=metadatas
            )
        except Exception as e:
            logger.error(f"Error adding documents to collection: {e}")
            return

        if debug:
            logger.debug(f"Crawled, chunked, calculated embeddings, and stored {results} chunks in collection '{collection_name}'")

        try:
            annotate_docs(collection_name)
        except Exception as e:
            logger.error(f"Error annotating docs: {e}")

    except Exception as e:
        logger.error(f"An unexpected error occurred during crawling: {e}", exc_info=True)

if __name__ == '__main__':
    # Your main execution code here
    pass
