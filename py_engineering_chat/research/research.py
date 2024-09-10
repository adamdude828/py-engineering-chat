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

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from urllib.parse import urlparse
from scrapy.signalmanager import dispatcher
from scrapy import signals

def crawl():
    data = request.json
    url = data.get('url')
    depth = data.get('depth')
    collection_name = data.get('collection_name')
    
    if not url or not depth or not collection_name:
        return jsonify({"error": "Missing parameters"}), 400

    crawl_and_store(url, depth, collection_name)
    return jsonify({"message": "Crawling started"}), 200

class WebsiteSpider(CrawlSpider):
    name = 'website_spider'

    custom_settings = {
        'SPIDER_MIDDLEWARES': {
            'scrapy.spidermiddlewares.offsite.OffsiteMiddleware': None,
        }
    }
    
    def __init__(self, start_url, max_depth, debug=False, *args, **kwargs):
        self.start_urls = [start_url]
        parsed_url = urlparse(start_url)
        self.allowed_domains = [parsed_url.hostname]
        self.port = parsed_url.port if parsed_url.port else 80
        self.max_depth = max_depth
        self.debug = debug  # Store debug as an instance variable
        
        WebsiteSpider.rules = (
            Rule(LinkExtractor(), callback='parse_item', follow=True, cb_kwargs={'depth': 0}),
        )
        super(WebsiteSpider, self).__init__(*args, **kwargs)

    def parse_item(self, response, depth):
        if not response.body:
            self.logger.warning(f"Empty response from {response.url}")
            return

        if depth == 0:  # Only yield on the first depth
            if self.debug:  # Use self.debug instead of debug
                print(f"Crawling: {response.url}")  # Debug output
            yield {
                'url': response.url,
                'content': ' '.join(response.css('*::text').getall())
            }
            return  # Prevent further processing for depth 0

        if depth < self.max_depth:
            for link in response.css('a::attr(href)').getall():
                if self.debug:  # Use self.debug instead of debug
                    print(f"Following link: {link}")  # Debug output
                yield response.follow(link, self.parse_item, cb_kwargs={'depth': depth + 1})

def crawl_and_store(url, depth, collection_name, keep_full_content=False, debug=False):
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

    # Initialize a list to store the crawled items
    crawled_items = []

    # Define a callback function to collect items
    def item_scraped(item, response, spider):
        summary = summarizer.summarize(item['content'])
        if keep_full_content:
            crawled_items.append({
                'url': item['url'],
                'content': item['content'],
                'summary': summary
            })
        else:
            crawled_items.append({
                'url': item['url'],
                'summary': summary
            })

    # Connect the callback to the item_scraped signal
    dispatcher.connect(item_scraped, signal=signals.item_scraped)

    # Set up the crawler
    process = CrawlerProcess(settings={
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'ROBOTSTXT_OBEY': False,  # Disable obeying robots.txt
        'CONCURRENT_REQUESTS': 32,
        'DOWNLOAD_DELAY': 3,
    })

    # Run the spider
    process.crawl(WebsiteSpider, start_url=url, max_depth=depth, keep_full_content=keep_full_content)
    process.start()

    # Use the collected items instead of accessing spider.items
    results = len(crawled_items)

    # Modify the data preparation for Chroma
    ids = [str(i) for i in range(results)]
    documents = [item['summary'] for item in crawled_items]

    if not documents:
        print("No documents to process. Exiting.")
        return

    if keep_full_content:
        metadatas = [{"url": item['url'], "full_content": item['content']} for item in crawled_items]
    else:
        metadatas = [{"url": item['url']} for item in crawled_items]
    
    # Calculate embeddings
    embeddings = model.encode(documents)

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    content_status = "full content and summaries" if keep_full_content else "summaries only"
    if debug:
        print(f"Crawled, summarized, calculated embeddings, and stored {results} pages ({content_status}) in collection '{collection_name}'")

if __name__ == '__main__':
    app.run(port=3000)  # Run Flask server on port 3000