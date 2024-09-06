import click
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import chromadb
from chromadb.config import Settings
import os
from dotenv import load_dotenv

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from urllib.parse import urlparse
from scrapy.signalmanager import dispatcher
from scrapy import signals

class WebsiteSpider(CrawlSpider):
    name = 'website_spider'

    custom_settings = {
        'SPIDER_MIDDLEWARES': {
            'scrapy.spidermiddlewares.offsite.OffsiteMiddleware': None,
        }
    }
    
    def __init__(self, start_url, max_depth, *args, **kwargs):
        self.start_urls = [start_url]
        self.allowed_domains = [urlparse(start_url).netloc]
        self.max_depth = max_depth
        
        WebsiteSpider.rules = (
            Rule(LinkExtractor(), callback='parse_item', follow=True, cb_kwargs={'depth': 0}),
        )
        super(WebsiteSpider, self).__init__(*args, **kwargs)

    def parse_item(self, response, depth):
        if depth <= self.max_depth:
            print(f"Crawling: {response.url}")
            yield {
                'url': response.url,
                'content': ' '.join(response.css('*::text').getall())
            }
            
            if depth < self.max_depth:
                for link in response.css('a::attr(href)').getall():
                    yield response.follow(link, self.parse_item, cb_kwargs={'depth': depth + 1})

def crawl_and_store(url, depth, collection_name):
    # Print the input parameters
    print(f"URL: {url}")
    print(f"Depth: {depth}")
    print(f"Collection: {collection_name}")

    # Load environment variables
    load_dotenv(dotenv_path='../.env')

    # Initialize Chroma client
    client = chromadb.PersistentClient(path="./.chroma_db")

    # Create or get collection
    collection = client.get_or_create_collection(name=collection_name)

    # Set up the crawler
    process = CrawlerProcess(settings={
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS': 32,
        'DOWNLOAD_DELAY': 3,
    })

    # Run the spider
    process.crawl(WebsiteSpider, start_url=url, max_depth=depth)
    process.start()

    # Get the results from the spider
    spider = process.spider
    results = spider.crawler.stats.get_stats()['item_scraped_count']

    # Insert data into Chroma
    ids = [str(i) for i in range(results)]
    documents = [item['content'] for item in spider.items]
    metadatas = [{"url": item['url']} for item in spider.items]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    print(f"Crawled and stored {results} pages in collection '{collection_name}'")

if __name__ == '__main__':
    crawl_and_store()