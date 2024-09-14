import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, urldefrag
import urllib.robotparser
import os

class WebCrawler:
    def __init__(self, starting_url, max_depth=2, max_urls=100):
        self.starting_url = starting_url
        self.max_depth = max_depth
        self.max_urls = max_urls
        self.visited_urls = set()
        self.num_urls_crawled = 0

        # Parse the domain from the starting URL
        self.domain = urlparse(starting_url).netloc

        # Initialize the robot parser and read the robots.txt file
        self.robotparser = urllib.robotparser.RobotFileParser()
        robots_url = urljoin(self.starting_url, '/robots.txt')
        self.robotparser.set_url(robots_url)
        self.robotparser.read()

    def crawl(self):
        yield from self._crawl_recursive(self.starting_url, 0)

    def _crawl_recursive(self, url, depth):
        # Check if the maximum depth or URL limit has been reached
        if depth > self.max_depth or self.num_urls_crawled >= self.max_urls:
            return

        # Remove fragment identifiers from the URL
        url, _ = urldefrag(url)

        # Skip if the URL has already been visited
        if url in self.visited_urls:
            return

        # Ensure the URL is within the same domain
        parsed_url = urlparse(url)
        if parsed_url.netloc != self.domain:
            return

        # Check robots.txt permissions
        if not self.robotparser.can_fetch('*', url):
            print(f"Blocked by robots.txt: {url}")
            return

        try:
            # Fetch the URL content
            headers = {'User-Agent': 'WebCrawlerBot'}
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                return

            # Mark the URL as visited
            self.visited_urls.add(url)
            self.num_urls_crawled += 1
            print(f"Crawled URL ({self.num_urls_crawled}): {url}")

            # Yield the raw HTML content instead
            yield (url, response.text)  # Yield a tuple with the URL and raw HTML content

            # Parse the page content for links
            soup = BeautifulSoup(response.content, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                next_url = urljoin(url, href)
                yield from self._crawl_recursive(next_url, depth + 1)

        except requests.RequestException as e:
            print(f"Request failed for {url}: {e}")

    def get_visited_urls(self):
        return self.visited_urls
