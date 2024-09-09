import pytest
from scrapy.http import TextResponse
from py_engineering_chat.research.research import WebsiteSpider  # Adjust the import based on your project structure

@pytest.fixture
def spider():
    return WebsiteSpider('http://example.com', max_depth=2)

def test_parse_item(spider):
    # Create a mock response
    url = 'http://example.com'
    body = '<html><body><a href="http://example.com/page">Link</a><p>Content here</p></body></html>'
    headers = {'Content-Type': 'text/html; charset=utf-8'}  # Set the content type
    response = TextResponse(url=url, body=body.encode('utf-8'), headers=headers)

    # Call the parse_item method
    results = list(spider.parse_item(response, depth=0))

    # Check the results
    assert len(results) == 1
    assert results[0]['url'] == url
    assert 'Content here' in results[0]['content']

def test_allowed_domains(spider):
    assert 'example.com' in spider.allowed_domains