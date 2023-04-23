import requests
from bs4 import BeautifulSoup

def get_text_from_url(url):
    # Send an HTTP request to the URL
    response = requests.get(url)

    # Use BeautifulSoup to parse the HTML content of the page
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract all text from the parsed HTML content
    text = soup.get_text()

    # Remove any remaining HTML tags from the text
    text = ' '.join(text.split())

    return text
