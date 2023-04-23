import os
import requests
import textract
from bs4 import BeautifulSoup

from typing import Optional

TMP_FILE_DIR = '/tmp/erica_scrape_file.pdf'

def text_from_pdf(filepath: str) -> Optional[str]:
    def clean_output(input_str: str) -> str:
        lines = input_str.split('\n')
        new_lines = [l for l in lines if l != '']
        return '\n'.join(new_lines)
    try:
        text = textract.process(filepath)
        text = clean_output(text.decode().strip())
        return text
    except:
        return None

def download_file(url: str) -> bool:
    try:
        r = requests.get(url, stream=True)
        with open(TMP_FILE_DIR, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return True
    except:
        return False

def get_text_from_url(url) -> Optional[str]:
    # if it's a pdf, grab the file and extract the text
    if url.endswith('.pdf'):
        status = download_file(url)
        if not status:
            return None
        text = text_from_pdf(TMP_FILE_DIR)
        if not text:
            return None
        os.remove(TMP_FILE_DIR)
        print(type(text))
        return text
    else:
        # if not, scrape the text from the html page
        if not url.startswith("https://"):
            url = f"https://{url}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        text = ' '.join(text.split())

    return text
