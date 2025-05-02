import os
import requests
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_confluence_page(url):
    session = requests.Session()
    cookies = {
        'JSESSIONID': os.getenv('JSESSIONID'),
        'atl-sticky-version': os.getenv('ATL_STICKY_VERSION'),
        'atl.xsrf.token': os.getenv('ATL_XSRF_TOKEN'),
        'atlassian.xsrf.token': os.getenv('ATLASSIAN_XSRF_TOKEN'),
        'tenant.session.token': os.getenv('TENANT_SESSION_TOKEN'),
        'ajs_anonymous_id': os.getenv('AJS_ANONYMOUS_ID')
    }
    response = session.get(url, cookies=cookies, verify=False)
    html_content = response.text
    # Print the response content
    soup = BeautifulSoup(html_content, 'html.parser')
    title = soup.find('title')
    main_content = soup.find('div', {'id': 'main-content'})
    text = f"{title}\n{main_content.prettify()}"
    return text

def save_confluence_pages(urls, output_dir='./input/html'):
    for url in urls:
        html_content = get_confluence_page(url)
        soup = BeautifulSoup(html_content, 'html.parser')
        title = soup.find('title').get_text()
        file_name = f"{title}.html"
        file_path = f"{output_dir}/{file_name}"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    return
