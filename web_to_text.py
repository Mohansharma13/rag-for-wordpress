import requests, re
# import pdfplumber  # Uncomment if you plan to work with PDF files
import logging
import streamlit as st
from typing import List, Any
from scrap_data import Scraper

# Configure logging for the application, specifying log level, format, and date format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Initialize a logger for this module to log errors and information messages
logger = logging.getLogger(__name__)

@st.cache_data  # Cache the extracted data to improve performance on repeated runs
def extract_all_pages(url) -> List[Any]:
    """
    Extract all pages from a given URL by invoking the Scraper class from the 'scrap_data' module.
    
    Args:
        url (str): The URL to scrape data from.
        
    Returns:
        List[Any]: A list containing extracted data from the pages.
    """
    # Initialize the scraper and call post_scraper method to get data from the URL
    data = Scraper(url).post_scraper()
    return data

def is_valid_url(url: str) -> bool:
    """
    Validate the provided URL format and ensure it is accessible.

    Args:
        url (str): The URL to validate.
        
    Returns:
        bool: True if the URL is valid and accessible; False otherwise.
    """
    # Basic URL pattern check using a regular expression
    url_pattern = re.compile(
        r'^(https?://)?'       # Matches http:// or https:// (optional)
        r'([a-zA-Z0-9-]+\.)+'  # Matches the domain part (e.g., example.com)
        r'[a-zA-Z]{2,}'        # Matches the top-level domain (e.g., .com, .org)
        r'(/.*)?$'             # Matches any additional path (optional)
    )
    
    # Check if the URL matches the defined pattern
    if not url_pattern.match(url):
        return False

    # Try sending a request to the URL to confirm it's accessible
    try:
        # Send a GET request to the URL with a timeout of 5 seconds
        response = requests.get(url, timeout=5)
        
        # Return True if the response status code is 200 (OK)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        # If there's any request error (e.g., DNS failure, connection error), URL is considered invalid
        return False
