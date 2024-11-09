import requests
import re
from bs4 import BeautifulSoup

class Scraper:
    """
    A class to scrape and clean content from WordPress posts.

    Attributes:
        url (str): The URL to scrape data from.
        data_list (list): List to store raw data retrieved from the URL.
    """
    
    def __init__(self, url):
        self.url = url  # URL to retrieve data from
        self.data_list = []  # Initialize an empty list to store retrieved data

    def data_retriever(self):
        """
        Retrieves and parses JSON data from the specified URL.
        
        Extracts the 'title' and 'content' fields from each post in the JSON response.
        
        Returns:
            list: A list of strings, where each string is a concatenated title and content of a post.
        """
        data = ""
        try:
            response = requests.get(self.url)  # Send GET request to URL
            response.raise_for_status()  # Raise an HTTPError for 4XX/5XX responses
            posts = response.json()  # Parse JSON content from the response
            
            for post in posts:
                # Extract 'title' and 'content' fields if they exist
                title = post.get('title', {}).get('rendered', '')
                content_html = post.get('content', {}).get('rendered', '')
                
                # Concatenate title and content into a single string
                data = f"{title}\n{content_html}\n"
                self.data_list.append(data)  # Append to the data list
                
        except requests.exceptions.RequestException as e:
            print(f"Failed to retrieve data: {e}")  # Handle network or HTTP errors
            return None
        except ValueError:
            print("Invalid JSON received.")  # Handle JSON decoding errors
            return None
            
        return self.data_list  # Return the list of post data

    def clean_content(self, content_list):
        """
        Cleans HTML and shortcode tags from a list of content strings.
        
        Args:
            content_list (list): List of raw content strings to clean.
        
        Returns:
            list: A list of cleaned strings with only plain text.
        """
        cleaned_content_list = []
        for content in content_list:
            # Step 1: Remove WordPress shortcodes like [et_pb_section], etc.
            content_no_shortcodes = re.sub(r'\[/?[a-zA-Z0-9_]+[^\]]*\]', '', content)
            
            # Step 2: Strip HTML tags using BeautifulSoup
            soup = BeautifulSoup(content_no_shortcodes, 'html.parser')
            text_only = soup.get_text()
            
            # Step 3: Remove excess whitespace and add cleaned content to list
            cleaned_content_list.append(re.sub(r'\s+', ' ', text_only).strip() + "\n")
        
        return cleaned_content_list  # Return the cleaned list of content

    def post_scraper(self):
        """
        Retrieves and cleans content from posts.
        
        Returns:
            list or None: Cleaned content if data is retrieved; otherwise, None.
        """
        content = self.data_retriever()  # Retrieve raw content
        if content is not None:
            return self.clean_content(content)  # Clean and return content
        else:
            print("No content to clean.")
            return None
