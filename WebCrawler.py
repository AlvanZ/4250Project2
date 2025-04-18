import os
import requests
from urllib.parse import urlparse, urljoin, quote
from bs4 import BeautifulSoup
from collections import deque
import re
from lingua import Language, LanguageDetectorBuilder
import tldextract
import csv
from collections import defaultdict

# 1. Make method to get the seed url
# 2. GO to page, error handle if it doesn't exist. Then you save that page
# 3. Check if page is in set, if not then add it and then go through all anchor tags that contain the seed url
# 4. Recurse to 2
# Needs set to prevent duplicates, and then language detection.

#Additions add inverted Index
inverted_index = defaultdict(set)

#Tokenizer for the text for inverted index
def tokenize(text):
    text = text.lower()
    return re.findall(r'\b\w+\b', text)


#Count the  number of files in the directory
def count_txt_files(directory):
    if(os.path.exists(directory)):
        return sum(1 for file in os.listdir(directory) if file.endswith(".txt"))
    return 0

detector = LanguageDetectorBuilder.from_all_languages().build()
def detect_language(text):
    soup = BeautifulSoup(text, 'html.parser')
    stripped_text = soup.get_text(separator=' ', strip = True)
    detected_language = detector.detect_language_of(stripped_text)
    if detected_language is None:
        return "unknown"
    
    return str(detected_language.iso_code_639_1.name)

    
def write_to_csv(url, outlink_count, domain, filename='report.csv'):
    """Appends URL and number of outlinks to a CSV file in the domain's directory."""
    try:
        # Ensure the domain folder exists
        domain_folder = os.path.join(domain)
        if not os.path.exists(domain_folder):
            os.makedirs(domain_folder)  # Create the folder if it doesn't exist
        
        # Define the file path with the desired filename in the domain folder
        file_path = os.path.join(domain_folder, filename)
        
        # Check if the file exists to append or create
        file_exists = os.path.isfile(file_path)
        
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["URL", "Outlinks"])  # Write header if file is new
            writer.writerow([url, outlink_count])  # Append data row
    except Exception as e:
        print(f"Error writing to CSV: {e}")
 
# Function to normalize and filter URLs
def normalize_url(base_url, link, allowed_domains, excluded_domains):
    absolute_url = urljoin(base_url, link)
    parsed_url = urlparse(absolute_url)

    # Encode non-ASCII characters in the URL
    absolute_url = quote(absolute_url, safe=":/?#[]@!$&'()*+,;=")

    extracted = tldextract.extract(parsed_url.netloc)
    root_domain = f"{extracted.domain}.{extracted.suffix}"

    if allowed_domains and not any(root_domain.endswith(domain) for domain in allowed_domains):
        return None
    if excluded_domains and any(root_domain.endswith(domain) for domain in excluded_domains):
        return None
    return absolute_url

# Function to extract links from a page
def extract_links(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        links = set()
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            if href.startswith("javascript") or href.startswith("#"):  # Ignore JavaScript and fragments
                continue
            links.add(href)
        
        return links
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return set()

# Function to save the HTML content of a page to a file
def save_page(url, content, base_dir):
    parsed_url = urlparse(url)
    # Sanitize URL to create a valid filename
    filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', parsed_url.path.strip('/')) or 'index'

    file_path = os.path.join(base_dir, f"{filename}.txt")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Save content to a file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    print(f"Saved: {file_path}")

# Web Crawler
def crawl(seed_urls, allowed_domains=None, excluded_domains=None, file_limit=500):
    visited = set()  # Keep track of visited URLs
    to_visit = deque()  # Queue to store URLs to visit
    
    # Initialize seed URLs
    for seed_url in seed_urls:
        extracted = tldextract.extract(urlparse(seed_url).netloc)
        domain = f"{extracted.domain}.{extracted.suffix}"
        
        # Check allowed and excluded domains
        if allowed_domains and domain not in allowed_domains:
            print(f"Skipping seed URL, not allowed: {seed_url}")
            continue
        if excluded_domains and domain in excluded_domains:
            print(f"Skipping seed URL (excluded): {seed_url}")
            continue
        
        to_visit.append(seed_url)  # Add seed URL to queue for crawling

    while to_visit:
        current_url = to_visit.popleft()
        
        # Skip if the URL has already been visited
        if current_url in visited:
            continue
        
        # Mark the URL as visited
        visited.add(current_url)

        print(f"Crawling: {current_url}".encode("utf-8", "ignore").decode("utf-8"))
        
        try:
            # Fetch the page content
            response = requests.get(current_url)
            if response.status_code == 200:
                extracted = tldextract.extract(current_url)
                domain = f"{extracted.domain}.{extracted.suffix}"
                language = detect_language(response.text)

                # Check file count before saving the page
                file_count = count_txt_files(f"{domain}/{language}")
                print("File count: ", file_count)
                
                if file_count >= file_limit:
                    print(f"Hit file limit for {domain}/{language}")
                    continue
                
                # Save the page content to file
                base_dir = os.path.join(domain, language)
                save_page(current_url, response.text, base_dir)
                
                #Analyze saved page for inverted index
                soup = BeautifulSoup(response.text, 'html.parser')
                visible_text = soup.get_text(separator=' ', strip=True)
                tokens = tokenize(visible_text)

                for word in tokens:
                    inverted_index[word].add(current_url)
                # Save inverted index to file
                with open('inverted_index.txt', 'w', encoding='utf-8') as f:
                    for word in sorted(inverted_index):
                        urls = ', '.join(inverted_index[word])
                        f.write(f"{word}: {urls}\n")
                print("Inverted index saved to inverted_index.txt")
                # Extract links from the page
                links = extract_links(current_url)
                outlink_count = len(links)
                write_to_csv(current_url, outlink_count, domain)
                
                # Process and normalize the links
                for link in links:
                    normalized_url = normalize_url(current_url, link, allowed_domains, excluded_domains)
                    
                    # Ensure only new URLs are added to the queue
                    if normalized_url and normalized_url not in visited and normalized_url not in to_visit:
                        # Normalizing link and ensuring we keep relative URLs
                        absolute_url = urljoin(current_url, normalized_url)  # Convert relative to absolute
                        if absolute_url not in visited:
                            to_visit.append(absolute_url)

        except requests.RequestException as e:
            print(f"Failed to fetch {current_url}: {e}")

# Example usage
seed_urls = [ 
    "https://www.taobao.com/",         # Should be saved under `taobao.com/` 
    "https://www.yahoo.co.jp/", # Should be saved under `yahoo.co.jp`
    "https://www.cpp.edu/",  # Should be saved under `cpp.edu`       
]

#doesnt check allowed domain for the frst parse

#use parsed_url.netloc to save to proper directory
allowed_domains = ["taobao.com", 'yahoo.co.jp', 'cpp.edu']  # Only crawl these
excluded_domains = []  # Ignore Taobao

crawl(seed_urls, allowed_domains=allowed_domains, excluded_domains=excluded_domains)
