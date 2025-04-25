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
import math
import json
# 1. Make method to get the seed url
# 2. GO to page, error handle if it doesn't exist. Then you save that page
# 3. Check if page is in set, if not then add it and then go through all anchor tags that contain the seed url
# 4. Recurse to 2
# Needs set to prevent duplicates, and then language detection.

# → term → { doc_url: raw_term_count }
inverted_index = defaultdict(lambda: defaultdict(int))
# → doc_url → total_terms_in_doc
doc_lengths     = defaultdict(int)
# total docs seen
N = 0


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


# Function to save inverted index to file
def save_inverted_index_to_file(filename='inverted_index.json'):
    # Check if file exists to do an incremental update
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                existing_index = json.load(f)
            
            # Merge existing index with new data
            for term, docs in inverted_index.items():
                if term in existing_index:
                    # Update existing term with new document counts
                    existing_index[term].update(docs)
                else:
                    # Add new term
                    existing_index[term] = docs
            
            # Write the merged index
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(existing_index, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error updating inverted index: {e}")
            # If there was an error reading the existing file, write a new one
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(inverted_index, f, ensure_ascii=False, indent=4)
    else:
        # If file doesn't exist, create it
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(inverted_index, f, ensure_ascii=False, indent=4)
    
    print(f"Inverted index updated in {filename}")
# Function to load inverted index from file
def load_inverted_index_from_file(filename='inverted_index.json'):
    global inverted_index
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            inverted_index = json.load(f)
        print(f"Inverted index loaded from {filename}")
    except FileNotFoundError:
        print(f"Error: {filename} not found. Please crawl first.")


# Web Crawler
def crawl(seed_urls, allowed_domains=None, excluded_domains=None, file_limit=500):
    global N

    # Prepare your extractor once
    extract = tldextract.TLDExtract(cache_dir="./.suffix_cache")
    page_counter = 0
    save_frequency = 10  # Save every 10 pages
    for seed_url in seed_urls:
        # Start fresh for this seed
        visited = set()
        to_visit = deque([seed_url])

        # Figure out the root‐domain of the seed
        parsed = urlparse(seed_url)
        parts  = extract(parsed.netloc)
        seed_domain = f"{parts.domain}.{parts.suffix}"

        print(f"\n=== Crawling seed: {seed_url} (domain: {seed_domain}) ===")

        while to_visit:
            current_url = to_visit.popleft()
            if current_url in visited:
                continue
            visited.add(current_url)

            try:
                resp = requests.get(current_url, timeout=5)
                resp.raise_for_status()
            except Exception as e:
                print(f"  [error] fetching {current_url}: {e}")
                continue

            # Determine domain & language
            parsed = urlparse(current_url)
            parts  = extract(parsed.netloc)
            domain = f"{parts.domain}.{parts.suffix}"
            language = detect_language(resp.text)

            # Check how many files we've already saved for this domain/language
            cnt = count_txt_files(os.path.join(domain, language))
            print(f"  File count for {domain}/{language}: {cnt}")
            if cnt >= file_limit:
                print(f"  → reached file_limit ({cnt} ≥ {file_limit}) for {domain}/{language}; skipping rest of this seed.")
                break   # abandon this seed and move on

            # Otherwise, save page & index it
            base_dir = os.path.join(domain, language)
            save_page(current_url, resp.text, base_dir)

            # TF counting
            text   = BeautifulSoup(resp.text, "html.parser").get_text(" ", strip=True)
            tokens = tokenize(text)
            N += 1
            doc_lengths[current_url] = len(tokens)
            for w in tokens:
                inverted_index[w][current_url] += 1

            # Save the inverted index after each page is processed
            page_counter += 1
            if page_counter % save_frequency == 0:
                save_inverted_index_to_file()
                

            # record outlink count
            links = extract_links(current_url)
            write_to_csv(current_url, len(links), domain)

            # enqueue normalized outlinks
            for href in links:
                norm = normalize_url(current_url, href, allowed_domains, excluded_domains)
                if norm and norm not in visited:
                    to_visit.append(norm)

        # end while for this seed
        print(f"--- done with seed {seed_url} ---")
    # end for each seed

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

# Compute document frequencies (df) and inverse‐df (idf)
df  = { term: len(postings) 
        for term, postings in inverted_index.items() }
idf = { term: math.log(N / df_t, 10) 
        for term, df_t in df.items() }

# Precedence for Boolean ops
precedence = {'NOT': 3, 'AND': 2, 'OR': 1}

def tokenize_query(q):
    return re.findall(r'\bAND\b|\bOR\b|\bNOT\b|\(|\)|\w+', q.upper())

def parse_boolean(q):
    output, ops = [], []
    for tok in tokenize_query(q):
        if tok not in precedence and tok not in ('(',')'):
            output.append(tok.lower())
        elif tok == '(':
            ops.append(tok)
        elif tok == ')':
            while ops and ops[-1] != '(':
                output.append(ops.pop())
            ops.pop()
        else:
            while ops and ops[-1] != '(' and precedence[ops[-1]] >= precedence[tok]:
                output.append(ops.pop())
            ops.append(tok)
    while ops:
        output.append(ops.pop())
    print("Parsing")
    print(output)
    return output

def eval_and_rank(rpn_tokens):
    # Prepare simple sets for each term
    postings = {
      term: set(inverted_index.get(term, {})) 
      for term in set(t for t in rpn_tokens if t not in precedence)
    }
    stack = []
    # Boolean evaluation
    for tok in rpn_tokens:
        if tok not in precedence:
            stack.append(postings.get(tok, set()))
        elif tok == 'NOT':
            a = stack.pop()
            stack.append(set(doc_lengths) - a)
        else:
            b, a = stack.pop(), stack.pop()
            if tok == 'AND': stack.append(a & b)
            else:            stack.append(a | b)
    result_docs = stack.pop() if stack else set()

    # Rank by sum of (1+log tf)*idf
    scores = {}
    for doc in result_docs:
        total = 0.0
        for term in postings:
            tf = inverted_index.get(term, {}).get(doc, 0)
            if tf > 0:
                total += (1 + math.log(tf, 10)) * idf.get(term, 0.0)
        scores[doc] = total

    # Return sorted list of (url, score)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

def search(query, top_k=10):
    rpn = parse_boolean(query)
    results = eval_and_rank(rpn)
    return results[:top_k]




    # compute df+idf as above...

print("\n=== TOKENS IN INVERTED INDEX ===")
for term in sorted(inverted_index.keys())[:50]:  # show only first 50 for brevity
    print(f"{term}: {len(inverted_index[term])} docs")
print("...")
print(f"Total unique terms (tokens): {len(inverted_index)}")

while True:
    q = input("Enter Boolean query (or 'exit'): ").strip()
    if q.lower() in ('exit','quit'): break
    for url, score in search(q):
        print(f"{score:.4f}\t{url}")




