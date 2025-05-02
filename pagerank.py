import os
import csv
import numpy as np
import pandas as pd
from collections import defaultdict
import json
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import argparse

class PageRank:
    def __init__(self):
        self.graph = defaultdict(list)  # URL -> list of outbound links
        self.pages = set()  # Set of all pages
        self.pagerank_scores = {}  # URL -> PageRank score
    
    def extract_links_from_html(self, file_path, base_url):
        """Extract links from an HTML file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                soup = BeautifulSoup(content, 'html.parser')
                links = []
                
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href'].strip()
                    if href.startswith("javascript") or href.startswith("#"):
                        continue
                    
                    # Convert relative URLs to absolute
                    absolute_url = urljoin(base_url, href)
                    links.append(absolute_url)
                
                return links
        except Exception as e:
            print(f"Error extracting links from {file_path}: {e}")
            return []
    
    def build_pagerank_graph_multi(self, base_dir, domains=None, csv_filename="report.csv"):
        """
        Build a PageRank graph from multiple domains.
        
        Args:
            base_dir: Base directory containing domain folders
            domains: List of domain folders to include (None means all available)
            csv_filename: Name of the CSV file with outlink data
        """
        # Clear any existing data
        self.graph = defaultdict(list)
        self.pages = set()
        
        # If no domains specified, use all available domain directories
        if domains is None:
            domains = []
            for item in os.listdir(base_dir):
                domain_dir = os.path.join(base_dir, item)
                if os.path.isdir(domain_dir):
                    domains.append(item)
        
        if not domains:
            print("No domains found or specified.")
            return
        
        print(f"Building PageRank graph for domains: {', '.join(domains)}")
        
        # Process each domain
        total_pages = 0
        total_processed_files = 0
        
        for domain in domains:
            domain_dir = os.path.join(base_dir, domain)
            if not os.path.isdir(domain_dir):
                print(f"Domain directory not found: {domain_dir}")
                continue
            
            # Load crawled pages for this domain
            csv_file = os.path.join(domain_dir, csv_filename)
            domain_pages = set()
            
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    for url in df['URL']:
                        domain_pages.add(url)
                        self.pages.add(url)
                    print(f"  {domain}: Loaded {len(domain_pages)} pages from CSV")
                    total_pages += len(domain_pages)
                except Exception as e:
                    print(f"  Error loading CSV for {domain}: {e}")
                    continue
            else:
                print(f"  CSV file not found for {domain}: {csv_file}")
                continue
            
            # Process HTML files for this domain
            processed_files = 0
            for root, dirs, files in os.walk(domain_dir):
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        
                        # Try to determine the original URL
                        file_basename = os.path.basename(file_path).replace('_', '/').replace('.txt', '')
                        potential_matches = [url for url in domain_pages if file_basename in url]
                        
                        if potential_matches:
                            source_url = potential_matches[0]
                            outlinks = self.extract_links_from_html(file_path, source_url)
                            
                            # Filter outlinks to only include other crawled pages from any domain
                            valid_outlinks = [link for link in outlinks if link in self.pages]
                            
                            if valid_outlinks:
                                self.graph[source_url] = valid_outlinks
                            
                            processed_files += 1
            
            total_processed_files += processed_files
            print(f"  {domain}: Processed {processed_files} HTML files")
        
        print(f"Built graph with {len(self.graph)} source pages and {len(self.pages)} total pages")
        print(f"Total processed files: {total_processed_files}")
    
    def calculate_pagerank(self, damping_factor=0.85, iterations=100, tolerance=1e-6):
        """Calculate PageRank scores for all pages."""
        n = len(self.pages)
        if n == 0:
            print("No pages found to calculate PageRank")
            return
        
        print(f"Calculating PageRank for {n} pages...")
        
        # Map URLs to indices for matrix operations
        url_to_index = {url: i for i, url in enumerate(self.pages)}
        index_to_url = {i: url for i, url in enumerate(self.pages)}
        
        # Initialize the transition matrix
        M = np.zeros((n, n))
        
        # Build the transition matrix from our graph
        for url, outlinks in self.graph.items():
            if url in url_to_index:
                i = url_to_index[url]
                # If page has outlinks, distribute probability equally
                if outlinks:
                    valid_outlinks = [link for link in outlinks if link in url_to_index]
                    if valid_outlinks:
                        for target in valid_outlinks:
                            j = url_to_index[target]
                            M[j, i] = 1.0 / len(valid_outlinks)
                    else:
                        # If no valid outlinks, distribute to all pages
                        for j in range(n):
                            M[j, i] = 1.0 / n
                # If page has no outlinks, distribute probability to all pages
                else:
                    for j in range(n):
                        M[j, i] = 1.0 / n
        
        # Handle pages not in the graph (distribute evenly)
        for url in self.pages:
            if url not in self.graph and url in url_to_index:
                i = url_to_index[url]
                for j in range(n):
                    M[j, i] = 1.0 / n
        
        # Initialize PageRank vector
        v = np.ones(n) / n
        
        # Power iteration
        last_iteration = 0
        for iteration in range(iterations):
            last_iteration = iteration
            v_prev = v.copy()
            v = damping_factor * np.dot(M, v) + (1 - damping_factor) / n
            
            # Check for convergence
            if np.linalg.norm(v - v_prev, 1) < tolerance:
                break
        
        # Store the results
        self.pagerank_scores = {index_to_url[i]: float(score) for i, score in enumerate(v)}
        
        print(f"PageRank calculation completed after {last_iteration + 1} iterations")
    
    def get_top_pages(self, n=100):
        """Return the top n pages by PageRank score."""
        sorted_pages = sorted(self.pagerank_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_pages[:n]
    
    def save_pagerank_scores(self, filename='pagerank_scores.json'):
        """Save PageRank scores to a JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.pagerank_scores, f, indent=4)
        print(f"PageRank scores saved to {filename}")
    
    def save_top_pages_list(self, filename='top_pages.txt', n=100):
        """
        Save the top n pages by PageRank score to a text file.
        
        Args:
            filename: Output filename
            n: Number of top pages to include
        """
        top_pages = self.get_top_pages(n)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Top {len(top_pages)} Pages by PageRank\n")
            f.write("="*50 + "\n\n")
            
            for i, (url, score) in enumerate(top_pages):
                f.write(f"{i+1}. {score:.8f}: {url}\n")
        
        print(f"Top {len(top_pages)} pages saved to {filename}")

# Example usage
if __name__ == "__main__":
    import sys
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Calculate PageRank for crawled web pages')
    parser.add_argument('--base_dir', type=str, default='.',
                        help='Base directory containing domain folders (default: current directory)')
    parser.add_argument('--domains', type=str, nargs='*',
                        help='Domain folders to process (default: all available)')
    parser.add_argument('--damping', type=float, default=0.85,
                        help='Damping factor (default: 0.85)')
    parser.add_argument('--top', type=int, default=100,
                        help='Number of top pages to output (default: 100)')
    parser.add_argument('--output', type=str, default='pagerank_scores.json',
                        help='Output JSON file for all PageRank scores (default: pagerank_scores.json)')
    parser.add_argument('--top_list', type=str, default='top_pages.txt',
                        help='Output text file for top pages (default: top_pages.txt)')
    
    args = parser.parse_args()
    
    # Normalize the base directory path
    base_dir = os.path.normpath(args.base_dir)
    
    # Verify the path exists
    if not os.path.exists(base_dir):
        print(f"Error: The specified base directory does not exist: {base_dir}")
        sys.exit(1)
    
    # If no domains specified, use only the known crawl directories
    if args.domains is None:
        # List of known crawled domains - modify as needed
        args.domains = ['cpp.edu', 'yahoo.co.jp', 'taobao.com']
        print(f"No domains specified, using default crawled domains: {', '.join(args.domains)}")
    
    # Create PageRank instance
    pr = PageRank()
    
    # Build graph from specified domains
    pr.build_pagerank_graph_multi(base_dir, args.domains)
    
    # Calculate PageRank
    pr.calculate_pagerank(damping_factor=args.damping)
    
    # Get top pages
    top_pages = pr.get_top_pages(args.top)
    
    print(f"\nTop 10 pages by PageRank:")
    for i, (url, score) in enumerate(top_pages[:10]):
        print(f"{i+1}. {score:.6f}: {url}")
    
    # Save results
    pr.save_pagerank_scores(args.output)
    pr.save_top_pages_list(args.top_list, args.top)