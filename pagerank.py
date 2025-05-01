import os
import csv
import numpy as np
import pandas as pd
from collections import defaultdict
import json
from urllib.parse import urljoin
from bs4 import BeautifulSoup

class PageRank:
    def __init__(self):
        self.graph = defaultdict(list)  # URL -> list of outbound links
        self.pages = set()  # Set of all pages
        self.pagerank_scores = {}  # URL -> PageRank score
        
    def load_data_from_csv(self, csv_path):
        """Load outlink data from a CSV file."""
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return
            
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} rows from {csv_path}")
            
            # Add all URLs to the pages set
            for url in df['URL']:
                self.pages.add(url)
                
            # Build a graph representation
            # For now, we'll use a simplified model where each page links to
            # a random subset of other pages based on its outlink count
            all_urls = list(self.pages)
            import random
            
            for _, row in df.iterrows():
                url = row['URL']
                outlink_count = row['Outlinks']
                
                if outlink_count > 0:
                    # Create a set of potential targets (all pages except self)
                    potential_targets = set(all_urls) - {url}
                    if potential_targets:
                        # Select random targets based on outlink count
                        # (This is a simplification - you'd want actual links)
                        targets = random.sample(
                            list(potential_targets), 
                            min(outlink_count, len(potential_targets))
                        )
                        self.graph[url] = targets
        
        except Exception as e:
            print(f"Error loading CSV {csv_path}: {e}")
    
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
    
    def build_graph_from_html_files(self, base_dir):
        """Build graph by scanning HTML files."""
        print(f"Building graph from HTML files in {base_dir}")
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    # We need to determine the original URL from the file path
                    # This is a simplification - adjust based on your crawler's naming scheme
                    url = f"file://{file_path}"
                    self.pages.add(url)
                    
                    # Extract links
                    outlinks = self.extract_links_from_html(file_path, url)
                    if outlinks:
                        self.graph[url] = outlinks
                        # Add outlinks to pages set
                        for link in outlinks:
                            self.pages.add(link)
    
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

# Example usage
if __name__ == "__main__":
    # Create PageRank instance
    pr = PageRank()
    
    # Method 1: Build graph from CSV file
    report_csv = "c:\\Users\\micha\\Code_windows\\cpp\\classes\\spring-2025\\CS4250\\assignment2\\4250Project2\\cpp.edu\\report.csv"
    pr.load_data_from_csv(report_csv)
    
    # Method 2: Build graph from HTML files (uncomment if you want to use this approach)
    # html_dir = "c:\\Users\\micha\\Code_windows\\cpp\\classes\\spring-2025\\CS4250\\assignment2\\4250Project2\\cpp.edu"
    # pr.build_graph_from_html_files(html_dir)
    
    # Calculate PageRank
    pr.calculate_pagerank()
    
    # Get top pages
    top_pages = pr.get_top_pages(100)
    
    print("\nTop 10 pages by PageRank:")
    for url, score in top_pages[:10]:
        print(f"{score:.6f}: {url}")
    
    # Save results
    pr.save_pagerank_scores()