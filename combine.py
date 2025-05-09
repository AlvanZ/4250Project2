import json

# get json file from vectormodel and pagerank.py
def load_tfidf_scores(filename='tfidf_results.json'):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {filename} not found. Run vectormodel.py first.")
        return {}

def load_pagerank_scores(filename='pagerank_scores.json'):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {filename} not found. Run pagerank.py first.")
        return {}

#combine TF-IDF score and PageRank score
def combine_scores_multiplicative(tfidf_scores, pagerank_scores):
    combined = []
    for url, tfidf_score in tfidf_scores.items():
        pagerank_score = pagerank_scores.get(url, 0)
        combined_score = tfidf_score * pagerank_score  
        combined.append((url, combined_score))
    return sorted(combined, key=lambda x: x[1], reverse=True)

def main():
    tfidf_scores = load_tfidf_scores()
    pagerank_scores = load_pagerank_scores()

    if not tfidf_scores or not pagerank_scores:
        return

    combined_ranked = combine_scores_multiplicative(tfidf_scores, pagerank_scores)
# saved to json file
    with open('combined_results.json', 'w', encoding='utf-8') as f:
        json.dump(combined_ranked, f, indent=2)
#print out the list of top 10
    print("\n=== Top 10 results based on combined score (TF-IDF * PageRank) ===")
    for i, (url, score) in enumerate(combined_ranked[:10], 1):
        print(f"{i}. {url} (Score: {score:.5f})")

if __name__ == "__main__":
    main()
