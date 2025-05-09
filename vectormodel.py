import json
from collections import defaultdict
import math

inverted_index = defaultdict(lambda: defaultdict(int))
# number of documents in the collection
N = 1500

# Function to load inverted index from file
def load_inverted_index_from_file(filename='inverted_index.json'):
    global inverted_index
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            inverted_index = json.load(f)
        print(f"Inverted index loaded from {filename}")
    except FileNotFoundError:
        print(f"Error: {filename} not found. Please crawl first.")

# returns the cosine similarity between two vectors
def cosine_similarity(query_vector, document_vector):
    numerator = 0

    for term in query_vector:
        numerator += query_vector.get(term, 0) * document_vector.get(term, 0)
    
    query_length = math.sqrt(sum(val ** 2 for val in query_vector.values()))

    document_length = math.sqrt(sum(val ** 2 for val in document_vector.values()))

    if query_length == 0 or document_length == 0:
        return 0

    return numerator / (query_length * document_length)

load_inverted_index_from_file()

# Compute document frequencies (df) and inverse‐df (idf)
df  = { term: len(postings) 
        for term, postings in inverted_index.items() }
idf = { term: math.log(N / df_t, 10) 
        for term, df_t in df.items() }

while True:
    query = input("Please enter your query (ctrl + C to quit): ")
    print("")
    query_terms = query.split()

    tf_query = defaultdict(int)

    # get a count for how many times each term appears in the query, aka the term frequency
    for term in query_terms:
        tf_query[term] += 1
    
    tf_idf_query = defaultdict(int)

    # using the formula (1 + log(tf)) * idf, compute the query vector
    for term in tf_query:
        tf_idf_query[term] = (1 + math.log(tf_query[term])) * idf.get(term, 0)
    

    query_vector = tf_idf_query

    relevant_documents = defaultdict(lambda: defaultdict(int))

    for term in query_vector:
        # calculate the tf_idf values for all the terms within the documents where at least 1 query term appears at least once
        for document in inverted_index.get(term, {}):
            relevant_documents[document][term] = (1 + math.log(inverted_index.get(term, {}).get(document, 0))) * idf.get(term, 0)

    scores = {}

    for document in relevant_documents:
        scores[document] = cosine_similarity(query_vector, relevant_documents[document])
    
    sorted_documents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    #print("sorted_documents:", sorted_documents)

    print("Relevant results and scores from vector space model (TF.IDF weights) are:\n")

    for document, score in sorted_documents:
        print(document + " | " + "Score: " + str(score))
        
    # Save TF-IDF scores to JSON for Part 5
    with open("tfidf_results.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

