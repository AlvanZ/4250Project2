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

    query_vector = defaultdict(int)

    relevant_documents = defaultdict(lambda: defaultdict(int))

    for term in query_terms:
        # set the value in the query vector to 1, indicating that the term appears
        query_vector[term] = 1

        # grab all documents where at least one word in the query appears at least once
        for document in inverted_index.get(term, {}):
            relevant_documents[document][term] = 1

    scores = {}

    for document in relevant_documents:
        scores[document] = cosine_similarity(query_vector, relevant_documents[document])
    
    sorted_documents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    #print("sorted_documents:", sorted_documents)

    print("Relevant results and scores from vector space model (boolean weights) are:\n")

    for document, score in sorted_documents:
        print(document + " | " + "Score: " + str(score))

