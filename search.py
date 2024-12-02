from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np

client = MongoClient("mongodb://localhost:27017/")
db = client["Indexing"]

documents = [
    "After the medication, headache and nausea were reported by the patient.",
    "The patient reported nausea and dizziness caused by the medication.",
    "Headache and dizziness are common effects of this medication.",
    "The medication caused a headache and nausea, but no dizziness was reported."
]

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

processed_docs = [preprocess(doc) for doc in documents]

def generate_engrams(text, n):
    tokens = text.split()
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(gram) for gram in ngrams]

vocabulary = {}
terms_data = []
doc_data = []
vectorizer = TfidfVectorizer(ngram_range=(1,3))

tfidf_matrix = vectorizer.fit_transform(processed_docs)
feature_names = vectorizer.get_feature_names_out()

for idx, term in enumerate(feature_names):
    vocabulary[term] = idx
    doc_ids = np.nonzero(tfidf_matrix[:, idx])[0]
    docs = [
        {"doc_id": int(doc_id), "tfidf": tfidf_matrix[doc_id, idx]}
        for doc_id in doc_ids
    ]
    terms_data.append({
        "_id": idx,
        "term": term,
        "pos": idx,
        "docs": docs
    })

db.terms.insert_many(terms_data)

for idx, doc in enumerate(docs):
    doc_data.append({"_id": idx, "content": doc})

db.documents.insert_many(doc_data)

def rank_documents(query):
    query = preprocess(query)
    query_vector = vectorizer.transform([query]).toarray().flatten()
    matching_terms = [term for term in feature_names if term in query]

    scores = {}
    for term in matching_terms:
        term_info = db.terms.find_one({"term": term})
        if term_info is None:
            print(f"Term '{term}' not found in the database.")
            continue

        for doc in term_info["docs"]:
            doc_id = doc["doc_id"]
            tfidf = doc["tfidf"]
            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += query_vector[term_info["pos"]] * tfidf

    for doc_id in scores:
        doc_vector = tfidf_matrix[doc_id].toarray().flatten()
        scores[doc_id] /= (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))

    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(db.documents.find_one({"_id": doc_id})["content"], score) for doc_id, score in ranked_docs]

queries = [
    "nausea and dizziness",
    "effects",
    "nausea was reported",
    "dizziness",
    "the medication"
]

for i, query in enumerate(queries, 1):
    print(f"Query q{i}: {query}")
    results = rank_documents(query)
    for content, score in results:
        print(f"Document: \"{content}\", Score: {score}")
    print()