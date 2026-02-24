import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from web_search import search_web

vectorizer = TfidfVectorizer()

def similarity(text1, text2):
    # Convert lists to strings
    if isinstance(text1, list):
        text1 = " ".join(text1)

    if isinstance(text2, list):
        text2 = " ".join(text2)

    if not text1 or not text2:
        return 0.0

    tfidf = vectorizer.fit_transform([text1, text2])
    sim_matrix = (tfidf * tfidf.T)

    return sim_matrix.toarray()[0, 1]
def extract_features(answer, context, query):
    length = len(answer.split())

    uncertainty_words = ["maybe", "probably", "possibly", "might", "could"]
    uncertainty = sum(word in answer.lower() for word in uncertainty_words)

    grounding = similarity(answer, context)

    # Web agreement score
    web_text = search_web(query)
    web_similarity = similarity(answer, web_text)

    return np.array(
        [length, uncertainty, grounding, web_similarity],
        dtype=np.float32
    )