from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
class SimpleRAG:
    def __init__(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        self.chunks = self.chunk_text(text)

        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(self.chunks)
    def chunk_text(self, text, size=300):
        words = text.split()
        return [
            " ".join(words[i:i+size])
            for i in range(0, len(words), size)
        ]

    def retrieve(self, query, k=3):
        query_vec = self.vectorizer.transform([query])

        sims = cosine_similarity(query_vec, self.doc_vectors)[0]

        idx = np.argsort(sims)[-k:][::-1]

        return [self.chunks[i] for i in idx]
