"""
retrieval.py
============
Grounding layer. Retrieves the most relevant IT FAQ article for a user message
using TF-IDF vectors and cosine similarity (scikit-learn).

Grounding matters more than fluency here: the language model is only permitted
to speak from a retrieved article, which is what keeps a small instruct model
from inventing IT procedures.
"""

import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_FAQ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "it_faq.json")
LOW_CONFIDENCE = 0.15      # below this the system must not answer from the KB


class Retriever:
    def __init__(self, faq_path=DEFAULT_FAQ):
        with open(faq_path, "r", encoding="utf-8") as fh:
            self.articles = json.load(fh)
        # keywords are repeated so they carry more weight than prose
        corpus = [
            f"{a['title']} {a['title']} {' '.join(a['keywords'])} {' '.join(a['keywords'])} {a['body']}"
            for a in self.articles
        ]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query, top_k=3):
        """Return [(article, score)] ranked by cosine similarity."""
        if not query or not query.strip():
            return []
        vec = self.vectorizer.transform([query])
        sims = cosine_similarity(vec, self.matrix).flatten()
        order = sims.argsort()[::-1][:top_k]
        return [(self.articles[i], float(sims[i])) for i in order]

    def best(self, query):
        """Return (article, score, is_confident)."""
        hits = self.search(query, top_k=1)
        if not hits:
            return None, 0.0, False
        article, score = hits[0]
        return article, score, score >= LOW_CONFIDENCE
