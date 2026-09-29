"""Offline embedder: hashed character n-grams weighted by TF-IDF.

This is not a neural embedding model. It is a deterministic, dependency-light stand-in that lets
the full pipeline (and CI) run without network access or API keys. Character n-grams complement the
word-level BM25 retriever: they tolerate morphology and typos ("calibrate" vs "calibration",
"O-ring" vs "oring"), which is where BM25 is weakest.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

from rag_assistant.embeddings.base import l2_normalize


class HashingTfidfEmbedder:
    name = "local-hashing-tfidf"

    def __init__(self, n_features: int = 2**14, ngram_range: tuple[int, int] = (3, 5)) -> None:
        self.dim = n_features
        self._vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            n_features=n_features,
            alternate_sign=False,
            norm=None,
            lowercase=True,
        )
        self._tfidf = TfidfTransformer(sublinear_tf=True)
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        if corpus:
            self._tfidf.fit(self._vectorizer.transform(corpus))
            self._fitted = True

    def _encode(self, texts: list[str]) -> np.ndarray:
        counts = self._vectorizer.transform(texts)
        if self._fitted:
            counts = self._tfidf.transform(counts)
        return l2_normalize(counts.toarray())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text])[0]
