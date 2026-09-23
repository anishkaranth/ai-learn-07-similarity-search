"""Tiny vector store: TF-IDF document embeddings + exact & LSH similarity search.

Educational NumPy implementation:
  - Fixed toy document corpus (themed short passages) in corpus.py
  - TF-IDF bag-of-words vectors, L2-normalized for cosine
  - Exact top-k retrieval by cosine similarity (brute force)
  - Approximate retrieval via random-projection LSH (bit signatures + Hamming)
  - Recall@k of approx vs exact ground truth
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

import numpy as np

from corpus import DOCUMENTS, LABELED_QUERIES

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def build_vocab(docs: Sequence[Dict[str, str]]) -> Tuple[Dict[str, int], List[str]]:
    counts: Dict[str, int] = {}
    for d in docs:
        for tok in set(tokenize(d["text"])):
            counts[tok] = counts.get(tok, 0) + 1
    words = sorted(counts.keys())
    word2id = {w: i for i, w in enumerate(words)}
    return word2id, words


def compute_idf(docs: Sequence[Dict[str, str]], word2id: Dict[str, int]) -> np.ndarray:
    """Smooth IDF: log((N + 1) / (df + 1)) + 1."""
    n = len(docs)
    df = np.zeros(len(word2id), dtype=np.float64)
    for d in docs:
        seen = set(tokenize(d["text"]))
        for tok in seen:
            if tok in word2id:
                df[word2id[tok]] += 1.0
    return np.log((n + 1.0) / (df + 1.0)) + 1.0


def tfidf_matrix(
    docs: Sequence[Dict[str, str]],
    word2id: Dict[str, int],
    idf: np.ndarray,
) -> np.ndarray:
    """Return (N, V) TF-IDF matrix (raw term frequency * idf), not yet normalized."""
    n, v = len(docs), len(word2id)
    X = np.zeros((n, v), dtype=np.float64)
    for i, d in enumerate(docs):
        toks = tokenize(d["text"])
        for tok in toks:
            if tok in word2id:
                X[i, word2id[tok]] += 1.0
        # optional log-tf
        mask = X[i] > 0
        X[i, mask] = 1.0 + np.log(X[i, mask])
        X[i] *= idf
    return X


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def embed_query(text: str, word2id: Dict[str, int], idf: np.ndarray) -> np.ndarray:
    """TF-IDF vector for a free-text query (same schema as docs), L2-normalized."""
    v = np.zeros(len(word2id), dtype=np.float64)
    toks = tokenize(text)
    for tok in toks:
        if tok in word2id:
            v[word2id[tok]] += 1.0
    mask = v > 0
    v[mask] = 1.0 + np.log(v[mask])
    v *= idf
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def cosine_scores(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """Both sides L2-normalized -> cosine = dot product."""
    return doc_matrix @ query_vec


def exact_topk(
    query_vec: np.ndarray,
    doc_matrix: np.ndarray,
    k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) of top-k docs by cosine, descending."""
    scores = cosine_scores(query_vec, doc_matrix)
    k = min(k, len(scores))
    # argpartition then sort the top-k slice
    part = np.argpartition(-scores, kth=k - 1)[:k]
    order = np.argsort(-scores[part])
    idx = part[order]
    return idx, scores[idx]


# ---------------------------------------------------------------------------
# Random-projection LSH (SimHash-style bit signatures)
# ---------------------------------------------------------------------------

class RandomProjectionLSH:
    """Approximate cosine NN via random hyperplanes -> bit codes + Hamming.

    For unit vectors, sign(R @ x) gives a bit string; Hamming distance
    approximates angular / cosine distance (Charikar / SimHash style).
    """

    def __init__(self, dim: int, n_bits: int = 64, seed: int = 42):
        self.dim = dim
        self.n_bits = n_bits
        rng = np.random.default_rng(seed)
        # Random hyperplane normals (n_bits, dim)
        self.R = rng.standard_normal((n_bits, dim))
        self.codes: np.ndarray | None = None  # (N, n_bits) bool

    def _sign_bits(self, X: np.ndarray) -> np.ndarray:
        """X: (N, dim) or (dim,) -> bool bits."""
        single = X.ndim == 1
        if single:
            X = X[None, :]
        proj = X @ self.R.T  # (N, n_bits)
        bits = proj >= 0
        return bits[0] if single else bits

    def fit(self, doc_matrix: np.ndarray) -> "RandomProjectionLSH":
        self.codes = self._sign_bits(doc_matrix)
        return self

    def hamming_distances(self, query_vec: np.ndarray) -> np.ndarray:
        assert self.codes is not None
        qbits = self._sign_bits(query_vec)
        # Hamming = xor sum
        return np.sum(self.codes != qbits, axis=1).astype(np.float64)

    def query(self, query_vec: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Return (indices, hamming_distances) of k nearest by Hamming."""
        dists = self.hamming_distances(query_vec)
        k = min(k, len(dists))
        part = np.argpartition(dists, kth=k - 1)[:k]
        order = np.argsort(dists[part])
        idx = part[order]
        return idx, dists[idx]

    def query_candidates(
        self,
        query_vec: np.ndarray,
        candidate_pool: int,
        doc_matrix: np.ndarray,
        k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """LSH shortlist then re-rank by exact cosine within the pool."""
        n = doc_matrix.shape[0]
        pool = min(candidate_pool, n)
        dists = self.hamming_distances(query_vec)
        part = np.argpartition(dists, kth=pool - 1)[:pool]
        # re-rank shortlist by cosine
        scores = doc_matrix[part] @ query_vec
        k = min(k, pool)
        top_local = np.argsort(-scores)[:k]
        idx = part[top_local]
        return idx, scores[top_local]


def recall_at_k(exact_ids: Sequence[int], approx_ids: Sequence[int], k: int | None = None) -> float:
    """Fraction of exact top-k that appear in approx top-k (same k unless overridden)."""
    if k is None:
        k = len(exact_ids)
    exact_set = set(list(exact_ids)[:k])
    approx_set = set(list(approx_ids)[:k])
    if not exact_set:
        return 0.0
    return len(exact_set & approx_set) / len(exact_set)


def topic_of(doc_idx: int, docs: Sequence[Dict[str, str]] = DOCUMENTS) -> str:
    return docs[doc_idx]["topic"]


def check_query_topic(
    query: str,
    expected_topic: str,
    word2id: Dict[str, int],
    idf: np.ndarray,
    doc_matrix: np.ndarray,
    docs: Sequence[Dict[str, str]] = DOCUMENTS,
    top_n: int = 3,
) -> Dict[str, object]:
    """Exact top-n retrieval; PASS if any of top-n (or top-1 preferred) matches topic."""
    qv = embed_query(query, word2id, idf)
    idx, scores = exact_topk(qv, doc_matrix, k=top_n)
    topics = [topic_of(int(i), docs) for i in idx]
    top1_ok = topics[0] == expected_topic if topics else False
    topn_ok = expected_topic in topics
    hits = [
        {
            "rank": r + 1,
            "doc_id": docs[int(i)]["id"],
            "topic": topics[r],
            "cosine": round(float(scores[r]), 4),
            "text": docs[int(i)]["text"][:80],
        }
        for r, i in enumerate(idx)
    ]
    return {
        "query": query,
        "expected_topic": expected_topic,
        "top1_ok": top1_ok,
        "topn_ok": topn_ok,
        "hits": hits,
    }
