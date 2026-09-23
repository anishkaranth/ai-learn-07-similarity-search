#!/usr/bin/env python3
"""TF-IDF vector store smoke: exact cosine + LSH approx -> results/."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from similarity_search import (
    DOCUMENTS,
    LABELED_QUERIES,
    RandomProjectionLSH,
    build_vocab,
    check_query_topic,
    compute_idf,
    embed_query,
    exact_topk,
    l2_normalize,
    recall_at_k,
    tfidf_matrix,
)
from smoke_plots import make_plots, write_artifacts

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42
N_BITS = 64
K_EVAL = 5
CANDIDATE_POOL = 8


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    t0 = time.perf_counter()

    word2id, id2word = build_vocab(DOCUMENTS)
    idf = compute_idf(DOCUMENTS, word2id)
    X = l2_normalize(tfidf_matrix(DOCUMENTS, word2id, idf))
    dim = X.shape[1]
    n_docs = X.shape[0]

    query_checks = [
        check_query_topic(
            str(item["query"]),
            str(item["expected_topic"]),
            word2id,
            idf,
            X,
            DOCUMENTS,
            top_n=3,
        )
        for item in LABELED_QUERIES
    ]
    n_pass_top1 = sum(1 for c in query_checks if c["top1_ok"])
    n_pass_top3 = sum(1 for c in query_checks if c["topn_ok"])
    retrieval_pass = n_pass_top3 == len(query_checks) and n_pass_top1 >= max(
        1, len(query_checks) - 1
    )

    lsh = RandomProjectionLSH(dim=dim, n_bits=N_BITS, seed=SEED).fit(X)
    eval_queries = [str(q["query"]) for q in LABELED_QUERIES] + [d["text"] for d in DOCUMENTS]
    recalls_raw, recalls_rerank, random_recalls = [], [], []
    rng = np.random.default_rng(SEED)

    for qtext in eval_queries:
        qv = embed_query(qtext, word2id, idf)
        exact_idx, _ = exact_topk(qv, X, k=K_EVAL)
        approx_idx, _ = lsh.query(qv, k=K_EVAL)
        recalls_raw.append(recall_at_k(exact_idx, approx_idx, k=K_EVAL))
        rr_idx, _ = lsh.query_candidates(qv, CANDIDATE_POOL, X, k=K_EVAL)
        recalls_rerank.append(recall_at_k(exact_idx, rr_idx, k=K_EVAL))
        rand_idx = rng.choice(n_docs, size=K_EVAL, replace=False)
        random_recalls.append(recall_at_k(exact_idx, rand_idx, k=K_EVAL))

    mean_recall_raw = float(np.mean(recalls_raw))
    mean_recall_rerank = float(np.mean(recalls_rerank))
    mean_recall_random = float(np.mean(random_recalls))
    approx_ok = mean_recall_rerank > mean_recall_random + 0.15

    k_max = min(10, n_docs)
    curve = []
    for k in range(1, k_max + 1):
        r_raw, r_rr, r_rand = [], [], []
        for qtext in eval_queries:
            qv = embed_query(qtext, word2id, idf)
            exact_idx, _ = exact_topk(qv, X, k=k)
            approx_idx, _ = lsh.query(qv, k=k)
            rr_idx, _ = lsh.query_candidates(qv, min(CANDIDATE_POOL, n_docs), X, k=k)
            r_raw.append(recall_at_k(exact_idx, approx_idx, k=k))
            r_rr.append(recall_at_k(exact_idx, rr_idx, k=k))
            rand_idx = rng.choice(n_docs, size=k, replace=False)
            r_rand.append(recall_at_k(exact_idx, rand_idx, k=k))
        curve.append(
            {
                "k": k,
                "recall_lsh_hamming": round(float(np.mean(r_raw)), 4),
                "recall_lsh_rerank": round(float(np.mean(r_rr)), 4),
                "recall_random": round(float(np.mean(r_rand)), 4),
            }
        )

    make_plots(RESULTS, word2id, idf, X, curve, N_BITS)
    runtime_s = time.perf_counter() - t0

    metrics = {
        "project": "ai-learn-07-similarity-search",
        "seed": SEED,
        "n_docs": n_docs,
        "vocab_size": len(id2word),
        "dim": dim,
        "n_bits_lsh": N_BITS,
        "candidate_pool": CANDIDATE_POOL,
        "k_eval": K_EVAL,
        "n_eval_queries": len(eval_queries),
        "retrieval_checks": [
            {
                "query": c["query"],
                "expected_topic": c["expected_topic"],
                "top1_ok": c["top1_ok"],
                "top3_ok": c["topn_ok"],
                "top_hits": [
                    {"doc_id": h["doc_id"], "topic": h["topic"], "cosine": h["cosine"]}
                    for h in c["hits"]
                ],
            }
            for c in query_checks
        ],
        "n_queries_top1_pass": n_pass_top1,
        "n_queries_top3_pass": n_pass_top3,
        "n_labeled_queries": len(query_checks),
        "retrieval_sanity": "PASS" if retrieval_pass else "FAIL",
        "mean_recall_at_5_lsh_hamming": round(mean_recall_raw, 4),
        "mean_recall_at_5_lsh_rerank": round(mean_recall_rerank, 4),
        "mean_recall_at_5_random": round(mean_recall_random, 4),
        "approx_beats_random": bool(approx_ok),
        "recall_curve": curve,
        "runtime_s": round(runtime_s, 3),
    }
    shot = {
        "project": metrics["project"],
        "n_docs": n_docs,
        "vocab_size": len(id2word),
        "dim": dim,
        "retrieval_sanity": metrics["retrieval_sanity"],
        "n_queries_top1_pass": n_pass_top1,
        "n_queries_top3_pass": n_pass_top3,
        "n_labeled_queries": len(query_checks),
        "mean_recall_at_5_lsh_rerank": metrics["mean_recall_at_5_lsh_rerank"],
        "mean_recall_at_5_lsh_hamming": metrics["mean_recall_at_5_lsh_hamming"],
        "mean_recall_at_5_random": metrics["mean_recall_at_5_random"],
        "approx_beats_random": metrics["approx_beats_random"],
        "runtime_s": metrics["runtime_s"],
        "seed": SEED,
    }
    write_artifacts(
        RESULTS, metrics, shot, query_checks, curve, SEED, n_docs, len(id2word), dim,
        N_BITS, retrieval_pass, n_pass_top1, n_pass_top3, mean_recall_rerank,
        mean_recall_raw, mean_recall_random, approx_ok, runtime_s,
    )
    print(json.dumps(shot, indent=2))
    print(
        f"\nWrote results/ in {runtime_s:.2f}s -- "
        f"retrieval={metrics['retrieval_sanity']} "
        f"recall@5_rerank={mean_recall_rerank:.3f} "
        f"random={mean_recall_random:.3f}"
    )


if __name__ == "__main__":
    main()
