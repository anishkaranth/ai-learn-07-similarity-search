"""Plots and RESULTS.md / metrics writers for the similarity-search smoke."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from similarity_search import DOCUMENTS, LABELED_QUERIES, embed_query, exact_topk, topic_of


def make_plots(results: Path, word2id, idf, X, curve, n_bits: int) -> None:
    showcase = "python java programming code"
    qv = embed_query(showcase, word2id, idf)
    idx, scores = exact_topk(qv, X, k=5)
    labels = [f"{DOCUMENTS[int(i)]['id']}\n({topic_of(int(i))})" for i in idx]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(idx)))
    bars = ax.bar(range(len(idx)), scores, color=colors)
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("cosine similarity")
    ax.set_title(f'Exact top-5 for query: "{showcase}"')
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    for b, s in zip(bars, scores):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.02, f"{s:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(results / "topk_scores.png", dpi=120)
    fig.savefig(results / "topk_scores.svg")
    plt.close(fig)

    q_labels = [str(q["query"])[:28] for q in LABELED_QUERIES]
    d_labels = [d["id"] for d in DOCUMENTS]
    n_docs = X.shape[0]
    heat = np.zeros((len(LABELED_QUERIES), n_docs), dtype=np.float64)
    for i, item in enumerate(LABELED_QUERIES):
        qv = embed_query(str(item["query"]), word2id, idf)
        heat[i] = X @ qv
    fig, ax = plt.subplots(figsize=(11, 5.5))
    im = ax.imshow(heat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(n_docs))
    ax.set_xticklabels(d_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(q_labels)))
    ax.set_yticklabels(q_labels, fontsize=8)
    ax.set_xlabel("document id")
    ax.set_ylabel("query")
    ax.set_title("Query -> document cosine similarity (TF-IDF)")
    topic_colors = {
        "animals": "#4c78a8",
        "weather": "#f58518",
        "programming": "#54a24b",
        "royalty": "#e45756",
        "science": "#b279a2",
    }
    for j, d in enumerate(DOCUMENTS):
        ax.add_patch(
            plt.Rectangle(
                (j - 0.5, -0.5), 1, 0.15,
                color=topic_colors.get(d["topic"], "gray"),
                clip_on=False,
            )
        )
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="cosine")
    fig.tight_layout()
    fig.savefig(results / "query_doc_heatmap.png", dpi=120)
    fig.savefig(results / "query_doc_heatmap.svg")
    plt.close(fig)

    ks = [c["k"] for c in curve]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ks, [c["recall_lsh_rerank"] for c in curve], "o-", label="LSH + cosine re-rank", lw=2)
    ax.plot(ks, [c["recall_lsh_hamming"] for c in curve], "s--", label="LSH Hamming only", lw=1.5)
    ax.plot(ks, [c["recall_random"] for c in curve], "^:", label="random baseline", lw=1.5)
    ax.set_xlabel("k")
    ax.set_ylabel("mean recall@k vs exact")
    ax.set_title(f"Approximate retrieval quality (n_bits={n_bits})")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(ks)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results / "recall_at_k.png", dpi=120)
    fig.savefig(results / "recall_at_k.svg")
    plt.close(fig)


def write_artifacts(
    results: Path,
    metrics: Dict[str, Any],
    shot: Dict[str, Any],
    query_checks: List[Dict[str, Any]],
    curve: Sequence[Dict[str, Any]],
    seed: int,
    n_docs: int,
    vocab_size: int,
    dim: int,
    n_bits: int,
    retrieval_pass: bool,
    n_pass_top1: int,
    n_pass_top3: int,
    mean_recall_rerank: float,
    mean_recall_raw: float,
    mean_recall_random: float,
    approx_ok: bool,
    runtime_s: float,
) -> None:
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (results / "JSON.shot").write_text(json.dumps(shot, indent=2) + "\n")
    check_rows = []
    for c in query_checks:
        status = "PASS" if c["topn_ok"] else "FAIL"
        tops = ", ".join(
            f"{h['doc_id']}({h['topic']},{h['cosine']:.2f})" for h in c["hits"][:3]
        )
        check_rows.append(
            f"| `{c['query'][:40]}` | {c['expected_topic']} | {tops} | {status} |"
        )
    curve_rows = "\n".join(
        f"| {c['k']} | {c['recall_lsh_rerank']:.3f} | {c['recall_lsh_hamming']:.3f} | {c['recall_random']:.3f} |"
        for c in curve
    )
    md = f"""# Smoke results -- ai-learn-07-similarity-search

**Seed:** `{seed}` | docs={n_docs} | vocab={vocab_size} | dim={dim} | LSH bits={n_bits}

## Headline metrics

| Metric | Value |
|--------|------:|
| Exact retrieval sanity (topic cluster) | {"PASS" if retrieval_pass else "FAIL"} |
| Labeled queries top-1 / top-3 pass | {n_pass_top1}/{len(query_checks)} ; {n_pass_top3}/{len(query_checks)} |
| Mean recall@5 (LSH + cosine re-rank) | {mean_recall_rerank:.4f} |
| Mean recall@5 (LSH Hamming only) | {mean_recall_raw:.4f} |
| Mean recall@5 (random baseline) | {mean_recall_random:.4f} |
| Approx beats random | {"PASS" if approx_ok else "FAIL"} |
| Wall time (CPU) | {runtime_s:.2f}s |

## Exact retrieval checks (expected topic in top-3)

| Query | Expected | Top-3 hits | Status |
|-------|----------|------------|--------|
{chr(10).join(check_rows)}

## Recall@k vs exact ground truth

| k | LSH+rerank | LSH Hamming | Random |
|--:|----------:|------------:|-------:|
{curve_rows}

## Plots

- [`topk_scores.png`](topk_scores.png) / [`topk_scores.svg`](topk_scores.svg)
- [`query_doc_heatmap.png`](query_doc_heatmap.png) / [`query_doc_heatmap.svg`](query_doc_heatmap.svg)
- [`recall_at_k.png`](recall_at_k.png) / [`recall_at_k.svg`](recall_at_k.svg)

## Takeaway

TF-IDF + L2-normalized cosine is a classic exact vector search baseline.
Random-projection LSH compresses each vector to a bit signature; Hamming
nearest neighbors approximate cosine neighbors. Re-ranking a short LSH
candidate list with exact cosine recovers most of exact top-k recall while
scanning far fewer documents -- the same idea behind production ANN indexes.
"""
    (results / "RESULTS.md").write_text(md)
