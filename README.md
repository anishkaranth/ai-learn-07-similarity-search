# AI Learn 07 — Similarity Search (Tiny Vector Store)

Build a **pure-NumPy vector store** over a toy document corpus: TF-IDF embeddings, exact top-k cosine retrieval, and approximate nearest neighbors via **random-projection LSH** (bit signatures + Hamming), compared by recall@k.

Phase B (AI components) — first topic after Phase A embeddings (`ai-learn-06`).

## Learning goals

- **TF-IDF** turns documents into sparse vectors; **L2-normalize** so cosine = dot product
- **Exact search** ranks every doc by cosine (brute force) — the ground-truth baseline
- **Random-projection LSH** maps each vector to a bit code; Hamming distance approximates angular distance
- **Recall@k** measures how well approximate top-k recovers exact top-k
- Re-ranking a short LSH candidate list with exact cosine recovers most quality cheaply

## Brief math

TF-IDF for term \(t\) in document \(d\):

\[
\mathrm{tfidf}(t,d) = (1 + \log \mathrm{tf}_{t,d}) \cdot \left(\log\frac{N+1}{\mathrm{df}_t+1} + 1\right)
\]

Cosine on L2-normalized vectors \(\hat{q}, \hat{d}\): \(\cos = \hat{q}^\top \hat{d}\).

LSH bits from random hyperplanes \(R \in \mathbb{R}^{b \times D}\):

\[
\mathrm{code}(x)_i = \mathbf{1}[R_i^\top x \ge 0]
\]

Hamming distance between codes approximates cosine distance (Charikar / SimHash).

## Layout

```
corpus.py              # toy document corpus + labeled queries
similarity_search.py   # TF-IDF, exact search, LSH, metrics helpers
run_smoke.py           # end-to-end smoke -> results/
notebooks/similarity_search.ipynb
results/               # committed metrics + plots
```

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_smoke.py
```

Runs on CPU in a few seconds. See `results/RESULTS.md` for the latest smoke metrics.

## What you'll learn next (Phase B)

Toy RAG retrieval over this vector store, tool-calling stubs, evals, and light fine-tuning (e.g. LoRA).
