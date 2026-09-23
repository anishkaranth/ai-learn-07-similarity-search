# Smoke results -- ai-learn-07-similarity-search

**Seed:** `42` | docs=19 | vocab=142 | dim=142 | LSH bits=64

## Headline metrics

| Metric | Value |
|--------|------:|
| Exact retrieval sanity (topic cluster) | PASS |
| Labeled queries top-1 / top-3 pass | 7/8 ; 8/8 |
| Mean recall@5 (LSH + cosine re-rank) | 0.5852 |
| Mean recall@5 (LSH Hamming only) | 0.4296 |
| Mean recall@5 (random baseline) | 0.2148 |
| Approx beats random | PASS |
| Wall time (CPU) | 0.70s |

## Exact retrieval checks (expected topic in top-3)

| Query | Expected | Top-3 hits | Status |
|-------|----------|------------|--------|
| `cat dog pets animals` | animals | s4(science,0.22), a2(animals,0.20), a1(animals,0.19) | PASS |
| `rain storm clouds weather` | weather | w2(weather,0.49), a1(animals,0.00), a3(animals,0.00) | PASS |
| `python java programming code` | programming | p1(programming,0.63), a1(animals,0.00), a3(animals,0.00) | PASS |
| `king queen castle royalty` | royalty | r1(royalty,0.61), a1(animals,0.00), a3(animals,0.00) | PASS |
| `math physics science school` | science | s1(science,0.52), s2(science,0.18), a3(animals,0.00) | PASS |
| `neural networks embeddings vectors` | programming | p3(programming,0.34), p2(programming,0.34), a1(animals,0.00) | PASS |
| `moon stars night sky` | weather | w3(weather,0.56), a3(animals,0.15), a2(animals,0.00) | PASS |
| `birds fish swim fly` | animals | a3(animals,0.62), a4(animals,0.14), a2(animals,0.00) | PASS |

## Recall@k vs exact ground truth

| k | LSH+rerank | LSH Hamming | Random |
|--:|----------:|------------:|-------:|
| 1 | 0.963 | 0.963 | 0.000 |
| 2 | 0.815 | 0.611 | 0.056 |
| 3 | 0.679 | 0.444 | 0.136 |
| 4 | 0.611 | 0.463 | 0.222 |
| 5 | 0.585 | 0.430 | 0.274 |
| 6 | 0.562 | 0.451 | 0.247 |
| 7 | 0.556 | 0.508 | 0.376 |
| 8 | 0.565 | 0.565 | 0.375 |
| 9 | 0.539 | 0.593 | 0.457 |
| 10 | 0.507 | 0.615 | 0.511 |

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
