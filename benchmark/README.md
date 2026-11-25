# LoCoMo benchmarks

These results cover 1,540 questions across all ten LoCoMo conversations. Both experiments ran against `mem0-decay` commit `8c9ce83`, Mem0 OSS JS `3.3.1`, and Qdrant `1.20.0`. The results describe that revision. [Dataset](https://github.com/snap-research/locomo/tree/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376) · [Pinned Mem0 harness](https://github.com/mem0ai/memory-benchmarks/tree/4b61c5d31b9c668a12b4f5e78064248a02c82d2b)

## Time-driven compression

Both arms used 30-day exponential half-life decay and a 0.95 search floor. The historical clock advanced on weekly checkpoints. Live sweeps in the compaction arm retired **4,958 source memories** and wrote **1,711 Tier 1 summaries**. Both arms answered questions as of the last timestamp in the dataset, **January 12, 2024 at 13:41 UTC**.

| Metric | Bounded control | With compaction |
| --- | ---: | ---: |
| Top-50 GPT-4o answer accuracy | 90.06% (1,387/1,540) | 88.77% (1,367/1,540) |
| Active memory records | 9,719 | 6,472 |
| Active entity records | 40,221 | 20,467 |
| Allocated Qdrant disk, memory collection | 113.98 MiB | 108.38 MiB |
| Allocated Qdrant disk, memory and entity collections | 425.45 MiB | 292.85 MiB |
| Retrieval p50 / p95, cached embeddings, 1,540 pairs | 116.1 / 211.5 ms | 106.9 / 196.8 ms |
| Retrieval p50 / p95, fresh embeddings, 50 pairs | 592.6 / 736.4 ms | 571.4 / 689.7 ms |

The paired change was **31.2% less allocated disk** across both collections and **1.30 percentage points lower accuracy**. The conversation-bootstrap 95% interval for the accuracy change was −2.37 to −0.14 points. Retrieval timing began after compression finished and excluded answer generation and judging. Estimated OpenAI spend was **$42.02** including the pilot, of which about **$0.10** was compression summaries and embeddings. [Full results](locomo-compaction-results.json) · [OpenAI pricing](https://developers.openai.com/api/docs/models/gpt-4o)

The replay used the existing `sweep()` implementation with seven-day buckets, pairwise cosine similarity of at least 0.75, and raw decay strength below 0.05. A benchmark adapter exposed the dense array from Qdrant's hybrid vector payload to the sweep reader. The client was pinned to Qdrant JS `1.17.0` for compatibility. [Parameters](parameters.json)

## Original ranking experiment

| Retrieval cutoff | Native Mem0 OSS ranking | `soft@v1` ranking |
| --- | ---: | ---: |
| Top 50 | 90.45% (1,393/1,540) | 76.62% (1,180/1,540) |
| Top 200 | 90.52% (1,394/1,540) | 90.52% (1,394/1,540) |

This run held the stored corpus fixed, used `soft@v1` with a 30-day half-life and a 0.30 floor, and queried each conversation at its own final session. The native OSS comparison to the compressed arm spans two query clock protocols. The paired compaction control above provides the same-clock comparison. Native OSS disk use and database-backed retrieval latency were unmeasured. [Full ranking results](locomo-ranking-results.json)

## Bounded-decay candidate replay

The top-50 evidence-hit rate counts questions where the selection contains a memory from at least one annotated supporting turn. On 1,531 eligible questions, the 30-day curve with a 0.95 floor hit **90.73%**, relevance-only ranking hit **90.53%**, and the original 0.30-floor curve hit **69.63%**. These are retrieval evidence rates. The replay also records the curve sweep and wrapper-only latency measurements. [Replay results](locomo-replay-results.json)
