# PipeOwl-1.0

Transformer-free semantic retrieval engine.

PipeOwl performs deterministic vocabulary scoring over a static embedding field:

score = α · cosine_similarity + β · scalar_bias

- O(n) over vocabulary.
- No attention.
- No transformer weights.

## Notice

Embedding files are hosted on HuggingFace due to size constraints.
Download from:
https://huggingface.co/WangKaiLin/PipeOwl

## Architecture

- Static embedding table (V × D)
- Aligned vocabulary index
- Optional scalar bias field
- Linear scoring
- Pluggable decoder stage
- Targeted for CPU environments and low-latency systems (e.g. IME).
- Single static field (~663MB), no runtime model weights.

## Attribution

The base embedding vectors were generated using BGE (Apache-2.0) via inference.
This repository does not redistribute any original BGE weights.

## Quickstart

```bash
pip install numpy
python quickstart.py
```

## Minimal usage:

from engine import PipeOwlEngine, PipeOwlConfig

engine = PipeOwlEngine(PipeOwlConfig())
q = engine.encode("雪鴞好可愛")

## Stress Test (Hard Retrieval Setting)

Corpus size = 1200
Eval size = 200
OOD ratio = 0.28

| Model | in-domain MRR@10 | OOD MRR@10 |
|--------|-----------------|------------|
| MiniLM | 0.019 | 0.026 |
| BGE | 0.026 | 0.009 |
| PipeOwl | 0.013 | 0.023 |

## See full experimental notes here:

https://hackmd.io/@galaxy4552/BkpUEnTwbl

## Repository Structure

```bash
pipeowl/
├─ engine.py
├─ quickstart.py
└─ data/
    ├─ L1_base_embeddings.npy
    ├─ delta_base_scalar.npy
    └─ L1_base_vocab.json
```

## PipeOwl 是一個基於靜態語義場的幾何檢索系統。

核心公式：

    score = α · base + β · delta

其中：
- base = embedding cosine similarity
- delta = 靜態場偏移量
- α / β 為可調權重

提供一種 O(n) 的輕量語義計分方法，
適合低延遲環境（如輸入法）。

## LICENSE

MIT