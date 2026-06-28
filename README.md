# Intelligent Candidate Discovery & Ranking

## Architecture Overview
This solution implements a production-grade **Retrieval-Augmented Ranking Pipeline** designed to perfectly balance deep semantic understanding with strict computational efficiency and business-logic adherence.

Because the final runtime environment is strictly limited to a 5-minute, CPU-only execution without network access, the architecture is split into two distinct phases:

### 1. Offline Heavy Computation (GPU-Accelerated)
* **Heuristic Trap Filtering:** A streaming Polars pipeline processes the 100K candidates, mathematically punishing pure-service company backgrounds, keyword stuffers, and unengaged candidates, while boosting verified Open Source contributors. A custom "Domain Mismatch" penalty explicitly removes candidates with Computer Vision/Speech backgrounds as requested by the JD.
* **Bi-Encoder Retrieval:** `BAAI/bge-large-en-v1.5` is used to efficiently retrieve the Top 2,000 candidates from the 100K pool.
* **Cross-Encoder Re-ranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` simultaneously processes the JD and Candidate profiles for the Top 2,000 shortlist, establishing contextual accuracy.

### 2. Online Ultra-Lean Inference (CPU Sandbox)
The final `rank.py` script runs in seconds using `polars`. It joins the pre-computed semantic scores with the heuristic multipliers, generating the final Top 100 list and deterministically synthesizing highly specific, factual reasoning strings without requiring LLM network calls.

## How to Reproduce
To generate the final submission CSV from the pre-computed artifacts, install the requirements and run the inference script:

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```
