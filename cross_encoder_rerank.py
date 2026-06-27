#!/usr/bin/env python3
"""
cross_encoder_rerank.py
=======================
Phase 2 — Offline Cross-Encoder Re-Ranking Pipeline.

Takes the Top-2000 bi-encoder shortlist produced by Phase 1
(heuristic_trap_filter + generate_embeddings) and re-scores every candidate
against the target Job Description using a pairwise Cross-Encoder model.

Cross-Encoders process both texts *jointly* through a single Transformer
forward pass, giving them access to full cross-attention between the JD
and the candidate profile.  This is dramatically more accurate than the
independent bi-encoder cosine similarity used in Phase 1, but ~100x slower
— hence the two-phase funnel.

Model:  cross-encoder/ms-marco-MiniLM-L-6-v2  (~80 MB, 6-layer MiniLM)
Target: RTX 5060 (8 GB VRAM) — 2000 pairs at batch_size=32 fits comfortably.

Output
------
  cross_encoder_semantics.parquet
    candidate_id    String   — unique identifier
    semantic_score  Float32  — sigmoid-normalised relevance score [0.0, 1.0]

Usage
-----
  python cross_encoder_rerank.py
  python cross_encoder_rerank.py --top-k 2000 --batch-size 32 --output cross_encoder_semantics.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from sentence_transformers import CrossEncoder

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s -- %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("cross_encoder_rerank")


# ---------------------------------------------------------------------------
# Target Job Description  (identical to generate_embeddings.py)
# ---------------------------------------------------------------------------

TARGET_JD: str = (
    "Company: Redrob AI (Series A AI-native talent intelligence platform). "
    "Role: Senior AI Engineer -- Founding Team. "
    "Total Experience Required: 5-9 years in applied ML/AI roles at product companies. "
    "Responsibilities: Own the intelligence layer, ranking, retrieval, and matching systems. "
    "Ship v2 ranking system involving embeddings, hybrid retrieval, and LLM-based re-ranking. "
    "Set up evaluation infrastructure (offline benchmarks, online A/B testing, NDCG, MRR). "
    "Core Required Skills: Production experience with embeddings-based retrieval systems "
    "(sentence-transformers, OpenAI embeddings, BGE, E5). "
    "Production experience with vector databases or hybrid search infrastructure "
    "(Pinecone, Weaviate, Qdrant, Milvus, FAISS). "
    "Strong Python code quality. Hands-on evaluation frameworks. "
    "Bonus Skills: LLM fine-tuning (LoRA, PEFT), XGBoost learning-to-rank, "
    "distributed systems NLP."
)


# ---------------------------------------------------------------------------
# Step 1 — Identify the Top-K shortlist from pre-computed signals
# ---------------------------------------------------------------------------

def identify_shortlist(
    heuristics_path: str,
    semantics_path: str,
    top_k: int,
) -> list[str]:
    """
    Load both Phase-1 Parquets, fuse them, and return the candidate_ids
    of the top *top_k* candidates (honeypots excluded).

    Parameters
    ----------
    heuristics_path : str
        Path to ``precomputed_heuristics.parquet``.
    semantics_path : str
        Path to ``precomputed_semantics.parquet``.
    top_k : int
        Number of candidates to shortlist for cross-encoder inference.

    Returns
    -------
    list[str]
        Ordered list of candidate_id strings (best first).
    """
    logger.info("Loading pre-computed signals for shortlist identification...")

    df_h: pl.DataFrame = pl.read_parquet(heuristics_path)
    df_s: pl.DataFrame = pl.read_parquet(semantics_path)

    df_fused: pl.DataFrame = (
        df_h.join(df_s, on="candidate_id", how="inner")
        .filter(pl.col("is_honeypot") == False)  # noqa: E712
        .with_columns(
            (pl.col("semantic_score") * pl.col("behavioral_multiplier"))
            .alias("phase1_score")
        )
        .sort("phase1_score", descending=True)
        .head(top_k)
    )

    shortlist: list[str] = df_fused["candidate_id"].to_list()
    logger.info(
        "Shortlist ready | top_k=%d | score_range=[%.4f, %.4f]",
        len(shortlist),
        df_fused["phase1_score"].min(),
        df_fused["phase1_score"].max(),
    )
    return shortlist


# ---------------------------------------------------------------------------
# Step 2 — Extract raw profiles for the shortlist from JSONL
# ---------------------------------------------------------------------------

def extract_profiles(
    jsonl_path: str,
    target_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """
    Stream the full JSONL file and extract raw JSON records only for
    candidates whose ``candidate_id`` is in *target_ids*.

    Uses a fast string pre-check before JSON parsing to skip irrelevant
    lines cheaply.

    Parameters
    ----------
    jsonl_path : str
        Path to ``candidates.jsonl``.
    target_ids : set[str]
        Set of candidate_id strings to extract.

    Returns
    -------
    dict[str, dict]
        Mapping from candidate_id to full parsed JSON record.
    """
    logger.info(
        "Streaming JSONL to extract %d shortlisted profiles...",
        len(target_ids),
    )
    profiles: dict[str, dict[str, Any]] = {}
    n_scanned: int = 0

    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            n_scanned += 1

            # Fast string pre-check — skip JSON parsing for irrelevant lines
            if not any(cid in line for cid in target_ids):
                continue

            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue

            cid: str = record.get("candidate_id", "")
            if cid in target_ids:
                profiles[cid] = record
                if len(profiles) == len(target_ids):
                    break  # Found all targets — stop early

            if n_scanned % 25_000 == 0:
                logger.info(
                    "  ...scanned %d lines, found %d/%d profiles",
                    n_scanned,
                    len(profiles),
                    len(target_ids),
                )

    logger.info(
        "Extraction complete | scanned=%d | found=%d/%d",
        n_scanned,
        len(profiles),
        len(target_ids),
    )
    return profiles


# ---------------------------------------------------------------------------
# Step 3 — Synthesize structured text from raw candidate dicts
# ---------------------------------------------------------------------------

def synthesize_profile(candidate: dict[str, Any]) -> str:
    """
    Convert a raw candidate JSON record into a dense, structured text block
    optimised for cross-encoder attention.

    Includes: title, YoE, skills (with proficiency), and top-3 career
    history entries — the same signals the bi-encoder used, but formatted
    more verbosely so the cross-attention layers can reason over details.

    Parameters
    ----------
    candidate : dict
        Full parsed JSON record for a single candidate.

    Returns
    -------
    str
        Synthesized profile string.
    """
    profile: dict[str, Any] = candidate.get("profile") or {}

    # -- Skills
    skills: list[dict[str, Any]] = candidate.get("skills") or []
    skill_parts: list[str] = [
        f"{s.get('name', 'Unknown')} ({s.get('proficiency', 'unknown')})"
        for s in skills
        if s.get("name")
    ]
    skills_text: str = ", ".join(skill_parts) if skill_parts else "None listed"

    # -- Career history (top 3 to stay within model's 512-token window)
    history: list[dict[str, Any]] = (candidate.get("career_history") or [])[:3]
    history_parts: list[str] = []
    for job in history:
        entry: str = (
            f"{job.get('title', 'Unknown Role')} at "
            f"{job.get('company', 'Unknown Company')} "
            f"for {job.get('duration_months', 0)} months "
            f"({job.get('industry', 'Unknown')}). "
            f"{job.get('description', '')}"
        )
        history_parts.append(entry)
    history_text: str = " | ".join(history_parts) if history_parts else "No history"

    synthesized: str = (
        f"Current Title: {profile.get('current_title', 'Unknown')}. "
        f"Total Experience: {profile.get('years_of_experience', 0)} years. "
        f"Summary: {profile.get('summary', 'N/A')} "
        f"Core Skills: {skills_text}. "
        f"Career History: {history_text}"
    )
    return synthesized


# ---------------------------------------------------------------------------
# Step 4 & 5 — Cross-Encoder inference + sigmoid normalisation
# ---------------------------------------------------------------------------

def run_cross_encoder(
    candidate_ids: list[str],
    candidate_texts: list[str],
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    """
    Run pairwise cross-encoder inference for each (JD, candidate) pair
    and return sigmoid-normalised scores in [0.0, 1.0].

    Parameters
    ----------
    candidate_ids : list[str]
        Ordered candidate IDs (used only for logging).
    candidate_texts : list[str]
        Synthesized text for each candidate, aligned with *candidate_ids*.
    model_name : str
        HuggingFace model identifier for the CrossEncoder.
    batch_size : int
        Inference batch size (GPU).

    Returns
    -------
    np.ndarray
        1-D array of float64 scores, shape ``(len(candidate_ids),)``.
    """
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Initializing CrossEncoder '%s' on %s...", model_name, device.upper())

    model: CrossEncoder = CrossEncoder(model_name, device=device)

    # Format as pairwise inputs: [[JD, Candidate_1], [JD, Candidate_2], ...]
    pairs: list[list[str]] = [[TARGET_JD, text] for text in candidate_texts]

    logger.info(
        "Starting inference on %d pairs (batch_size=%d)...",
        len(pairs),
        batch_size,
    )
    t0: float = time.perf_counter()

    # model.predict() returns raw logits (can be negative)
    with torch.no_grad():
        raw_logits: np.ndarray = model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=True,
        )

    elapsed: float = time.perf_counter() - t0
    logger.info(
        "Inference complete in %.2f s | "
        "logit_range=[%.4f, %.4f] | pairs_per_sec=%.1f",
        elapsed,
        float(np.min(raw_logits)),
        float(np.max(raw_logits)),
        len(pairs) / elapsed,
    )

    # -- Sigmoid normalisation: bound scores strictly to [0.0, 1.0]
    scores: np.ndarray = 1.0 / (1.0 + np.exp(-raw_logits))

    logger.info(
        "Sigmoid normalisation applied | "
        "score_range=[%.6f, %.6f] | mean=%.6f",
        float(np.min(scores)),
        float(np.max(scores)),
        float(np.mean(scores)),
    )
    return scores


# ---------------------------------------------------------------------------
# Step 6 — Save output Parquet
# ---------------------------------------------------------------------------

def save_output(
    candidate_ids: list[str],
    scores: np.ndarray,
    output_path: str,
) -> None:
    """
    Write the cross-encoder scores to a two-column Parquet file.

    Schema
    ------
    candidate_id    String
    semantic_score  Float32

    Parameters
    ----------
    candidate_ids : list[str]
        Ordered candidate IDs.
    scores : np.ndarray
        Sigmoid-normalised scores aligned with *candidate_ids*.
    output_path : str
        Destination file path.
    """
    df: pl.DataFrame = pl.DataFrame(
        {
            "candidate_id": candidate_ids,
            "semantic_score": scores.tolist(),
        },
        schema={
            "candidate_id": pl.String,
            "semantic_score": pl.Float32,
        },
    )

    out: Path = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out, compression="zstd", statistics=True)

    logger.info(
        "Output saved | rows=%d | path=%s | schema=%s",
        df.height,
        out.resolve(),
        df.schema,
    )


# ---------------------------------------------------------------------------
# CLI & Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cross_encoder_rerank",
        description=(
            "Phase 2: Cross-Encoder re-ranking of the Top-K bi-encoder "
            "shortlist for maximum NDCG@10 accuracy."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--heuristics", default="precomputed_heuristics.parquet",
        help="Path to Phase 1 heuristics Parquet.",
    )
    parser.add_argument(
        "--semantics", default="precomputed_semantics.parquet",
        help="Path to Phase 1 bi-encoder semantics Parquet.",
    )
    parser.add_argument(
        "--candidates", default="datasets/candidates.jsonl",
        help="Path to the raw JSONL candidate pool.",
    )
    parser.add_argument(
        "--output", "-o", default="cross_encoder_semantics.parquet",
        help="Output path for cross-encoder re-ranked scores.",
    )
    parser.add_argument(
        "--top-k", type=int, default=2000,
        help="Number of candidates to shortlist for cross-encoder inference.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="GPU inference batch size.",
    )
    parser.add_argument(
        "--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="HuggingFace CrossEncoder model identifier.",
    )
    args = parser.parse_args()

    pipeline_start: float = time.perf_counter()

    # ── Step 1: Identify shortlist ────────────────────────────────────
    shortlist_ids: list[str] = identify_shortlist(
        args.heuristics, args.semantics, args.top_k,
    )

    # ── Step 2: Extract raw profiles ──────────────────────────────────
    profiles: dict[str, dict[str, Any]] = extract_profiles(
        args.candidates, set(shortlist_ids),
    )

    # ── Step 3: Synthesize text (preserve shortlist order) ────────────
    logger.info("Synthesizing %d candidate profiles into text...", len(shortlist_ids))
    ordered_ids: list[str] = [
        cid for cid in shortlist_ids if cid in profiles
    ]
    ordered_texts: list[str] = [
        synthesize_profile(profiles[cid]) for cid in ordered_ids
    ]

    if len(ordered_ids) < len(shortlist_ids):
        logger.warning(
            "Could not find profiles for %d/%d shortlisted candidates "
            "(missing from JSONL). Proceeding with %d.",
            len(shortlist_ids) - len(ordered_ids),
            len(shortlist_ids),
            len(ordered_ids),
        )

    # ── Step 4 & 5: Cross-Encoder inference + sigmoid ─────────────────
    scores: np.ndarray = run_cross_encoder(
        ordered_ids, ordered_texts, args.model, args.batch_size,
    )

    # ── Step 6: Save output ───────────────────────────────────────────
    save_output(ordered_ids, scores, args.output)

    total_elapsed: float = time.perf_counter() - pipeline_start
    logger.info("Phase 2 pipeline complete in %.2f s.", total_elapsed)


if __name__ == "__main__":
    main()
