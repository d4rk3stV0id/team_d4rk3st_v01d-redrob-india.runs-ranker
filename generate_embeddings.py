#!/usr/bin/env python3
"""
generate_embeddings.py
======================
Offline Phase 1 (Bi-Encoder) - GPU-Accelerated Semantic Scoring Pipeline using BAAI/bge-large-en-v1.5.

This script processes a 100K JSONL candidate pool in memory-safe chunks, synthesizes
a highly structured natural language profile for each candidate, and computes cosine
similarity against the target Job Description using a local NVIDIA GPU.

Hardware Target: RTX 5060 (8GB VRAM) & 16GB System RAM.
"""

import json
import logging
import argparse
import time
from typing import List, Dict, Tuple, Any

import torch
import polars as pl
from sentence_transformers import SentenceTransformer, util

# ==========================================
# TARGET JOB DESCRIPTION (PRE-FORMATTED)
# ==========================================
# We strip the formatting and distill the JD down to the core requirements
# so the embedding model isn't distracted by "culture fit" boilerplate.
TARGET_JD = """
Company: Redrob AI (Series A AI-native talent intelligence platform).
Role: Senior AI Engineer — Founding Team.
Total Experience Required: 5-9 years in applied ML/AI roles at product companies.
Responsibilities: Own the intelligence layer, ranking, retrieval, and matching systems. 
Ship v2 ranking system involving embeddings, hybrid retrieval, and LLM-based re-ranking.
Set up evaluation infrastructure (offline benchmarks, online A/B testing, NDCG, MRR).
Core Required Skills: Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5).
Production experience with vector databases or hybrid search infrastructure (Pinecone, Weaviate, Qdrant, Milvus, FAISS).
Strong Python code quality. Hands-on evaluation frameworks.
Bonus Skills: LLM fine-tuning (LoRA, PEFT), XGBoost learning-to-rank, distributed systems NLP.
"""

class SemanticEmbedder:
    def __init__(self, input_path: str, output_path: str, chunk_size: int = 10_000, batch_size: int = 64):
        self.input_path = input_path
        self.output_path = output_path
        self.chunk_size = chunk_size   # How many JSON rows to load into RAM at once
        self.batch_size = batch_size   # How many strings to pass to the GPU at once
        
        # Check for RTX 5060
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Initializing model on device: {self.device.upper()}")
        
        # BAAI/bge-large-en-v1.5 is a top-tier retrieval model (1.2GB memory footprint)
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=self.device)
        
        # BGE requires queries (but not the corpus) to use this exact instruction prefix
        instruction = "Represent this sentence for searching relevant passages: "
        self.jd_vector = self.model.encode(
            instruction + TARGET_JD, 
            convert_to_tensor=True, 
            normalize_embeddings=True # Normalizing allows fast dot-product/cosine sim
        )

    def _synthesize_profile(self, candidate: Dict[str, Any]) -> str:
        """
        Converts the nested JSON dictionary into a highly structured, dense text block.
        This focuses the attention mechanism of the embedding model on what matters.
        """
        profile = candidate.get('profile', {})
        
        # Compile Skills
        skills = candidate.get('skills', [])
        skill_strings = [f"{s.get('name', '')} ({s.get('proficiency', '')})" for s in skills if s.get('name')]
        skills_text = ", ".join(skill_strings)
        
        # Compile Career History (Limit to top 3 to avoid exceeding 512 token limit)
        history = candidate.get('career_history', [])[:3]
        history_strings = []
        for job in history:
            job_str = (
                f"Role: {job.get('title', '')} at {job.get('company', '')} "
                f"for {job.get('duration_months', 0)} months. "
                f"Industry: {job.get('industry', 'Unknown')}. "
                f"Description: {job.get('description', '')}"
            )
            history_strings.append(job_str)
        history_text = " | ".join(history_strings)
        
        # Final Synth String
        synthesized_text = (
            f"Candidate Title: {profile.get('current_title', '')}. "
            f"Total Experience: {profile.get('years_of_experience', 0)} years. "
            f"Summary: {profile.get('summary', '')} "
            f"Core Skills: {skills_text}. "
            f"Recent Career History: {history_text}"
        )
        return synthesized_text

    def process_chunk(self, chunk: List[Dict[str, Any]]) -> Tuple[List[str], List[float]]:
        """Processes a chunk of candidates, computes similarities, and returns ids & scores."""
        candidate_ids = [c.get('candidate_id') for c in chunk]
        texts = [self._synthesize_profile(c) for c in chunk]
        
        # Generate embeddings on GPU (batch_size=64 prevents 8GB VRAM OOM)
        with torch.no_grad():
            candidate_vectors = self.model.encode(
                texts, 
                batch_size=self.batch_size, 
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            
            # Compute Cosine Similarity (Since vectors are normalized, dot product = cosine sim)
            # Resulting shape: (1, len(chunk)). Squeeze to 1D array.
            cosine_scores = util.dot_score(self.jd_vector, candidate_vectors)[0]
            
        return candidate_ids, cosine_scores.cpu().numpy().tolist()

    def run(self):
        start_time = time.time()
        results_ids = []
        results_scores = []
        
        chunk = []
        total_processed = 0
        
        logging.info("Starting JSONL stream and embedding generation...")
        
        with open(self.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                chunk.append(json.loads(line))
                
                if len(chunk) >= self.chunk_size:
                    ids, scores = self.process_chunk(chunk)
                    results_ids.extend(ids)
                    results_scores.extend(scores)
                    total_processed += len(chunk)
                    
                    logging.info(f"Processed {total_processed} candidates...")
                    chunk = [] # Clear RAM chunk
                    torch.cuda.empty_cache() # Prevent VRAM fragmentation
            
            # Process remaining records
            if chunk:
                ids, scores = self.process_chunk(chunk)
                results_ids.extend(ids)
                results_scores.extend(scores)
                total_processed += len(chunk)
                logging.info(f"Processed final batch. Total: {total_processed} candidates.")

        # Save via Polars
        logging.info(f"Saving results to {self.output_path}...")
        df = pl.DataFrame({
            "candidate_id": results_ids,
            "semantic_score": results_scores
        }).with_columns(
            pl.col("semantic_score").cast(pl.Float32)
        )
        
        df.write_parquet(self.output_path)
        
        elapsed = time.time() - start_time
        logging.info(f"Semantic Pipeline complete in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    
    embedder = SemanticEmbedder(
        input_path="datasets/candidates.jsonl",
        output_path="precomputed_semantics.parquet",
        chunk_size=10_000, # Load 10k strings into RAM
        batch_size=64      # Process 64 tensors at a time on the GPU
    )
    
    embedder.run()