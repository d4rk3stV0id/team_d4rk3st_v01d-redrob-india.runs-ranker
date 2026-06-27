#!/usr/bin/env python3
"""
heuristic_trap_filter.py
========================
Offline Phase 3 (Constraints) - Production-grade heuristic pre-computation pipeline.

Streams a JSONL file of candidate profiles, applies four categories of
business-rule-based heuristic modifiers sequentially, and persists the
results as a lean, three-column Parquet file.  Designed for strict RAM budgets
(≤ 16 GB) by processing the stream in configurable batches and *never* loading
the full dataset into memory at once.

Output Parquet Schema
---------------------
  candidate_id          String   – unique identifier copied verbatim from source
  is_honeypot           Boolean  – True when stated vs. computed experience
                                   diverges by more than 3 years
  behavioral_multiplier Float32  – composite heuristic score (base value = 1.0)

Usage
-----
  python heuristic_trap_filter.py
  python heuristic_trap_filter.py --input data/candidates.jsonl \\
                                   --output out/precomputed_heuristics.parquet \\
                                   --batch-size 10000 --validate

Expected Input Record Shape
---------------------------
  {
    "candidate_id":       "abc-123",
    "current_title":      "Marketing Manager",
    "years_of_experience": 7,
    "career_history": [
        {"company": "TCS",    "duration_months": 36},
        {"company": "Wipro",  "duration_months": 24}
    ],
    "skills": [
        {"name": "PyTorch",    "proficiency": "expert"},
        {"name": "Excel",      "proficiency": "intermediate"}
    ],
    "redrob_signals": {
        "recruiter_response_rate": 0.12,
        "notice_period_days":      60,
        "github_activity_score":   28,
        "verified_email":          true
    }
  }

Dependencies
------------
  polars >= 0.20  (pip install polars)
  Python >= 3.10
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import polars as pl

# ---------------------------------------------------------------------------
# Logging — structured, timestamped, single source of truth
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("heuristic_trap_filter")


# ---------------------------------------------------------------------------
# Domain lookup sets  (frozenset → O(1) membership; immutable by construction)
# ---------------------------------------------------------------------------

DEFAULT_SERVICE_COMPANIES: frozenset[str] = frozenset({
    "TCS",
    "Infosys",
    "Wipro",
    "Accenture",
    "Cognizant",
    "Capgemini",
})

DEFAULT_NON_AI_TITLES: frozenset[str] = frozenset({
    "Marketing Manager",
    "Accountant",
    "HR Manager",
    "Operations Manager",
    "Customer Support",
    "Civil Engineer",
    "Project Manager",
})

DEFAULT_DEEP_AI_SKILLS: frozenset[str] = frozenset({
    "FAISS",
    "Pinecone",
    "LangChain",
    "LLMs",
    "PyTorch",
    "Embeddings",
    "Vector Search",
    "Recommendation Systems",
})

DEFAULT_MISMATCHED_DOMAIN_SKILLS: frozenset[str] = frozenset({
    "computer vision", "cv", "speech", "speech recognition", "tts", 
    "text-to-speech", "robotics", "ros", "autonomous", "object detection", 
    "image processing", "audio processing"
})

DEFAULT_REQUIRED_DOMAIN_SKILLS: frozenset[str] = frozenset({
    "nlp", "natural language processing", "information retrieval", "ir", 
    "search", "ranking", "recommendation systems", "llm", "llms", "rag", 
    "embeddings", "sentence transformers", "semantic search", "haystack", "lora"
})

# Proficiency levels considered "high" for the stuffer-detection rule
_HIGH_PROFICIENCIES: frozenset[str] = frozenset({"advanced", "expert"})


# ---------------------------------------------------------------------------
# Business-rule multipliers & thresholds
# (centralised so they can be patched in unit tests without subclassing)
# ---------------------------------------------------------------------------

_SERVICE_PENALTY: float = 0.10          # Rule 1 — pure service-company background
_STUFFER_PENALTY: float = 0.10          # Rule 2 — keyword stuffer detected

_LOW_RESPONSE_PENALTY: float = 0.50    # Rule 3a — recruiter_response_rate < threshold
_LONG_NOTICE_PENALTY:  float = 0.80    # Rule 3b — notice_period_days > threshold
_HIGH_SIGNAL_BONUS:    float = 1.20    # Rule 3c — github_activity_score + verified_email

_RESPONSE_RATE_THRESHOLD: float = 0.20  # below this → low-response penalty
_NOTICE_PERIOD_THRESHOLD: int   = 90    # above this → long-notice penalty
_GITHUB_SCORE_THRESHOLD:  float = 20.0  # above this (+ verified_email) → bonus

_HONEYPOT_DELTA_THRESHOLD: float = 3.0  # Rule 4 — years discrepancy ceiling

_JOB_HOPPER_PENALTY:       float = 0.20  # Rule 5 — title-chaser / job hopper
_JOB_HOPPER_MIN_YOE:       float = 3.0   # only penalise if computed YoE > this
_JOB_HOPPER_MAX_TENURE:    float = 1.5   # avg tenure below this triggers penalty

_OSS_HIGH_BONUS:           float = 1.30  # Rule 6a — github_activity_score > 50
_OSS_MID_BONUS:            float = 1.10  # Rule 6b — github_activity_score 20–50
_OSS_HIGH_THRESHOLD:       float = 50.0  # above this → high OSS bonus
_OSS_MID_THRESHOLD:        float = 20.0  # above this (≤ high) → mid OSS bonus

_DOMAIN_MISMATCH_PENALTY:  float = 0.10  # Rule 7 — mismatched domain


# ---------------------------------------------------------------------------
# Data Transfer Object  (slots=True cuts per-instance memory vs plain dict)
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class ProcessedRecord:
    """
    Immutable value object carrying exactly the three output columns.

    Using ``slots=True`` eliminates the ``__dict__`` overhead on each
    instance, which matters when tens of thousands of objects coexist in
    the in-flight batch list.
    """

    candidate_id:          str
    is_honeypot:           bool
    behavioral_multiplier: float


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class HeuristicTrapFilter:
    """
    Streaming heuristic filter and scoring engine for candidate profiles.

    Design Philosophy
    -----------------
    * **Single-pass streaming** — the JSONL input is opened once and consumed
      one line at a time.  Peak memory usage is bounded by ``batch_size``
      :class:`ProcessedRecord` objects plus a list of small Polars frames —
      well within a 16 GB budget even for multi-million-row files.

    * **Multiplicative scoring** — each rule returns a scalar multiplier
      (≤ 1.0 for a penalty, > 1.0 for a bonus).  Multipliers compose
      naturally: a service-company keyword-stuffer ends at 0.1 × 0.1 = 0.01
      before any availability adjustments.

    * **Defensive key access** — every dict key is read through ``.get()``
      with a typed default.  The pipeline never raises ``KeyError`` on
      malformed or partially-filled records.

    * **Atomic output** — the Parquet file is written to a ``*.tmp`` path
      and renamed into place only on success, so a partial run never leaves
      a corrupt file at the destination path.

    Parameters
    ----------
    input_path : str | Path
        Path to the source ``candidates.jsonl`` file.
    output_path : str | Path
        Destination for ``precomputed_heuristics.parquet``.
    batch_size : int
        Records per in-memory batch before flushing to a Polars DataFrame.
        Default ``10_000``; reduce if memory pressure is observed.
    service_companies : frozenset[str] | None
        Inject a custom set of pure-service companies (useful in tests).
    non_ai_titles : frozenset[str] | None
        Inject a custom set of non-AI job titles.
    deep_ai_skills : frozenset[str] | None
        Inject a custom set of deep-AI skill names.
    """

    def __init__(
        self,
        input_path:        str | Path,
        output_path:       str | Path,
        batch_size:        int = 10_000,
        service_companies: frozenset[str] | None = None,
        non_ai_titles:     frozenset[str] | None = None,
        deep_ai_skills:    frozenset[str] | None = None,
        mismatched_domain_skills: frozenset[str] | None = None,
        required_domain_skills:   frozenset[str] | None = None,
    ) -> None:
        self.input_path  = Path(input_path)
        self.output_path = Path(output_path)
        self.batch_size  = batch_size

        # Allow test injection; fall back to module-level defaults
        self._service_companies: frozenset[str] = (
            service_companies if service_companies is not None
            else DEFAULT_SERVICE_COMPANIES
        )
        self._non_ai_titles: frozenset[str] = (
            non_ai_titles if non_ai_titles is not None
            else DEFAULT_NON_AI_TITLES
        )
        self._deep_ai_skills: frozenset[str] = (
            deep_ai_skills if deep_ai_skills is not None
            else DEFAULT_DEEP_AI_SKILLS
        )
        self._mismatched_domain_skills: frozenset[str] = (
            mismatched_domain_skills if mismatched_domain_skills is not None
            else DEFAULT_MISMATCHED_DOMAIN_SKILLS
        )
        self._required_domain_skills: frozenset[str] = (
            required_domain_skills if required_domain_skills is not None
            else DEFAULT_REQUIRED_DOMAIN_SKILLS
        )

        # Telemetry counters (reset at the start of each run())
        self._n_processed:       int = 0
        self._n_honeypots:       int = 0
        self._n_stuffers:        int = 0
        self._n_service_penalty: int = 0
        self._n_job_hoppers:     int = 0
        self._n_oss_bonuses:     int = 0
        self._n_domain_mismatches: int = 0
        self._n_parse_errors:    int = 0

        logger.info(
            "HeuristicTrapFilter ready | input=%s | output=%s | batch_size=%d",
            self.input_path,
            self.output_path,
            self.batch_size,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input={str(self.input_path)!r}, "
            f"output={str(self.output_path)!r}, "
            f"batch_size={self.batch_size})"
        )

    # ------------------------------------------------------------------
    # Type-safe casting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        """
        Coerce *value* to ``float``; silently return *default* on failure.

        Guards against ``None``, empty strings, and non-numeric literals
        that raw JSON data occasionally contains.

        Parameters
        ----------
        value : object
            The raw value retrieved from a dict via ``.get()``.
        default : float
            Fallback returned when coercion fails.

        Returns
        -------
        float
        """
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: object, default: int = 0) -> int:
        """
        Coerce *value* to ``int`` via ``float`` truncation; return *default*
        on failure.

        The intermediate ``float()`` call handles values stored as ``90.0``
        in JSON, which would make a direct ``int()`` cast raise ``ValueError``.

        Parameters
        ----------
        value : object
            The raw value retrieved from a dict via ``.get()``.
        default : int
            Fallback returned when coercion fails.

        Returns
        -------
        int
        """
        try:
            return int(float(value))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------------
    # Rule 1 — Pure Service-Company Penalty
    # ------------------------------------------------------------------

    def _apply_service_company_penalty(
        self,
        career_history: list[dict],
    ) -> float:
        """
        Return ``0.1`` when *every* employer in the candidate's history is a
        pure service / body-shop company; otherwise return ``1.0``.

        Rationale
        ---------
        Candidates from pure service companies rarely have the product-thinking
        depth we need.  When a candidate has *zero* product-company exposure,
        their profile is down-weighted significantly so that the dense embedding
        layer cannot overrule this structural signal.

        Edge Cases
        ----------
        * An empty ``career_history`` is **not** penalised — absence of data
          is not evidence of a service background.
        * Entries with a missing or empty ``"company"`` key are treated as
          *unknown* (not service) so the penalty is not triggered by
          data-quality issues alone.
        * Company name matching is case-sensitive and exact to avoid false
          positives (e.g. "Accenture Federal" ≠ "Accenture").

        Parameters
        ----------
        career_history : list[dict]
            Sequence of job objects.  Each is expected to carry a
            ``"company"`` string key.

        Returns
        -------
        float
            ``0.1`` (penalty) or ``1.0`` (no penalty).
        """
        if not career_history:
            return 1.0

        companies: set[str] = {
            entry.get("company", "") for entry in career_history
        }
        companies.discard("")   # empty string → unknown, not service

        if not companies:
            return 1.0

        if companies.issubset(self._service_companies):
            self._n_service_penalty += 1
            logger.debug(
                "Rule 1 triggered — service-company penalty | companies=%s",
                companies,
            )
            return _SERVICE_PENALTY

        return 1.0

    # ------------------------------------------------------------------
    # Rule 2 — Keyword-Stuffer Penalty
    # ------------------------------------------------------------------

    def _apply_keyword_stuffer_penalty(
        self,
        current_title: str,
        skills:        list[dict],
    ) -> float:
        """
        Return ``0.1`` when a candidate's job title is non-technical yet they
        claim ``advanced`` or ``expert`` proficiency in a deep-AI skill.

        Rationale
        ---------
        Dense embedding models can be deceived by keyword stuffing: an
        "Operations Manager" who lists "expert PyTorch" will score highly
        on AI-role similarity despite the obvious mismatch.  This rule
        intercepts such profiles *before* the embedding stage.

        Implementation Notes
        --------------------
        The inner loop short-circuits on the first matching skill, making the
        method O(1) amortised for typical profiles where the first deep-AI
        match is found quickly.  Proficiency strings are lowercased before
        comparison to handle capitalisation variants ("Expert", "EXPERT").

        Parameters
        ----------
        current_title : str
            The candidate's most recent job title (exact match, case-sensitive).
        skills : list[dict]
            Skill objects, each expected to carry ``"name"`` and
            ``"proficiency"`` string keys.

        Returns
        -------
        float
            ``0.1`` (penalty) or ``1.0`` (no penalty).
        """
        if current_title not in self._non_ai_titles:
            return 1.0

        for skill in skills:
            name:        str = skill.get("name", "")
            proficiency: str = (skill.get("proficiency") or "").lower().strip()

            if name in self._deep_ai_skills and proficiency in _HIGH_PROFICIENCIES:
                self._n_stuffers += 1
                logger.debug(
                    "Rule 2 triggered — keyword-stuffer penalty | "
                    "title=%r skill=%r proficiency=%r",
                    current_title,
                    name,
                    proficiency,
                )
                return _STUFFER_PENALTY

        return 1.0

    # ------------------------------------------------------------------
    # Rule 3 — Availability Modifiers
    # ------------------------------------------------------------------

    def _apply_availability_modifiers(
        self,
        redrob_signals: dict,
    ) -> float:
        """
        Compose three independent availability signals into a single
        multiplicative modifier drawn from the ``redrob_signals``
        sub-dictionary.

        Modifier Logic (applied sequentially)
        --------------------------------------
        a. ``recruiter_response_rate < 0.20`` → ×0.50
           Candidate is consistently unresponsive to recruiter outreach,
           making successful hiring unlikely regardless of raw talent.

        b. ``notice_period_days > 90`` → ×0.80
           Long notice period increases time-to-hire risk for urgent roles.

        c. ``github_activity_score > 20 AND verified_email == True`` → ×1.20
           Active open-source contribution verified by a confirmed contact
           address is a strong, hard-to-fake quality signal.

        The three modifiers are independent and fully composable:
        a candidate who is unresponsive (×0.5) AND has a verified active
        GitHub (×1.2) ends up at 0.5 × 1.2 = 0.60 — still penalised
        overall due to the reachability concern.

        Parameters
        ----------
        redrob_signals : dict
            The ``"redrob_signals"`` sub-dictionary from the candidate record.
            All keys are accessed defensively via ``.get()``.

        Returns
        -------
        float
            Composite availability modifier.  Values < 1.0 represent a net
            penalty; values > 1.0 represent a net quality bonus.
        """
        modifier: float = 1.0

        response_rate:  float = self._safe_float(
            redrob_signals.get("recruiter_response_rate"), default=1.0
        )
        notice_days:    int   = self._safe_int(
            redrob_signals.get("notice_period_days"),      default=0
        )
        github_score:   float = self._safe_float(
            redrob_signals.get("github_activity_score"),   default=0.0
        )
        verified_email: bool  = bool(redrob_signals.get("verified_email", False))

        # 3a — unresponsive to recruiter outreach
        if response_rate < _RESPONSE_RATE_THRESHOLD:
            modifier *= _LOW_RESPONSE_PENALTY
            logger.debug(
                "Rule 3a — low response-rate penalty | rate=%.4f", response_rate
            )

        # 3b — long notice period
        if notice_days > _NOTICE_PERIOD_THRESHOLD:
            modifier *= _LONG_NOTICE_PENALTY
            logger.debug(
                "Rule 3b — long notice-period penalty | days=%d", notice_days
            )

        # 3c — high-quality signal (OSS activity + verified contact)
        if github_score > _GITHUB_SCORE_THRESHOLD and verified_email:
            modifier *= _HIGH_SIGNAL_BONUS
            logger.debug(
                "Rule 3c — high-quality signal bonus | "
                "github_score=%.1f verified_email=%s",
                github_score,
                verified_email,
            )

        return modifier

    # ------------------------------------------------------------------
    # Rule 5 — Title-Chaser / Job Hopper Penalty
    # ------------------------------------------------------------------

    def _apply_job_hopper_penalty(
        self,
        career_history: list[dict],
    ) -> float:
        """
        Return ``0.2`` when a candidate's career history reveals a pattern
        of very short tenures (average < 1.5 years) *and* they have enough
        total experience (> 3 years) that the hopping cannot be attributed
        to being early-career.

        Rationale
        ---------
        The hiring manager has stated explicitly: "If your career trajectory
        shows you switching companies every 1.5 years, we're not a fit.  We
        need someone who plans to be here for 3+ years."  This rule encodes
        that requirement as a steep multiplicative penalty applied *before*
        the semantic layer can overrule it.

        Protecting Juniors
        ------------------
        Candidates with ≤ 3.0 computed years of experience are exempted.
        A 2-year career spanning 2 short roles is expected behaviour for
        someone who has done an internship followed by their first full-time
        position — penalising them would be a false positive.

        Computation
        -----------
        ``total_months   = sum(entry["duration_months"] for each entry)``
        ``total_years    = total_months / 12``
        ``n_roles        = len(career_history)``  (only entries with non-empty company)
        ``avg_tenure_yrs = total_years / n_roles``
        ``penalty fires  ⟺  total_years > 3.0  AND  avg_tenure_yrs < 1.5``

        Parameters
        ----------
        career_history : list[dict]
            Sequence of job objects.  Each is expected to carry a
            ``"duration_months"`` numeric key and a ``"company"`` string key.

        Returns
        -------
        float
            ``0.2`` (penalty) or ``1.0`` (no penalty).
        """
        if not career_history:
            return 1.0

        # Only count entries that have a non-empty company name — guard
        # against placeholder / padding entries inflating the role count.
        valid_entries: list[dict] = [
            entry for entry in career_history
            if (entry.get("company") or "").strip()
        ]
        n_roles: int = len(valid_entries)
        if n_roles == 0:
            return 1.0

        total_months: float = sum(
            self._safe_float(entry.get("duration_months"), default=0.0)
            for entry in valid_entries
        )
        total_years: float = total_months / 12.0

        # Protect juniors — not enough history to judge
        if total_years <= _JOB_HOPPER_MIN_YOE:
            return 1.0

        avg_tenure_yrs: float = total_years / n_roles

        if avg_tenure_yrs < _JOB_HOPPER_MAX_TENURE:
            self._n_job_hoppers += 1
            logger.debug(
                "Rule 5 triggered — job-hopper penalty | "
                "total_yrs=%.2f n_roles=%d avg_tenure=%.2f",
                total_years,
                n_roles,
                avg_tenure_yrs,
            )
            return _JOB_HOPPER_PENALTY

        return 1.0

    # ------------------------------------------------------------------
    # Rule 6 — Open Source / Validation Reward
    # ------------------------------------------------------------------

    def _apply_open_source_reward(
        self,
        redrob_signals: dict,
    ) -> float:
        """
        Reward candidates with meaningful open-source activity as an
        independent quality signal.

        Rationale
        ---------
        The JD calls for "external validation (papers, talks, open-source)".
        ``github_activity_score`` (range -1 to 100) is the strongest proxy
        available in the dataset.  This rule is intentionally *decoupled*
        from Rule 3c (which also looks at ``github_activity_score`` but
        gates on ``verified_email``): a high-activity contributor should be
        rewarded regardless of email verification status.

        Tier Thresholds
        ---------------
        * **High OSS** (score > 50) → ×1.30  —  top-quartile contributor;
          strong evidence of engineering depth and public code review.
        * **Mid OSS** (20 < score ≤ 50) → ×1.10  —  moderate activity;
          meaningful but not exceptional.
        * Below 20 or -1 (no GitHub linked) → no modifier.

        Parameters
        ----------
        redrob_signals : dict
            The ``"redrob_signals"`` sub-dictionary from the candidate record.

        Returns
        -------
        float
            ``1.3``, ``1.1``, or ``1.0``.
        """
        github_score: float = self._safe_float(
            redrob_signals.get("github_activity_score"), default=-1.0
        )

        if github_score > _OSS_HIGH_THRESHOLD:
            self._n_oss_bonuses += 1
            logger.debug(
                "Rule 6a — high OSS bonus | github_score=%.1f",
                github_score,
            )
            return _OSS_HIGH_BONUS

        if github_score > _OSS_MID_THRESHOLD:
            self._n_oss_bonuses += 1
            logger.debug(
                "Rule 6b — mid OSS bonus | github_score=%.1f",
                github_score,
            )
            return _OSS_MID_BONUS

        return 1.0

    # ------------------------------------------------------------------
    # Rule 7 — Domain Mismatch Penalty
    # ------------------------------------------------------------------

    def _apply_domain_mismatch_penalty(self, skills: list[dict]) -> float:
        """
        Penalise candidates whose primary expertise is in an excluded sub-domain
        (e.g., Computer Vision, Speech) with no corresponding advanced skills in
        the required domain (e.g., NLP, Search).
        """
        if not skills:
            return 1.0

        has_mismatched = False
        has_required = False

        for s in skills:
            name: str = str(s.get("name", "")).strip().lower()
            prof: str = str(s.get("proficiency", "")).strip().lower()

            if prof in _HIGH_PROFICIENCIES:
                if name in self._mismatched_domain_skills:
                    has_mismatched = True
                if name in self._required_domain_skills:
                    has_required = True

        if has_mismatched and not has_required:
            self._n_domain_mismatches += 1
            logger.debug("Rule 7 triggered — domain mismatch penalty applied.")
            return _DOMAIN_MISMATCH_PENALTY

        return 1.0

    # ------------------------------------------------------------------
    # Rule 4 — Honeypot Flag
    # ------------------------------------------------------------------

    def _compute_honeypot_flag(
        self,
        career_history:      list[dict],
        years_of_experience: float,
    ) -> bool:
        """
        Flag profiles where computed career tenure diverges from the
        self-reported ``years_of_experience`` by more than 3 years.

        Detection Patterns
        ------------------
        * **Inflated self-report** — candidate claims 15 YoE but career
          entries sum to only 4 years: delta = 11 → ``is_honeypot = True``.
        * **Over-padded history** — candidate lists many short roles to
          inflate apparent experience beyond what they claimed: delta > 3.

        Computation
        -----------
        ``total_months = sum(entry["duration_months"] for entry in career_history)``
        ``computed_years = total_months / 12``
        ``is_honeypot = abs(computed_years − years_of_experience) > 3.0``

        Missing ``"duration_months"`` keys are silently treated as ``0``
        via :py:meth:`_safe_float`, so data gaps skew toward *not* flagging,
        which is the safer default.

        Parameters
        ----------
        career_history : list[dict]
            Sequence of job objects; each may carry a ``"duration_months"``
            numeric key.
        years_of_experience : float
            The candidate's self-reported total years of professional
            experience.

        Returns
        -------
        bool
            ``True`` if the absolute discrepancy exceeds the threshold.
        """
        total_months: float = sum(
            self._safe_float(entry.get("duration_months"), default=0.0)
            for entry in career_history
        )
        computed_years: float = total_months / 12.0
        delta:          float = abs(computed_years - years_of_experience)
        is_honeypot:    bool  = delta > _HONEYPOT_DELTA_THRESHOLD

        if is_honeypot:
            self._n_honeypots += 1
            logger.debug(
                "Rule 4 triggered — honeypot flag | "
                "computed_yrs=%.2f stated_yrs=%.2f delta=%.2f",
                computed_years,
                years_of_experience,
                delta,
            )

        return is_honeypot

    # ------------------------------------------------------------------
    # Core single-record orchestrator
    # ------------------------------------------------------------------

    def _process_candidate(self, record: dict) -> ProcessedRecord | None:
        """
        Orchestrate all four heuristic rules for a single parsed JSON record
        and return a typed :class:`ProcessedRecord` DTO.

        Scoring Flow
        ------------
        1. Extract top-level fields with typed defaults (all via ``.get()``).
        2. Initialise ``behavioral_multiplier = 1.0``.
        3. Apply Rules 1–3 multiplicatively in sequence.
        4. Compute the independent Rule 4 boolean flag.
        5. Return a frozen :class:`ProcessedRecord`.

        Parameters
        ----------
        record : dict
            A single parsed JSON object from the JSONL stream.

        Returns
        -------
        ProcessedRecord | None
            ``None`` when the record lacks a valid ``"candidate_id"`` field,
            making it useless for any downstream join.  The parse-error counter
            is incremented in this case.
        """
        candidate_id: str | None = record.get("candidate_id")
        if not candidate_id:
            self._n_parse_errors += 1
            logger.warning("Skipping record: missing or empty 'candidate_id'.")
            return None

        # ── Extract fields with defensive defaults ────────────────────
        # NOTE: current_title and years_of_experience are nested inside the
        # "profile" sub-dictionary in the actual dataset schema.
        profile:             dict       = record.get("profile")        or {}
        career_history:      list[dict] = record.get("career_history") or []
        skills:              list[dict] = record.get("skills")         or []
        current_title:       str        = str(profile.get("current_title") or "")
        years_of_experience: float      = self._safe_float(
            profile.get("years_of_experience"), default=0.0
        )
        redrob_signals:      dict       = record.get("redrob_signals") or {}

        # ── Apply rules sequentially ──────────────────────────────────
        multiplier: float = 1.0

        multiplier *= self._apply_service_company_penalty(career_history)        # Rule 1
        multiplier *= self._apply_keyword_stuffer_penalty(current_title, skills) # Rule 2
        multiplier *= self._apply_availability_modifiers(redrob_signals)         # Rule 3
        multiplier *= self._apply_job_hopper_penalty(career_history)             # Rule 5
        multiplier *= self._apply_open_source_reward(redrob_signals)             # Rule 6
        multiplier *= self._apply_domain_mismatch_penalty(skills)                # Rule 7

        is_honeypot: bool = self._compute_honeypot_flag(                         # Rule 4
            career_history, years_of_experience
        )

        return ProcessedRecord(
            candidate_id          = str(candidate_id),
            is_honeypot           = is_honeypot,
            behavioral_multiplier = multiplier,
        )

    # ------------------------------------------------------------------
    # Streaming iterator (single I/O entry point)
    # ------------------------------------------------------------------

    def _iter_records(self) -> Iterator[dict]:
        """
        Lazily yield one parsed JSON object per non-blank line.

        This is the *only* point where the file is opened; all other methods
        receive already-parsed dicts.  Keeping I/O strictly contained here
        simplifies testing (callers can swap this iterator for a fixture).

        Error Handling
        --------------
        * Blank lines are silently skipped.
        * ``json.JSONDecodeError`` on a line increments the parse-error
          counter, logs a WARNING, and **continues** — a single bad line
          must not abort a 100 k-row run.

        Yields
        ------
        dict
            One candidate record per iteration.
        """
        with self.input_path.open("r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    self._n_parse_errors += 1
                    logger.warning(
                        "JSON parse error — line %d skipped | %s",
                        line_no,
                        exc,
                    )

    # ------------------------------------------------------------------
    # Batch → Polars DataFrame flush
    # ------------------------------------------------------------------

    def _flush_batch(
        self,
        batch:              list[ProcessedRecord],
        accumulated_frames: list[pl.DataFrame],
    ) -> None:
        """
        Materialise a completed batch of :class:`ProcessedRecord` DTOs into a
        typed Polars DataFrame and append it to *accumulated_frames*.

        Memory Strategy
        ---------------
        Collecting frames in a list and concatenating once at the end
        (in :py:meth:`run`) avoids the repeated full-copy re-allocation that
        would occur if we grew a single mutable DataFrame incrementally.
        At 10 k rows per batch and ~200 bytes of output data per row, each
        frame is roughly 2 MB — the full list of 10 frames for 100 k records
        stays well under 30 MB.

        Schema Enforcement
        ------------------
        The ``schema`` kwarg is passed explicitly so that Polars infers the
        correct types even for edge cases (e.g., a batch where every
        ``is_honeypot`` happens to be ``False``, which Polars might otherwise
        infer as ``Null``).

        Parameters
        ----------
        batch : list[ProcessedRecord]
            Records since the last flush.  No-op if empty.
        accumulated_frames : list[pl.DataFrame]
            Mutable accumulator to which the new frame is appended in-place.
        """
        if not batch:
            return

        frame = pl.DataFrame(
            data={
                "candidate_id":          [r.candidate_id          for r in batch],
                "is_honeypot":           [r.is_honeypot           for r in batch],
                "behavioral_multiplier": [r.behavioral_multiplier for r in batch],
            },
            schema={
                "candidate_id":          pl.String,
                "is_honeypot":           pl.Boolean,
                "behavioral_multiplier": pl.Float32,
            },
        )
        accumulated_frames.append(frame)

        logger.info(
            "Batch flushed | batch_rows=%d | cumulative_processed=%d",
            len(batch),
            self._n_processed,
        )

    # ------------------------------------------------------------------
    # Output validation (QA gate)
    # ------------------------------------------------------------------

    def validate_output(self) -> bool:
        """
        Read back the written Parquet file and verify its schema, column
        names, data types, null counts, and row count.

        Call this *after* :py:meth:`run` has completed successfully.

        Returns
        -------
        bool
            ``True`` when all checks pass; ``False`` otherwise.  Failure
            details are emitted at ``ERROR`` log level so CI pipelines can
            capture them.
        """
        if not self.output_path.exists():
            logger.error(
                "Validation failed: output file not found at %s",
                self.output_path.resolve(),
            )
            return False

        try:
            df = pl.read_parquet(self.output_path)
        except Exception as exc:  # noqa: BLE001 — catch-all intentional for QA gate
            logger.error("Validation failed: could not read Parquet — %s", exc)
            return False

        # ── Column existence check ───────────────────────────────────
        expected_cols: list[str] = [
            "candidate_id",
            "is_honeypot",
            "behavioral_multiplier",
        ]
        if df.columns != expected_cols:
            logger.error(
                "Column mismatch | expected=%s | got=%s",
                expected_cols,
                df.columns,
            )
            return False

        # ── Dtype check ──────────────────────────────────────────────
        dtype_checks: list[tuple[str, pl.PolarsDataType]] = [
            ("candidate_id",          pl.String),
            ("is_honeypot",           pl.Boolean),
            ("behavioral_multiplier", pl.Float32),
        ]
        for col_name, expected_dtype in dtype_checks:
            actual_dtype = df[col_name].dtype
            if actual_dtype != expected_dtype:
                logger.error(
                    "Dtype mismatch | column=%r expected=%s got=%s",
                    col_name,
                    expected_dtype,
                    actual_dtype,
                )
                return False

        # ── Null check ───────────────────────────────────────────────
        total_nulls: int = sum(df.null_count().row(0))
        if total_nulls != 0:
            logger.error(
                "Unexpected nulls in output | null_counts=%s",
                df.null_count(),
            )
            return False

        # ── Row count cross-check ────────────────────────────────────
        if df.height != self._n_processed:
            logger.error(
                "Row count mismatch | processed=%d parquet_rows=%d",
                self._n_processed,
                df.height,
            )
            return False

        logger.info(
            "Output validation passed | rows=%d | schema=%s",
            df.height,
            df.schema,
        )
        return True

    # ------------------------------------------------------------------
    # Public pipeline entry-point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the full streaming pipeline end-to-end.

        Pipeline Steps
        --------------
        1. Validate input file exists and reset telemetry counters.
        2. Open the JSONL file and stream records lazily via
           :py:meth:`_iter_records`.
        3. Pass each record through :py:meth:`_process_candidate` to apply
           all four heuristic rules.
        4. Accumulate :class:`ProcessedRecord` DTOs in a rolling batch list.
        5. When the batch reaches ``self.batch_size``, flush to a typed Polars
           DataFrame via :py:meth:`_flush_batch` and clear the batch
           (releasing references so the GC can reclaim memory).
        6. After the stream is exhausted, flush any remaining partial batch.
        7. Concatenate all accumulated frames (``rechunk=True`` for read-time
           performance) and write the Parquet file atomically via a tmp→rename
           pattern.

        Raises
        ------
        FileNotFoundError
            If ``self.input_path`` does not exist.
        RuntimeError
            If zero valid records were produced (prevents writing an empty
            output that would silently mislead downstream consumers).
        """
        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {self.input_path.resolve()}"
            )

        # Reset telemetry — supports re-running the same instance in tests
        (
            self._n_processed,
            self._n_honeypots,
            self._n_stuffers,
            self._n_service_penalty,
            self._n_job_hoppers,
            self._n_oss_bonuses,
            self._n_domain_mismatches,
            self._n_parse_errors,
        ) = (0, 0, 0, 0, 0, 0, 0, 0)

        start_ts: float = time.perf_counter()
        logger.info("Pipeline started | input=%s", self.input_path.resolve())

        batch:              list[ProcessedRecord] = []
        accumulated_frames: list[pl.DataFrame]   = []

        for record in self._iter_records():
            result = self._process_candidate(record)
            if result is None:
                continue

            batch.append(result)
            self._n_processed += 1

            if len(batch) >= self.batch_size:
                self._flush_batch(batch, accumulated_frames)
                batch.clear()   # drop references → GC reclaims batch memory

        # Flush the final (possibly partial) batch
        if batch:
            self._flush_batch(batch, accumulated_frames)
            batch.clear()

        if not accumulated_frames:
            raise RuntimeError(
                "Zero valid records were produced.  "
                "Verify the input file and check the logs for parse errors."
            )

        # ── Concatenate and write ─────────────────────────────────────
        logger.info("Concatenating %d Polars frame(s) …", len(accumulated_frames))
        final_df: pl.DataFrame = pl.concat(accumulated_frames, rechunk=True)

        # Atomic write: write to *.tmp then rename into place
        tmp_path = self.output_path.with_suffix(".parquet.tmp")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Writing %d rows → %s  (compression=zstd) …",
            final_df.height,
            self.output_path,
        )
        final_df.write_parquet(
            tmp_path,
            compression="zstd",
            statistics=True,        # enables Parquet column statistics for predicate push-down
        )
        tmp_path.replace(self.output_path)  # atomic overwrite on both POSIX and Windows

        elapsed: float = time.perf_counter() - start_ts
        logger.info(
            "Pipeline complete in %.2f s | "
            "rows=%d | honeypots=%d | stuffers=%d | "
            "service_penalties=%d | job_hoppers=%d | "
            "oss_bonuses=%d | domain_mismatches=%d | parse_errors=%d",
            elapsed,
            self._n_processed,
            self._n_honeypots,
            self._n_stuffers,
            self._n_service_penalty,
            self._n_job_hoppers,
            self._n_oss_bonuses,
            self._n_domain_mismatches,
            self._n_parse_errors,
        )

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """
        Print a formatted post-run summary table to ``stdout``.

        Must be called *after* :py:meth:`run` has completed.  The numbers
        reflect the most recent invocation (counters are reset at run-start).
        """
        sep = "-" * 56
        rows: list[tuple[str, int]] = [
            ("Total records processed",        self._n_processed),
            ("  > Honeypot flags set",          self._n_honeypots),
            ("  > Keyword stuffers penalised",  self._n_stuffers),
            ("  > Service-company penalised",   self._n_service_penalty),
            ("  > Job hoppers penalised",       self._n_job_hoppers),
            ("  > OSS bonuses applied",         self._n_oss_bonuses),
            ("  > Domain mismatches penalised", self._n_domain_mismatches),
            ("Parse / skip errors",             self._n_parse_errors),
        ]
        print(f"\n  {sep}")
        print("  HeuristicTrapFilter -- Post-Run Summary")
        print(f"  {sep}")
        for label, count in rows:
            print(f"  {label:<40} {count:>10,}")
        print(f"  {sep}")
        print(f"  Output -> {self.output_path.resolve()}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Construct and return the argument parser for the CLI entry-point.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="heuristic_trap_filter",
        description=(
            "Stream a JSONL candidate file, apply four categories of heuristic "
            "business rules, and write pre-computed scores to Parquet."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        default="candidates.jsonl",
        metavar="PATH",
        help="Path to the source JSONL file.",
    )
    parser.add_argument(
        "--output", "-o",
        default="precomputed_heuristics.parquet",
        metavar="PATH",
        help="Destination path for the output Parquet file.",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=10_000,
        metavar="N",
        help="Records per in-memory processing batch.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Read back the output Parquet and validate its schema after writing.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def main() -> None:
    """
    Parse CLI arguments, configure logging, and run the pipeline.

    Exits with code 1 if ``--validate`` is requested and validation fails,
    so CI pipelines can detect a corrupt output without inspecting logs.
    """
    args = _build_arg_parser().parse_args()
    logging.getLogger().setLevel(args.log_level)

    pipeline = HeuristicTrapFilter(
        input_path  = args.input,
        output_path = args.output,
        batch_size  = args.batch_size,
    )

    pipeline.run()
    pipeline.print_summary()

    if args.validate:
        ok = pipeline.validate_output()
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()