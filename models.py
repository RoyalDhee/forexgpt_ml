"""
models.py — Dataclasses that represent pipeline data structures.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    id: str
    text: str
    # "CFO", "CEO", "Analyst", "Operator", "Unknown"
    speaker: Optional[str] = None
    section: Optional[str] = None    # "prepared_remarks" | "qa_session"
    token_count: int = 0
    has_forex: bool = False          # True if chunk passed forex keyword filter


@dataclass
class TranscriptDoc:
    """Everything produced by the PreprocessingAgent for one PDF."""

    # Metadata
    company_name: str
    ticker: str
    earnings_date: str               # ISO-8601 date string or "" if unknown
    quarter: str                     # e.g. "Q2 2024"
    source: str = "Earnings Call"
    preprocessed_at: str = ""

    # Text
    raw_text: str = ""
    cleaned_text: str = ""

    # Speaker statements
    speakers: dict[str, list[str]] = field(default_factory=dict)

    #  Sections
    sections: dict[str, str] = field(default_factory=lambda: {
        "prepared_remarks": "",
        "qa_session": "",
    })

    #  Chunks (ALL chunks, forex-flagged or not)
    chunks: list[Chunk] = field(default_factory=list)

    #  Stats
    preprocessing_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "metadata": {
                "company_name":    self.company_name,
                "ticker":          self.ticker,
                "earnings_date":   self.earnings_date,
                "quarter":         self.quarter,
                "source":          self.source,
                "preprocessed_at": self.preprocessed_at,
            },
            "raw_text":    self.raw_text,
            "cleaned_text": self.cleaned_text,
            "speakers":    self.speakers,
            "sections":    self.sections,
            "chunks": [
                {
                    "id":          c.id,
                    "text":        c.text,
                    "speaker":     c.speaker,
                    "section":     c.section,
                    "token_count": c.token_count,
                    "has_forex":   c.has_forex,
                }
                for c in self.chunks
            ],
            "preprocessing_stats": self.preprocessing_stats,
        }


@dataclass
class Label:
    """One forex signal label extracted from a single chunk."""

    #  Input (the exact chunk text that produced the signal) ─
    input: str

    #  Output
    signal: bool
    currency_pair: Optional[str]
    direction: Optional[str]         # "LONG" | "SHORT" | "NEUTRAL" | null
    confidence: Optional[float]
    reasoning: str
    magnitude: Optional[str]         # "low" | "moderate" | "high" | null
    # "current_quarter" | "next_quarter" | "long_term" | null
    time_horizon: Optional[str]

    #  Metadata
    company: str
    ticker: str
    earnings_date: str
    quarter: str
    speaker: Optional[str]
    section: Optional[str]
    chunk_id: str
    labeled_at: str
    labeler: str = "openai-gpt4o"

    def to_dict(self) -> dict:
        return {
            "input": self.input,
            "output": {
                "signal":        self.signal,
                "currency_pair": self.currency_pair,
                "direction":     self.direction,
                "confidence":    self.confidence,
                "reasoning":     self.reasoning,
                "magnitude":     self.magnitude,
                "time_horizon":  self.time_horizon,
            },
            "metadata": {
                "company":       self.company,
                "ticker":        self.ticker,
                "earnings_date": self.earnings_date,
                "quarter":       self.quarter,
                "speaker":       self.speaker,
                "section":       self.section,
                "chunk_id":      self.chunk_id,
                "labeler":       self.labeler,
                "labeled_at":    self.labeled_at,
            },
        }
