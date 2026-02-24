"""
labeler.py — LabelingAgent

Reads preprocessed JSON files from data/preprocessed/.
For each chunk flagged has_forex=True, calls OpenAI to extract forex signals.

Output layout (mirrors preprocessing):
  data/labeled/<company>_<quarter>_labels.json   ← one file per document
  data/labeled/training_data.json                ← full aggregate (all docs combined)

One label entry per signal extracted; a single chunk can yield multiple entries
if multiple currency pairs are discussed.

Usage via run.py:
    python run.py label                            # Label all preprocessed docs
    python run.py label --limit 3                  # First 3 docs only
    python run.py label --skip 5                   # Skip first 5, label rest
    python run.py label --skip 5 --limit 3         # Skip 5, then next 3
    python run.py label --pattern sony             # Only docs whose filename contains 'sony'
    python run.py label --pattern sony --limit 2   # First 2 sony docs
    python run.py label --no-resume                # Re-label everything from scratch
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from openai import OpenAI

from config import (
    LABELED_DIR,
    LABELED_OUTPUT,
    PREPROCESSED_DIR,
    OPENAI_MODEL,
    MAX_TOKENS,
)
from models import Label


#  Logging setup ─
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def _elapsed(start: float) -> str:
    return f"{time.time() - start:.1f}s"


#  System prompt ─
_SYSTEM_PROMPT = """You are a financial analyst specialising in foreign exchange (FX) markets.
Your task is to read an excerpt from an earnings call transcript and extract ALL forex trading signals present.

For EACH distinct currency pair / signal in the text, output one JSON object in the array.
If no actionable signal exists, output a single object with signal=false.

Each JSON object must follow this exact schema:
{
  "signal": true | false,
  "currency_pair": "EUR/USD" | "USD/JPY" | ... | null,
  "direction": "LONG" | "SHORT" | "NEUTRAL" | null,
  "confidence": 0.0–1.0 | null,
  "reasoning": "string",
  "magnitude": "low" | "moderate" | "high" | null,
  "time_horizon": "current_quarter" | "next_quarter" | "long_term" | null
}

Return ONLY a valid JSON array — no markdown, no explanation outside the array."""

_USER_TEMPLATE = """Company: {company} ({ticker})
Quarter: {quarter}
Speaker: {speaker}
Section: {section}

Transcript excerpt:
\"\"\"{text}\"\"\"

Extract all forex signals as a JSON array."""


class LabelingAgent:
    """
    Reads preprocessed JSONs, filters forex chunks,
    calls OpenAI, and writes per-doc label files + a combined aggregate.
    """

    def __init__(self, api_key: Optional[str] = None):
        import os
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Add it to your .env file."
            )
        self.client = OpenAI(api_key=key)

    #  Call OpenAI ─
    def _call_openai(self, user_msg: str) -> list[dict]:
        """Call OpenAI and return a list of signal dicts."""
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        return self._parse_response(raw)

    @staticmethod
    def _parse_response(raw: str) -> list[dict]:
        """Robustly parse OpenAI JSON array response."""
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        try:
            data = json.loads(cleaned)
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
        log.warning("    Could not parse OpenAI response: %s", cleaned[:120])
        return []

    #  Build Label objects ─
    @staticmethod
    def _build_labels(
        signals:    list[dict],
        chunk_text: str,
        meta:       dict,
        chunk_id:   str,
        speaker:    Optional[str],
        section:    Optional[str],
    ) -> list[Label]:
        now = datetime.now(timezone.utc).isoformat()
        labels = []
        for sig in signals:
            labels.append(Label(
                input=chunk_text,
                signal=bool(sig.get("signal", False)),
                currency_pair=sig.get("currency_pair"),
                direction=sig.get("direction"),
                confidence=sig.get("confidence"),
                reasoning=sig.get("reasoning", ""),
                magnitude=sig.get("magnitude"),
                time_horizon=sig.get("time_horizon"),
                company=meta["company_name"],
                ticker=meta["ticker"],
                earnings_date=meta["earnings_date"],
                quarter=meta["quarter"],
                speaker=speaker,
                section=section,
                chunk_id=chunk_id,
                labeled_at=now,
            ))
        return labels

    #  Per-doc output path ─
    @staticmethod
    def _doc_label_path(meta: dict, labeled_dir: Path) -> Path:
        """
        Derive a per-document label filename from company + quarter metadata,
        mirroring the convention used by PreprocessingAgent.save():
            sony_q4_2024_labels.json
        """
        safe_company = re.sub(r"[^\w]", "_", meta["company_name"].lower())
        safe_quarter = re.sub(r"\s+", "_", meta["quarter"].lower())
        return labeled_dir / f"{safe_company}_{safe_quarter}_labels.json"

    #  Process one preprocessed JSON file ─
    def label_document(self, json_path: Path, labeled_dir: Path) -> tuple[list[Label], Path]:
        """
        Label all forex chunks in one preprocessed JSON.

        Returns (labels, output_path) where output_path is the per-doc label file.
        The per-doc file is written immediately after all chunks are processed.
        """
        t0 = time.time()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        meta = data["metadata"]
        out_path = self._doc_label_path(meta, labeled_dir)

        forex_chunks = [c for c in data["chunks"] if c.get("has_forex")]
        total_chunks = len(data["chunks"])
        log.info(
            "  Forex chunks: %d / %d  (skipping %d non-forex)",
            len(forex_chunks), total_chunks, total_chunks - len(forex_chunks),
        )

        all_labels: list[Label] = []

        for ci, chunk in enumerate(forex_chunks, 1):
            chunk_id = chunk["id"]
            log.info(
                "  [%d/%d] Calling OpenAI for %s (speaker=%s, section=%s, ~%d tokens)...",
                ci, len(forex_chunks),
                chunk_id,
                chunk.get("speaker") or "Unknown",
                chunk.get("section") or "Unknown",
                chunk.get("token_count", 0),
            )
            t_chunk = time.time()

            user_msg = _USER_TEMPLATE.format(
                company=meta["company_name"],
                ticker=meta["ticker"],
                quarter=meta["quarter"],
                speaker=chunk.get("speaker") or "Unknown",
                section=chunk.get("section") or "Unknown",
                text=chunk["text"],
            )

            try:
                signals = self._call_openai(user_msg)
                labels = self._build_labels(
                    signals,
                    chunk_text=chunk["text"],
                    meta=meta,
                    chunk_id=chunk_id,
                    speaker=chunk.get("speaker"),
                    section=chunk.get("section"),
                )
                all_labels.extend(labels)

                signal_count = sum(1 for lbl in labels if lbl.signal)
                log.info(
                    "    -> %d label(s) extracted (%d with signal=true)  (%s)",
                    len(labels), signal_count, _elapsed(t_chunk),
                )
                time.sleep(0.3)   # polite rate-limit buffer

            except Exception as e:
                log.error("    ✗ Error on %s: %s", chunk_id, e)

        # Write per-doc file immediately
        out_path.write_text(
            json.dumps([lbl.to_dict() for lbl in all_labels],
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        true_signals = sum(1 for lbl in all_labels if lbl.signal)
        log.info(
            "  Saved %d label(s) [%d signal, %d no-signal] -> %s  (%s)",
            len(all_labels), true_signals, len(all_labels) - true_signals,
            out_path.name, _elapsed(t0),
        )
        return all_labels, out_path

    #  Rebuild aggregate from all per-doc files
    @staticmethod
    def _rebuild_aggregate(labeled_dir: Path, output_path: Path) -> int:
        """
        Scan labeled_dir for *_labels.json files and merge them into
        training_data.json.  Returns total label count.
        """
        all_labels: list[dict] = []
        for lf in sorted(labeled_dir.glob("*_labels.json")):
            try:
                entries = json.loads(lf.read_text(encoding="utf-8"))
                all_labels.extend(entries)
            except Exception as e:
                log.warning("  Could not read %s: %s", lf.name, e)
        output_path.write_text(
            json.dumps(all_labels, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return len(all_labels)

    #  Batch run ─
    def run_all(
        self,
        preprocessed_dir: Path = PREPROCESSED_DIR,
        labeled_dir:      Path = LABELED_DIR,
        output_path:      Path = LABELED_OUTPUT,
        resume:           bool = True,
        limit:            int | None = None,
        skip:             int | None = None,
        pattern:          str | None = None,
    ) -> Path:
        """
        Label all preprocessed JSONs and write per-doc label files +
        a combined training_data.json.

        Args:
            resume:   Skip docs that already have a per-doc label file.
            limit:    Process only the first N docs (after skip/pattern filtering).
            skip:     Skip the first N files in sorted order.
            pattern:  Only process files whose name contains this string
                      (case-insensitive).  E.g. 'sony', 'apple'.
        """
        batch_start = time.time()

        json_files = sorted(preprocessed_dir.glob("*.json"))
        if not json_files:
            log.warning("No preprocessed JSON files found in %s",
                        preprocessed_dir)
            return output_path

        #  Filter: --pattern ─
        if pattern:
            json_files = [p for p in json_files if pattern.lower()
                          in p.name.lower()]
            log.info("--pattern '%s': %d matching file(s)",
                     pattern, len(json_files))
            if not json_files:
                log.warning("No files matched pattern '%s'", pattern)
                return output_path

        #  Filter: --skip
        if skip:
            json_files = json_files[skip:]
            log.info(
                "--skip %d: starting from file %d onward (%d remaining)",
                skip, skip + 1, len(json_files),
            )

        #  Filter: --limit ─
        if limit:
            json_files = json_files[:limit]
            log.info("--limit %d: processing %d file(s)",
                     limit, len(json_files))

        log.info("Found %d preprocessed file(s) to label", len(json_files))

        total_labels = 0

        for i, jf in enumerate(json_files, 1):
            log.info("")
            log.info("=" * 60)
            log.info("[%d/%d] %s", i, len(json_files), jf.name)
            log.info("=" * 60)

            #  Resume: check if per-doc file already exists
            if resume:
                try:
                    preview = json.loads(jf.read_text(encoding="utf-8"))
                    doc_out = self._doc_label_path(
                        preview["metadata"], labeled_dir)
                    if doc_out.exists():
                        existing_count = len(json.loads(
                            doc_out.read_text(encoding="utf-8")))
                        log.info(
                            "  Skipping (already labeled — %d labels in %s)",
                            existing_count, doc_out.name,
                        )
                        total_labels += existing_count
                        continue
                except Exception as e:
                    log.warning("  Could not check resume status: %s", e)

            try:
                labels, out_path = self.label_document(jf, labeled_dir)
                total_labels += len(labels)
                log.info("  -> %d label(s) this document", len(labels))
            except Exception as e:
                log.error("  ✗ Document-level error: %s", e, exc_info=True)

        # Rebuild aggregate training_data.json
        log.info("")
        log.info("Rebuilding aggregate -> %s", output_path.name)
        agg_count = self._rebuild_aggregate(labeled_dir, output_path)
        log.info("Aggregate: %d total label(s) across all docs", agg_count)

        log.info("")
        log.info("*" * 60)
        log.info(
            "Labeling complete: %d label(s) in this run | Total in aggregate: %d | %s",
            total_labels, agg_count, _elapsed(batch_start),
        )
        log.info("*" * 60)
        return output_path
