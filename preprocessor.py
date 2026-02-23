"""
preprocessor.py — PreprocessingAgent

Reads raw PDF transcripts -> produces structured TranscriptDoc objects
and writes them as JSON to data/preprocessed/.

Handles four distinct transcript formats:
  1. Alphabet/official inline format:  "Name, Title: speech text"
  2. FactSet/CallStreet format:         Name on own line, title below, dotted separators
  3. Microsoft official format:         "FIRSTNAME LASTNAME, Firm:" or "Firstname Lastname:"
  4. Seeking Alpha (image-based PDFs):  flagged and skipped with clear warning

Filename convention (separator is _ or - or mixed, case-insensitive):
    suncor_energy_q2_2024_ect.pdf  |  MICROSOFT_Q1_2024_ECT.PDF  |  shell-q1-2024-ect.pdf
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pypdf import PdfReader
try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    _PDFPLUMBER_AVAILABLE = False

from config import (
    PREPROCESSED_DIR,
    TRANSCRIPTS_DIR,
    FOREX_KEYWORDS,
    CHUNK_TOKENS,
    OVERLAP_TOKENS,
    TICKER_MAP,
)
from models import Chunk, TranscriptDoc

#  Logging setup
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


#  Quarter -> approximate earnings date heuristic
_QUARTER_MONTH = {"Q1": "04", "Q2": "07", "Q3": "10", "Q4": "01"}


# =============================================================================
class FilenameParser:
    """
    Extracts (company_name, quarter, year) from an earnings transcript filename.
    Separator between parts is _ or - (or a mix), case-insensitive.
    Company name may be one or more separated words.
    """

    _PATTERN = re.compile(
        r"^(?P<company>.+?)[_-](Q[1-4])[_-](\d{4})[_-]ect$",
        re.IGNORECASE,
    )

    @classmethod
    def parse(cls, path: Path) -> tuple[str, str, str]:
        stem = path.stem
        m = cls._PATTERN.match(stem)
        if not m:
            fallback = re.sub(r"[_-]+", " ", stem).strip().title()
            log.warning(
                "Filename did not match pattern: %s — using '%s'", path.name, fallback)
            return fallback, "Unknown", "Unknown"
        raw_company = m.group("company")
        quarter = m.group(2).upper()
        year = m.group(3)
        company_name = re.sub(r"[_-]+", " ", raw_company).strip().title()
        return company_name, quarter, year


# =============================================================================
class TickerResolver:
    """Looks up ticker from company name using TICKER_MAP (longest-key-first).

    Tries multiple normalised variants of the company name so that filenames
    like 'CocaCola' still match TICKER_MAP keys like 'coca-cola' and 'coca cola'.
    """

    @staticmethod
    def resolve(company_name: str) -> str:
        import re as _re
        lower = company_name.lower()

        # Variant 1: plain lowercase  ('cocacola')
        # Variant 2: insert space before caps following lowercase  ('coca Cola' -> 'coca cola')
        spaced = _re.sub(r'([a-z])([A-Z])', r'\1 \2', company_name).lower()
        # Variant 3: hyphens become spaces  ('coca-cola' -> 'coca cola')
        hyphen_spaced = lower.replace("-", " ")
        # Variant 4: underscores become spaces
        under_spaced = lower.replace("_", " ")

        candidates = list(dict.fromkeys(
            [lower, spaced, hyphen_spaced, under_spaced]))

        for key in sorted(TICKER_MAP, key=len, reverse=True):
            # Also try the key without hyphens/spaces to broaden matching
            key_plain = key.replace("-", "").replace(" ", "")
            for candidate in candidates:
                candidate_plain = candidate.replace("-", "").replace(" ", "")
                if key in candidate or key_plain in candidate_plain:
                    return TICKER_MAP[key]

        log.warning("No ticker found for: '%s'", company_name)
        return "UNKNOWN"


# =============================================================================
class PDFExtractor:
    """
    PDF text extraction with automatic image-PDF detection and OCR fallback.

    Strategy 1 — pypdf (no timeout, no threads)
        Extracts text from native text-layer PDFs. Fast for text PDFs (1-3s).
        For image PDFs pypdf also runs slowly (~0.6s/page) but returns 0 chars.

    Detection — 3-page sampling heuristic (runs BEFORE full extraction)
        Sample the first 3 pages: time how long they take and count chars.
        Text PDFs: fast (<0.1s/page) and produce chars → proceed with full extract.
        Image PDFs: slow (>0.25s/page) and produce 0 chars → go straight to OCR.
        This correctly handles large text PDFs (e.g. Alphabet, 100+ pages) that
        would have been wrongly killed by a fixed timeout.

    Strategy 2 — OCR via pytesseract + pdf2image
        Converts PDF pages to 200dpi images and runs Tesseract OCR.
        Used for image-based PDFs (SA screenshots, scanned docs).
        ~2-5s per page — used once per document, result is cached in JSON.

    Returns (raw_text, is_image_based).
    """

    # Seconds per page threshold: slower than this AND 0 chars = image PDF
    _IMAGE_SPP_THRESHOLD = 0.25

    @classmethod
    def _sample_pages(cls, reader: "PdfReader", n: int = 3) -> tuple[int, float]:
        """
        Extract text from first n pages, return (total_chars, seconds_per_page).
        Used to distinguish image PDFs from large text PDFs without a fixed timeout.
        """
        t0 = time.time()
        chars = 0
        pages_checked = 0
        for page in reader.pages[:n]:
            chars += len(page.extract_text() or "")
            pages_checked += 1
        elapsed = time.time() - t0
        spp = elapsed / max(pages_checked, 1)
        return chars, spp

    # Number of parallel threads for OCR.
    # Each Tesseract process is CPU-bound; 4 workers typically 3-4x faster
    # than sequential on a quad-core machine. Raise to 6-8 on more cores.
    _OCR_WORKERS = 4
    _OCR_DPI = 150   # 150dpi is sufficient for printed transcript text; 200dpi adds ~44% cost

    @classmethod
    def _ocr_page(cls, args: tuple) -> tuple[int, str]:
        """OCR a single page image. Returns (page_index, text). Called in thread pool."""
        import pytesseract
        idx, img = args
        try:
            text = pytesseract.image_to_string(
                img, lang="eng", config="--psm 1")
            return idx, text
        except Exception as e:
            log.warning("    OCR page %d failed: %s", idx + 1, e)
            return idx, ""

    @classmethod
    def _ocr_pdf(cls, path: Path, total_pages: int) -> str:
        """
        OCR all pages in parallel using a thread pool.

        Speed improvements over the original sequential version:
          - DPI reduced 200 -> 150 (44% smaller images, proportionally faster)
          - Pages processed in parallel (_OCR_WORKERS threads)
          - Combined effect: typically 3-4x faster (5-6 min -> 90-120s for 23 pages)
        """
        try:
            from pdf2image import convert_from_path
            import pytesseract
        except ImportError as e:
            log.error("    OCR dependencies missing: %s", e)
            return ""

        log.info("    Starting OCR (%d pages @ %ddpi, %d parallel workers)...",
                 total_pages, cls._OCR_DPI, cls._OCR_WORKERS)
        try:
            images = convert_from_path(str(path), dpi=cls._OCR_DPI)
        except Exception as e:
            log.error("    pdf2image failed: %s", e)
            return ""

        t_start = time.time()
        n = len(images)
        results: list[tuple[int, str]] = []

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=cls._OCR_WORKERS) as pool:
            futures = {pool.submit(cls._ocr_page, (i, img)): i
                       for i, img in enumerate(images)}
            done_count = 0
            for future in concurrent.futures.as_completed(futures):
                done_count += 1
                idx, text = future.result()
                results.append((idx, text))
                if done_count % 5 == 0 or done_count == n:
                    log.info("    OCR: %d / %d pages done  (%s)",
                             done_count, n, _elapsed(t_start))

        # Sort by original page order
        results.sort(key=lambda x: x[0])
        pages_text = [text for _, text in results]

        combined = "\n".join(pages_text).strip()
        log.info("    OCR complete: %d chars extracted  (%s)",
                 len(combined), _elapsed(t_start))
        return combined

    @classmethod
    def extract(cls, path: Path) -> tuple[str, bool]:
        """
        Returns (text, is_image_based).

        Flow:
          1. Open PDF, sample first 3 pages (fast).
          2. If sample has chars → full pypdf extract → done.
          3. If sample is slow+empty → image PDF → run OCR → done.
          4. If sample is fast+empty → blank/corrupt → run OCR → done.
        """
        try:
            reader = PdfReader(str(path))
            total_pages = len(reader.pages)
            log.info("    PDF opened: %d pages", total_pages)
        except Exception as e:
            log.warning("    Cannot open PDF: %s", e)
            return "", True

        #  Step 1: 3-page sample to classify PDF type ────────────────────────
        log.info("    Sampling first 3 pages to detect PDF type...")
        sample_chars, spp = cls._sample_pages(reader, n=3)
        log.info("    Sample: %d chars, %.3fs/page", sample_chars, spp)

        # ── Step 2: Text PDF → full pypdf extraction ──────────────────────────
        if sample_chars > 0:
            log.info("    Text layer confirmed — running full pypdf extraction...")
            pages_text: list[str] = []
            for i, page in enumerate(reader.pages, 1):
                if i % 10 == 0 or i == total_pages:
                    log.info("    Extracting page %d / %d", i, total_pages)
                pages_text.append(page.extract_text() or "")
            combined = "\n".join(pages_text).strip()
            if combined:
                log.info("    pypdf extracted %d chars", len(combined))
                return combined, False

        # ── Step 3 & 4: No text → run OCR ────────────────────────────────────
        is_image = spp > cls._IMAGE_SPP_THRESHOLD
        if is_image:
            log.info("    Image-based PDF detected (%.3fs/page > %.2f threshold) — running OCR",
                     spp, cls._IMAGE_SPP_THRESHOLD)
        else:
            log.info("    Blank/empty pages detected — attempting OCR anyway")

        ocr_text = cls._ocr_pdf(path, total_pages)
        if ocr_text:
            return ocr_text, True   # is_image_based=True so format detector uses SA logic
        else:
            log.warning("    OCR produced no text — document unprocessable")
            return "", True


# =============================================================================
class FormatDetector:
    """
    Detects which of the known transcript formats a document uses.

    Formats:
      "factset"    — FactSet/CallStreet: dotted separator lines between speakers
      "msft"       — Microsoft official: ALLCAPS NAME, Firm: or Firstname Lastname: inline
      "inline"     — Alphabet/generic: "Name, Title: speech" inline
      "unknown"    — fallback, treated as inline
    """

    # FactSet uses long dotted separator lines between speaker turns
    _FACTSET_DOTS = re.compile(r"\.{20,}")

    # Microsoft format: ALLCAPS NAME, optional firm in title case, then colon
    # e.g. "KEITH WEISS, Morgan Stanley:" or "SATYA NADELLA:"  or "AMY HOOD:"
    _MSFT_ALLCAPS = re.compile(
        r"(?:^|\n)[A-Z]{2,}(?:\s+[A-Z]{2,}){0,4}"   # 1-5 ALLCAPS words
        r"(?:,\s*[A-Za-z\s]+)?"                        # optional ", Firm Name"
        r"\s*:",                                        # colon
        re.MULTILINE,
    )

    # Inline format: "Name Surname, Title: text" — already collapses well
    _INLINE_COLON = re.compile(
        r"[A-Z][a-z]+(?:\s+[A-Z][a-z\'\-]+){0,4}"
        r"(?:,\s*[A-Za-z &/()\-]{2,60})?"
        r"\s*:\s+[A-Z]"
    )

    # Seeking Alpha text-based: SA header in first 1000 chars
    _SA_HEADER = re.compile(
        r"SA Transcripts|Seeking Alpha|seekingalpha\.com",
        re.IGNORECASE,
    )

    @classmethod
    def detect(cls, raw_text: str, force_sa: bool = False) -> str:
        # OCR'd image PDFs are always SA format (all our image PDFs are SA screenshots)
        if force_sa:
            log.info("    Format detected: Seeking Alpha (OCR/image-based)")
            return "seeking_alpha"

        # FactSet detection: dotted lines are a definitive marker
        if cls._FACTSET_DOTS.search(raw_text):
            log.info("    Format detected: FactSet/CallStreet")
            return "factset"

        # Microsoft: several ALLCAPS speaker lines
        msft_hits = cls._MSFT_ALLCAPS.findall(raw_text)
        if len(msft_hits) >= 3:
            log.info(
                "    Format detected: Microsoft official (%d ALLCAPS markers)", len(msft_hits))
            return "msft"

        # Seeking Alpha text-based: SA header near top of document
        if cls._SA_HEADER.search(raw_text[:1500]):
            log.info("    Format detected: Seeking Alpha (text-based)")
            return "seeking_alpha"

        # Default to inline (Alphabet / generic official)
        log.info("    Format detected: inline (Alphabet/generic)")
        return "inline"


# =============================================================================
class TextCleaner:
    """
    Cleans raw PDF-extracted text.

    The cleaning strategy depends on format:
      - "inline": collapse ALL newlines into spaces so "Name, Title: speech"
        can be matched as one continuous string by the regex.
      - "factset" / "msft": preserve single newlines as structural markers
        (speaker names live on their own lines in these formats), but still
        collapse consecutive blank lines and normalise whitespace.
    """

    _REPLACEMENTS = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }

    @classmethod
    def clean(cls, text: str, fmt: str = "inline") -> str:
        text = unicodedata.normalize("NFC", text)
        for s, d in cls._REPLACEMENTS.items():
            text = text.replace(s, d)
        # Strip HTML tags (rare but present in some SA-derived docs)
        text = re.sub(r"<[^>]+>", " ", text)

        if fmt == "inline":
            # Collapse every newline to space — makes inline speaker regex work
            text = re.sub(r"\n+", " ", text)
            text = re.sub(r" {2,}", " ", text)
        else:
            # seeking_alpha / factset / msft: preserve newlines as structure
            # normalise horizontal space
            text = re.sub(r"[ \t]+", " ", text)
            # trim spaces around newlines
            text = re.sub(r" *\n *", "\n", text)
            # max 2 consecutive newlines
            text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()


# =============================================================================
class SpeakerIdentifier:
    """
    Segments transcript text into per-speaker blocks.

    Supports four detection strategies selected by the detected format:

    Strategy A — INLINE (Alphabet/generic)
        Works on a single flat string (newlines already collapsed).
        Anchors on sentence-ending punctuation OR start-of-string.
        Pattern: "Name [, Title]: speech"

    Strategy B — FACTSET (FactSet/CallStreet)
        Works on newline-preserved text.
        Speaker blocks are separated by long dotted lines (.....).
        Speaker name is a Title Case line; next line is italic title (ignored).
        "Operator:" inline is also caught.

    Strategy C — MSFT (Microsoft official)
        Works on newline-preserved text.
        Speakers appear as "FIRSTNAME LASTNAME, Firm:  speech" (ALLCAPS name)
        OR "Firstname Lastname:  speech" (Title Case, exec-style).

    Strategy D — SEEKING ALPHA (text-based SA transcripts)
        Works on newline-preserved text.
        Speaker name appears on its OWN LINE (Title Case, 2-5 words).
        Speech text follows on the next line(s).
        NO dotted separators, NO title line below name.
        Participant list at top ("Name - Title") is explicitly skipped.
        SA header noise lines are stripped before parsing.
    """

    _ROLE_MAP = [
        (re.compile(r"\b(chief executive|ceo|president & ceo|president/ceo|chairman.*ceo)\b", re.I), "CEO"),
        (re.compile(r"\b(chief financial|cfo|chief finance|finance department|finance division|group finance|accounting.*chief|chief.*accounting|treasurer)\b", re.I), "CFO"),
        (re.compile(r"\b(chief operating|coo)\b", re.I), "COO"),
        (re.compile(r"\b(svp|evp|senior vice president|executive vice president|cbo|cmo|cto|general counsel|deputy general|chief communications|communications officer)\b", re.I), "Executive"),
        (re.compile(r"\b(analyst|equity research|research analyst|securities|capital markets?|asset management)\b", re.I), "Analyst"),
        (re.compile(r"\b(investor relation|director.*investor|ir director|vice president.*ir|vp.*investor|vice president.*investor|head.*ir|head.*investor|investor.*relations)\b", re.I), "IR"),
        (re.compile(r"\boperator\b", re.I), "Operator"),
    ]

    # ── Strategy A: inline speaker regex ─────────────────────────────────────
    # Anchor: start-of-string OR after sentence-ending punctuation + whitespace
    _INLINE_SPEAKER = re.compile(
        r"(?:^|(?<=[.!?])\s+)"
        r"((?:[A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+){0,5})"
        r"(?:(?:,\s*[A-Za-z &/()\"\-\']{2,80})"
        r"|(?:\s*[\-\u2013]\s*[A-Za-z &/()\"\-\']{2,60}))?)"
        r"\s*:\s+",
    )

    # ── Strategy B: FactSet dotted separator ─────────────────────────────────
    _FACTSET_SEPARATOR = re.compile(r"\.{10,}")
    # Title-case name on its own line (FactSet speaker name line)
    _FACTSET_NAME_LINE = re.compile(
        r"^([A-Z][a-z]+(?:[\s\.\-][A-Z][a-z\'\-]+){1,5})\s*$"
    )
    # Inline Operator: prefix in FactSet docs
    _FACTSET_OPERATOR = re.compile(r"^Operator\s*:", re.MULTILINE)

    # ── Strategy C: Microsoft ALLCAPS speaker ────────────────────────────────
    # Pattern: "FIRSTNAME [LASTNAME[, Firm]]:  text"  (all on one line, at line start)
    # Examples: "BRETT IVERSEN:  Good afternoon"
    #           "KARL KEIRSTEAD, UBS:  Thank you"
    #           "SATYA NADELLA:  Thank you"
    _MSFT_ALLCAPS_SPEAKER = re.compile(
        r"(?:^|\n)"
        r"([A-Z]{2,}(?:\s+[A-Z]{2,}){0,4})"          # ALLCAPS name (1-5 words)
        r"(?:,\s*([A-Za-z][A-Za-z\s&\.\']{0,50}))?"  # optional ", Firm"
        r"\s*:\s{1,3}",                                # colon + 1-3 spaces
        re.MULTILINE,
    )
    # Also catches title-case exec names on their own line: "Satya Nadella: " / "Amy Hood: "
    _MSFT_TITLECASE_SPEAKER = re.compile(
        r"(?:^|\n)"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z\'\-]+){1,4})"  # Title Case name
        r"\s*:\s+",
        re.MULTILINE,
    )

    # ── Shared ───
    # Lines to skip: FactSet headers, page footers, copyright, URLs, timestamps
    _SKIP_LINE = re.compile(
        r"(1-877-FACTSET|callstreet\.com|Copyright\s*©|www\.|"
        r"Corrected Transcript|\bTotal Pages\b|^\s*\d+\s*$|"
        r"MANAGEMENT DISCUSSION SECTION|QUESTION AND ANSWER SECTION|"
        r"CORPORATE PARTICIPANTS|OTHER PARTICIPANTS|Disclaimer|"
        r"The information herein|"
        # SA-specific noise lines
        r"SA Transcripts|Seeking Alpha|\d+\.\d+K Followers|"
        r"Call Transcript|Earnings Summary|Transcripts Consumer|"
        r"EPS of \$|Revenue of \$|beats by|misses by|"
        r"Company Participants|Conference Call Participants)",
        re.IGNORECASE,
    )

    @classmethod
    def _classify_role(cls, text: str) -> str:
        for pattern, role in cls._ROLE_MAP:
            if pattern.search(text):
                return role
        return "Other"

    # ── Strategy A ────────────────────────────────────────────────────────────
    @classmethod
    def _segment_inline(cls, text: str) -> list[dict]:
        segments: list[dict] = []
        matches = list(cls._INLINE_SPEAKER.finditer(text))

        if not matches:
            log.info("        Inline strategy: no speaker markers found")
            return [{"speaker": "Unknown", "text": text}]

        log.info("        Inline strategy: %d speaker markers found", len(matches))
        if matches[0].start() > 0:
            preamble = text[:matches[0].start()].strip()
            if preamble:
                segments.append({"speaker": "Unknown", "text": preamble})

        for i, m in enumerate(matches):
            speaker_str = m.group(1).strip()
            text_start = m.end()
            text_end = matches[i + 1].start() if i + \
                1 < len(matches) else len(text)
            block_text = text[text_start:text_end].strip()
            role = cls._classify_role(speaker_str)
            if block_text:
                segments.append({"speaker": role, "text": block_text})

        return segments

    # ── Strategy B ────────────────────────────────────────────────────────────
    @classmethod
    def _segment_factset(cls, text: str) -> list[dict]:
        """
        Split on the dotted separator lines.
        Each block starts with: Name\nTitle\ntext
        The Operator sometimes appears inline as 'Operator: ...' without a separator.
        """
        segments: list[dict] = []

        # Split by dotted separator lines
        blocks = cls._FACTSET_SEPARATOR.split(text)

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # Skip pure header/footer blocks
            if cls._SKIP_LINE.search(block):
                # Check if there's actual speech content mixed in
                lines = [l.strip() for l in block.split("\n") if l.strip()]
                content_lines = [
                    l for l in lines if not cls._SKIP_LINE.search(l)]
                if not content_lines:
                    continue
                block = "\n".join(content_lines)

            lines = [l.strip() for l in block.split("\n") if l.strip()]
            if not lines:
                continue

            # Check if first line looks like an inline "Operator: ..."
            first_line = lines[0]
            inline_op = re.match(r"^(Operator)\s*:\s*(.*)",
                                 first_line, re.IGNORECASE)
            if inline_op:
                op_text = inline_op.group(2) + " " + " ".join(lines[1:])
                role = "Operator"
                if op_text.strip():
                    segments.append({"speaker": role, "text": op_text.strip()})
                continue

            # Check if first line is a speaker name (Title Case, 2+ words)
            name_match = cls._FACTSET_NAME_LINE.match(first_line)
            if name_match:
                speaker_name = name_match.group(1)
                # Second line is typically the title
                title_line = lines[1] if len(lines) > 1 else ""
                # Content starts at line 2 if title present, else line 1
                content_start = 2 if (len(lines) > 2 and title_line and not
                                      cls._FACTSET_NAME_LINE.match(title_line)
                                      and len(title_line.split()) >= 2) else 1
                speech = " ".join(lines[content_start:]).strip()
                role = cls._classify_role(speaker_name + " " + title_line)
                if speech:
                    segments.append({"speaker": role, "text": speech})
            else:
                # No recognized speaker — treat as continuation / unknown
                speech = " ".join(lines).strip()
                if speech:
                    segments.append({"speaker": "Unknown", "text": speech})

        log.info("        FactSet strategy: %d segments found", len(segments))
        return segments if segments else [{"speaker": "Unknown", "text": text}]

    # Known MSFT exec name -> role mapping (covers the most common official transcripts)
    _MSFT_KNOWN_EXECS = {
        "satya nadella": "CEO", "amy hood": "CFO",
        "brett iversen": "IR", "alice jolla": "Executive",
        "keith dolliver": "Executive",
    }

    @classmethod
    def _classify_msft_speaker(cls, speaker_str: str) -> str:
        """
        Classify MSFT-format speakers.
        'SATYA NADELLA' -> CEO, 'KEITH WEISS, Morgan Stanley' -> Analyst, etc.
        """
        lower = speaker_str.lower()
        # Check known exec names first
        for name, role in cls._MSFT_KNOWN_EXECS.items():
            if name in lower:
                return role
        # Firm suffix after comma -> Analyst
        if "," in speaker_str:
            return "Analyst"
        # Moderator / IR keywords
        if re.search(r"iversen|operator", lower):
            return "IR"
        # Generic role keywords
        role = cls._classify_role(lower)
        return role if role != "Other" else "Other"

    # ── Strategy C ────────────────────────────────────────────────────────────
    @classmethod
    def _segment_msft(cls, text: str) -> list[dict]:
        """
        Combine ALLCAPS and TitleCase speaker patterns for Microsoft format.
        Both patterns anchor at newline/start boundaries.
        Examples handled:
          'BRETT IVERSEN:  Good afternoon'
          'KEITH WEISS, Morgan Stanley:  Excellent.'
          'Satya Nadella: Thank you, Brett.'
          'Amy Hood: Thank you, Satya.'
        """
        hits: list[tuple[int, str, int]] = []

        for m in cls._MSFT_ALLCAPS_SPEAKER.finditer(text):
            name = m.group(1).strip()
            firm = (m.group(2) or "").strip()
            speaker_str = f"{name}, {firm}" if firm else name
            hits.append((m.start(), speaker_str, m.end()))

        for m in cls._MSFT_TITLECASE_SPEAKER.finditer(text):
            name = m.group(1).strip()
            # Skip if an ALLCAPS match already captured this position
            already = any(abs(h[0] - m.start()) < 5 for h in hits)
            if not already:
                hits.append((m.start(), name, m.end()))

        hits.sort(key=lambda x: x[0])

        if not hits:
            log.info("        MSFT strategy: no speaker markers found")
            return [{"speaker": "Unknown", "text": text}]

        log.info("        MSFT strategy: %d speaker markers found", len(hits))
        segments: list[dict] = []

        if hits[0][0] > 0:
            preamble = text[:hits[0][0]].strip()
            if preamble:
                segments.append({"speaker": "Unknown", "text": preamble})

        for i, (pos, speaker_str, end) in enumerate(hits):
            text_end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
            block_text = text[end:text_end].strip()
            role = cls._classify_msft_speaker(speaker_str)
            if block_text:
                segments.append({"speaker": role, "text": block_text})

        return segments

    # ── Strategy D ────────────────────────────────────────────────────────────
    # SA standalone speaker name: Title Case, 2-5 words, nothing else on the line
    _SA_NAME_LINE = re.compile(
        r"^([A-Z][a-z]+(?:\s+[A-Z][a-z\'\-]+){1,4})\s*$"
    )
    # SA participant list line: "Name - Title"  (one dash separating name from title)
    _SA_ROSTER_LINE = re.compile(
        r"^([A-Z][a-z]+(?:\s+[A-Z][a-z\'\-]+){1,4})\s*-\s*(.+)$"
    )
    # Firm names that identify an analyst in the roster
    _FIRM_NAMES = re.compile(
        r"\b(Goldman|JPMorgan|Morgan Stanley|Evercore|Jefferies|Barclays|"
        r"UBS|Citi|Deutsche|Cowen|Wells Fargo|Bank of America|RBC|"
        r"Credit Suisse|HSBC|Truist|Bernstein|Stifel|Guggenheim|"
        r"Piper|Wolfe|Needham|KeyBanc|Oppenheimer|Mizuho|TD|BTIG)\b",
        re.IGNORECASE,
    )
    # "Operator" appears as a standalone line in SA format (no colon)
    _SA_OPERATOR_LINE = re.compile(r"^Operator$", re.IGNORECASE)
    # Lines that mark section headers inside SA docs (not speakers)
    _SA_SECTION_HEADER = re.compile(
        r"^(Call Transcript|Earnings Summary|Company Participants|"
        r"Conference Call Participants|Transcripts|Participants)$",
        re.IGNORECASE,
    )

    # ── SA run-together fix ───────────────────────────────────────────────────
    @classmethod
    def _inject_sa_newlines(cls, text: str) -> str:
        """
        Fix SA transcripts where pypdf extracted speaker turns without newlines,
        producing run-together text like:
            'Bernard ArnaultChairman & CEOGood evening...'
            'Alexandra SteigerGood evening, everyone...'
            'Hiroki TotokiToday, I will explain...'

        Three-pass strategy — all passes operate on the full document text:

        Pass 1 — inject a newline BEFORE each roster name when not already at
                  line start.  Anchors on the Company Participants block for
                  the definitive name list.
        Pass 2 — inject a newline AFTER each roster name when directly followed
                  by non-newline content (title or speech running on immediately).
        Pass 3 — inject a newline AFTER known job titles when they are directly
                  followed by speech text (e.g. 'Chairman & CEOGood evening' →
                  'Chairman & CEO\\nGood evening').

        After these three passes the SA line-based parser can see each speaker
        name on its own line, followed by an optional title line, followed by
        speech lines — exactly the expected format.
        """
        # ── Build roster map: name → job title ───────────────────────────────
        roster_map: dict[str, str] = {}

        # Prefer the structured Company Participants block when present
        block_m = re.search(
            r"Company Participants(.+?)"
            r"(?:Presentation\b|Conference Call Participants|Question-and-Answer)",
            text[:4000],
            re.DOTALL | re.IGNORECASE,
        )
        if block_m:
            block = block_m.group(1).strip()
            # Split on each "TitleCase Name - " boundary
            name_title_pat = re.compile(
                r"([A-Z][a-z\u00c0-\u00ff]+(?:\s+[A-Z][a-z\u00c0-\u00ff'\-]+){1,4})\s*-\s*"
            )
            parts = name_title_pat.split(block)
            for i in range(1, len(parts) - 1, 2):
                name = parts[i].strip()
                raw_title = parts[i + 1]
                # Trim the title: stop before the next person's name at the end
                trimmed = re.sub(
                    r"[A-Z][a-z\u00c0-\u00ff]+(?:\s+[A-Z][a-z\u00c0-\u00ff'\-]+){1,4}\s*$",
                    "",
                    raw_title,
                ).strip()
                roster_map[name] = trimmed if trimmed else raw_title.strip()

        # Fallback: simple "Name - Title" scan anywhere in the first 3 000 chars
        for m in re.finditer(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z'\-]+){1,4})\s*-\s*[A-Z]",
            text[:3000],
        ):
            name = m.group(1).strip()
            if name not in roster_map:
                roster_map[name] = ""

        # Add bare "Operator" if present in roster area
        if "Operator" not in roster_map and re.search(r"\bOperator\b", text[:3000]):
            roster_map["Operator"] = ""

        if not roster_map:
            return text  # no roster found — nothing to fix

        names = sorted(roster_map, key=len, reverse=True)
        # skip Operator for after-name pass
        alt_names = [n for n in names if n != "Operator"]

        # ── Pass 1: inject \n BEFORE each name (when not already line-initial) ─
        pat_before = re.compile(
            r"(?<!\n)(" + "|".join(re.escape(n) for n in names) + r")(?!\s*-)"
        )

        def _before(m: re.Match) -> str:
            return m.group(1) if m.start() == 0 else "\n" + m.group(1)

        text = pat_before.sub(_before, text)

        # ── Pass 2: inject \n AFTER each name (when followed by non-newline content)
        if alt_names:
            pat_after = re.compile(
                r"(" + "|".join(re.escape(n) for n in alt_names) + r")"
                r"(?![\n\s]*-)(?=[^\n])"
            )
            text = pat_after.sub(lambda m: m.group(1) + "\n", text)

        # ── Pass 3: inject \n AFTER known job titles (when run into speech) ──
        for name, title in roster_map.items():
            if title and len(title) > 3:
                text = re.sub(
                    re.escape(title) + r"(?=[A-Z][a-z])",
                    title + "\n",
                    text,
                )

        return text

    @classmethod
    def _build_sa_name_role_map(cls, lines: list[str]) -> dict[str, str]:
        """
        Parse the SA participant roster (Name - Title lines near the top)
        to build a name -> role lookup used during segmentation.

        This is the KEY fix: without it, speakers like 'Brian Olsavsky'
        can't be classified because their title isn't on the same line as
        their name in SA format.
        """
        name_role: dict[str, str] = {}
        for line in lines:
            stripped = line.strip()
            m = cls._SA_ROSTER_LINE.match(stripped)
            if not m:
                continue
            name = m.group(1).strip()
            title = m.group(2).strip()
            # Classify by title
            role = cls._classify_role(title)
            if role == "Other":
                # Firm name in title = analyst
                if cls._FIRM_NAMES.search(title):
                    role = "Analyst"
            name_role[name.lower()] = role
        return name_role

    # SA inline-colon pattern: "Name: text" on a single line (Exxon / similar)
    # Used as fallback when no standalone-name lines are found after injection.
    _SA_INLINE_COLON = re.compile(
        r"(?:^|(?<=[.!?\n])\s*)"
        r"((?:[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){0,5})"
        r"(?:,\s*[A-Za-z &/()\"-']{2,80})?)"
        r"\s*:\s*\|?\s+",          # colon, optional pipe (Exxon quirk), space
        re.MULTILINE,
    )

    @classmethod
    def _segment_seeking_alpha(cls, text: str) -> list[dict]:
        """
        Seeking Alpha text-based format:
          - Participant roster at top:  'Name - Title'  (parsed for role lookup)
          - Speaker name on its own standalone line (Title Case, 2-5 words)
          - Speech text on the following line(s)
          - 'Operator' appears as a bare standalone line (no colon)
          - No dotted separators, no title line below the name

        Two-pass approach:
          Pass 1: scan all lines, build name->role map from participant roster
          Pass 2: walk lines, use name->role map to classify each speaker turn

        Pre-step: inject newlines where pypdf merged speaker name into speech text
        (run-together format: 'Bernard ArnaultChairman & CEOGood evening...').

        Fallback: if name-on-own-line yields nothing, try inline 'Name: text'
        colon-separated pattern (covers Exxon-style SA transcripts).
        """
        # ── Pre-step: fix run-together name/speech (LVMH, SAP, Sony, etc.) ──
        # Build a name->role map from the ORIGINAL text (before injection scrambles roster lines)
        pre_injection_name_role: dict[str, str] = {}
        _block_m = re.search(
            r"Company Participants(.+?)"
            r"(?:Presentation\b|Conference Call Participants|Question-and-Answer)",
            text[:4000],
            re.DOTALL | re.IGNORECASE,
        )
        if _block_m:
            _block = _block_m.group(1).strip()
            _ntp = re.compile(
                r"([A-Z][a-z\u00c0-\u00ff]+(?:\s+[A-Z][a-z\u00c0-\u00ff'\-]+){1,4})\s*-\s*"
            )
            _parts = _ntp.split(_block)
            for _i in range(1, len(_parts) - 1, 2):
                _name = _parts[_i].strip()
                _raw_title = _parts[_i + 1]
                _trimmed = re.sub(
                    r"[A-Z][a-z\u00c0-\u00ff]+(?:\s+[A-Z][a-z\u00c0-\u00ff'\-]+){1,4}\s*$",
                    "",
                    _raw_title,
                ).strip()
                _title = _trimmed if _trimmed else _raw_title.strip()
                _role = cls._classify_role(_title)
                pre_injection_name_role[_name.lower(
                )] = _role if _role != "Other" else "Other"

        # Also scan Conference Call Participants for analyst names
        _conf_m = re.search(
            r"Conference Call Participants(.+?)(?:\Z|$)",
            text[:4000],
            re.DOTALL | re.IGNORECASE,
        )
        if _conf_m:
            _conf_block = _conf_m.group(1)
            _cntp = re.compile(
                r"([A-Z][a-z\u00c0-\u00ff]+(?:\s+[A-Z][a-z\u00c0-\u00ff'\-]+){1,4})\s*-\s*([^\n]+?)(?=[A-Z][a-z]|\Z)"
            )
            for _cm in _cntp.finditer(_conf_block[:2000]):
                _name = _cm.group(1).strip()
                _title = _cm.group(2).strip()
                if _name.lower() not in pre_injection_name_role:
                    _role = cls._classify_role(_title)
                    if _role == "Other" and cls._FIRM_NAMES.search(_title):
                        _role = "Analyst"
                    pre_injection_name_role[_name.lower()] = _role

        log.info("        SA pre-injection roster: %d named speakers",
                 len(pre_injection_name_role))

        text = cls._inject_sa_newlines(text)

        lines = text.split("\n")

        # Pass 1: build name->role lookup from participant list (post-injection lines)
        name_role_map = cls._build_sa_name_role_map(lines)
        # Merge pre-injection roster (higher fidelity for run-together transcripts)
        for _k, _v in pre_injection_name_role.items():
            if _k not in name_role_map or name_role_map[_k] == "Other":
                name_role_map[_k] = _v
        log.info("        SA roster parsed: %d named speakers",
                 len(name_role_map))

        segments: list[dict] = []
        current_speaker: str | None = None
        current_lines: list[str] = []
        roster_done = False   # True once we've passed the participant list section

        def flush():
            if current_speaker is not None and current_lines:
                speech = " ".join(current_lines).strip()
                if speech:
                    segments.append(
                        {"speaker": current_speaker, "text": speech})

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Skip known boilerplate lines
            if cls._SKIP_LINE.search(stripped):
                continue

            # Skip SA section header lines
            if cls._SA_SECTION_HEADER.match(stripped):
                continue

            # Skip roster lines (Name - Title) — already parsed in pass 1
            if cls._SA_ROSTER_LINE.match(stripped) and not roster_done:
                continue

            # 'Operator' as standalone line
            if cls._SA_OPERATOR_LINE.match(stripped):
                flush()
                current_speaker = "Operator"
                current_lines = []
                roster_done = True
                continue

            # Check if this line is a standalone speaker name
            name_m = cls._SA_NAME_LINE.match(stripped)
            if name_m:
                candidate = name_m.group(1)
                # Look up in roster map first (most accurate)
                role = name_role_map.get(candidate.lower())
                if role is None:
                    # Not in roster — try keyword classification
                    role = cls._classify_role(candidate)
                    if role == "Other":
                        # Unknown name, not in roster = skip as noise
                        # (catches stray header words like 'Call Transcript')
                        if not roster_done:
                            continue
                flush()
                current_speaker = role
                current_lines = []
                roster_done = True
                # The very next line may be a job title (injected from run-together text)
                # e.g. "Chairman & CEO" — we handle it by checking in the next iteration:
                # if the line looks like a title phrase and current_lines is still empty,
                # use it to refine the role rather than treating it as speech.
                continue

            # Regular speech content
            if current_speaker is None:
                current_speaker = "Unknown"
            roster_done = True

            # Skip job-title lines that immediately follow an injected speaker name
            # e.g. "Chairman & CEO", "Chief Financial Officer", "Head, IR"
            # These appear as the first content line when _inject_sa_newlines split
            # a run-together block.  Heuristic: short line (≤8 words), no verb,
            # current_lines is still empty (nothing spoken yet).
            if not current_lines and len(stripped.split()) <= 8:
                role_upgrade = cls._classify_role(stripped)
                if role_upgrade != "Other":
                    current_speaker = role_upgrade  # refine role from title line
                    continue                         # don't add title to speech

            current_lines.append(stripped)

        flush()

        log.info("        SA strategy: %d segments found", len(segments))

        # ── Fallback: inline 'Name: text' colon pattern (e.g. Exxon) ─────────
        if not segments or all(s["speaker"] == "Unknown" for s in segments):
            log.info("        SA line-based failed — trying inline colon fallback")
            flat = " ".join(text.split("\n"))  # collapse for inline matching
            flat = re.sub(r" {2,}", " ", flat)
            colon_matches = list(cls._SA_INLINE_COLON.finditer(flat))
            if colon_matches:
                log.info("        Inline colon fallback: %d markers",
                         len(colon_matches))
                # Rebuild name->role from whatever the roster gave us
                colon_segs: list[dict] = []
                if colon_matches[0].start() > 0:
                    preamble = flat[:colon_matches[0].start()].strip()
                    if preamble:
                        colon_segs.append(
                            {"speaker": "Unknown", "text": preamble})
                for i, cm in enumerate(colon_matches):
                    spk_str = cm.group(1).strip()
                    t_start = cm.end()
                    t_end = colon_matches[i + 1].start() if i + \
                        1 < len(colon_matches) else len(flat)
                    blk = flat[t_start:t_end].strip()
                    role = name_role_map.get(spk_str.lower())
                    if role is None:
                        role = cls._classify_role(spk_str)
                        if role == "Other":
                            if "operator" in spk_str.lower():
                                role = "Operator"
                            elif "," in spk_str:
                                role = "Analyst"
                    if blk:
                        colon_segs.append({"speaker": role, "text": blk})
                if colon_segs:
                    segments = colon_segs

        return segments if segments else [{"speaker": "Unknown", "text": text}]

    # ── Public entry point ────────────────────────────────────────────────────
    @classmethod
    def segment(cls, text: str, fmt: str) -> tuple[dict[str, list[str]], list[dict]]:
        if fmt == "factset":
            segments = cls._segment_factset(text)
        elif fmt == "msft":
            segments = cls._segment_msft(text)
        elif fmt == "seeking_alpha":
            segments = cls._segment_seeking_alpha(text)
        else:
            # inline / unknown
            segments = cls._segment_inline(text)

        # If primary strategy produced nothing useful, fall back to inline
        real_segs = [s for s in segments if s["speaker"]
                     != "Unknown" and s["text"]]
        if not real_segs and fmt not in ("inline", "seeking_alpha"):
            log.info("        Primary strategy yielded no labelled segments — "
                     "falling back to inline")
            segments = cls._segment_inline(text)

        # Aggregate into role buckets for the speakers dict
        speakers: dict[str, list[str]] = {}
        for seg in segments:
            role = seg["speaker"]
            speakers.setdefault(role, [])
            if seg["text"]:
                speakers[role].append(seg["text"])

        log.info("        Speaker roles found: %s",
                 {r: len(v) for r, v in speakers.items()})
        return speakers, segments


# =============================================================================
class SectionExtractor:
    """
    Splits transcript into Prepared Remarks and Q&A session.
    Works on both flat (inline) and newline-preserved (factset/msft) text.
    """

    _QA_MARKERS = re.compile(
        r"(question[- ]and[- ]answer\s+section"
        r"|q\s*&\s*a\s+section"
        r"|q\s*and\s*a"
        r"|open.*floor.*question"
        r"|now.*take.*question"
        r"|we.ll now move over to q&a"
        r"|operator.*instruct"
        r"|please.*go ahead"
        r"|your (first )?question)",
        re.IGNORECASE,
    )

    @classmethod
    def extract(cls, text: str) -> dict[str, str]:
        match = cls._QA_MARKERS.search(text)
        if match:
            split_idx = match.start()
            return {
                "prepared_remarks": text[:split_idx].strip(),
                "qa_session":       text[split_idx:].strip(),
            }
        return {"prepared_remarks": text.strip(), "qa_session": ""}


# =============================================================================
class BoilerplateStripper:
    """
    Strips known boilerplate sections from transcripts before processing.

    - FactSet: disclaimer page at the end
    - Seeking Alpha: user comment section (not applicable to image PDFs,
      but included for any text-layer SA docs)
    - General: participant lists, header/footer lines (page numbers, URLs)
    """

    # Marks the beginning of FactSet legal disclaimer
    _FACTSET_DISCLAIMER = re.compile(
        r"Disclaimer\s*\nThe information herein",
        re.IGNORECASE,
    )

    # Seeking Alpha comment sections
    _SA_COMMENTS = re.compile(
        r"\n(?:Comments\s*\(\d+\)|Show\s+\d+\s+Comments?|"
        r"Submit\s+Comments?|Leave\s+a\s+Comment)",
        re.IGNORECASE,
    )

    # FactSet participant roster header — strip the whole block up to content
    _PARTICIPANT_BLOCK = re.compile(
        r"(CORPORATE PARTICIPANTS.*?OTHER PARTICIPANTS.*?\n\n)",
        re.IGNORECASE | re.DOTALL,
    )

    @classmethod
    def strip(cls, text: str, fmt: str) -> str:
        # Remove FactSet disclaimer at end
        m = cls._FACTSET_DISCLAIMER.search(text)
        if m:
            text = text[:m.start()].strip()
            log.info("        Stripped FactSet disclaimer")

        # Remove Seeking Alpha comment section
        m = cls._SA_COMMENTS.search(text)
        if m:
            text = text[:m.start()].strip()
            log.info("        Stripped SA comment section")

        return text


# =============================================================================
class SemanticChunker:
    """
    Splits speaker segments into overlapping token-aware chunks.

    Chunking is done PER SPEAKER SEGMENT so every chunk retains its speaker
    and section attribution. The step size is guaranteed positive so the
    window always advances (no infinite loop possible).

    Token approximation: 1 token ~= 0.75 words.
    """

    _WORDS_PER_TOKEN = 0.75

    def __init__(self, chunk_tokens: int = CHUNK_TOKENS, overlap_tokens: int = OVERLAP_TOKENS):
        self.chunk_size = int(chunk_tokens * self._WORDS_PER_TOKEN)
        self.step_size = self.chunk_size - \
            int(overlap_tokens * self._WORDS_PER_TOKEN)
        if self.step_size < 1:
            self.step_size = max(1, self.chunk_size // 2)

    @staticmethod
    def _approx_tokens(n_words: int) -> int:
        return int(n_words / 0.75)

    def chunk(
        self,
        words: list[str],
        speaker: Optional[str] = None,
        section: Optional[str] = None,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        total = len(words)
        if total == 0:
            return chunks
        start = 0
        idx = 0
        while start < total:
            end = min(start + self.chunk_size, total)
            chunk_text = " ".join(words[start:end])
            chunks.append(Chunk(
                id=f"chunk_{idx}",
                text=chunk_text,
                speaker=speaker,
                section=section,
                token_count=self._approx_tokens(end - start),
                has_forex=_has_forex(chunk_text),
            ))
            idx += 1
            if end == total:
                break
            start += self.step_size
        return chunks


def _has_forex(text: str) -> bool:
    """
    Returns True only if the chunk contains language specific to FX/currency
    signals — not generic financial or business language.
    """
    lower = text.lower()
    return any(kw in lower for kw in FOREX_KEYWORDS)


# =============================================================================
class PreprocessingAgent:
    """
    Orchestrates the full pipeline for one PDF:

    PDF read -> format detect -> boilerplate strip -> clean ->
    speaker segmentation -> section split -> per-speaker chunking -> JSON output.

    Chunking is done per speaker segment (not on the whole text) so every
    chunk carries the correct speaker and section label.
    """

    def __init__(self):
        self.chunker = SemanticChunker()

    @staticmethod
    def _infer_date(quarter: str, year: str) -> str:
        if year == "Unknown" or quarter == "Unknown":
            return ""
        month = _QUARTER_MONTH.get(quarter.upper(), "01")
        report_year = str(int(year) + 1) if quarter.upper() == "Q4" else year
        return f"{report_year}-{month}-15"

    def process(self, pdf_path: Path) -> TranscriptDoc:
        t0 = time.time()

        # ── Step 1: parse filename ────────────────────────────────────────────
        log.info("  [1/7] Parsing filename...")
        company_name, quarter, year = FilenameParser.parse(pdf_path)
        ticker = TickerResolver.resolve(company_name)
        log.info("        -> %s (%s) | %s %s",
                 company_name, ticker, quarter, year)

        # ── Step 2: extract text from PDF ─────────────────────────────────────
        log.info("  [2/7] Extracting text from PDF...")
        t = time.time()
        raw_text, is_image = PDFExtractor.extract(pdf_path)
        log.info("        -> %d chars extracted | image_based=%s  (%s)",
                 len(raw_text), is_image, _elapsed(t))

        if not raw_text.strip():
            # OCR was attempted but produced nothing — truly unprocessable
            log.warning(
                "  !! No text could be extracted from %s (even after OCR). "
                "Document is unprocessable — saving empty stub.",
                pdf_path.name,
            )
            reason = "ocr_produced_no_text" if is_image else "empty_pdf_unreadable"
            return TranscriptDoc(
                company_name=company_name,
                ticker=ticker,
                earnings_date=self._infer_date(quarter, year),
                quarter=f"{quarter} {year}" if year != "Unknown" else quarter,
                preprocessed_at=datetime.now(timezone.utc).isoformat(),
                raw_text="",
                cleaned_text="",
                speakers={},
                sections={"prepared_remarks": "", "qa_session": ""},
                chunks=[],
                preprocessing_stats={
                    "total_chunks": 0,
                    "forex_chunks": 0,
                    "total_speakers": 0,
                    "avg_chunk_length": 0,
                    "skipped_reason": reason,
                },
            )

        if is_image:
            log.info(
                "  OCR text extracted successfully — continuing with SA format pipeline")

        # ── Step 3: detect format ─────────────────────────────────────────────
        log.info("  [3/7] Detecting transcript format...")
        fmt = FormatDetector.detect(raw_text, force_sa=is_image)

        # ── Step 4: strip boilerplate ─────────────────────────────────────────
        log.info("  [4/7] Stripping boilerplate...")
        t = time.time()
        raw_text = BoilerplateStripper.strip(raw_text, fmt)
        log.info("        -> done  (%s)", _elapsed(t))

        # ── Step 5: clean text ────────────────────────────────────────────────
        log.info("  [5/7] Cleaning text...")
        t = time.time()
        cleaned_text = TextCleaner.clean(raw_text, fmt)
        log.info("        -> %d chars after clean  (%s)",
                 len(cleaned_text), _elapsed(t))

        # ── Step 6: speaker segmentation ──────────────────────────────────────
        log.info("  [6/7] Identifying speakers (format=%s)...", fmt)
        t = time.time()
        speakers_dict, segments = SpeakerIdentifier.segment(cleaned_text, fmt)
        log.info("        -> %d role(s), %d segment(s)  (%s)",
                 len(speakers_dict), len(segments), _elapsed(t))

        # Section extraction (works on the cleaned text regardless of format)
        sections = SectionExtractor.extract(cleaned_text)
        log.info("        -> Q&A detected: %s", bool(sections["qa_session"]))

        # ── Step 7: chunking ──────────────────────────────────────────────────
        log.info("  [7/7] Chunking per speaker segment...")
        t = time.time()
        all_chunks: list[Chunk] = []
        global_idx = 0

        # Determine section boundary by character position
        qa_start_pos = len(sections["prepared_remarks"]
                           ) if sections["qa_session"] else None
        char_pos = 0

        for seg in segments:
            seg_text = seg["text"]
            seg_speaker = seg["speaker"]
            seg_section = (
                "qa_session"
                if (qa_start_pos is not None and char_pos >= qa_start_pos)
                else "prepared_remarks"
            )
            char_pos += len(seg_text) + 1

            seg_words = seg_text.split()
            sub_chunks = self.chunker.chunk(
                seg_words, speaker=seg_speaker, section=seg_section)
            for c in sub_chunks:
                c.id = f"chunk_{global_idx}"
                global_idx += 1
                all_chunks.append(c)

        forex_count = sum(1 for c in all_chunks if c.has_forex)
        log.info("        -> %d chunks | %d forex-flagged  (%s)",
                 len(all_chunks), forex_count, _elapsed(t))

        doc = TranscriptDoc(
            company_name=company_name,
            ticker=ticker,
            earnings_date=self._infer_date(quarter, year),
            quarter=f"{quarter} {year}" if year != "Unknown" else quarter,
            preprocessed_at=datetime.now(timezone.utc).isoformat(),
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            speakers=speakers_dict,
            sections=sections,
            chunks=all_chunks,
            preprocessing_stats={
                "total_chunks":     len(all_chunks),
                "forex_chunks":     forex_count,
                "total_speakers":   len(speakers_dict),
                "avg_chunk_length": (
                    int(sum(c.token_count for c in all_chunks) / len(all_chunks))
                    if all_chunks else 0
                ),
                "format_detected":  fmt,
            },
        )
        log.info("  DONE: %s  (total: %s)", pdf_path.name, _elapsed(t0))
        return doc

    @staticmethod
    def save(doc: TranscriptDoc, out_dir: Path = PREPROCESSED_DIR) -> Path:
        safe_name = re.sub(r"[^\w]", "_", doc.company_name.lower())
        quarter = re.sub(r"\s+", "_", doc.quarter.lower())
        out_path = out_dir / f"{safe_name}_{quarter}.json"
        t = time.time()
        out_path.write_text(
            json.dumps(doc.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("  Saved -> %s  (%s)", out_path.name, _elapsed(t))
        return out_path

    def run_all(
        self,
        transcripts_dir: Path = TRANSCRIPTS_DIR,
        out_dir: Path = PREPROCESSED_DIR,
        limit: int | None = None,
        skip: int | None = None,
        pattern: str | None = None,
    ) -> list[Path]:
        pdf_files = sorted(transcripts_dir.glob("*.[Pp][Dd][Ff]"))
        if not pdf_files:
            log.warning("No PDF files found in %s", transcripts_dir)
            return []

        # Apply --pattern filter first (case-insensitive filename match)
        if pattern:
            pdf_files = [p for p in pdf_files if pattern.lower()
                         in p.name.lower()]
            log.info("--pattern '%s': %d matching file(s)",
                     pattern, len(pdf_files))
            if not pdf_files:
                log.warning("No files matched pattern '%s'", pattern)
                return []

        # Apply --skip (drop first N from sorted list)
        if skip:
            pdf_files = pdf_files[skip:]
            log.info("--skip %d: starting from file %d onward (%d remaining)",
                     skip, skip + 1, len(pdf_files))

        # Apply --limit (cap total processed)
        if limit:
            pdf_files = pdf_files[:limit]
            log.info("--limit %d: processing %d file(s)",
                     limit, len(pdf_files))

        log.info("Found %d PDF(s) to process", len(pdf_files))
        outputs: list[Path] = []
        batch_start = time.time()

        for i, pdf in enumerate(pdf_files, 1):
            log.info("")
            log.info("=" * 60)
            log.info("[%d/%d] %s", i, len(pdf_files), pdf.name)
            log.info("=" * 60)
            try:
                doc = self.process(pdf)
                path = self.save(doc, out_dir)
                outputs.append(path)
            except Exception as e:
                log.error("FAILED: %s | %s", pdf.name, e, exc_info=True)

        log.info("")
        log.info("*" * 60)
        log.info("Batch complete: %d/%d succeeded | Total: %s",
                 len(outputs), len(pdf_files), _elapsed(batch_start))
        log.info("*" * 60)
        return outputs
