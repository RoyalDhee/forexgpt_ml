# Earnings Transcript FX Pipeline

A two-stage pipeline that converts raw earnings call PDFs into labeled forex
training data.

## File Structure

```
pipeline/
├── config.py           # Paths, ticker map, forex keywords, chunking settings
├── models.py           # Dataclasses: Chunk, TranscriptDoc, Label
├── preprocessor.py     # Stage 1 — PDF → structured JSON
├── labeler.py          # Stage 2 — JSON chunks → OpenAI forex labels
├── run.py              # CLI entry point
├── pipeline.log        # Rolling log file (appended on every run)
├── requirements.txt
├── .env.template       # Copy to .env and add your key
└── data/
    ├── transcripts/    # ← Drop your PDFs here
    ├── preprocessed/   # ← Per-transcript JSONs (auto-created)
    └── labeled/
        ├── <company>_<quarter>_labels.json   # Per-doc label files
        └── training_data.json                # Full aggregate (all docs)
```

## Setup

```bash
pip install -r requirements.txt

cp .env.template .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

## PDF Naming Convention

Files must follow this pattern (case-insensitive).
Separators can be **underscores, hyphens, or a mix** — all are handled:

```
<company_name><sep><Q1|Q2|Q3|Q4><sep><YEAR><sep>ect.pdf
```

where `<sep>` is `_` or `-`.

Examples:

```
suncor_energy_q2_2024_ect.pdf
MICROSOFT_Q1_2024_ECT.PDF
taiwan_semiconductor_q3_2023_ect.pdf
shell-q1-2024-ect.pdf
rio_tinto-q3-2023_ect.PDF
```

Multi-word company names use underscores or hyphens between words.

---

## Usage

### Stage 1 — Preprocessing

```bash
# All PDFs in data/transcripts/
python run.py preprocess

# Single file by path
python run.py preprocess --file data/transcripts/suncor_energy_q2_2024_ect.pdf

# First 5 files only (useful for testing)
python run.py preprocess --limit 5

# Skip first 32 files, process the rest
python run.py preprocess --skip 32

# Skip 32, then process next 5
python run.py preprocess --skip 32 --limit 5

# Only files whose name contains 'apple' (case-insensitive)
python run.py preprocess --pattern apple

# First 3 apple files
python run.py preprocess --pattern apple --limit 3
```

### Stage 2 — Labeling

```bash
# Label all preprocessed docs (skips already-labeled ones by default)
python run.py label

# Re-label everything from scratch (ignores existing per-doc label files)
python run.py label --no-resume

# First 3 docs only
python run.py label --limit 3

# Skip first 5 docs, label the rest
python run.py label --skip 5

# Skip 5, then label next 3
python run.py label --skip 5 --limit 3

# Only docs whose filename contains 'sony'
python run.py label --pattern sony

# First 2 sony docs
python run.py label --pattern sony --limit 2
```

### Both stages

```bash
# Run preprocess then label sequentially
python run.py all

# Both stages, first 2 docs only
python run.py all --limit 2

# Both stages for apple docs only
python run.py all --pattern apple
```

---

## Supported Transcript Formats

The preprocessor auto-detects the transcript format and applies the appropriate
parsing strategy. Four formats are supported:

| Format          | Description                                                  | Example sources                |
| --------------- | ------------------------------------------------------------ | ------------------------------ |
| `inline`        | `"Name, Title: speech text"` all on one line                 | Alphabet official transcripts  |
| `factset`       | Dotted separator lines (`......`) between speaker blocks     | FactSet / CallStreet           |
| `msft`          | `ALLCAPS NAME, Firm:` or `Firstname Lastname:` at line start | Microsoft official transcripts |
| `seeking_alpha` | Speaker name on its own line, speech on following lines      | Seeking Alpha text transcripts |

### Seeking Alpha — run-together fix

Some PDFs (e.g. LVMH, SAP, Sony, Siemens, Toyota) are extracted by pypdf
without newlines between the speaker name, their job title, and their speech,
producing run-together text like:

```
Bernard ArnaultChairman & CEOGood evening. I'm delighted to present...
```

The preprocessor handles this automatically via a three-pass injection step
that reconstructs the correct line structure before parsing:

1. **Pass 1** — inject `\n` before each roster name when not already at line start
2. **Pass 2** — inject `\n` after each roster name when directly followed by content
3. **Pass 3** — inject `\n` after known job title strings when run into speech

Speaker roles are resolved from the `Company Participants` and
`Conference Call Participants` roster blocks at the top of each transcript,
before any injection takes place.

---

## Output Formats

### Preprocessed JSON (`data/preprocessed/<company>_<quarter>.json`)

```json
{
  "metadata": {
    "company_name": "Suncor Energy",
    "ticker": "SU",
    "earnings_date": "2024-07-15",
    "quarter": "Q2 2024",
    "source": "Earnings Call",
    "preprocessed_at": "2024-08-01T10:30:00Z"
  },
  "raw_text": "...",
  "cleaned_text": "...",
  "speakers": {
    "CEO": ["statement1", "statement2"],
    "CFO": ["statement1"]
  },
  "sections": {
    "prepared_remarks": "...",
    "qa_session": "..."
  },
  "chunks": [
    {
      "id": "chunk_0",
      "text": "...",
      "speaker": "CFO",
      "section": "prepared_remarks",
      "token_count": 312,
      "has_forex": true
    }
  ],
  "preprocessing_stats": {
    "total_chunks": 24,
    "forex_chunks": 7,
    "total_speakers": 5,
    "avg_chunk_length": 410,
    "format_detected": "factset"
  }
}
```

### Per-doc label file (`data/labeled/<company>_<quarter>_labels.json`)

Written immediately after each document is fully labeled. Contains the same
structure as `training_data.json` but scoped to a single transcript. Used for
resume support — if this file exists, the document is skipped on the next run.

### Aggregate training data (`data/labeled/training_data.json`)

Rebuilt from all per-doc label files at the end of every labeling run.
A JSON array where each entry is one forex signal:

```json
[
  {
    "input": "The exact chunk text that contains the signal...",
    "output": {
      "signal": true,
      "currency_pair": "EUR/USD",
      "direction": "SHORT",
      "confidence": 0.85,
      "reasoning": "Company reported 4% revenue headwind from dollar strength vs euro...",
      "magnitude": "moderate",
      "time_horizon": "next_quarter"
    },
    "metadata": {
      "company": "Microsoft",
      "ticker": "MSFT",
      "earnings_date": "2024-04-15",
      "quarter": "Q1 2024",
      "speaker": "CFO",
      "section": "prepared_remarks",
      "chunk_id": "chunk_3",
      "labeler": "openai-gpt4o",
      "labeled_at": "2024-08-01T11:00:00Z"
    }
  }
]
```

If one chunk contains signals for multiple currency pairs, it produces
**multiple entries** — each with the same `input` but different `output`.

---

## Speaker Roles

Speakers are classified into the following roles based on their job title:

| Role        | Matched titles                                                                              |
| ----------- | ------------------------------------------------------------------------------------------- |
| `CEO`       | Chief Executive, CEO, President & CEO, Chairman & CEO                                       |
| `CFO`       | Chief Financial, CFO, Finance Department/Division, Accounting Chief, Treasurer              |
| `COO`       | Chief Operating, COO                                                                        |
| `Executive` | SVP, EVP, Senior/Executive VP, CTO, CMO, CBO, General Counsel, Chief Communications         |
| `Analyst`   | Analyst, Equity Research, Securities, Capital Markets, firm names (Goldman, JPMorgan, etc.) |
| `IR`        | Investor Relations, Head of IR, Director of Investor Relations, VP Investor Relations       |
| `Operator`  | Operator                                                                                    |
| `Other`     | All other identified speakers                                                               |
| `Unknown`   | Preamble text before the first identified speaker                                           |

---

## Key Design Decisions

**Forex filtering** — only chunks with `has_forex=true` are sent to OpenAI,
saving API cost. The filter uses a curated keyword list covering explicit FX
mechanisms, CFO-style institutional phrasing, hedging terminology, directional
language, and specific currency pair mentions. All chunks are still stored in
the preprocessed JSON for full traceability via `chunk_id`.

**Per-doc label files** — each document gets its own
`<company>_<quarter>_labels.json` written immediately upon completion.
Resume logic checks for this file's existence, so a run interrupted mid-batch
resumes cleanly from the next unprocessed document without re-labeling anything.
The aggregate `training_data.json` is always rebuilt from all per-doc files
at the end of every run.

**Speaker identification** — four format-specific strategies handle the variety
of transcript layouts in the wild. A keyword-based role classifier maps free-text
job titles to canonical roles. For Seeking Alpha transcripts the `Company
Participants` roster is parsed before injection to build a name→role lookup
that survives the newline reconstruction step.

**Format detection** — the preprocessor samples the first three pages to
distinguish text-layer PDFs from image-based PDFs, then inspects structural
markers (dotted separator lines, ALLCAPS speaker names, Seeking Alpha headers)
to select the right parsing strategy. Image-based PDFs are OCR'd via
Tesseract with parallel page processing before the same pipeline applies.

**Chunking** — word-count approximation of token limits (1 token ≈ 0.75 words),
1 000-token chunks with 200-token overlap to preserve cross-paragraph context.
Chunking is done per speaker segment so every chunk carries the correct
speaker and section label.

**Logging** — both stages write timestamped logs to stdout and append to
`pipeline.log`. The labeling stage logs every OpenAI call individually,
showing chunk ID, speaker, section, token count, how many labels were
extracted, how many had `signal=true`, and elapsed time per chunk and
per document.

---

## Data Quality Reference

Based on 1 491 labeled records across the initial batch:

| Category                                                | Count   | % of total |
| ------------------------------------------------------- | ------- | ---------- |
| `signal = true`                                         | 1 181   | 79.2%      |
| `signal = false`                                        | 310     | 20.8%      |
| **Fully complete** (signal=true + all fields populated) | **807** | **54.1%**  |
| Partial (signal=true, ≥1 field null)                    | 374     | 25.1%      |

Among `signal=true` records, field population rates:

| Field           | Populated | Null | % populated |
| --------------- | --------- | ---- | ----------- |
| `signal`        | 1 181     | 0    | 100%        |
| `reasoning`     | 1 181     | 0    | 100%        |
| `time_horizon`  | 1 165     | 16   | 98.6%       |
| `magnitude`     | 1 149     | 32   | 97.2%       |
| `direction`     | 1 143     | 38   | 96.8%       |
| `confidence`    | 1 103     | 78   | 93.4%       |
| `currency_pair` | 813       | 368  | 68.8%       |

<!-- The 368 records missing `currency_pair` typically describe general FX exposure -->
<!-- (headwinds, constant-currency adjustments) without specifying a pair — -->
<!-- the model correctly returns `null` rather than guessing. -->
