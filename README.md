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
├── requirements.txt
├── .env.template       # Copy to .env and add your key
└── data/
    ├── transcripts/    # ← Drop your PDFs here
    ├── preprocessed/   # ← Per-transcript JSONs (auto-created)
    └── labeled/
        └── training_data.json   # ← Final output
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

## Usage

```bash
# Stage 1: Preprocess all PDFs
python run.py preprocess

# Stage 1: Preprocess a single PDF
python run.py preprocess --file data/transcripts/suncor_energy_q2_2024_ect.pdf

# Stage 2: Label all preprocessed files (resumes from last run by default)
python run.py label

# Stage 2: Re-label everything from scratch
python run.py label --no-resume

# Both stages sequentially
python run.py all
```

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
    "avg_chunk_length": 410
  }
}
```

### Labeled Training Data (`data/labeled/training_data.json`)

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
      "reasoning": "Company reported 4% revenue headwind...",
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

## Key Design Decisions

- **Forex filtering**: Only chunks with `has_forex=true` are sent to OpenAI,
  saving API cost. All chunks are still stored in the preprocessed JSON for
  full traceability via `chunk_id`.

- **Resume support**: `python run.py label` checks which documents are already
  in `training_data.json` and skips them. Use `--no-resume` to redo all.

- **Speaker identification**: Uses regex with role-keyword fallback to handle
  inconsistent transcript formats across documents.

- **Chunking**: Word-count-based approximation of token limits
  (1 token ≈ 0.75 words), 1000-token chunks with 200-token overlap to
  preserve cross-paragraph context.
