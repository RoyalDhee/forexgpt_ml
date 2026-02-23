"""
run.py — Pipeline entry point.

Usage:
    # ── Preprocessing ──────────────────────────────────────────────────────────
    python run.py preprocess                           # All PDFs
    python run.py preprocess --limit 5                 # First 5 files only
    python run.py preprocess --skip 32                 # Skip first 32, process rest
    python run.py preprocess --skip 32 --limit 5       # Skip 32, then next 5
    python run.py preprocess --pattern apple           # Only files matching 'apple'
    python run.py preprocess --pattern apple --limit 3 # First 3 apple files
    python run.py preprocess --file path/to/file.pdf   # Single file by path

    # ── Labeling ───────────────────────────────────────────────────────────────
    python run.py label                                # All preprocessed docs
    python run.py label --limit 3                      # First 3 docs only
    python run.py label --skip 5                       # Skip first 5, label rest
    python run.py label --skip 5 --limit 3             # Skip 5, then next 3
    python run.py label --pattern sony                 # Only docs matching 'sony'
    python run.py label --pattern sony --limit 2       # First 2 sony docs
    python run.py label --no-resume                    # Re-label everything from scratch

    # ── Both stages ────────────────────────────────────────────────────────────
    python run.py all --limit 2                        # Preprocess + label, 2 docs
    python run.py all --pattern apple                  # Both stages for apple docs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def run_preprocess(
    single_file: str | None = None,
    limit:       int | None = None,
    skip:        int | None = None,
    pattern:     str | None = None,
) -> None:
    from preprocessor import PreprocessingAgent
    agent = PreprocessingAgent()

    if single_file:
        pdf = Path(single_file)
        if not pdf.exists():
            print(f"File not found: {pdf}")
            sys.exit(1)
        doc = agent.process(pdf)
        path = agent.save(doc)
        print(f"Preprocessed -> {path}")
    else:
        outputs = agent.run_all(limit=limit, skip=skip, pattern=pattern)
        print(f"\nPreprocessing complete. {len(outputs)} file(s) written.")


def run_label(
    resume:  bool = True,
    limit:   int | None = None,
    skip:    int | None = None,
    pattern: str | None = None,
) -> None:
    from labeler import LabelingAgent
    agent = LabelingAgent()
    agent.run_all(resume=resume, limit=limit, skip=skip, pattern=pattern)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Earnings Transcript FX Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "stage",
        choices=["preprocess", "label", "all"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--file",
        help="(preprocess only) Process a single PDF file by path",
        default=None,
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="(label only) Re-label all documents, ignoring existing per-doc label files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N documents (after --skip / --pattern filtering)",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=None,
        metavar="N",
        help="Skip the first N files in sorted order before applying --limit",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        metavar="TEXT",
        help=(
            "Only process files whose filename contains TEXT (case-insensitive). "
            "Works for both preprocess and label stages. "
            "E.g. --pattern apple, --pattern sony"
        ),
    )

    args = parser.parse_args()

    if args.stage in ("preprocess", "all"):
        run_preprocess(
            single_file=args.file,
            limit=args.limit,
            skip=args.skip,
            pattern=args.pattern,
        )

    if args.stage in ("label", "all"):
        run_label(
            resume=not args.no_resume,
            limit=args.limit,
            skip=args.skip,
            pattern=args.pattern,
        )


if __name__ == "__main__":
    main()


# # python run.py preprocess --skip 67 --limit 1
# # python run.py preprocess --pattern Apple --limit 3
# # python run.py preprocess --pattern Apple
# python run.py label --limit 2
