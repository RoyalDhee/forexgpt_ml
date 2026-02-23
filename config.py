"""
config.py — Central configuration for the earnings transcript pipeline.
"""

from pathlib import Path

#  Directory layout
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
LABELED_DIR = DATA_DIR / "labeled"

for _d in (TRANSCRIPTS_DIR, PREPROCESSED_DIR, LABELED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

LABELED_OUTPUT = LABELED_DIR / "training_data.json"

#  OpenAI model config
OPENAI_MODEL = "gpt-4o-mini"
MAX_TOKENS = 1024

#  Chunking
CHUNK_TOKENS = 1000
OVERLAP_TOKENS = 200

#  Ticker mapping (company name fragment → ticker)
TICKER_MAP: dict[str, str] = {
    "sony":                  "SONY",
    "suncor":                "SU",
    "toyota":                "TM",
    "sap":                   "SAP",
    "shell":                 "SHEL",
    "alphabet":              "GOOGL",
    "google":                "GOOGL",
    "amazon":                "AMZN",
    "apple":                 "AAPL",
    "bhp":                   "BHP",
    "microsoft":             "MSFT",
    "procter":               "PG",
    "pg":                    "PG",
    "shopify":               "SHOP",
    "siemens":               "SIEGY",
    "unilever":              "UL",
    "taiwan semiconductor":  "TSM",
    "tsmc":                  "TSM",
    "rio tinto":             "RIO",
    "rio":                   "RIO",
    "bumble":                "BMBL",
    "coca cola":             "KO",
    "coca-cola":             "KO",
    "exxon":                 "XOM",
    "intel":                 "INTC",
    "louis vuitton":         "LVMH",
    "lvmh":                  "LVMH",
    "nestle":                "NSRGY",
    "nestlé":                "NSRGY",
    "novartis":              "NVS",
    "pfizer":                "PFE",
}

#  Forex keyword filter (chunks must contain at least one)
# FOREX_KEYWORDS = [
#     # Explicit FX mechanisms — must imply currency movement/exposure, not just money
#     "foreign exchange", "forex", "fx rate", "fx impact", "fx headwind", "fx tailwind",
#     "exchange rate", "currency headwind", "currency tailwind", "currency impact",
#     "currency movement", "currency fluctuation", "currency exposure",
#     "currency risk", "currency translation", "constant currency",
#     "local currency", "reported currency", "functional currency",
#     # Hedging — only meaningful in FX context
#     "currency hedge", "currency hedg", "hedged our", "fx hedge",
#     "forward contract", "currency swap",
#     # FX directional language
#     "dollar strength", "dollar weakness", "dollar headwind", "dollar tailwind",
#     "euro weakness", "euro strength", "yen weakness", "yen strength",
#     "pound weakness", "pound strength",
#     "strengthening of the dollar", "weakening of the dollar",
#     "strengthened versus", "weakened versus",
#     "u.s. dollar strengthened", "u.s. dollar weakened",
#     # Impact phrasing (specific enough to avoid false positives)
#     "headwind from currency", "tailwind from currency",
#     "headwind from foreign", "tailwind from foreign",
#     "impact from foreign exchange", "impact from fx",
#     "impact from currency", "point headwind", "point tailwind",
#     "3 point headwind", "2 point headwind", "3 point tailwind", "2 point tailwind",
#     "spot rates", "current spot rates",
#     # Specific currency pair mentions (precise, not just "dollar")
#     "eur/usd", "usd/jpy", "gbp/usd", "usd/cny", "usd/cad", "usd/aud",
#     "dollar versus", "dollar vs", "versus the euro", "versus the yen",
#     "versus the pound", "versus the dollar",
# ]

FOREX_KEYWORDS = [

    # =========================
    # Explicit FX mechanisms
    # =========================
    "foreign exchange", "forex", "fx rate", "fx impact", "fx headwind", "fx tailwind",
    "exchange rate", "currency headwind", "currency tailwind", "currency impact",
    "currency movement", "currency fluctuation", "currency exposure",
    "currency risk", "currency translation", "constant currency",
    "local currency", "reported currency", "functional currency",

    # =========================
    # CFO-style institutional phrasing (CRITICAL ADDITION)
    # =========================
    "excluding the impact of foreign exchange",
    "excluding the impact of currency",
    "excluding fx",
    "on a constant currency basis",
    "on a reported basis",
    "reduced by currency",
    "offset by currency",
    "drag from fx",
    "benefit from fx",
    "currency was a drag",
    "currency was a benefit",
    "fx was a drag",
    "fx was a benefit",
    "foreign currency impact",
    "foreign currency translation",
    "translation impact",
    "translation loss",
    "translation gain",

    # =========================
    # Basis point / percentage impact phrasing
    # =========================
    "basis point headwind",
    "basis point tailwind",
    "bps headwind",
    "bps tailwind",
    "percentage point headwind",
    "percentage point tailwind",
    "growth was impacted by fx",
    "growth was impacted by currency",
    "revenue was impacted by fx",
    "revenue was impacted by currency",

    # =========================
    # Hedging terminology
    # =========================
    "currency hedge", "currency hedg", "hedged our", "fx hedge",
    "forward contract", "currency swap",
    "cross currency swap",
    "foreign exchange forward",
    "hedging program",
    "hedging strategy",

    # =========================
    # FX directional language
    # =========================
    "dollar strength", "dollar weakness", "dollar headwind", "dollar tailwind",
    "euro weakness", "euro strength", "yen weakness", "yen strength",
    "pound weakness", "pound strength",
    "strengthening of the dollar", "weakening of the dollar",
    "strengthened versus", "weakened versus",
    "u.s. dollar strengthened", "u.s. dollar weakened",
    "appreciation of the dollar",
    "depreciation of the dollar",
    "appreciated versus",
    "depreciated versus",

    # =========================
    # Impact phrasing
    # =========================
    "headwind from currency", "tailwind from currency",
    "headwind from foreign", "tailwind from foreign",
    "impact from foreign exchange", "impact from fx",
    "impact from currency", "point headwind", "point tailwind",
    "3 point headwind", "2 point headwind", "3 point tailwind", "2 point tailwind",
    "spot rates", "current spot rates",
    "based on current spot rates",
    "at current exchange rates",

    # =========================
    # Constant currency comparison triggers
    # =========================
    "up in constant currency",
    "down in constant currency",
    "growth in constant currency",
    "decline in constant currency",
    "compared to constant currency",

    # =========================
    # Balance sheet FX exposure (ADVANCED ADDITION)
    # =========================
    "foreign denominated debt",
    "foreign currency denominated",
    "repatriation of foreign cash",
    "accumulated other comprehensive income",
    "oci impact",
    "remeasurement gain",
    "remeasurement loss",

    # =========================
    # Specific currency pair mentions
    # =========================
    "eur/usd", "usd/jpy", "gbp/usd", "usd/cny", "usd/cad", "usd/aud",
    "dollar versus", "dollar vs", "versus the euro", "versus the yen",
    "versus the pound", "versus the dollar",
]
