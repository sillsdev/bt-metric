"""Where the SIL NLP data bucket is mounted: SIL_NLP_DATA_PATH, as in silnlp, or M:/ (its usual Windows mount)."""

import os
from pathlib import Path

DATA_DIR = Path(os.getenv("SIL_NLP_DATA_PATH") or "M:/")
PARATEXT_DIR = DATA_DIR / "Paratext" / "projects"
MT_DIR = DATA_DIR / "MT"
EXPERIMENTS_DIR = MT_DIR / "experiments"
SCRIPTURE_DIR = MT_DIR / "scripture"
TERMS_DIR = MT_DIR / "terms"
EBIBLE_SCRIPTURE_DIR = DATA_DIR / "eBible" / "MT" / "scripture"
