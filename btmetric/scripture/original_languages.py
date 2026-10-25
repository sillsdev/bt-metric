"""Builds one verse-aligned extract of the original-language Bible from unfoldingWord's openly licensed Hebrew OT (UHB)
and Greek NT (UGNT), for scoring coverage against the original text."""

import argparse
import logging
import re
import unicodedata
from pathlib import Path

LOGGER = logging.getLogger(__name__)

SCRIPTURE_DIR = Path("M:/MT/scripture")
HEBREW = "hbo-hbo_uhb_2024_07_10"
GREEK = "grc-el_x_koine_ugnt_2024_07_11"
OUTPUT = Path("data/scripture/orig-UHB_UGNT.txt")

# UHB joins prefixes (and, the, in...) to their word with a word joiner, and compounds with a maqaf.
_WORD_SEPARATORS = re.compile("[\u2060\u05be]")
# Paseq and nun hafukha are reading marks with no counterpart in a translation.
_READING_MARKS = re.compile("[\u05c0\u05c6]")


def plain(text: str) -> str:
    """Without accents, vowel points or cantillation, which split one word into many forms the aligner sees rarely, and
    with UHB's prefixes as separate words, as many target languages write them."""
    # A few UGNT elisions use a modifier-letter apostrophe, which counts as a letter and stays attached to the word.
    text = _READING_MARKS.sub("", _WORD_SEPARATORS.sub(" ", text)).replace("ʼ", "’")
    # UGNT marks questions with both the Greek question mark (";" after NFC) and "?".
    text = unicodedata.normalize("NFC", text).replace(";", "?")
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return " ".join(unicodedata.normalize("NFC", stripped).split())


def combine(hebrew: list, greek: list) -> list:
    """Each verse from whichever testament has it; the two extracts don't overlap."""
    return [plain(h) if h.strip() else plain(g) for h, g in zip(hebrew, greek)]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scripture-dir", type=Path, default=SCRIPTURE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    read = lambda name: (args.scripture_dir / f"{name}.txt").read_text(encoding="utf-8-sig").splitlines()
    lines = combine(read(HEBREW), read(GREEK))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info(f"Wrote {sum(1 for line in lines if line)} verses to {args.output}")


if __name__ == "__main__":
    main()
