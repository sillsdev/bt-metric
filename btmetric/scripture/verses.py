"""Verse references and verse-aligned Scripture text, paired the way silnlp pairs them for training."""

import re
from typing import AbstractSet, List, Sequence, Tuple

_VREF = re.compile(r"^(\S+) (\d+):(\d+)(?:-(\d+))?$")


def expand_vref(vref: str) -> List[str]:
    """The single verses of a reference that may be a range ("MAT 13:5-6")."""
    match = _VREF.match(vref)
    if not match:
        return [vref]
    book, chapter, first, last = match.group(1), match.group(2), int(match.group(3)), match.group(4)
    return [f"{book} {chapter}:{v}" for v in range(first, int(last or first) + 1)]


def parallel_verses(
    vrefs: Sequence[str], source_lines: Sequence[str], target_lines: Sequence[str], excluded: AbstractSet[str]
) -> List[Tuple[str, str, str]]:
    """(reference, source, target) for each verse both texts have, as silnlp's get_scripture_parallel_corpus pairs
    them: a verse one text combines with the one before it ("<range>") joins that verse's row, whose reference becomes
    a range ("GEN 1:1-2"). Rows missing either text or holding "..." are dropped, and so is any row with an excluded
    verse."""
    rows: List[List[str]] = []  # first verse, last verse, source, target
    for vref, source, target in zip(vrefs, source_lines, target_lines):
        source, target = source.strip(), target.strip()
        if "<range>" not in (source, target) or not rows:
            rows.append([vref, vref, source, target])
            continue
        row = rows[-1]
        if _chapter(vref) == _chapter(row[0]):
            row[1] = vref
        if source != "<range>":
            row[2] = f"{row[2]} {source}".strip()
        if target != "<range>":
            row[3] = f"{row[3]} {target}".strip()
    verses = []
    for first, last, source, target in rows:
        vref = first if last == first else f"{first}-{last.split(':')[-1]}"
        if source in ("", "...") or target in ("", "...") or any(v in excluded for v in expand_vref(vref)):
            continue
        verses.append((vref, source, target))
    return verses


def _chapter(vref: str) -> str:
    return vref.split(":")[0]
