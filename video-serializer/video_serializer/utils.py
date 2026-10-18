"""Small shared helpers: timestamp formatting and safe output filenames.

These are used by ``cutter`` (filenames) and ``cli`` (metadata timestamps).
Kept dependency-free and pure so they are trivially testable.
"""
from __future__ import annotations

import re
import unicodedata

_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


def format_timestamp(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS.d`` (one decimal), clamping negatives to 0.

    >>> format_timestamp(0.0)
    '00:00:00.0'
    >>> format_timestamp(123.4)
    '00:02:03.4'
    >>> format_timestamp(3661.0)
    '01:01:01.0'
    """
    if seconds is None or seconds < 0:
        seconds = 0.0
    whole = int(seconds)
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    tenths = int(round((seconds - whole) * 10))
    if tenths == 10:
        tenths = 0
        secs += 1
        if secs == 60:
            secs = 0
            minutes += 1
            if minutes == 60:
                minutes = 0
                hours += 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{tenths}"


def make_slug(text: str, max_len: int = 40) -> str:
    """Build a filesystem-safe ASCII slug from possibly non-Latin text.

    Strips diacritics (``chào`` -> ``chao``) and removes non-alphanumerics.
    Falls back to ``"untitled"`` when nothing ASCII remains (e.g. pure CJK).
    """
    if not text:
        return "untitled"
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = _SLUG_RE.sub("-", ascii_text).strip("-").lower()
    slug = slug[:max_len].rstrip("-")
    return slug or "untitled"


def build_filename(index: int, title: str) -> str:
    """Return ``episode_<NN>_<slug>.mp4`` for a 1-based episode index."""
    return f"episode_{index:02d}_{make_slug(title)}.mp4"
