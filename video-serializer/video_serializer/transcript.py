"""Transcript parsing: read an existing SRT or VTT file into transcript segments.

This module is deliberately dependency-free (standard library only) and treats
the transcript as the single source of truth for *when* each piece of spoken
text occurs. It does not perform speech recognition.

Public API:
    TranscriptParseError
    parse_transcript(path) -> list[TranscriptSegment]
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from .models import TranscriptSegment


class TranscriptParseError(Exception):
    """Raised when a transcript file cannot be read or contains no usable cues."""


# Matches a subtitle timing line. Accepts either SRT (comma millisecond
# separator) or VTT (dot separator), with optional hours:
#     00:00:03,000 --> 00:00:06,500     (SRT)
#     00:00:03.000 --> 00:00:06.500     (VTT)
#     00:03.500   --> 00:06.500         (VTT, no hours)
_TIMING_RE = re.compile(
    r"(?:(?P<h1>\d{1,2}):)?(?P<m1>\d{1,2}):(?P<s1>\d{1,2})[,.](?P<ms1>\d{1,3})"
    r"\s*-->\s*"
    r"(?:(?P<h2>\d{1,2}):)?(?P<m2>\d{1,2}):(?P<s2>\d{1,2})[,.](?P<ms2>\d{1,3})"
)


def _ts(hours, minutes, seconds, ms) -> float:
    """Convert timestamp parts into float seconds (milliseconds zero-padded)."""
    h = int(hours) if hours else 0
    m = int(minutes)
    s = int(seconds)
    frac = int((ms or "0").ljust(3, "0")) / 1000.0
    return h * 3600 + m * 60 + s + frac


def _decode(raw: bytes) -> str:
    """Decode file bytes, tolerating a BOM and falling back for legacy encodings."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def parse_transcript(path: str) -> List[TranscriptSegment]:
    """Parse an SRT/VTT file into a sorted, re-indexed list of segments.

    Rules:
      * Format is detected from the file extension (case-insensitive).
      * Cues are located by their timing line; following text lines (up to the
        next blank line or timing line) are joined into a single normalized string.
      * Cues with empty text or non-positive duration are dropped.
      * The result is sorted by start time and re-indexed from 0.
    """
    p = Path(path)
    if not p.is_file():
        raise TranscriptParseError(f"Transcript file not found: {path}")

    ext = p.suffix.lower()
    if ext not in (".srt", ".vtt"):
        raise TranscriptParseError(
            f"Unsupported transcript format '{ext or '(no extension)'}' for '{path}'. "
            "Expected a .srt or .vtt file."
        )

    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise TranscriptParseError(f"Could not read transcript '{path}': {exc}") from exc

    text = _decode(raw)
    if not text.strip():
        raise TranscriptParseError(f"Transcript file is empty: {path}")

    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]

    cues: List[tuple] = []
    i = 0
    n = len(lines)
    while i < n:
        match = _TIMING_RE.search(lines[i])
        if not match:
            i += 1
            continue

        start = _ts(match.group("h1"), match.group("m1"), match.group("s1"), match.group("ms1"))
        end = _ts(match.group("h2"), match.group("m2"), match.group("s2"), match.group("ms2"))

        i += 1
        buf: List[str] = []
        while i < n:
            nxt = lines[i]
            if not nxt.strip():
                break
            if _TIMING_RE.search(nxt):  # next cue started without a blank separator
                break
            buf.append(nxt.strip())
            i += 1

        joined = " ".join(buf)
        seg_text = re.sub(r"\s+", " ", joined).strip()
        cues.append((start, end, seg_text))

    kept = [(s, e, t) for (s, e, t) in cues if t and e > s]
    if not kept:
        raise TranscriptParseError(f"No valid cues found in transcript: {path}")

    kept.sort(key=lambda c: c[0])
    return [
        TranscriptSegment(index=idx, start=float(s), end=float(e), text=t)
        for idx, (s, e, t) in enumerate(kept)
    ]


if __name__ == "__main__":
    import os
    import tempfile

    sample_srt = (
        "1\n00:00:01,000 --> 00:00:03,000\nHello world\n\n"
        "2\n00:00:05,000 --> 00:00:07,500\nSecond cue\n\n"
        "3\n00:00:00,500 --> 00:00:00,900\nout of order, early\n"
    )
    sample_vtt = (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:03.000\nFirst VTT cue\n\n"
        "NOTE this note block must be ignored\n\n"
        "00:00:05.000 --> 00:00:07.500\nSecond VTT cue\n"
    )

    d = tempfile.mkdtemp()
    f_srt = os.path.join(d, "t.srt")
    f_vtt = os.path.join(d, "t.vtt")
    with open(f_srt, "w", encoding="utf-8") as fh:
        fh.write(sample_srt)
    with open(f_vtt, "w", encoding="utf-8") as fh:
        fh.write(sample_vtt)

    srt_segs = parse_transcript(f_srt)
    assert [s.index for s in srt_segs] == [0, 1, 2], srt_segs
    assert [s.text for s in srt_segs] == ["out of order, early", "Hello world", "Second cue"], srt_segs
    assert srt_segs[0].start == 0.5 and abs(srt_segs[1].end - 3.0) < 1e-9

    vtt_segs = parse_transcript(f_vtt)
    assert [s.text for s in vtt_segs] == ["First VTT cue", "Second VTT cue"], vtt_segs
    assert vtt_segs[1].start == 5.0 and abs(vtt_segs[1].end - 7.5) < 1e-9

    # Empty-text cue + bad duration are dropped; unsupported extension raises.
    bad = os.path.join(d, "bad.srt")
    with open(bad, "w", encoding="utf-8") as fh:
        fh.write("1\n00:00:01,000 --> 00:00:01,000\nzero duration\n")
    try:
        parse_transcript(bad)
        raise AssertionError("expected TranscriptParseError for zero-duration-only transcript")
    except TranscriptParseError:
        pass
    try:
        parse_transcript(os.path.join(d, "x.txt"))
        raise AssertionError("expected TranscriptParseError for missing file")
    except TranscriptParseError:
        pass

    print("transcript self-test OK")
