"""Topic-aware hook loader for retention-cutting + sensational teasers.

Reads ``hooks.json`` (lives next to the project root, sibling of
``video_serializer/``) and exposes a tiny, dependency-free accessor layer so the
rest of the pipeline can:

- pick a topic from CLI/web (see :func:`topic_options`)
- score retention-mode clips against a topic's cliffhanger keywords
  (see :func:`cut_keywords_for`)
- build a sensational "Watch Part 2!" intro card per episode
  (see :func:`pick_teaser`)

The file is loaded **once** and cached in :data:`_CACHE` so repeated lookups do
not re-hit disk; any failure (missing file, invalid JSON) degrades to an empty
dict, which makes every accessor a safe no-op -- the pipeline then behaves
exactly as if no topic had been selected.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HOOKS_PATH: Path = Path(__file__).resolve().parent.parent / "hooks.json"

# Module-level cache; ``None`` means "not loaded yet". A second call returns the
# same dict (no disk re-read, no per-call I/O).
_CACHE: Optional[Dict[str, dict]] = None


def load_hooks() -> dict:
    """Read ``HOOKS_PATH`` once and return the parsed topic dict.

    Returns an empty dict on *any* error (missing file, permission error,
    invalid JSON, non-dict root). The result is cached in :data:`_CACHE` so a
    second call is O(1) and never raises.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    try:
        text = HOOKS_PATH.read_text(encoding="utf-8")
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            _CACHE = {}
            return _CACHE
        # Keep only dict-shaped entries so downstream helpers can assume a
        # uniform shape; bad entries are silently dropped (better than crashing
        # the whole pipeline).
        cleaned: Dict[str, dict] = {}
        for key, value in parsed.items():
            if isinstance(value, dict):
                cleaned[str(key)] = value
        _CACHE = cleaned
    except (OSError, ValueError, json.JSONDecodeError):
        _CACHE = {}
    return _CACHE


def topic_options() -> List[Tuple[str, str]]:
    """Return ``[(key, label), ...]`` for every topic defined in ``hooks.json``.

    File order is preserved. The label falls back to the key when an entry has
    no ``"label"`` field. Returns an empty list when no hooks file is loaded.
    """
    hooks = load_hooks()
    options: List[Tuple[str, str]] = []
    for key, entry in hooks.items():
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label", key))
        options.append((str(key), label))
    return options


def get_topic(topic: str) -> Optional[dict]:
    """Return the topic entry for ``topic`` (``None`` when missing/invalid)."""
    entry = load_hooks().get(topic)
    return entry if isinstance(entry, dict) else None


def cut_keywords_for(topic: str) -> List[str]:
    """Return the topic's lowercased cliffhanger phrases, dropping empties.

    Returns an empty list when the topic is missing or has no keywords; the
    segmentation scorer treats an empty list as "no bonus", which is the
    correct off-switch.
    """
    entry = get_topic(topic)
    if entry is None:
        return []
    keywords = entry.get("cut_keywords") or []
    return [str(k).lower() for k in keywords if str(k).strip()]


def pick_teaser(topic: str, index: int) -> str:
    """Return ``teasers[index % len(teasers)]`` for ``topic``; ``""`` if none.

    ``index`` is clamped via modulo so callers can pass any non-negative
    episode number; an empty string is returned when the topic or its teaser
    list is missing/empty (lets the caller fall back to the neutral recap).
    """
    entry = get_topic(topic)
    if entry is None:
        return ""
    teasers = [str(t) for t in (entry.get("teasers") or []) if str(t).strip()]
    if not teasers:
        return ""
    return teasers[index % len(teasers)]


if __name__ == "__main__":
    # Offline self-test: assume ``hooks.json`` ships with the project.
    data = load_hooks()
    assert isinstance(data, dict) and data, "hooks.json missing or empty"
    for key in ("travel", "pranks", "sports"):
        assert key in data, f"hooks.json missing topic: {key}"
        entry = data[key]
        assert isinstance(entry, dict)
        assert "label" in entry and "cut_keywords" in entry and "teasers" in entry, (
            f"hooks.json entry {key!r} missing required keys"
        )

    opts = topic_options()
    assert len(opts) == 3, f"expected 3 topic options, got {len(opts)}"

    first_travel = pick_teaser("travel", 0)
    travel_kws = cut_keywords_for("travel")
    assert isinstance(travel_kws, list) and travel_kws, "travel keywords empty"
    assert all(k == k.lower() for k in travel_kws), "travel keywords not lowercased"
    travel_entry = data["travel"]
    first_teaser_expected = str(travel_entry["teasers"][0])
    assert first_travel == first_teaser_expected, (
        f"pick_teaser('travel', 0) = {first_travel!r}, expected {first_teaser_expected!r}"
    )

    assert get_topic("nonexistent") is None, "get_topic('nonexistent') must be None"

    print("hooks self-test OK")
