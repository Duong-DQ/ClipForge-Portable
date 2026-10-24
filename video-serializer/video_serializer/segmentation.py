"""Group a timed transcript into coherent topics/episodes.

This is the "brain" of the pipeline: it takes the flat list of
:class:`~video_serializer.models.TranscriptSegment` produced by ``parse_transcript``
and partitions it into a list of :class:`~video_serializer.models.Topic` in time
order. Two interchangeable backends are supported:

- **embedding** -- sentence-transformer embeddings compared against a running
  centroid of the current topic (cosine similarity).
- **heuristic** -- a dependency-free lexical similarity over word tokens and
  character 3-grams. No model, no network.

Both backends cut only *between whole segments*, so the concatenation of all
topic texts always equals the concatenation of all input segment texts (no text
is ever dropped, duplicated or reordered).

After the initial boundary pass the topics are adjusted to honour the duration
constraints in :class:`~video_serializer.models.PipelineConfig`: too-short
topics are merged into a neighbour and too-long topics are force-split at their
best internal boundary.

Importing this module is dependency-free; ``sentence_transformers`` is imported
lazily, and only when the embedding backend is actually used.
"""
from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

from .llm import chat_json, is_configured
from .models import PipelineConfig, Topic, TranscriptSegment

__all__ = ["SegmentationError", "segment_topics"]

# ``\w`` with Unicode semantics keeps non-ASCII letters (Vietnamese, Chinese,
# ...) and drop-kicks punctuation, which is exactly what we want for a
# language-agnostic tokenizer.
_WORD_RE = re.compile(r"\w+", re.UNICODE)

_CHAR_NGRAM = 3

# --------------------------------------------------------------------------- #
# Retention-mode signals (English-focused; stdlib-only)
# --------------------------------------------------------------------------- #
# First-word interrogatives that turn a declarative clip-ending into a question.
_INTERROGATIVES: tuple = (
    "how", "why", "what", "when", "where", "who", "whom", "whose", "which",
    "can", "could", "will", "would", "should",
    "do", "does", "did",
    "is", "are", "was", "were",
    "have", "has", "had",
)

# Cliffhanger / hook phrases. Stored in lowercase; matched against the lowercased
# end of the clip text. Both curly and straight apostrophe variants are listed
# so subtitle transcripts (which vary wildly in apostrophe style) all hit.
_CLIFFHANGERS: tuple = (
    "but",
    "however",
    "here's the thing",
    "heres the thing",
    "the problem is",
    "the catch",
    "surprisingly",
    "interestingly",
    "unfortunately",
    "that's when",
    "thats when",
    "that's where",
    "thats where",
    "the secret",
    "the question is",
    "what if",
    "imagine",
    "now the real question",
    "one more thing",
    "here's where it gets",
    "heres where it gets",
    "but first",
    "the real answer",
    "so why",
    "which is why",
)

# Target clip length for retention mode (~4 minutes). The scorer rewards clips
# that reach this target and gently penalises overshoot; the partitioner still
# honours ``max_topic_duration`` as a hard upper bound.
_RETENTION_TARGET: float = 240.0


class SegmentationError(Exception):
    """Raised when segmentation cannot proceed.

    Typical causes: an empty segment list, or forcing the embedding backend
    while ``sentence_transformers`` is not installed.
    """


# --------------------------------------------------------------------------- #
# Lexical similarity (heuristic backend + max-pass tie-break)
# --------------------------------------------------------------------------- #
def _word_tokens(text: str) -> List[str]:
    """Lowercased word tokens of ``text`` (Unicode-aware, punctuation ignored)."""
    return _WORD_RE.findall(text.lower())


def _char_ngrams(text: str, n: int = _CHAR_NGRAM) -> Set[str]:
    """Set of character ``n``-grams of ``text`` (whitespace collapsed)."""
    collapsed = " ".join(text.lower().split())
    if not collapsed:
        return set()
    if len(collapsed) < n:
        return {collapsed}
    return {collapsed[i:i + n] for i in range(len(collapsed) - n + 1)}


def _jaccard(left: Set[str], right: Set[str]) -> float:
    """Jaccard index of two token/gram sets, in ``0..1``.

    Two empty sets are considered identical (similarity ``1.0``) so that empty
    or punctuation-only segments do not spuriously trigger a boundary.
    """
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _keyword_similarity(a: str, b: str) -> float:
    """Lexical similarity of two strings, in ``0..1``.

    The score is the mean of the Jaccard index over lowercased word tokens and
    the Jaccard index over character 3-grams. Token-only overlap misses
    morphological variants (``star``/``stars``) and CJK runs, while gram-only
    overlap misses word structure; averaging the two keeps the measure
    language-agnostic and robust for short subtitle lines.
    """
    tokens = _jaccard(set(_word_tokens(a)), set(_word_tokens(b)))
    grams = _jaccard(_char_ngrams(a), _char_ngrams(b))
    return (tokens + grams) / 2.0


# --------------------------------------------------------------------------- #
# Embedding helpers (embedding backend only; never imported at module load)
# --------------------------------------------------------------------------- #
def _embedding_available() -> bool:
    """Return ``True`` when ``sentence_transformers`` can be imported."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity of two dense vectors, in ``-1..1`` (``0`` if degenerate)."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        fx = float(x)
        fy = float(y)
        dot += fx * fy
        norm_a += fx * fx
        norm_b += fy * fy
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a ** 0.5 * norm_b ** 0.5)


def _encode_segments(model, segments: List[TranscriptSegment]) -> List[List[float]]:
    """Encode every segment text, returning one float vector per segment."""
    encoded = model.encode([s.text for s in segments], convert_to_numpy=True)
    if hasattr(encoded, "ndim") and encoded.ndim == 1:
        # A single-segment transcript may yield a bare vector instead of a row.
        return [list(map(float, encoded))]
    return [list(map(float, row)) for row in encoded]


# --------------------------------------------------------------------------- #
# Boundary passes
# --------------------------------------------------------------------------- #
def _group_heuristic(
    segments: List[TranscriptSegment], config: PipelineConfig
) -> List[List[int]]:
    """Partition segment indices by lexical similarity to the previous segment."""
    groups: List[List[int]] = [[0]]
    for i in range(1, len(segments)):
        previous = segments[i - 1]
        current = segments[i]
        gap = current.start - previous.end
        similarity = _keyword_similarity(previous.text, current.text)
        if similarity < config.similarity_threshold or gap >= config.gap_threshold:
            groups.append([i])
        else:
            groups[-1].append(i)
    return groups


def _group_embeddings(
    segments: List[TranscriptSegment], config: PipelineConfig
) -> List[List[int]]:
    """Partition segment indices by cosine similarity to the topic centroid."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised via use_embeddings=True
        raise SegmentationError(
            "use_embeddings=True requires the 'sentence-transformers' package, "
            "which is not importable. Install it with: "
            "pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(config.embedding_model)
    vectors = _encode_segments(model, segments)

    groups: List[List[int]] = [[0]]
    centroid = list(vectors[0])
    count = 1  # number of segments currently contributing to ``centroid``
    for i in range(1, len(segments)):
        previous = segments[i - 1]
        current = segments[i]
        gap = current.start - previous.end
        similarity = _cosine_similarity(vectors[i], centroid)
        if similarity < config.similarity_threshold or gap >= config.gap_threshold:
            groups.append([i])
            centroid = list(vectors[i])
            count = 1
        else:
            groups[-1].append(i)
            count += 1
            centroid = [
                (mean * (count - 1) + value) / count
                for mean, value in zip(centroid, vectors[i])
            ]
    return groups


# --------------------------------------------------------------------------- #
# Retention-mode boundary pass (English-focused, stdlib-only)
# --------------------------------------------------------------------------- #
def _retention_score(
    segments: List[TranscriptSegment],
    start: int,
    end_idx: int,
    duration: float,
    config: PipelineConfig,
) -> float:
    """Score how well a candidate clip ``segments[start..end_idx]`` hooks a viewer.

    Higher = better. Components are summed:

    - ``+3.0`` when the closing segment contains a literal ``?`` (clear hook).
    - ``+1.5`` when the closing segment starts with an interrogative word
      (e.g. ``"Why is this..."``) even without punctuation.
    - ``+1.0`` when the tail of the clip text mentions a known cliffhanger
      phrase (``"but"``, ``"however"``, ``"the problem is"``, ...).
    - duration term: rewards length up to ``_RETENTION_TARGET`` (~4 min),
      mildly penalises too-short clips, very lightly penalises overshoot.
    - ``+0.5`` when there is a next segment and the gap to it is a clean
      ``>=0.5s`` pause (an easy ffmpeg cut point).
    """
    score = 0.0
    last = segments[end_idx]
    last_text = last.text or ""

    # Literal question mark anywhere in the closing segment.
    if "?" in last_text:
        score += 3.0

    # Closing segment begins with an interrogative (first word via _WORD_RE).
    first_word_match = _WORD_RE.search(last_text)
    if first_word_match and first_word_match.group(0).lower() in _INTERROGATIVES:
        score += 1.5

    # Cliffhanger phrase in the tail of the clip's combined text (checked once).
    clip_text = " ".join(segments[i].text for i in range(start, end_idx + 1))
    tail = clip_text[-60:].lower() if clip_text else ""
    if tail:
        for phrase in _CLIFFHANGERS:
            if phrase in tail:
                score += 1.0
                break

    # Topic-keyword bonus: when the user picked a topic (Travel / Pranks / ...),
    # prefer ending the clip on one of that topic's cliffhanger phrases. This is
    # only a scoring nudge -- the cut still falls between whole segments, so
    # content is preserved exactly. ``topic == "none"`` (default) is a no-op.
    topic = getattr(config, "topic", "none")
    if topic and topic != "none":
        from .hooks import cut_keywords_for
        kws = cut_keywords_for(topic)
        if kws:
            tail120 = clip_text[-120:].lower() if clip_text else ""
            if tail120 and any(k in tail120 for k in kws):
                score += 1.5

    # Duration term: reward up to target, gently punish short, lightly punish overshoot.
    score += min(duration, _RETENTION_TARGET) * 0.005
    score -= max(0.0, _RETENTION_TARGET - duration) * 0.01
    score -= max(0.0, duration - _RETENTION_TARGET) * 0.003

    # Clean pause to the next segment.
    if end_idx + 1 < len(segments):
        next_seg = segments[end_idx + 1]
        if next_seg.start - last.end >= 0.5:
            score += 0.5

    return score


def _group_retention(
    segments: List[TranscriptSegment], config: PipelineConfig
) -> List[List[int]]:
    """Greedy retention-mode partition: end each clip on the strongest hook.

    From ``start``, scans forward while ``duration <= config.max_topic_duration``,
    remembering the candidate end-index with the highest ``_retention_score``.
    Closes the clip at that best end and continues from ``best_end + 1``.

    Each group is always a contiguous index range (cuts fall only between whole
    segments), so the concatenation of all topic texts still equals the
    concatenation of all segment texts. Forward progress is guaranteed: a group
    always contains at least one segment even when no candidate scored above
    the initial single-segment end.
    """
    n = len(segments)
    groups: List[List[int]] = []
    start = 0
    while start < n:
        best_end = start
        best_score = _retention_score(
            segments, start, start, 0.0, config
        )
        end_idx = start
        while end_idx < n:
            duration = segments[end_idx].end - segments[start].start
            if duration > config.max_topic_duration:
                break
            score = _retention_score(
                segments, start, end_idx, duration, config
            )
            if score > best_score:
                best_score = score
                best_end = end_idx
            end_idx += 1

        # Forward-progress guarantee: at least one segment per group. If the
        # while-loop never found any candidate (e.g. every candidate was over
        # the cap), ``best_end == start`` and we still advance one step.
        if best_end < start:
            best_end = start
        groups.append(list(range(start, best_end + 1)))
        start = best_end + 1
    return groups


# --------------------------------------------------------------------------- #
# Duration constraint passes
# --------------------------------------------------------------------------- #
def _group_duration(group: List[int], segments: List[TranscriptSegment]) -> float:
    """Wall-clock duration of a group: last segment end minus first segment start."""
    return segments[group[-1]].end - segments[group[0]].start


def _merge_across(
    group: List[int], neighbour: List[int], on_left: bool
) -> List[int]:
    """Return the concatenation that keeps segment order intact."""
    return neighbour + group if on_left else group + neighbour


def _enforce_min(
    groups: List[List[int]],
    segments: List[TranscriptSegment],
    config: PipelineConfig,
) -> List[List[int]]:
    """Merge topics shorter than ``min_topic_duration`` into the cheaper neighbour.

    The neighbour yielding the smaller combined duration is preferred. A merge
    that would push the combined topic past ``max_topic_duration`` is skipped,
    leaving the short topic in place.
    """
    i = 0
    while i < len(groups):
        duration = _group_duration(groups[i], segments)
        if duration < config.min_topic_duration and len(groups) > 1:
            has_left = i > 0
            has_right = i + 1 < len(groups)

            best_side = ""
            best_combined = 0.0
            if has_left:
                best_combined = _group_duration(
                    _merge_across(groups[i], groups[i - 1], on_left=True), segments
                )
                best_side = "left"
            if has_right:
                combined = _group_duration(
                    _merge_across(groups[i], groups[i + 1], on_left=False), segments
                )
                if not best_side or combined < best_combined:
                    best_combined = combined
                    best_side = "right"

            if best_side and best_combined <= config.max_topic_duration:
                if best_side == "left":
                    groups[i - 1] = groups[i - 1] + groups[i]
                    del groups[i]
                    i = i - 1
                else:
                    groups[i] = groups[i] + groups[i + 1]
                    del groups[i + 1]
                continue  # re-evaluate the merged topic from the same position
        i += 1
    return groups


def _best_split(group: List[int], segments: List[TranscriptSegment]) -> int:
    """Index of the best internal cut: largest time gap, then lowest similarity.

    Returns a value in ``1..len(group)-1`` so the cut is always between two
    whole segments.
    """
    best_index = 1
    best_gap = float("-inf")
    best_similarity = 1.0
    for j in range(1, len(group)):
        left = segments[group[j - 1]]
        right = segments[group[j]]
        gap = right.start - left.end
        similarity = _keyword_similarity(left.text, right.text)
        if gap > best_gap or (gap == best_gap and similarity < best_similarity):
            best_gap = gap
            best_similarity = similarity
            best_index = j
    return best_index


def _enforce_max(
    groups: List[List[int]],
    segments: List[TranscriptSegment],
    config: PipelineConfig,
) -> List[List[int]]:
    """Recursively force-split topics longer than ``max_topic_duration``."""
    result: List[List[int]] = []

    def split(group: List[int]) -> None:
        if _group_duration(group, segments) <= config.max_topic_duration or len(group) <= 1:
            result.append(group)
            return
        cut = _best_split(group, segments)
        split(group[:cut])
        split(group[cut:])

    for group in groups:
        split(group)
    return result


# --------------------------------------------------------------------------- #
# LLM cut-point path (opt-in, degrades gracefully to the offline backends)
# --------------------------------------------------------------------------- #
_MAX_SIGNAL_TIMES = 12  # how many of each signal to inline in the LLM prompt


def _format_signals(
    source_video: "Optional[str]",
) -> "tuple[str, list[tuple[float, float]]]":
    """Return ``(signals_text, highlights_seed)`` for ``source_video``.

    Runs the audio/visual analyzers (ffmpeg-backed) and produces a compact
    human-readable summary suitable for inclusion in the LLM prompt:

    - Scene cuts at: 12.50s, 34.20s, ...
    - Energy peaks at: 15.05s, 42.30s, ...

    Returns ``("", [])`` when no source video is provided or when both
    analyzers fail (graceful degradation). ``highlights_seed`` is a list of
    ``(start, end)`` pairs computed from the energy peaks (3s windows around
    each peak, padded/clamped to the cut-point extent) -- used as a fallback
    when the LLM is not asked for highlights, or as a sanity-default when it
    returns an empty list.
    """
    if not source_video:
        return ("", [])
    try:
        from . import analysis  # local import keeps module load dependency-free
    except Exception:  # noqa: BLE001 - analysis is purely optional at this stage
        return ("", [])

    lines: list[str] = []
    highlights_seed: list[tuple[float, float]] = []

    try:
        cuts = analysis.scene_cuts(source_video, threshold=0.30) or []
    except Exception:
        cuts = []
    if cuts:
        cuts_sorted = sorted(cuts, key=lambda c: float(c.get("t", 0.0)))
        ts = ", ".join(f"{float(c['t']):.2f}s" for c in cuts_sorted[:_MAX_SIGNAL_TIMES])
        lines.append(f"Scene cuts at: {ts}")

    try:
        env = analysis.audio_energy_envelope(source_video, hop_sec=0.05) or []
    except Exception:
        env = []
    if env:
        times = [float(t) for t, _ in env]
        energy = [float(e) for _, e in env]
        peaks = analysis.beat_peaks(times, energy)
        if peaks:
            ts = ", ".join(f"{p:.2f}s" for p in peaks[:_MAX_SIGNAL_TIMES])
            lines.append(f"Energy peaks at: {ts}")
            # Seed: 3s window centred on each peak (clamped to non-negative).
            for p in peaks[:6]:
                start = max(0.0, p - 1.5)
                end = p + 1.5
                highlights_seed.append((start, end))

    return ("\n".join(lines), highlights_seed)


def _validate_highlights(
    raw: object, cuts: List[int], segments: List[TranscriptSegment]
) -> List[Tuple[float, float]]:
    """Coerce ``raw`` into a list of ``(start, end)`` floats aligned with ``cuts``.

    Accepts either a flat list ``[[s, e], [s, e], ...]`` (one per cut) or a
    single pair ``[s, e]`` broadcast across all cuts. Out-of-range entries
    are clamped to the corresponding cut's segment range; malformed pairs
    are dropped. Returns an empty list on any structural failure.
    """
    if not cuts:
        return []
    n_segments = len(segments)
    if n_segments == 0:
        return []
    # Compute (region_start, region_end) for each cut: the time window whose
    # "most dramatic moments" the highlights should be picked from.
    region_bounds: list[tuple[float, float]] = []
    prev_idx = 0
    for cut in cuts:
        region_start = segments[prev_idx].start
        region_end = segments[cut].end
        region_bounds.append((region_start, region_end))
        prev_idx = cut + 1

    out: list[tuple[float, float]] = []
    pairs_input: list = []
    if isinstance(raw, list):
        if len(raw) == 2 and all(isinstance(x, (int, float)) for x in raw):
            # Broadcast a single [s, e] to every cut.
            for _ in cuts:
                pairs_input.append(list(raw))
        else:
            for item in raw:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    pairs_input.append(list(item))
                else:
                    pairs_input.append(None)  # placeholder; will be skipped
    elif isinstance(raw, dict) and isinstance(raw.get("highlights"), list):
        # Tolerate the model wrapping the array under "highlights" too.
        return _validate_highlights(raw["highlights"], cuts, segments)

    for i, (region_start, region_end) in enumerate(region_bounds):
        if i >= len(pairs_input):
            break
        pair = pairs_input[i]
        if pair is None:
            continue
        try:
            s = float(pair[0])
            e = float(pair[1])
        except (TypeError, ValueError, IndexError):
            continue
        if e <= s:
            continue
        # Clamp into the region's actual time range (and never negative).
        if s < region_start:
            s = region_start
        if e > region_end:
            e = region_end
        if e - s < 0.5:
            continue  # pair is too short to be useful
        out.append((s, e))

    return out


def _llm_cut_points(
    segments: List[TranscriptSegment],
    config: PipelineConfig,
    source_video: "Optional[str]" = None,
) -> "tuple[list[int] | None, list[tuple[float, float]] | None]":
    """Ask the LLM to choose content-driven cut indices; ``None`` on any failure.

    Builds a compact transcript (per segment: ``i\\tstart-end\\ttext``), asks the
    model to split into ~3-5 min clips that end at real topic/narrative changes
    or hook/question/cliffhanger endings, and parses a strict JSON
    ``{"cuts": [c1, c2, ...], "highlights": [[s, e], ...]}`` response where:

    - each ``ci`` is the 0-based index of the LAST segment of clip ``i``.
    - ``highlights[i]`` is a ``[start_sec, end_sec]`` pair of the most
      dramatic 3-5s moment **inside** the region covered by clip ``i`` (used
      by the video-style recap stage to assemble a video-highlight intro from
      the previous clip). The highlights array has the same length as ``cuts``
      (one entry per cut); the last clip has no preceding cut and therefore
      receives no recap, but the highlights array still covers every cut.

    When ``source_video`` is provided, the prompt is enriched with audio/
    visual signals (scene cuts + energy peaks) extracted via
    :mod:`video_serializer.analysis`. Signal extraction failures are silent:
    the prompt just omits that section.

    Returns the validated ``(cuts, highlights)`` tuple on success,
    ``(None, None)`` otherwise. Never raises -- every error path returns the
    failure tuple.
    """
    n = len(segments)
    if n == 0:
        return (None, None)

    lines: list[str] = []
    for seg in segments:
        text = (seg.text or "").replace("\t", " ").replace("\n", " ").strip()
        lines.append(f"{seg.index}\t{seg.start:.2f}-{seg.end:.2f}\t{text}")
    transcript = "\n".join(lines)

    signals_text, _seed = _format_signals(source_video)
    signals_section = ""
    if signals_text:
        signals_section = (
            "\nAUDIO/VISUAL SIGNALS (from ffmpeg analysis of the source video):\n"
            + signals_text
            + "\n"
        )

    system = "You split video transcripts into short-form clips and return strict JSON only."
    user = (
        "Split the following timed transcript into ~3-5 minute clips.\n"
        "Rules:\n"
        "  - Cut ONLY at segment boundaries (use the segment index).\n"
        "  - Prefer cuts at real topic/narrative changes or at strong "
        "hook / question / cliffhanger endings.\n"
        "  - 'cuts' is the list of 0-based indices of the LAST segment of each "
        "clip. The final clip ends at the last segment automatically.\n"
        + signals_section
        + "In your response, ALSO include a 'highlights' array of the form "
        "\"highlights\": [[start_sec, end_sec], ...] with one [start, end] pair "
        "per cut (each 3-5 seconds long), covering the most dramatic moment "
        "inside that clip's region. Use the audio/visual signals above as a "
        "hint when picking the moments.\n"
        "Return STRICT JSON of the form: "
        "{\"cuts\": [c1, c2, ...], \"highlights\": [[s1, e1], [s2, e2], ...]} "
        "and nothing else.\n\n"
        "Transcript:\n" + transcript
    )

    obj = chat_json(config, system, user, temperature=0.0, max_tokens=600)
    if not isinstance(obj, dict):
        return (None, None)
    raw_cuts = obj.get("cuts")
    if not isinstance(raw_cuts, list):
        return (None, None)

    validated: list[int] = []
    last = -1
    for item in raw_cuts:
        # Coerce to int, drop anything non-numeric / out of range.
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value < 0 or value >= n:
            continue
        if value <= last:
            # Not strictly increasing -- drop; do not break so the model can
            # still produce a valid tail if the early entries are messy.
            continue
        validated.append(value)
        last = value

    if not validated:
        return (None, None)

    # Highlights are optional but desirable; missing/garbage highlights
    # don't fail the whole call -- the offline path in ``add_recaps`` will
    # degrade to the text card.
    raw_highlights = obj.get("highlights")
    highlights = _validate_highlights(raw_highlights, validated, segments)
    return (validated, highlights)


def _groups_from_cuts(cuts: List[int], n: int) -> List[List[int]]:
    """Convert a list of clip-ending indices into contiguous index groups.

    ``cuts = [c1, c2, ...]`` with ``0 <= c1 < c2 < ... < n-1`` yields::

        [0..c1], [c1+1..c2], ..., [last+1..n-1]

    Edge guards: if the first cut is past ``n-1`` we get a single empty group
    and return an empty list (caller should treat as failure); if the last cut
    is the final segment there is no trailing partial clip.
    """
    if not cuts or n <= 0:
        return []
    groups: List[List[int]] = []
    start = 0
    for cut in cuts:
        if cut >= n:
            break
        if cut < start:
            # Defensive: should not happen if _llm_cut_points validated properly.
            continue
        groups.append(list(range(start, cut + 1)))
        start = cut + 1
    if start < n:
        groups.append(list(range(start, n)))
    return groups


# --------------------------------------------------------------------------- #
# Viral selective mode (LLM picks + ranks the top-N standalone moments)
# --------------------------------------------------------------------------- #
def _validate_viral_clips(
    raw: object, segments: List[TranscriptSegment], config: PipelineConfig
) -> "list[dict]":
    """Coerce + validate the LLM's viral-clip selection into a clean list.

    ``raw`` is the parsed ``"clips"`` array; each entry is a dict with
    ``start``/``end`` (0-based segment indices, inclusive), ``score`` (0-100) and
    an optional ``hook``. Returns a list of
    ``{"start": int, "end": int, "score": float, "hook": str}`` dicts that are:

      * clamped to valid segment ranges (``0 <= start <= end < len(segments)``),
      * clamped so each clip's wall-clock duration <= ``config.max_topic_duration``
        (``end`` is shrunk when needed),
      * de-duplicated so no two selected moments overlap (higher score wins),
      * sorted by score descending (most viral first),
      * capped at ``config.max_clips``.

    Pure and network-free so it can be unit-tested. Returns ``[]`` on any
    structural failure (mirrors the never-raise style of the LLM helpers).
    """
    if not segments or not isinstance(raw, list):
        return []
    n = len(segments)
    max_dur = float(config.max_topic_duration)
    candidates: "list[dict]" = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            start = int(item.get("start"))
            end = int(item.get("end"))
        except (TypeError, ValueError):
            continue
        if start < 0 or end < start or end >= n:
            continue
        # Shrink ``end`` so the clip duration never exceeds the hard max.
        while end > start and (segments[end].end - segments[start].start) > max_dur:
            end -= 1
        if end < start:
            continue
        try:
            score = float(item.get("score", 50.0))
        except (TypeError, ValueError):
            score = 50.0
        score = max(0.0, min(100.0, score))
        hook = str(item.get("hook", "") or "").strip()[:80]
        candidates.append({"start": start, "end": end, "score": score, "hook": hook})

    # Most viral first.
    candidates.sort(key=lambda c: c["score"], reverse=True)

    # Drop overlapping selections (keep the higher-score one, which sorts first).
    accepted: "list[dict]" = []
    for cand in candidates:
        if any(cand["start"] <= a["end"] and cand["end"] >= a["start"] for a in accepted):
            continue
        accepted.append(cand)

    max_clips = int(getattr(config, "max_clips", 5) or 5)
    if max_clips > 0:
        accepted = accepted[:max_clips]
    return accepted


def _build_viral_topics(
    clips: "list[dict]", segments: List[TranscriptSegment]
) -> List[Topic]:
    """Materialise validated viral clips into :class:`Topic` objects.

    ``clips`` is already sorted by score descending (most viral first), so the
    resulting topics carry ``index`` ``0..N-1`` in virality order (topic 0 = the
    most viral clip). Each clip is a contiguous segment range; different clips
    may be separated by gaps (skipped content) -- that is the point of the
    selector, so no merge/split passes are applied here.
    """
    topics: List[Topic] = []
    for rank, clip in enumerate(clips):
        members = [segments[i] for i in range(clip["start"], clip["end"] + 1)]
        if not members:
            continue
        topics.append(
            Topic(
                index=rank,
                start=members[0].start,
                end=members[-1].end,
                segments=members,
                score=clip["score"],
                hook=clip["hook"],
            )
        )
    return topics


def _llm_viral_clips(
    segments: List[TranscriptSegment],
    config: PipelineConfig,
    source_video: "Optional[str]" = None,
) -> "list[dict]":
    """Ask the LLM to SELECT the top-N most engaging standalone moments.

    Unlike :func:`_llm_cut_points` (which partitions the whole transcript into
    contiguous clips), this sends the transcript to the model and asks it to pick
    ONLY the best self-contained moments -- a selective, non-contiguous ranking
    of short-form clips, each with a 0-100 virality score and a hook line.
    Returns the validated clip list from :func:`_validate_viral_clips`, or
    ``[]`` on any failure. Never raises.
    """
    n = len(segments)
    if n == 0:
        return []

    lines: "list[str]" = []
    for seg in segments:
        text = (seg.text or "").replace("\t", " ").replace("\n", " ").strip()
        lines.append(f"{seg.index}\t{seg.start:.2f}-{seg.end:.2f}\t{text}")
    transcript = "\n".join(lines)

    max_clips = int(getattr(config, "max_clips", 5) or 5)
    min_s = int(getattr(config, "min_topic_duration", 20) or 20)
    max_s = int(getattr(config, "max_topic_duration", 300) or 300)

    signals_text, _seed = _format_signals(source_video)
    signals_section = ""
    if signals_text:
        signals_section = (
            "\nAUDIO/VISUAL SIGNALS (from ffmpeg analysis of the source video):\n"
            + signals_text + "\n"
        )

    system = (
        "You select the most engaging standalone short-form moments from a video "
        "transcript and return strict JSON only."
    )
    user = (
        "From the timed transcript below, SELECT the "
        + str(max_clips) + " MOST ENGAGING, self-contained moments that would "
        "work as standalone short-form clips (roughly "
        + str(min_s) + "-" + str(max_s) + " seconds each).\n"
        "Rules:\n"
        "  - DO NOT cover the whole video. Pick ONLY the best moments; skip filler.\n"
        "  - Each moment must be a complete standalone unit (hook -> body -> payoff) "
        "understandable on its own, without the rest of the video.\n"
        "  - Use segment indices: 'start' = 0-based index of the FIRST segment, "
        "'end' = 0-based index of the LAST segment of the moment (start <= end).\n"
        "  - 'score' = 0-100, how viral/engaging it is as a standalone clip.\n"
        "  - 'hook' = a short punchy opening line/caption phrase (max ~10 words).\n"
        "  - Sort 'clips' by score, best first.\n"
        + signals_section
        + "Return STRICT JSON of the form: "
        '{"clips": [{"start": i, "end": j, "score": 0-100, "hook": "..."}, ...]} '
        "and nothing else.\n\n"
        "Transcript:\n" + transcript
    )

    obj = chat_json(config, system, user, temperature=0.0, max_tokens=800)
    if not isinstance(obj, dict):
        return []
    return _validate_viral_clips(obj.get("clips"), segments, config)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _attach_highlights(
    topics: List[Topic],
    highlights: "Optional[List[Tuple[float, float]]]",
    segments: List[TranscriptSegment],
) -> None:
    """Copy ``highlights`` (one entry per cut) onto ``topics`` in place.

    ``highlights[i]`` describes the dramatic moment inside the time region
    covered by ``topics[i]`` (which ends at the i-th cut). When the lists do
    not line up -- fewer highlights than topics, or the LLM returned none --
    the tail of topics keeps an empty ``highlights`` list. Out-of-range
    highlights are silently dropped.
    """
    if not topics or not highlights:
        return
    for index, topic in enumerate(topics):
        if index >= len(highlights):
            break
        start, end = highlights[index]
        # Final safety net: clamp into the topic's actual time range and drop
        # degenerate pairs. ``enforce_min`` / ``enforce_max`` may have shifted
        # the topic bounds slightly after the LLM call.
        if start < topic.start:
            start = float(topic.start)
        if end > topic.end:
            end = float(topic.end)
        if end - start < 0.5:
            continue
        topic.highlights = [(float(start), float(end))]


def segment_topics(
    segments: List[TranscriptSegment],
    config: PipelineConfig,
    source_video: "Optional[str]" = None,
) -> List[Topic]:
    """Group ``segments`` into time-ordered :class:`Topic` objects.

    Parameters
    ----------
    segments:
        Transcript segments in time order (as produced by ``parse_transcript``).
    config:
        Pipeline tunables; ``use_embeddings`` selects the backend and the
        duration/similarity thresholds drive boundary and merge decisions.
    source_video:
        Optional path to the original video file. When provided AND the LLM
        backend is selected, the LLM prompt is enriched with audio/visual
        signals extracted via :mod:`video_serializer.analysis` and the
        returned highlights are attached to the produced topics for the
        video-style recap stage. Always safe to omit (falls back to no
        signals, no highlights).

    Returns
    -------
    List[Topic]
        Topics in time order with ``index`` ``0..N-1``, ``segments`` populated,
        and ``start``/``end`` set from the first/last segment. ``title``,
        ``summary``, ``file``, and ``highlights`` are left empty for later
        pipeline stages (highlights only populated when the LLM path ran
        with a ``source_video`` AND the model returned valid highlights).

    Raises
    ------
    SegmentationError
        If ``segments`` is empty, or the embedding backend is requested but
        ``sentence_transformers`` is not installed.
    """
    if not segments:
        raise SegmentationError("Cannot segment an empty transcript (no segments).")

    # Viral selective mode: the LLM picks the top-N most engaging self-contained
    # moments (non-contiguous, may skip content) and ranks them by a 0-100 score.
    # This is a SELECTOR, not a partitioner, so unlike the paths below it does not
    # cover 100% of the transcript. When the LLM is unavailable or returns
    # nothing, fall back to the hook-based ``retention`` partition.
    if getattr(config, "cut_strategy", "topic") == "viral":
        if is_configured(config):
            clips = _llm_viral_clips(segments, config, source_video)
            if clips:
                topics = _build_viral_topics(clips, segments)
                print(
                    "[segmentation] viral mode: selected %d clip(s) (ranked by score)"
                    % len(topics)
                )
                return topics
        print(
            "[segmentation] viral mode needs the LLM (key+url); "
            "falling back to 'retention' hook mode"
        )
        groups = _group_retention(segments, config)
        groups = _enforce_min(groups, segments, config)
        groups = _enforce_max(groups, segments, config)
        return _build_topics(groups, segments)

    # Opt-in: when the LLM is configured, ask it for content-driven cut points.
    # Any failure (LLM disabled, no key, call failed, malformed response) falls
    # through to the offline backends below so the pipeline never crashes.
    if is_configured(config):
        cuts, highlights = _llm_cut_points(segments, config, source_video)
        if cuts:
            groups = _groups_from_cuts(cuts, len(segments))
            if groups:
                groups = _enforce_min(groups, segments, config)
                groups = _enforce_max(groups, segments, config)
                topics = _build_topics(groups, segments)
                _attach_highlights(topics, highlights, segments)
                print(
                    "[segmentation] using LLM cut points: %d clip(s), "
                    "%d highlight(s)" % (len(groups), len(highlights or []))
                )
                return topics

    # Retention strategy overrides similarity/embedding grouping entirely: cuts
    # are placed where the heuristic hook score is highest, then re-checked
    # against min/max duration in the passes below.
    if getattr(config, "cut_strategy", "topic") == "retention":
        groups = _group_retention(segments, config)
    elif config.use_embeddings is True:
        groups = _group_embeddings(segments, config)
    elif config.use_embeddings is False:
        groups = _group_heuristic(segments, config)
    elif _embedding_available():
        groups = _group_embeddings(segments, config)
    else:
        groups = _group_heuristic(segments, config)

    groups = _enforce_min(groups, segments, config)
    groups = _enforce_max(groups, segments, config)
    return _build_topics(groups, segments)


def _build_topics(
    groups: List[List[int]], segments: List[TranscriptSegment]
) -> List[Topic]:
    """Materialise index-groups into :class:`Topic` objects in time order."""
    topics: List[Topic] = []
    for index, group in enumerate(groups):
        members = [segments[i] for i in group]
        topics.append(
            Topic(
                index=index,
                start=members[0].start,
                end=members[-1].end,
                segments=members,
            )
        )
    return topics


# --------------------------------------------------------------------------- #
# Offline self-test (no network, no model, no third-party imports)
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Exercise the heuristic backend end-to-end on a synthetic transcript."""
    topic_a = [  # ~kitchen / cooking
        "cook the recipe in the kitchen",
        "cook the recipe using kitchen ingredients",
        "cook the recipe with the kitchen ingredients",
        "cook the recipe and the kitchen ingredients",
    ]
    topic_b = [  # ~football / sport
        "the football team scores a goal",
        "the football team scores a goal again",
        "the football team plays the match",
        "the football team wins the match",
        "the football team ends the match",
    ]
    topic_c = [  # ~space / astronomy
        "the planet and the star in space",
        "the planet and the galaxy in space",
        "the galaxy and the star in space",
        "the planet and the star in the galaxy",
        "the planet and the star and the galaxy",
    ]

    segments: List[TranscriptSegment] = []
    cursor = 0.0
    for group_index, texts in enumerate((topic_a, topic_b, topic_c)):
        if group_index > 0:
            cursor += 3.0  # ~3s gap between topics
        for text in texts:
            start = cursor
            end = start + 1.0
            segments.append(
                TranscriptSegment(index=len(segments), start=start, end=end, text=text)
            )
            cursor = end

    config = PipelineConfig(
        use_embeddings=False, min_topic_duration=3.0, max_topic_duration=30.0
    )
    topics = segment_topics(segments, config)

    # (a) time order, contiguous, indices 0..N-1, bounds from first/last segment
    assert topics, "expected at least one topic"
    for expected_index, topic in enumerate(topics):
        assert topic.index == expected_index, (
            f"topic index {topic.index} != {expected_index}"
        )
        assert topic.start == topic.segments[0].start
        assert topic.end == topic.segments[-1].end
    for previous, current in zip(topics, topics[1:]):
        assert previous.end <= current.start, "topics out of order or overlapping"

    # (b) content preserved exactly
    expected_text = " ".join(s.text for s in segments)
    actual_text = " ".join(t.text for t in topics)
    assert actual_text == expected_text, "segment text was dropped or reordered"

    # (c) no topic exceeds max duration
    for topic in topics:
        assert topic.duration <= 30.0, f"topic duration {topic.duration} > 30.0"

    # (d) at least 2 topics produced
    assert len(topics) >= 2, f"expected >= 2 topics, got {len(topics)}"

    # title/summary must be left for the enrichment stage
    for topic in topics:
        assert topic.title == "" and topic.summary == ""

    # ----- retention-mode self-test -----
    # Synthetic English transcript: two loosely-related paragraphs with a
    # question mark, a "but" cliffhanger, and a long flat stretch. The
    # similarity score on these sentences is high (shared "machine learning"
    # vocabulary), so the default "topic" strategy would happily merge them
    # all; retention mode must still cut somewhere and must preserve content.
    english_texts = [
        "we begin with a quick overview of machine learning fundamentals",
        "machine learning is a field that keeps evolving every year",
        "neural networks are at the heart of modern machine learning",
        "training data quality matters more than model size",
        "the trick is to balance bias and variance carefully",
        "but here is the part nobody tells you about machine learning",
        "why does overfitting ruin so many production models in practice",
        "regularization and cross validation can fix most of these problems",
        "in production you also need to monitor for data drift over time",
        "the real answer is to keep iterating and measure everything you ship",
        "now the real question is which metric actually tracks user value",
        "imagine you could detect drift before users even notice it",
    ]
    english_segments: List[TranscriptSegment] = []
    cursor = 0.0
    for text in english_texts:
        start = cursor
        end = start + 6.0  # 6s per line -> 72s total transcript
        english_segments.append(
            TranscriptSegment(index=len(english_segments), start=start, end=end, text=text)
        )
        cursor = end

    retention_config = PipelineConfig(
        cut_strategy="retention",
        use_embeddings=False,
        min_topic_duration=1.0,
        max_topic_duration=120.0,
    )
    retention_topics = segment_topics(english_segments, retention_config)

    # (a) at least one topic
    assert retention_topics, "retention mode produced no topics"
    # (b) every topic within max duration
    for topic in retention_topics:
        assert topic.duration <= 120.0, (
            f"retention topic duration {topic.duration} > max_topic_duration"
        )
    # (c) content preservation: joined topic text == joined segment text
    # (whitespace-insensitive -- topic.text joins segments with single spaces
    # but the segment texts themselves may carry arbitrary whitespace).
    def _flatten_texts(items) -> str:
        chunks = []
        for item in items:
            chunks.append(" ".join((item.text or "").split()))
        return " ".join(chunks).strip()

    assert _flatten_texts(retention_topics) == _flatten_texts(english_segments), (
        "retention mode dropped / duplicated / reordered segment text"
    )
    # (d) retention groups must still be contiguous index ranges (sanity)
    expected_index = 0
    for topic in retention_topics:
        for offset, seg in enumerate(topic.segments):
            assert seg.index == expected_index + offset, (
                f"retention topic breaks contiguity at segment index "
                f"{seg.index} (expected {expected_index + offset})"
            )
        expected_index = topic.segments[-1].index + 1

    print("PASS")


# --------------------------------------------------------------------------- #
# Pure offline tests for the LLM cut-point helpers (NO network, NO key)
# --------------------------------------------------------------------------- #
def _test_llm_helpers() -> None:
    """Exercise ``_groups_from_cuts`` and the cut-list validator (no I/O)."""
    # (a) Contiguous groups with the canonical cuts [c1, c2, ...] -> [0..c1],
    # [c1+1..c2], ..., [last+1..n-1]. Content (joined text) is preserved.
    segments = [
        TranscriptSegment(index=0, start=0.0, end=1.0, text="alpha one"),
        TranscriptSegment(index=1, start=1.0, end=2.0, text="alpha two"),
        TranscriptSegment(index=2, start=2.0, end=3.0, text="beta one"),
        TranscriptSegment(index=3, start=3.0, end=4.0, text="beta two"),
        TranscriptSegment(index=4, start=4.0, end=5.0, text="gamma one"),
        TranscriptSegment(index=5, start=5.0, end=6.0, text="gamma two"),
    ]
    cuts = [1, 3]  # clip1=0..1, clip2=2..3, clip3=4..5
    groups = _groups_from_cuts(cuts, len(segments))
    assert groups == [[0, 1], [2, 3], [4, 5]], f"got {groups!r}"
    # content preserved in time order across the partition
    joined = " ".join(segments[i].text for g in groups for i in g)
    assert joined == " ".join(s.text for s in segments), "content dropped"

    # (b) Last cut == n-1 collapses to a single trailing group with no leftovers.
    groups_tail = _groups_from_cuts([2, 5], len(segments))
    assert groups_tail == [[0, 1, 2], [3, 4, 5]], f"got {groups_tail!r}"

    # (c) Empty cuts = "no cuts at all" (caller falls back to offline); n=0 is
    # also a no-op so we never produce a phantom group.
    assert _groups_from_cuts([], len(segments)) == []
    assert _groups_from_cuts([0], 0) == []

    # (d) Out-of-range cut beyond n-1 is truncated; content after the cap still
    # forms a trailing group so nothing is dropped.
    groups_truncated = _groups_from_cuts([1, 99], len(segments))
    assert groups_truncated == [[0, 1], [2, 3, 4, 5]], f"got {groups_truncated!r}"

    print("PASS")


def _test_viral_helpers() -> None:
    """Exercise the viral-mode pure helpers (no network, no key)."""
    # 12 segments, 1s each, contiguous from 0s.
    segments = [
        TranscriptSegment(index=i, start=float(i), end=float(i) + 1.0, text=f"line {i}")
        for i in range(12)
    ]
    config = PipelineConfig(
        cut_strategy="viral", max_clips=3,
        min_topic_duration=1.0, max_topic_duration=5.0,
    )
    raw = [
        {"start": 0, "end": 9, "score": 80, "hook": "A"},   # 10s -> clamped to end 4 (5s)
        {"start": 1, "end": 3, "score": 95, "hook": "B"},   # overlaps clamped A; B wins (95>80)
        {"start": 6, "end": 8, "score": 70, "hook": "C"},
        {"start": 10, "end": 11, "score": 60, "hook": "D"},
        "garbage",                                           # dropped (not a dict)
        {"start": 50, "end": 60, "score": 99},               # dropped (out of range)
    ]
    clips = _validate_viral_clips(raw, segments, config)
    # A is dropped (overlaps higher-score B); result sorted by score desc, capped at 3.
    assert [c["score"] for c in clips] == [95.0, 70.0, 60.0], clips
    assert (clips[0]["start"], clips[0]["end"]) == (1, 3), clips
    assert (clips[1]["start"], clips[1]["end"]) == (6, 8), clips
    assert (clips[2]["start"], clips[2]["end"]) == (10, 11), clips
    # no two selected clips may overlap
    for x in range(len(clips)):
        for y in range(x + 1, len(clips)):
            assert not (
                clips[x]["start"] <= clips[y]["end"]
                and clips[x]["end"] >= clips[y]["start"]
            ), "overlap"
    # build topics: virality-ordered indices, contiguous members, gaps allowed.
    topics = _build_viral_topics(clips, segments)
    assert len(topics) == 3, topics
    assert topics[0].index == 0 and topics[0].score == 95.0, topics
    assert topics[0].start == 1.0 and topics[0].end == 4.0, topics
    assert topics[0].segments[0].index == 1 and topics[0].segments[-1].index == 3, topics
    # gap between topic0 (segs 1..3) and topic1 (segs 6..8) is allowed
    assert topics[1].start == 6.0, topics
    # structural failures -> []
    assert _validate_viral_clips(None, segments, config) == []
    assert _validate_viral_clips([], segments, config) == []
    assert _validate_viral_clips("not a list", segments, config) == []

    print("PASS")


if __name__ == "__main__":
    _self_test()
    _test_llm_helpers()
    _test_viral_helpers()
