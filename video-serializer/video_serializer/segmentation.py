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
from typing import List, Set

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
def _llm_cut_points(
    segments: List[TranscriptSegment], config: PipelineConfig
) -> "list[int] | None":
    """Ask the LLM to choose content-driven cut indices; ``None`` on any failure.

    Builds a compact transcript (per segment: ``i\\tstart-end\\ttext``), asks the
    model to split into ~3-5 min clips that end at real topic/narrative changes
    or hook/question/cliffhanger endings, and parses a strict JSON
    ``{"cuts": [c1, c2, ...]}`` response where each ``ci`` is the 0-based index
    of the LAST segment of that clip.

    Returns the validated list (strictly increasing, in ``[0, n)``) on success,
    ``None`` otherwise. Never raises -- every error path returns ``None``.
    """
    n = len(segments)
    if n == 0:
        return None

    lines: list[str] = []
    for seg in segments:
        text = (seg.text or "").replace("\t", " ").replace("\n", " ").strip()
        lines.append(f"{seg.index}\t{seg.start:.2f}-{seg.end:.2f}\t{text}")
    transcript = "\n".join(lines)

    system = "You split video transcripts into short-form clips and return strict JSON only."
    user = (
        "Split the following timed transcript into ~3-5 minute clips.\n"
        "Rules:\n"
        "  - Cut ONLY at segment boundaries (use the segment index).\n"
        "  - Prefer cuts at real topic/narrative changes or at strong "
        "hook / question / cliffhanger endings.\n"
        "  - 'cuts' is the list of 0-based indices of the LAST segment of each "
        "clip. The final clip ends at the last segment automatically.\n"
        "Return STRICT JSON of the form: {\"cuts\": [c1, c2, ...]} and nothing else.\n\n"
        "Transcript:\n" + transcript
    )

    obj = chat_json(config, system, user, temperature=0.0, max_tokens=400)
    if not isinstance(obj, dict):
        return None
    raw_cuts = obj.get("cuts")
    if not isinstance(raw_cuts, list):
        return None

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

    return validated if validated else None


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
# Public API
# --------------------------------------------------------------------------- #
def segment_topics(
    segments: List[TranscriptSegment], config: PipelineConfig
) -> List[Topic]:
    """Group ``segments`` into time-ordered :class:`Topic` objects.

    Parameters
    ----------
    segments:
        Transcript segments in time order (as produced by ``parse_transcript``).
    config:
        Pipeline tunables; ``use_embeddings`` selects the backend and the
        duration/similarity thresholds drive boundary and merge decisions.

    Returns
    -------
    List[Topic]
        Topics in time order with ``index`` ``0..N-1``, ``segments`` populated,
        and ``start``/``end`` set from the first/last segment. ``title``,
        ``summary`` and ``file`` are left empty for later pipeline stages.

    Raises
    ------
    SegmentationError
        If ``segments`` is empty, or the embedding backend is requested but
        ``sentence_transformers`` is not installed.
    """
    if not segments:
        raise SegmentationError("Cannot segment an empty transcript (no segments).")

    # Opt-in: when the LLM is configured, ask it for content-driven cut points.
    # Any failure (LLM disabled, no key, call failed, malformed response) falls
    # through to the offline backends below so the pipeline never crashes.
    if is_configured(config):
        cuts = _llm_cut_points(segments, config)
        if cuts:
            groups = _groups_from_cuts(cuts, len(segments))
            if groups:
                groups = _enforce_min(groups, segments, config)
                groups = _enforce_max(groups, segments, config)
                print(
                    "[segmentation] using LLM cut points: %d clip(s)"
                    % len(_groups_from_cuts(cuts, len(segments)))
                )
                return _build_topics(groups, segments)

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


if __name__ == "__main__":
    _self_test()
    _test_llm_helpers()
