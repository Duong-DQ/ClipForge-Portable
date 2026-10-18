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

from .models import PipelineConfig, Topic, TranscriptSegment

__all__ = ["SegmentationError", "segment_topics"]

# ``\w`` with Unicode semantics keeps non-ASCII letters (Vietnamese, Chinese,
# ...) and drop-kicks punctuation, which is exactly what we want for a
# language-agnostic tokenizer.
_WORD_RE = re.compile(r"\w+", re.UNICODE)

_CHAR_NGRAM = 3


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

    if config.use_embeddings is True:
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

    print("PASS")


if __name__ == "__main__":
    _self_test()
