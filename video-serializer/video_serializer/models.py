"""Shared data types for the video serializer pipeline.

Every pipeline module imports its inputs/outputs from here so the modules stay
loosely coupled and can be developed/tested independently.

Conventions
-----------
- All timestamps/durations are **float seconds** throughout the pipeline.
- ``index`` on :class:`TranscriptSegment` is 0-based in time order.
- ``Topic.title`` / ``Topic.summary`` are filled by ``enrich``; ``Topic.file``
  is filled by ``cutter`` after the clips are written.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TranscriptSegment:
    """A single timed unit from the transcript (typically one subtitle line).

    Attributes
    ----------
    index:
        Position in time order, 0-based.
    start:
        Start time in seconds.
    end:
        End time in seconds (>= start).
    text:
        Transcript text for this segment, whitespace-normalized.
    """

    index: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Topic:
    """A coherent slice of the transcript/video that becomes one episode clip.

    ``segments`` are the original transcript segments that make up this topic,
    kept in time order. ``title`` and ``summary`` default to empty and are
    populated by the enrichment step; ``file`` is populated by the cutter.
    ``intro_seconds`` records the length (seconds) of the optional recap intro
    prepended to the output clip (0.0 when no recap was added), so the actual
    output file is ``duration + intro_seconds``.
    """

    index: int
    start: float
    end: float
    segments: List[TranscriptSegment] = field(default_factory=list)
    title: str = ""
    summary: str = ""
    file: str = ""
    intro_seconds: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def text(self) -> str:
        """Full transcript text of this topic (segments joined in time order)."""
        return " ".join(s.text for s in self.segments).strip()


@dataclass
class PipelineConfig:
    """Tunables for the whole pipeline. All durations are in seconds.

    Segmentation
    ------------
    min_topic_duration:
        Topics shorter than this (seconds) are merged into a neighbor so the
        output never contains clips that are too short to be useful.
    max_topic_duration:
        Topics longer than this (seconds) are force-split so a single clip is
        never excessively long.
    gap_threshold:
        A silent gap between two consecutive segments that is >= this many
        seconds is a strong candidate for a topic boundary.
    similarity_threshold:
        A similarity score (0..1) between adjacent content that drops below
        this value indicates a topic change. Higher = fewer, larger topics.
    use_embeddings:
        ``None`` (default) auto-detects: use sentence-transformer embeddings
        if the package is importable, otherwise fall back to the heuristic.
        ``True`` forces the embedding path (raises a clear error if the
        package is missing); ``False`` forces the heuristic path.
    embedding_model:
        Hugging Face model name for the embedding backend (must be
        multilingual to handle Vietnamese/Chinese/English).

    Cutting
    -------
    cut_mode:
        ``"accurate"`` (default) re-encodes for frame-accurate cuts;
        ``"fast"`` stream-copies for speed (boundaries snap to keyframes).
    cut_strategy:
        ``"topic"`` (default) = current content-similarity splitting (unchanged).
        ``"retention"`` = cut ~3-5 min clips that end on questions/cliffhangers.

    Recap intros (optional, post-cut)
    ---------------------------------
    add_recaps:
        When True, every clip from the 2nd onward gets a short intro prepended
        to its start that recaps the previous clip (a text card, plus spoken
        audio via edge-tts when reachable). Clip 1 is never changed.
    recap_duration:
        Length (seconds) of each recap intro (clamped to >= 1).
    recap_voice:
        edge-tts voice name for the spoken recap. Empty (default) auto-picks a
        voice matching the recap language (vi-VN / en-US / zh-CN). The intro
        silently falls back to no audio if TTS is unavailable.
    recap_lang:
        Language for the recap intro card labels (header + "Episode N:").
        "auto" (default) detects it from the transcript; or force "vi"/"en"/"zh".

    Enrichment / optional LLM
    -------------------------
    use_llm:
        When True, request concise titles/summaries from an
        OpenAI-compatible chat endpoint. Falls back to the heuristic if the
        call fails or the endpoint is not configured.
    llm_base_url:
        Base URL of the OpenAI-compatible API (e.g. ``http://host/v1``).
    llm_api_key:
        API key (Bearer token) for the LLM endpoint.
    llm_model:
        Model identifier to request from the endpoint.

    Output
    ------
    metadata_format:
        ``"json"`` (default) or ``"csv"`` for the generated metadata file.
    """

    # --- segmentation ---
    min_topic_duration: float = 20.0
    max_topic_duration: float = 300.0
    gap_threshold: float = 1.5
    similarity_threshold: float = 0.45
    use_embeddings: Optional[bool] = None
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # --- cutting ---
    cut_mode: str = "accurate"
    cut_strategy: str = "topic"  # "topic" = similarity split (default); "retention" = hook-ending clips
    topic: str = "none"  # topic key for hook-based cutting + sensational teaser (see hooks.json); "none" = off

    # --- recap intros (optional, post-cut) ---
    add_recaps: bool = False
    recap_duration: float = 4.0
    recap_voice: str = ""  # empty = auto-pick a voice matching the recap language
    recap_lang: str = "auto"  # recap card labels: auto / vi / en / zh

    # --- enrichment / optional LLM ---
    use_llm: bool = False
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    # --- output ---
    metadata_format: str = "json"
