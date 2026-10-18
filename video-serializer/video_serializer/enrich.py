"""Topic enrichment: assign a concise ``title`` and ``summary`` to each topic.

The pipeline runs this step between segmentation and cutting; downstream the
cutter builds output filenames from ``Topic.title``, so every topic must always
end up with a non-empty title.

Two backends are supported:

* **Heuristic** (always available, offline, standard-library only) -- the first
  sentence becomes the title and the first one-to-two sentences become the
  summary, each cleaned and truncated.
* **LLM** (opt-in) -- when ``config.use_llm`` is true *and* both the base URL
  and API key are non-empty, an OpenAI-compatible ``/chat/completions`` endpoint
  is queried with :mod:`urllib.request`. Any failure (network error, non-200,
  malformed JSON, missing/empty fields) silently falls back to the heuristic for
  that topic, so the pipeline never crashes because of the optional LLM.

Deliberately language-agnostic: content may be Vietnamese, Chinese or English,
so sentence splitting understands both Latin (``.!?``) and CJK (``。！？``)
terminators and truncation never assumes space-delimited words.
"""
from __future__ import annotations

import json
import re
import urllib.request
from typing import List, Optional, Tuple

from .models import PipelineConfig, Topic

__all__ = ["enrich_topics"]

#: Sentence boundary: a run of terminators (Latin + CJK, keeping any trailing
#: quotes/brackets) followed by whitespace. The boundary is captured so the
#: text on either side can be recovered with the same regex.
_SENTENCE_RE = re.compile(r"""([.!?。！？]+["')\]]*\s+)""")

#: Characters stripped from the ends of titles/summaries: whitespace plus stray
#: punctuation (Latin and CJK quotes, brackets, commas, dashes, ...).
_TRIM_CHARS = (
    " \t\r\n"
    "\"'`\u201c\u201d\u2018\u2019\u300c\u300d\u300e\u300f"
    "()[]{}<>\uff08\uff09\u300a\u300b\u3010\u3011"
    ".,;:!?;\u3002\uff0c\u3001\uff1b\uff1a\uff01\uff1f\u2026-\u2014\u2013"
)

_MAX_TITLE = 70
_MAX_SUMMARY = 220
_ELLIPSIS = "\u2026"


def _clean(text: str) -> str:
    """Collapse whitespace and strip leading/trailing punctuation + spaces."""
    if not text:
        return ""
    collapsed = re.sub(r"\s+", " ", text)
    return collapsed.strip(_TRIM_CHARS).strip()


def _truncate(text: str, limit: int) -> str:
    """Trim ``text`` to at most ``limit`` characters, breaking at a word edge."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = cut.rfind(" ")
    # Only honour the word boundary when it is not pathologically early (CJK
    # text has no spaces, so we simply hard-cut in that case).
    if boundary >= limit // 2:
        cut = cut[:boundary]
    return cut.rstrip(_TRIM_CHARS) + _ELLIPSIS


def _first_sentence(text: str) -> str:
    """Return the first sentence of ``text`` (whole trimmed text if none)."""
    text = (text or "").strip()
    if not text:
        return ""
    parts = _SENTENCE_RE.split(text, maxsplit=1)
    first = parts[0].strip()
    return first or text


def _first_sentences(text: str, count: int = 2) -> str:
    """Return the first ``count`` sentences of ``text`` joined by a space."""
    text = (text or "").strip()
    if not text:
        return ""
    parts = _SENTENCE_RE.split(text)
    sentences: List[str] = []
    # ``parts`` alternates: sentence, separator, sentence, separator, ...
    for i in range(0, len(parts), 2):
        chunk = parts[i].strip()
        if chunk:
            sentences.append(chunk)
        if len(sentences) >= count:
            break
    return " ".join(sentences) if sentences else text


def _heuristic_title(topic: Topic) -> str:
    """First sentence of the topic, cleaned and truncated; ``Episode N`` fallback."""
    title = _clean(_first_sentence(topic.text))
    if not title:
        title = f"Episode {topic.index + 1}"
    return _truncate(title, _MAX_TITLE)


def _heuristic_summary(topic: Topic) -> str:
    """First one-to-two sentences of the topic, cleaned and truncated."""
    summary = _clean(_first_sentences(topic.text, 2))
    if not summary:
        summary = _clean(_first_sentence(topic.text))
    if not summary:
        summary = _heuristic_title(topic)
    return _truncate(summary, _MAX_SUMMARY)


def _strip_code_fences(text: str) -> str:
    """Remove a surrounding markdown code fence (```` ```json ... ``` ````)."""
    text = (text or "").strip()
    if text.startswith("```"):
        newline = text.find("\n")
        text = text[newline + 1 :] if newline != -1 else text.lstrip("`")
    if text.rstrip().endswith("```"):
        text = text.rstrip()[:-3]
    return text.strip()


def _llm_prompt(episode_text: str) -> str:
    """Build the chat prompt requesting a JSON title/summary object."""
    return (
        "You are labeling a single episode of a long video.\n"
        "Read the episode transcript text and reply with ONLY a JSON object of "
        'the form {"title": "...", "summary": "..."}.\n'
        "- title: a concise, human, descriptive episode title.\n"
        "- summary: a 1-2 sentence summary of the episode.\n"
        "- Write BOTH in the same language as the transcript text.\n"
        "- No markdown, no code fences, no extra keys, no commentary.\n\n"
        "Episode text:\n" + episode_text
    )


def _llm_request_body(config: PipelineConfig, prompt: str) -> bytes:
    """Serialize the OpenAI-compatible chat request body to UTF-8 bytes."""
    body = {
        "model": config.llm_model,
        "temperature": 0,
        "max_tokens": 140,
        "messages": [{"role": "user", "content": prompt}],
    }
    return json.dumps(body).encode("utf-8")


def _llm_enrich(topic: Topic, config: PipelineConfig) -> Optional[Tuple[str, str]]:
    """Query the LLM for ``(title, summary)`` or return ``None`` on any failure.

    Never raises: every error path (network, HTTP, JSON, missing/empty fields)
    is swallowed and reported as ``None`` so the caller can fall back.
    """
    try:
        episode_text = topic.text
        if not episode_text:
            return None
        url = config.llm_base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            url,
            data=_llm_request_body(config, _llm_prompt(episode_text)),
            headers={
                "Authorization": f"Bearer {config.llm_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read().decode("utf-8", "replace")
        parsed = json.loads(payload)
        content = parsed["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            return None
        obj = json.loads(_strip_code_fences(content))
        if not isinstance(obj, dict):
            return None
        title = _clean(str(obj.get("title", "")))
        summary = _clean(str(obj.get("summary", "")))
        if not title or not summary:
            return None
        return _truncate(title, _MAX_TITLE), _truncate(summary, _MAX_SUMMARY)
    except Exception:
        return None


def enrich_topics(topics: List[Topic], config: PipelineConfig) -> List[Topic]:
    """Fill ``title`` and ``summary`` for every topic (in place) and return them.

    Uses the LLM when it is enabled and configured, falling back to the offline
    heuristic whenever the LLM is disabled, unconfigured, or fails. The input
    list is mutated so each ``Topic.title`` is always non-empty for downstream
    filename generation.
    """
    for topic in topics:
        title = ""
        summary = ""
        if config.use_llm and config.llm_base_url and config.llm_api_key:
            result = _llm_enrich(topic, config)
            if result is not None:
                title, summary = result
        if not title:
            title = _heuristic_title(topic)
        if not summary:
            summary = _heuristic_summary(topic)
        topic.title = title.strip()
        topic.summary = summary.strip()
    return topics


if __name__ == "__main__":
    from .models import TranscriptSegment

    def _make_topic(index: int, sentences: List[str]) -> Topic:
        segments = [
            TranscriptSegment(index=i, start=float(i), end=float(i) + 1.0, text=s)
            for i, s in enumerate(sentences)
        ]
        return Topic(index=index, start=0.0, end=float(len(sentences)), segments=segments)

    vietnamese = _make_topic(
        0,
        [
            "Hôm nay chúng ta sẽ nói về cách nấu phở ngon tại nhà.",
            "Bí quyết nằm ở nước dùng hầm xương trong nhiều giờ.",
        ],
    )
    english = _make_topic(
        1,
        [
            "In this episode we explore the basics of machine learning.",
            "We cover supervised learning, overfitting, and evaluation metrics.",
        ],
    )
    topics = [vietnamese, english]

    enriched = enrich_topics(topics, PipelineConfig())

    assert enriched is topics, "enrich_topics must return the same list"
    for topic in enriched:
        assert topic.title, f"topic {topic.index} has an empty title"
        assert topic.summary, f"topic {topic.index} has an empty summary"
        assert len(topic.title) <= 80, f"title too long: {topic.title!r}"
        assert len(topic.summary) <= 240, f"summary too long: {topic.summary!r}"

    print("PASS")
