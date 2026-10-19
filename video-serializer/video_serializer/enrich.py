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

# A standard ~120-word English stopword set used by the offline extractive
# summary. Lower-cased. Keeping this inline (rather than downloading NLTK) keeps
# the core offline and stdlib-only.
_EN_STOPWORDS: set = {
    "a", "about", "above", "across", "after", "again", "against", "all",
    "almost", "along", "already", "also", "although", "always", "am", "among",
    "an", "and", "another", "any", "anyone", "anything", "are", "as", "at",
    "back", "be", "because", "been", "before", "being", "between", "both",
    "but", "by", "came", "can", "cannot", "could", "did", "do", "does",
    "doing", "done", "down", "during", "each", "either", "else", "enough",
    "etc", "even", "ever", "every", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "however", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "later", "least", "less", "let",
    "like", "made", "make", "many", "may", "me", "might", "more", "most",
    "much", "must", "my", "myself", "neither", "never", "no", "nor",
    "not", "now", "of", "off", "on", "once", "one", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "per", "perhaps",
    "quite", "rather", "really", "said", "same", "say", "says", "she",
    "should", "since", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "thing", "things", "this", "those", "though", "through",
    "thus", "to", "too", "under", "until", "up", "upon", "us", "very",
    "was", "we", "well", "were", "what", "when", "where", "which", "while",
    "who", "whom", "whose", "why", "will", "with", "within", "without",
    "would", "yes", "yet", "you", "your", "yours", "yourself", "yourselves",
}


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


def _sentences(text: str) -> list[str]:
    """Split ``text`` into sentences using the existing ``_SENTENCE_RE``.

    Reuses the same Latin + CJK terminator regex used elsewhere in this module
    so the splitter behaves identically to the title/summary code paths. Empty
    fragments (from runs of whitespace) are dropped.
    """
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    # ``parts`` alternates: sentence, separator, sentence, separator, ...
    result: list[str] = []
    for i in range(0, len(parts), 2):
        chunk = parts[i].strip()
        if chunk:
            result.append(chunk)
    return result


def _extractive_summary(topic: Topic) -> str:
    """Pick the 2-3 most content-bearing sentences of ``topic.text``.

    A standard frequency-based extractive summariser (TextRank-adjacent,
    without the graph): build word frequencies over lowercased, non-stopword
    tokens across all sentences, score each sentence by the sum of its
    non-stopword frequencies, then length-normalise by the count of non-stop
    words to ``**0.6`` so a long bag-of-words sentence is not unfairly rewarded
    over a shorter, denser one. The 2-3 best sentences are re-ordered by their
    original position in the text so the digest reads chronologically.

    Returns the joined chosen sentences; the caller truncates to
    ``_MAX_SUMMARY``.
    """
    sents = _sentences(topic.text)
    if len(sents) < 2:
        return topic.text.strip()

    # Word frequencies across all sentences (stopwords excluded).
    freq: dict = {}
    for s in sents:
        for w in re.findall(r"\w+", s.lower()):
            if w in _EN_STOPWORDS:
                continue
            freq[w] = freq.get(w, 0) + 1

    # Score each sentence, length-normalised to avoid long-sentence bias.
    scored: list[tuple[int, float]] = []
    for idx, s in enumerate(sents):
        nonstop = [w for w in re.findall(r"\w+", s.lower()) if w not in _EN_STOPWORDS]
        if not nonstop:
            continue
        raw = sum(freq.get(w, 0) for w in nonstop)
        norm = raw / (max(1, len(nonstop)) ** 0.6)
        scored.append((idx, norm))

    if not scored:
        return topic.text.strip()

    pick_count = 2 if len(sents) <= 4 else 3
    scored.sort(key=lambda pair: pair[1], reverse=True)
    chosen_indexes = sorted(idx for idx, _ in scored[:pick_count])
    chosen = [sents[i] for i in chosen_indexes]
    return " ".join(chosen)


def _heuristic_title(topic: Topic) -> str:
    """First sentence of the topic, cleaned and truncated; ``Episode N`` fallback."""
    title = _clean(_first_sentence(topic.text))
    if not title:
        title = f"Episode {topic.index + 1}"
    return _truncate(title, _MAX_TITLE)


def _heuristic_summary(topic: Topic) -> str:
    """Frequency-based extractive summary, cleaned and truncated.

    Picks the 2-3 most content-bearing sentences (not just the opening lines),
    re-orders them chronologically, and truncates to ``_MAX_SUMMARY``.
    """
    summary = _clean(_extractive_summary(topic))
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

    # ----- extractive summary self-test -----
    # Topic whose first sentence is a generic intro and whose later sentences
    # repeat the key terms (machine learning) heavily. The offline extractive
    # summary must NOT just echo the first sentence -- it has to pick at
    # least one of the later, denser sentences for the digest.
    extractive_topic = _make_topic(
        2,
        [
            "in this part of the lecture we will continue from last week",
            "machine learning models can be trained on raw data",
            "the key trick is to feed enough machine learning examples",
            "machine learning pipelines need clean machine learning data",
            "we close with a quick recap of the main machine learning idea",
        ],
    )
    enriched_extractive = enrich_topics([extractive_topic], PipelineConfig(use_llm=False))[0]
    first_sentence = _sentences(extractive_topic.text)[0]
    assert enriched_extractive.summary, "extractive summary is empty"
    assert enriched_extractive.summary != first_sentence, (
        f"summary is just the first sentence: {enriched_extractive.summary!r}"
    )
    assert len(enriched_extractive.summary) <= _MAX_SUMMARY, (
        f"summary too long: {len(enriched_extractive.summary)} > {_MAX_SUMMARY}"
    )
    chosen_sents = _sentences(enriched_extractive.summary)
    assert any(s != first_sentence for s in chosen_sents), (
        "extractive summary did not pick any non-first sentence: "
        f"{enriched_extractive.summary!r}"
    )

    print("PASS")
