"""Command-line interface for the video-serializer pipeline.

Wires the five stages together:

    parse transcript  ->  segment into topics  ->  enrich (title/summary)
        ->  (optionally) cut clips with ffmpeg  ->  write metadata

Exposes ``main()`` which is bound to the ``video-serializer`` console script
(see pyproject.toml) and to ``python -m video_serializer`` (see __main__.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from .cutter import FFmpegError, cut_video
from .recap import add_recaps
from .enrich import enrich_topics
from .models import PipelineConfig
from .segmentation import SegmentationError, segment_topics
from .transcript import TranscriptParseError, parse_transcript
from .utils import build_filename, format_timestamp

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: "list[str] | None") -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="video-serializer",
        description=(
            "Split a long video into a series of topic-based short clips and "
            "write a metadata file (JSON or CSV) describing each episode."
        ),
    )
    parser.add_argument("video", help="Path to the source video file.")
    parser.add_argument("transcript", help="Path to the SRT or VTT transcript file.")

    parser.add_argument(
        "-o", "--output", default=None,
        help="Output directory (default: <video_stem>_clips next to the video).",
    )
    parser.add_argument("--min-duration", type=float, default=20.0,
                        help="Minimum topic duration in seconds (default: 20).")
    parser.add_argument("--max-duration", type=float, default=300.0,
                        help="Maximum topic duration in seconds (default: 300).")
    parser.add_argument("--gap", type=float, default=1.5,
                        help="Silence-gap boundary threshold in seconds (default: 1.5).")
    parser.add_argument("--similarity", type=float, default=0.45,
                        help="Similarity threshold below which a new topic starts (default: 0.45).")

    parser.add_argument("--mode", choices=["accurate", "fast"], default="accurate",
                        help="Cut mode: 'accurate' re-encodes (precise), 'fast' stream-copies (default: accurate).")

    emb = parser.add_mutually_exclusive_group()
    emb.add_argument("--embeddings", dest="use_embeddings", action="store_const", const=True,
                     help="Force semantic (embedding) segmentation.")
    emb.add_argument("--no-embeddings", dest="use_embeddings", action="store_const", const=False,
                     help="Force keyword-heuristic segmentation (offline, no dependencies).")
    parser.set_defaults(use_embeddings=None)  # None => auto (use embeddings if available)

    parser.add_argument("--llm", action="store_true",
                        help="Use an LLM for titles/summaries (requires base URL + API key).")
    parser.add_argument("--llm-base-url", default=None,
                        help="OpenAI-compatible base URL (env: LLM_BASE_URL).")
    parser.add_argument("--llm-api-key", default=None,
                        help="LLM API key (env: LLM_API_KEY).")
    parser.add_argument("--llm-model", default=None,
                        help="LLM model name (env: LLM_MODEL, else gpt-4o-mini).")

    parser.add_argument("--metadata", choices=["json", "csv"], default="json",
                        help="Metadata output format (default: json).")
    parser.add_argument("--no-clips", action="store_true",
                        help="Skip cutting clips; only produce metadata.")
    parser.add_argument("--recaps", action="store_true",
                        help="Prepend a short recap intro (text card + optional voice) to each clip "
                             "from the 2nd onward, summarizing the previous clip.")
    parser.add_argument("--recap-duration", type=float, default=4.0,
                        help="Length in seconds of each recap intro (default: 4, clamped to >= 1).")
    parser.add_argument("--recap-lang", choices=["auto", "vi", "en", "zh"], default="auto",
                        help="Language for the recap intro card labels (default: auto = "
                             "detect from the transcript; or force vi / en / zh).")
    parser.add_argument("--recap-voice", default="",
                        help="edge-tts voice for the spoken recap (default: empty = auto by "
                             "recap language); falls back to silent if TTS is unavailable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")

    return parser.parse_args(argv)


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        min_topic_duration=args.min_duration,
        max_topic_duration=args.max_duration,
        gap_threshold=args.gap,
        similarity_threshold=args.similarity,
        use_embeddings=args.use_embeddings,
        cut_mode=args.mode,
        use_llm=args.llm,
        llm_base_url=args.llm_base_url or os.environ.get("LLM_BASE_URL", ""),
        llm_api_key=args.llm_api_key or os.environ.get("LLM_API_KEY", ""),
        llm_model=args.llm_model or os.environ.get("LLM_MODEL", "") or "gpt-4o-mini",
        metadata_format=args.metadata,
        add_recaps=args.recaps,
        recap_duration=args.recap_duration,
        recap_voice=args.recap_voice,
        recap_lang=args.recap_lang,
    )


# ---------------------------------------------------------------------------
# Metadata writers
# ---------------------------------------------------------------------------

def _episode_row(index: int, topic) -> dict:
    return {
        "index": index,
        "title": topic.title,
        "start": round(topic.start, 3),
        "end": round(topic.end, 3),
        "duration": round(topic.duration, 3),
        "intro_seconds": round(getattr(topic, "intro_seconds", 0.0), 3),
        "start_timestamp": format_timestamp(topic.start),
        "end_timestamp": format_timestamp(topic.end),
        "file": topic.file,
        "summary": topic.summary,
    }


_METADATA_COLUMNS = [
    "index", "title", "start", "end", "duration", "intro_seconds",
    "start_timestamp", "end_timestamp", "file", "summary",
]


def _write_metadata(topics, path: Path, source_video: str, transcript: str,
                    fmt: str) -> None:
    if fmt == "csv":
        # utf-8-sig so Excel opens non-ASCII (Vietnamese/Chinese) titles correctly.
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=_METADATA_COLUMNS)
            writer.writeheader()
            for i, topic in enumerate(topics, start=1):
                writer.writerow(_episode_row(i, topic))
    else:
        data = {
            "source_video": source_video,
            "transcript": transcript,
            "total_episodes": len(topics),
            "total_duration_seconds": round(sum(t.duration for t in topics), 3),
            "episodes": [_episode_row(i, t) for i, t in enumerate(topics, start=1)],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: "list[str] | None" = None) -> int:
    # Ensure non-ASCII titles (e.g. Vietnamese/Chinese) print cleanly. On Windows the
    # console default code page is often cp1252, where a plain print() would raise
    # UnicodeEncodeError; reconfigure to UTF-8 and fall back to replacement so the
    # tool never crashes because of console encoding.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass

    args = _parse_args(argv)

    video = Path(args.video)
    transcript = Path(args.transcript)
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = video.parent / f"{video.stem}_clips"

    if not video.is_file():
        print(f"Error: video not found: {video}", file=sys.stderr)
        return 1
    if not transcript.is_file():
        print(f"Error: transcript not found: {transcript}", file=sys.stderr)
        return 1

    config = _build_config(args)

    try:
        # 1) Parse transcript
        segments = parse_transcript(str(transcript))
        print(f"Parsed {len(segments)} transcript segments.")
        if not segments:
            print("Error: transcript contained no usable segments.", file=sys.stderr)
            return 1

        # 2) Segment into topics
        topics = segment_topics(segments, config)
        print(f"Detected {len(topics)} topic(s).")
        if not topics:
            print("Error: no topics produced.", file=sys.stderr)
            return 1

        # 3) Enrich titles + summaries (before cutting, so filenames include titles)
        topics = enrich_topics(topics, config)

        # 4) Cut clips (optional)
        if not args.no_clips:
            cut_video(str(video), topics, str(output_dir), config)
            produced = sum(1 for t in topics if t.file)
            print(f"Cut {produced} clip(s) to {output_dir}/")
            if config.add_recaps:
                print("Adding recap intros to clips 2..N ...")
                add_recaps(topics, str(output_dir), config)
        else:
            for i, t in enumerate(topics, start=1):
                t.file = build_filename(i, t.title)
            print("Skipped clip cutting (--no-clips); intended filenames listed in metadata.")

        # 5) Metadata
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_name = "metadata.csv" if config.metadata_format == "csv" else "metadata.json"
        meta_path = output_dir / meta_name
        _write_metadata(topics, meta_path, str(video), str(transcript), config.metadata_format)
        print(f"Wrote metadata to {meta_path}")

        # 6) Summary table
        print("\n" + "=" * 78)
        print(f"{'#':>2}  {'START':>10}  {'END':>10}  {'DUR':>7}  TITLE")
        print("-" * 78)
        for i, t in enumerate(topics, start=1):
            print(f"{i:>2}  {format_timestamp(t.start):>10}  {format_timestamp(t.end):>10}"
                  f"  {t.duration:>6.1f}s  {t.title}")
        print("=" * 78)
        print(f"\nDone. {len(topics)} episode(s). Output: {output_dir}")
        return 0

    except (TranscriptParseError, SegmentationError, FFmpegError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
