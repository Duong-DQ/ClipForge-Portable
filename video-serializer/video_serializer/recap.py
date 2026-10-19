"""Prepend a short "recap intro" to the start of each clip (from clip 2 onward).

The recap stage of the pipeline. Each clip N (N >= 2) gets a short (1-10s)
intro prepended whose content recaps the *previous* clip (topic N-1). An intro
is composed of two independent parts:

1. A **text card** rendered with ffmpeg's ``drawtext`` filter over a solid dark
   background. This is the always-works, fully-offline path: it needs nothing
   beyond the ffmpeg build (libx264 + freetype).
2. An **optional spoken voice-over** synthesized with ``edge-tts``. That package
   is imported lazily and is entirely optional; on *any* failure (missing
   package, no network, corporate-proxy TLS errors, timeout, empty output, ...)
   the stage degrades gracefully to a *silent* intro and never raises.

Everything else is stdlib + the ffmpeg/ffprobe CLI. To avoid the escaping
nightmares that come with non-Latin text (Vietnamese diacritics) and Windows
paths, all drawtext text is written to UTF-8 temp files and referenced via the
``textfile=`` option; the intro is probed against the real clip so its
resolution/fps/audio parameters match and the final concat stays clean.

Safety notes
------------
- ffmpeg/ffprobe are always invoked as **argument lists** (``shell=False``).
- :func:`generate_tts` is bulletproof: it returns ``(None, 0.0)`` on any error.
- :func:`add_recaps` is best-effort per clip: one failure never stops the rest.
- All temporary files (textfiles, mp3, concat listfile, interim intro/final)
  are cleaned up best-effort.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from .cutter import FFmpegError
from .models import PipelineConfig, Topic

# Vietnamese-capable fonts shipped with Windows (both render the diacritics used
# by the recap text). The first one that exists on disk wins.
_FONT_CANDIDATES = (
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
)

# Solid near-black background colour (matches a dark card aesthetic).
_BG_COLOR = "0x0f1419"

# ---------------------------------------------------------------------------
# Language-aware recap labels
# ---------------------------------------------------------------------------
# The recap card used to hard-code Vietnamese ("Nhắc lại tập trước" / "Tập N:").
# Now the header, the "Episode N:" prefix and the default spoken voice all follow
# the transcript language, so an English transcript gets English labels. Language
# is detected with a dependency-free *script* heuristic (Vietnamese is signalled
# by its distinctive letters, Chinese by CJK ideographs, plain Latin defaults to
# English). A manual override is available via ``PipelineConfig.recap_lang``.

_HEADER_BY_LANG = {
    "vi": "Nhắc lại tập trước",
    "en": "Recap of previous episode",
    "zh": "回顾上一集",
}
_VOICE_BY_LANG = {
    "vi": "vi-VN-HuyNeural",
    "en": "en-US-AriaNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
}

# Letters that (practically) occur only in Vietnamese: the horn letters
# o-horn / u-horn, the d-hook, and the toned circumflex/horn variants. Plain
# Latin (and English) never contain these, so this is a low-false-positive signal.
_VIETNAMESE_SIGNALS = frozenset(
    "ơưđƠƯĐ"
    "ếềểễệỆỀỂỄẼ"
    "ắằẳẵặấầẩẫậẢÃẠẮẰẲẴẶẤẦẨẪẬ"
    "ỏốờớỡởỔỐỜỚỠỞ"
    "ủụừửữựỤỪỬỮỰ"
)

_CJK_RANGE = (0x4E00, 0x9FFF)


def _detect_language(text: str) -> str:
    """Dependency-free script heuristic -> ``"vi"`` | ``"zh"`` | ``"en"``.

    Vietnamese is signalled by its distinctive horn/toned letters, Chinese by
    CJK ideographs, and anything else (plain Latin) defaults to English so an
    English transcript receives English recap labels.
    """
    for ch in text:
        if ch in _VIETNAMESE_SIGNALS:
            return "vi"
    for ch in text:
        code = ord(ch)
        if _CJK_RANGE[0] <= code <= _CJK_RANGE[1]:
            return "zh"
    return "en"


def _resolve_lang(prev_topic: "Topic", config: "PipelineConfig") -> str:
    """Pick the recap language: an explicit ``config.recap_lang`` (vi/en/zh) wins;
    otherwise (``auto``) detect it from the previous topic's text."""
    lang = str(getattr(config, "recap_lang", "auto") or "auto").strip().lower()
    if lang in ("vi", "en", "zh"):
        return lang
    source = (prev_topic.summary or "").strip() or (prev_topic.text or "").strip()
    return _detect_language(source)


def _recap_label(lang: str, index_1based: int, base: str) -> str:
    """Build the localized ``"Episode N: <base>"`` (Vietnamese/English/Chinese)."""
    if lang == "zh":
        return f"第{index_1based}集：{base}"
    word = {"vi": "Tập", "en": "Episode"}.get(lang, "Episode")
    return f"{word} {index_1based}: {base}"


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _short_reason(exc: BaseException) -> str:
    """Return a short, single-line reason string for ``exc`` (for warnings)."""
    msg = " ".join(f"{type(exc).__name__}: {exc}".split())
    return msg[:120] or type(exc).__name__


def make_recap_text(
    prev_topic: "Topic",
    prev_index_1based: int,
    max_chars: int = 140,
    lang: str = "auto",
) -> str:
    """Build the one-line recap string for the *previous* topic.

    The base text is ``prev_topic.summary`` when non-empty, else
    ``prev_topic.title``, else the first ~60 chars of ``prev_topic.text``. The
    result is prefixed with a localized "Episode N:" label (``_recap_label``)
    and, when longer than ``max_chars``, trimmed at the last space before the
    limit with an ellipsis appended.

    ``lang`` may be ``"auto"`` (detect from the previous topic's text) or one of
    ``"vi"`` / ``"en"`` / ``"zh"``. Always returns a single-line string (embedded
    newlines are collapsed).
    """
    base = (prev_topic.summary or "").strip()
    if not base:
        base = (prev_topic.title or "").strip()
    if not base:
        base = (prev_topic.text or "").strip()[:60]

    if lang == "auto":
        lang = _detect_language(base)
    result = _recap_label(lang, prev_index_1based, base)
    # Guarantee a single physical line (drawtext honours embedded newlines).
    result = " ".join(result.split())
    if not result:
        result = _recap_label(lang, prev_index_1based, "")

    if len(result) > max_chars:
        cut = result[:max_chars]
        space = cut.rfind(" ")
        if space > 0:
            cut = cut[:space]
        result = cut.rstrip() + "…"
    return result


def _wrap_text(text: str, width: int = 42) -> List[str]:
    """Word-wrap ``text`` into lines of at most ``width`` characters.

    Splits on whitespace. A single word longer than ``width`` is hard-cut at
    ``width`` characters so no line ever exceeds the limit. Returns a list of
    lines (empty list for empty/whitespace-only input).
    """
    lines: List[str] = []
    current = ""
    for word in text.split():
        # Break over-long words into hard chunks of <= width.
        while len(word) > width:
            if current:
                lines.append(current)
                current = ""
            lines.append(word[:width])
            word = word[width:]
        if not word:
            continue
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------

def _parse_fps(raw: str) -> float:
    """Parse an ffprobe frame-rate string like ``"30000/1001"`` to a float.

    Returns ``25.0`` when the value is missing, ``"0/0"`` or otherwise
    unparseable.
    """
    try:
        if not raw or "/" not in raw:
            return 25.0
        num_s, den_s = raw.split("/", 1)
        num = float(num_s)
        den = float(den_s)
        if den == 0 or num == 0:
            return 25.0
        fps = num / den
        return fps if fps > 0 else 25.0
    except (ValueError, ZeroDivisionError):
        return 25.0


def probe_clip(path: str) -> Dict[str, Any]:
    """Probe the video + first audio stream of ``path`` via ffprobe.

    Returns a dict ``{"width", "height", "fps", "sample_rate", "channels"}``.
    Video geometry comes from the first video stream and audio parameters from
    the first audio stream. Any failure (missing file, no streams, bad JSON,
    ffprobe missing) yields the safe defaults
    ``width=1280, height=720, fps=25.0, sample_rate=44100, channels=2``. Never
    raises.
    """
    info: Dict[str, Any] = {
        "width": 1280,
        "height": 720,
        "fps": 25.0,
        "sample_rate": 44100,
        "channels": 2,
    }
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries",
            "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,"
            "sample_rate,channels",
            "-of", "json",
            path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0 or not res.stdout:
            return info

        import json

        data = json.loads(res.stdout)
        streams = data.get("streams", []) or []

        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        if video is not None:
            if isinstance(video.get("width"), int) and video["width"] > 0:
                info["width"] = video["width"]
            if isinstance(video.get("height"), int) and video["height"] > 0:
                info["height"] = video["height"]
            fps = _parse_fps(video.get("avg_frame_rate") or "")
            if fps == 25.0:
                fps = _parse_fps(video.get("r_frame_rate") or "")
            info["fps"] = fps

        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
        if audio is not None:
            try:
                sr = int(audio.get("sample_rate"))
                if sr > 0:
                    info["sample_rate"] = sr
            except (TypeError, ValueError):
                pass
            try:
                ch = int(audio.get("channels"))
                if ch > 0:
                    info["channels"] = ch
            except (TypeError, ValueError):
                pass
    except Exception:
        # Never raise out of a probe: defaults are always usable.
        return info
    return info


def _probe_duration(path: str) -> float:
    """Return the container duration of ``path`` in seconds, or ``0.0`` on error."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            return 0.0
        return float((res.stdout or "").strip().splitlines()[0])
    except (ValueError, IndexError, subprocess.SubprocessError, OSError):
        return 0.0


# ---------------------------------------------------------------------------
# Optional TTS (edge-tts) — lazy + bulletproof
# ---------------------------------------------------------------------------

def generate_tts(
    text: str,
    out_path: str,
    voice: str = "vi-VN-HuyNeural",
    timeout: int = 45,
) -> Tuple[Optional[str], float]:
    """Synthesize ``text`` to MP3 at ``out_path`` using edge-tts.

    ``edge_tts`` is imported lazily so this module imports cleanly even when the
    package is absent. On *any* failure — ImportError, network/proxy TLS errors,
    ``asyncio.TimeoutError``, an empty output file, etc. — a single warning line
    is printed to stderr and ``(None, 0.0)`` is returned. This function never
    raises.

    On success the MP3's duration is probed with ffprobe and
    ``(out_path, duration)`` is returned; if the probe fails the duration is
    ``0.0``.
    """
    try:
        import edge_tts  # type: ignore

        async def _synth() -> None:
            communicate = edge_tts.Communicate(text, voice)
            await asyncio.wait_for(communicate.save(out_path), timeout=timeout)

        asyncio.run(_synth())

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError("tts produced an empty file")

        duration = _probe_duration(out_path)
        return (out_path, duration)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, must never raise
        print(
            f"[recap] TTS unavailable (reason={_short_reason(exc)}); "
            f"using silent intro",
            file=sys.stderr,
        )
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except OSError:
            pass
        return (None, 0.0)


# ---------------------------------------------------------------------------
# Intro construction
# ---------------------------------------------------------------------------

def _pick_font() -> str:
    """Return the first existing Vietnamese-capable font path, else segoeui."""
    for candidate in _FONT_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return _FONT_CANDIDATES[0]


def _esc_filter_path(path: str) -> str:
    """Escape a filesystem path for use *inside* a single argv filtergraph.

    No shell is involved, but the filtergraph parser treats ``:`` as an option
    separator and ``\\`` as an escape char. We normalise to forward slashes and
    escape literal colons (e.g. the drive letter ``C:`` -> ``C\\:``) so the path
    survives as a single option value.
    """
    normalized = os.path.abspath(path).replace("\\", "/")
    return normalized.replace(":", "\\:")


def _fmt_num(value: float) -> str:
    """Format a number compactly for ffmpeg (integers without a trailing .0)."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def build_intro(
    prev_topic: "Topic",
    prev_index_1based: int,
    ref_clip_path: str,
    out_path: str,
    config: "PipelineConfig",
    header_override: "Optional[str]" = None,
    body_override: "Optional[str]" = None,
) -> Tuple[str, float]:
    """Render a recap intro MP4 for clip ``prev_index_1based + 1``.

    The intro matches the geometry/audio parameters of ``ref_clip_path`` (the
    real clip the intro will be prepended to) so the subsequent concat is clean.
    A solid dark background carries two drawtext layers: a header
    (``"Nhắc lại tập trước"``) near the top and the wrapped recap text centered.
    Audio is either the optional edge-tts voice-over (padded/trimmed to the
    intro duration) or trailing silence.

    When ``header_override`` / ``body_override`` are provided (non-``None``),
    they REPLACE the default localized header / recap body entirely; everything
    else (background colour, font, drawtext layout, ffmpeg encode + concat
    machinery) is identical. The optional TTS voice-over reads the body text
    shown on the card (the override when supplied, otherwise the neutral recap).

    Returns ``(out_path, intro_duration)`` where ``intro_duration`` is
    ``max(1.0, config.recap_duration)``. Raises :class:`FFmpegError` when the
    ffmpeg invocation exits non-zero. Temp text/audio files are always cleaned
    up.
    """
    info = probe_clip(ref_clip_path)
    width = int(info["width"])
    height = int(info["height"])
    fps = float(info["fps"])
    sample_rate = int(info["sample_rate"])
    channels = int(info["channels"])

    duration = max(1.0, float(config.recap_duration))

    lang = _resolve_lang(prev_topic, config)
    default_header = _HEADER_BY_LANG.get(lang, _HEADER_BY_LANG["en"])
    # Overrides (used by topic-aware hooks to show a sensational teaser card)
    # take the slot wholesale; ``None`` keeps the existing neutral default.
    header_text = header_override if header_override is not None else default_header
    if body_override is not None:
        body_text = body_override
    else:
        body_text = make_recap_text(prev_topic, prev_index_1based, lang=lang)
    body_wrapped = "\n".join(_wrap_text(body_text, width=42)) or body_text

    font_path = _pick_font()

    tmp_paths: List[str] = []
    fd, header_file = tempfile.mkstemp(suffix=".txt", prefix="recap_hdr_")
    os.close(fd)
    tmp_paths.append(header_file)
    fd, body_file = tempfile.mkstemp(suffix=".txt", prefix="recap_body_")
    os.close(fd)
    tmp_paths.append(body_file)

    mp3_path: Optional[str] = None
    try:
        with open(header_file, "w", encoding="utf-8") as fh:
            fh.write(header_text)
        with open(body_file, "w", encoding="utf-8") as fh:
            fh.write(body_wrapped)

        # Wrap the (colon-escaped) paths in single quotes: ffmpeg needs the
        # quotes *and* the escaped colon to accept a Windows drive-letter path
        # (e.g. ``fontfile='C\:/Windows/Fonts/segoeui.ttf'``).
        font_esc = "'" + _esc_filter_path(font_path) + "'"
        header_esc = "'" + _esc_filter_path(header_file) + "'"
        body_esc = "'" + _esc_filter_path(body_file) + "'"

        drawtext_header = (
            f"drawtext=fontfile={font_esc}:textfile={header_esc}:"
            f"fontsize=24:fontcolor=white:x=(w-text_w)/2:y=h/4"
        )
        drawtext_body = (
            f"drawtext=fontfile={font_esc}:textfile={body_esc}:"
            f"fontsize=30:fontcolor=white:x=(w-text_w)/2:y=(h-th)/2:"
            f"box=1:boxcolor=black@0.30:boxborderw=16"
        )
        video_filter = f"{drawtext_header},{drawtext_body}"

        # --- audio: optional TTS voice-over, else silent ---
        audio_input: List[str]
        audio_filter: str
        if config.add_recaps:
            fd, mp3_path = tempfile.mkstemp(suffix=".mp3", prefix="recap_tts_")
            os.close(fd)
            os.remove(mp3_path)  # let edge-tts create the file
            # Empty recap_voice = "auto": pick a matching voice for the language.
            voice = config.recap_voice or _VOICE_BY_LANG.get(lang, _VOICE_BY_LANG["en"])
            tts_out, _tts_dur = generate_tts(
                body_text, mp3_path, voice=voice
            )
            if tts_out:
                audio_input = ["-i", tts_out]
                # Pad speech with silence, then trim to the intro length.
                audio_filter = f"apad,atrim=0:{duration:.3f}"
            else:
                audio_input = [
                    "-f", "lavfi",
                    "-i", f"anullsrc=r={sample_rate}:cl=stereo",
                ]
                audio_filter = f"atrim=0:{duration:.3f}"
        else:
            audio_input = [
                "-f", "lavfi",
                "-i", f"anullsrc=r={sample_rate}:cl=stereo",
            ]
            audio_filter = f"atrim=0:{duration:.3f}"

        cmd: List[str] = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i",
            f"color=c={_BG_COLOR}:s={width}x{height}:"
            f"d={duration:.3f}:r={_fmt_num(fps)}",
            *audio_input,
            "-vf", video_filter,
            "-af", audio_filter,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-r", _fmt_num(fps),
            "-c:a", "aac",
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-t", f"{duration:.3f}",
            "-movflags", "+faststart",
            out_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            stderr_tail = (result.stderr or "")[-400:]
            raise FFmpegError(
                f"ffmpeg failed to build intro for episode "
                f"{prev_index_1based + 1}: {stderr_tail.strip()}"
            )
        return (out_path, float(duration))
    finally:
        for path in tmp_paths:
            try:
                os.remove(path)
            except OSError:
                pass
        if mp3_path:
            try:
                os.remove(mp3_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Concatenation
# ---------------------------------------------------------------------------

def _concat_listfile_line(path: str) -> str:
    """Return a concat-demuxer ``file '<path>'`` line (forward slashes, '' escaped)."""
    normalized = os.path.abspath(path).replace("\\", "/").replace("'", "''")
    return f"file '{normalized}'"


def prepend_intro(clip_path: str, intro_path: str, out_path: str) -> str:
    """Write ``out_path`` = ``intro_path`` followed by ``clip_path`` (intro first).

    Preferred path is the concat **demuxer with stream copy** (no re-encode), via
    a temp list file. If that fails or yields a missing/empty output, it falls
    back to the concat **filter** (re-encode). Returns ``out_path`` and raises
    :class:`FFmpegError` only when *both* methods fail. The temp list file is
    always removed.
    """
    fd, listfile = tempfile.mkstemp(suffix=".txt", prefix="recap_concat_")
    os.close(fd)
    first_error = ""
    try:
        with open(listfile, "w", encoding="utf-8") as fh:
            fh.write(_concat_listfile_line(intro_path) + "\n")
            fh.write(_concat_listfile_line(clip_path) + "\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", listfile,
            "-c", "copy",
            "-movflags", "+faststart",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if (
            result.returncode == 0
            and os.path.exists(out_path)
            and os.path.getsize(out_path) > 0
        ):
            return out_path
        first_error = (result.stderr or "")[-400:].strip()
    except Exception as exc:  # noqa: BLE001 - fall through to the re-encode path
        first_error = _short_reason(exc)
    finally:
        try:
            os.remove(listfile)
        except OSError:
            pass

    # --- fallback: concat filter (re-encode) ---
    cmd2 = [
        "ffmpeg", "-y",
        "-i", intro_path,
        "-i", clip_path,
        "-filter_complex",
        "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]",
        "-map", "[v]",
        "-map", "[a]",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        out_path,
    ]
    result2 = subprocess.run(cmd2, capture_output=True, text=True, check=False)
    if (
        result2.returncode != 0
        or not os.path.exists(out_path)
        or os.path.getsize(out_path) == 0
    ):
        stderr_tail = (result2.stderr or "")[-400:].strip()
        raise FFmpegError(
            f"could not prepend intro to {clip_path!r} (copy failed: "
            f"{first_error or 'n/a'}; re-encode failed: {stderr_tail or 'n/a'})"
        )
    return out_path


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------

def add_recaps(topics: List[Topic], output_dir: str, config: "PipelineConfig") -> None:
    """Prepend a recap intro to every clip from the second onward, in place.

    For each topic ``i`` in ``1..len-1`` (0-based; episode ``i + 1``), an intro
    recapping episode ``i`` is built and prepended to that clip, replacing it at
    its original path (so the nice ``episode_NN_*.mp4`` filename is preserved).
    ``clip.intro_seconds`` records the added duration. A per-clip failure is
    logged and skipped without stopping the others. Returns immediately when
    ``config.add_recaps`` is falsy, and removes its temp dir best-effort.

    Topic-aware teaser cards
    ------------------------
    When ``config.topic`` is set (not ``"none"``), every intro from clip 2
    onward (ep = i + 1, with the *first* introed clip being ep 2) shows a
    sensational teaser from that topic instead of the neutral recap: header
    ``"PHẦN {ep}"`` + the topic's ``teasers[ep - 2]`` (modulo the list length)
    via :func:`build_intro`'s ``header_override`` / ``body_override`` knobs.
    When the topic has no teasers (or the topic is ``"none"``) the existing
    neutral recap is used -- byte-identical to the original behaviour.
    """
    if not config.add_recaps:
        return

    tmpdir = os.path.join(output_dir, "_recap_tmp")
    os.makedirs(tmpdir, exist_ok=True)

    topic = str(getattr(config, "topic", "none") or "none").strip().lower()
    teaser_header: "Optional[str]" = None
    teaser_body: "Optional[str]" = None
    pick_teaser = None
    if topic and topic != "none":
        try:
            from .hooks import pick_teaser as _pick_teaser
            pick_teaser = _pick_teaser
        except Exception:  # noqa: BLE001 - missing hooks.json must never crash the pipeline
            pick_teaser = None

    try:
        for i in range(1, len(topics)):
            clip = topics[i]
            prev = topics[i - 1]
            prev_index_1based = i  # clip i is episode (i+1)

            if not clip.file or not os.path.exists(clip.file):
                continue

            ep = i + 1
            # Topic teaser: the first introed clip (ep == 2) gets teaser[0],
            # ep 3 gets teaser[1], etc.  An empty teaser (or missing topic)
            # silently falls back to the neutral recap below.
            teaser_header = None
            teaser_body = None
            if pick_teaser is not None:
                teaser = pick_teaser(topic, ep - 2)
                if teaser:
                    teaser_header = "PHẦN %d" % ep
                    teaser_body = teaser

            intro = os.path.join(tmpdir, "intro_ep%02d.mp4" % ep)
            final = os.path.join(tmpdir, "final_ep%02d.mp4" % ep)
            try:
                intro_path, intro_dur = build_intro(
                    prev, prev_index_1based, clip.file, intro, config,
                    header_override=teaser_header,
                    body_override=teaser_body,
                )
                prepend_intro(clip.file, intro_path, final)
                # Replace the clip in place (keep its nice filename).
                os.replace(final, clip.file)
                clip.intro_seconds = float(intro_dur)
                if teaser_body:
                    print(
                        "[recap] prepended teaser intro to episode %d "
                        "(topic=%s, ~%.1fs)" % (ep, topic, intro_dur)
                    )
                else:
                    print(
                        "[recap] prepended intro to episode %d (recap of ep %d, ~%.1fs)"
                        % (ep, prev_index_1based, intro_dur)
                    )
            except Exception as exc:  # noqa: BLE001 - one clip must not stop the rest
                print(
                    "[recap] WARNING: could not add intro to episode %d: %s "
                    "(left unchanged)" % (ep, exc)
                )
            finally:
                for f in (intro, final):
                    try:
                        os.remove(f)
                    except OSError:
                        pass
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Offline self-test. edge-tts is expected to fail here (corporate proxy) ->
    # the intro is silent, which is fine; the test must still pass.
    from .models import PipelineConfig, Topic, TranscriptSegment

    tmp_root = tempfile.mkdtemp(prefix="video_serializer_recap_")
    try:
        ref = os.path.join(tmp_root, "ref.mp4")
        gen = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=5:size=320x180:rate=25",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-t", "5",
            ref,
        ]
        res = subprocess.run(gen, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(
                f"failed to build reference clip: {(res.stderr or '')[-400:]}"
            )

        prev = Topic(
            index=0,
            start=0.0,
            end=5.0,
            segments=[
                TranscriptSegment(
                    index=0,
                    start=0.0,
                    end=5.0,
                    text="Đây là phần giới thiệu công cụ tách video.",
                )
            ],
            title="Giới thiệu dự án Video Serializer",
            summary=(
                "Phần này giới thiệu cách dùng công cụ tách video thành các tập ngắn."
            ),
        )

        cfg = PipelineConfig()
        cfg.add_recaps = True
        cfg.recap_duration = 3.0
        cfg.recap_voice = "vi-VN-HuyNeural"

        # Text helpers (language-aware)
        recap_text = make_recap_text(prev, 1)
        assert recap_text.startswith("Tập 1: "), recap_text
        assert "\n" not in recap_text
        assert _wrap_text("a b c d", width=3) == ["a b", "c d"]

        # Language detection + English label path
        assert _detect_language("Xin chào, cảm ơn các bạn") == "vi"
        assert _detect_language("Hello everyone, welcome to the video") == "en"
        assert _detect_language("大家好，今天我们来聊聊视频") == "zh"
        en_prev = Topic(
            index=0,
            start=0.0,
            end=5.0,
            segments=[
                TranscriptSegment(
                    index=0,
                    start=0.0,
                    end=5.0,
                    text="This is an introduction to the tool.",
                )
            ],
            title="Introduction",
            summary=(
                "This clip introduces the video splitting tool and how to use it."
            ),
        )
        en_text = make_recap_text(en_prev, 1)
        assert en_text.startswith("Episode 1: "), en_text

        intro = os.path.join(tmp_root, "intro.mp4")
        out = os.path.join(tmp_root, "out.mp4")

        intro_path, intro_dur = build_intro(prev, 1, ref, intro, cfg)
        assert os.path.exists(intro_path) and os.path.getsize(intro_path) > 0, (
            "intro not produced"
        )
        assert intro_dur >= 1.0, f"intro duration too small: {intro_dur}"

        prepend_intro(ref, intro_path, out)
        assert os.path.exists(out) and os.path.getsize(out) > 0, "output not produced"

        out_dur = _probe_duration(out)
        assert out_dur > 5.0, f"intro was not added (duration={out_dur})"

        print("recap self-test OK")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
