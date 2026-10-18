"""Cut one source video into one MP4 clip per topic using the ffmpeg CLI.

This is the ``cutter`` stage of the pipeline. It takes an already-enriched list
of :class:`~video_serializer.models.Topic` objects (each with a ``start``/``end``
in seconds and a human ``title``) and a source video file, and writes one MP4
clip per topic into an output directory. Each topic's ``file`` attribute is set
to the absolute path of the clip that was produced for it.

Only the ffmpeg command-line tool is required (no Python ffmpeg bindings). The
tool is expected to be on ``PATH``; :func:`ensure_ffmpeg` checks for it up front
and raises a clear :class:`FFmpegError` when it is missing.

Safety notes
------------
- ffmpeg is always invoked as a **list of arguments** via
  ``subprocess.run(..., shell=False)``. No shell string concatenation is ever
  used, so topic titles / paths can never inject or break the command.
- A single bad topic does *not* silently abort the run: ffmpeg failures raise
  :class:`FFmpegError` carrying the topic index and a tail of ffmpeg's stderr so
  the CLI can surface a useful message.
- Topics whose computed duration is <= 0 are skipped (no ffmpeg call, no empty
  clip) and keep ``topic.file == ""``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

from .models import Topic, PipelineConfig
from .utils import build_filename


class FFmpegError(Exception):
    """Raised when ffmpeg is unavailable or a cut operation fails.

    The message is always actionable: it states what went wrong and, for cut
    failures, includes the offending topic index plus a tail of ffmpeg's stderr.
    """


def ensure_ffmpeg() -> None:
    """Verify that the ``ffmpeg`` executable is available on ``PATH``.

    Raises
    ------
    FFmpegError
        If ``ffmpeg`` cannot be located. The message explains how to fix it.
    """
    if shutil.which("ffmpeg") is None:
        raise FFmpegError(
            "ffmpeg was not found on PATH. Please install ffmpeg (e.g. "
            "'winget install Gyan.FFmpeg' on Windows, 'brew install ffmpeg' on "
            "macOS, or 'apt-get install ffmpeg' on Debian/Ubuntu) and make sure "
            "the 'ffmpeg' binary is reachable from the PATH environment variable, "
            "then try again."
        )


def _build_command(
    video_path: str,
    start: float,
    duration: float,
    output_path: str,
    cut_mode: str,
) -> List[str]:
    """Return the ffmpeg argv (as a list of strings) for one cut.

    Parameters
    ----------
    video_path:
        Path to the source video.
    start:
        Cut start in seconds (formatted to 3 decimals).
    duration:
        Cut length in seconds (formatted to 3 decimals), already clamped >= 0.
    output_path:
        Destination MP4 path.
    cut_mode:
        ``"accurate"`` (re-encode) or ``"fast"`` (stream copy). Any other value
        is rejected.

    Raises
    ------
    FFmpegError
        If ``cut_mode`` is not one of the supported values.
    """
    cmd: List[str] = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        video_path,
        "-t",
        f"{duration:.3f}",
    ]
    if cut_mode == "accurate":
        cmd += [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
        ]
    elif cut_mode == "fast":
        cmd += ["-c", "copy", "-movflags", "+faststart"]
    else:
        raise FFmpegError(
            f"invalid cut_mode: {cut_mode!r} (expected 'accurate' or 'fast')"
        )
    cmd.append(output_path)
    return cmd


def cut_video(
    video_path: str,
    topics: List[Topic],
    output_dir: str,
    config: PipelineConfig,
) -> None:
    """Cut ``video_path`` into one MP4 clip per topic, writing into ``output_dir``.

    For each topic (1-based ``i``) a file named ``build_filename(i, topic.title)``
    is produced. After a successful cut, ``topic.file`` is set to the absolute
    path of the produced clip. Topics with a non-positive duration are skipped
    (no ffmpeg call, no empty clip) and retain ``topic.file == ""``.

    Parameters
    ----------
    video_path:
        Path to the source video to cut from.
    topics:
        Enriched topics, in time order. Each clip's 1-based position in this
        list drives its filename.
    output_dir:
        Directory to write clips into. Created (with parents) if missing.
    config:
        Pipeline config; only ``config.cut_mode`` is used here.

    Raises
    ------
    FFmpegError
        If ffmpeg is missing, ``video_path`` does not exist, ``config.cut_mode``
        is invalid, or any ffmpeg invocation exits non-zero.
    """
    ensure_ffmpeg()

    if not os.path.exists(video_path):
        raise FFmpegError(f"source video not found: {video_path!r}")

    os.makedirs(output_dir, exist_ok=True)

    # Fail fast on an unsupported mode before doing any work.
    cut_mode = config.cut_mode
    if cut_mode not in ("accurate", "fast"):
        raise FFmpegError(
            f"invalid cut_mode: {cut_mode!r} (expected 'accurate' or 'fast')"
        )

    for i, topic in enumerate(topics, start=1):
        duration = max(0.0, topic.end - topic.start)
        if duration <= 0:
            # Nothing to cut: leave topic.file empty and skip ffmpeg entirely.
            topic.file = ""
            continue

        filename = build_filename(i, topic.title)
        output_path = os.path.join(output_dir, filename)
        cmd = _build_command(
            video_path=video_path,
            start=topic.start,
            duration=duration,
            output_path=output_path,
            cut_mode=cut_mode,
        )

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            stderr_tail = (result.stderr or "")[-400:]
            raise FFmpegError(
                f"ffmpeg failed for topic #{i} "
                f"({topic.title or 'untitled'}): {stderr_tail.strip()}"
            )

        topic.file = str(Path(output_path).resolve())


if __name__ == "__main__":
    # Self-test: synthesize a tiny 6s video, cut it into two 3s clips in
    # "accurate" mode, and assert both clips exist and are non-empty.
    from .models import PipelineConfig

    def _run_ffmpeg(cmd: List[str]) -> None:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(
                f"ffmpeg self-test helper failed: {(res.stderr or '')[-400:]}"
            )

    tmp_root = tempfile.mkdtemp(prefix="video_serializer_cutter_")
    try:
        src_dir = Path(tmp_root)
        src = src_dir / "src.mp4"
        _run_ffmpeg([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=6:size=128x72:rate=25",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
            "-c:v", "libx264", "-c:a", "aac", "-t", "6",
            str(src),
        ])

        out_dir = src_dir / "out"
        topics = [
            Topic(index=0, start=0.0, end=3.0, title="Alpha"),
            Topic(index=1, start=3.0, end=6.0, title="Beta"),
        ]
        cut_video(str(src), topics, str(out_dir), PipelineConfig(cut_mode="accurate"))

        expected = [
            (out_dir / "episode_01_alpha.mp4", topics[0].file),
            (out_dir / "episode_02_beta.mp4", topics[1].file),
        ]
        for path, topic_file in expected:
            assert path.exists(), f"missing clip: {path}"
            assert path.stat().st_size > 0, f"empty clip: {path}"
            assert topic_file == str(path.resolve()), (
                f"topic.file mismatch: {topic_file!r} != {str(path.resolve())!r}"
            )

        print("PASS")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
