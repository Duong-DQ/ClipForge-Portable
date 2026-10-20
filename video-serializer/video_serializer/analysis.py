"""Audio and visual signal analysis via FFmpeg.

This module is dependency-free (stdlib only) and never raises: every function
returns an empty collection on any failure (missing ffmpeg, unsupported codec,
truncated file, etc.). Callers can therefore always treat the result as "best
effort" and continue with the rest of the pipeline.

Three independent signals are exposed:

- :func:`audio_energy_envelope` -- RMS amplitude sampled at a fixed hop across
  the audio track of ``video_path``. Used to find beat-like peaks.
- :func:`beat_peaks` -- pure-math local-maxima picker over an energy envelope.
  No ffmpeg needed; takes the envelope from ``audio_energy_envelope``.
- :func:`scene_cuts` -- ffmpeg's ``scene`` filter thresholded to find hard
  visual cuts. Each entry has ``t`` (timestamp seconds) and ``score``.

All ffmpeg invocations use ``subprocess.run(..., capture_output=True,
timeout=...)`` and a list argv (``shell=False``) so paths can never be
interpreted as shell metacharacters. WAV extraction goes through a temp file
created with :func:`tempfile.NamedTemporaryFile` so concurrent runs do not
collide.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
import wave
from typing import Dict, List, Tuple

__all__ = ["audio_energy_envelope", "beat_peaks", "scene_cuts"]


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _ffmpeg_failed(cmd: list, exc: BaseException) -> str:
    """Format a single-line stderr warning for a failed ffmpeg invocation.

    Never raises; safe to use inside ``except`` blocks.
    """
    return (
        f"[analysis] ffmpeg failed ({type(exc).__name__}: {exc}); "
        f"cmd={cmd[:4]}..."
    )


def _ffmpeg_missing() -> bool:
    """Return ``True`` when the ``ffmpeg`` executable is not on PATH."""
    try:
        import shutil

        return shutil.which("ffmpeg") is None
    except Exception:  # noqa: BLE001
        return True


# --------------------------------------------------------------------------- #
# Audio energy envelope (RMS, fixed hop)
# --------------------------------------------------------------------------- #

def audio_energy_envelope(
    video_path: str, hop_sec: float = 0.05
) -> List[Tuple[float, float]]:
    """Return ``[(timestamp_sec, rms_energy_0_to_1), ...]`` sampled every ``hop_sec``.

    The audio is first extracted to a temporary WAV (mono, 8 kHz, 16-bit PCM)
    via ffmpeg, then read with the stdlib :mod:`wave` module and reduced to
    per-hop RMS values in pure Python. The hop is ``int(hop_sec * sample_rate)``
    samples; the last partial chunk is dropped so all returned entries cover
    exactly ``hop_sec`` seconds each.

    On any failure (missing file, ffmpeg absent, unsupported codec, truncated
    file, empty stream) returns an empty list and emits a single stderr line.
    Never raises.
    """
    if not video_path or not os.path.exists(video_path):
        return []
    if _ffmpeg_missing():
        return []
    if hop_sec <= 0:
        return []

    sample_rate = 8000
    wav_path = ""
    try:
        # mkstemp + close + reopen keeps Windows happy (NamedTemporaryFile
        # cannot be reopened for writing on Windows while still open).
        fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="analysis_audio_")
        os.close(fd)
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ar", str(sample_rate),
            "-ac", "1",
            "-f", "wav",
            wav_path,
        ]
        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, check=False, timeout=120,
            )
        except Exception as exc:  # noqa: BLE001
            print(_ffmpeg_failed(cmd, exc), flush=True)
            return []
        if res.returncode != 0 or not os.path.exists(wav_path):
            return []

        try:
            with wave.open(wav_path, "rb") as wf:
                n_channels = wf.getnchannels()
                samp_width = wf.getsampwidth()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
        except Exception:
            return []

        if not raw or samp_width != 2:
            return []

        # Decode 16-bit signed little-endian PCM. We assume mono per the
        # ffmpeg filter (-ac 1); if the file is somehow multi-channel we still
        # get a deterministic value by treating the stream as interleaved.
        hop_samples = max(1, int(hop_sec * sample_rate))
        bytes_per_sample = samp_width  # 2 bytes for 16-bit
        bytes_per_frame = bytes_per_sample * max(1, n_channels)

        envelope: List[Tuple[float, float]] = []
        max_amp = 32768.0  # 16-bit signed peak
        t = 0.0
        # Iterate frame-aligned chunks; the trailing partial frame is dropped
        # so every returned entry covers exactly ``hop_samples`` samples.
        offset = 0
        while offset + bytes_per_frame * hop_samples <= len(raw):
            sq_sum = 0
            count = 0
            end = offset + bytes_per_frame * hop_samples
            # Inner loop over each sample in the hop (interleaved if >1 ch).
            j = offset
            while j < end:
                # little-endian signed int16
                sample = raw[j] | (raw[j + 1] << 8)
                if sample >= 0x8000:
                    sample -= 0x10000
                sq_sum += sample * sample
                count += 1
                j += bytes_per_frame
            if count > 0:
                rms = math.sqrt(sq_sum / count) / max_amp
                envelope.append((round(t, 4), float(min(1.0, max(0.0, rms)))))
            t += hop_sec
            offset = end
        return envelope
    except Exception as exc:  # noqa: BLE001 - never raise from analysis
        print(f"[analysis] audio envelope failed: {type(exc).__name__}: {exc}",
              flush=True)
        return []
    finally:
        if wav_path:
            try:
                os.remove(wav_path)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# Beat / peak picker (pure math)
# --------------------------------------------------------------------------- #

def beat_peaks(times: List[float], energy: List[float]) -> List[float]:
    """Return timestamps of local maxima in ``energy`` above an adaptive threshold.

    The algorithm matches what ClipForge uses:

    1. Smooth ``energy`` with a length-5 moving average (zero-padded ends).
    2. Threshold = ``max(p75 * 1.25, mean * 1.4)`` where ``p75`` is the 75th
       percentile of the smoothed signal. The ``max(...)`` ensures loud clips
       still produce peaks even when their mean energy is high.
    3. A sample is a peak when ``smoothed[i] >= threshold`` AND it is
       ``>=`` both neighbours (``i - 1`` and ``i + 1``, guarded at the edges).
    4. Consecutive candidates are deduplicated: keep at most one peak per
       ``0.35`` seconds of timeline.

    Returns a list of timestamps (seconds) drawn from ``times``. Returns an
    empty list when the inputs are empty, of different lengths, or all-zero
    (i.e. nothing exceeds the threshold). Never raises.
    """
    try:
        if not times or not energy or len(times) != len(energy):
            return []
        n = len(energy)
        if n < 3:
            return []

        # 1) Smooth with a length-5 box filter (zero-padded ends).
        kernel = 5
        half = kernel // 2
        smoothed = [0.0] * n
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            acc = 0.0
            for j in range(lo, hi):
                acc += float(energy[j])
            smoothed[i] = acc / (hi - lo)

        # 2) Adaptive threshold.
        srt = sorted(smoothed)
        p75 = srt[int(0.75 * (n - 1))]
        mean = sum(smoothed) / n
        threshold = max(p75 * 1.25, mean * 1.4)

        # 3) Local-maxima pass.
        peaks: List[float] = []
        min_gap = 0.35
        last_peak_t = -1e9
        for i in range(1, n - 1):
            v = smoothed[i]
            if v < threshold:
                continue
            if v < smoothed[i - 1] or v < smoothed[i + 1]:
                continue
            t = float(times[i])
            if t - last_peak_t < min_gap:
                continue
            peaks.append(t)
            last_peak_t = t
        return peaks
    except Exception as exc:  # noqa: BLE001
        print(f"[analysis] beat_peaks failed: {type(exc).__name__}: {exc}",
              flush=True)
        return []


# --------------------------------------------------------------------------- #
# Scene-cut detection (ffmpeg scene filter)
# --------------------------------------------------------------------------- #

# ffmpeg's metadata=print frame lines look like:
#   frame:0    pts:120    pts_time:0.500
#   lavfi.scene_score=0.123
_PTS_TIME_RE = re.compile(r"pts_time:\s*([0-9]+(?:\.[0-9]+)?)")
_SCORE_RE = re.compile(r"scene_score\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)")


def scene_cuts(
    video_path: str, threshold: float = 0.30
) -> List[Dict[str, float]]:
    """Detect visual scene cuts in ``video_path`` via ffmpeg's ``scene`` filter.

    Runs ``ffmpeg -skip_frame nokey -vf "select='gt(scene,THRESHOLD)',
    metadata=print" -an -f null -`` and parses the metadata lines to recover
    one ``{t, score}`` entry per emitted frame (i.e. per hard cut). Frames
    without audio are skipped to keep the scan fast.

    The default ``threshold=0.30`` matches ffmpeg's recommended starting
    point; raise it for noisier footage and lower it for slow-paced content.

    On any failure returns an empty list and emits a single stderr line.
    Never raises.
    """
    if not video_path or not os.path.exists(video_path):
        return []
    if _ffmpeg_missing():
        return []

    select_expr = f"select='gt(scene,{threshold:.3f})',metadata=print"
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-skip_frame", "nokey",
        "-vf", select_expr,
        "-an",
        "-f", "null",
        "-",
    ]
    try:
        res = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=180,
        )
    except Exception as exc:  # noqa: BLE001 - never raise from analysis
        print(_ffmpeg_failed(cmd, exc), flush=True)
        return []

    if res.returncode != 0 or not res.stderr:
        return []

    cuts: List[Dict[str, float]] = []
    current_t: float = -1.0
    for line in (res.stderr.splitlines() if res.stderr else []):
        pts_match = _PTS_TIME_RE.search(line)
        if pts_match:
            try:
                current_t = float(pts_match.group(1))
            except ValueError:
                current_t = -1.0
            continue
        score_match = _SCORE_RE.search(line)
        if score_match and current_t >= 0.0:
            try:
                score = float(score_match.group(1))
            except ValueError:
                continue
            cuts.append({"t": round(current_t, 3), "score": round(score, 4)})
            current_t = -1.0  # consume the timestamp so it is not reused
    return cuts


# --------------------------------------------------------------------------- #
# Offline self-test (no ffmpeg, no media required)
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Exercise the pure-math helpers without touching ffmpeg or media."""
    # (a) beat_peaks on a synthetic envelope with two clear peaks.
    times = [i * 0.05 for i in range(200)]  # 0..9.95s
    energy = [0.1] * 200
    energy[40] = 0.9
    energy[41] = 0.95  # the local maximum inside the first cluster
    energy[42] = 0.85
    energy[120] = 0.9
    energy[121] = 0.95  # the local maximum inside the second cluster
    energy[122] = 0.85
    peaks = beat_peaks(times, energy)
    assert peaks, "expected at least one peak in synthetic envelope"
    # Both peaks should land near t = 41 * 0.05 ~= 2.05s and 121 * 0.05 ~= 6.05s.
    assert any(1.5 <= p <= 2.5 for p in peaks), (
        f"first peak not found near 2s: {peaks}"
    )
    assert any(5.5 <= p <= 6.5 for p in peaks), (
        f"second peak not found near 6s: {peaks}"
    )

    # (b) empty / mismatched inputs -> empty result, never raise.
    assert beat_peaks([], []) == []
    assert beat_peaks([0.0, 1.0], [0.5]) == []  # length mismatch

    # (c) too-short inputs are tolerated (no peak possible).
    assert beat_peaks([0.0, 1.0], [0.5, 0.5]) == []

    # (d) flat / monotonous envelope produces no peaks (threshold not crossed).
    flat = [0.5] * 50
    assert beat_peaks([i * 0.05 for i in range(50)], flat) == []

    # (e) all ffmpeg-dependent helpers return [] on missing inputs.
    assert audio_energy_envelope("") == []
    assert audio_energy_envelope("does_not_exist.mp4") == []
    assert scene_cuts("") == []
    assert scene_cuts("does_not_exist.mp4") == []

    print("PASS")


if __name__ == "__main__":
    _self_test()
