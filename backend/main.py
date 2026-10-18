"""ClipForge AI backend - video download, streaming and clip splitting.

Prototype FastAPI backend for the ClipForge AI frontend. It:

  * downloads videos from YouTube / TikTok / any generic web URL via yt-dlp
  * streams full videos and generated clips with HTTP Range support
    (so the browser <video> element can seek)
  * splits a video into clips with FFmpeg using stream copy (no re-encode)

Run with:  python main.py   (or)   uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import random
import re
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import urlparse

import aiofiles
import requests as http_requests
import urllib3
import yt_dlp
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

# Corporate proxy (pxw.exe) terminates TLS with self-signed chains that are
# not in the default trust store, so outbound calls (HuggingFace model
# download, MyMemory translation) need SSL verification disabled - mirroring
# how pip is configured on this machine via trusted-host.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# huggingface_hub >= 1.x performs its model download through httpx, and it has
# no env var to disable TLS verify anymore. Force every httpx client (created
# after this point) to skip certificate verification.
import httpx as _httpx  # noqa: E402

_orig_httpx_client_init = _httpx.Client.__init__


def _patched_httpx_client_init(self, *args, **kwargs):
    """Force all httpx clients (incl. huggingface_hub) to skip TLS verify."""
    kwargs["verify"] = False
    _orig_httpx_client_init(self, *args, **kwargs)


_httpx.Client.__init__ = _patched_httpx_client_init

# ---------------------------------------------------------------------------
# Directories (created on startup)
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = BASE_DIR / "videos"
CLIPS_DIR = BASE_DIR / "clips"
THUMBS_DIR = BASE_DIR / "thumbnails"

for _directory in (VIDEOS_DIR, CLIPS_DIR, THUMBS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("video-clipper")

# ---------------------------------------------------------------------------
# In-memory store (no database for this prototype)
# Record shape:
#   {
#       "id": str, "title": str, "platform": str, "duration": float,
#       "thumbnail": str, "path": str, "created_at": str,
#       "clips": {clip_id: {"id", "start", "end", "label", "duration",
#                           "url", "thumbnail"}},
#   }
# ---------------------------------------------------------------------------

videos: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(title="Video Clipper API", version="0.1.0")

# CORS - allow all origins for development (prototype only).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the single-file frontend at the site root so one public URL (e.g. via a
# tunnel) hosts both the UI and the API on the same origin.
FRONTEND_PATH = BASE_DIR.parent / "video-clipper.html"


@app.get("/")
def serve_frontend():
    """Serve video-clipper.html at the site root (same-origin UI + API)."""
    if FRONTEND_PATH.exists():
        return FileResponse(FRONTEND_PATH, media_type="text/html")
    raise HTTPException(status_code=404, detail="Frontend video-clipper.html not found")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ImportRequest(BaseModel):
    """Body for POST /api/import."""

    url: str = Field(..., min_length=1, description="URL of the video to download")


class ClipSpec(BaseModel):
    """A single clip request for POST /api/split."""

    start: float = Field(..., ge=0, description="Clip start time in seconds")
    end: float = Field(..., gt=0, description="Clip end time in seconds")
    label: str = Field(default="", description="Optional clip label")


class SplitRequest(BaseModel):
    """Body for POST /api/split."""

    videoId: str = Field(..., description="ID of the video to split")
    clips: list[ClipSpec] = Field(..., min_length=1, description="Clips to cut")
    removeSilence: bool = Field(
        default=False, description="Drop silence/dead-air inside each clip (jump-cut)"
    )


# ---------------------------------------------------------------------------
# Helpers - misc
# ---------------------------------------------------------------------------

MEDIA_TYPES: dict[str, str] = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
}


def media_type_for(path: Path) -> str:
    """Return the MIME type for a media file based on its extension."""
    return MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def is_valid_url(url: str) -> bool:
    """Return True if *url* is a well-formed http(s) URL."""
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def detect_platform(url: str) -> str:
    """Detect the source platform: 'youtube', 'tiktok' or 'web'."""
    host = (urlparse(url).netloc or "").lower()
    if "youtube.com" in host or "youtu.be" in host:
        return "youtube"
    if "tiktok.com" in host:
        return "tiktok"
    return "web"


# ---------------------------------------------------------------------------
# Helpers - file management
# ---------------------------------------------------------------------------


def _cleanup_leftovers(directory: Path, prefix: str) -> None:
    """Remove partial/interrupted files matching ``prefix.*`` in ``directory``."""
    for leftover in directory.glob(f"{prefix}.*"):
        try:
            leftover.unlink()
        except OSError:
            logger.warning("Could not remove leftover file %s", leftover)


def _find_output(directory: Path, prefix: str) -> Path | None:
    """Locate the file yt-dlp produced for ``prefix`` (usually prefix.mp4)."""
    candidates = sorted(
        directory.glob(f"{prefix}.*"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


def _get_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except (ValueError, subprocess.TimeoutExpired, FileNotFoundError):
        return 0.0


def _public_video(video_id: str) -> dict:
    """Build the public, path-free view of a stored video record."""
    record = videos[video_id]
    return {
        "id": video_id,
        "title": record["title"],
        "platform": record["platform"],
        "duration": record["duration"],
        "thumbnail": record["thumbnail"],
        "createdAt": record["created_at"],
        "previewUrl": f"/api/videos/{video_id}/preview",
        "clips": [
            {
                "id": clip_id,
                "start": meta["start"],
                "end": meta["end"],
                "label": meta["label"],
                "duration": meta["duration"],
                "url": meta["url"],
                "thumbnail": meta["thumbnail"],
            }
            for clip_id, meta in record["clips"].items()
        ],
    }


# ---------------------------------------------------------------------------
# Helpers - FFmpeg
# ---------------------------------------------------------------------------


def _run_ffmpeg(args: list[str], context: str, timeout: int = 300) -> None:
    """Run an ffmpeg command, translating failures into clear HTTP errors."""
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="FFmpeg is not installed or not in PATH (see backend/README.md)",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=500, detail=f"FFmpeg timed out while {context}"
        ) from exc

    if completed.returncode != 0:
        stderr_lines = (completed.stderr or completed.stdout or "").strip().splitlines()
        snippet = "\n".join(stderr_lines[-5:]) if stderr_lines else "unknown error"
        logger.error(
            "FFmpeg failed while %s (exit %s): %s", context, completed.returncode, snippet
        )
        raise HTTPException(
            status_code=400, detail=f"FFmpeg failed while {context}: {snippet}"
        )


# ---------------------------------------------------------------------------
# Pacing analysis helpers (scene cuts, silence/dead-air, audio energy, beats,
# brightness day/night, face presence). All are best-effort and non-fatal.
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402  (opencv dependency; third-party)

FACE_MODEL_PATH = BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"
_face_detector = None


def _has_audio(video_path: Path) -> bool:
    """Return True if the file has at least one audio stream (via ffprobe)."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, timeout=60,
        )
        return bool((proc.stdout or "").strip())
    except Exception:  # noqa: BLE001
        return False


def _silencedetect_ranges(
    video_path: Path, noise_db: float = -35.0, min_dur: float = 0.5
) -> list[tuple[float, float]]:
    """Detect silence runs across the whole file. Returns absolute [(start, end), ...]."""
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(video_path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}", "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except Exception as exc:  # noqa: BLE001
        logger.warning("silencedetect failed: %s", exc)
        return []
    text = (proc.stderr or "") + (proc.stdout or "")
    ranges: list[tuple[float, float]] = []
    cur: float | None = None
    for m in re.finditer(r"silence_(start|end):\s*(-?\d+(?:\.\d+)?)", text):
        kind, val = m.group(1), float(m.group(2))
        if kind == "start":
            cur = val
        elif cur is not None:
            ranges.append((cur, val))
            cur = None
    return ranges


def _read_wav_energy(wav_path: Path, hop_sec: float = 0.05) -> tuple[list[float], list[float]]:
    """Read a 16-bit mono WAV via the stdlib and return (times, rms_energy) per hop."""
    import wave

    try:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate() or 16000
            n = wf.getnframes()
            raw = wf.readframes(n)
    except Exception as exc:  # noqa: BLE001
        logger.warning("wav read failed: %s", exc)
        return [], []
    if not raw:
        return [], []
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    hop = max(1, int(sr * hop_sec))
    times: list[float] = []
    energy: list[float] = []
    for i in range(0, len(samples) - hop + 1, hop):
        chunk = samples[i:i + hop]
        rms = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
        times.append(round(i / sr, 3))
        energy.append(rms)
    return times, energy


def _energy_envelope(video_path: Path, hop_sec: float = 0.05) -> tuple[list[float], list[float]]:
    """Extract audio and return (times, rms energy). Best-effort (empty on failure)."""
    audio_path = None
    try:
        audio_path = _extract_audio(video_path)
        return _read_wav_energy(audio_path, hop_sec)
    except Exception as exc:  # noqa: BLE001
        logger.warning("energy envelope failed: %s", exc)
        return [], []
    finally:
        if audio_path is not None:
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass


def _beat_peaks(times: list[float], energy: list[float]) -> list[float]:
    """Crude beat/onset peaks: local maxima above an adaptive threshold."""
    if len(times) < 5:
        return []
    env = np.asarray(energy, dtype=np.float32)
    if env.size == 0 or float(env.max()) <= 1e-6:
        return []
    # smooth
    k = 5
    kernel = np.ones(k, dtype=np.float32) / k
    sm = np.convolve(env, kernel, mode="same")
    thr = max(float(np.percentile(sm, 75)) * 1.25, float(sm.mean()) * 1.4, 1e-4)
    peaks: list[float] = []
    last_t = -10.0
    for i in range(1, len(sm) - 1):
        if sm[i] >= thr and sm[i] >= sm[i - 1] and sm[i] > sm[i + 1]:
            t = times[i]
            if t - last_t >= 0.35:  # avoid machine-gun peaks
                peaks.append(round(t, 2))
                last_t = t
    return peaks


def _ffmpeg_metadata_pairs(
    video_path: Path, vf: str, key: str, timeout: int = 900
) -> list[tuple[float, float]]:
    """Run ffmpeg with a metadata=print filter and return [(pts_time, key_value), ...]."""
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(video_path),
        "-vf", vf, "-an", "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ffmpeg metadata pass failed: %s", exc)
        return []
    text = (proc.stderr or "") + (proc.stdout or "")
    pairs: list[tuple[float, float]] = []
    last_t: float | None = None
    for line in text.splitlines():
        mt = re.search(r"pts_time:\s*(-?\d+(?:\.\d+)?)", line)
        if mt:
            last_t = float(mt.group(1))
            continue
        mk = re.search(re.escape(key) + r"=\s*(-?\d+(?:\.\d+)?)", line)
        if mk and last_t is not None:
            pairs.append((round(last_t, 3), float(mk.group(1))))
    return pairs


def _scene_cuts(video_path: Path, threshold: float = 0.30) -> list[dict]:
    """Detect visual scene cuts. Returns [{'t': sec, 'score': s}, ...]."""
    pairs = _ffmpeg_metadata_pairs(
        video_path, f"select='gt(scene,{threshold})',metadata=print", "lavfi.scene_score"
    )
    return [{"t": t, "score": round(s, 3)} for t, s in pairs]


def _brightness_curve(video_path: Path, fps_hop: float = 1.0) -> list[dict]:
    """Sample average luma (YAVG) over time. Returns [{'t': s,'y': 0-255}, ...]."""
    pairs = _ffmpeg_metadata_pairs(
        video_path, f"fps=1/{max(0.25, fps_hop)},signalstats,metadata=print",
        "lavfi.signalstats.YAVG",
    )
    return [{"t": t, "y": round(y, 1)} for t, y in pairs]


def _day_night_transitions(bright: list[dict], drop: float = 60.0) -> list[dict]:
    """Flag brightness transitions (day->night = big YAVG drop; night->day = rise)."""
    out: list[dict] = []
    for a, b in zip(bright, bright[1:]):
        d = b["y"] - a["y"]
        if d <= -drop:
            out.append({"t": b["t"], "kind": "day->night", "delta": round(d, 1)})
        elif d >= drop:
            out.append({"t": b["t"], "kind": "night->day", "delta": round(d, 1)})
    return out


def _get_face_detector():
    """Lazy FaceDetectorYN (YuNet). Returns None if model/lib unavailable."""
    global _face_detector
    import cv2
    if _face_detector is False:
        return None
    if _face_detector is None:
        try:
            if not FACE_MODEL_PATH.exists():
                logger.warning("Face model missing at %s", FACE_MODEL_PATH)
                _face_detector = False
                return None
            _face_detector = cv2.FaceDetectorYN.create(str(FACE_MODEL_PATH), "", (320, 320), 0.7, 0.3, 5000)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FaceDetectorYN init failed: %s", exc)
            _face_detector = False
            return None
    return _face_detector


def _face_samples(video_path: Path, hop_sec: float = 2.0) -> list[dict]:
    """Sample frames every *hop_sec* and count faces. Returns [{'t','count','cx','cy'}, ...]."""
    det = _get_face_detector()
    if det is None:
        return []
    import cv2
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        pattern = str(Path(td) / "f_%05d.jpg")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
            "-vf", f"fps=1/{max(0.5, hop_sec)}", "-q:v", "4", pattern,
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except Exception as exc:  # noqa: BLE001
            logger.warning("face frame dump failed: %s", exc)
            return []
        frames = sorted(Path(td).glob("f_*.jpg"))
        out: list[dict] = []
        for idx, fp in enumerate(frames):
            img = cv2.imread(str(fp))
            if img is None:
                continue
            h, w = img.shape[:2]
            det.setInputSize((w, h))
            try:
                _, boxes = det.detect(img)
            except Exception:  # noqa: BLE001
                boxes = None
            t = round(idx * max(0.5, hop_sec), 2)
            if boxes is not None and len(boxes) > 0:
                b = boxes[0]
                cx = float(b[0] + b[2] / 2) / max(1, w)
                cy = float(b[1] + b[3] / 2) / max(1, h)
                out.append({"t": t, "count": int(len(boxes)),
                            "cx": round(cx, 3), "cy": round(cy, 3)})
            else:
                out.append({"t": t, "count": 0})
        return out


def _energy_at(times: list[float], energy: list[float], t: float, win: float = 0.5) -> float:
    """Average energy within [t-win, t+win]."""
    if not times:
        return 0.0
    lo, hi = t - win, t + win
    vals = [e for tt, e in zip(times, energy) if lo <= tt <= hi]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _count_in(ranges: list, lo: float, hi: float) -> int:
    """Count items (floats or {'t':..}) whose time falls within [lo, hi]."""
    n = 0
    for r in ranges:
        v = r["t"] if isinstance(r, dict) else r
        if lo <= v <= hi:
            n += 1
    return n


def _build_windows(duration: float, cuts: list[dict], target: float) -> list[dict]:
    """Partition [0, duration] at scene cuts into windows close to *target* seconds."""
    boundaries = sorted({0.0, duration} | {c["t"] for c in cuts if 0 < c["t"] < duration})
    if len(boundaries) <= 2:
        n = max(1, int(round(duration / max(1.0, target))))
        boundaries = [round(i * duration / n, 3) for i in range(n + 1)]
    # merge consecutive boundaries so each window is roughly >= target
    merged = [boundaries[0]]
    for b in boundaries[1:]:
        if b - merged[-1] >= target * 0.7 or b == boundaries[-1]:
            merged.append(b)
    windows = []
    for a, b in zip(merged, merged[1:]):
        if b - a >= 0.3:
            windows.append({"start": round(a, 2), "end": round(b, 2)})
    return windows


class PacingRequest(BaseModel):
    videoId: str = Field(..., description="ID of a registered video to analyze")
    minDuration: float = Field(default=10, ge=2, le=3600, description="Min window (s)")
    maxDuration: float = Field(default=30, ge=3, le=7200, description="Max window/target (s)")


@app.post("/api/pacing")
def pacing_analysis(request: PacingRequest) -> dict:
    """Analyze pacing cues (cuts, dead-air, beats, face, day/night, hook/cliffhanger)."""
    record = videos.get(request.videoId)
    if record is None:
        raise HTTPException(status_code=404, detail="Video not found")
    video_path = Path(record["path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file is missing on disk")
    duration = float(record.get("duration") or 0.0)

    # ---- audio energy + beats + silence ----
    e_times, e_energy = _energy_envelope(video_path)
    beats = _beat_peaks(e_times, e_energy)
    silence = _silencedetect_ranges(video_path)
    dead_air = [
        {"start": round(s, 2), "end": round(e, 2), "dur": round(e - s, 2)}
        for s, e in silence if e - s >= 0.8
    ]
    dead_total = round(sum(d["dur"] for d in dead_air), 1)

    # ---- visual: scene cuts + brightness + faces ----
    cuts = _scene_cuts(video_path)
    bright = _brightness_curve(video_path)
    daynight = _day_night_transitions(bright)
    faces = _face_samples(video_path)

    # ---- windows -> hook / cliffhanger scoring ----
    windows = _build_windows(duration, cuts, target=max(4.0, request.maxDuration))
    env_max = max(e_energy) if e_energy else 1.0
    for w in windows:
        mid = (w["start"] + w["end"]) / 2.0
        avg_e = _energy_at(e_times, e_energy, mid, win=(w["end"] - w["start"]) / 2.0)
        w["energy"] = round(avg_e / env_max, 3) if env_max else 0.0
        w["cuts"] = _count_in(cuts, w["start"], w["end"])
        w["beats"] = _count_in(beats, w["start"], w["end"])
        w["faces"] = sum(1 for f in faces if w["start"] <= f["t"] <= w["end"] and f.get("count"))
        span = max(1.0, w["end"] - w["start"])
        changes = w["cuts"] + w["beats"]
        breath = min(1.0, (changes / span) / 0.5)  # ~1 change / 2s => 1.0
        face_ratio = min(1.0, w["faces"] / max(1.0, span / 3.0))
        w["score"] = round(0.45 * w["energy"] + 0.3 * breath + 0.25 * face_ratio, 3)

    ranked = sorted(windows, key=lambda w: w["score"], reverse=True)
    hooks = []
    for w in ranked[:3]:
        reasons = []
        if w["energy"] >= 0.5:
            reasons.append("năng lượng cao")
        if w["beats"] > 0:
            reasons.append(f"{w['beats']} nhịp")
        if w["cuts"] > 0:
            reasons.append(f"{w['cuts']} cắt cảnh")
        if w["faces"] > 0:
            reasons.append("có mặt người")
        hooks.append({
            "start": w["start"], "end": w["end"], "score": w["score"],
            "reason": "🔥 Hook: " + (", ".join(reasons) if reasons else "đoạn sôi động nhất"),
        })

    # cliffhanger: sharp energy drop right after a local peak (buildup -> payoff)
    cliffhangers = []
    if e_energy and len(e_energy) > 4:
        env = np.asarray(e_energy, dtype=np.float32)
        for i in range(2, len(env) - 3):
            pre = float(np.mean(env[max(0, i - 4):i]))
            post = float(np.mean(env[i + 1:i + 5]))
            if pre > env_max * 0.45 and post < pre * 0.55:
                t = round(e_times[i], 2)
                if not cliffhangers or t - cliffhangers[-1]["t"] >= 3.0:
                    cliffhangers.append({"t": t, "reason": "⛓️ Cliffhanger: cắt ngay sau cao trào"})
    cliffhangers = cliffhangers[:5]

    return {
        "videoId": request.videoId,
        "duration": round(duration, 2),
        "hasAudio": _has_audio(video_path),
        "sceneCuts": cuts,
        "brightness": bright,
        "dayNight": daynight,
        "deadAir": dead_air,
        "deadAirTotal": dead_total,
        "beats": beats,
        "faces": faces,
        "windows": windows,
        "hooks": hooks,
        "cliffhangers": cliffhangers,
    }


def _subtract_silence(
    start: float, end: float, silence_map: list[tuple[float, float]],
    pad: float = 0.15, min_keep: float = 0.25,
) -> list[tuple[float, float]]:
    """Return kept [a,b] pieces of [start,end] with silence runs removed (padded)."""
    pieces: list[tuple[float, float]] = []
    cur = start
    for s, e in sorted(silence_map):
        if e <= start or s >= end:
            continue
        s2 = max(start, s + pad)
        e2 = min(end, e - pad)
        if e2 <= s2:
            continue
        if s2 > cur:
            pieces.append((cur, s2))
        cur = max(cur, e2)
    if cur < end:
        pieces.append((cur, end))
    pieces = [(a, b) for a, b in pieces if b - a >= min_keep]
    return pieces if pieces else [(start, end)]


def _render_kept_segments(
    video_path: Path, clip_path: Path, pieces: list[tuple[float, float]], has_audio: bool,
) -> None:
    """Render multiple kept pieces as one clip via trim+concat (re-encode)."""
    base = pieces[0][0]
    parts: list[str] = []
    for i, (a, b) in enumerate(pieces):
        ra, rb = a - base, b - base
        parts.append(f"[0:v]trim=start={ra}:end={rb},setpts=PTS-STARTPTS[v{i}]")
        if has_audio:
            parts.append(f"[0:a]atrim=start={ra}:end={rb},asetpts=PTS-STARTPTS[a{i}]")
    n = len(pieces)
    if has_audio:
        ins = "".join(f"[v{i}][a{i}]" for i in range(n))
        fc = ";".join(parts) + f";{ins}concat=n={n}:v=1:a=1[v][a]"
        maps = ["-map", "[v]", "-map", "[a]"]
    else:
        ins = "".join(f"[v{i}]" for i in range(n))
        fc = ";".join(parts) + f";{ins}concat=n={n}:v=1:a=0[v]"
        maps = ["-map", "[v]"]
    cmd = [
        "ffmpeg", "-y", "-ss", str(base), "-i", str(video_path),
        "-filter_complex", fc, *maps,
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
    ]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "128k"]
    cmd += [str(clip_path)]
    _run_ffmpeg(cmd, context="trimming silence from clip")


# --- Re-encode progress tracking (single active anti-duplicate job) ----------
_exprogress: dict = {"running": False, "percent": 0}


def _run_ffmpeg_progress(
    args: list[str], total_seconds: float, on_progress, context: str, timeout: int = 3600
) -> None:
    """Run ffmpeg while parsing -progress output and reporting percent (0-99)."""
    prog_file = THUMBS_DIR / f"_progress_{uuid.uuid4().hex}.txt"
    cmd = [args[0], "-progress", str(prog_file), "-nostats", "-loglevel", "error"] + args[1:]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500, detail="FFmpeg is not installed or not in PATH"
        ) from exc
    start = time.time()
    while proc.poll() is None:
        if time.time() - start > timeout:
            proc.kill()
            raise HTTPException(status_code=500, detail=f"FFmpeg timed out while {context}")
        try:
            txt = prog_file.read_text()
            found = re.findall(r"out_time_ms=(\d+)", txt)
            if found and total_seconds > 0:
                secs = int(found[-1]) / 1_000_000.0
                on_progress(min(99, max(0, int(secs / total_seconds * 100))))
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    stderr_txt = ""
    try:
        stderr_txt = (proc.stderr.read() or "") if proc.stderr else ""
    except Exception:  # noqa: BLE001
        pass
    try:
        prog_file.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass
    if proc.returncode != 0:
        snippet = "\n".join(stderr_txt.strip().splitlines()[-4:])
        raise HTTPException(status_code=400, detail=f"FFmpeg failed while {context}: {snippet}")


@app.get("/api/exprogress")
def exprocess_progress() -> dict:
    """Poll progress (0-100) of the currently running anti-duplicate job."""
    return dict(_exprogress)


def _cut_clip(
    video_path: Path, video_id: str, start: float, end: float, label: str,
    trim_silence: bool = False, silence_map: list[tuple[float, float]] | None = None,
) -> dict:
    """Cut [start, end] out of *video_path* and return clip metadata.

    When *trim_silence* is set and a *silence_map* is provided, silence/dead-air
    runs inside the range are dropped and the remaining pieces are jump-cut.
    """
    clip_id = str(uuid.uuid4())
    clip_path = CLIPS_DIR / f"{clip_id}.mp4"
    trimmed_seconds = 0.0

    kept: list[tuple[float, float]] | None = None
    if trim_silence and silence_map:
        candidate = _subtract_silence(start, end, silence_map)
        if len(candidate) >= 2:  # something was actually removed
            kept = candidate
            trimmed_seconds = round((end - start) - sum(b - a for a, b in kept), 3)

    if kept is not None:
        _render_kept_segments(video_path, clip_path, kept, has_audio=_has_audio(video_path))
    else:
        # -ss before -i is an input seek; -c copy avoids a re-encode.
        _run_ffmpeg(
            ["ffmpeg", "-y", "-i", str(video_path), "-ss", str(start), "-to", str(end),
             "-c", "copy", str(clip_path)],
            context="splitting clip",
        )

    # Best-effort thumbnail extracted from the clip midpoint.
    thumbnail_url = ""
    midpoint = (start + end) / 2.0
    try:
        thumb_out = THUMBS_DIR / f"{clip_id}.jpg"
        _run_ffmpeg(
            ["ffmpeg", "-y", "-ss", str(midpoint), "-i", str(clip_path),
             "-vframes", "1", "-q:v", "2", str(thumb_out)],
            context="generating clip thumbnail",
        )
        thumbnail_url = f"/api/clips/{clip_id}/thumb.jpg"
    except HTTPException:
        logger.warning(
            "Could not generate thumbnail for clip %s; continuing without one", clip_id
        )

    meta = {
        "id": clip_id,
        "start": start,
        "end": end,
        "label": label,
        "duration": round(end - start, 3),
        "url": f"/api/clips/{clip_id}",
        "thumbnail": thumbnail_url,
    }
    if trimmed_seconds > 0:
        meta["trimmedSeconds"] = trimmed_seconds
    videos[video_id]["clips"][clip_id] = meta
    logger.info("Created clip %s (%.1fs-%.1fs) for video %s", clip_id, start, end, video_id)
    return meta


# ---------------------------------------------------------------------------
# Streaming - HTTP Range support
# ---------------------------------------------------------------------------

_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
_CHUNK_SIZE = 1024 * 1024  # 1 MiB


async def _iter_file(path: Path, start: int, end: int) -> AsyncIterator[bytes]:
    """Yield the byte range [start, end] of *path* in chunks."""
    remaining = end - start + 1
    async with aiofiles.open(path, "rb") as file:
        await file.seek(start)
        while remaining > 0:
            data = await file.read(min(_CHUNK_SIZE, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


class RangeResponse(StreamingResponse):
    """Stream a file, honouring the HTTP ``Range`` header so browsers can seek.

    Returns 200 + the full body when no Range header is sent, and 206 with a
    ``Content-Range`` header for partial requests.  Malformed Range headers are
    ignored (full body served); unsatisfiable ranges yield 416.
    """

    def __init__(self, path: Path, range_header: str | None = None) -> None:
        file_size = path.stat().st_size
        start, end = 0, file_size - 1
        status_code = 200
        headers = {"Accept-Ranges": "bytes", "Content-Length": str(file_size)}

        if range_header:
            match = _RANGE_RE.match(range_header)
            if match:
                first, last = match.group(1), match.group(2)
                if first:
                    start = int(first)
                    end = int(last) if last else file_size - 1
                elif last:  # suffix range, e.g. "bytes=-500" -> last 500 bytes
                    start = max(0, file_size - int(last))
                    end = file_size - 1

                if start >= file_size:
                    raise HTTPException(
                        status_code=416,
                        detail="Requested range not satisfiable "
                        f"(file size: {file_size} bytes)",
                    )

                end = min(end, file_size - 1)
                headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
                headers["Content-Length"] = str(end - start + 1)
                status_code = 206

        super().__init__(
            _iter_file(path, start, end),
            media_type=media_type_for(path),
            status_code=status_code,
            headers=headers,
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health_check() -> dict:
    """Report service health and whether ffmpeg / yt-dlp are available."""
    return {
        "status": "ok",
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ytdlp": importlib.util.find_spec("yt_dlp") is not None,
    }


@app.post("/api/import")
def import_video(request: ImportRequest) -> dict:
    """Download a video from a URL (YouTube / TikTok / generic web) and register it."""
    url = request.url.strip()
    if not is_valid_url(url):
        raise HTTPException(
            status_code=400, detail="Invalid URL: must be an http(s) URL"
        )

    platform = detect_platform(url)
    logger.info("Importing %s video from %s", platform, url)

    video_id = str(uuid.uuid4())
    ydl_opts = {
        # Best mp4 stream up to 1080p, with audio; mp4 output.
        "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]"
        "/best[height<=1080][ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(VIDEOS_DIR / video_id),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        logger.warning("yt-dlp download failed for %s: %s", url, exc)
        _cleanup_leftovers(VIDEOS_DIR, video_id)
        raise HTTPException(
            status_code=400, detail=f"Download failed: {exc}"
        ) from exc

    video_path = _find_output(VIDEOS_DIR, video_id)
    if video_path is None:
        raise HTTPException(
            status_code=500, detail="Download finished but no output file was found"
        )

    duration = float(info.get("duration") or 0.0)
    title = info.get("title") or url
    thumbnail = info.get("thumbnail") or ""

    videos[video_id] = {
        "id": video_id,
        "title": title,
        "platform": platform,
        "duration": duration,
        "thumbnail": thumbnail,
        "path": str(video_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "clips": {},
    }
    logger.info("Imported video %s (%s, %.1fs)", video_id, title, duration)

    return {
        "videoId": video_id,
        "duration": duration,
        "title": title,
        "platform": platform,
        "thumbnail": thumbnail,
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)) -> dict:
    """Upload a local video file and register it for splitting."""
    allowed_exts = {".mp4", ".m4v", ".webm", ".mov", ".mkv", ".avi"}
    filename = file.filename or "upload.mp4"
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {', '.join(allowed_exts)}",
        )

    video_id = str(uuid.uuid4())
    dest = VIDEOS_DIR / f"{video_id}{suffix}"

    # Save uploaded file
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            out.write(chunk)

    # Get duration via ffprobe
    duration = _get_duration(dest)

    videos[video_id] = {
        "id": video_id,
        "title": Path(filename).stem,
        "platform": "local",
        "duration": duration,
        "thumbnail": "",
        "path": str(dest),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "clips": {},
    }
    logger.info("Uploaded video %s (%s, %.1fs)", video_id, filename, duration)

    return {
        "videoId": video_id,
        "duration": duration,
        "title": Path(filename).stem,
        "platform": "local",
        "thumbnail": "",
    }


@app.post("/api/split")
def split_video(request: SplitRequest) -> dict:
    """Split a registered video into clips and write them (with thumbnails) to disk."""
    record = videos.get(request.videoId)
    if record is None:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = Path(record["path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file is missing on disk")

    duration = record["duration"]
    generated: list[dict] = []
    silence_map = _silencedetect_ranges(video_path) if request.removeSilence else None

    for spec in request.clips:
        if spec.start >= spec.end:
            raise HTTPException(
                status_code=400,
                detail="Clip 'end' must be greater than 'start'",
            )

        start = spec.start
        end = spec.end
        if duration > 0:
            end = min(end, duration)
        if end <= start:
            raise HTTPException(
                status_code=400,
                detail=f"Clip range {spec.start}-{spec.end} is empty or out of bounds",
            )

        generated.append(
            _cut_clip(video_path, request.videoId, start, end, spec.label,
                      trim_silence=request.removeSilence, silence_map=silence_map)
        )

    return {"clips": generated}


# ---------------------------------------------------------------------------
# Speech-to-Text (transcription) + Translation
# ---------------------------------------------------------------------------

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
WHISPER_MODELS_DIR = BASE_DIR / "models"
_whisper_model = None
_whisper_model_lock = threading.Lock()


def _get_whisper_model():
    """Lazily load (and cache) the openai-whisper model from a LOCAL directory.

    HuggingFace (where the faster-whisper CTranslate2 models live) is blocked
    by the corporate proxy, so we use openai-whisper with its official
    checkpoint that was pre-downloaded from openaipublic.azureedge.net into
    backend/models/<size>/. If the checkpoint is missing, whisper tries to
    download it from the OpenAI CDN (also reachable through the proxy).
    """
    global _whisper_model
    if _whisper_model is None:
        with _whisper_model_lock:
            if _whisper_model is None:
                logger.info("Loading OpenAI Whisper model '%s'...", WHISPER_MODEL_SIZE)
                try:
                    import whisper

                    _whisper_model = whisper.load_model(
                        WHISPER_MODEL_SIZE,
                        download_root=str(WHISPER_MODELS_DIR),
                    )
                except Exception as exc:  # noqa: BLE001 - surface a clean API error
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to load Whisper model: {exc}",
                    ) from exc
    return _whisper_model


def _extract_audio(video_path: Path) -> Path:
    """Extract 16 kHz mono WAV audio from a video for transcription."""
    audio_path = video_path.parent / f"audio_{uuid.uuid4().hex}.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(audio_path),
    ]
    _run_ffmpeg(cmd, context="extracting audio for transcription")
    return audio_path


class TranscribeRequest(BaseModel):
    videoId: str = Field(..., description="ID of a registered video to transcribe")


@app.post("/api/transcribe")
def transcribe_video(request: TranscribeRequest) -> dict:
    """Transcribe a video's audio into timestamped text segments (auto language)."""
    record = videos.get(request.videoId)
    if record is None:
        raise HTTPException(status_code=404, detail="Video not found")
    video_path = Path(record["path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file is missing on disk")

    audio_path = None
    try:
        audio_path = _extract_audio(video_path)
        model = _get_whisper_model()
        result = model.transcribe(
            str(audio_path),
            language=None,  # auto-detect
            fp16=False,     # CPU only
        )
        segments = [
            {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": (s.get("text") or "").strip()}
            for s in result.get("segments", [])
            if (s.get("text") or "").strip()
        ]
        return {
            "language": result.get("language", ""),
            "segments": segments,
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Transcription failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    finally:
        if audio_path is not None:
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass


class TranslateSegment(BaseModel):
    start: float = 0.0
    end: float = 0.0
    text: str


class TranslateRequest(BaseModel):
    segments: list[TranslateSegment] = Field(..., min_length=1)
    targetLang: str = Field(..., min_length=2, description="ISO 639-1 target code, e.g. 'vi'")


_MY_MEMORY_URL = "https://api.mymemory.translated.net/get"


def _my_memory_translate(text: str, target: str) -> str:
    """Translate one segment via the free MyMemory API (auto source detect)."""
    resp = http_requests.get(
        _MY_MEMORY_URL,
        params={"q": text[:450], "langpair": f"autodetect|{target}"},
        headers={"User-Agent": "Mozilla/5.0"},
        verify=False,
        timeout=25,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("responseStatus") != 200:
        raise HTTPException(
            status_code=502,
            detail=f"MyMemory translation error: {data.get('responseDetails', 'unknown')}",
        )
    return data.get("responseData", {}).get("translatedText", "")


@app.post("/api/translate")
def translate_text(request: TranslateRequest) -> dict:
    """Translate transcript segments into a target language (MyMemory, free)."""
    out: list[dict] = []
    for i, seg in enumerate(request.segments):
        text = (seg.text or "").strip()
        if not text:
            out.append({"start": seg.start, "end": seg.end, "text": seg.text, "translated": ""})
            continue
        try:
            translated = _my_memory_translate(text, request.targetLang)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - keep original on single-segment failure
            logger.warning("Translation failed for segment %d: %s", i, exc)
            translated = seg.text
        out.append({"start": seg.start, "end": seg.end, "text": seg.text, "translated": translated})
        # Be gentle on the anonymous free tier.
        if i < len(request.segments) - 1:
            time.sleep(0.3)
    return {"targetLang": request.targetLang, "segments": out}


# ---------------------------------------------------------------------------
# Semantic segmentation (content-aware clip boundaries from transcript cues)
# ---------------------------------------------------------------------------

class SegmentRequest(BaseModel):
    videoId: str = Field(..., description="ID of a registered video to segment")
    minDuration: float = Field(default=60, ge=5, le=3600, description="Min clip length (s)")
    maxDuration: float = Field(default=180, ge=10, le=7200, description="Max clip length (s)")


_CUE_OPEN = [
    "now for", "next up", "up next", "moving on", "move on to", "let's move", "lets move",
    "now let's", "now lets", "welcome back", "so now", "for today", "and today", "today we",
    "today's", "now it is time", "it's time to", "its time to", "let's get into", "lets get into",
    "tiếp theo là", "tiếp theo", "bây giờ chúng ta", "bây giờ", "chuyển sang", "sang phần",
    "hôm nay chúng ta", "và bây giờ", "tiếp tục với", "bắt đầu",
]

_CUE_CLOSE = [
    "that's all for", "thats all for", "that's it for", "thats it for", "so that's", "so thats",
    "see you", "until next time", "thanks for watching", "thank you for watching", "that wraps up",
    "that's it", "thats it", "for today", "come back for", "the end", "that's a wrap", "thats a wrap",
    "đó là tất cả", "hết phần", "kết thúc", "hẹn gặp lại", "cảm ơn đã xem", "tạm biệt",
    "đến đây là hết", "xin cảm ơn", "phần này kết thúc",
]

_CUE_GREET = [
    "hello everyone", "hello everybody", "hey guys", "hi everyone", "hi guys", "welcome back to",
    "welcome to", "hello and welcome", "good morning", "good afternoon",
    "xin chào", "chào mừng", "chào cả nhà", "xin chào các bạn",
]

_CUE_MARKER_RE = [
    r"\bday (?:one|two|three|four|five|six|seven|eight|nine|\d+)\b",
    r"\bpart (?:one|two|three|four|\d+)\b",
    r"\bgame (?:one|two|three|four|\d+)\b",
    r"\bround (?:one|two|three|\d+)\b",
    r"\b(?:episode|chapter|level|stage|match) (?:one|two|three|\d+)\b",
    r"\bngày (?:thứ )?(?:một|hai|ba|bốn|năm|sáu|bảy|tám|chín|\d+)\b",
    r"\bphần (?:một|hai|ba|\d+)\b",
    r"\bvòng (?:một|hai|ba|\d+)\b",
    r"\btrận (?:đấu )?(?:một|hai|ba|\d+)\b",
    r"\bmàn (?:một|hai|ba|\d+)\b",
    r"\btập (?:một|hai|ba|\d+)\b",
]


def _hit_cue(text: str, plain: list[str], regexps: list[str]) -> bool:
    t = (" " + (text or "") + " ").lower()
    for p in plain:
        if p in t:
            return True
    for r in regexps:
        if re.search(r, t):
            return True
    return False


# Signal priority for the "reason" label (lower index = more meaningful).
_SIGNAL_PRIORITY = ["marker", "close", "open", "greet", "silence"]


def _score_cut(segs: list[dict], k: int) -> tuple[float, str, str, float]:
    """Score a boundary between seg[k-1] and seg[k]. Returns (score, signal, snippet, gap)."""
    prev = segs[k - 1]["text"] or ""
    nxt = segs[k]["text"] or ""
    score = 0.0
    signals: list[str] = []

    if _hit_cue(nxt, [], _CUE_MARKER_RE):
        score += 0.9
        signals.append("marker")
    if _hit_cue(nxt, _CUE_OPEN, []):
        score += 0.7
        signals.append("open")
    if _hit_cue(nxt, _CUE_GREET, []):
        score += 0.6
        signals.append("greet")
    if _hit_cue(prev, _CUE_CLOSE, []):
        score += 0.7
        signals.append("close")

    gap = float(segs[k]["start"]) - float(segs[k - 1]["end"])
    if gap >= 1.5:
        score += min(1.0, 0.5 + (gap - 1.5) / 5.0)  # 1.5s→0.5, 3s→0.8, 5s+→1.0
        signals.append("silence")

    signal = min(signals, key=lambda s: _SIGNAL_PRIORITY.index(s)) if signals else ""
    snippet = (nxt or prev).strip()
    return score, signal, snippet[:60], gap


def _reason_text(signal: str, snippet: str, gap: float) -> str:
    """Human-readable reason for a boundary choice."""
    if signal == "marker":
        return f'🏷️ Mở phần mới: "{snippet}"'
    if signal == "close":
        return f'🏁 Kết phần: "{snippet}"'
    if signal == "open":
        return f'⏭️ Dẫn chuyển: "{snippet}"'
    if signal == "greet":
        return f'👋 Chào mở đầu: "{snippet}"'
    if signal == "silence":
        return f"🤫 Im lặng {gap:.1f}s"
    return "📐 Chia đều (fallback)"


def _label_at(segs: list[dict], pos: float, clip_no: int) -> str:
    """Build a short label from the first words of the transcript near *pos*."""
    for s in segs:
        if float(s["start"]) >= pos - 0.5:
            words = (s["text"] or "").strip().split()
            if words:
                return " ".join(words[:8])[:70]
            break
    return f"Clip {clip_no}"


def _semantic_clips(segs: list[dict], duration: float, min_dur: float, max_dur: float) -> list[dict]:
    """Greedy windowed segmentation: cut at the strongest semantic boundary in [min, max]."""
    clips: list[dict] = []
    last = 0.0
    clip_no = 1
    n = len(segs)

    while True:
        low_early = last + min_dur * 0.8   # allow strong cues to cut slightly early
        low = last + min_dur
        high = last + max_dur
        if high >= duration - 1.0:
            break

        best = None  # (score, pos, signal, snippet, gap)
        for k in range(1, n):
            pos = float(segs[k]["start"])
            if pos > high:
                break
            if pos < low_early:
                continue
            sc, signal, snippet, gap = _score_cut(segs, k)
            if best is None or sc > best[0]:
                best = (sc, pos, signal, snippet, gap)

        pos: float | None = None
        reason = "📐 Chia đều (fallback)"
        if best and best[0] >= 0.85 and low_early <= best[1] <= high:
            pos, reason = best[1], _reason_text(best[2], best[3], best[4])
        elif best and best[0] >= 0.4 and low <= best[1] <= high:
            pos, reason = best[1], _reason_text(best[2], best[3], best[4])

        if pos is None:
            pos = last + max_dur  # hard cut

        # Enforce minimum spacing / guaranteed progress
        if pos - last < min_dur * 0.45:
            pos = last + min_dur
            reason = "📐 Chia đều (fallback)"
        if pos <= last + 1.0:
            pos = max(last + 1.0, last + max_dur)
            reason = "📐 Chia đều (fallback)"

        end = min(pos, duration)
        label = _label_at(segs, pos, clip_no) if reason.startswith(("🏷️", "🏁", "⏭️", "👋")) else f"Clip {clip_no}"
        clips.append({"start": round(last, 2), "end": round(end, 2), "label": label, "reason": reason})
        clip_no += 1

        last = end
        if last >= duration - 0.5:
            break

    # Final tail (uncovered remainder)
    if last < duration - 1.0:
        clips.append({
            "start": round(last, 2),
            "end": round(duration, 2),
            "label": _label_at(segs, last, clip_no),
            "reason": "📐 Phần cuối",
        })
        clip_no += 1

    # Merge a too-short tail into the previous clip
    if len(clips) >= 2 and clips[-1]["end"] - clips[-1]["start"] < max(5.0, min_dur * 0.3):
        clips[-2]["end"] = clips[-1]["end"]
        clips.pop()

    return clips


def _equal_clips(duration: float, min_dur: float, max_dur: float, reason: str) -> list[dict]:
    """Fallback: split the video into equal time slots sized from the target range."""
    count = max(1, round(duration / ((min_dur + max_dur) / 2.0)))
    step = duration / count
    clips: list[dict] = []
    for i in range(count):
        s = i * step
        e = duration if i == count - 1 else (i + 1) * step
        clips.append({"start": round(s, 2), "end": round(e, 2), "label": f"Clip {i + 1}", "reason": reason})
    return clips


@app.post("/api/segment")
def segment_video(request: SegmentRequest) -> dict:
    """Return content-aware clip boundaries (transcript cues + silence). """
    record = videos.get(request.videoId)
    if record is None:
        raise HTTPException(status_code=404, detail="Video not found")
    video_path = Path(record["path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file is missing on disk")

    duration = float(record.get("duration") or 0.0)
    if duration <= 0:
        try:
            duration = _get_duration(video_path)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Could not determine video duration: {exc}") from exc

    min_dur = min(request.minDuration, request.maxDuration)
    max_dur = max(request.minDuration, request.maxDuration)
    if max_dur < min_dur + 1:
        max_dur = min_dur + 1

    # Cache transcript on the record so repeated /segment or /split calls don't re-transcribe.
    segs = record.get("transcript")  # None if not transcribed yet
    language = record.get("transcript_language", "")
    audio_path = None
    model = None
    if segs is None:
        try:
            audio_path = _extract_audio(video_path)
            model = _get_whisper_model()
            result = model.transcribe(str(audio_path), language=None, fp16=False)
            language = result.get("language", "") or ""
            segs = [
                {"start": float(s["start"]), "end": float(s["end"]), "text": (s.get("text") or "").strip()}
                for s in result.get("segments", [])
                if (s.get("text") or "").strip()
            ]
            # Only cache a non-empty transcript.
            if segs:
                record["transcript"] = segs
                record["transcript_language"] = language
        except Exception as exc:  # noqa: BLE001 - non-fatal: fall back to equal split
            logger.warning("Segment: transcription failed (%s) - falling back to equal split", exc)
            segs = []
        finally:
            if audio_path is not None:
                try:
                    audio_path.unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass

    if not segs:
        return {
            "mode": "equal",
            "language": language,
            "clips": _equal_clips(duration, min_dur, max_dur, "📐 Chia đều (không có transcript)"),
        }

    clips = _semantic_clips(segs, duration, min_dur, max_dur)
    # Safety net: if the algorithm produced nothing, fall back to equal.
    if not clips:
        clips = _equal_clips(duration, min_dur, max_dur, "📐 Chia đều (fallback)")
    return {"mode": "semantic", "language": language, "clips": clips}


# ---------------------------------------------------------------------------
# Anti-duplicate / uniqueness processing (make repurposed content look new)
# ---------------------------------------------------------------------------

class ExProcessRequest(BaseModel):
    videoId: str = Field(..., description="ID of a registered source video")
    outputAspect: str = Field(default="9:16", description="'9:16', '1:1' or '16:9'")
    zoom: float = Field(default=1.20, ge=1.0, le=1.30, description="Center zoom after crop (120% default)")
    flip: bool = Field(default=True, description="Mirror (hflip) the whole video")
    colorAdjust: bool = Field(default=True, description="Eq: contrast +2%, brightness +3%")
    colorMatrix: bool = Field(default=True, description="Change pixel color matrix (bt709->bt601)")
    noiseOpacity: float = Field(default=0.03, ge=0.0, le=0.10, description="Noise/haze overlay opacity (3% default)")
    speed: float = Field(default=0.0, ge=0.0, le=0.25, description="Speed 1.03-1.06; 0 = random within range")
    maxHeight: int = Field(default=1920, ge=480, le=2160, description="Cap output height (e.g. 1280 = faster/lighter render)")
    subjectZoom: bool = Field(default=True, description="Dynamic punch-in on the subject (face/motion) during quiet moments")
    subjectZoomLevel: float = Field(default=1.35, ge=1.05, le=1.8, description="Zoom level during quiet punch-in")


def _file_md5(path: Path, chunk: int = 1 << 20) -> str:
    """Compute an MD5 hash of a file with bounded memory (streaming)."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


_ASPECT_SIZES = {"9:16": (1080, 1920), "1:1": (1080, 1080), "16:9": (1920, 1080)}


def _frame_thumb(video_path: Path, out_name: str, t_sec: float) -> str:
    """Extract a small JPEG frame at *t_sec* for before/after comparison. URL or ''."""
    out = THUMBS_DIR / f"{out_name}.jpg"
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{max(0.0, t_sec):.2f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", "scale=240:-2",
        "-q:v", "4",
        str(out),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception:  # noqa: BLE001 - best effort, non-fatal
        return ""
    if out.exists():
        return f"/api/thumbs/{out_name}.jpg"
    return ""


def _get_fps(video_path: Path) -> float:
    """Best-effort source FPS via ffprobe (defaults to 30 on failure)."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate", "-of", "default=nw=1:nk=1",
        str(video_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        raw = (r.stdout or "").strip().splitlines()[0]
        num, _, den = raw.partition("/")
        fps = float(num) / float(den or 1)
        return fps if 5.0 <= fps <= 120.0 else 30.0
    except Exception:  # noqa: BLE001
        return 30.0


def _subject_samples(video_path: Path, hop_sec: float = 0.75) -> list[dict]:
    """Per-hop subject position: largest face, else frame-diff motion region.

    Returns [{'t','kind','fx','fy','face_ratio','mx','my','motion'}, ...] with
    fx/fy/mx/my normalized to 0-1. 'kind' is 'face' | 'motion' | 'none'.
    """
    import cv2
    import tempfile

    hop = max(0.4, hop_sec)
    det = _get_face_detector()
    with tempfile.TemporaryDirectory() as td:
        pattern = str(Path(td) / "s_%05d.jpg")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
            "-vf", f"fps=1/{hop:.4f}", "-q:v", "4", pattern,
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except Exception as exc:  # noqa: BLE001
            logger.warning("subject frame dump failed: %s", exc)
            return []

        frames = sorted(Path(td).glob("s_*.jpg"))
        out: list[dict] = []
        prev_gray = None
        for idx, fp in enumerate(frames):
            img = cv2.imread(str(fp))
            if img is None:
                continue
            h, w = img.shape[:2]
            t = round(idx * hop, 2)

            # Largest face (closest / most prominent = likely the speaker)
            fx = fy = None
            face_ratio = 0.0
            if det is not None:
                det.setInputSize((w, h))
                try:
                    _, boxes = det.detect(img)
                except Exception:  # noqa: BLE001
                    boxes = None
                if boxes is not None and len(boxes) > 0:
                    b = max(boxes, key=lambda bx: float(bx[2]) * float(bx[3]))
                    bw, bh = float(b[2]), float(b[3])
                    fx = float(b[0] + bw / 2) / max(1, w)
                    fy = float(b[1] + bh / 2) / max(1, h)
                    face_ratio = (bw * bh) / max(1, w * h)

            # Motion region (frame difference) — crude proxy for the action/object
            small = cv2.resize(img, (160, max(1, int(160 * h / max(1, w)))))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            mx = my = None
            motion = 0.0
            if prev_gray is not None and prev_gray.shape == gray.shape:
                diff = cv2.absdiff(gray, prev_gray)
                motion = float(diff.mean()) / 255.0
                if motion > 0.01:
                    _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    ys, xs = np.where(th > 0)
                    if len(xs) > 0:
                        mx = float(xs.mean()) / gray.shape[1]
                        my = float(ys.mean()) / gray.shape[0]
            prev_gray = gray

            if fx is not None:
                kind = "face"
            elif mx is not None:
                kind = "motion"
            else:
                kind = "none"
            out.append({
                "t": t, "kind": kind,
                "fx": round(fx, 3) if fx is not None else None,
                "fy": round(fy, 3) if fy is not None else None,
                "face_ratio": round(face_ratio, 4),
                "mx": round(mx, 3) if mx is not None else None,
                "my": round(my, 3) if my is not None else None,
                "motion": round(motion, 4),
            })
        return out


def _quiet_focus_plan(video_path: Path, duration: float, max_windows: int = 16) -> list[dict]:
    """Silence windows each with a focus point: largest face -> motion -> center."""
    sil = [(s, e) for (s, e) in _silencedetect_ranges(video_path) if e - s >= 0.6]
    if not sil:
        return []
    sil.sort()
    merged: list[list[float]] = []
    for s, e in sil:
        if merged and s - merged[-1][1] < 1.0:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    if len(merged) > max_windows:
        merged = sorted(sorted(merged, key=lambda r: r[1] - r[0], reverse=True)[:max_windows])

    hop = max(0.5, min(1.5, duration / 120.0)) if duration > 0 else 0.75
    samples = _subject_samples(video_path, hop_sec=hop)

    plan: list[dict] = []
    for s, e in merged:
        seg = [x for x in samples if s - hop <= x["t"] <= e + hop]
        faces = [x for x in seg if x["fx"] is not None and x["face_ratio"] >= 0.003]
        if faces:
            best = max(faces, key=lambda x: x["face_ratio"])
            fx, fy, src = best["fx"], best["fy"], "face"
        else:
            mot = [x for x in seg if x["mx"] is not None]
            if mot:
                best = max(mot, key=lambda x: x["motion"])
                fx, fy, src = best["mx"], best["my"], "motion"
            else:
                fx, fy, src = 0.5, 0.5, "center"
        plan.append({"start": round(s, 2), "end": round(e, 2),
                     "fx": float(fx), "fy": float(fy), "src": src})
    return plan


def _zoompan_expr(plan: list[dict], base_zoom: float, extra: float, ramp: float = 0.4) -> tuple[str, str, str]:
    """Build (z, x, y) ffmpeg zoompan expressions with smooth trapezoid gates.

    Expressions use PLAIN commas (callers embed them inside single quotes).
    """
    if not plan:
        return f"{base_zoom:.4f}", "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"

    R = max(0.15, ramp)
    gates: list[str] = []
    for w in plan:
        s, e = float(w["start"]), float(w["end"])
        gates.append(f"max(0,min(1,min((in_time-{s:.3f})/{R:.3f},({e:.3f}-in_time)/{R:.3f})))")

    gsum = "(" + "+".join(gates) + ")"                       # ~0..1 (windows don't overlap)
    gcap = f"min(1,{gsum})"
    z = f"{base_zoom:.4f}+{extra:.4f}*{gcap}"

    fx_num = "+".join(f"({g})*{float(w['fx']):.4f}" for g, w in zip(gates, plan))
    fy_num = "+".join(f"({g})*{float(w['fy']):.4f}" for g, w in zip(gates, plan))
    fx = f"(if(gt({gsum},0.0001),({fx_num})/{gsum},0.5))"
    fy = f"(if(gt({gsum},0.0001),({fy_num})/{gsum},0.5))"

    x = f"max(0,min(iw-iw/zoom,{fx}*iw-(iw/zoom)/2))"
    y = f"max(0,min(ih-ih/zoom,{fy}*ih-(ih/zoom)/2))"
    return z, x, y


@app.post("/api/exprocess")
def exprocess_video(request: ExProcessRequest) -> dict:
    """Re-render a video so its pixels/audio metadata differ from the source.

    Applies (controllable): aspect crop+center zoom, mirror flip, contrast/brightness
    tweak, color-matrix change, a low-opacity noise overlay, small speed change with
    pitch compensation, metadata strip, and a brand-new MD5. Registers the result as a
    NEW video record so the user can keep clipping the unique copy.
    """
    record = videos.get(request.videoId)
    if record is None:
        raise HTTPException(status_code=404, detail="Video not found")
    video_path = Path(record["path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file is missing on disk")

    w, h = _ASPECT_SIZES.get(request.outputAspect, _ASPECT_SIZES["9:16"])
    # Optional downscale cap -> much faster render (fewer pixels). Aspect preserved.
    mh = max(480, min(2160, int(request.maxHeight)))
    if mh < h:
        w = int(w * (mh / h)) // 2 * 2
        h = mh // 2 * 2
    zoom = max(1.0, min(1.30, request.zoom))
    speed = request.speed if request.speed >= 1.03 else round(random.uniform(1.03, 1.06), 4)
    speed = max(1.01, min(1.10, speed))

    # Fast center-crop to the target aspect at NATIVE resolution, then a single
    # scale to output. This avoids the old cover-fit upscale + zoom-upscale +
    # crop (two large upscales) and is ~30% faster on CPU.
    ar = w / h  # target aspect ratio (w/h)

    # Shared link map (aspect crop is shared; zoom applied by scale OR zoompan).
    def _tail_filters() -> str:
        t = ""
        if request.flip:
            t += ",hflip"
        if request.colorAdjust:
            t += ",eq=contrast=1.02:brightness=0.03"
        if request.colorMatrix:
            t += ",colormatrix=bt709:bt601"
        t += f",setpts=PTS/{speed}"
        return t

    # Static pipeline (default + fallback): center crop at native res, single scale.
    vf = (
        f"crop=w=min(iw\\,ih*{ar:.6f})/{zoom:.4f}"
        f":h=min(ih\\,iw/{ar:.6f})/{zoom:.4f}"
        f":x=(iw-ow)/2:y=(ih-oh)/2,"
        f"scale={w}:{h},setsar=1"
    )
    vf += _tail_filters()

    # Optional dynamic "quiet punch-in": during silence, zoom/pan to the subject
    # (largest face -> motion region -> center). Built as a time-varying zoompan.
    vf_dynamic = None
    focus_plan: list[dict] = []
    if request.subjectZoom:
        try:
            focus_plan = _quiet_focus_plan(video_path, float(record.get("duration") or 0.0))
        except Exception as exc:  # noqa: BLE001
            focus_plan = []
            logger.warning("quiet focus plan failed: %s", exc)
        if focus_plan:
            zexpr, xexpr, yexpr = _zoompan_expr(plan=focus_plan, base_zoom=zoom,
                                                extra=(request.subjectZoomLevel - 1.0))
            fps = _get_fps(video_path)
            aspect = (
                f"crop=w=min(iw\\,ih*{ar:.6f}):h=min(ih\\,iw/{ar:.6f})"
                f":x=(iw-ow)/2:y=(ih-oh)/2"
            )
            vf_dynamic = (
                aspect
                + f",zoompan=z='{zexpr}':x='{xexpr}':y='{yexpr}':d=1:s={w}x{h}:fps={fps:.3f}"
                + _tail_filters()
            )

    def _build_cmd(vf_use: str, out_path: Path) -> list[str]:
        fc = [f"[0:v]{vf_use}[vt]"]
        noise = max(0.0, min(0.10, request.noiseOpacity))
        if noise > 0:
            fc.append(f"nullsrc=s={w}x{h}:d=1,format=gray,noise=alls=20:allf=t+u,format=yuv420p[un]")
            fc.append(f"[vt][un]blend=all_mode=overlay:all_opacity={noise:.3f},format=yuv420p[v]")
            vout = "[v]"
        else:
            fc[0] = fc[0].replace("[vt]", "[v]")
            vout = "[v]"
        fc_parts = ";".join(fc)
        has_audio = _has_audio(video_path)
        if has_audio:
            fc_parts += f";[0:a]atempo={speed}[a]"
        c = ["ffmpeg", "-y", "-i", str(video_path), "-filter_complex", fc_parts, "-map", vout]
        if has_audio:
            c += ["-map", "[a]", "-c:a", "aac", "-b:a", "128k"]
        else:
            c += ["-an"]
        c += [
            "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
            "-movflags", "+faststart",
            "-map_metadata", "-1", "-map_chapters", "-1",
            str(out_path),
        ]
        return c

    new_id = uuid.uuid4().hex
    out_path = VIDEOS_DIR / f"{new_id}.mp4"
    # A full re-encode of a long source can take many minutes: allow up to 1 hour
    # and stream progress so the UI can show a moving bar.
    total_out = (float(record.get("duration") or 0.0) / max(1.01, speed)) or 0.0
    _exprogress["running"] = True
    _exprogress["percent"] = 0
    subject_mode = "static"
    try:
        if vf_dynamic:
            try:
                _run_ffmpeg_progress(
                    _build_cmd(vf_dynamic, out_path), total_out,
                    lambda p: _exprogress.__setitem__("percent", p),
                    context="anti-duplicate processing (subject zoom)", timeout=3600,
                )
                subject_mode = "dynamic"
            except Exception as exc:  # noqa: BLE001
                logger.warning("dynamic subject-zoom render failed (%s); falling back to static", exc)
                vf_dynamic = None
        if vf_dynamic is None:
            _run_ffmpeg_progress(
                _build_cmd(vf, out_path), total_out,
                lambda p: _exprogress.__setitem__("percent", p),
                context="anti-duplicate processing", timeout=3600,
            )
            subject_mode = "static" if subject_mode != "dynamic" else subject_mode
    finally:
        _exprogress["running"] = False
        _exprogress["percent"] = 100

    md5 = _file_md5(out_path)
    try:
        dur = _get_duration(out_path)
    except Exception:  # noqa: BLE001
        dur = 0.0

    videos[new_id] = {
        "id": new_id,
        "title": f"{record['title']} (unique)",
        "platform": "local",
        "duration": dur,
        "thumbnail": "",
        "path": str(out_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "clips": {},
    }
    logger.info("Exprocessed %s -> %s (md5 %s, speed %s, zoom %s)", request.videoId, new_id, md5[:12], speed, zoom)

    # Before/after comparison frames (same relative moment; speed-corrected).
    t_orig = min(2.0, max(0.0, float(record.get("duration") or 0.0) * 0.1))
    t_new = t_orig / max(1.01, speed)
    orig_thumb = _frame_thumb(video_path, f"{new_id}_orig", t_orig)
    new_thumb = _frame_thumb(out_path, f"{new_id}_new", t_new)

    return {
        "videoId": new_id,
        "title": videos[new_id]["title"],
        "duration": dur,
        "previewUrl": f"/api/videos/{new_id}/preview",
        "md5": md5,
        "origThumb": orig_thumb,
        "newThumb": new_thumb,
        "aspect": request.outputAspect,
        "zoom": round(zoom, 3),
        "speed": speed,
        "subjectZoom": subject_mode,
        "focusWindows": len(focus_plan),
        "sizeBytes": out_path.stat().st_size,
    }


@app.get("/api/thumbs/{filename}")
def serve_thumb(filename: str) -> FileResponse:
    """Serve a generated before/after comparison thumbnail (name sanitized)."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+\.(jpg|jpeg|png)", filename):
        raise HTTPException(status_code=400, detail="Invalid thumbnail filename")
    path = THUMBS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/videos/{video_id}")
def get_video(video_id: str) -> dict:
    """Return metadata (and clip list) for a registered video."""
    if video_id not in videos:
        raise HTTPException(status_code=404, detail="Video not found")
    return _public_video(video_id)


@app.get("/api/videos/{video_id}/preview")
async def preview_video(video_id: str, request: Request) -> RangeResponse:
    """Stream the full downloaded video (supports HTTP Range for seeking)."""
    record = videos.get(video_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Video not found")

    path = Path(record["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file is missing on disk")
    return RangeResponse(path, request.headers.get("range"))


@app.get("/api/clips/{clip_id}")
async def serve_clip(clip_id: str, request: Request) -> RangeResponse:
    """Stream a generated clip (supports HTTP Range for seeking)."""
    path = CLIPS_DIR / f"{clip_id}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Clip not found")
    return RangeResponse(path, request.headers.get("range"))


@app.get("/api/clips/{clip_id}/thumb.jpg")
def serve_clip_thumbnail(clip_id: str) -> FileResponse:
    """Serve the JPEG thumbnail for a generated clip."""
    path = THUMBS_DIR / f"{clip_id}.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Clip thumbnail not found")
    return FileResponse(path, media_type="image/jpeg")


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str) -> dict:
    """Remove a video plus all of its clips and thumbnails from disk and the store."""
    record = videos.pop(video_id, None)
    if record is None:
        raise HTTPException(status_code=404, detail="Video not found")

    removed_clips = 0
    for clip_id in list(record["clips"].keys()):
        (CLIPS_DIR / f"{clip_id}.mp4").unlink(missing_ok=True)
        (THUMBS_DIR / f"{clip_id}.jpg").unlink(missing_ok=True)
        removed_clips += 1

    # Remove the video file (and any stray part-file remnants).
    _cleanup_leftovers(VIDEOS_DIR, video_id)

    logger.info("Deleted video %s and %d clip(s)", video_id, removed_clips)
    return {"status": "deleted", "videoId": video_id, "clipsRemoved": removed_clips}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
