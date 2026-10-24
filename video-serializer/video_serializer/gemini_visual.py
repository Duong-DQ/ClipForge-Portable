"""Multimodal Gemini visual segmentation module.

Sends the actual video file + SRT transcript to Gemini in a SINGLE call.
Gemini watches the video (sees scenes, actions, transitions) AND reads the
transcript, then returns:
  - Cut points (segment indices)
  - Per-clip: title, summary, narration

This module requires the `google-genai` package (pip install google-genai).
If the package is not installed or the call fails, the caller should fall
back to the text-only path.
"""
from __future__ import annotations

import json
import os
import time
from typing import List, Optional, Tuple

from .models import PipelineConfig, Topic, TranscriptSegment

__all__ = ["VisualSegmentationError", "visual_segment"]


class VisualSegmentationError(Exception):
    """Raised when visual segmentation fails."""


def _get_client(api_key: str):
    """Create a google-genai client with extended timeout for large files.
    Raises if package missing or key empty.
    """
    if not api_key:
        raise VisualSegmentationError("API key is required for visual mode.")
    try:
        from google import genai
        from google.genai import types as _types
    except ImportError as exc:
        raise VisualSegmentationError(
            "google-genai package not installed. Run: pip install google-genai"
        ) from exc
    # 600s timeout — large video uploads need extended time
    return genai.Client(
        api_key=api_key,
        http_options=_types.HttpOptions(timeout=600_000),  # 10 min in ms
    )


def _upload_video(client, file_path: str, mime_type: str = "video/mp4") -> str:
    """Upload video to Gemini Files API and wait until ACTIVE. Returns file URI.
    
    Includes retry logic (3 attempts) for the upload step and scales the
    processing wait time based on file size (up to 15 min for 2GB files).
    """
    from google.genai import types

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > 2048:
        raise VisualSegmentationError(
            f"Video size {file_size_mb:.0f}MB exceeds 2GB free-tier limit. "
            f"A paid Google AI project is required for larger files."
        )

    # --- Upload with retry ---
    max_retries = 3
    uploaded = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[visual] Uploading video ({file_size_mb:.1f} MB) to Gemini Files API... "
                  f"(attempt {attempt}/{max_retries})")
            uploaded = client.files.upload(
                file=file_path,
                config=types.UploadFileConfig(mime_type=mime_type),
            )
            break
        except Exception as exc:
            if attempt == max_retries:
                raise VisualSegmentationError(
                    f"Failed to upload video after {max_retries} attempts. "
                    f"Error: {exc}\n"
                    f"Tips: Check your internet connection. For files > 1GB, "
                    f"ensure stable connection and sufficient upload bandwidth."
                ) from exc
            wait = attempt * 5  # 5s, 10s backoff
            print(f"[visual] Upload attempt {attempt} failed: {exc}. Retrying in {wait}s...")
            time.sleep(wait)

    file_uri = uploaded.name
    print(f"[visual] File accepted by Gemini: {file_uri}")

    # --- Poll until ACTIVE (scaled timeout) ---
    # Small files (< 500MB): max 5 min. Large files: max 15 min.
    max_wait_min = 5 if file_size_mb < 500 else 15
    max_polls = max_wait_min * 60 // 5  # poll every 5s
    for i in range(max_polls):
        state = client.files.get(name=file_uri)
        if state.state == types.FileState.ACTIVE:
            elapsed = (i + 1) * 5
            print(f"[visual] Video ready after {elapsed}s: {file_uri}")
            return file_uri
        if state.state == types.FileState.FAILED:
            err_msg = getattr(state, "error_message", "") or ""
            raise VisualSegmentationError(
                f"Gemini file processing failed: {state.uri}\n{err_msg}"
            )
        # Progress indicator every 30s
        if (i + 1) % 6 == 0:
            print(f"[visual] Still processing... ({(i+1)*5}s elapsed)")
        time.sleep(5)

    raise VisualSegmentationError(
        f"Timed out waiting for Gemini file processing ({max_wait_min} min). "
        f"File: {file_uri}. This can happen with very large files — try again."
    )


def _build_prompt(segments: List[TranscriptSegment], custom_prompt: str, min_dur: float, max_dur: float) -> str:
    """Build the multimodal prompt asking Gemini to watch video + read SRT."""
    lines = []
    for seg in segments:
        text = (seg.text or "").replace("\t", " ").replace("\n", " ").strip()
        lines.append(f"{seg.index}\t{seg.start:.2f}-{seg.end:.2f}\t{text}")
    transcript = "\n".join(lines)

    user_instruction = custom_prompt.strip() if custom_prompt else (
        "Split into ~3-5 minute clips at natural topic/scene changes. "
        "For each clip, write a short title, a 1-2 sentence summary of what "
        "happens visually, and a 2-3 sentence engaging narration as if "
        "storytelling what the viewer is watching."
    )

    prompt = f"""You are a professional video editor and narrator.
You are watching a video (attached) and reading its subtitles (below).

TASK: {user_instruction}

CONSTRAINTS:
- Cut ONLY at segment boundaries (use the segment index numbers).
- Each clip should be between {min_dur:.0f}s and {max_dur:.0f}s.
- Cuts should fall at natural scene/topic changes visible in the video.
- 'narration' should describe what is ACTUALLY happening in the video (actions, scenes, visual elements), not just repeat the subtitle text.
- Match the language of the original subtitles.

Return STRICT JSON (nothing else, no markdown fences) of the form:
{{
  "cuts": [c1, c2, ...],
  "clips": [
    {{"title": "...", "summary": "...", "narration": "..."}},
    ...
  ]
}}

Where:
- "cuts" is a list of 0-based indices of the LAST segment of each clip (strictly increasing).
- "clips" has one entry per clip (same length as cuts + 1 if the last clip extends to the end).
- Each clip entry has "title" (short), "summary" (1-2 sentences about visuals), "narration" (2-3 sentences storytelling).

TRANSCRIPT ({len(segments)} segments):
{transcript}"""
    return prompt


def _parse_response(text: str) -> Tuple[List[int], List[dict]]:
    """Parse the JSON response from Gemini. Returns (cuts, clips_info)."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    obj = json.loads(text)
    raw_cuts = obj.get("cuts", [])
    clips_raw = obj.get("clips", [])

    # Validate cuts
    cuts: List[int] = []
    last = -1
    for item in raw_cuts:
        try:
            val = int(item)
        except (TypeError, ValueError):
            continue
        if val < 0 or val <= last:
            continue
        cuts.append(val)
        last = val

    if not cuts:
        raise VisualSegmentationError("Gemini returned no valid cut points.")

    # Validate clips
    clips: List[dict] = []
    for i, c in enumerate(clips_raw):
        if isinstance(c, dict):
            clips.append({
                "title": str(c.get("title", "")).strip(),
                "summary": str(c.get("summary", "")).strip(),
                "narration": str(c.get("narration", "")).strip(),
            })
        else:
            clips.append({"title": "", "summary": "", "narration": ""})

    return cuts, clips


def visual_segment(
    video_path: str,
    segments: List[TranscriptSegment],
    config: PipelineConfig,
) -> List[Topic]:
    """Full visual segmentation: upload video + ONE Gemini call → topics.

    Parameters
    ----------
    video_path:
        Path to the video file on disk.
    segments:
        Parsed transcript segments in time order.
    config:
        Pipeline configuration (uses llm_api_key, llm_model, llm_custom_prompt,
        min_topic_duration, max_topic_duration).

    Returns
    -------
    List[Topic]
        Topics with title, summary, and narration populated.

    Raises
    ------
    VisualSegmentationError
        On any failure (missing key, upload failure, API error, bad response).
    """
    if not segments:
        raise VisualSegmentationError("No transcript segments provided.")
    if not os.path.isfile(video_path):
        raise VisualSegmentationError(f"Video file not found: {video_path}")

    api_key = config.llm_api_key
    model = config.llm_model or "gemini-2.5-flash"
    custom_prompt = config.llm_custom_prompt
    mime_type = "video/mp4"
    ext = os.path.splitext(video_path)[1].lower()
    mime_map = {
        ".mp4": "video/mp4", ".mov": "video/quicktime",
        ".avi": "video/x-msvideo", ".mkv": "video/x-matroska",
        ".webm": "video/webm",
    }
    if ext in mime_map:
        mime_type = mime_map[ext]

    client = _get_client(api_key)

    # Step 1: Upload video
    file_uri = _upload_video(client, video_path, mime_type)

    # Step 2: Build prompt and call Gemini
    print("[visual] Sending video + transcript to Gemini for analysis...")
    prompt = _build_prompt(segments, custom_prompt, config.min_topic_duration, config.max_topic_duration)

    from google.genai import types as genai_types

    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                genai_types.Part.from_uri(uri=file_uri, mime_type=mime_type),
                genai_types.Part.from_text(text=prompt),
            ],
            config=genai_types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=8192,
            ),
        )
    except Exception as exc:
        raise VisualSegmentationError(f"Gemini API call failed: {exc}") from exc

    text = response.text
    if not text or not text.strip():
        raise VisualSegmentationError("Gemini returned empty response.")

    # Step 3: Parse response
    try:
        cuts, clips_info = _parse_response(text)
    except (json.JSONDecodeError, VisualSegmentationError) as exc:
        raise VisualSegmentationError(f"Failed to parse Gemini response: {exc}\nRaw: {text[:500]}") from exc

    # Step 4: Build topics from cuts
    n = len(segments)
    topics: List[Topic] = []
    start = 0
    clip_idx = 0
    for cut in cuts:
        cut = min(cut, n - 1)  # clamp
        group = list(range(start, cut + 1))
        members = [segments[i] for i in group]
        info = clips_info[clip_idx] if clip_idx < len(clips_info) else {}
        topics.append(Topic(
            index=len(topics),
            start=members[0].start,
            end=members[-1].end,
            segments=members,
            title=info.get("title", ""),
            summary=info.get("summary", ""),
            narration=info.get("narration", ""),
        ))
        start = cut + 1
        clip_idx += 1

    # Trailing segments after last cut
    if start < n:
        group = list(range(start, n))
        members = [segments[i] for i in group]
        info = clips_info[clip_idx] if clip_idx < len(clips_info) else {}
        topics.append(Topic(
            index=len(topics),
            start=members[0].start,
            end=members[-1].end,
            segments=members,
            title=info.get("title", ""),
            summary=info.get("summary", ""),
            narration=info.get("narration", ""),
        ))

    print(f"[visual] Segmentation complete: {len(topics)} clip(s)")
    return topics
