import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FREE_TIER_MAX_MB = 2048  # 2GB free tier limit


def _get_client(api_key: str = None) -> genai.Client:
    key = api_key or API_KEY
    if not key:
        raise RuntimeError("No API key provided.")
    # Extended timeout for large file uploads (10 min)
    return genai.Client(
        api_key=key,
        http_options=types.HttpOptions(timeout=600_000),  # 600s in ms
    )


def upload_video_to_gemini(file_path: str, mime_type: str = "video/mp4", api_key: str = None) -> str:
    """Upload video to Gemini Files API with retry. Returns the file URI.
    
    - 3 upload attempts with backoff
    - Processing wait scaled by size: 5 min (<500MB) or 15 min (>=500MB)
    """
    client = _get_client(api_key)

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > FREE_TIER_MAX_MB:
        raise ValueError(
            f"Video size {file_size_mb:.0f}MB exceeds free tier limit ({FREE_TIER_MAX_MB}MB). "
            f"Requires a paid Google AI project."
        )

    # --- Upload with retry ---
    max_retries = 3
    uploaded = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[subtitle-rewriter] Uploading video ({file_size_mb:.1f} MB)... attempt {attempt}/{max_retries}")
            uploaded = client.files.upload(
                file=file_path,
                config=types.UploadFileConfig(mime_type=mime_type),
            )
            break
        except Exception as exc:
            if attempt == max_retries:
                raise ValueError(
                    f"Failed to upload video after {max_retries} attempts. Error: {exc}\n"
                    f"Tips: Check internet connection. For files > 1GB, ensure stable "
                    f"connection and sufficient upload bandwidth."
                ) from exc
            wait = attempt * 5
            print(f"[subtitle-rewriter] Upload attempt {attempt} failed: {exc}. Retrying in {wait}s...")
            time.sleep(wait)

    file_uri = uploaded.name
    print(f"[subtitle-rewriter] File accepted: {file_uri}")

    # --- Poll until ACTIVE (scaled timeout) ---
    max_wait_min = 5 if file_size_mb < 500 else 15
    max_polls = max_wait_min * 60 // 5  # poll every 5s
    for i in range(max_polls):
        state = client.files.get(name=file_uri)
        if state.state == types.FileState.ACTIVE:
            print(f"[subtitle-rewriter] Video ready after {(i+1)*5}s")
            return uploaded  # Return the File object (not just URI string)
        if state.state == types.FileState.FAILED:
            err = getattr(state, "error_message", "") or ""
            raise RuntimeError(f"File processing failed: {state.uri}\n{err}")
        if (i + 1) % 6 == 0:
            print(f"[subtitle-rewriter] Still processing... ({(i+1)*5}s)")
        time.sleep(5)

    raise ValueError(
        f"Timed out waiting for Gemini file processing ({max_wait_min} min). "
        f"File: {file_uri}. Try again or check if file is corrupt."
    )


def generate_rewritten_srt(video_file, srt_content: str, api_key: str = None, custom_prompt: str = "") -> str:
    """Send video + SRT to Gemini, get back rewritten SRT.
    
    Args:
        video_file: The File object from client.files.upload() (or file URI string).
        srt_content: The SRT text content.
        api_key: Optional API key override.
        custom_prompt: Optional custom instruction.
    """
    client = _get_client(api_key)

    if custom_prompt:
        instruction = custom_prompt
    else:
        instruction = "Rewrite ALL subtitle lines as a smooth, engaging storytelling/review narration that matches what actually happens in the video."

    prompt = f"""You are a professional video narrator and reviewer.

Given:
1. A video (attached above)
2. Original subtitles in SRT format (below)

TASK: {instruction}

Rules:
- Keep the EXACT same timestamps, sequence numbers, and total number of subtitle entries
- Match the language of the original subtitles
- The rewritten text should reference what is actually happening in the video (actions, scenes, visual elements)
- Output ONLY the complete new SRT content. No explanations, no markdown code fences, no extra text.

Original SRT:
{srt_content}"""

    # Pass the File object directly (works across SDK versions)
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            video_file,
            prompt,
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=65536,
        ),
    )

    text = response.text
    if not text:
        raise RuntimeError("Gemini returned empty response")

    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (fences)
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    return text.strip()


def process_job(video_path: str, srt_content: str, mime_type: str = "video/mp4", api_key: str = None) -> str:
    """Full pipeline: upload video → generate rewritten SRT. Returns the rewritten SRT text."""
    video_file = upload_video_to_gemini(video_path, mime_type, api_key)
    rewritten = generate_rewritten_srt(video_file, srt_content, api_key)
    return rewritten
