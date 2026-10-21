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
    return genai.Client(api_key=key)


def upload_video_to_gemini(file_path: str, mime_type: str = "video/mp4", api_key: str = None) -> str:
    """Upload video to Gemini Files API. Returns the file URI."""
    client = _get_client(api_key)

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > FREE_TIER_MAX_MB:
        raise ValueError(
            f"Video size {file_size_mb:.0f}MB exceeds free tier limit ({FREE_TIER_MAX_MB}MB). "
            f"Requires a paid Google AI project."
        )

    uploaded = client.files.upload(file=file_path, config=types.UploadFileConfig(mime_type=mime_type))
    file_uri = uploaded.name  # e.g. "files/abc123"

    # Wait for file processing to complete
    while True:
        state = client.files.get(name=file_uri)
        if state.state == types.FileState.ACTIVE:
            break
        if state.state == types.FileState.FAILED:
            raise RuntimeError(f"File processing failed: {state.uri}")
        time.sleep(2)

    return file_uri


def generate_rewritten_srt(video_uri: str, srt_content: str, api_key: str = None, custom_prompt: str = "") -> str:
    """Send video + SRT to Gemini, get back rewritten SRT."""
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

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_uri(uri=video_uri, mime_type="video/mp4"),
            types.Part.from_text(text=prompt),
        ],
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


def process_job(video_path: str, srt_content: str, mime_type: str = "video/mp4") -> str:
    """Full pipeline: upload video → generate rewritten SRT. Returns the rewritten SRT text."""
    video_uri = upload_video_to_gemini(video_path, mime_type)
    rewritten = generate_rewritten_srt(video_uri, srt_content)
    return rewritten
