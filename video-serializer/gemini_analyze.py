#!/usr/bin/env python3
"""
Gemini Transcript Analyzer — standalone script.

Phân tích transcript (SRT/VTT/txt) bằng Gemini API để:
  - Chia video thành các tập (episodes) theo chủ đề
  - Viết recap/tóm tắt cho từng tập
  - Gợi ý điểm kịch tính (highlight timestamps)
  - Xác định điểm cắt tối ưu (cliffhanger)

Yêu cầu:
  - Python 3.9+ (stdlib only, không cần pip install)
  - API key: Google AI Studio (free) -> https://aistudio.google.com/apikey
  - FFmpeg trên PATH (không bắt buộc, chỉ cho inforation)

Cách dùng:
  python gemini_analyze.py transcript.txt --duration 600
  python gemini_analyze.py transcript.srt --duration 600 --episodes 3
  python gemini_analyze.py transcript.txt --duration 600 --json  # output JSON

Config (theo thứ tự ưu tiên):
  1. Env vars: LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_PROXY
  2. Sua cac hang so ben duoi
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

# =============================================================================
# CONFIG — sửa các giá trị này hoặc dùng env vars
# =============================================================================

API_KEY = os.environ.get("LLM_API_KEY", "")  # ← dán Google AI key vào đây
BASE_URL = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
MODEL = os.environ.get("LLM_MODEL", "gemini-2.0-flash")  # gemini-2.0-flash / gemini-1.5-flash
PROXY = os.environ.get("LLM_PROXY", "")  # ví dụ: "http://127.0.0.1:8899" — để trống = auto
TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "60"))


# =============================================================================
# PROXY — auto-detect từ Windows registry / env vars
# =============================================================================

def _auto_proxy() -> str:
    """Detect proxy from env vars or Windows registry (IE settings)."""
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        val = os.environ.get(var, "").strip()
        if val:
            return val
    # Windows registry
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
            0, winreg.KEY_READ,
        )
        try:
            enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
            if enabled:
                server, _ = winreg.QueryValueEx(key, "ProxyServer")
                if server and "=" not in server:
                    return "http://{}".format(server)
                return server
        finally:
            winreg.CloseKey(key)
    except Exception:
        pass
    return ""


def _build_opener() -> urllib.request.OpenerDirector:
    """Build urllib opener with proxy support."""
    proxy = PROXY or _auto_proxy()
    if proxy:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        )
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


# =============================================================================
# GEMINI API
# =============================================================================

def gemini_chat(system: str, user: str, temperature: float = 0.4, max_tokens: int = 4096) -> str:
    """Call Gemini API (OpenAI-compatible). Returns response text or raises."""
    api_key = API_KEY
    if not api_key:
        raise RuntimeError(
            "No API key. Set LLM_API_KEY env var or edit API_KEY constant in this script.\n"
            "Get free key: https://aistudio.google.com/apikey"
        )

    base = BASE_URL.rstrip("/")
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    url = base + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
            "User-Agent": "video-serializer-analyzer/1.0",
        },
        method="POST",
    )

    opener = _build_opener()
    with opener.open(req, timeout=TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    msg = (data.get("choices") or [{}])[0].get("message") or {}
    content = msg.get("content") or msg.get("reasoning") or msg.get("refusal") or ""
    if not isinstance(content, str):
        content = str(content)
    content = content.strip()
    if not content:
        raise RuntimeError("Gemini returned empty response.")
    return content


def gemini_chat_json(system: str, user: str, temperature: float = 0.3, max_tokens: int = 4096) -> dict:
    """Call Gemini and parse JSON from response."""
    raw = gemini_chat(system, user, temperature, max_tokens)
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Extract from markdown code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # Find first { ... } block
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError("Could not parse JSON from response: " + raw[:200] + "...")


# =============================================================================
# TRANSCRIPT PARSING (SRT / VTT / plain text)
# =============================================================================

def parse_transcript(path: str) -> str:
    """Read transcript file, return clean text (strip SRT/VTT headers)."""
    text = Path(path).read_text(encoding="utf-8", errors="replace").strip()

    # SRT: remove line numbers + timestamp lines
    if re.search(r"^\d+\s*\n\d{2}:\d{2}:\d{2}[,.]\d+\s*-->", text, re.MULTILINE):
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if re.match(r"^\d+$", line):
                continue
            if re.match(r"^\d{2}:\d{2}:\d{2}[,.]\d+\s*-->", line):
                continue
            lines.append(line)
        return "\n".join(lines)

    # VTT
    if text.startswith("WEBVTT"):
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("WEBVTT") or line == "NOTE":
                continue
            if re.match(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->", line):
                continue
            if re.match(r"^\d+$", line):
                continue
            lines.append(line)
        return "\n".join(lines)

    return text


# =============================================================================
# LLM PROMPTS
# =============================================================================

SYSTEM_PROMPT = """Bạn là một chuyên gia biên tập video (video editor) và phân tích nội dung.
Nhiệm vụ: phân tích transcript video để xác định:
1. Điểm chia tập (episode) tự nhiên nhất
2. Các moment kịch tính nhất (highlights)
3. Viết recap tóm tắt hấp dẫn cho intro tập tiếp theo

Trả về CHỈ JSON hợp lệ. Không markdown, không giải thích ngoài JSON.
"""

USER_PROMPT_TEMPLATE = """Phân tích transcript video sau và trả về JSON.

THỜI LƯỢNG VIDEO: {duration} giây
SỐ TẬP MUỐN CHIA: {episodes}

TRANSCRIPT:
{transcript}

TRẢ VỀ JSON CẤU TRÚC SAU:
{{
  "topics": [
    {{
      "title": "Tên tập ngắn gọn (tối đa 8 từ, kịch tính)",
      "start": <giây - điểm bắt đầu tập này>,
      "end": <giây - điểm kết thúc tập này>,
      "summary": "Tóm tắt 1-2 câu về nội dung chính",
      "highlights": [[start_sec, end_sec], [start_sec, end_sec]],
      "dramatic_score": <0.0-1.0 - mức độ kịch tính>
    }}
  ],
  "recap_text": "Recap 2-3 câu kịch tính (tối đa 80 từ) cho intro 'Trước đó...' — tóm tắt mâu thuẫn chính, gợi tò mò. Giọng văn: drama/gossip.",
  "cut_points": [<giây>, <giây>],
  "cliffhanger_point": <giây - điểm cliffhanger hay nhất>,
  "overall_genre": "thể loại (drama/news/gaming/vlog/...)"
}}

QUY TẮC:
- Các topics PHẢI liên tục (không gap, không overlap)
- Topic đầu bắt đầu ở 0, topic cuối kết thúc ở {duration}
- Highlights là các đoạn 3-5 giây kịch tính nhất trong topic đó
- cut_points ở các điểm nghỉ tự nhiên hoặc cliffhanger
- Dựa vào nội dung transcript để ước lượng thời gian (trung bình ~150 từ/phút nếu không có timestamp)
"""


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_transcript(
    transcript_path: str,
    duration: float,
    episodes: int = 3,
) -> dict:
    """Main entry: read transcript file, call Gemini, return structured result."""
    print("📖 Reading: {}".format(transcript_path), file=sys.stderr)
    text = parse_transcript(transcript_path)
    if not text:
        raise ValueError("Transcript is empty")

    # Cap for context window safety
    if len(text) > 30000:
        text = text[:30000] + "\n[... truncated ...]"

    print("📝 Transcript: {} chars, {} words".format(len(text), len(text.split())), file=sys.stderr)
    print("⏱️  Duration: {}s | 🎬 Episodes: {}".format(duration, episodes), file=sys.stderr)
    print("🤖 Calling Gemini API...", file=sys.stderr)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        duration=duration,
        episodes=episodes,
        transcript=text,
    )

    result = gemini_chat_json(SYSTEM_PROMPT, user_prompt, temperature=0.3, max_tokens=4096)

    if "topics" not in result:
        raise ValueError("Gemini response missing 'topics'. Keys: {}".format(list(result.keys())))

    print("✅ Done! {} episodes found.".format(len(result["topics"])), file=sys.stderr)
    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python gemini_analyze.py <transcript.txt|srt|vtt> --duration <sec> [--episodes N] [--json]")
        print("  --duration:  Video duration in seconds (REQUIRED)")
        print("  --episodes:  Number of episodes (default: 3)")
        print("  --json:      Output raw JSON only")
        sys.exit(1)

    transcript_path = sys.argv[1]
    duration = 0.0
    episodes = 3
    output_json = False

    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--duration" and i + 1 < len(sys.argv):
            duration = float(sys.argv[i + 1])
            i += 2
        elif arg == "--episodes" and i + 1 < len(sys.argv):
            episodes = int(sys.argv[i + 1])
            i += 2
        elif arg == "--json":
            output_json = True
            i += 1
        else:
            print("Unknown arg: " + arg, file=sys.stderr)
            i += 1

    if duration <= 0:
        print("Error: --duration is required (video length in seconds)", file=sys.stderr)
        sys.exit(1)

    if not Path(transcript_path).exists():
        print("Error: File not found: " + transcript_path, file=sys.stderr)
        sys.exit(1)

    try:
        result = analyze_transcript(transcript_path, duration, episodes)
    except Exception as e:
        print("\n❌ Error: {}".format(e), file=sys.stderr)
        sys.exit(2)

    if output_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # Human-readable
        print("\n" + "=" * 60)
        print("📺 Video Analysis Result")
        print("=" * 60)
        print("Genre: {}".format(result.get("overall_genre", "unknown")))
        print("\n📝 RECAP:")
        print("   " + result.get("recap_text", "N/A"))
        print("\n🎬 EPISODES ({}):".format(len(result.get("topics", []))))
        for idx, topic in enumerate(result.get("topics", []), 1):
            start = topic.get("start", 0)
            end = topic.get("end", duration)
            print("\n   {}. [{}s → {}s] {}".format(idx, start, end, topic.get("title", "Untitled")))
            print("      " + topic.get("summary", ""))
            hl = topic.get("highlights", [])
            if hl:
                hl_str = ", ".join("{}-{}s".format(a, b) for a, b in hl)
                print("      🔥 Highlights: " + hl_str)
            score = topic.get("dramatic_score")
            if score is not None:
                print("      💥 Drama: {:.2f}/1.0".format(score))

        cuts = result.get("cut_points", [])
        if cuts:
            print("\n✂️  CUT POINTS: " + ", ".join("{}s".format(c) for c in cuts))

        cliff = result.get("cliffhanger_point")
        if cliff is not None:
            print("⛓️  CLIFFHANGER: {}s".format(cliff))

        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
