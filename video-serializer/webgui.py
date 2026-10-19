#!/usr/bin/env python3
"""
webgui.py - Local web UI for the video serializer (stdlib only + ffmpeg).

Run:
    python webgui.py            # opens the browser automatically
    python webgui.py --port 9000
    python webgui.py --no-open

What it does:
  * Serves a single-page UI (webgui.html) at http://127.0.0.1:<port>/
  * You drag-drop a VIDEO file and a TRANSCRIPT (.srt / .vtt).
  * The transcript is parsed and shown as a readable script with detected
    topics (a live preview of how the video will be split).
  * Click "Tach video" to cut the video into clips; each clip plays inline
    in the browser and can be downloaded, along with metadata.json.

Uses ONLY the Python standard library for the server. Cutting uses ffmpeg
(ffmpeg must be on PATH, same as the CLI). No pip installs required.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
import urllib.parse
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Make the sibling `video_serializer` package importable regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from video_serializer.models import PipelineConfig  # noqa: E402
from video_serializer.transcript import parse_transcript, TranscriptParseError  # noqa: E402
from video_serializer.segmentation import segment_topics, SegmentationError  # noqa: E402
from video_serializer.enrich import enrich_topics  # noqa: E402
from video_serializer.cutter import cut_video, ensure_ffmpeg, FFmpegError  # noqa: E402
from video_serializer.recap import add_recaps  # noqa: E402
from video_serializer.utils import format_timestamp  # noqa: E402
from video_serializer.hooks import load_hooks  # noqa: E402

# Windows console is cp1252 by default; make stdout/stderr UTF-8 safe.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

TOOL_DIR = _HERE
HTML_FILE = os.path.join(TOOL_DIR, "webgui.html")
WORK_DIR = tempfile.mkdtemp(prefix="video_serializer_web_")
DEFAULT_OUTPUT_ROOT = os.path.join(TOOL_DIR, "web_output")

# Serialize the (slow) /api/process so two runs don't clobber each other's output dir.
_PROCESS_LOCK = threading.Lock()
_LAST_OUTPUT_DIR = {"dir": ""}  # newest /api/process output dir -> served via /clips + /metadata.json

_OPTION_KEYS = ("min_duration", "max_duration", "gap", "similarity",
                "use_embeddings", "cut_mode", "cut_strategy", "metadata_format")

# Gemini defaults for the opt-in AI mode. Mirror the same values the CLI uses
# so the two front-ends stay in sync. These are filled in by _build_config when
# the user enables use_llm and leaves the advanced fields blank. The API key is
# never read from a URL query string -- it only travels in the JSON body of
# /api/process, where it stays in process memory.
_LLM_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
_LLM_DEFAULT_MODEL = "gemini-3.8-flash"


def _build_config(payload: dict) -> PipelineConfig:
    """Build a PipelineConfig from a flat dict of option strings/numbers."""
    def _f(key: str, default: float) -> float:
        try:
            return float(payload.get(key, default))
        except (TypeError, ValueError):
            return default

    ue = str(payload.get("use_embeddings", "auto")).strip().lower()
    use_embeddings = None if ue in ("", "auto") else (ue in ("true", "1", "yes"))

    cut_mode = str(payload.get("cut_mode", "accurate")).strip().lower()
    if cut_mode not in ("accurate", "fast"):
        cut_mode = "accurate"

    strategy = str(payload.get("strategy", "topic")).strip().lower()
    if strategy not in ("topic", "retention"):
        strategy = "topic"

    meta = str(payload.get("metadata_format", "json")).strip().lower()
    if meta not in ("json", "csv"):
        meta = "json"

    add_recaps = str(payload.get("add_recaps", "false")).strip().lower() in ("1", "true", "yes", "on")
    recap_voice = str(payload.get("recap_voice", "")).strip()  # empty = auto by recap language
    recap_lang = str(payload.get("recap_lang", "auto")).strip().lower()
    if recap_lang not in ("auto", "vi", "en", "zh"):
        recap_lang = "auto"

    # Topic hook selection: accept only values that exist in hooks.json (plus
    # "none" = off). Unknown / missing values degrade to "none" so the rest of
    # the pipeline behaves exactly as before.
    topic = str(payload.get("topic", "none")).strip().lower()
    if topic != "none" and topic not in load_hooks():
        topic = "none"

    # Opt-in AI (Gemini) mode. The flag comes from the Web UI checkbox; the key
    # is never persisted on disk -- it lives only in this request payload. The
    # base URL and model default to the confirmed-working Gemini endpoint when
    # use_llm is on and the user left them blank.
    use_llm = str(payload.get("use_llm", "false")).strip().lower() in (
        "1", "true", "yes", "on",
    )
    llm_api_key = str(payload.get("llm_api_key", "")).strip()
    llm_base_url = str(payload.get("llm_base_url", "")).strip()
    llm_model = str(payload.get("llm_model", "")).strip()
    if use_llm:
        if not llm_base_url:
            llm_base_url = _LLM_DEFAULT_BASE_URL
        if not llm_model:
            llm_model = _LLM_DEFAULT_MODEL

    return PipelineConfig(
        min_topic_duration=_f("min_duration", 20.0),
        max_topic_duration=_f("max_duration", 300.0),
        gap_threshold=_f("gap", 1.5),
        similarity_threshold=_f("similarity", 0.45),
        use_embeddings=use_embeddings,
        cut_mode=cut_mode,
        cut_strategy=strategy,
        topic=topic,
        metadata_format=meta,
        add_recaps=add_recaps,
        recap_duration=max(1.0, _f("recap_duration", 4.0)),
        recap_voice=recap_voice,
        recap_lang=recap_lang,
        use_llm=use_llm,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
    )


def _ts(seconds: float) -> str:
    return format_timestamp(seconds)


class Handler(BaseHTTPRequestHandler):
    server_version = "VideoSerializerWeb/1.0"
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------------ utils
    def _send_json(self, obj, code: int = 200) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, text: str) -> None:
        data = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file_full(self, path: str, content_type: str, download: bool = False,
                        filename: 'str | None' = None) -> None:
        size = os.path.getsize(path)
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        if download:
            self.send_header("Content-Disposition",
                             'attachment; filename="%s"' % (filename or os.path.basename(path)))
        else:
            self.send_header("Content-Disposition", "inline")
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def _handle_export(self, qs: dict) -> None:
        """Zip every file in an allowed output dir and serve it as a download."""
        requested = qs.get("dir", [""])[0] or _LAST_OUTPUT_DIR["dir"]
        target = os.path.realpath(requested) if requested else ""
        allowed_bases = [
            b for b in (os.path.realpath(DEFAULT_OUTPUT_ROOT),
                        os.path.realpath(_LAST_OUTPUT_DIR["dir"]))
            if b and os.path.isdir(b)
        ]
        if (not target or not os.path.isdir(target)
                or not any(target == b or target.startswith(b + os.sep)
                           for b in allowed_bases)):
            self._send_json({"ok": False, "error": "export dir not allowed"}, 403)
            return
        files = [
            os.path.join(target, n)
            for n in sorted(os.listdir(target))
            if os.path.isfile(os.path.join(target, n))
        ]
        if not files:
            self._send_json({"ok": False, "error": "nothing to export"}, 404)
            return
        fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="vs_export_")
        os.close(fd)
        os.remove(zip_path)  # let zipfile create the file
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for fp in files:
                    zf.write(fp, os.path.basename(fp))
            friendly = (re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(target))
                        or "video_clips") + ".zip"
            self._send_file_full(zip_path, "application/zip", download=True,
                                 filename=friendly)
        finally:
            try:
                os.remove(zip_path)
            except OSError:
                pass

    def _send_video(self, path: str) -> None:
        """Serve a video with HTTP Range support so <video> can seek/play."""
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        match = re.match(r"bytes=(\d*)-(\d*)", rng) if rng else None
        if match and (match.group(1) or match.group(2)):
            start = int(match.group(1)) if match.group(1) else 0
            end = int(match.group(2)) if match.group(2) else size - 1
            end = min(end, size - 1)
            if start >= size or start > end:
                self.send_response(416)
                self.send_header("Content-Range", "bytes */%d" % size)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
            self.send_header("Content-Length", str(length))
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            return

        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(length) if length > 0 else b""

    def _stream_body_to(self, dest_path: str) -> int:
        length = int(self.headers.get("Content-Length", 0) or 0)
        written = 0
        with open(dest_path, "wb") as f:
            while written < length:
                chunk = self.rfile.read(min(1024 * 1024, length - written))
                if not chunk:
                    break
                f.write(chunk)
                written += len(chunk)
        return written

    def _query_and_path(self):
        parsed = urllib.parse.urlparse(self.path)
        return urllib.parse.parse_qs(parsed.query), parsed.path

    def log_message(self, fmt, *args):  # keep the console quiet except errors
        try:
            status = str(args[1]) if len(args) >= 2 else ""
            if status.startswith("5") or "Traceback" in fmt:
                sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))
        except Exception:
            pass

    # -------------------------------------------------------------------- GET
    def do_GET(self) -> None:
        qs, path = self._query_and_path()
        try:
            if path in ("/", "/index.html"):
                if os.path.isfile(HTML_FILE):
                    with open(HTML_FILE, "r", encoding="utf-8") as f:
                        self._send_html(f.read())
                else:
                    self._send_html(
                        "<h1>webgui.html chua co</h1>"
                        "<p>Front-end chua duoc tao. API van hoat dong: <code>/api/health</code></p>"
                    )
                return

            if path == "/api/health":
                self._send_json({
                    "ok": True,
                    "work_dir": WORK_DIR,
                    "last_output_dir": _LAST_OUTPUT_DIR["dir"],
                    "ffmpeg_found": shutil.which("ffmpeg") is not None,
                })
                return

            if path == "/api/export":
                self._handle_export(qs)
                return

            if path.startswith("/clips/"):
                name = os.path.basename(urllib.parse.unquote(path[len("/clips/"):]))
                if not name or name in (".", ".."):
                    self._send_json({"ok": False, "error": "bad path"}, 400)
                    return
                outdir = _LAST_OUTPUT_DIR["dir"]
                if not outdir:
                    self._send_json({"ok": False, "error": "no run yet"}, 404)
                    return
                full = os.path.join(outdir, name)
                if not os.path.isfile(full):
                    self._send_json({"ok": False, "error": "clip not found"}, 404)
                    return
                download = qs.get("download", ["0"])[0].lower() in ("1", "true", "yes")
                if download:
                    self._send_file_full(full, "video/mp4", download=True)
                else:
                    self._send_video(full)
                return

            if path == "/metadata.json":
                outdir = _LAST_OUTPUT_DIR["dir"]
                meta = os.path.join(outdir, "metadata.json") if outdir else ""
                if meta and os.path.isfile(meta):
                    self._send_file_full(meta, "application/json; charset=utf-8")
                else:
                    self._send_json({"ok": False, "error": "no metadata yet"}, 404)
                return

            self._send_json({"ok": False, "error": "not found: " + path}, 404)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "error": repr(exc)}, 500)

    # ------------------------------------------------------------------- POST
    def do_POST(self) -> None:
        qs, path = self._query_and_path()
        try:
            if path == "/api/parse":
                self._handle_parse(qs)
            elif path == "/api/upload":
                self._handle_upload(qs)
            elif path == "/api/process":
                self._handle_process()
            else:
                self._send_json({"ok": False, "error": "not found: " + path}, 404)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "error": repr(exc)}, 500)

    def _transcript_ext(self, filename: str, head: bytes) -> str:
        name = (filename or "").lower()
        if name.endswith(".srt"):
            return "srt"
        if name.endswith(".vtt"):
            return "vtt"
        return "vtt" if head.lstrip().startswith(b"WEBVTT") else "srt"

    def _handle_parse(self, qs) -> None:
        filename = qs.get("filename", ["transcript"])[0]
        body = self._body()
        if not body.strip():
            self._send_json({"ok": False, "error": "empty transcript body"}, 400)
            return
        ext = self._transcript_ext(filename, body)
        tmp = os.path.join(WORK_DIR, "tr_%d_%d.%s" % (os.getpid(), int(time.time() * 1000), ext))
        with open(tmp, "wb") as f:
            f.write(body)

        # Build config from any option query params so the preview matches the run.
        cfg_payload = {k: qs[k][0] for k in _OPTION_KEYS if k in qs}
        try:
            cfg = _build_config(cfg_payload)
            segments = parse_transcript(tmp)
            if not segments:
                self._send_json({"ok": False, "error": "No usable cues found in the transcript."}, 400)
                return
            topics = segment_topics(segments, cfg)
            enrich_topics(topics, cfg)
        except (TranscriptParseError, SegmentationError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, 400)
            return

        transcript = [
            {"index": s.index, "start": s.start, "end": s.end,
             "start_ts": _ts(s.start), "end_ts": _ts(s.end), "text": s.text}
            for s in segments
        ]
        topics_out = [
            {"index": t.index + 1, "title": t.title, "summary": t.summary,
             "start": t.start, "end": t.end, "start_ts": _ts(t.start), "end_ts": _ts(t.end),
             "duration": round(t.end - t.start, 3)}
            for t in topics
        ]
        self._send_json({
            "ok": True,
            "transcript_path": tmp,
            "segment_count": len(segments),
            "transcript": transcript,
            "topics": topics_out,
        })

    def _handle_upload(self, qs) -> None:
        filename = qs.get("filename", ["video.mp4"])[0]
        safe_name = os.path.basename(filename) or "video.mp4"
        ext = os.path.splitext(safe_name)[1] or ".mp4"
        dest = os.path.join(WORK_DIR, "vid_%d_%d%s" % (os.getpid(), int(time.time() * 1000), ext))
        written = self._stream_body_to(dest)
        if written <= 0:
            self._send_json({"ok": False, "error": "empty upload"}, 400)
            return
        self._send_json({"ok": True, "video_path": dest, "name": safe_name})

    def _handle_process(self) -> None:
        raw = self._body()
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:  # noqa: BLE001
            self._send_json({"ok": False, "error": "invalid JSON body"}, 400)
            return

        video_path = payload.get("video_path", "")
        transcript_path = payload.get("transcript_path", "")
        if not video_path or not os.path.isfile(video_path):
            self._send_json({"ok": False, "error": "video_path missing or not found on server"}, 400)
            return
        if not transcript_path or not os.path.isfile(transcript_path):
            self._send_json({"ok": False, "error": "transcript_path missing or not found on server"}, 400)
            return

        cfg = _build_config(payload)
        outdir = str(payload.get("output_dir", "")).strip()
        if not outdir:
            outdir = os.path.join(DEFAULT_OUTPUT_ROOT, time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(outdir, exist_ok=True)

        with _PROCESS_LOCK:
            try:
                ensure_ffmpeg()
                segments = parse_transcript(transcript_path)
                if not segments:
                    self._send_json({"ok": False, "error": "No usable cues found in the transcript."}, 400)
                    return
                topics = segment_topics(segments, cfg)
                enrich_topics(topics, cfg)
                cut_video(video_path, topics, outdir, cfg)
                if cfg.add_recaps:
                    add_recaps(topics, outdir, cfg)
            except (TranscriptParseError, SegmentationError, FFmpegError) as exc:
                self._send_json({"ok": False, "error": str(exc)}, 500)
                return

            episodes = []
            for t in topics:
                base = os.path.basename(t.file) if t.file else ""
                episodes.append({
                    "index": t.index + 1, "title": t.title, "summary": t.summary,
                    "start": t.start, "end": t.end, "duration": round(t.end - t.start, 3),
                    "intro_seconds": round(getattr(t, "intro_seconds", 0.0), 3),
                    "start_timestamp": _ts(t.start), "end_timestamp": _ts(t.end),
                    "file": t.file or "",
                    "clip": ("/clips/" + base) if base else "",
                    "download": ("/clips/" + base + "?download=1") if base else "",
                })
            meta = {
                "source_video": video_path,
                "transcript": transcript_path,
                "total_episodes": len(episodes),
                "total_duration_seconds": round(sum(e["duration"] for e in episodes), 3),
                "episodes": episodes,
            }
            with open(os.path.join(outdir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            _LAST_OUTPUT_DIR["dir"] = outdir

        self._send_json({
            "ok": True,
            "output_dir": outdir,
            "episode_count": len(episodes),
            "episodes": episodes,
            "metadata_url": "/metadata.json",
        })


def find_free_port(preferred: int = 8765) -> int:
    for candidate in (preferred,):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", candidate))
            return candidate
        except OSError:
            pass
        finally:
            s.close()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    port = 8765
    open_browser = True
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--port" and i + 1 < len(argv):
            try:
                port = int(argv[i + 1])
            except ValueError:
                port = 8765
            i += 1
        elif arg == "--no-open":
            open_browser = False
        i += 1

    if shutil.which("ffmpeg") is None:
        print("[CANH Bao] Khong thay ffmpeg trong PATH - mo hinh / tach video se that bai. "
              "Caai ffmpeg roi chay lai.")

    port = find_free_port(port)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = "http://127.0.0.1:%d/" % port

    print("=" * 62)
    print("  Video Serializer - Web GUI")
    print("  URL:            %s" % url)
    print("  Upload work dir: %s" % WORK_DIR)
    print("  Clips go to the newest folder under:")
    print("                  %s" % DEFAULT_OUTPUT_ROOT)
    print("  Stop: press Ctrl+C in this window (or close it).")
    print("=" * 62)

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[BYE] Web server stopped.")
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
