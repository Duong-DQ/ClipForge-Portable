#!/usr/bin/env python3
"""ClipForge portable bootstrap / launcher.

This is the ONE file you need to run on a fresh machine. It is stdlib-only at
the top level: everything else (FastAPI, opencv, whisper, ...) is installed
into a private virtual environment (``.venv``) created next to this file.

What it does, in order:

  [1/5] Locate the project + create the virtual environment (``.venv``)
  [2/5] Upgrade pip and install ``backend/requirements.txt`` into the venv
  [3/5] Download the AI models that are NOT shipped in the zip (Whisper small)
  [4/5] (nothing to build - marker for a future step / keeps banners uniform)
  [5/5] Print the local + LAN URLs and start the FastAPI/uvicorn server

Usage:
    python setup_and_run.py                # bootstrap (if needed) then run
    python setup_and_run.py --skip-models  # do not download models
    python setup_and_run.py --reinstall    # delete .venv first, reinstall all
    python setup_and_run.py --no-run       # bootstrap + models only, do not run

Env:
    CLIPFORGE_SKIP_MODELS=1  # same as --skip-models
"""

from __future__ import annotations

import argparse
import os
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
import venv
from pathlib import Path

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
BACKEND = ROOT / "backend"
MODELS = BACKEND / "models"
REQUIREMENTS = BACKEND / "requirements.txt"

# ---------------------------------------------------------------------------
# Model definitions (official URLs only)
# ---------------------------------------------------------------------------

# YuNet face-detection ONNX (OpenCV Zoo). ~232589 bytes.
# NOTE: the canonical "media.githubusercontent.com/media/..." URL serves the
# real binary; the plain raw.githubusercontent.com URL serves a ~130 byte git
# LFS pointer for this repo, which is why we use the /media/ host and still
# sanity-check the size.
YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_PATH = MODELS / "face_detection_yunet_2023mar.onnx"
YUNET_MIN_BYTES = 10000  # anything smaller than this is an LFS pointer / error

# Whisper "small" checkpoint from the official OpenAI CDN (~461 MiB).
WHISPER_URL = (
    "https://openaipublic.azureedge.net/main/whisper/models/"
    "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt"
)
WHISPER_PATH = MODELS / "small.pt"
WHISPER_MIN_BYTES = 100 * 1024 * 1024  # ~461MB real; <100MB == broken

MODELS_TO_FETCH = [
    # (label, url, destination path, minimum plausible size in bytes)
    ("YuNet face-detection ONNX", YUNET_URL, YUNET_PATH, YUNET_MIN_BYTES),
    ("Whisper 'small' checkpoint", WHISPER_URL, WHISPER_PATH, WHISPER_MIN_BYTES),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def banner(step: str, text: str) -> None:
    print()
    print("=" * 68)
    print(f"[{step}] {text}")
    print("=" * 68)


def venv_python() -> Path:
    """Return the python executable inside the venv for this platform."""
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def human_bytes(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def make_ssl_context() -> ssl.SSLContext:
    """Unverified SSL context (corporate proxies terminate TLS with a private CA)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


# ---------------------------------------------------------------------------
# Step 1 - virtual environment
# ---------------------------------------------------------------------------


def ensure_venv(reinstall: bool) -> Path:
    banner("1/5", "Virtual environment")
    if reinstall and VENV.exists():
        import shutil

        print(f"  --reinstall: removing existing venv at {VENV}")
        shutil.rmtree(VENV, ignore_errors=True)

    py = venv_python()
    if py.exists():
        print(f"  Reusing existing venv: {VENV}")
        return py

    print(f"  Creating venv at {VENV} ...")
    venv.create(VENV, with_pip=True)
    if not py.exists():
        raise RuntimeError(
            f"venv was created but {py} is missing - aborting. "
            "Check that your Python install (python -m venv) works."
        )
    print("  venv created.")
    return py


# ---------------------------------------------------------------------------
# Step 2 - dependencies
# ---------------------------------------------------------------------------


def install_deps(py: Path) -> None:
    banner("2/5", "Installing Python dependencies (this can take a while)")
    if not REQUIREMENTS.exists():
        raise RuntimeError(f"requirements.txt not found at {REQUIREMENTS}")

    run([str(py), "-m", "pip", "install", "--upgrade", "pip"])

    print()
    print("  Installing backend/requirements.txt ...")
    print("  (openai-whisper pulls PyTorch, which is ~2 GB on first install.)")
    run([str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    print("  Dependencies ready.")


# ---------------------------------------------------------------------------
# Step 3 - models
# ---------------------------------------------------------------------------


def _download_urllib(url: str, dest: Path) -> None:
    """Download with urllib in 1 MiB chunks, printing a progress percentage."""
    ctx = make_ssl_context()
    req = urllib.request.Request(url, headers={"User-Agent": "ClipForge-setup/1.0"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:  # noqa: S310
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        last_pct = -1
        with open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = int(done * 100 / total)
                    if pct != last_pct:
                        sys.stdout.write(
                            f"\r    {pct:3d}%  ({human_bytes(done)} / {human_bytes(total)})"
                        )
                        sys.stdout.flush()
                        last_pct = pct
                else:
                    sys.stdout.write(f"\r    {human_bytes(done)} downloaded")
                    sys.stdout.flush()
    sys.stdout.write("\n")
    tmp.replace(dest)


def _download_curl(url: str, dest: Path) -> bool:
    """Fallback downloader using curl -k -L -o. Returns True on success."""
    import shutil

    curl = shutil.which("curl")
    if not curl:
        return False
    print("    urllib failed - retrying with curl -k -L ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    rc = subprocess.call([curl, "-k", "-L", "--fail", "-o", str(tmp), url])
    if rc == 0 and tmp.exists():
        tmp.replace(dest)
        return True
    return False


def fetch_model(label: str, url: str, dest: Path, min_bytes: int) -> bool:
    """Download one model if missing. Returns True if the file is now present+valid."""
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"  [skip] {label} already present ({human_bytes(dest.stat().st_size)})")
        return True

    print(f"  [get ] {label}")
    print(f"         -> {dest}")
    print(f"         from {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    ok = False
    try:
        _download_urllib(url, dest)
        ok = True
    except Exception as exc:  # noqa: BLE001
        print(f"         urllib error: {exc}")
        try:
            ok = _download_curl(url, dest)
        except Exception as exc2:  # noqa: BLE001
            print(f"         curl error: {exc2}")

    if ok and dest.exists():
        size = dest.stat().st_size
        if size >= min_bytes:
            print(f"         done ({human_bytes(size)})")
            return True
        print(
            f"         WARNING: downloaded file is only {size} bytes "
            f"(expected >= {min_bytes}). It is probably an HTML error page or a "
            "git-LFS pointer, not the real model."
        )
        if "githubusercontent" in url:
            print("         (The /media/ URL is required for git-LFS files.)")
        return False

    print("         FAILED to download.")
    return False


def ensure_models() -> None:
    banner("3/5", "AI models")
    if os.environ.get("CLIPFORGE_SKIP_MODELS", "").strip() in ("1", "true", "yes"):
        print("  CLIPFORGE_SKIP_MODELS is set - skipping downloads.")
        return

    all_ok = True
    for label, url, dest, min_bytes in MODELS_TO_FETCH:
        if not fetch_model(label, url, dest, min_bytes):
            all_ok = False
            print()
            print(f"  !! Could not obtain {label}.")
            print(f"     Manual fix: download this file yourself")
            print(f"       {url}")
            print(f"     and place it at:")
            print(f"       {dest}")

    if all_ok:
        print("  All models present.")
    else:
        print()
        print("  NOTE: the server will still start, but transcription / face")
        print("  auto-zoom will fail until the missing model is in place.")


# ---------------------------------------------------------------------------
# Step 4 - placeholder (kept so the [n/5] banners stay uniform)
# ---------------------------------------------------------------------------


def step_placeholder() -> None:
    banner("4/5", "Prepare runtime directories")
    for sub in ("videos", "clips", "thumbnails"):
        (BACKEND / sub).mkdir(parents=True, exist_ok=True)
    print(f"  Ensured {BACKEND}/videos, clips, thumbnails exist.")


# ---------------------------------------------------------------------------
# Step 5 - run
# ---------------------------------------------------------------------------


def detect_lan_ip() -> str:
    """Best-effort LAN IP discovery (no packets are actually sent)."""
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:  # noqa: BLE001
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:  # noqa: BLE001
            return "127.0.0.1"


def run_server(py: Path) -> int:
    banner("5/5", "Starting ClipForge server")
    lan = detect_lan_ip()
    print()
    print("  Server will be reachable at:")
    print("    Local : http://localhost:8000")
    print(f"    LAN   : http://{lan}:8000   (share on the same Wi-Fi/LAN)")
    print()
    print("  Open the Local URL in your browser. Press Ctrl+C here to stop.")
    print()

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"  # Windows console encoding safety
    try:
        return subprocess.call([str(py), "main.py"], cwd=str(BACKEND), env=env)
    except KeyboardInterrupt:
        print("\n  Stopped by user.")
        return 0


# ---------------------------------------------------------------------------
# subprocess helper
# ---------------------------------------------------------------------------


def run(cmd: list[str]) -> None:
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ClipForge portable bootstrap + launcher.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Do not download model files (same as CLIPFORGE_SKIP_MODELS=1).",
    )
    parser.add_argument(
        "--reinstall",
        action="store_true",
        help="Delete the existing .venv and recreate it from scratch.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Bootstrap + download models only; do not start the server.",
    )
    args = parser.parse_args(argv)

    if args.skip_models:
        os.environ["CLIPFORGE_SKIP_MODELS"] = "1"

    print("ClipForge portable launcher")
    print(f"  ROOT    = {ROOT}")
    print(f"  Python  = {sys.version.split()[0]}  ({sys.executable})")

    try:
        py = ensure_venv(args.reinstall)
        install_deps(py)
        if not os.environ.get("CLIPFORGE_SKIP_MODELS"):
            ensure_models()
        else:
            banner("3/5", "AI models")
            print("  Skipped (--skip-models).")
        step_placeholder()

        if args.no_run:
            banner("done", "Bootstrap complete (--no-run)")
            print("  Run again without --no-run to start the server.")
            return 0

        return run_server(py)
    except KeyboardInterrupt:
        print("\n  Interrupted - exiting.")
        return 130
    except Exception as exc:  # noqa: BLE001
        print()
        print(f"  ERROR: {exc}")
        print("  See the messages above; fix the issue and run again.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
