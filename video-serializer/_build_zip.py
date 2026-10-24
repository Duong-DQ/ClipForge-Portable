"""One-shot builder (disposable): package the whole runnable video-serializer
project into video-serializer.zip, rooted at 'video-serializer/' so it extracts
to a clean, ready-to-run folder. Excludes only generated artifacts + cache.
"""
import os
import zipfile

BASE = r"C:\Users\tien1_le\Desktop\opencode_project\video-serializer"
OUT = os.path.join(BASE, "video-serializer.zip")
ROOTNAME = "video-serializer"

# Generated / non-project files to keep OUT of the zip.
EXCLUDE_FILES = {
    "TOAN_BO_CODE.txt",
    "video_serializer_all_in_one.py",
    "video-serializer.zip",
}


def excluded(rel):
    parts = rel.split(os.sep)
    if "__pycache__" in parts:
        return True
    if parts[-1].endswith(".pyc"):
        return True
    if len(parts) == 1 and parts[-1] in EXCLUDE_FILES:
        return True
    return False


count = 0
size = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for dirpath, dirnames, filenames in os.walk(BASE):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, BASE)
            if excluded(rel):
                continue
            arcname = os.path.join(ROOTNAME, rel).replace(os.sep, "/")
            zf.write(full, arcname)
            count += 1
            size += os.path.getsize(full)

print("WROTE:", OUT)
print("FILES_ZIPPED:", count)
print("SOURCE_KB:", round(size / 1024.0, 1))
print("ZIP_KB:", round(os.path.getsize(OUT) / 1024.0, 1))

# --- verify ---
with zipfile.ZipFile(OUT) as zf:
    names = [n.replace("\\", "/") for n in zf.namelist()]

    def has(p):
        return any(n == p or n.endswith("/" + p) for n in names)

    print("--- verify ---")
    print("bat webgui:", has("video-serializer/chay_webgui.bat"))
    print("bat cli:", has("video-serializer/chay_video_serializer.bat"))
    print("config.json:", has("video-serializer/config.json"))
    print("hooks.json:", has("video-serializer/hooks.json"))
    print("webgui.py:", has("video-serializer/webgui.py"))
    print("webgui.html:", has("video-serializer/webgui.html"))
    print("pkg segmentation.py:", has("video-serializer/video_serializer/segmentation.py"))
    print("pkg llm.py:", has("video-serializer/video_serializer/llm.py"))
    print("examples sample.srt:", has("video-serializer/examples/sample.srt"))
    print("EXCLUDED TOAN_BO_CODE.txt:", not has("video-serializer/TOAN_BO_CODE.txt"))
    print("EXCLUDED all_in_one.py:", not has("video-serializer/video_serializer_all_in_one.py"))
    print("TOTAL_ENTRIES:", len(names))
