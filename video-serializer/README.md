# video-serializer

Turn **one long video + its existing transcript** (SRT/VTT) into a **series** of short,
topic-based episode clips — plus a metadata file describing every episode.

The tool reads your already-timed transcript, automatically groups it into coherent
**topics**, cuts the video into one clip per topic with ffmpeg, and writes a metadata
file (JSON or CSV) with titles, timestamps, durations and summaries.

> No speech recognition is required — you provide the transcript (e.g. from YouTube
> auto-captions, a transcription service, or your own subtitles).

---

## How it works

```
        parse transcript        segment into topics        enrich          cut clips         write metadata
video + (SRT/VTT)  ───────────►  ────────────────────────►  ────────────►  ──────────────►  ──────────────────
   SRT/VTT             segments (start,end,text)         title + summary    MP4 per topic     metadata.json / .csv
```

1. **Parse** — read the `.srt`/`.vtt` into timed segments.
2. **Segment** — group segments into topics (heuristic, or semantic if embeddings are installed).
3. **Enrich** — give each topic a title + summary (heuristic, or LLM if enabled).
4. **Cut** — ffmpeg produces one MP4 per topic.
5. **Metadata** — a JSON/CSV file describing the whole series.

---

## Requirements

- **Python 3.8+**
- **ffmpeg** on your `PATH` (any recent build; tested on ffmpeg 9.x)

Optional:
- `sentence-transformers` — enables higher-quality **semantic** topic segmentation.
- Any **OpenAI-compatible** LLM endpoint — for polished titles/summaries (stdlib only, no extra package).

### Install

No install is required — run directly from this directory:

```bash
python -m video_serializer --help
```

Or install it for a `video-serializer` command:

```bash
pip install -e .
video-serializer --help
```

Enable semantic segmentation (optional, downloads a small multilingual model once):

```bash
pip install sentence-transformers
```

---

## Quick start

```bash
python -m video_serializer talk.mp4 talk.srt -o out/
```

This produces:

```
out/
├── episode_01_xin_chao_moi_nguoi.mp4
├── episode_02_truoc_tien_hay_noi_ve.mp4
├── episode_03_giur_chung_ta_chuyen_sang.mp4
├── episode_04_cuoi_cung_toi_se.tom.mp4
└── metadata.json
```

For short test clips, lower the minimum topic duration so you get several episodes:

```bash
python -m video_serializer examples/video.mp4 examples/sample.srt -o out/ --min-duration 5
```

---

## Options

| Flag | Default | Description |
|---|---|---|
| `video` *(positional)* | — | Input video file (`mp4`, `mov`, `mkv`, …) |
| `transcript` *(positional)* | — | Transcript file (`.srt` or `.vtt`) |
| `-o, --output DIR` | `<video_stem>_clips` | Output directory for clips + metadata |
| `--min-duration FLOAT` | `20` | Minimum topic duration (seconds). Shorter topics are merged. |
| `--max-duration FLOAT` | `300` | Maximum topic duration (seconds). Longer topics are force-split. |
| `--gap FLOAT` | `1.5` | A silent gap ≥ this (seconds) suggests a topic boundary. |
| `--similarity FLOAT` | `0.45` | Similarity threshold (0..1). Lower = more, shorter topics. |
| `--mode {accurate,fast}` | `accurate` | `accurate` re-encodes (frame-accurate); `fast` stream-copies (fast, keyframe-aligned). |
| `--embeddings` | auto | Force semantic (embedding) segmentation. |
| `--no-embeddings` | auto | Force heuristic segmentation (no model). |
| `--llm` | off | Use an LLM for titles/summaries (needs endpoint + key). |
| `--llm-base-url URL` | env `LLM_BASE_URL` | OpenAI-compatible base URL, e.g. `http://host:8000/v1` |
| `--llm-api-key KEY` | env `LLM_API_KEY` | Bearer token for the LLM endpoint. |
| `--llm-model MODEL` | env `LLM_MODEL`, else `gpt-4o-mini` | Model identifier to request. |
| `--metadata {json,csv}` | `json` | Metadata file format. |
| `--no-clips` | off | Skip cutting — only segment + write metadata (great for preview/tuning). |
| `-v, --verbose` | off | Verbose logging. |

---

## Topic segmentation

Two interchangeable backends:

- **Heuristic (default, zero-dependency)** — detects boundaries from (a) silent gaps
  between cues and (b) a lightweight keyword/n-gram similarity drop, then enforces
  min/max durations. Works out of the box, fully offline.
- **Semantic (optional)** — if `sentence-transformers` is installed, each cue is
  embedded with a multilingual model and boundaries are detected from similarity to a
  running topic centroid. This follows topic changes more faithfully, especially when
  gaps are small and wording shifts gradually.

The backend is chosen automatically (`--embeddings` to force it, `--no-embeddings` to
force the heuristic). Content is always preserved — no transcript text is dropped or
reordered.

### Tuning the cut points

| Want | Adjust |
|---|---|
| More, shorter episodes | lower `--similarity` (e.g. `0.35`), lower `--min-duration` |
| Fewer, longer episodes | raise `--similarity` (e.g. `0.55`), raise `--min-duration` |
| Respect natural pauses | lower `--gap` (e.g. `0.8`) |
| Cap episode length | `--max-duration` (e.g. `120` for ~2-minute episodes) |

Use `--no-clips` to iterate on boundaries without re-cutting the video each time.

---

## Metadata output

`metadata.json` (or `metadata.csv`) is written into the output directory:

```json
{
  "source_video": "talk.mp4",
  "transcript": "talk.srt",
  "total_episodes": 4,
  "total_duration_seconds": 7200.0,
  "episodes": [
    {
      "index": 1,
      "title": "Xin chào mọi người, hôm nay…",
      "start": 0.0,
      "end": 92.5,
      "duration": 92.5,
      "start_timestamp": "00:00:00.0",
      "end_timestamp": "00:01:32.5",
      "file": "out/episode_01_xin_chao_moi_nguoi.mp4",
      "summary": "Trong phần mở đầu này, tôi sẽ giới thiệu…"
    }
  ]
}
```

CSV columns: `index,title,start,end,duration,start_timestamp,end_timestamp,file,summary`.

---

## LLM titles & summaries (optional)

By default titles/summaries are generated heuristically (first sentence, trimmed).
For cleaner, more descriptive titles you can point the tool at any
OpenAI-compatible endpoint:

```bash
python -m video_serializer talk.mp4 talk.srt -o out/ \
  --llm \
  --llm-base-url http://localhost:8000/v1 \
  --llm-api-key sk-... \
  --llm-model gpt-4o-mini
```

The tool asks the model to return a JSON `{"title": ..., "summary": ...}` per episode.
If the call fails for any reason, it silently falls back to the heuristic for that
episode — the pipeline never crashes because of the LLM.

---

## Troubleshooting

- **`ffmpeg not found`** — install ffmpeg and ensure it's on `PATH`
  (`ffmpeg -version` should work in your terminal).
- **Boundaries feel wrong** — try `--no-clips` and tweak `--similarity` / `--gap` /
  `--min-duration`; or install `sentence-transformers` for semantic segmentation.
- **Clips start slightly early (fast mode)** — expected; `fast` mode snaps to keyframes.
  Use `--mode accurate` for exact cuts.
- **Vietnamese/Chinese titles look plain** — enable `--llm` for natural titles, or rely
  on the heuristic (uses the episode's opening sentence).
- **Empty output** — check the transcript covers the video's time range and that cue
  times are valid.

---

## Project structure

```
video-serializer/
├── pyproject.toml
├── requirements.txt
├── README.md
├── examples/
│   └── sample.srt
└── video_serializer/
    ├── __init__.py
    ├── __main__.py        # python -m video_serializer
    ├── cli.py             # argument parsing + pipeline orchestration
    ├── models.py          # TranscriptSegment, Topic, PipelineConfig
    ├── utils.py           # timestamp formatting, safe filenames
    ├── transcript.py      # SRT/VTT parser
    ├── segmentation.py    # topic detection (heuristic + optional semantic)
    ├── enrich.py          # titles/summaries (heuristic + optional LLM)
    └── cutter.py          # ffmpeg clip cutting
```

## License

MIT
