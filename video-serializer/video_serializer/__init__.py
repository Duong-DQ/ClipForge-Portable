"""video_serializer - turn one long video + its transcript into a clip series.

Pipeline: parse transcript (SRT/VTT) -> segment into topics -> enrich with
titles/summaries -> cut clips with ffmpeg -> write metadata (JSON/CSV).
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
