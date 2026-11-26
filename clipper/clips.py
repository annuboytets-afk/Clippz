"""Cutting clips out of recorded chunks with ffmpeg."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .recorder import Segment


def _run(cmd: list[str]):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr[-1500:]}")


def cut_clip(segments: list[Segment], start: float, end: float, out: Path, vertical: bool = False) -> Path:
    """Cut [start, end) (seconds since recording start) spanning any number of chunks.

    Chunks keep continuous MPEG-TS timestamps, so they can be byte-concatenated.
    """
    segs = [s for s in segments if s.end > start and s.start < end]
    if not segs:
        raise ValueError(f"no recorded segments cover {start:.0f}-{end:.0f}s")
    start = max(start, segs[0].start)
    end = min(end, segs[-1].end)
    src = "concat:" + "|".join(str(s.path) for s in segs)
    vf = ["crop=ih*9/16:ih,scale=1080:1920"] if vertical else []
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
          "-ss", f"{start - segs[0].start:.3f}", "-i", src, "-t", f"{end - start:.3f}",
          "-map", "0:v:0?", "-map", "0:a:0?", *(["-vf", vf[0]] if vf else []),
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)])
    return out


def make_preview(src: Path, out: Path, duration: float, limit_mb: float) -> Path:
    """Low-res copy that fits under Discord's upload limit."""
    total_kbps = limit_mb * 8192 * 0.9 / max(duration, 1)
    v_kbps = max(150, int(total_kbps - 96))
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
          "-vf", "scale=-2:'min(480,ih)'", "-c:v", "libx264", "-preset", "veryfast",
          "-b:v", f"{v_kbps}k", "-maxrate", f"{v_kbps}k", "-bufsize", f"{2 * v_kbps}k",
          "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(out)])
    return out
