"""Cutting clips out of recorded chunks with ffmpeg, and framing them for TikTok / Shorts.

Vertical clips are always a 1080x1920 (9:16) video. The picture inside it is one of:
  9:16  - crop that fills the whole phone screen
  4:5   - 1080x1350 crop in the middle, background above and below
  16:9  - the full stream picture (1080x608) in the middle, background above and below
With format "auto" the clip is analysed first: where the motion is decides how much of
the width has to stay visible, and where the crop goes.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import numpy as np

from .recorder import Segment

log = logging.getLogger(__name__)

FORMATS = ("auto", "9:16", "4:5", "16:9")
BACKGROUNDS = ("blur", "black")
OUT_W, OUT_H = 1080, 1920
# a crop is only used when it keeps at least this share of the clip's motion
KEEP = {"9:16": 0.85, "4:5": 0.80}


def _run(cmd: list[str]):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr[-1500:]}")


def _even(v: float) -> int:
    return max(2, int(round(v)) // 2 * 2)


def _video_size(path: Path) -> tuple[int, int]:
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height", "-of", "json", str(path)], capture_output=True, text=True)
    s = (json.loads(p.stdout or "{}").get("streams") or [{}])[0]
    return int(s.get("width", 1920)), int(s.get("height", 1080))


def motion_profile(src: str, offset: float, dur: float, n: int = 60) -> np.ndarray:
    """How much movement there is in each column of the picture over the clip (320 columns)."""
    sw, sh = 320, 180
    p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{offset:.3f}", "-i", src,
                        "-t", f"{dur:.3f}", "-vf", f"fps={n / max(dur, 1):.4f},scale={sw}:{sh},format=gray",
                        "-f", "rawvideo", "-"], capture_output=True)
    count = len(p.stdout) // (sw * sh)
    if count < 3:
        return np.zeros(sw)
    frames = np.frombuffer(p.stdout[: count * sw * sh], np.uint8).reshape(count, sh, sw).astype(np.int16)
    diff = np.abs(np.diff(frames, axis=0))
    diff[diff < 12] = 0  # ignore compression noise
    return diff.sum(axis=(0, 1)).astype(float)


def _best_window(profile: np.ndarray, frac: float) -> tuple[float, float]:
    """(share of motion kept, window centre as 0-1) for the best window of the given width fraction."""
    n = len(profile)
    w = max(1, min(n, int(round(frac * n))))
    total = profile.sum()
    if total <= 0:
        return 1.0, 0.5
    sums = np.convolve(profile, np.ones(w), mode="valid")
    i = int(np.argmax(sums))
    # Several positions can keep the same amount of motion. Centre the crop on the middle of
    # the area where the motion is (ignoring 1% stray specks at either side), so the action is
    # framed evenly instead of being pushed against an edge.
    part = profile[i:i + w]
    if part.sum() > 0:
        cum = np.cumsum(part) / part.sum()
        a, b = int(np.searchsorted(cum, 0.01)), int(np.searchsorted(cum, 0.99))
        mid = i + (a + b + 1) / 2
    else:
        mid = i + w / 2
    start = min(max(0.0, mid - w / 2), n - w)
    kept = float(profile[int(start):int(start) + w].sum() / total)
    return kept, (start + w / 2) / n


def choose_format(profile: np.ndarray, W: int, H: int, requested: str = "auto") -> tuple[str, float]:
    """Pick 9:16 / 4:5 / 16:9 and the horizontal centre (0-1) of the crop."""
    fracs = {"9:16": min(1.0, H * 9 / 16 / W), "4:5": min(1.0, H * 4 / 5 / W)}
    total = profile.sum()
    if requested in fracs:
        return requested, _best_window(profile, fracs[requested])[1] if total > 0 else 0.5
    if requested == "16:9":
        return "16:9", 0.5
    if total == 0:  # nothing moves at all: a centred full-screen crop is safest
        return "9:16", 0.5
    for fmt in ("9:16", "4:5"):
        kept, centre = _best_window(profile, fracs[fmt])
        if kept >= KEEP[fmt]:
            return fmt, centre
    return "16:9", 0.5


def vertical_filter(fmt: str, W: int, H: int, centre: float, background: str = "blur") -> str:
    if fmt == "9:16":
        cw = min(W, _even(H * 9 / 16))
        x = _even(min(max(0, centre * W - cw / 2), W - cw))
        return f"crop={cw}:{H}:{x}:0,scale={OUT_W}:{OUT_H},setsar=1"
    if fmt == "4:5":
        cw = min(W, _even(H * 4 / 5))
        x = _even(min(max(0, centre * W - cw / 2), W - cw))
        fg = f"crop={cw}:{H}:{x}:0,scale={OUT_W}:-2"
    else:  # 16:9: whole picture
        fg = f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease"
    if background == "black":
        return f"{fg},pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
    return (f"split[a][b];[a]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
            f"crop={OUT_W}:{OUT_H},boxblur=25:5,eq=brightness=-0.08[bg];[b]{fg}[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1")


def cut_clip(segments: list[Segment], start: float, end: float, out: Path, vertical: bool = False,
             fmt: str = "auto", background: str = "blur") -> tuple[Path, str]:
    """Cut [start, end) (seconds since recording start) spanning any number of chunks.

    Chunks keep continuous MPEG-TS timestamps, so they can be byte-concatenated.
    Returns the output path and the framing used ("landscape" when not vertical).
    """
    segs = [s for s in segments if s.end > start and s.start < end]
    if not segs:
        raise ValueError(f"no recorded segments cover {start:.0f}-{end:.0f}s")
    start = max(start, segs[0].start)
    end = min(end, segs[-1].end)
    src = "concat:" + "|".join(str(s.path) for s in segs)
    offset, dur = start - segs[0].start, end - start
    vf: list[str] = []
    used = "landscape"
    if vertical:
        W, H = _video_size(segs[0].path)
        profile = motion_profile(src, offset, dur) if fmt in ("auto", "9:16", "4:5") else np.zeros(1)
        used, centre = choose_format(profile, W, H, fmt)
        vf = ["-vf", vertical_filter(used, W, H, centre, background)]
        log.info("Framing %s as %s (crop centre %.0f%%)", out.name, used, centre * 100)
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
          "-ss", f"{offset:.3f}", "-i", src, "-t", f"{dur:.3f}",
          "-map", "0:v:0?", "-map", "0:a:0?", *vf,
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)])
    return out, used


def make_preview(src: Path, out: Path, duration: float, limit_mb: float) -> Path:
    """Smaller copy for the Discord approval message that fits under the upload limit.

    Only the preview is reduced; the clip that gets posted stays full quality. To spend
    Discord's few MB on sharpness: 540 px on the short side (540x960 for vertical clips),
    30 fps instead of 60, a more efficient encoder preset and lighter audio.
    """
    w, h = _video_size(src)
    short = min(540, w, h)
    scale = f"scale=-2:{short}" if w >= h else f"scale={short}:-2"
    audio_kbps = 64
    total_kbps = limit_mb * 8192 * 0.92 / max(duration, 1)
    v_kbps = max(150, int(total_kbps - audio_kbps))
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
          "-vf", f"{scale},fps=30", "-c:v", "libx264", "-preset", "medium",
          "-b:v", f"{v_kbps}k", "-maxrate", f"{int(v_kbps * 1.2)}k", "-bufsize", f"{2 * v_kbps}k",
          "-c:a", "aac", "-b:a", f"{audio_kbps}k", "-movflags", "+faststart", str(out)])
    if out.stat().st_size > limit_mb * 1024 * 1024:  # rare: very busy footage overshot the budget
        _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
              "-vf", f"{scale},fps=30", "-c:v", "libx264", "-preset", "medium",
              "-b:v", f"{int(v_kbps * 0.75)}k", "-maxrate", f"{int(v_kbps * 0.8)}k",
              "-bufsize", f"{v_kbps}k", "-c:a", "aac", "-b:a", f"{audio_kbps}k",
              "-movflags", "+faststart", str(out)])
    return out
