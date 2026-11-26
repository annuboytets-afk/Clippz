"""Cutting clips out of recorded chunks with ffmpeg, plus vertical (9:16) layouts that keep faces visible.

Vertical layouts (clips.vertical_layout, or per streamer):
  auto   - look for the streamer's face in the clip and pick one of the layouts below
  split  - facecam on top, gameplay underneath (for gaming streams with a facecam in a corner)
  face   - 9:16 crop that follows where the face is (for IRL / just-chatting streams)
  blur   - the whole 16:9 picture, centred, on a blurred background (nothing is cut off)
  center - plain centre crop (the old behaviour)
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import numpy as np

from .recorder import Segment

log = logging.getLogger(__name__)

LAYOUTS = ("auto", "split", "face", "blur", "center")
OUT_W, OUT_H = 1080, 1920
TOP_H = 640  # split layout: height of the facecam panel


def _run(cmd: list[str]):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr[-1500:]}")


def _even(v: float) -> int:
    return max(2, int(v) // 2 * 2)


def _video_size(path: Path) -> tuple[int, int]:
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height", "-of", "json", str(path)], capture_output=True, text=True)
    s = json.loads(p.stdout or "{}").get("streams") or [{}]
    return int(s[0].get("width", 1920)), int(s[0].get("height", 1080))


def _sample_frames(src: str, offset: float, dur: float, w: int, h: int, n: int = 12) -> list[np.ndarray]:
    """n greyscale frames spread over the clip, scaled to 640 px wide."""
    sw, sh = 640, _even(h * 640 / w)
    p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{offset:.3f}", "-i", src,
                        "-t", f"{dur:.3f}", "-vf", f"fps={n / max(dur, 1):.4f},scale={sw}:{sh},format=gray",
                        "-f", "rawvideo", "-"], capture_output=True)
    count = len(p.stdout) // (sw * sh)
    return list(np.frombuffer(p.stdout[: count * sw * sh], np.uint8).reshape(count, sh, sw))


def detect_faces(frames: list[np.ndarray]) -> list[tuple[float, float, float, float]]:
    """Largest face per frame as (x, y, w, h) fractions of the frame. Empty if OpenCV is missing."""
    try:
        import cv2
    except ImportError:
        log.warning("opencv-python-headless not installed; using the blur layout")
        return []
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    found = []
    for f in frames:
        faces = cascade.detectMultiScale(f, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))
        if len(faces):
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            H, W = f.shape
            found.append((x / W, y / H, w / W, h / H))
    return found


def choose_layout(requested: str, faces: list, n_frames: int) -> tuple[str, tuple | None]:
    """Decide the layout and the (median) face box to build it around."""
    if requested in ("blur", "center"):
        return requested, None
    if n_frames == 0 or len(faces) < max(2, 0.3 * n_frames):
        return "blur", None  # no reliable face: show everything rather than guess
    x, y, w, h = np.median(np.array(faces), axis=0)
    if requested == "auto":
        cx = x + w / 2
        big_and_central = h >= 0.15 and 0.3 <= cx <= 0.7
        requested = "face" if big_and_central else "split"
    return requested, (x, y, w, h)


def vertical_filter(layout: str, W: int, H: int, box=None, exact_cam: bool = False) -> str:
    if layout == "center":
        cw = min(W, _even(H * 9 / 16))
        return f"crop={cw}:{H}:{(W - cw) // 2}:0,scale={OUT_W}:{OUT_H},setsar=1"
    if layout == "blur" or box is None:
        return (f"split[a][b];[a]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
                f"crop={OUT_W}:{OUT_H},boxblur=20:5[bg];[b]scale={OUT_W}:-2[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1")
    x, y, w, h = box
    fcx, fcy = (x + w / 2) * W, (y + h / 2) * H
    if layout == "face":
        cw = min(W, _even(H * 9 / 16))
        left = _even(min(max(0, fcx - cw / 2), W - cw))
        return f"crop={cw}:{H}:{left}:0,scale={OUT_W}:{OUT_H},setsar=1"
    # split: facecam panel on top, centre of the gameplay underneath
    top_aspect = OUT_W / TOP_H
    if exact_cam:
        cw, ch, cx, cy = _even(w * W), _even(h * H), _even(x * W), _even(y * H)
        cam = (f"crop={cw}:{ch}:{cx}:{cy},scale={OUT_W}:{TOP_H}:force_original_aspect_ratio=increase,"
               f"crop={OUT_W}:{TOP_H}")
    else:
        ch = min(H, 2.8 * h * H)  # a facecam is roughly 3x the height of the face in it
        cw = ch * top_aspect
        if cw > W:
            cw, ch = W, W / top_aspect
        cw, ch = _even(cw), _even(ch)
        cx = _even(min(max(0, fcx - cw / 2), W - cw))
        cy = _even(min(max(0, fcy - ch / 2), H - ch))
        cam = f"crop={cw}:{ch}:{cx}:{cy},scale={OUT_W}:{TOP_H}"
    gw = min(W, _even(H * OUT_W / (OUT_H - TOP_H)))
    return (f"split[a][b];[a]{cam}[top];[b]crop={gw}:{H}:{(W - gw) // 2}:0,"
            f"scale={OUT_W}:{OUT_H - TOP_H}[bot];[top][bot]vstack,setsar=1")


def cut_clip(segments: list[Segment], start: float, end: float, out: Path, vertical: bool = False,
             layout: str = "auto", facecam: list[float] | None = None) -> tuple[Path, str]:
    """Cut [start, end) (seconds since recording start) spanning any number of chunks.

    Chunks keep continuous MPEG-TS timestamps, so they can be byte-concatenated.
    Returns the output path and the layout used ("landscape" when not vertical).
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
        if facecam and layout in ("auto", "split"):
            used, box, exact = "split", tuple(facecam), True
        else:
            frames = _sample_frames(src, offset, dur, W, H) if layout in ("auto", "split", "face") else []
            used, box = choose_layout(layout, detect_faces(frames) if frames else [], len(frames))
            exact = False
        vf = ["-vf", vertical_filter(used, W, H, box, exact)]
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
          "-ss", f"{offset:.3f}", "-i", src, "-t", f"{dur:.3f}",
          "-map", "0:v:0?", "-map", "0:a:0?", *vf,
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)])
    return out, used


def make_preview(src: Path, out: Path, duration: float, limit_mb: float) -> Path:
    """Low-res copy that fits under Discord's upload limit."""
    total_kbps = limit_mb * 8192 * 0.9 / max(duration, 1)
    v_kbps = max(150, int(total_kbps - 96))
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
          "-vf", "scale=-2:'min(480,ih)'", "-c:v", "libx264", "-preset", "veryfast",
          "-b:v", f"{v_kbps}k", "-maxrate", f"{v_kbps}k", "-bufsize", f"{2 * v_kbps}k",
          "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(out)])
    return out
