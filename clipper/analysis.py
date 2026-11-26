"""Hype detection from per-second audio loudness and chat activity.

Two ways to pick clips (clips.mode):

* "spikes": each signal becomes a robust z-score against a trailing baseline (median / MAD
  over the previous ~10 minutes). score = chat_weight * z_chat + audio_weight * z_audio
  (audio only when no chat is available). Seconds above the threshold form one event;
  its peak second is the hype moment. Quiet stretches produce no clips.
* "best_per_chunk": every finished chunk yields exactly one clip, its most exciting moment,
  scored relative to that chunk itself, with no threshold.
* "both": spike clips, plus a best-of-chunk clip for any chunk that had no spike.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ClipConfig

SAMPLE_RATE = 8000
MODES = ("spikes", "best_per_chunk", "both")


def audio_loudness(path: Path) -> np.ndarray:
    """Per-second RMS loudness in dBFS for a media file (blocking; run in a thread)."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
        capture_output=True, check=False)
    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)
    n = len(pcm) // SAMPLE_RATE
    if n == 0:
        return np.zeros(0)
    rms = np.sqrt(np.mean(pcm[: n * SAMPLE_RATE].reshape(n, SAMPLE_RATE) ** 2, axis=1))
    return np.maximum(20 * np.log10(rms / 32768 + 1e-9), -90.0)


@dataclass
class HypeMoment:
    t: float  # seconds since recording start
    score: float
    chat_z: float
    audio_z: float
    kind: str = "spike"  # "spike" or "best"


def _robust_z(x: np.ndarray, t: int, lo: int, hi: int, floor: float) -> float:
    base = x[lo:hi]
    base = base[~np.isnan(base)]
    if len(base) < 30 or np.isnan(x[t]):
        return 0.0
    med = np.median(base)
    mad = np.median(np.abs(base - med)) * 1.4826
    return float((x[t] - med) / max(mad, floor))


def _local_z(x: np.ndarray, floor: float) -> np.ndarray:
    """z-score of every point relative to the window itself (NaN -> 0)."""
    valid = x[~np.isnan(x)]
    if len(valid) == 0:
        return np.zeros(len(x))
    med = np.median(valid)
    mad = np.median(np.abs(valid - med)) * 1.4826
    return np.nan_to_num((x - med) / max(mad, floor))


class HypeDetector:
    def __init__(self, cfg: ClipConfig):
        self.cfg = cfg
        self.audio = np.zeros(0)
        self.cursor = 0       # spikes: first second not yet analysed
        self.best_cursor = 0  # best_per_chunk: start of the next chunk window
        self.clip_times: list[float] = []

    @property
    def end(self) -> int:
        return len(self.audio)

    def add_audio(self, start: float, loudness: np.ndarray):
        s = int(round(start))
        e = s + len(loudness)
        if e > len(self.audio):
            self.audio = np.concatenate([self.audio, np.full(e - len(self.audio), np.nan)])
        self.audio[s:e] = loudness

    def _signals(self, chat: np.ndarray | None):
        n = self.end
        # Edge-padded 3 s average (zero padding would read as "loud" in dBFS at the edges).
        audio = np.convolve(np.pad(np.nan_to_num(self.audio, nan=-90.0), 1, mode="edge"),
                            np.ones(3) / 3, mode="valid")
        audio[np.isnan(self.audio)] = np.nan
        chat_rate = None
        if chat is not None:
            w = max(1, int(self.cfg.chat_window_seconds))
            chat_rate = np.convolve(chat[:n], np.ones(w))[:n]  # trailing sum over the window
        return audio, chat_rate

    def _combine(self, zc: np.ndarray, za: np.ndarray, has_chat: bool) -> np.ndarray:
        if not has_chat:
            return za
        c = self.cfg
        return (c.chat_weight * zc + c.audio_weight * za) / (c.chat_weight + c.audio_weight)

    def _free(self, t: float) -> bool:
        """True if a clip at t would not overlap / crowd an existing clip."""
        return all(abs(t - ct) >= self.cfg.cooldown_seconds for ct in self.clip_times)

    def scores(self, chat: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = self.cfg
        n = self.end
        audio, chat_rate = self._signals(chat)
        za, zc = np.zeros(n), np.zeros(n)
        for t in range(n):
            hi = t - 30
            lo = max(0, t - int(c.baseline_window_seconds))
            if t < c.min_baseline_seconds or hi - lo < 60:
                continue
            za[t] = _robust_z(audio, t, lo, hi, floor=1.5)
            if chat_rate is not None and chat_rate[t] >= c.min_chat_messages:
                zc[t] = _robust_z(chat_rate, t, lo, hi, floor=2.0)
        return self._combine(zc, za, chat is not None), zc, za

    def detect(self, chat: np.ndarray | None, final: bool = False) -> list[HypeMoment]:
        """Spike mode. Needs post_roll seconds of audio after a moment unless final."""
        c = self.cfg
        upto = self.end if final else self.end - int(c.post_roll_seconds)
        if upto <= self.cursor:
            return []
        score, zc, za = self.scores(chat)
        hot = [t for t in range(self.cursor, upto) if score[t] >= c.threshold]
        runs: list[list[int]] = []
        for t in hot:
            if runs and t - runs[-1][-1] <= 5:
                runs[-1].append(t)
            else:
                runs.append([t])
        next_cursor = upto
        if runs and not final and runs[-1][-1] >= upto - 1:
            next_cursor = runs.pop()[0]  # event still rising; finish it next time
        out = []
        for run in runs:
            t = max(run, key=lambda i: score[i])
            if self._free(t):
                out.append(HypeMoment(float(t), float(score[t]), float(zc[t]), float(za[t]), "spike"))
                self.clip_times.append(t)
        self.cursor = next_cursor
        return out

    def best(self, chat: np.ndarray | None, final: bool = False,
             skip_if_clipped: bool = False) -> HypeMoment | None:
        """Best-per-chunk mode: the single most exciting second of the newest chunk window.

        Windows tile the recording: [previous window end, chunk end - post_roll), so a moment
        in the last post_roll seconds of a chunk is considered together with the next chunk.
        """
        c = self.cfg
        lo = self.best_cursor
        hi = self.end if final else self.end - int(c.post_roll_seconds)
        if hi - lo < 30:
            if final:
                self.best_cursor = max(lo, hi)
            return None
        self.best_cursor = hi
        if skip_if_clipped and any(lo <= ct < hi for ct in self.clip_times):
            return None
        audio, chat_rate = self._signals(chat)
        za = _local_z(audio[lo:hi], floor=1.5)
        zc = _local_z(chat_rate[lo:hi], floor=2.0) if chat_rate is not None else np.zeros(hi - lo)
        score = self._combine(zc, za, chat_rate is not None)
        score = np.convolve(np.pad(score, 2, mode="edge"), np.ones(5) / 5, mode="valid")  # favour sustained excitement
        score[np.isnan(audio[lo:hi])] = -np.inf
        score[: max(0, int(c.pre_roll_seconds) - lo)] = -np.inf  # need a full pre-roll before the moment
        for i in np.argsort(score)[::-1]:
            if not np.isfinite(score[i]):
                break
            t = lo + int(i)
            if not self._free(t):
                continue
            if c.best_min_score is not None and score[i] < c.best_min_score:
                return None
            self.clip_times.append(t)
            return HypeMoment(float(t), float(score[i]), float(zc[i]), float(za[i]), "best")
        return None

    def process(self, chat: np.ndarray | None, final: bool = False) -> list[HypeMoment]:
        """Run whichever selection the configured mode asks for."""
        mode = self.cfg.mode
        out: list[HypeMoment] = []
        if mode in ("spikes", "both"):
            out += self.detect(chat, final)
        if mode in ("best_per_chunk", "both"):
            m = self.best(chat, final, skip_if_clipped=(mode == "both"))
            if m:
                out.append(m)
        return out
