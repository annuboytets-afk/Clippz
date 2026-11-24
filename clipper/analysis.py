"""Hype detection from per-second audio loudness and chat activity.

Each signal is turned into a robust z-score against a trailing baseline (median / MAD over
the previous ~10 minutes), so "loud" and "busy" are relative to the current stream.
score = chat_weight * z_chat + audio_weight * z_audio  (audio only when no chat is available)
Contiguous seconds above the threshold form one event; its peak second is the hype moment.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ClipConfig

SAMPLE_RATE = 8000


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


def _robust_z(x: np.ndarray, t: int, lo: int, hi: int, floor: float) -> float:
    base = x[lo:hi]
    base = base[~np.isnan(base)]
    if len(base) < 30 or np.isnan(x[t]):
        return 0.0
    med = np.median(base)
    mad = np.median(np.abs(base - med)) * 1.4826
    return float((x[t] - med) / max(mad, floor))


class HypeDetector:
    def __init__(self, cfg: ClipConfig):
        self.cfg = cfg
        self.audio = np.zeros(0)
        self.cursor = 0  # first second not yet analysed
        self.last_peak = -1e9

    @property
    def end(self) -> int:
        return len(self.audio)

    def add_audio(self, start: float, loudness: np.ndarray):
        s = int(round(start))
        e = s + len(loudness)
        if e > len(self.audio):
            self.audio = np.concatenate([self.audio, np.full(e - len(self.audio), np.nan)])
        self.audio[s:e] = loudness

    def scores(self, chat: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = self.cfg
        n = self.end
        audio = np.convolve(np.nan_to_num(self.audio, nan=-90.0), np.ones(3) / 3, mode="same")
        audio[np.isnan(self.audio)] = np.nan
        if chat is not None:
            w = max(1, int(c.chat_window_seconds))
            chat_rate = np.convolve(chat[:n], np.ones(w))[:n]  # trailing sum over the window
        za, zc = np.zeros(n), np.zeros(n)
        for t in range(n):
            hi = t - 30
            lo = max(0, t - int(c.baseline_window_seconds))
            if t < c.min_baseline_seconds or hi - lo < 60:
                continue
            za[t] = _robust_z(audio, t, lo, hi, floor=1.5)
            if chat is not None and chat_rate[t] >= c.min_chat_messages:
                zc[t] = _robust_z(chat_rate, t, lo, hi, floor=2.0)
        if chat is None:
            score = za
        else:
            tot = c.chat_weight + c.audio_weight
            score = (c.chat_weight * zc + c.audio_weight * za) / tot
        return score, zc, za

    def detect(self, chat: np.ndarray | None, final: bool = False) -> list[HypeMoment]:
        """Find new hype moments. Needs post_roll seconds of audio after a moment unless final."""
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
            if t - self.last_peak >= c.cooldown_seconds:
                out.append(HypeMoment(float(t), float(score[t]), float(zc[t]), float(za[t])))
                self.last_peak = t
        self.cursor = next_cursor
        return out
