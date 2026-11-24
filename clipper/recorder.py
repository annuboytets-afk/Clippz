"""Streamlink -> ffmpeg segmenter. Produces fixed-length .ts chunks with a CSV index."""
from __future__ import annotations

import asyncio
import csv
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable

log = logging.getLogger(__name__)


@dataclass
class Segment:
    path: Path
    start: float  # seconds since recording start
    end: float


class Recording:
    def __init__(self, url: str, out_dir: Path, segment_seconds: int, quality: str,
                 streamlink_args: list[str],
                 on_segment: Callable[["Recording", Segment], None],
                 on_finished: Callable[["Recording"], Awaitable[None]]):
        self.url = url
        self.dir = out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.segment_seconds, self.quality, self.streamlink_args = segment_seconds, quality, streamlink_args
        self.on_segment, self.on_finished = on_segment, on_finished
        self.segments: list[Segment] = []
        self.t0: float | None = None  # wall-clock time of the first received byte
        self.finished = asyncio.Event()
        self._sl = self._ff = None
        self._task: asyncio.Task | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self.finished.is_set()

    async def start(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        self._log = open(self.dir / "recorder.log", "ab")
        self._sl = await asyncio.create_subprocess_exec(
            "streamlink", *self.streamlink_args, "--stdout", self.url, self.quality,
            stdout=asyncio.subprocess.PIPE, stderr=self._log)
        self._ff = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-fflags", "+genpts", "-i", "pipe:0",
            "-map", "0:v?", "-map", "0:a?", "-c", "copy",
            "-f", "segment", "-segment_time", str(self.segment_seconds), "-segment_format", "mpegts",
            "-segment_list", "segments.csv", "-segment_list_type", "csv",
            "chunk_%05d.ts", cwd=self.dir, stdin=asyncio.subprocess.PIPE, stderr=self._log)
        self._task = asyncio.create_task(self._run())
        log.info("Recording %s -> %s", self.url, self.dir)

    async def stop(self):
        if self._sl and self._sl.returncode is None:
            self._sl.terminate()
        await self.finished.wait()

    async def _run(self):
        watcher = asyncio.create_task(self._watch_index())
        try:
            while True:
                chunk = await self._sl.stdout.read(1 << 16)
                if not chunk:
                    break
                if self.t0 is None:
                    self.t0 = time.time()
                self._ff.stdin.write(chunk)
                await self._ff.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as e:
            log.warning("ffmpeg pipe closed: %s", e)
        finally:
            if self._sl.returncode is None:
                self._sl.terminate()
            await self._sl.wait()
            try:
                self._ff.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            await self._ff.wait()
            watcher.cancel()
            self._scan_index()  # pick up the final segment
            self._log.close()
            self.finished.set()
            log.info("Recording finished: %s (%d segments)", self.dir, len(self.segments))
            await self.on_finished(self)

    async def _watch_index(self):
        while True:
            await asyncio.sleep(5)
            self._scan_index()

    def _scan_index(self):
        """Synchronous on purpose: on_segment only enqueues work, so cancellation can't drop a segment."""
        idx = self.dir / "segments.csv"
        if not idx.exists():
            return
        with idx.open(newline="") as f:
            rows = list(csv.reader(f))
        for row in rows[len(self.segments):]:
            if len(row) < 3:
                break
            seg = Segment(self.dir / row[0], float(row[1]), float(row[2]))
            self.segments.append(seg)
            try:
                self.on_segment(self, seg)
            except Exception:  # noqa: BLE001
                log.exception("Segment handler failed for %s", seg.path)
