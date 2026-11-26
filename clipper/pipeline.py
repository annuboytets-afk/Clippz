"""Orchestrates polling, recording, chat, hype detection, clipping and approval."""
from __future__ import annotations

import asyncio
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import aiohttp

from . import chat as chatmod
from .analysis import HypeDetector, HypeMoment, audio_loudness
from .clips import cut_clip, make_preview
from .config import Config, Streamer
from .db import ClipDB
from .platforms import KickAPI, LiveInfo, TwitchAPI, YouTubeAPI
from .postforme import PostForMe
from .recorder import Recording, Segment

log = logging.getLogger(__name__)


@dataclass
class Part:
    """One continuous recording (a stream that drops and resumes gets several parts)."""
    rec: Recording
    detector: HypeDetector
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    worker: asyncio.Task | None = None


@dataclass
class LiveSession:
    streamer: Streamer
    info: LiveInfo
    counter: chatmod.ChatCounter
    chat_task: asyncio.Task | None = None
    parts: list[Part] = field(default_factory=list)
    offline_polls: int = 0

    @property
    def current(self) -> Part | None:
        return self.parts[-1] if self.parts else None


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.db = ClipDB(cfg.db_path)
        self.sessions: dict[str, LiveSession] = {}
        self.session: aiohttp.ClientSession | None = None
        self.bot = None  # set by __main__
        self._clip_sem = asyncio.Semaphore(2)  # limit concurrent ffmpeg encodes

    # ---------- polling ----------
    async def run(self):
        s = self.cfg.secrets
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
        self.twitch = TwitchAPI(self.session, s.twitch_client_id, s.twitch_client_secret)
        self.youtube = YouTubeAPI(self.session, s.youtube_api_key)
        self.kick = KickAPI(self.session, s.kick_client_id, s.kick_client_secret)
        self.pfm = PostForMe(self.session, s.postforme_api_key, self.cfg.postforme.base_url)
        asyncio.create_task(self._cleanup_loop())
        log.info("Watching %d streamers every %ss", len(self.cfg.streamers), self.cfg.poll_interval_seconds)
        while True:
            started = time.monotonic()
            try:
                await self.poll()
            except Exception:  # noqa: BLE001
                log.exception("Poll failed")
            await asyncio.sleep(max(5, self.cfg.poll_interval_seconds - (time.monotonic() - started)))

    async def poll(self):
        groups = {p: [s for s in self.cfg.streamers if s.platform == p] for p in ("twitch", "youtube", "kick")}
        apis = {"twitch": self.twitch, "youtube": self.youtube, "kick": self.kick}
        live: dict[str, LiveInfo] = {}
        failed: set[str] = set()
        for platform, streamers in groups.items():
            if not streamers:
                continue
            try:
                live |= await apis[platform].live(streamers)
            except Exception as e:  # noqa: BLE001
                failed.add(platform)
                log.warning("%s live check failed: %s", platform, e)

        for s in self.cfg.streamers:
            if s.platform in failed:
                continue  # unknown state; don't start or stop anything
            info, sess = live.get(s.key), self.sessions.get(s.key)
            if info and not sess:
                await self._go_live(info)
            elif info and sess:
                sess.offline_polls = 0
                sess.info = info
                if not sess.current or not sess.current.rec.running:
                    log.info("%s still live but recorder stopped; starting a new part", s.name)
                    await self._start_part(sess)
            elif sess:
                sess.offline_polls += 1
                if sess.offline_polls >= 2 or not (sess.current and sess.current.rec.running):
                    await self._end_session(sess)

    async def _go_live(self, info: LiveInfo):
        s = info.streamer
        log.info("%s is live on %s: %s", s.name, s.platform, info.title)
        c = self.cfg.clips
        sess = LiveSession(s, info, chatmod.ChatCounter(c.keywords, c.keyword_weight))
        self.sessions[s.key] = sess
        if s.platform == "twitch":
            sess.chat_task = asyncio.create_task(chatmod.twitch_chat(s.channel, sess.counter))
        elif s.platform == "kick":
            sess.chat_task = asyncio.create_task(chatmod.kick_chat(self.session, s, sess.counter))
        elif s.platform == "youtube" and info.live_chat_id:
            sess.chat_task = asyncio.create_task(
                chatmod.youtube_chat(self.youtube, info.live_chat_id, sess.counter, c.youtube_chat_min_interval))
        await self._start_part(sess)
        try:
            await self.bot.notify(f"🔴 **{s.name}** is live on {s.platform.title()}: {info.title}\n{info.url}\n"
                                  f"Recording in {self.cfg.recording.segment_seconds / 60:g}-minute chunks.")
        except Exception:  # noqa: BLE001
            log.exception("Discord notify failed")

    async def _start_part(self, sess: LiveSession):
        r = self.cfg.recording
        rec = Recording(sess.info.url, r.dir / sess.streamer.slug, r.segment_seconds, r.quality,
                        r.streamlink_args, self._on_segment, self._on_finished)
        part = Part(rec, HypeDetector(self.cfg.clips))
        rec.part, rec.session = part, sess  # back-references for callbacks
        part.worker = asyncio.create_task(self._part_worker(sess, part))
        sess.parts.append(part)
        await rec.start()

    async def _end_session(self, sess: LiveSession):
        log.info("%s went offline", sess.streamer.name)
        self.sessions.pop(sess.streamer.key, None)
        if sess.current and sess.current.rec.running:
            await sess.current.rec.stop()
        if sess.chat_task:
            sess.chat_task.cancel()
        try:
            await self.bot.notify(f"⚫ **{sess.streamer.name}** went offline.")
        except Exception:  # noqa: BLE001
            log.exception("Discord notify failed")

    # ---------- analysis ----------
    def _on_segment(self, rec: Recording, seg: Segment):
        rec.part.queue.put_nowait(("segment", seg))

    async def _on_finished(self, rec: Recording):
        rec.part.queue.put_nowait(("final", None))

    async def _part_worker(self, sess: LiveSession, part: Part):
        """Processes one recording's chunks in order."""
        while True:
            kind, seg = await part.queue.get()
            try:
                if kind == "segment":
                    loud = await asyncio.to_thread(audio_loudness, seg.path)
                    part.detector.add_audio(seg.start, loud)
                    log.info("%s: analysed chunk %s (%.0f-%.0fs)", sess.streamer.name, seg.path.name, seg.start, seg.end)
                final = kind == "final"
                chat = None
                if sess.counter.active and part.rec.t0:
                    chat = sess.counter.series(part.rec.t0, part.detector.end)
                for m in part.detector.process(chat, final=final):
                    asyncio.create_task(self._make_clip(sess, part, m))
                if final:
                    return
            except Exception:  # noqa: BLE001
                log.exception("Analysis failed for %s", sess.streamer.name)

    async def _make_clip(self, sess: LiveSession, part: Part, m: HypeMoment):
        c = self.cfg.clips
        s = sess.streamer
        start = max(0.0, m.t - c.pre_roll_seconds)
        end = min(float(part.detector.end), m.t + c.post_roll_seconds)
        stamp = datetime.fromtimestamp((part.rec.t0 or time.time()) + m.t).strftime("%Y%m%d_%H%M%S")
        out = c.dir / s.slug / f"{s.slug}_{stamp}.mp4"
        async with self._clip_sem:
            try:
                await asyncio.to_thread(cut_clip, part.rec.segments, start, end, out, c.vertical)
                preview = await asyncio.to_thread(make_preview, out, out.with_suffix(".preview.mp4"),
                                                  end - start, self.cfg.discord.upload_limit_mb)
            except Exception:  # noqa: BLE001
                log.exception("Cutting clip failed")
                return
        caption = self.cfg.postforme.caption_template.format(
            title=sess.info.title, streamer=s.name, platform=s.platform)
        clip_id = self.db.add(streamer=s.name, platform=s.platform, title=sess.info.title, path=str(out),
                              preview_path=str(preview), score=m.score, caption=caption)
        hhmmss = time.strftime("%H:%M:%S", time.gmtime(m.t))
        if m.kind == "best":
            why = f"🏆 Best moment of this {self.cfg.recording.segment_seconds / 60:g}-minute chunk"
        else:
            why = "🔥 Hype spike"
        text = (f"🎬 **Clip #{clip_id}** — {s.name} ({s.platform}) at {hhmmss} into the recording\n"
                f"{why} · score {m.score:.1f} (chat z={m.chat_z:.1f}, audio z={m.audio_z:.1f}), "
                f"{end - start:.0f}s\nCaption: {caption}")
        try:
            msg_id = await self.bot.send_for_approval(clip_id, text, preview, self.cfg.discord.upload_limit_mb)
            self.db.update(clip_id, discord_message_id=msg_id)
        except Exception:  # noqa: BLE001
            log.exception("Sending clip %s to Discord failed", clip_id)

    # ---------- posting ----------
    async def post_clip(self, clip_id: int, caption: str) -> str:
        clip = self.db.get(clip_id)
        post = await self.pfm.post_video(Path(clip["path"]), caption, self.cfg.postforme.social_account_ids)
        post_id = str(post.get("id", ""))
        self.db.update(clip_id, status="posted", post_id=post_id, error=None)
        return post_id

    # ---------- housekeeping ----------
    async def _cleanup_loop(self):
        while True:
            try:
                self._cleanup()
            except Exception:  # noqa: BLE001
                log.exception("Cleanup failed")
            await asyncio.sleep(3600)

    def _cleanup(self):
        """Delete recording chunks older than retention_hours (clips are kept)."""
        root = self.cfg.recording.dir
        if not root.exists():
            return
        cutoff = time.time() - self.cfg.recording.retention_hours * 3600
        active = {p.rec.dir for s in self.sessions.values() for p in s.parts}
        for d in root.glob("*/*"):
            if d.is_dir() and d not in active and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                log.info("Removed old recording %s", d)
