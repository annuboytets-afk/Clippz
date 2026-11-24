"""Chat listeners. Each one feeds a ChatCounter with (timestamp, weight) events."""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from collections import deque
from datetime import datetime

import aiohttp
import numpy as np

from .config import Streamer

log = logging.getLogger(__name__)

KICK_PUSHER_URL = ("wss://ws-us2.pusher.com/app/32cbd69e4b950bf97679"
                   "?protocol=7&client=js&version=8.4.0&flash=false")
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/131.0 Safari/537.36")


class ChatCounter:
    """Stores weighted chat message timestamps; messages with hype keywords weigh more."""

    def __init__(self, keywords: list[str], keyword_weight: float = 1.0, keep_seconds: float = 6 * 3600):
        words = sorted({k.lower() for k in keywords}, key=len, reverse=True)
        self._re = re.compile(r"(?<![a-z0-9])(" + "|".join(map(re.escape, words)) + r")(?![a-z0-9])") if words else None
        self.keyword_weight = keyword_weight
        self.keep = keep_seconds
        self.events: deque[tuple[float, float]] = deque()
        self.active = False  # set once a listener has connected

    def add(self, text: str, ts: float | None = None):
        ts = time.time() if ts is None else ts
        w = 1.0
        if self._re and self._re.search(text.lower()):
            w += self.keyword_weight
        self.events.append((ts, w))
        cutoff = time.time() - self.keep
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def series(self, t0: float, n: int) -> np.ndarray:
        """Weighted message counts per second for seconds [0, n) after wall-clock t0."""
        out = np.zeros(n)
        for ts, w in self.events:
            i = int(ts - t0)
            if 0 <= i < n:
                out[i] += w
        return out


async def twitch_chat(channel: str, counter: ChatCounter):
    """Anonymous read-only Twitch IRC connection (no token needed)."""
    channel = channel.lower()
    while True:
        try:
            reader, writer = await asyncio.open_connection("irc.chat.twitch.tv", 6697, ssl=True)
            writer.write(f"NICK justinfan{random.randint(10000, 99999)}\r\nJOIN #{channel}\r\n".encode())
            await writer.drain()
            counter.active = True
            log.info("Twitch chat connected: #%s", channel)
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=360)
                if not line:
                    break
                s = line.decode("utf-8", "ignore").rstrip("\r\n")
                if s.startswith("PING"):
                    writer.write(b"PONG :tmi.twitch.tv\r\n")
                    await writer.drain()
                elif " PRIVMSG " in s:
                    _, _, msg = s.partition(" PRIVMSG ")[2].partition(" :")
                    counter.add(msg)
            writer.close()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("Twitch chat #%s error: %s; reconnecting", channel, e)
        await asyncio.sleep(5)


async def _kick_chatroom_id(session: aiohttp.ClientSession, streamer: Streamer) -> int | None:
    if streamer.kick_chatroom_id:
        return streamer.kick_chatroom_id
    # Unofficial endpoint (the public API does not expose the chatroom id). May be blocked by Cloudflare.
    try:
        async with session.get(f"https://kick.com/api/v2/channels/{streamer.channel}",
                               headers={"User-Agent": BROWSER_UA, "Accept": "application/json"}) as r:
            r.raise_for_status()
            j = await r.json(content_type=None)
            return int(j["chatroom"]["id"])
    except Exception as e:  # noqa: BLE001
        log.warning("Could not look up Kick chatroom id for %s (%s). Set kick_chatroom_id in config.yaml "
                    "to enable chat analysis; continuing with audio only.", streamer.channel, e)
        return None


async def kick_chat(session: aiohttp.ClientSession, streamer: Streamer, counter: ChatCounter):
    room = await _kick_chatroom_id(session, streamer)
    if room is None:
        return
    while True:
        try:
            async with session.ws_connect(KICK_PUSHER_URL, heartbeat=60) as ws:
                await ws.send_json({"event": "pusher:subscribe",
                                    "data": {"auth": "", "channel": f"chatrooms.{room}.v2"}})
                counter.active = True
                log.info("Kick chat connected: %s (room %s)", streamer.channel, room)
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    d = json.loads(msg.data)
                    ev = d.get("event", "")
                    if ev == "pusher:ping":
                        await ws.send_json({"event": "pusher:pong", "data": {}})
                    elif ev.endswith("ChatMessageEvent"):
                        data = json.loads(d["data"]) if isinstance(d.get("data"), str) else d.get("data", {})
                        counter.add(data.get("content", ""))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("Kick chat %s error: %s; reconnecting", streamer.channel, e)
        await asyncio.sleep(5)


async def youtube_chat(yt, live_chat_id: str, counter: ChatCounter, min_interval: float):
    """Official liveChatMessages API (5 quota units per call; interval is configurable)."""
    page = None
    counter.active = True
    while True:
        try:
            j = await yt.chat_messages(live_chat_id, page)
            for item in j.get("items", []):
                sn = item.get("snippet", {})
                try:
                    ts = datetime.fromisoformat(sn["publishedAt"].replace("Z", "+00:00")).timestamp()
                except Exception:  # noqa: BLE001
                    ts = None
                counter.add(sn.get("displayMessage", ""), ts)
            page = j.get("nextPageToken", page)
            await asyncio.sleep(max(min_interval, j.get("pollingIntervalMillis", 5000) / 1000))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("YouTube chat error: %s", e)
            if "liveChatEnded" in str(e) or "forbidden" in str(e).lower():
                return
            await asyncio.sleep(60)
