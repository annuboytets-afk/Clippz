"""Live-status checks for Twitch (Helix), YouTube (Data API v3) and Kick (public API)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import aiohttp

from .config import Streamer

log = logging.getLogger(__name__)


@dataclass
class LiveInfo:
    streamer: Streamer
    title: str
    url: str  # what streamlink records
    started_at: str | None = None
    live_chat_id: str | None = None  # YouTube only


class _AppToken:
    """Client-credentials OAuth token with auto refresh."""

    def __init__(self, session: aiohttp.ClientSession, url: str, client_id: str, client_secret: str):
        self.session, self.url = session, url
        self.client_id, self.client_secret = client_id, client_secret
        self._token: str | None = None
        self._expires = 0.0

    async def get(self) -> str:
        if self._token and time.time() < self._expires - 120:
            return self._token
        data = {"client_id": self.client_id, "client_secret": self.client_secret,
                "grant_type": "client_credentials"}
        async with self.session.post(self.url, data=data) as r:
            if r.status >= 400:
                raise RuntimeError(f"token request to {self.url} failed: {r.status} {await r.text()}")
            j = await r.json()
        self._token = j["access_token"]
        self._expires = time.time() + float(j.get("expires_in", 3600))
        return self._token

    def invalidate(self):
        self._token = None


class TwitchAPI:
    def __init__(self, session, client_id, client_secret):
        self.session, self.client_id = session, client_id
        self.token = _AppToken(session, "https://id.twitch.tv/oauth2/token", client_id, client_secret)

    async def live(self, streamers: list[Streamer]) -> dict[str, LiveInfo]:
        out: dict[str, LiveInfo] = {}
        by_login = {s.channel.lower(): s for s in streamers}
        logins = list(by_login)
        for i in range(0, len(logins), 100):
            params = [("user_login", l) for l in logins[i:i + 100]]
            for attempt in range(2):
                headers = {"Client-Id": self.client_id, "Authorization": f"Bearer {await self.token.get()}"}
                async with self.session.get("https://api.twitch.tv/helix/streams", params=params,
                                            headers=headers) as r:
                    if r.status == 401 and attempt == 0:
                        self.token.invalidate()
                        continue
                    r.raise_for_status()
                    j = await r.json()
                    break
            for st in j.get("data", []):
                if st.get("type") != "live":
                    continue
                s = by_login.get(st["user_login"].lower())
                if s:
                    out[s.key] = LiveInfo(s, st.get("title", ""), f"https://www.twitch.tv/{st['user_login']}",
                                          st.get("started_at"))
        return out


class KickAPI:
    def __init__(self, session, client_id, client_secret):
        self.session = session
        self.token = _AppToken(session, "https://id.kick.com/oauth/token", client_id, client_secret)

    async def live(self, streamers: list[Streamer]) -> dict[str, LiveInfo]:
        out: dict[str, LiveInfo] = {}
        by_slug = {s.channel.lower(): s for s in streamers}
        slugs = list(by_slug)
        for i in range(0, len(slugs), 50):
            params = [("slug", s) for s in slugs[i:i + 50]]
            for attempt in range(2):
                headers = {"Authorization": f"Bearer {await self.token.get()}", "Accept": "application/json"}
                async with self.session.get("https://api.kick.com/public/v1/channels", params=params,
                                            headers=headers) as r:
                    if r.status == 401 and attempt == 0:
                        self.token.invalidate()
                        continue
                    r.raise_for_status()
                    j = await r.json()
                    break
            for ch in j.get("data", []):
                stream = ch.get("stream") or {}
                s = by_slug.get(str(ch.get("slug", "")).lower())
                if s and stream.get("is_live"):
                    out[s.key] = LiveInfo(s, ch.get("stream_title", ""), f"https://kick.com/{ch['slug']}",
                                          stream.get("start_time"))
        return out


class YouTubeAPI:
    """Quota-friendly live check: uploads playlist (1 unit/channel) + one videos.list (1 unit).

    search.list?eventType=live costs 100 units per call, which would burn the 10k/day
    quota in minutes at a 2-minute poll interval, so it is deliberately avoided.
    """

    BASE = "https://www.googleapis.com/youtube/v3"

    def __init__(self, session, api_key):
        self.session, self.key = session, api_key
        self._channel_ids: dict[str, str] = {}

    async def _get(self, path, **params):
        params["key"] = self.key
        async with self.session.get(f"{self.BASE}/{path}", params=params) as r:
            if r.status >= 400:
                raise RuntimeError(f"YouTube {path} failed: {r.status} {await r.text()}")
            return await r.json()

    async def channel_id(self, s: Streamer) -> str:
        if s.key in self._channel_ids:
            return self._channel_ids[s.key]
        ch = s.channel
        if not ch.startswith("UC"):
            j = await self._get("channels", part="id", forHandle=ch if ch.startswith("@") else f"@{ch}")
            items = j.get("items") or []
            if not items:
                raise RuntimeError(f"YouTube channel {ch!r} not found")
            ch = items[0]["id"]
        self._channel_ids[s.key] = ch
        return ch

    async def live(self, streamers: list[Streamer]) -> dict[str, LiveInfo]:
        candidates: dict[str, Streamer] = {}
        for s in streamers:
            try:
                cid = await self.channel_id(s)
                j = await self._get("playlistItems", part="contentDetails", maxResults=5,
                                    playlistId="UU" + cid[2:])
                for it in j.get("items", []):
                    candidates[it["contentDetails"]["videoId"]] = s
            except Exception as e:  # noqa: BLE001
                log.warning("YouTube check for %s failed: %s", s.name, e)
        out: dict[str, LiveInfo] = {}
        ids = list(candidates)
        for i in range(0, len(ids), 50):
            j = await self._get("videos", part="snippet,liveStreamingDetails", id=",".join(ids[i:i + 50]))
            for v in j.get("items", []):
                sn, lsd = v.get("snippet", {}), v.get("liveStreamingDetails", {})
                if sn.get("liveBroadcastContent") != "live" or lsd.get("actualEndTime"):
                    continue
                s = candidates[v["id"]]
                out.setdefault(s.key, LiveInfo(s, sn.get("title", ""), f"https://www.youtube.com/watch?v={v['id']}",
                                               lsd.get("actualStartTime"), lsd.get("activeLiveChatId")))
        return out

    async def chat_messages(self, live_chat_id: str, page_token: str | None):
        params = {"liveChatId": live_chat_id, "part": "snippet", "maxResults": 2000}
        if page_token:
            params["pageToken"] = page_token
        return await self._get("liveChat/messages", **params)
