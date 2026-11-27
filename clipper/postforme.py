"""Post for Me API client (https://api.postforme.dev/docs).

Flow: create a signed upload URL -> PUT the video -> create a social post that references
the returned media URL and the target social account ids. Publishing to each platform then
happens asynchronously; the outcome per account is in /v1/social-post-results.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp

log = logging.getLogger(__name__)


class PostForMeError(RuntimeError):
    pass


API_TIMEOUT = aiohttp.ClientTimeout(total=120, sock_connect=30)
# Uploads of 50-100 MB clips can take minutes on a home connection: no overall limit,
# only fail if the connection stalls completely for 2 minutes.
UPLOAD_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=120)
UPLOAD_ATTEMPTS = 3


def describe_error(e: BaseException) -> str:
    """Readable message even for exceptions whose str() is empty (timeouts, disconnects)."""
    if isinstance(e, (asyncio.TimeoutError, TimeoutError)):
        return "timed out talking to Post for Me (slow or interrupted internet connection?)"
    text = str(e).strip()
    return f"{type(e).__name__}: {text}" if text else type(e).__name__


@dataclass
class PostResult:
    account_id: str
    platform: str
    username: str
    success: bool
    url: str | None
    error: str | None


def youtube_title(caption: str) -> str:
    """YouTube title from the caption's first line: no hashtags, max 100 characters, no angle brackets."""
    first = caption.strip().splitlines()[0] if caption.strip() else ""
    title = re.sub(r"\s+", " ", re.sub(r"#\w+", "", re.sub(r"[<>]", "", first))).strip() or "Stream clip"
    return title if len(title) <= 100 else title[:97].rstrip() + "..."


class PostForMe:
    def __init__(self, session: aiohttp.ClientSession, api_key: str, base_url: str):
        self.session, self.base = session, base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async def _req(self, method: str, path: str, **kw):
        kw.setdefault("timeout", API_TIMEOUT)
        async with self.session.request(method, f"{self.base}{path}", headers=self.headers, **kw) as r:
            body = await r.text()
            if r.status >= 400:
                raise PostForMeError(f"{method} {path} -> {r.status}: {body[:500]}")
            return await r.json(content_type=None) if body else {}

    async def social_accounts(self, connected_only: bool = False) -> list[dict]:
        params = {"limit": 100}
        if connected_only:
            params["status"] = "connected"
        j = await self._req("GET", "/v1/social-accounts", params=params)
        return j.get("data", j) if isinstance(j, dict) else j

    async def upload(self, path: Path) -> str:
        size = path.stat().st_size
        last: BaseException | None = None
        for attempt in range(1, UPLOAD_ATTEMPTS + 1):
            # fresh signed URL each attempt, since they expire quickly
            j = await self._req("POST", "/v1/media/create-upload-url", json={})
            upload_url, media_url = j["upload_url"], j["media_url"]
            started = time.monotonic()
            log.info("Uploading %s (%.1f MB), attempt %d/%d", path.name, size / 1e6, attempt, UPLOAD_ATTEMPTS)
            try:
                with path.open("rb") as f:
                    async with self.session.put(upload_url, data=f, timeout=UPLOAD_TIMEOUT,
                                                headers={"Content-Type": "video/mp4",
                                                         "Content-Length": str(size)}) as r:
                        if r.status >= 400:
                            raise PostForMeError(f"media upload failed: HTTP {r.status} {(await r.text())[:300]}")
                secs = time.monotonic() - started
                log.info("Uploaded %s in %.0fs (%.1f Mbit/s)", path.name, secs, size * 8 / 1e6 / max(secs, 0.1))
                return media_url
            except (asyncio.TimeoutError, aiohttp.ClientError, PostForMeError) as e:
                last = e
                log.warning("Upload attempt %d failed: %s", attempt, describe_error(e))
                if attempt < UPLOAD_ATTEMPTS:
                    await asyncio.sleep(5 * attempt)
        raise PostForMeError(f"upload failed after {UPLOAD_ATTEMPTS} attempts: {describe_error(last)}")

    async def create_post(self, caption: str, media_url: str, account_ids: list[str]) -> dict:
        return await self._req("POST", "/v1/social-posts", json={
            "caption": caption,
            "social_accounts": account_ids,
            "media": [{"url": media_url}],
            "platform_configurations": {"youtube": {"title": youtube_title(caption)}},
        })

    async def post_video(self, path: Path, caption: str, account_ids: list[str]) -> tuple[dict, list[str]]:
        """Upload and create the post. Returns (post, account ids it was sent to)."""
        if not account_ids:
            account_ids = [a["id"] for a in await self.social_accounts(connected_only=True)]
            if not account_ids:
                raise PostForMeError("no connected social accounts in Post for Me — connect one in its dashboard")
        media_url = await self.upload(path)
        post = await self.create_post(caption, media_url, account_ids)
        log.info("Created Post for Me post %s for %s (status %s)", post.get("id"), path.name, post.get("status"))
        return post, account_ids

    async def results(self, post_id: str) -> list[PostResult]:
        j = await self._req("GET", "/v1/social-post-results", params={"post_id": post_id, "limit": 100})
        accounts = {a["id"]: a for a in await self.social_accounts()}
        out = []
        for r in j.get("data", []):
            acct = accounts.get(r.get("social_account_id"), {})
            err = r.get("error")
            if err is not None and not isinstance(err, str):
                err = err.get("message") if isinstance(err, dict) and "message" in err else str(err)
            out.append(PostResult(r.get("social_account_id", ""), acct.get("platform", "?"),
                                  acct.get("username") or "", bool(r.get("success")),
                                  (r.get("platform_data") or {}).get("url"), err))
        return out

    async def wait_for_results(self, post_id: str, expected: int, timeout: float = 900,
                               interval: float = 15) -> list[PostResult]:
        """Poll until every account has a result (or timeout). Platforms like YouTube can take minutes."""
        deadline = time.monotonic() + timeout
        res: list[PostResult] = []
        while time.monotonic() < deadline:
            await asyncio.sleep(interval)
            try:
                res = await self.results(post_id)
            except PostForMeError as e:
                log.warning("Fetching results for %s failed: %s", post_id, e)
                continue
            if len(res) >= expected:
                break
        return res
