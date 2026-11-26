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


@dataclass
class PostResult:
    account_id: str
    platform: str
    username: str
    success: bool
    url: str | None
    error: str | None


def youtube_title(caption: str) -> str:
    """YouTube titles: max 100 characters, no angle brackets, not empty."""
    title = re.sub(r"[<>]", "", caption).strip() or "Stream clip"
    return title if len(title) <= 100 else title[:97].rstrip() + "..."


class PostForMe:
    def __init__(self, session: aiohttp.ClientSession, api_key: str, base_url: str):
        self.session, self.base = session, base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async def _req(self, method: str, path: str, **kw):
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
        j = await self._req("POST", "/v1/media/create-upload-url", json={})
        upload_url, media_url = j["upload_url"], j["media_url"]
        with path.open("rb") as f:
            async with self.session.put(upload_url, data=f, headers={"Content-Type": "video/mp4"}) as r:
                if r.status >= 400:
                    raise PostForMeError(f"media upload failed: {r.status} {await r.text()}")
        return media_url

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
