"""Post for Me API client (https://www.postforme.dev).

Flow: create a signed upload URL -> PUT the video -> create a social post that references
the returned media URL and the target social account ids.
"""
from __future__ import annotations

import logging
from pathlib import Path

import aiohttp

log = logging.getLogger(__name__)


class PostForMeError(RuntimeError):
    pass


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

    async def social_accounts(self) -> list[dict]:
        j = await self._req("GET", "/v1/social-accounts")
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
        })

    async def post_video(self, path: Path, caption: str, account_ids: list[str]) -> dict:
        if not account_ids:
            accounts = await self.social_accounts()
            account_ids = [a["id"] for a in accounts if a.get("status", "connected") == "connected"]
            if not account_ids:
                raise PostForMeError("no connected social accounts in Post for Me")
        media_url = await self.upload(path)
        post = await self.create_post(caption, media_url, account_ids)
        log.info("Posted %s via Post for Me: %s", path.name, post.get("id"))
        return post
