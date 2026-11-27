"""Discord bot: go-live notifications and Approve / Edit caption / Reject buttons for clips."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .postforme import describe_error

import discord

if TYPE_CHECKING:
    from .pipeline import Pipeline

log = logging.getLogger(__name__)


class CaptionModal(discord.ui.Modal, title="Edit caption and post"):
    def __init__(self, bot: "ClipperBot", clip_id: int, caption: str):
        super().__init__()
        self.bot, self.clip_id = bot, clip_id
        self.caption = discord.ui.TextInput(label="Caption", style=discord.TextStyle.paragraph,
                                            default=caption[:2000], max_length=2000)
        self.add_item(self.caption)

    async def on_submit(self, interaction: discord.Interaction):
        await self.bot.approve(interaction, self.clip_id, str(self.caption.value))


class ApprovalView(discord.ui.View):
    """Persistent view: custom_ids are fixed and the clip is looked up by message id."""

    def __init__(self, bot: "ClipperBot"):
        super().__init__(timeout=None)
        self.bot = bot

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        allowed = self.bot.pipeline.cfg.discord.approver_ids
        if allowed and interaction.user.id not in allowed:
            await interaction.response.send_message("You're not allowed to approve clips.", ephemeral=True)
            return False
        return True

    def _clip(self, interaction):
        return self.bot.pipeline.db.by_message(interaction.message.id)

    @discord.ui.button(label="Approve & post", style=discord.ButtonStyle.success, custom_id="clip:approve")
    async def approve(self, interaction: discord.Interaction, _):
        clip = self._clip(interaction)
        if not clip:
            return await interaction.response.send_message("Unknown clip.", ephemeral=True)
        await self.bot.approve(interaction, clip["id"], clip["caption"])

    @discord.ui.button(label="Edit caption", style=discord.ButtonStyle.primary, custom_id="clip:edit")
    async def edit(self, interaction: discord.Interaction, _):
        clip = self._clip(interaction)
        if not clip:
            return await interaction.response.send_message("Unknown clip.", ephemeral=True)
        await interaction.response.send_modal(CaptionModal(self.bot, clip["id"], clip["caption"] or ""))

    @discord.ui.button(label="Reject", style=discord.ButtonStyle.danger, custom_id="clip:reject")
    async def reject(self, interaction: discord.Interaction, _):
        clip = self._clip(interaction)
        if not clip or not self.bot.pipeline.db.transition(clip["id"], "pending", "rejected"):
            return await interaction.response.send_message("Already handled.", ephemeral=True)
        await interaction.response.edit_message(
            content=f"{interaction.message.content}\n\n❌ Rejected by {interaction.user.mention}", view=None)


class ClipperBot(discord.Client):
    def __init__(self, pipeline: "Pipeline"):
        super().__init__(intents=discord.Intents.default())
        self.pipeline = pipeline
        self._started = False

    async def setup_hook(self):
        self.add_view(ApprovalView(self))

    async def on_ready(self):
        log.info("Discord bot logged in as %s", self.user)
        if not self._started:
            self._started = True
            self.loop.create_task(self.pipeline.run())

    async def _channel(self, channel_id: int):
        return self.get_channel(channel_id) or await self.fetch_channel(channel_id)

    async def notify(self, text: str):
        mention = self.pipeline.cfg.discord.mention_user_id
        ch = await self._channel(self.pipeline.cfg.discord.notify_channel_id)
        await ch.send(f"<@{mention}> {text}" if mention else text)

    async def send_for_approval(self, clip_id: int, text: str, preview: Path | None, limit_mb: float) -> int:
        ch = await self._channel(self.pipeline.cfg.discord.approval_channel_id)
        if preview and preview.exists() and preview.stat().st_size <= limit_mb * 1024 * 1024:
            msg = await ch.send(text, file=discord.File(preview), view=ApprovalView(self))
        else:
            msg = await ch.send(text + "\n_(preview too large to upload; check the clips folder)_",
                                view=ApprovalView(self))
        return msg.id

    async def approve(self, interaction: discord.Interaction, clip_id: int, caption: str):
        db = self.pipeline.db
        if not db.transition(clip_id, "pending", "posting"):
            return await interaction.response.send_message("Already handled.", ephemeral=True)
        db.update(clip_id, caption=caption)
        await interaction.response.defer()
        msg = interaction.message
        if msg is None:  # modal submit on some clients
            ch = await self._channel(self.pipeline.cfg.discord.approval_channel_id)
            msg = await ch.fetch_message(db.get(clip_id)["discord_message_id"])
        base = msg.content
        await msg.edit(content=f"{base}\n\n⏳ Posting (approved by {interaction.user.mention})…", view=None)
        who = interaction.user.mention
        try:
            post_id, expected = await self.pipeline.post_clip(clip_id, caption)
        except Exception as e:  # noqa: BLE001
            log.exception("Posting clip %s failed", clip_id)
            db.update(clip_id, status="pending", error=describe_error(e)[:1000])
            await msg.edit(content=f"{base}\n\n⚠️ Posting failed: `{describe_error(e)[:300]}` — try again.",
                           view=ApprovalView(self))
            return
        await msg.edit(content=f"{base}\n\n⏳ Approved by {who}. Uploaded to Post for Me (`{post_id}`), "
                               f"waiting for {expected} account(s) to publish…")
        try:
            results = await self.pipeline.finish_post(clip_id, post_id, expected)
        except Exception as e:  # noqa: BLE001
            log.exception("Checking results for clip %s failed", clip_id)
            results, err = [], describe_error(e)
        else:
            err = ""
        lines = []
        for r in results:
            name = f"{r.platform}" + (f" @{r.username}" if r.username else "")
            lines.append(f"✅ {name}: {r.url or 'published'}" if r.success
                         else f"❌ {name}: `{(r.error or 'unknown error')[:300]}`")
        if len(results) < expected:
            lines.append(f"⏳ {expected - len(results)} account(s) still publishing after 15 min "
                         f"(post `{post_id}`). Run `python -m clipper results` later to check."
                         + (f" Error: `{err[:200]}`" if err else ""))
        text = f"{base}\n\nApproved by {who}:\n" + "\n".join(lines)
        await msg.edit(content=text if len(text) <= 2000 else text[:1990] + "…")
