"""Configuration loading: config.yaml for settings, .env for secrets."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

PLATFORMS = ("twitch", "youtube", "kick")
VERTICAL_LAYOUTS = ("auto", "split", "face", "blur", "center")


@dataclass
class Streamer:
    platform: str
    channel: str  # twitch login, kick slug, or youtube channel id / @handle
    name: str = ""
    kick_chatroom_id: int | None = None
    vertical_layout: str | None = None  # override clips.vertical_layout for this streamer
    facecam: list[float] | None = None  # [x, y, width, height] of the facecam, as fractions of the screen

    def __post_init__(self):
        self.platform = self.platform.lower()
        if self.platform not in PLATFORMS:
            raise ValueError(f"Unknown platform {self.platform!r} (use one of {PLATFORMS})")
        if not self.name:
            self.name = self.channel.lstrip("@")
        if self.vertical_layout and self.vertical_layout not in VERTICAL_LAYOUTS:
            raise ValueError(f"{self.name}: vertical_layout must be one of {VERTICAL_LAYOUTS}")
        if self.facecam is not None and (len(self.facecam) != 4 or not all(0 <= v <= 1 for v in self.facecam)):
            raise ValueError(f"{self.name}: facecam must be [x, y, width, height] with values between 0 and 1")

    @property
    def key(self) -> str:
        return f"{self.platform}:{self.channel.lower()}"

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier."""
        return re.sub(r"[^A-Za-z0-9_-]+", "_", f"{self.platform}_{self.name}")


@dataclass
class RecordingConfig:
    dir: Path = Path("data/recordings")
    segment_seconds: int = 600
    quality: str = "best"
    streamlink_args: list[str] = field(default_factory=lambda: ["--twitch-disable-ads"])
    retention_hours: float = 24


@dataclass
class ClipConfig:
    dir: Path = Path("data/clips")
    mode: str = "best_per_chunk"  # spikes | best_per_chunk | both
    best_min_score: float | None = None  # best_per_chunk: skip chunks whose best moment scores below this
    pre_roll_seconds: float = 40
    post_roll_seconds: float = 20
    min_clip_seconds: float = 30
    max_clip_seconds: float = 60
    threshold: float = 3.5
    cooldown_seconds: float = 90
    min_baseline_seconds: float = 180
    baseline_window_seconds: float = 600
    chat_window_seconds: int = 10
    chat_weight: float = 0.6
    audio_weight: float = 0.4
    min_chat_messages: int = 0
    keywords: list[str] = field(default_factory=lambda: [
        "pog", "poggers", "pogchamp", "kekw", "lul", "lmao", "lmfao", "omegalul",
        "clip", "clip it", "holy", "wtf", "no way", "insane", "w", "lets go", "letsgo",
    ])
    keyword_weight: float = 1.0
    vertical: bool = False
    vertical_layout: str = "auto"  # auto | split | face | blur | center
    youtube_chat_min_interval: float = 30


@dataclass
class DiscordConfig:
    notify_channel_id: int = 0
    approval_channel_id: int = 0
    mention_user_id: int | None = None
    approver_ids: list[int] = field(default_factory=list)
    upload_limit_mb: float = 9.5


@dataclass
class PostForMeConfig:
    base_url: str = "https://api.postforme.dev"
    social_account_ids: list[str] = field(default_factory=list)
    caption_template: str = "{title} 🔥 ({streamer})"


@dataclass
class Secrets:
    twitch_client_id: str = ""
    twitch_client_secret: str = ""
    youtube_api_key: str = ""
    kick_client_id: str = ""
    kick_client_secret: str = ""
    discord_bot_token: str = ""
    postforme_api_key: str = ""


@dataclass
class Config:
    streamers: list[Streamer]
    poll_interval_seconds: float = 120
    db_path: Path = Path("data/clipper.db")
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    clips: ClipConfig = field(default_factory=ClipConfig)
    discord: DiscordConfig = field(default_factory=DiscordConfig)
    postforme: PostForMeConfig = field(default_factory=PostForMeConfig)
    secrets: Secrets = field(default_factory=Secrets)


def _section(cls, data: dict | None):
    data = dict(data or {})
    known = cls.__dataclass_fields__
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown keys in {cls.__name__}: {sorted(unknown)}")
    if "dir" in data:
        data["dir"] = Path(data["dir"])
    obj = cls(**data)
    if cls is ClipConfig and obj.mode not in ("spikes", "best_per_chunk", "both"):
        raise ValueError(f"clips.mode must be spikes, best_per_chunk or both (got {obj.mode!r})")
    if cls is ClipConfig and obj.vertical_layout not in VERTICAL_LAYOUTS:
        raise ValueError(f"clips.vertical_layout must be one of {VERTICAL_LAYOUTS} (got {obj.vertical_layout!r})")
    return obj


def load_config(path: str | Path = "config.yaml", env_file: str | Path = ".env") -> Config:
    load_dotenv(env_file)
    raw = yaml.safe_load(Path(path).read_text()) or {}
    streamers = [Streamer(**s) for s in raw.pop("streamers", [])]
    if not streamers:
        raise ValueError("config.yaml must list at least one streamer")
    cfg = Config(
        streamers=streamers,
        poll_interval_seconds=float(raw.pop("poll_interval_seconds", 120)),
        db_path=Path(raw.pop("db_path", "data/clipper.db")),
        recording=_section(RecordingConfig, raw.pop("recording", None)),
        clips=_section(ClipConfig, raw.pop("clips", None)),
        discord=_section(DiscordConfig, raw.pop("discord", None)),
        postforme=_section(PostForMeConfig, raw.pop("postforme", None)),
        secrets=Secrets(**{k: os.getenv(k.upper(), "") for k in Secrets.__dataclass_fields__}),
    )
    if raw:
        raise ValueError(f"Unknown top-level keys in config.yaml: {sorted(raw)}")
    return cfg


def check_secrets(cfg: Config) -> list[str]:
    """Return a list of missing secrets needed for the configured streamers."""
    s, missing = cfg.secrets, []
    used = {st.platform for st in cfg.streamers}
    need = {"DISCORD_BOT_TOKEN": s.discord_bot_token, "POSTFORME_API_KEY": s.postforme_api_key}
    if "twitch" in used:
        need |= {"TWITCH_CLIENT_ID": s.twitch_client_id, "TWITCH_CLIENT_SECRET": s.twitch_client_secret}
    if "youtube" in used:
        need |= {"YOUTUBE_API_KEY": s.youtube_api_key}
    if "kick" in used:
        need |= {"KICK_CLIENT_ID": s.kick_client_id, "KICK_CLIENT_SECRET": s.kick_client_secret}
    missing = [k for k, v in need.items() if not v]
    if not cfg.discord.notify_channel_id or not cfg.discord.approval_channel_id:
        missing.append("discord.notify_channel_id / discord.approval_channel_id (config.yaml)")
    return missing
