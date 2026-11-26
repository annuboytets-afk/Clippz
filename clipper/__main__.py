"""Entry point.

  python -m clipper run                 # start the bot + pipeline
  python -m clipper check               # validate config and secrets, test API logins
  python -m clipper accounts            # list Post for Me social account ids
  python -m clipper analyze FILE        # offline: find hype moments in a recording (audio only)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import sys
from pathlib import Path

import aiohttp
import numpy as np


def _setup_logging(verbose: bool):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    logging.getLogger("discord").setLevel(logging.WARNING)


def cmd_run(args):
    from .config import check_secrets, load_config
    from .discord_bot import ClipperBot
    from .pipeline import Pipeline

    cfg = load_config(args.config, args.env)
    missing = check_secrets(cfg)
    if missing:
        sys.exit("Missing configuration:\n  " + "\n  ".join(missing))
    for tool in ("streamlink", "ffmpeg"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found on PATH")
    pipeline = Pipeline(cfg)
    bot = ClipperBot(pipeline)
    pipeline.bot = bot
    bot.run(cfg.secrets.discord_bot_token, log_handler=None)


async def _check(cfg):
    from .platforms import KickAPI, TwitchAPI, YouTubeAPI
    from .postforme import PostForMe

    s = cfg.secrets
    async with aiohttp.ClientSession() as sess:
        apis = {"twitch": TwitchAPI(sess, s.twitch_client_id, s.twitch_client_secret),
                "youtube": YouTubeAPI(sess, s.youtube_api_key),
                "kick": KickAPI(sess, s.kick_client_id, s.kick_client_secret)}
        for p, api in apis.items():
            group = [st for st in cfg.streamers if st.platform == p]
            if not group:
                continue
            try:
                live = await api.live(group)
                for st in group:
                    info = live.get(st.key)
                    print(f"  {p:8} {st.name:20} {'LIVE: ' + info.title if info else 'offline'}")
            except Exception as e:  # noqa: BLE001
                print(f"  {p:8} ERROR: {e}")
        try:
            accts = await PostForMe(sess, s.postforme_api_key, cfg.postforme.base_url).social_accounts()
            print(f"  Post for Me: {len(accts)} social account(s) connected")
        except Exception as e:  # noqa: BLE001
            print(f"  Post for Me ERROR: {e}")


def cmd_check(args):
    from .config import check_secrets, load_config
    cfg = load_config(args.config, args.env)
    missing = check_secrets(cfg)
    print("Missing: " + ", ".join(missing) if missing else "All required settings present.")
    asyncio.run(_check(cfg))


def cmd_accounts(args):
    from .config import load_config
    from .postforme import PostForMe
    cfg = load_config(args.config, args.env)

    async def go():
        async with aiohttp.ClientSession() as sess:
            for a in await PostForMe(sess, cfg.secrets.postforme_api_key, cfg.postforme.base_url).social_accounts():
                print(f"{a.get('id')}\t{a.get('platform')}\t{a.get('username') or a.get('name')}\t{a.get('status', '')}")
    asyncio.run(go())


def cmd_analyze(args):
    from .analysis import HypeDetector, audio_loudness
    from .clips import cut_clip
    from .config import ClipConfig, load_config
    from .recorder import Segment

    if Path(args.config).exists():
        full = load_config(args.config, args.env)
        clip_cfg, seg_len = full.clips, full.recording.segment_seconds
    else:
        clip_cfg, seg_len = ClipConfig(), 600
    if args.threshold is not None:
        clip_cfg.threshold = args.threshold
    if args.mode:
        clip_cfg.mode = args.mode
    seg_len = args.segment or seg_len
    loud = audio_loudness(Path(args.file))
    det = HypeDetector(clip_cfg)
    moments = []
    for s in range(0, len(loud), seg_len):  # feed it chunk by chunk, like a live recording
        det.add_audio(s, loud[s:s + seg_len])
        moments += det.process(None, final=s + seg_len >= len(loud))
    print(f"{len(loud)}s analysed, median loudness {np.median(loud):.1f} dBFS, {len(moments)} hype moment(s):")
    seg = [Segment(Path(args.file).resolve(), 0.0, float(len(loud)))]
    for i, m in enumerate(moments):
        start = max(0, m.t - clip_cfg.pre_roll_seconds)
        line = (f"  {int(m.t) // 3600:02d}:{int(m.t) % 3600 // 60:02d}:{int(m.t) % 60:02d}  "
                f"{m.kind:5}  score {m.score:.1f}")
        if args.cut:
            out = Path(args.cut) / f"clip_{i:03d}.mp4"
            cut_clip(seg, start, min(len(loud), m.t + clip_cfg.post_roll_seconds), out, clip_cfg.vertical)
            line += f"  -> {out}"
        print(line)


def main():
    ap = argparse.ArgumentParser(prog="clipper")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--env", default=".env")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("run")
    sub.add_parser("check")
    sub.add_parser("accounts")
    a = sub.add_parser("analyze")
    a.add_argument("file")
    a.add_argument("--threshold", type=float)
    a.add_argument("--mode", choices=["spikes", "best_per_chunk", "both"])
    a.add_argument("--segment", type=int, help="chunk length in seconds (default: from config)")
    a.add_argument("--cut", metavar="DIR", help="also cut the clips into DIR")
    args = ap.parse_args()
    _setup_logging(args.verbose)
    {"run": cmd_run, "check": cmd_check, "accounts": cmd_accounts, "analyze": cmd_analyze,
     None: cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
