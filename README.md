# stream-clipper

Watches Twitch, YouTube and Kick streamers, records them with Streamlink in 10-minute
chunks, detects hype moments from chat and audio, cuts clips, and sends them to Discord
for approval. Approved clips are posted through the Post for Me API.

```
poll every 2 min ──► live? ──► Discord "🔴 X is live" + start Streamlink │ ffmpeg (10-min .ts chunks)
                                 + chat listener (Twitch IRC / Kick Pusher / YouTube API)
each finished chunk ──► audio loudness/sec + chat msgs/sec ──► robust z-scores vs trailing 10 min
   ──► peak above threshold ──► clip [peak-40s, peak+20s] (crosses chunk boundaries)
   ──► Discord post with preview + [Approve & post] [Edit caption] [Reject]
Approve ──► Post for Me: upload video ──► create social post on your connected accounts
```

## Requirements

- Python 3.11+
- `ffmpeg` on PATH (`apt install ffmpeg` / `brew install ffmpeg` / `winget install ffmpeg`)
- Streamlink (installed by `requirements.txt`)
- Enough disk: a 1080p60 stream is roughly 2.5–4 GB/hour. Raw chunks are deleted after `retention_hours`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # put your API keys here
cp config.example.yaml config.yaml # streamers, Discord channel ids, tuning
```

### Keys (`.env`)

| Variable | Where to get it |
|---|---|
| `TWITCH_CLIENT_ID`, `TWITCH_CLIENT_SECRET` | dev.twitch.tv/console/apps → Register app (redirect URL `http://localhost`) |
| `YOUTUBE_API_KEY` | console.cloud.google.com → enable *YouTube Data API v3* → Credentials → API key |
| `KICK_CLIENT_ID`, `KICK_CLIENT_SECRET` | kick.com → Settings → Developer → create app |
| `DISCORD_BOT_TOKEN` | discord.com/developers/applications → New app → Bot → Reset token. Invite with OAuth2 URL generator: scope `bot`, permissions *View Channels, Send Messages, Attach Files, Read Message History* |
| `POSTFORME_API_KEY` | postforme.dev dashboard → API keys. Connect your TikTok/YouTube/Instagram/X accounts there first. |

You only need keys for platforms you actually list in `config.yaml`.

### Discord channel ids

Discord → Settings → Advanced → Developer Mode on, then right-click a channel → *Copy Channel ID*.
Put them in `discord.notify_channel_id` and `discord.approval_channel_id` (they can be the same channel).

## Run

```bash
python -m clipper check      # validates config, tests every API login, shows who is live
python -m clipper accounts   # lists Post for Me social account ids (optional: pin them in config)
python -m clipper run        # start the pipeline (Ctrl+C to stop)
```

To keep it running permanently on Linux, use a systemd user service:

```ini
# ~/.config/systemd/user/stream-clipper.service
[Unit]
Description=stream-clipper
After=network-online.target

[Service]
WorkingDirectory=%h/stream-clipper
ExecStart=%h/stream-clipper/.venv/bin/python -m clipper run
Restart=always

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now stream-clipper && loginctl enable-linger $USER
```

## Tuning hype detection

Try settings offline on any recording first (audio only, no chat):

```bash
python -m clipper analyze data/recordings/twitch_x/20261124_200000/chunk_00003.ts --threshold 3 --cut /tmp/test-clips
```

- More or fewer clips: change `clips.threshold` (default 3.5; try 2.5–5).
- Chat-heavy streamers: raise `chat_weight`. Loud-music streamers: lower `audio_weight`.
- Clips are only cut once a 10-minute chunk finishes, so a clip arrives up to ~10 minutes
  after the moment. Lower `segment_seconds` if you want them sooner.

## Notes and limits

- **YouTube quota:** the live check uses ~2 units per channel per poll, which fits the 10,000/day
  quota for about 6 channels at a 2-minute poll interval. Chat uses `liveChatMessages`
  (5 units per call, every `youtube_chat_min_interval` s). Request a quota increase if you need more.
- **Kick chat** uses Kick's Pusher websocket and an unofficial endpoint to find the chatroom id.
  If that lookup is blocked, set `kick_chatroom_id` for the streamer (find it in the browser dev tools
  network tab on the channel page). Without chat the detector uses audio alone.
- **Discord previews** are re-encoded at 480p to fit `upload_limit_mb`. The full-quality clip is what gets posted.
- Clip state lives in `data/clipper.db`, so the Approve buttons keep working after a restart.
- Only clip and repost streams you have the rights to (your own channel, or with the streamer's permission).
