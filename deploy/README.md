# Cattle Auction on Coolify

Use [coolify.compose.yaml](coolify.compose.yaml) as a raw Docker Compose service.
The intended placement is Oracle, in a dedicated `cattle-auction` project and
`development` environment, on the server's `coolify` Docker destination.

The worker waits for manual CLI jobs. It exposes no ports and keeps media,
checkpoints, JSON results, and the optional SQLite database in a named volume
at `/data`. Keep the same Coolify service and volume when upgrading.

## Release and credentials

The defaults build tag `v2.0.0` at commit
`fb8b1d7a6a822b86255840888b19d9fd14314235`. The build checks the tag against
the expected commit and installs the frozen lockfile. The runtime image smoke
check imports OpenCV, SQLite and yt-dlp EJS, initializes RapidOCR, and verifies
ffmpeg, Deno and the CLI.

To upgrade, set both `CATTLE_AUCTION_RELEASE` and `CATTLE_AUCTION_REVISION` on
the service to the published tag and its full commit SHA before rebuilding.
Set `OPENAI_API_KEY` and `GROQ_API_KEY` as runtime secrets for the default run.
Set `OPENROUTER_API_KEY` if you use `--provider openrouter`. Keep secret values
in Coolify.

## Run on Oracle

Check the worker and available storage from its Coolify terminal:

```sh
python /app/main.py --help
test -w /data
df -h /data
```

Before a paid run, check that Oracle can retrieve formats from a short
authorized YouTube video. This check makes no Groq or OpenAI calls.

```sh
python -m yt_dlp --no-remote-components --js-runtimes deno --list-formats 'YOUTUBE_URL'
```

Run one auction at a time. The lock prevents two jobs in this worker from
writing the same checkpoints concurrently.

```sh
flock -n /data/.auction.lock python /app/main.py --output-dir /data 'YOUTUBE_URL'
```

Use a terminal session that stays open for a long job. If interrupted, run the
same command again to reuse compatible checkpoints. Results are written to
`/data/<video_id>/result_<video_id>.json`.

Select v2.1.0 or a later release to use SQLite history. Run the CLI with
`--sqlite-db /data/auction-history.sqlite3` to add results to the same database
across jobs. Confirm the flag appears in `python /app/main.py --help` after
upgrading. The `v2.0.0` default writes JSON and checkpoints.

The Compose health check covers the worker's local binaries and output mount.
YouTube downloads and provider access need their own checks on Oracle.
