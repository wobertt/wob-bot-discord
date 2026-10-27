# WobBot

A small Python Discord bot for interactions with the Codeforces API.

## Features

- `/probrat [contest_id]`: Estimate problem ratings using published pre-contest ratings or cached rating history, including before rating changes are released.
- `/solvetimes [contest_id] [problem_id]`: Plot solve times by rating for a specified problem.

## Setup (untested, generated with codex)

Create a virtual environment, install dependencies, and add your server ID to `.env`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Your real `.env` should include:

```bash
APP_ID=...
DISCORD_TOKEN=...
PUBLIC_KEY=...
DISCORD_GUILD_ID=...
```

`PUBLIC_KEY` is kept for your Discord application settings, but this gateway bot only needs `DISCORD_TOKEN` at runtime.

## Run

In foreground:

```bash
python3 -m wob_bot
```

In background:

```bash
nohup python3 -u script.py > script.log 2>&1 < /dev/null &
```

Invite the bot with the `bot` and `applications.commands` OAuth2 scopes. The bot needs permission to send messages and attach files.

## Codeforces rating cache

The bot automatically downloads historical rating changes in the background on startup. It fetches the full rated field of each finished public regular contest, saves progress after each response, and resumes after restarts. The database is stored at `data/codeforces.sqlite3` by default; set `CF_CACHE_PATH` in `.env` to use another persistent location. Keep this file across deployments. No additional dependency is required.

Allow roughly **1.5–3 hours** for the initial download and **1–3 GB** of disk space (about **5 GB free** recommended). These estimates are based on approximately 2,157 finished contests and a compact SQLite storage benchmark; actual time and size depend on API responses and retries. Requests across all commands and the background updater share Codeforces' limit of one request every two seconds. Progress appears in the bot logs as `Rating cache: checked/total contests checked`.

The updater refreshes the contest list and rating changes for contests finished within the last seven days every ten minutes. It also retries older contests without published changes daily. Recent successful downloads are rechecked to capture partial publication and recalculations; older successful history is retained.

`/probrat` prefers a contest's published `oldRating`, falling back to the participant's latest cached rating update strictly before the contest started. Historical fallback waits for the cache to cover earlier finished contests, so a partial download cannot silently substitute an outdated rating. Published ratings remain usable while the history cache warms. Unresolved participants and unrated newcomers are excluded from both the rating inputs and the solve counts. A zero `oldRating` is included only when cached history establishes prior rated participation. The command does not fetch individual user histories.

Every result includes the chat text `Excluded [n] participants missing from cache.` The embed shows how many participants were included, how many historical ratings were used, and whether the cache is warming or the standings are provisional. Without any eligible participants, predictions show `N/A`. Zero-solves and everyone-solves cases show the search boundaries (`>=8000` and `<=-1000`). Estimates use the retained rated participant population and support individual CF/ICPC contests; divisions are calculated separately. The plot commands continue to require published rating changes.

Run the offline tests with:

```bash
python -m pytest -q
```


## Plotly Image Export

Plotly uses Kaleido to export figures to PNG. Kaleido v1 needs a compatible Chrome or Chromium installation. If rendering fails because Chrome is missing, install Chrome locally or run:

```bash
plotly_get_chrome
```

## Adding New Commands

Create a new file in the `wob_bot` directory with a function that returns text or a Plotly `Figure`. Also add a slash-command wrapper in `wob_bot/commands.py`. See `problemratings_command` for an example of a text command and `solvetimes` for an example of a Figure command.
