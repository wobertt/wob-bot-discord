# WobBot

A small Python Discord bot for interactions with the Codeforces API.

## Features

- `/probrat [contest_id]`: Obtain estimated problem ratings for a past contest. Rating changes must have been released.

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

```bash
python -m wob_bot
```

Invite the bot with the `bot` and `applications.commands` OAuth2 scopes. The bot needs permission to send messages and attach files.

Run `/problemratings 2033` in your server. The current implementation returns a placeholder chart from `wob_bot.charts.problemratings`.

Run `/solves contest-id:2033 problem-name:A` to get a placeholder solve count from `wob_bot.solves.get_problem_solves`.

## TLS Certificates

The bot configures `discord.py` to use Certifi's trusted CA bundle. This avoids a common macOS `CERTIFICATE_VERIFY_FAILED` issue with Python.org installs while keeping normal HTTPS certificate verification enabled.

## Plotly Image Export

Plotly uses Kaleido to export figures to PNG. Kaleido v1 needs a compatible Chrome or Chromium installation. If rendering fails because Chrome is missing, install Chrome locally or run:

```bash
plotly_get_chrome
```

## Add A New Chart Command

1. Add a pure function in `wob_bot/charts.py` that returns a Plotly `Figure`.
2. Add a slash-command wrapper in `wob_bot/commands.py`.
3. In the wrapper, defer the interaction, call your plotting function, and pass the figure to `send_plotly_figure`.

For text commands, follow the `/solves` pattern: write a pure lookup function, then add a slash-command wrapper that defers, calls the function, and sends a text response.

## Test

```bash
pytest
```
