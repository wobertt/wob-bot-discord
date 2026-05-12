# WobBot

A small Python Discord bot for interactions with the Codeforces API.

## Features

- `/probrat [contest_id]`: Obtain estimated problem ratings for a past contest. Rating changes must have been released.
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

```bash
python -m wob_bot
```

Invite the bot with the `bot` and `applications.commands` OAuth2 scopes. The bot needs permission to send messages and attach files.


## Plotly Image Export

Plotly uses Kaleido to export figures to PNG. Kaleido v1 needs a compatible Chrome or Chromium installation. If rendering fails because Chrome is missing, install Chrome locally or run:

```bash
plotly_get_chrome
```

## Adding New Commands

1. Create a new file in the `wob_bot` directory that returns text or a Plotly `Figure`.
2. Add a slash-command wrapper in `wob_bot/commands.py`.
3. In the wrapper, defer the interaction, call your plotting function, and pass the figure to `send_plotly_figure`.

In `wob_bot/commands.py`, see `problemratings_command` for an example of a text command
and `solvetimes` for an example of a Figure command.