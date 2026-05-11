"""Entrypoint for running the Discord bot with ``python -m wob_bot``."""

from __future__ import annotations

import asyncio
import logging
import os
import ssl

import aiohttp
import certifi
from dotenv import load_dotenv

from .bot import WobBot


def configure_logging() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def main() -> None:
    load_dotenv()
    configure_logging()

    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_TOKEN is required in your environment or .env file.")

    asyncio.run(run_bot(token))


async def run_bot(token: str) -> None:
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(limit=0, ssl=ssl_context)
    bot = WobBot(connector=connector)

    async with bot:
        await bot.start(token)


if __name__ == "__main__":
    main()
