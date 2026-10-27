"""Discord bot setup and slash command synchronization."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import logging
import os
from typing import Any

import discord
from discord.ext import commands

from .commands import register_commands
from .rating_cache import RatingCache, RatingCacheUpdater

logger = logging.getLogger(__name__)


class WobBot(commands.Bot):
    """Bot configured for slash commands and guild-first command sync."""

    def __init__(
        self, *, rating_cache: RatingCache | None = None, **options: Any
    ) -> None:
        super().__init__(
            command_prefix=commands.when_mentioned,
            intents=discord.Intents.default(),
            **options,
        )
        self._registered_local_commands = False
        self.rating_cache = rating_cache if rating_cache is not None else RatingCache()
        self._cache_updater = RatingCacheUpdater(self.rating_cache)
        self._cache_task: asyncio.Task | None = None

    async def setup_hook(self) -> None:
        if not self._registered_local_commands:
            register_commands(self, self.rating_cache)
            self._registered_local_commands = True

        if self._cache_task is None:
            self._cache_task = asyncio.create_task(
                self._cache_updater.run(), name="codeforces-rating-cache"
            )

        guild_id = os.getenv("DISCORD_GUILD_ID")
        if guild_id:
            guild = self._guild_object_from_env(guild_id)
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            logger.info(
                "Synced %s slash command(s) to guild %s.", len(synced), guild_id
            )
            return

        synced = await self.tree.sync()
        logger.warning(
            "DISCORD_GUILD_ID is not set; synced %s global command(s). "
            "Global commands can take up to an hour to appear.",
            len(synced),
        )

    async def close(self) -> None:
        self._cache_updater.stop()
        if self._cache_task is not None:
            self._cache_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._cache_task
        await super().close()

    async def on_ready(self) -> None:
        if self.user is None:
            logger.info("Bot is ready.")
            return

        logger.info("Logged in as %s (ID: %s).", self.user, self.user.id)

    @staticmethod
    def _guild_object_from_env(guild_id: str) -> discord.Object:
        try:
            return discord.Object(id=int(guild_id))
        except ValueError as exc:
            raise RuntimeError(
                "DISCORD_GUILD_ID must be an integer Discord server ID."
            ) from exc
