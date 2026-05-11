"""Discord bot setup and slash command synchronization."""

from __future__ import annotations

import logging
import os
from typing import Any

import discord
from discord.ext import commands

from .commands import register_commands

logger = logging.getLogger(__name__)


class WobBot(commands.Bot):
    """Bot configured for slash commands and guild-first command sync."""

    def __init__(self, **options: Any) -> None:
        super().__init__(
            command_prefix=commands.when_mentioned,
            intents=discord.Intents.default(),
            **options,
        )
        self._registered_local_commands = False

    async def setup_hook(self) -> None:
        if not self._registered_local_commands:
            register_commands(self)
            self._registered_local_commands = True

        guild_id = os.getenv("DISCORD_GUILD_ID")
        if guild_id:
            guild = self._guild_object_from_env(guild_id)
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            logger.info("Synced %s slash command(s) to guild %s.", len(synced), guild_id)
            return

        synced = await self.tree.sync()
        logger.warning(
            "DISCORD_GUILD_ID is not set; synced %s global command(s). "
            "Global commands can take up to an hour to appear.",
            len(synced),
        )

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
            raise RuntimeError("DISCORD_GUILD_ID must be an integer Discord server ID.") from exc
