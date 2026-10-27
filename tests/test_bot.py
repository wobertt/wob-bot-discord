import asyncio
from unittest.mock import AsyncMock

import discord
from discord.ext import commands

from wob_bot.bot import WobBot
from wob_bot.commands import register_commands
from wob_bot.problemratings import ProblemRatingsResult
import wob_bot.commands as command_module


def test_command_sends_exact_chat_text_with_embed(cache, monkeypatch):
    async def run():
        bot = commands.Bot(command_prefix="!", intents=discord.Intents.none())
        register_commands(bot, cache)
        embed = discord.Embed(title="Contest")
        monkeypatch.setattr(
            command_module,
            "problemratings",
            lambda *args: ProblemRatingsResult(embed, 10, 3),
        )
        interaction = AsyncMock()
        await bot.tree.get_command("probrat").callback(interaction, 1)
        interaction.response.defer.assert_awaited_once_with(thinking=True)
        interaction.followup.send.assert_awaited_once_with(
            content="Excluded 3 participants missing from cache.", embed=embed
        )
        await bot.close()

    asyncio.run(run())


def test_command_reports_expected_errors(cache, monkeypatch):
    async def run():
        bot = commands.Bot(command_prefix="!", intents=discord.Intents.none())
        register_commands(bot, cache)

        def invalid(*args):
            raise ValueError("This contest has not started yet.")

        monkeypatch.setattr(command_module, "problemratings", invalid)
        interaction = AsyncMock()
        await bot.tree.get_command("probrat").callback(interaction, 1)
        interaction.followup.send.assert_awaited_once_with(
            "This contest has not started yet."
        )
        await bot.close()

    asyncio.run(run())


def test_bot_starts_one_worker_and_cancels_it_on_close(cache, monkeypatch):
    async def run():
        monkeypatch.delenv("DISCORD_GUILD_ID", raising=False)
        bot = WobBot(rating_cache=cache)
        monkeypatch.setattr(bot.tree, "sync", AsyncMock(return_value=[]))
        started = asyncio.Event()

        async def update():
            started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(bot._cache_updater, "run", update)
        await bot.setup_hook()
        task = bot._cache_task
        await started.wait()
        await bot.setup_hook()
        assert task is bot._cache_task
        await bot.close()
        assert task.cancelled()
        assert bot._cache_updater.cancel_event.is_set()

    asyncio.run(run())
