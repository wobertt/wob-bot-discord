"""Slash command registration for the bot."""

from __future__ import annotations

import asyncio
import logging

import discord
from discord import app_commands
from discord.ext import commands

from .problemratings import problemratings
from .plotting import send_plotly_figure
from .solvetimes import solvetimes

logger = logging.getLogger(__name__)


def register_commands(bot: commands.Bot) -> None:
    """Register slash commands on the bot's command tree."""

    @bot.tree.command(
        name="probrat",
        description="Show predicted problem ratings for a contest.",
    )
    @app_commands.describe(contest_id="Contest ID.")
    async def problemratings_command(
        interaction: discord.Interaction, contest_id: int
    ) -> None:
        await interaction.response.defer(thinking=True)

        try:
            res = await asyncio.to_thread(problemratings, contest_id)
            await interaction.followup.send(res)

        except Exception:
            logger.exception(
                "Failed to calculate problem ratings for contest_id=%s.", contest_id
            )
            await interaction.followup.send(
                f"Error: Failed to calculate problem ratings for contest_id={contest_id}",
            )

    @bot.tree.command(
        name="solvetimes",
        description="Show solve times by rating for a contest problem.",
    )
    @app_commands.rename(contest_id="contest-id", problem_name="problem-name")
    @app_commands.describe(
        contest_id="Contest ID.",
        problem_name="Problem name (e.g., A, B, C1).",
    )
    async def solvetimes_command(
        interaction: discord.Interaction,
        contest_id: int,
        problem_name: str,
    ) -> None:
        await interaction.response.defer(thinking=True)

        try:
            fig = await asyncio.to_thread(solvetimes, contest_id, problem_name)
            await send_plotly_figure(
                interaction,
                fig,
                filename=f"solvetimes-{contest_id}{problem_name}.png",
            )
        except Exception:
            logger.exception(
                "Failed to get solve times for contest_id=%s problem_name=%s.",
                contest_id,
                problem_name,
            )
            await interaction.followup.send(
                f"Failed to get solve times for contest_id={contest_id} problem_name={problem_name}.",
            )
