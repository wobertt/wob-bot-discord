"""Slash command registration for the bot."""

from __future__ import annotations

import asyncio
import logging

import discord
from discord import app_commands
from discord.ext import commands

from .problemratings import problemratings
from .plotting import send_plotly_figure
from .solves import get_problem_solves

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

"""
    @bot.tree.command(
        name="solves",
        description="Show the number of solves for a contest problem.",
    )
    @app_commands.rename(contest_id="contest-id", problem_name="problem-name")
    @app_commands.describe(
        contest_id="Contest ID containing the problem.",
        problem_name="Problem name, letter, or slug.",
    )
    async def solves_command(
        interaction: discord.Interaction,
        contest_id: int,
        problem_name: str,
    ) -> None:
        await interaction.response.defer(thinking=True)

        try:
            # TODO: complete this command.
            # this is how you send an image
            # fig = await asyncio.to_thread(problemratings, id)
            # await send_plotly_figure(
            #     interaction,
            #     fig,
            #     filename=f"problemratings-{id}.png",
            # )

            solves = await asyncio.to_thread(
                get_problem_solves, contest_id, problem_name
            )
            await interaction.followup.send(
                f"{problem_name} in contest {contest_id} has {solves} solves."
            )
        except Exception:
            logger.exception(
                "Failed to look up solves for contest_id=%s problem_name=%s.",
                contest_id,
                problem_name,
            )
            await interaction.followup.send(
                "Sorry, I couldn't look up solves for that problem. Please try again in a minute."
            )
"""