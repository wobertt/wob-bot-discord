"""Slash command registration for the bot."""

from __future__ import annotations

import asyncio
import io
import logging
import random
import re

import discord
from discord import app_commands
from discord.ext import commands

from .fetchuser import RANK_COLORS, fetchuser
from .problemratings import problemratings
from .cf_api.cf_api import CodeforcesApiError
from .performance import perf, perfgraph
from .plotting import send_plotly_figure
from .rating_cache import RatingCache
from .solve_info import solvetimes, solvechance

logger = logging.getLogger(__name__)


def register_commands(bot: commands.Bot, rating_cache: RatingCache) -> None:
    """Register slash commands on the bot's command tree."""

    @bot.tree.command(name="fetchuser", description="Find a Codeforces user at or near a rating.")
    @app_commands.describe(rating="Target current Codeforces rating.")
    async def fetchuser_command(interaction: discord.Interaction, rating: int) -> None:
        await interaction.response.defer(thinking=True)
        try:
            embed = await asyncio.to_thread(fetchuser, rating, rating_cache)
            await interaction.followup.send(embed=embed)
        except (CodeforcesApiError, ValueError) as exc:
            await interaction.followup.send(str(exc))
        except Exception:
            logger.exception("Failed to fetch a user for rating=%s.", rating)
            await interaction.followup.send("Failed to find a Codeforces user. Please try again later.")

    @bot.tree.command(
        name="perf",
        description="Show a user's performance in a contest (rated or virtual).",
    )
    @app_commands.rename(contest_id="contest-id")
    @app_commands.describe(handle="Codeforces handle.", contest_id="Contest ID.")
    async def perf_command(
        interaction: discord.Interaction, handle: str, contest_id: int
    ) -> None:
        await interaction.response.defer(thinking=True)
        try:
            embed = await asyncio.to_thread(perf, handle, contest_id, rating_cache)
            await interaction.followup.send(embed=embed)
        except (CodeforcesApiError, ValueError) as exc:
            await interaction.followup.send(str(exc))
        except Exception:
            logger.exception(
                "Failed to calculate performance for handle=%s contest_id=%s.",
                handle,
                contest_id,
            )
            await interaction.followup.send(
                f"Failed to calculate performance for {handle} in contest {contest_id}."
            )

    @bot.tree.command(
        name="perfgraph",
        description="Plot a user's contest performances over time.",
    )
    @app_commands.rename(date_from="from", date_to="to")
    @app_commands.describe(
        handle="Codeforces handle.",
        include="Which participations to plot (default: both).",
        date_from="Start date, inclusive: yyyy, mmyyyy or ddmmyyyy.",
        date_to="End date, exclusive: yyyy, mmyyyy or ddmmyyyy.",
    )
    @app_commands.choices(
        include=[
            app_commands.Choice(name="Rated + virtual", value="both"),
            app_commands.Choice(name="Rated only", value="rated"),
            app_commands.Choice(name="Virtual only", value="virtual"),
        ]
    )
    async def perfgraph_command(
        interaction: discord.Interaction,
        handle: str,
        include: app_commands.Choice[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> None:
        await interaction.response.defer(thinking=True)
        try:
            result = await asyncio.to_thread(
                perfgraph,
                handle,
                rating_cache,
                include.value if include else "both",
                date_from,
                date_to,
            )
            filename = f"perfgraph-{re.sub(r'[^A-Za-z0-9_.-]', '_', handle)}.png"
            # Same presentation as TLE: the graph inside a random CF-colored embed.
            embed = discord.Embed(
                title="Performance graph on Codeforces",
                description=result.note or None,
                color=random.choice(list(RANK_COLORS.values())),
            )
            embed.set_image(url=f"attachment://{filename}")
            await interaction.followup.send(
                embed=embed, file=discord.File(io.BytesIO(result.png), filename=filename)
            )
        except (CodeforcesApiError, ValueError) as exc:
            await interaction.followup.send(str(exc))
        except Exception:
            logger.exception("Failed to plot performance for handle=%s.", handle)
            await interaction.followup.send(
                f"Failed to plot performance for {handle}. Please try again later."
            )

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
            result = await asyncio.to_thread(problemratings, contest_id, rating_cache)
            await interaction.followup.send(
                content=result.chat_text, embed=result.embed
            )

        except (CodeforcesApiError, ValueError) as exc:
            await interaction.followup.send(str(exc))

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

    @bot.tree.command(
        name="solvechance",
        description="Show solve chance by rating for a contest problem.",
    )
    @app_commands.rename(contest_id="contest-id", problem_name="problem-name")
    @app_commands.describe(
        contest_id="Contest ID.",
        problem_name="Problem name (e.g., A, B, C1).",
    )
    async def solvechance_command(
        interaction: discord.Interaction,
        contest_id: int,
        problem_name: str,
    ) -> None:
        await interaction.response.defer(thinking=True)

        try:
            fig = await asyncio.to_thread(solvechance, contest_id, problem_name)
            await send_plotly_figure(
                interaction,
                fig,
                filename=f"solvechance-{contest_id}{problem_name}.png",
            )
        except Exception:
            logger.exception(
                "Failed to get solve chance for contest_id=%s problem_name=%s.",
                contest_id,
                problem_name,
            )
            await interaction.followup.send(
                f"Failed to get solve chance for contest_id={contest_id} problem_name={problem_name}.",
            )
