"""Helpers for rendering Plotly figures and sending them to Discord."""

from __future__ import annotations

import asyncio
import io

import discord
import plotly.graph_objects as go

DEFAULT_IMAGE_WIDTH = 1000
DEFAULT_IMAGE_HEIGHT = 650
DEFAULT_IMAGE_SCALE = 2


def figure_to_png_bytes(
    fig: go.Figure,
    *,
    width: int = DEFAULT_IMAGE_WIDTH,
    height: int = DEFAULT_IMAGE_HEIGHT,
    scale: int = DEFAULT_IMAGE_SCALE,
) -> bytes:
    """Render a Plotly figure as PNG bytes."""
    return fig.to_image(format="png", width=width, height=height, scale=scale)


async def send_plotly_figure(
    interaction: discord.Interaction,
    fig: go.Figure,
    filename: str,
) -> None:
    """Render a Plotly figure off the event loop and send it as a Discord file."""
    png_bytes = await asyncio.to_thread(figure_to_png_bytes, fig)
    file = discord.File(io.BytesIO(png_bytes), filename=filename)
    await interaction.followup.send(file=file)
