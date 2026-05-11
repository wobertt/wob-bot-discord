from __future__ import annotations

import plotly.graph_objects as go

from wob_bot.charts import problemratings
from wob_bot.plotting import figure_to_png_bytes


def test_problemratings_returns_plotly_figure() -> None:
    fig = problemratings(2033)

    assert isinstance(fig, go.Figure)


def test_figure_to_png_bytes_returns_png() -> None:
    fig = problemratings(2033)

    png_bytes = figure_to_png_bytes(fig, width=400, height=260, scale=1)

    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
