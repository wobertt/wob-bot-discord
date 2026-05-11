"""Plotly chart builders used by Discord slash commands."""

from __future__ import annotations

import plotly.graph_objects as go


def problemratings(id: int) -> go.Figure:
    """Return a placeholder problem rating distribution chart."""
    ratings = [800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]
    counts = [3, 4, 7, 6, 9, 8, 5, 4, 2]

    fig = go.Figure(
        data=[
            go.Bar(
                x=ratings,
                y=counts,
                marker_color="#5865F2",
                hovertemplate="Rating %{x}<br>Problems %{y}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=f"Problem Ratings for {id}",
        xaxis_title="Rating",
        yaxis_title="Number of problems",
        template="plotly_white",
        bargap=0.25,
        margin=dict(l=80, r=40, t=90, b=80),
    )
    fig.update_xaxes(type="category")

    return fig
