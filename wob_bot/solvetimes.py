"""Solve-count lookups used by Discord slash commands."""

from __future__ import annotations
from .cf_api import cf_api
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from math import isqrt


def solvetimes(contest_id: int, problem_name: str) -> go.Figure:
    """Plot solve times by rating."""
    _, problems, _, problem_results = cf_api.get_contest_standings(contest_id)
    rating_changes = cf_api.get_rating_changes(contest_id)

    problem_index = problems.index[problems.label == problem_name][0]
    results_for = problem_results[problem_results.problem_index == problem_index]

    time_data = pd.merge(results_for, rating_changes, on="handle")[
        ["handle", "oldRating", "bestSubmissionTimeSeconds"]
    ]
    time_data = time_data.rename(
        columns=dict(oldRating="rating", bestSubmissionTimeSeconds="solve_time")
    )
    time_data = time_data[time_data.solve_time.notna()]
    time_data["solve_time"] /= 60.0

    time_data["rating_bin"] = pd.qcut(time_data.rating, q=isqrt(time_data.shape[0]))

    grouped_data = time_data.groupby("rating_bin").agg(
        rating=("rating", "median"),
        p25=("solve_time", lambda r: r.quantile(0.25)),
        p50=("solve_time", lambda r: r.quantile(0.50)),
        p75=("solve_time", lambda r: r.quantile(0.75)),
    )

    return px.line(
        grouped_data,
        x="rating",
        y=["p25", "p50", "p75"],
        title=f"Solve time by rating for {contest_id}{problem_name}",
    ).update_layout(
        xaxis_title="Rating",
        yaxis_title="Solve time (mins)",
        legend_title_text="Percentile",
    )
