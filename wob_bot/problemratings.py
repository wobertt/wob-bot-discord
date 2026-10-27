"""Problem difficulty estimates using published and cached pre-contest ratings."""

from __future__ import annotations

from dataclasses import dataclass

import discord
import numpy as np
import pandas as pd
from tabulate import tabulate

from .cf_api import cf_api
from .rating_cache import RatingCache


@dataclass(frozen=True)
class ProblemRatingsResult:
    embed: discord.Embed
    included_count: int
    excluded_count: int

    @property
    def chat_text(self) -> str:
        return f"Excluded {self.excluded_count} participants missing from cache."


def calc_problem_rating(user_ratings: np.ndarray, solve_count: int) -> int | str:
    """Fit expected solves to observed solves, displaying unbounded estimates."""
    if len(user_ratings) == 0:
        return "N/A"
    if solve_count == 0:
        return ">=8000"
    if solve_count == len(user_ratings):
        return "<=-1000"

    def expected_solves(difficulty: int) -> float:
        exponent = np.clip((difficulty - user_ratings) * np.log(10) / 400, -700, 700)
        return float((1 / (1 + np.exp(exponent))).sum())

    lo, hi = -1000, 8000
    if expected_solves(lo) <= solve_count:
        return "<=-1000"
    if expected_solves(hi) >= solve_count:
        return ">=8000"
    while hi - lo > 1:
        test_rating = (lo + hi) // 2
        if solve_count < expected_solves(test_rating):
            lo = test_rating
        else:
            hi = test_rating
    return hi


def problemratings(
    contest_id: int, cache: RatingCache | None = None
) -> ProblemRatingsResult:
    if contest_id <= 0:
        raise ValueError("Contest ID must be positive.")
    contest, problems, participants, problem_results = cf_api.get_contest_standings(
        contest_id
    )
    if contest["phase"] == "BEFORE":
        raise ValueError("This contest has not started yet.")
    if contest["type"] not in ("CF", "ICPC"):
        raise ValueError(
            "Problem ratings support individual CF and ICPC contests only."
        )
    if contest.get("startTimeSeconds") is None:
        raise ValueError("Codeforces has not provided the contest start time.")

    cache = cache if cache is not None else RatingCache()
    cache.record_contest(contest)
    changes = cf_api.get_rating_changes(contest_id)
    cache.store_rating_changes(contest_id, changes)
    published = cache.published_ratings(contest_id)

    participants = participants[participants.participantType == "CONTESTANT"].copy()
    participants["handle_key"] = participants.handle.astype("string").str.casefold()
    participants = participants.drop_duplicates("handle_key")
    history = cache.ratings_before(
        participants.handle_key.tolist(), contest["startTimeSeconds"]
    )

    ratings = {}
    cached_count = 0
    for handle in participants.handle_key:
        if handle in published and (
            published[handle] != 0 or handle in history.ratings
        ):
            ratings[handle] = published[handle]
        elif history.history_ready and handle in history.ratings:
            ratings[handle] = history.ratings[handle]
            cached_count += 1

    results = problem_results.copy()
    results["handle_key"] = results.handle.astype("string").str.casefold()
    solved = results[results.handle_key.isin(ratings) & (results.points > 0)]
    solve_counts = solved.groupby("problem_index").handle_key.nunique()
    data = pd.DataFrame(
        {
            "label": problems.label,
            "solve_count": solve_counts.reindex(problems.index, fill_value=0),
        }
    )
    data["official"] = [
        int(value) if pd.notna(value) else "None"
        for value in problems.get(
            "rating", pd.Series(index=problems.index, dtype=float)
        )
    ]
    user_ratings = np.array(list(ratings.values()), dtype=float)
    data["predicted"] = data.solve_count.map(
        lambda count: calc_problem_rating(user_ratings, count)
    )
    data = data.set_index("label").rename_axis("problem")

    description = (
        f"```{tabulate(data, headers=[data.index.name] + list(data.columns))}```"
    )
    if not ratings:
        description += (
            "\nNo participants with usable pre-contest ratings are available."
        )
    embed = discord.Embed(
        title=contest["name"],
        url=f"https://codeforces.com/contest/{contest_id}",
        description=description,
    )
    notes = [
        f"Based on {len(ratings)} participants",
        f"{cached_count} cached historical ratings",
    ]
    if not history.history_ready:
        notes.append("History cache warming")
    if (
        contest["phase"] != "FINISHED"
        or contest.get("frozen", False)
        or (
            not results.empty
            and "type" in results
            and (results["type"] == "PRELIMINARY").any()
        )
    ):
        notes.append("Provisional standings")
    embed.set_footer(text=" · ".join(notes))
    return ProblemRatingsResult(embed, len(ratings), len(participants) - len(ratings))


if __name__ == "__main__":
    result = problemratings(2033)
    print(result.chat_text)
    print(result.embed.description)
