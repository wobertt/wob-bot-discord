"""Logic for the problemratings command."""

from __future__ import annotations
import pandas as pd
import numpy as np
from tabulate import tabulate
from .cf_api import cf_api
import discord


def problemratings(contest_id: int) -> discord.Embed:
    """
    Example: output of problemratings(2033)

           solve_count  predicted_rating
    label
    A            13710               778
    B            11197               925
    C             3872              1366
    D             4325              1332
    E             3568              1391
    F              598              1799
    G               15              2459
    """

    def calc_problem_rating(user_ratings: pd.Series, solve_count: int) -> int:
        """Calculate the rating of a problem."""

        lo, hi = -1000, 8000  # binary search. problem rating in (lo, hi]
        while hi - lo > 1:
            test_rating = (lo + hi) // 2
            expected_solves = (
                1 / (1 + np.exp(-(user_ratings - test_rating) / 400 * np.log(10)))
            ).sum()
            if solve_count < expected_solves:
                lo = test_rating
            else:
                hi = test_rating
        return hi

    contest, problems, _, problem_results = cf_api.get_contest_standings(contest_id)
    rating_changes = cf_api.get_rating_changes(contest_id)

    data = pd.DataFrame(
        dict(
            label=problems.label,
            solve_count=problem_results.groupby(
                "problem_index"
            ).bestSubmissionTimeSeconds.count(),  # field is NaN iff unsolved by that handle
        )
    )
    if "rating" in problems.columns:
        data["official"] = problems.rating
    else:
        data["official"] = "None"
    data["predicted"] = data.solve_count.map(
        lambda cnt: calc_problem_rating(rating_changes.oldRating, cnt)
    )

    data = data.set_index("label").rename_axis("problem")

    return discord.Embed(
        title=contest["name"],
        url=f"https://codeforces.com/contest/{contest_id}",
        description=f"```{tabulate(data, headers=[data.index.name] + list(data.columns))}```",
    )


if __name__ == "__main__":
    # test
    print(problemratings(2033))
