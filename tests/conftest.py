import pandas as pd
import pytest

from wob_bot.rating_cache import RatingCache


def contest(contest_id=1, start=100, phase="FINISHED"):
    return {
        "id": contest_id,
        "name": f"Contest {contest_id}",
        "type": "CF",
        "phase": phase,
        "startTimeSeconds": start,
        "durationSeconds": 10,
    }


def changes(*rows):
    return pd.DataFrame(
        rows, columns=["handle", "oldRating", "newRating", "ratingUpdateTimeSeconds"]
    )


@pytest.fixture
def cache(tmp_path):
    return RatingCache(tmp_path / "ratings.sqlite3")
