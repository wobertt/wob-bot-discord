import gzip
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wob_bot import performance as module
from wob_bot.performance import (
    FetchBudget,
    RatingCalculator,
    _virtual_rank_key,
    group_virtuals,
    parse_cf_date,
    virtual_score,
)

from conftest import contest

FIXTURE = Path(__file__).parent / "data" / "carrot-round-1342-edu-86.json.gz"


def test_calculator_matches_carrot_fixture():
    # Real Codeforces deltas and true performances from carrot's test data.
    with gzip.open(FIXTURE, "rt") as file:
        data = json.load(file)
    rows = data["rows"]
    real = {
        handle: math.inf if value == "Infinity" else value
        for handle, value in data["perfs"]
    }
    calculator = RatingCalculator(
        [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows]
    )
    assert (calculator.deltas == np.array([r[4] for r in rows])).all()
    perfs = calculator.performance(np.arange(len(rows)))
    for row, value in zip(rows, perfs):
        expected = real[row[0]]
        if math.isinf(expected):
            assert math.isinf(value)
        else:
            assert 0 <= value - expected <= 4


def test_ties_take_the_last_position_of_their_group():
    calculator = RatingCalculator([3, 2, 2, 1], [0, 5, 5, 0], [1500] * 4)
    assert calculator.rank.tolist() == [1, 3, 3, 4]
    assert math.isinf(calculator.performance(0)[0])
    better, tied = calculator.performance([1, 3])
    assert better > tied


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2023", (2023, 1, 1)),
        ("052023", (2023, 5, 1)),
        ("15052023", (2023, 5, 15)),
        ("15-05-2023", (2023, 5, 15)),
        ("05/2023", (2023, 5, 1)),
        (" 1.5.2023 ", (2023, 5, 1)),
    ],
)
def test_parse_cf_date(text, expected):
    assert parse_cf_date(text) == datetime(*expected, tzinfo=timezone.utc).timestamp()


@pytest.mark.parametrize("text", ["", "23", "13202023", "31022023", "2023-05", "abc"])
def test_parse_cf_date_rejects_invalid(text):
    with pytest.raises(ValueError, match="yyyy"):
        parse_cf_date(text)


def submission(index, verdict, seconds, *, passed=1, start=1000, contest_id=7,
               kind="VIRTUAL", sub_id=0):
    return {
        "id": sub_id or seconds,
        "contestId": contest_id,
        "relativeTimeSeconds": seconds,
        "verdict": verdict,
        "passedTestCount": passed,
        "problem": {"index": index},
        "author": {
            "participantType": kind,
            "startTimeSeconds": start,
            "members": [{"handle": "Virt"}],
        },
    }


def test_group_virtuals_splits_runs_and_ignores_other_types():
    subs = [
        submission("A", "OK", 60, start=1),
        submission("A", "OK", 60, start=2),
        submission("A", "OK", 60, kind="PRACTICE"),
        submission("A", "OK", 60, contest_id=100_001),
    ]
    groups = group_virtuals(subs)
    assert list(groups) == [7]
    assert sorted(groups[7]) == [1, 2]


def icpc_official():
    participants = pd.DataFrame(
        {"handle": ["x", "y"], "points": [2.0, 1.0], "penalty": [130, 10]}
    )
    results = pd.DataFrame(
        {
            "handle": ["x", "x", "y", "y"],
            "problem_index": [0, 1, 0, 1],
            "points": [1.0, 1.0, 1.0, 0.0],
            "rejectedAttemptCount": [1, 0, 0, 3],
            "bestSubmissionTimeSeconds": [600, 6000, 600, np.nan],
        }
    )
    return participants, results


def test_icpc_virtual_score_infers_penalty_and_skips_unpenalized_attempts():
    participants, results = icpc_official()  # x: 10 + 100 + 20*1 -> 20 per wrong
    problems = pd.DataFrame({"label": ["A", "B"]})
    subs = [
        submission("A", "WRONG_ANSWER", 100, passed=3),
        submission("A", "WRONG_ANSWER", 110, passed=0),  # failed test 1
        submission("A", "COMPILATION_ERROR", 120, passed=0),
        submission("A", "OK", 300),
        submission("A", "OK", 400),
        submission("B", "WRONG_ANSWER", 500, passed=2),
        submission("B", "OK", 20000),  # after the contest
    ]
    metadata = contest(7) | {"type": "ICPC", "durationSeconds": 7200}
    assert virtual_score(metadata, problems, participants, results, subs) == (1, 5 + 20)


def test_cf_virtual_score_uses_fitted_decay_and_last_accepted_submission():
    problems = pd.DataFrame({"label": ["A"], "points": [500.0]})
    participants = pd.DataFrame({"handle": ["x"], "points": [480.0], "penalty": [0]})
    results = pd.DataFrame(
        {
            "handle": ["x"],
            "problem_index": [0],
            "points": [480.0],
            "rejectedAttemptCount": [0],
            "bestSubmissionTimeSeconds": [600],
        }
    )  # 20 points over 10 minutes -> 2 per minute
    subs = [
        submission("A", "OK", 300),
        submission("A", "WRONG_ANSWER", 400, passed=2),
        submission("A", "OK", 1200),
    ]
    metadata = contest(7) | {"type": "CF", "durationSeconds": 7200}
    points, penalty = virtual_score(metadata, problems, participants, results, subs)
    assert (points, penalty) == (500 - 2 * 20 - 50 * 2, 0)


def test_virtual_rank_key_ties_and_interpolates_hidden_rows():
    visible = pd.DataFrame(
        {"points": [5.0, 4.0, 4.0, 2.0], "penalty": [10, 20, 50, 0],
         "cf_rank": [3, 10, 20, 40]}
    )
    assert _virtual_rank_key(visible, 4.0, 20) == 10
    assert _virtual_rank_key(visible, 4.0, 30) == (10 + 20) / 2 + 0.25
    assert _virtual_rank_key(visible, 9.0, 0) == (0 + 3) / 2 + 0.25
    assert _virtual_rank_key(visible, 0.0, 0) == 40.5


def seed_contest(cache, contest_id=2000, *, start=200, rows=None):
    cache.store_contests([contest(contest_id, start)])
    frame = pd.DataFrame(
        rows or [
            ("Alice", 1, start + 10, 1500, 1600),
            ("bob", 2, start + 10, 1500, 1500),
            ("carol", 3, start + 10, 1500, 1400),
        ],
        columns=["handle", "rank", "ratingUpdateTimeSeconds", "oldRating", "newRating"],
    )
    cache.store_rating_changes(contest_id, frame)


def test_rated_perf_uses_cached_field_and_memoizes(cache, monkeypatch):
    seed_contest(cache)
    monkeypatch.setattr(
        module.cf_api, "get_rating_changes",
        lambda _: pytest.fail("ranks are cached; no fetch expected"),
    )
    value = module.rated_perf("ALICE", 2000, cache)
    assert math.isinf(value)
    middle = module.rated_perf("bob", 2000, cache)
    assert 1400 < middle < 1600
    assert cache.get_rated_perf(2000, "Bob") == middle


def test_rated_perf_fetches_missing_ranks_within_budget(cache, monkeypatch):
    cache.store_contests([contest(2000, 200)])
    fetched = []

    def fetch(contest_id):
        fetched.append(contest_id)
        return pd.DataFrame(
            [("a", 1, 210, 1500, 1600), ("b", 2, 210, 1500, 1400)],
            columns=["handle", "rank", "ratingUpdateTimeSeconds", "oldRating", "newRating"],
        )

    monkeypatch.setattr(module.cf_api, "get_rating_changes", fetch)
    with pytest.raises(module.BudgetExhausted):
        module.rated_perf("b", 2000, cache, FetchBudget(0))
    assert module.rated_perf("b", 2000, cache, FetchBudget(1)) < 1500
    assert fetched == [2000]


def test_newcomer_offsets_apply_with_complete_history(cache):
    # 'new' first played contest 1400 (shown 0 -> 400), then contest 1401.
    cache.store_contests([contest(1400, 100), contest(1401, 200)])
    columns = ["handle", "rank", "ratingUpdateTimeSeconds", "oldRating", "newRating"]
    cache.store_rating_changes(1400, pd.DataFrame(
        [("new", 1, 110, 0, 400), ("old", 2, 110, 1500, 1490)], columns=columns))
    cache.store_rating_changes(1401, pd.DataFrame(
        [("new", 1, 210, 400, 700), ("old", 2, 210, 1490, 1480),
         ("fresh", 3, 210, 0, 300)], columns=columns))
    field = cache.contest_field(1401).set_index("handle").effective_rating
    assert field.to_dict() == {"new": 400 + 900, "old": 1490, "fresh": 1400}


def test_newcomer_offsets_fall_back_without_complete_history(cache):
    cache.store_contests([contest(1400, 100), contest(1401, 200)])
    columns = ["handle", "rank", "ratingUpdateTimeSeconds", "oldRating", "newRating"]
    cache.store_rating_changes(1401, pd.DataFrame(
        [("new", 1, 210, 400, 700), ("fresh", 2, 210, 0, 300)], columns=columns))
    field = cache.contest_field(1401).set_index("handle").effective_rating
    assert field.to_dict() == {"new": 400, "fresh": 1400}


def rating_entry(contest_id, when, old=1500, new=1550, handle="Alice"):
    return {
        "contestId": contest_id, "contestName": f"Round {contest_id}",
        "handle": handle, "rank": 5, "ratingUpdateTimeSeconds": when,
        "oldRating": old, "newRating": new,
    }


def test_perf_embed_lists_rated_and_virtual_runs(cache, monkeypatch):
    monkeypatch.setattr(module.cf_api, "get_user_rating",
                        lambda _: [rating_entry(2000, 300)])
    monkeypatch.setattr(module.cf_api, "get_contest_status",
                        lambda *_: [submission("A", "OK", 60, contest_id=2000)])
    monkeypatch.setattr(module, "rated_perf", lambda *_args, **_kw: 1750.0)
    monkeypatch.setattr(
        module, "virtual_perfs",
        lambda *_args, **_kw: ([module.VirtualResult(
            2000, 1000, "Round 2000", rank=12, points=3, penalty=40, perf=1620.0)], 0),
    )
    embed = module.perf("alice", 2000, cache)
    assert embed.title == "Round 2000"
    assert [f.name for f in embed.fields] == ["Contest", "Virtual"]
    assert "**1750**" in embed.fields[0].value and "+50" in embed.fields[0].value
    assert "**1620** (estimated)" in embed.fields[1].value
    assert "penalty 40" in embed.fields[1].value
    assert embed.color.value == module.RANK_COLORS["expert"]


def test_perf_without_participation_is_a_user_error(cache, monkeypatch):
    monkeypatch.setattr(module.cf_api, "get_user_rating", lambda _: [])
    monkeypatch.setattr(module.cf_api, "get_contest_status", lambda *_: [])
    with pytest.raises(ValueError, match="no rated or virtual participation"):
        module.perf("alice", 2000, cache)


def test_perf_reports_unknown_handle(cache, monkeypatch):
    def missing(_):
        raise module.cf_api.UserNotFoundError("not found")

    monkeypatch.setattr(module.cf_api, "get_user_rating", missing)
    with pytest.raises(ValueError, match="was not found"):
        module.perf("ghost", 2000, cache)


def ts(*parts):
    return int(datetime(*parts, tzinfo=timezone.utc).timestamp())


def setup_graph(monkeypatch):
    history = [
        rating_entry(1, ts(2022, 6, 1)),
        rating_entry(2, ts(2023, 1, 1)),
        rating_entry(3, ts(2023, 12, 31)),
        rating_entry(4, ts(2024, 1, 1), new=1600),
    ]
    monkeypatch.setattr(module.cf_api, "get_user_rating", lambda _: history)
    monkeypatch.setattr(
        module.cf_api, "get_user_status",
        lambda _: [submission("A", "OK", 60, contest_id=9, start=ts(2023, 5, 5))],
    )
    rated_calls = []

    def fake_rated(handle, contest_id, cache, budget=None):
        rated_calls.append(contest_id)
        return math.inf if contest_id == 3 else 1500.0 + contest_id

    def fake_virtual(handle, groups, cache, budget=None):
        return [module.VirtualResult(cid, start, "V", perf=1700.0)
                for cid, starts in groups.items() for start in starts], 0

    monkeypatch.setattr(module, "rated_perf", fake_rated)
    monkeypatch.setattr(module, "virtual_perfs", fake_virtual)
    return rated_calls


def test_perfgraph_filters_dates_and_connects_participations(cache, monkeypatch):
    rated_calls = setup_graph(monkeypatch)
    result = module.perfgraph("alice", cache, "both", "2023", "2024")
    assert rated_calls == [2, 3]
    # One chronological series; the rank-1 finish sits at max(3000, best + 200).
    assert [(p.perf, p.virtual) for p in result.points] == [
        (1502.0, False), (1700.0, True), (3000.0, False),
    ]
    assert result.labels == ["Alice (1600)", "Alice (virtual)"]
    assert "Infinite performances (rank 1): 1" in result.note
    assert "plot" not in result.note.split("\n")[0].lower()
    assert "estimates" in result.note
    assert result.png.startswith(b"\x89PNG")


def test_infinite_points_sit_above_the_best_finite_performance():
    points = [
        module.GraphPoint(1, 3100.0, False),
        module.GraphPoint(2, math.inf, True),
    ]
    assert [p.perf for p in module.plotted_points(points)] == [3100.0, 3300.0]
    only_infinite = [module.GraphPoint(1, math.inf, False)]
    assert [p.perf for p in module.plotted_points(only_infinite)] == [3000.0]


def test_perfgraph_include_choices(cache, monkeypatch):
    setup_graph(monkeypatch)
    rated = module.perfgraph("alice", cache, "rated")
    assert rated.labels == ["Alice (1600)"]
    assert not any(p.virtual for p in rated.points)
    virtual = module.perfgraph("alice", cache, "virtual")
    assert virtual.labels == ["Alice (virtual)"]
    assert [p.perf for p in virtual.points] == [1700.0]
    with pytest.raises(ValueError, match="no virtual contest performances"):
        module.perfgraph("alice", cache, "virtual", "2025")
    with pytest.raises(ValueError, match="earlier"):
        module.perfgraph("alice", cache, "both", "2024", "2023")
