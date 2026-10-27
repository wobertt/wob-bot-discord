from concurrent.futures import ThreadPoolExecutor
import threading
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from wob_bot.cf_api import cf_api
from wob_bot.solve_info import get_time_data


def response(payload, status=200):
    return Mock(json=Mock(return_value=payload), status_code=status, ok=status < 400)


@pytest.fixture(autouse=True)
def no_real_requests(monkeypatch):
    monkeypatch.setattr(cf_api, "_wait_for_request_slot", Mock())
    monkeypatch.setattr(
        cf_api.requests,
        "get",
        Mock(side_effect=AssertionError("Unexpected network request")),
    )


@pytest.mark.parametrize(
    "payload,status",
    [
        ({"status": "OK", "result": []}, 200),
        ({"status": "FAILED", "comment": "Rating changes are unavailable"}, 400),
    ],
)
def test_unavailable_rating_changes_have_stable_columns(monkeypatch, payload, status):
    monkeypatch.setattr(
        cf_api.requests, "get", Mock(return_value=response(payload, status))
    )
    frame = cf_api.get_rating_changes(1)
    assert frame.empty
    assert list(frame.columns) == [
        "handle",
        "rank",
        "ratingUpdateTimeSeconds",
        "oldRating",
        "newRating",
    ]


def test_invalid_contest_is_not_treated_as_unavailable(monkeypatch):
    monkeypatch.setattr(
        cf_api.requests,
        "get",
        Mock(
            return_value=response(
                {"status": "FAILED", "comment": "contestId: Contest not found"}, 400
            )
        ),
    )
    with pytest.raises(cf_api.CodeforcesApiError, match="Contest not found"):
        cf_api.get_rating_changes(1)


def test_api_failures_are_handled_even_with_http_200(monkeypatch):
    monkeypatch.setattr(
        cf_api.requests,
        "get",
        Mock(return_value=response({"status": "FAILED", "comment": "Request denied"})),
    )
    with pytest.raises(cf_api.CodeforcesApiError, match="Request denied"):
        cf_api.make_request("contest.list")


def test_transient_failures_retry_with_timeouts_and_shared_limiter(monkeypatch):
    limiter = cf_api._wait_for_request_slot
    get = Mock(
        side_effect=[
            requests.Timeout(),
            response({"status": "FAILED", "comment": "Call limit exceeded"}),
            response({"status": "OK", "result": [1]}),
        ]
    )
    monkeypatch.setattr(cf_api.requests, "get", get)
    assert cf_api.make_request("contest.list") == [1]
    assert limiter.call_count == 3
    assert get.call_count == 3
    assert get.call_args.kwargs["timeout"] == (5, 30)


def test_persistent_network_failure_is_bounded(monkeypatch):
    get = Mock(side_effect=requests.ConnectionError())
    monkeypatch.setattr(cf_api.requests, "get", get)
    with pytest.raises(cf_api.CodeforcesApiError, match="could not be reached"):
        cf_api.get_rating_changes(1)
    assert get.call_count == 3


def test_server_errors_and_invalid_json_are_bounded(monkeypatch):
    bad = response(None, 503)
    bad.json.side_effect = ValueError("not JSON")
    get = Mock(return_value=bad)
    monkeypatch.setattr(cf_api.requests, "get", get)
    with pytest.raises(cf_api.CodeforcesApiError, match="invalid response"):
        cf_api.get_contests()
    assert get.call_count == 3


def test_non_object_json_is_reported_as_invalid(monkeypatch):
    monkeypatch.setattr(cf_api.requests, "get", Mock(return_value=response([])))
    with pytest.raises(cf_api.CodeforcesApiError, match="invalid response"):
        cf_api.get_contests()


@pytest.mark.parametrize("empty", [False, True])
def test_standings_allow_missing_times_and_empty_rows(monkeypatch, empty):
    row = {
        "party": {"members": [{"handle": "a"}], "participantType": "CONTESTANT"},
        "rank": 1,
        "points": 0,
        "penalty": 0,
        "successfulHackCount": 0,
        "unsuccessfulHackCount": 0,
        "problemResults": [{"points": 0}],
    }
    payload = {
        "contest": {"id": 1},
        "problems": [{"index": "A"}],
        "rows": [] if empty else [row],
    }
    monkeypatch.setattr(cf_api, "make_request", lambda _: payload)
    _, problems, participants, results = cf_api.get_contest_standings(1)
    assert problems.label.tolist() == ["A"]
    assert "participantType" in participants
    assert "bestSubmissionTimeSeconds" in results
    assert results.bestSubmissionTimeSeconds.isna().all()
    assert len(results) == (0 if empty else 1)


def test_team_standings_raise_descriptive_error(monkeypatch):
    monkeypatch.setattr(
        cf_api,
        "make_request",
        lambda _: {
            "contest": {},
            "problems": [],
            "rows": [{"party": {"members": [{"handle": "a"}, {"handle": "b"}]}}],
        },
    )
    with pytest.raises(ValueError, match="individual participants"):
        cf_api.get_contest_standings(1)


def test_solve_plot_data_remains_compatible(monkeypatch):
    problems = pd.DataFrame({"label": ["A"]})
    results = pd.DataFrame(
        {"handle": ["a"], "problem_index": [0], "bestSubmissionTimeSeconds": [120]}
    )
    ratings = pd.DataFrame({"handle": ["a"], "oldRating": [1200]})
    monkeypatch.setattr(
        cf_api,
        "get_contest_standings",
        lambda _: ({}, problems, pd.DataFrame(), results),
    )
    monkeypatch.setattr(cf_api, "get_rating_changes", lambda _: ratings)
    assert get_time_data(1, "A").to_dict("records") == [
        {"handle": "a", "rating": 1200, "solve_time": 2.0}
    ]


# Keep a reference before the per-test network guard replaces the function.
real_wait_for_slot = cf_api._wait_for_request_slot


def test_rate_limit_is_shared_across_threads(monkeypatch):
    fake_time = [100.0]
    slots = []

    def sleep(delay):
        fake_time[0] += delay
        slots.append(fake_time[0])

    monkeypatch.setattr(cf_api, "_next_request_at", 0.0)
    monkeypatch.setattr(cf_api.time, "monotonic", lambda: fake_time[0])
    monkeypatch.setattr(cf_api.time, "sleep", sleep)
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(lambda _: real_wait_for_slot(None), range(3)))
    assert slots == [100.0, 102.0, 104.0]


def test_stopped_background_request_is_not_sent():
    stopped = threading.Event()
    stopped.set()
    with pytest.raises(cf_api.RequestCancelled):
        real_wait_for_slot(stopped)
