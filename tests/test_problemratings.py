import importlib

import numpy as np
import pandas as pd
import pytest

from conftest import changes, contest

module = importlib.import_module("wob_bot.problemratings")


def setup_command(
    monkeypatch, handles, scores, published=None, *, phase="FINISHED", official=None
):
    participants = pd.DataFrame({"handle": handles, "participantType": "CONTESTANT"})
    problems = pd.DataFrame({"label": ["A", "B"]})
    if official is not None:
        problems["rating"] = official
    results = pd.DataFrame(
        [
            {"handle": handle, "problem_index": index, "points": score}
            for handle, per_problem in zip(handles, scores)
            for index, score in enumerate(per_problem)
        ],
        columns=["handle", "problem_index", "points"],
    )
    metadata = contest(2, 200, phase)
    monkeypatch.setattr(
        module.cf_api,
        "get_contest_standings",
        lambda _: (metadata, problems, participants, results),
    )
    monkeypatch.setattr(
        module.cf_api,
        "get_rating_changes",
        lambda _: published if published is not None else changes(),
    )
    return metadata, problems, participants, results


def seed_history(cache, *rows):
    cache.store_contests([contest(1), contest(2, 200)])
    cache.store_rating_changes(1, changes(*rows))


def test_excluded_solves_do_not_change_predictions(cache, monkeypatch):
    seed_history(cache, ("a", 1000, 1600, 110), ("b", 1000, 1600, 110))
    setup_command(monkeypatch, ["a", "b", "missing"], [[1, 0], [0, 0], [1, 1]])
    result = module.problemratings(2, cache)
    setup_command(monkeypatch, ["a", "b", "missing"], [[1, 0], [0, 0], [0, 0]])
    no_excluded_solves = module.problemratings(2, cache)
    assert result.embed.description == no_excluded_solves.embed.description
    assert "1600" in result.embed.description
    assert ">=8000" in result.embed.description
    assert result.included_count == 2
    assert result.excluded_count == 1
    assert result.chat_text == "Excluded 1 participants missing from cache."


def test_published_ratings_override_history_and_first_timers_are_excluded(
    cache, monkeypatch
):
    seed_history(cache, ("a", 1000, 1200, 110), ("b", 1000, 1200, 110))
    setup_command(
        monkeypatch,
        ["a", "b", "first"],
        [[1, 0], [0, 1], [1, 1]],
        changes(
            ("a", 1800, 1900, 210), ("b", 1800, 1700, 210), ("first", 0, 1000, 210)
        ),
    )
    result = module.problemratings(2, cache)
    assert "1800" in result.embed.description
    assert result.included_count == 2
    assert result.excluded_count == 1
    assert "0 cached historical ratings" in result.embed.footer.text


def test_real_zero_rating_is_retained_with_evidence_of_prior_participation(
    cache, monkeypatch
):
    seed_history(cache, ("a", 100, 0, 110))
    setup_command(
        monkeypatch,
        ["a", "first"],
        [[0, 1], [1, 0]],
        changes(("a", 0, 100, 210), ("first", 0, 100, 210)),
    )
    result = module.problemratings(2, cache)
    assert result.included_count == 1
    assert result.excluded_count == 1


def test_incomplete_history_does_not_silently_use_an_outdated_rating(
    cache, monkeypatch
):
    seed_history(cache, ("a", 1000, 1200, 110))
    cache.store_contests([contest(9, 150)])
    setup_command(
        monkeypatch, ["a", "b"], [[1, 0], [0, 1]], changes(("b", 1500, 1600, 210))
    )
    result = module.problemratings(2, cache)
    assert result.included_count == 1
    assert result.excluded_count == 1
    assert "warming" in result.embed.footer.text


def test_no_eligible_participants_returns_na_and_counts(cache, monkeypatch):
    setup_command(monkeypatch, ["first", "missing"], [[1, 1], [0, 0]])
    result = module.problemratings(2, cache)
    assert result.included_count == 0
    assert result.excluded_count == 2
    assert "N/A" in result.embed.description
    assert "No participants with usable" in result.embed.description


def test_case_variants_are_counted_once_and_unofficial_users_are_ignored(
    cache, monkeypatch
):
    _, _, participants, _ = setup_command(
        monkeypatch,
        ["a", "A", "practice"],
        [[1, 0], [1, 0], [1, 1]],
        changes(("A", 1500, 1600, 210)),
    )
    participants.loc[2, "participantType"] = "PRACTICE"
    result = module.problemratings(2, cache)
    assert result.included_count == 1
    assert result.excluded_count == 0
    assert result.chat_text == "Excluded 0 participants missing from cache."


def test_empty_standings_and_partial_official_ratings(cache, monkeypatch):
    setup_command(monkeypatch, [], [], official=[800, np.nan])
    result = module.problemratings(2, cache)
    assert result.included_count == result.excluded_count == 0
    assert "None" in result.embed.description
    assert "800" in result.embed.description


def test_live_and_preliminary_standings_are_marked_provisional(cache, monkeypatch):
    setup_command(
        monkeypatch, ["a"], [[1, 0]], changes(("a", 1200, 1300, 210)), phase="CODING"
    )
    assert "Provisional" in module.problemratings(2, cache).embed.footer.text
    _, _, _, results = setup_command(
        monkeypatch, ["a"], [[1, 0]], changes(("a", 1200, 1300, 210))
    )
    results["type"] = "PRELIMINARY"
    assert "Provisional" in module.problemratings(2, cache).embed.footer.text


@pytest.mark.parametrize(
    "phase,kind,message",
    [("BEFORE", "CF", "not started"), ("FINISHED", "IOI", "CF and ICPC")],
)
def test_unsupported_contests_have_clear_errors(
    cache, monkeypatch, phase, kind, message
):
    metadata, _, _, _ = setup_command(monkeypatch, [], [], phase=phase)
    metadata["type"] = kind
    with pytest.raises(ValueError, match=message):
        module.problemratings(2, cache)


def test_binary_search_and_boundary_cases():
    ratings = np.array([1200, 1600, 2000, 2400])
    assert module.calc_problem_rating(ratings, 2) == 1800
    assert module.calc_problem_rating(ratings, 0) == ">=8000"
    assert module.calc_problem_rating(ratings, 4) == "<=-1000"
    assert module.calc_problem_rating(np.array([]), 0) == "N/A"
