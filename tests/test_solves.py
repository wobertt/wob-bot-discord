from __future__ import annotations

from wob_bot.solves import get_problem_solves


def test_get_problem_solves_returns_placeholder_count() -> None:
    assert get_problem_solves(2033, "A") == 0
