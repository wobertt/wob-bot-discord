"""Contest performance: carrot's rating-calculator method for rated and virtual runs.

Performance is the pre-contest rating at which a contestant's rating change
(including the contest-wide adjustment) would have been zero, holding everyone
else's real pre-contest rating fixed. The calculator is a numpy port of carrot
(https://github.com/meooow25/carrot, src/background/predict.js), which adapted
TLE's rating calculator, itself based on Mike Mirzayanov's published formula.

Virtual results can no longer be read from contest.standings (non-admin calls
only accept contestId), so a virtual score is rebuilt from the user's own
submissions and ranked against the official rated field. Those are estimates.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import math
import re
import threading
from urllib.parse import quote

import discord
import numpy as np
import pandas as pd

from .cf_api import cf_api
from .fetchuser import RANK_COLORS
from .rating_cache import RECENT_WINDOW, RatingCache

DEFAULT_RATING = 1400
MAX_RATING_LIMIT = 6000
MIN_RATING_LIMIT = -500
_RATING_RANGE_LEN = MAX_RATING_LIMIT - MIN_RATING_LIMIT
_ELO_OFFSET = _RATING_RANGE_LEN
_RATING_OFFSET = -MIN_RATING_LIMIT
GYM_ID_THRESHOLD = 100_000

# ELO_WIN_PROB[y - x + ELO_OFFSET]: probability that rating x beats rating y.
_ELO_WIN_PROB = 1 / (
    1 + np.power(10.0, np.arange(-_RATING_RANGE_LEN, _RATING_RANGE_LEN + 1) / 400)
)
_SEED_LEN = len(_ELO_WIN_PROB) + _RATING_RANGE_LEN - 1
_FFT_LEN = 1 << (_SEED_LEN - 1).bit_length()
_ELO_WIN_PROB_FFT = np.fft.rfft(_ELO_WIN_PROB, _FFT_LEN)


def _binary_search(lo: np.ndarray, hi: np.ndarray, predicate) -> np.ndarray:
    """Vectorized first value in [lo, hi) where predicate holds (hi if none)."""
    lo, hi = lo.copy(), hi.copy()
    while True:
        active = lo < hi
        if not active.any():
            return lo
        mid = (lo + hi) // 2
        hit = predicate(mid) & active
        hi = np.where(hit, mid, hi)
        lo = np.where(active & ~hit, mid + 1, lo)


class RatingCalculator:
    """Codeforces rating deltas, adjustment and performances for one contest."""

    def __init__(self, points, penalty, ratings) -> None:
        points = np.asarray(points, dtype=float)
        penalty = np.asarray(penalty, dtype=float)
        self.ratings = np.clip(
            np.asarray(ratings, dtype=np.int64), MIN_RATING_LIMIT, MAX_RATING_LIMIT - 1
        )
        self.n = n = len(self.ratings)
        if n == 0:
            raise ValueError("A contest needs at least one rated contestant.")

        # Standings order: points desc, penalty asc. Ties share the rank of
        # the group's last position, as in carrot's reassignRanks.
        order = np.lexsort((penalty, -points))
        sorted_key = np.stack([points[order], penalty[order]], axis=1)
        new_group = np.ones(n, dtype=bool)
        new_group[1:] = np.any(sorted_key[1:] != sorted_key[:-1], axis=1)
        group_id = np.cumsum(new_group) - 1
        group_last = np.zeros(group_id[-1] + 1, dtype=np.int64)
        np.maximum.at(group_last, group_id, np.arange(n))
        self.rank = np.empty(n, dtype=np.int64)
        self.rank[order] = group_last[group_id] + 1

        counts = np.bincount(self.ratings + _RATING_OFFSET, minlength=_RATING_RANGE_LEN)
        seed = np.fft.irfft(_ELO_WIN_PROB_FFT * np.fft.rfft(counts, _FFT_LEN), _FFT_LEN)
        self.seed = seed[:_SEED_LEN] + 1

        all_index = np.arange(n)
        deltas = self.calc_delta(all_index, self.ratings).astype(np.int64)
        # Adjustment passes operate on contestants sorted by rating desc,
        # stable with respect to standings order.
        by_rating = order[np.argsort(-self.ratings[order], kind="stable")]
        inc = math.trunc(-int(deltas.sum()) / n) - 1
        deltas += inc
        zero_sum_count = min(4 * math.floor(math.sqrt(n) + 0.5), n)
        top_sum = int(deltas[by_rating[:zero_sum_count]].sum())
        inc2 = min(max(math.trunc(-top_sum / zero_sum_count), -10), 0)
        deltas += inc2
        self.deltas = deltas
        self.adjustment = inc + inc2

    def get_seed(self, rating: np.ndarray, exclude: np.ndarray) -> np.ndarray:
        """Expected rank at `rating`, leaving out a contestant rated `exclude`."""
        return (
            self.seed[rating + _ELO_OFFSET + _RATING_OFFSET]
            - _ELO_WIN_PROB[rating - exclude + _ELO_OFFSET]
        )

    def rank_to_rating(self, rank: np.ndarray, self_rating: np.ndarray) -> np.ndarray:
        lo = np.full(len(rank), 2, dtype=np.int64)
        hi = np.full(len(rank), MAX_RATING_LIMIT, dtype=np.int64)
        return (
            _binary_search(lo, hi, lambda r: self.get_seed(r, self_rating) < rank) - 1
        )

    def calc_delta(self, index: np.ndarray, assumed: np.ndarray) -> np.ndarray:
        own = self.ratings[index]
        seed = self.get_seed(assumed, own)
        mid_rank = np.sqrt(self.rank[index] * seed)
        need = self.rank_to_rating(mid_rank, own)
        return np.trunc((need - assumed) / 2)

    def performance(self, index) -> np.ndarray:
        """Performance of contestants at `index`; rank 1 is infinite."""
        index = np.atleast_1d(np.asarray(index, dtype=np.int64))
        lo = np.full(len(index), MIN_RATING_LIMIT, dtype=np.int64)
        hi = np.full(len(index), MAX_RATING_LIMIT, dtype=np.int64)
        perf = _binary_search(
            lo, hi, lambda r: self.calc_delta(index, r) + self.adjustment <= 0
        ).astype(float)
        perf[self.rank[index] == 1] = math.inf
        return perf


class PerfUnavailable(Exception):
    """A participation that cannot be given a performance, with the reason."""


class BudgetExhausted(Exception):
    """A command used up its allowance of slow Codeforces requests."""


@dataclass
class FetchBudget:
    remaining: int | None = None

    def spend(self) -> None:
        if self.remaining is None:
            return
        if self.remaining <= 0:
            raise BudgetExhausted
        self.remaining -= 1


def _history_complete(cache: RatingCache) -> bool:
    checked, total = cache.progress()
    return total > 0 and checked == total


def _ensure_ranks(
    contest_id: int, handle: str | None, cache: RatingCache, budget: FetchBudget
) -> None:
    if cache.contest_has_ranks(contest_id, handle):
        return
    budget.spend()
    try:
        changes = cf_api.get_rating_changes(contest_id)
    except cf_api.ContestNotFoundError as exc:
        raise PerfUnavailable("contest is not public") from exc
    cache.store_rating_changes(contest_id, changes)
    if not cache.contest_has_ranks(contest_id):
        raise PerfUnavailable("rating changes are not published")
    if handle is not None and not cache.contest_has_ranks(contest_id, handle):
        raise PerfUnavailable("user has no rating change in this contest")


def rated_perf(
    handle: str, contest_id: int, cache: RatingCache, budget: FetchBudget | None = None
) -> float:
    """Performance of an official rated participation, from the cached field."""
    budget = budget or FetchBudget()
    cached = cache.get_rated_perf(contest_id, handle)
    if cached is not None:
        return cached
    _ensure_ranks(contest_id, handle, cache, budget)
    contest_field = cache.contest_field(contest_id)
    calculator = RatingCalculator(
        -contest_field["rank"].to_numpy(), np.zeros(len(contest_field)),
        contest_field.effective_rating.to_numpy(),
    )
    index = np.flatnonzero(contest_field.handle.to_numpy() == handle.casefold())
    perf = float(calculator.performance(index[0])[0])
    if _history_complete(cache):
        cache.store_rated_perf(contest_id, handle, perf)
    return perf


# ---------------------------------------------------------------------------
# Virtual participations
# ---------------------------------------------------------------------------

_NOT_PENALIZED = {"OK", "COMPILATION_ERROR", "SKIPPED", "TESTING", None}


def _penalized(submission: dict) -> bool:
    # Codeforces does not penalize compilation errors or failures on test 1.
    return (
        submission.get("verdict") not in _NOT_PENALIZED
        and submission.get("passedTestCount", 0) > 0
    )


def group_virtuals(submissions: list[dict]) -> dict[int, dict[int, list[dict]]]:
    """VIRTUAL submissions as {contest_id: {virtual_start_time: [submissions]}}."""
    groups: dict[int, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for submission in submissions:
        author = submission.get("author") or {}
        contest_id = submission.get("contestId")
        start = author.get("startTimeSeconds")
        if (
            author.get("participantType") == "VIRTUAL"
            and contest_id is not None
            and contest_id < GYM_ID_THRESHOLD
            and start is not None
        ):
            groups[contest_id][start].append(submission)
    return {cid: dict(starts) for cid, starts in groups.items()}


def _icpc_penalty_per_wrong(participants: pd.DataFrame, results: pd.DataFrame) -> int:
    """Infer the minutes per rejected attempt from official rows (10 or 20)."""
    solved = results[results.points > 0]
    if solved.empty or "rejectedAttemptCount" not in solved:
        return 10
    per_row = solved.groupby("handle").agg(
        minutes=("bestSubmissionTimeSeconds", lambda s: (s // 60).sum()),
        rejected=("rejectedAttemptCount", "sum"),
    )
    per_row = per_row.join(participants.set_index("handle").penalty, how="inner")
    per_row = per_row[per_row.rejected > 0]
    if per_row.empty:
        return 10
    estimate = ((per_row.penalty - per_row.minutes) / per_row.rejected).median()
    return int(round(estimate)) if np.isfinite(estimate) and estimate > 0 else 10


def _cf_decay(max_points: float, problem_results: pd.DataFrame) -> float:
    """Points lost per minute on a CF-format problem, fitted to official rows."""
    clean = problem_results[
        (problem_results.points > 0.3 * max_points + 1)
        & (problem_results.get("rejectedAttemptCount", 0) == 0)
        & (problem_results.bestSubmissionTimeSeconds >= 60)
    ]
    if clean.empty:
        return max_points / 250
    minutes = clean.bestSubmissionTimeSeconds // 60
    decay = ((max_points - clean.points) / minutes).median()
    return float(decay) if np.isfinite(decay) and decay >= 0 else max_points / 250


def virtual_score(
    contest: dict,
    problems: pd.DataFrame,
    participants: pd.DataFrame,
    problem_results: pd.DataFrame,
    submissions: list[dict],
) -> tuple[float, int]:
    """Rebuild (points, penalty) for one virtual run, scored like the official field."""
    duration = contest.get("durationSeconds")
    in_window = [
        s for s in submissions
        if duration is None or s.get("relativeTimeSeconds", 0) <= duration
    ]
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for submission in sorted(
        in_window, key=lambda s: (s.get("relativeTimeSeconds", 0), s.get("id", 0))
    ):
        by_problem[submission["problem"]["index"]].append(submission)

    points, penalty = 0.0, 0
    if contest["type"] == "ICPC":
        per_wrong = _icpc_penalty_per_wrong(participants, problem_results)
        for subs in by_problem.values():
            accepted = next(
                (i for i, s in enumerate(subs) if s.get("verdict") == "OK"), None
            )
            if accepted is None:
                continue
            wrong = sum(_penalized(s) for s in subs[:accepted])
            points += 1
            penalty += subs[accepted]["relativeTimeSeconds"] // 60 + per_wrong * wrong
        return points, penalty

    if "points" not in problems or problems.points.isna().any():
        raise PerfUnavailable("problem point values are not published")
    for label, subs in by_problem.items():
        matches = problems.index[problems.label == label]
        if len(matches) == 0:
            continue
        problem_index = matches[0]
        accepted = [i for i, s in enumerate(subs) if s.get("verdict") == "OK"]
        if not accepted:
            continue
        last = accepted[-1]
        # Earlier passing submissions count as rejected after a resubmission.
        wrong = sum(_penalized(s) or s.get("verdict") == "OK" for s in subs[:last])
        max_points = float(problems.points[problem_index])
        decay = _cf_decay(
            max_points, problem_results[problem_results.problem_index == problem_index]
        )
        minute = subs[last]["relativeTimeSeconds"] // 60
        points += max(0.3 * max_points, max_points - decay * minute - 50 * wrong)
    return round(points, 2), penalty


@dataclass
class VirtualResult:
    contest_id: int
    start_time: int
    contest_name: str
    rank: int | None = None
    points: float | None = None
    penalty: int | None = None
    perf: float | None = None
    skip_reason: str | None = None

    def record(self) -> dict:
        return self.__dict__.copy()


@dataclass
class _ContestStandings:
    contest: dict
    problems: pd.DataFrame
    participants: pd.DataFrame
    problem_results: pd.DataFrame
    field: pd.DataFrame


def _load_standings(
    contest_id: int, cache: RatingCache, budget: FetchBudget
) -> _ContestStandings:
    budget.spend()
    try:
        contest, problems, participants, results = cf_api.get_contest_standings(contest_id)
    except cf_api.ContestNotFoundError as exc:
        raise PerfUnavailable("contest is not public") from exc
    except ValueError as exc:  # team standings
        raise PerfUnavailable("team contests are not supported") from exc
    if contest.get("type") not in ("CF", "ICPC"):
        raise PerfUnavailable(f"{contest.get('type', 'this')} contests are not supported")
    if contest.get("phase") != "FINISHED":
        raise PerfUnavailable("contest has not finished")
    cache.record_contest(contest)
    _ensure_ranks(contest_id, None, cache, budget)
    # The rated field comes from rating changes. Anonymous standings can hide
    # rated rows (e.g. untrusted Div. 3 participants), so visible rows are only
    # used to place a virtual score in official-rank order.
    rated_field = cache.contest_field(contest_id)
    ranks = rated_field.set_index("handle")["rank"]
    official = participants[participants.participantType == "CONTESTANT"].copy()
    official["handle_key"] = official.handle.astype(str).str.casefold()
    official = official.drop_duplicates("handle_key")
    official = official[official.handle_key.isin(ranks.index)]
    official["cf_rank"] = official.handle_key.map(ranks)
    if official.empty:
        raise PerfUnavailable("no rated official contestants")
    official_results = results[results.handle.isin(official.handle)]
    return _ContestStandings(
        contest, problems, official, official_results, rated_field
    )


def _virtual_rank_key(visible: pd.DataFrame, points: float, penalty: int) -> float:
    """Official-rank position for a virtual score, from the visible rows around it.

    A tie with visible rows shares their rank. Otherwise the score sits
    between the last better row and the first worse row; hidden rows ranked
    in that gap have unknown scores, so take the midpoint (offset to avoid an
    accidental tie with a hidden rank).
    """
    better = (visible.points > points) | (
        (visible.points == points) & (visible.penalty < penalty)
    )
    tied = (visible.points == points) & (visible.penalty == penalty)
    if tied.any():
        return float(visible.cf_rank[tied].min())
    above = float(visible.cf_rank[better].max()) if better.any() else 0.0
    worse = visible.cf_rank[~better]
    if worse.empty:
        return above + 0.5
    return (above + float(worse.min())) / 2 + 0.25


def _virtual_result(
    handle: str,
    standings: _ContestStandings,
    start: int,
    submissions: list[dict],
    cache: RatingCache,
) -> VirtualResult:
    contest = standings.contest
    points, penalty = virtual_score(
        contest, standings.problems, standings.participants,
        standings.problem_results, submissions,
    )
    rated_field = standings.field
    key = _virtual_rank_key(standings.participants, points, penalty)
    own_rating = cache.rating_before(handle, start)
    # Order by official rank (lower is better), expressed as points = -rank.
    calculator = RatingCalculator(
        np.append(-rated_field["rank"].to_numpy(dtype=float), -key),
        np.zeros(len(rated_field) + 1),
        np.append(rated_field.effective_rating.to_numpy(), own_rating or DEFAULT_RATING),
    )
    return VirtualResult(
        contest_id=contest["id"],
        start_time=start,
        contest_name=contest["name"],
        rank=int((rated_field["rank"] < key).sum()) + 1,
        points=points,
        penalty=penalty if contest["type"] == "ICPC" else None,
        perf=float(calculator.performance(calculator.n - 1)[0]),
    )


def virtual_perfs(
    handle: str,
    groups: dict[int, dict[int, list[dict]]],
    cache: RatingCache,
    budget: FetchBudget | None = None,
    *,
    use_cache: bool = True,
) -> tuple[list[VirtualResult], int]:
    """Virtual results for the grouped submissions; returns (results, pending)."""
    budget = budget or FetchBudget()
    stored = cache.get_virtual_perfs(handle) if use_cache else {}
    complete = _history_complete(cache)
    results, pending = [], 0
    for contest_id in sorted(groups):
        starts = groups[contest_id]
        todo = []
        for start in sorted(starts):
            record = stored.get((contest_id, start))
            if record is not None:
                record = {k: v for k, v in record.items() if k != "handle"}
                results.append(VirtualResult(**record))
            else:
                todo.append(start)
        if not todo:
            continue
        try:
            standings = _load_standings(contest_id, cache, budget)
        except BudgetExhausted:
            pending += len(todo)
            continue
        except PerfUnavailable as exc:
            for start in todo:
                results.append(
                    VirtualResult(contest_id, start, f"Contest {contest_id}",
                                  skip_reason=str(exc))
                )
            continue
        finished_long_ago = (
            standings.contest.get("startTimeSeconds", 0)
            + standings.contest.get("durationSeconds", 0)
            < datetime.now(timezone.utc).timestamp() - RECENT_WINDOW
        )
        for start in todo:
            try:
                result = _virtual_result(
                    handle, standings, start, starts[start], cache
                )
            except PerfUnavailable as exc:
                result = VirtualResult(contest_id, start, standings.contest["name"],
                                       skip_reason=str(exc))
            results.append(result)
            if complete and finished_long_ago:
                cache.store_virtual_perf(handle, result.record())
    return results, pending


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------

# (low, high, name, band color) — TLE's RATED_RANKS graph colors.
RANK_BANDS = (
    (-10**9, 1200, "newbie", "#CCCCCC"),
    (1200, 1400, "pupil", "#77FF77"),
    (1400, 1600, "specialist", "#77DDBB"),
    (1600, 1900, "expert", "#AAAAFF"),
    (1900, 2100, "candidate master", "#FF88FF"),
    (2100, 2300, "master", "#FFCC88"),
    (2300, 2400, "international master", "#FFBB55"),
    (2400, 2600, "grandmaster", "#FF7777"),
    (2600, 3000, "international grandmaster", "#FF3333"),
    (3000, 10**9, "legendary grandmaster", "#AA0000"),
)


def rank_name(rating: float) -> str:
    for low, high, name, _ in RANK_BANDS:
        if low <= rating < high:
            return name
    return RANK_BANDS[-1][2]


def format_perf(perf: float) -> str:
    return "∞ (rank 1)" if math.isinf(perf) else f"{int(perf)}"


def _profile_url(handle: str) -> str:
    return f"https://codeforces.com/profile/{quote(handle, safe='')}"


def _format_date(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


_DATE_HINT = "Use yyyy, mmyyyy or ddmmyyyy (e.g. 2023, 052023, 15052023)."


def parse_cf_date(text: str) -> int:
    """Parse TLE-style [[dd]mm]yyyy (separators allowed) to a UTC timestamp."""
    value = text.strip()
    if re.search(r"[-/.]", value):
        parts = re.split(r"[-/.]", value)
    elif value.isdigit() and len(value) in (4, 6, 8):
        parts = [value[i:i + 2] for i in range(0, len(value) - 4, 2)] + [value[-4:]]
    else:
        raise ValueError(f"Invalid date `{text}`. {_DATE_HINT}")
    if not 1 <= len(parts) <= 3 or not all(p.isdigit() for p in parts) or len(parts[-1]) != 4:
        raise ValueError(f"Invalid date `{text}`. {_DATE_HINT}")
    numbers = [int(p) for p in parts]
    year = numbers[-1]
    month = numbers[-2] if len(numbers) >= 2 else 1
    day = numbers[-3] if len(numbers) == 3 else 1
    try:
        return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())
    except ValueError as exc:
        raise ValueError(f"Invalid date `{text}`. {_DATE_HINT}") from exc


def _user_rating(handle: str) -> list[dict]:
    try:
        return cf_api.get_user_rating(handle)
    except cf_api.UserNotFoundError as exc:
        raise ValueError(f"Codeforces user `{handle}` was not found.") from exc


# ---------------------------------------------------------------------------
# /perf
# ---------------------------------------------------------------------------


def perf(handle: str, contest_id: int, cache: RatingCache) -> discord.Embed:
    """Embed describing every rated and virtual participation in one contest."""
    if contest_id <= 0:
        raise ValueError("Contest ID must be positive.")
    if contest_id >= GYM_ID_THRESHOLD:
        raise ValueError("Gym contests have no rated field to compare against.")
    history = _user_rating(handle)
    rated = next((h for h in history if h.get("contestId") == contest_id), None)
    try:
        submissions = cf_api.get_contest_status(contest_id, handle)
    except cf_api.ContestNotFoundError as exc:
        raise ValueError(f"Contest {contest_id} was not found.") from exc
    virtual_runs = group_virtuals(submissions).get(contest_id, {})
    if rated is None and not virtual_runs:
        raise ValueError(
            f"`{handle}` has no rated or virtual participation in contest {contest_id}."
        )

    display_handle = rated["handle"] if rated else next(
        iter(virtual_runs.values()))[0]["author"]["members"][0]["handle"]
    contest_name = rated["contestName"] if rated else f"Contest {contest_id}"
    fields: list[tuple[str, str]] = []
    perfs: list[float] = []

    if rated is not None:
        delta = rated["newRating"] - rated["oldRating"]
        try:
            value = rated_perf(handle, contest_id, cache)
            perfs.append(value)
            perf_text = f"**{format_perf(value)}**"
        except PerfUnavailable as exc:
            perf_text = f"unavailable ({exc})"
        fields.append((
            "Contest",
            f"Perf: {perf_text}\nRank: {rated['rank']:,}\n"
            f"Rating: {rated['oldRating']} → {rated['newRating']} ({delta:+d})",
        ))

    if virtual_runs:
        results, _ = virtual_perfs(handle, {contest_id: virtual_runs}, cache)
        for number, result in enumerate(sorted(results, key=lambda r: r.start_time), 1):
            if not result.contest_name.startswith("Contest "):
                contest_name = result.contest_name
            name = "Virtual" if len(results) == 1 else f"Virtual #{number}"
            if result.perf is None:
                fields.append((name, f"Perf: unavailable ({result.skip_reason})"))
                continue
            perfs.append(result.perf)
            score = f"{result.points:g}" + (
                f", penalty {result.penalty}" if result.penalty is not None else ""
            )
            fields.append((
                name,
                f"Perf: **{format_perf(result.perf)}** (estimated)\n"
                f"Equivalent rank: {result.rank:,}\nScore: {score}\n"
                f"Started: {_format_date(result.start_time)}",
            ))

    best = max(perfs) if perfs else None
    embed = discord.Embed(
        title=contest_name,
        url=f"https://codeforces.com/contest/{contest_id}",
        description=f"[{display_handle}]({_profile_url(display_handle)})",
        color=RANK_COLORS.get(rank_name(best), 0x808080) if best is not None else 0x808080,
    )
    for name, value in fields:
        embed.add_field(name=name, value=value, inline=True)
    return embed


# ---------------------------------------------------------------------------
# /perfgraph
# ---------------------------------------------------------------------------

INCLUDE_CHOICES = ("both", "rated", "virtual")
# Slow requests (2s each) allowed per graph before the rest is left for next time.
GRAPH_FETCH_BUDGET = 60
INFINITE_PERF_FLOOR = 3000
INFINITE_PERF_MARGIN = 200
# TLE's rating_color_cycler: first color for contests, second for virtuals.
CONTEST_COLOR = "#5d4dff"
VIRTUAL_COLOR = "#009ccc"
# TLE's RATED_RANKS bands, including its 4000+ tier.
GRAPH_BANDS = RANK_BANDS[:-1] + (
    (3000, 4000, "legendary grandmaster", "#AA0000"),
    (4000, 10**9, "tourist", "#330000"),
)


@dataclass(frozen=True)
class GraphPoint:
    timestamp: int
    perf: float
    virtual: bool


@dataclass
class GraphResult:
    png: bytes
    note: str
    points: list[GraphPoint]
    labels: list[str]


def plotted_points(points: list[GraphPoint]) -> list[GraphPoint]:
    """Chronological points with infinite perfs drawn at a fixed height."""
    finite = [p.perf for p in points if not math.isinf(p.perf)]
    ceiling = max(INFINITE_PERF_FLOOR, max(finite, default=-math.inf) + INFINITE_PERF_MARGIN)
    return sorted(
        (
            GraphPoint(p.timestamp, ceiling if math.isinf(p.perf) else p.perf, p.virtual)
            for p in points
        ),
        key=lambda p: (p.timestamp, p.virtual),
    )


def perfgraph(
    handle: str,
    cache: RatingCache,
    include: str = "both",
    date_from: str | None = None,
    date_to: str | None = None,
) -> GraphResult:
    if include not in INCLUDE_CHOICES:
        raise ValueError(f"`include` must be one of {', '.join(INCLUDE_CHOICES)}.")
    lo = parse_cf_date(date_from) if date_from else None
    hi = parse_cf_date(date_to) if date_to else None
    if lo is not None and hi is not None and lo >= hi:
        raise ValueError("`from` must be earlier than `to`.")

    def in_range(timestamp: int) -> bool:
        return (lo is None or timestamp >= lo) and (hi is None or timestamp < hi)

    budget = FetchBudget(GRAPH_FETCH_BUDGET)
    history = _user_rating(handle)
    display_handle = history[-1]["handle"] if history else handle
    current = history[-1]["newRating"] if history else None
    points: list[GraphPoint] = []
    skipped = pending = 0

    if include in ("both", "rated"):
        for entry in history:
            when = entry["ratingUpdateTimeSeconds"]
            if not in_range(when):
                continue
            try:
                value = rated_perf(handle, entry["contestId"], cache, budget)
            except BudgetExhausted:
                pending += 1
                continue
            except PerfUnavailable:
                skipped += 1
                continue
            points.append(GraphPoint(when, value, False))

    if include in ("both", "virtual"):
        try:
            submissions = cf_api.get_user_status(handle)
        except cf_api.UserNotFoundError as exc:
            raise ValueError(f"Codeforces user `{handle}` was not found.") from exc
        groups = {
            cid: {start: subs for start, subs in starts.items() if in_range(start)}
            for cid, starts in group_virtuals(submissions).items()
        }
        groups = {cid: starts for cid, starts in groups.items() if starts}
        results, waiting = virtual_perfs(handle, groups, cache, budget)
        pending += waiting
        for result in results:
            if result.perf is None:
                skipped += 1
            else:
                points.append(GraphPoint(result.start_time, result.perf, True))
        if submissions and not history:
            display_handle = submissions[0]["author"]["members"][0]["handle"]

    if not points:
        what = {"both": "rated or virtual", "rated": "rated", "virtual": "virtual"}[include]
        extra = (
            f" {pending} still need to be downloaded; try again in a few minutes."
            if pending else ""
        )
        raise ValueError(
            f"`{display_handle}` has no {what} contest performances to plot"
            f"{' in that date range' if lo or hi else ''}.{extra}"
        )

    infinite = sum(math.isinf(p.perf) for p in points)
    plotted = plotted_points(points)
    labels = []
    if any(not p.virtual for p in plotted):
        labels.append(f"{display_handle} ({current if current is not None else 'Unrated'})")
    if any(p.virtual for p in plotted):
        labels.append(f"{display_handle} (virtual)")
    png = render_perf_graph(plotted, labels)

    notes = []
    if infinite:
        notes.append(f"Infinite performances (rank 1): {infinite}")
    if skipped:
        notes.append(f"{skipped} participation(s) skipped (unrated, unsupported or unavailable).")
    if pending:
        notes.append(
            f"{pending} participation(s) still need data from Codeforces; "
            "run the command again to include them."
        )
    if any(p.virtual for p in plotted):
        notes.append("Virtual performances are estimates.")
    return GraphResult(png, "\n".join(notes), plotted, labels)


# matplotlib's rc_context mutates global state; one render at a time.
_RENDER_LOCK = threading.Lock()


def _tle_style() -> dict:
    """TLE's theme: sns.set() then darkgrid with its axes overrides, 7x3.5in."""
    import seaborn as sns

    style = {}
    style.update(sns.plotting_context("notebook"))
    style.update(sns.axes_style("darkgrid", {
        "axes.edgecolor": "#A0A0C5",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }))
    style["figure.figsize"] = (7.0, 3.5)
    return style


def render_perf_graph(points: list[GraphPoint], labels: list[str]) -> bytes:
    """Render like TLE's ;plot perf, with one line through every participation."""
    import matplotlib

    matplotlib.use("agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    when = [datetime.fromtimestamp(p.timestamp, timezone.utc).replace(tzinfo=None)
            for p in points]
    values = [p.perf for p in points]
    marker_style = dict(marker="o", markersize=3, markerfacecolor="white",
                        markeredgewidth=0.5)

    with _RENDER_LOCK, plt.rc_context(_tle_style()):
        fig, ax = plt.subplots()
        try:
            # Each segment takes the color of the participation it leads to.
            if len(points) == 1:
                ax.plot(when, values, linestyle="none", **marker_style,
                        color=VIRTUAL_COLOR if points[0].virtual else CONTEST_COLOR)
            for i in range(1, len(points)):
                color = VIRTUAL_COLOR if points[i].virtual else CONTEST_COLOR
                ax.plot(when[i - 1:i + 1], values[i - 1:i + 1], linestyle="-",
                        color=color, solid_capstyle="round")
            for virtual, color in ((False, CONTEST_COLOR), (True, VIRTUAL_COLOR)):
                xs = [w for w, p in zip(when, points) if p.virtual == virtual]
                ys = [p.perf for p in points if p.virtual == virtual]
                if xs:
                    ax.plot(xs, ys, linestyle="none", color=color, zorder=3, **marker_style)

            # gc.plot_rating_bg: rank bands plus background-colored x gridlines.
            ymin, ymax = ax.get_ylim()
            bgcolor = ax.get_facecolor()
            for low, high, _, color in GRAPH_BANDS:
                ax.axhspan(low, high, facecolor=color, alpha=0.8, edgecolor=bgcolor,
                           linewidth=0.5)
            for loc in ax.get_xticks():
                ax.axvline(loc, color=bgcolor, linewidth=0.5)
            ax.set_ylim(ymin, ymax)
            fig.autofmt_xdate()

            handles = []
            if any(not p.virtual for p in points):
                handles.append(Line2D([], [], color=CONTEST_COLOR, **marker_style))
            if any(p.virtual for p in points):
                handles.append(Line2D([], [], color=VIRTUAL_COLOR, **marker_style))
            ax.legend(handles, labels, bbox_to_anchor=(0, 1, 1, 0), loc="lower left",
                      mode="expand", ncol=2)

            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", facecolor=ax.get_facecolor(),
                        bbox_inches="tight", pad_inches=0.25)
            return buffer.getvalue()
        finally:
            plt.close(fig)
