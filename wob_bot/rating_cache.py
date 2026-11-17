"""Persistent Codeforces rating history and its resumable background updater."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import logging
import os
import random
from pathlib import Path
import sqlite3
import threading
import time
from typing import Iterator

import pandas as pd

from .cf_api import cf_api

logger = logging.getLogger(__name__)
REFRESH_INTERVAL = 10 * 60
RECENT_WINDOW = 7 * 24 * 60 * 60
UNAVAILABLE_RETRY_INTERVAL = 24 * 60 * 60
# Codeforces' newcomer system: hidden rating 1400, shown rating boosted over 6 contests.
NEW_RATING_SYSTEM_CONTEST = 1360
NEWCOMER_OFFSETS = (1400, 900, 550, 300, 150, 50)


@dataclass(frozen=True)
class HistoricalRatings:
    ratings: dict[str, int]
    history_ready: bool


class RatingCache:
    """Small connections per operation let command and updater threads share SQLite."""

    def __init__(self, path: str | Path | None = None) -> None:
        default = Path(__file__).resolve().parent.parent / "data" / "codeforces.sqlite3"
        self.path = Path(path or os.getenv("CF_CACHE_PATH") or default).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS contest (
                    id INTEGER PRIMARY KEY,
                    start_time INTEGER NOT NULL,
                    end_time INTEGER NOT NULL,
                    phase TEXT NOT NULL,
                    last_checked REAL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    retry_after REAL NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS rating_change (
                    contest_id INTEGER NOT NULL,
                    handle TEXT NOT NULL,
                    rating_update_time INTEGER NOT NULL,
                    old_rating INTEGER NOT NULL,
                    new_rating INTEGER NOT NULL,
                    PRIMARY KEY (contest_id, handle)
                );
                CREATE INDEX IF NOT EXISTS rating_history
                    ON rating_change(handle, rating_update_time DESC);
                CREATE TABLE IF NOT EXISTS current_rating (
                    handle TEXT PRIMARY KEY, rating INTEGER NOT NULL,
                    rating_update_time INTEGER NOT NULL, contest_id INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS current_rating_value ON current_rating(rating);
                CREATE TABLE IF NOT EXISTS cache_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rated_perf (
                    contest_id INTEGER NOT NULL,
                    handle TEXT NOT NULL,
                    perf REAL NOT NULL,
                    PRIMARY KEY (contest_id, handle)
                );
                CREATE TABLE IF NOT EXISTS virtual_perf (
                    handle TEXT NOT NULL,
                    contest_id INTEGER NOT NULL,
                    start_time INTEGER NOT NULL,
                    contest_name TEXT NOT NULL,
                    rank INTEGER,
                    points REAL,
                    penalty INTEGER,
                    perf REAL,
                    skip_reason TEXT,
                    PRIMARY KEY (handle, contest_id, start_time)
                );
            """)
            # Columns added after the first release; ALTER keeps existing caches.
            for table, column, definition in (
                ("rating_change", "rank", "INTEGER"),
                ("contest", "ranks_stored", "INTEGER NOT NULL DEFAULT 0"),
            ):
                columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
                if column not in columns:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def initialize_current_ratings(self) -> None:
        """Build once atomically; WAL readers can still report warming up."""
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM cache_state WHERE key='current_ratings_ready'").fetchone():
                return
            conn.execute("DELETE FROM current_rating")
            conn.execute("""INSERT INTO current_rating
                SELECT handle, new_rating, rating_update_time, contest_id FROM (
                    SELECT *, row_number() OVER (
                        PARTITION BY handle ORDER BY rating_update_time DESC, contest_id DESC
                    ) AS position FROM rating_change
                ) WHERE position=1""")
            conn.execute("INSERT INTO cache_state VALUES ('current_ratings_ready', '1')")

    def nearest_users(self, rating: int, limit: int = 5) -> list[tuple[str, int]]:
        with self.connection() as conn:
            conn.execute("BEGIN")
            if not conn.execute("SELECT 1 FROM cache_state WHERE key='current_ratings_ready'").fetchone():
                raise ValueError("The user lookup is warming up. Please try again shortly.")
            lower = conn.execute(
                "SELECT DISTINCT rating FROM current_rating WHERE rating <= ? ORDER BY rating DESC LIMIT ?",
                (rating, limit),
            ).fetchall()
            upper = conn.execute(
                "SELECT DISTINCT rating FROM current_rating WHERE rating > ? ORDER BY rating LIMIT ?",
                (rating, limit),
            ).fetchall()
            distances = {}
            for (value,) in lower + upper:
                distances.setdefault(abs(value - rating), []).append(value)
            result = []
            for distance in sorted(distances):
                candidates = []
                for value in distances[distance]:
                    candidates.extend(conn.execute(
                        "SELECT handle, rating FROM current_rating WHERE rating=?", (value,)
                    ).fetchall())
                random.shuffle(candidates)
                result.extend(candidates[:limit - len(result)])
                if len(result) == limit:
                    break
            return result

    @staticmethod
    def _record_contests(conn: sqlite3.Connection, contests: list[dict]) -> None:
        conn.executemany(
            """INSERT INTO contest(id, start_time, end_time, phase) VALUES (?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   start_time=excluded.start_time, end_time=excluded.end_time,
                   phase=excluded.phase""",
            (
                (
                    contest["id"],
                    contest["startTimeSeconds"],
                    contest["startTimeSeconds"] + contest.get("durationSeconds", 0),
                    contest["phase"],
                )
                for contest in contests
                if contest.get("startTimeSeconds") is not None
            ),
        )

    def record_contest(self, contest: dict) -> None:
        with self.connection() as conn:
            self._record_contests(conn, [contest])

    def store_contests(self, contests: list[dict]) -> None:
        with self.connection() as conn:
            self._record_contests(conn, contests)
            conn.execute(
                "INSERT OR REPLACE INTO cache_state VALUES ('catalog_updated_at', ?)",
                (str(time.time()),),
            )

    def store_rating_changes(
        self, contest_id: int, changes: pd.DataFrame, *, now: float | None = None
    ) -> None:
        now = time.time() if now is None else now
        has_rank = "rank" in changes.columns
        with self.connection() as conn:
            # Keep members missing from a partial re-fetch; replace corrected values.
            conn.executemany(
                """INSERT INTO rating_change(contest_id, handle, rating_update_time,
                       old_rating, new_rating, rank) VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(contest_id, handle) DO UPDATE SET
                       rating_update_time=excluded.rating_update_time,
                       old_rating=excluded.old_rating, new_rating=excluded.new_rating,
                       rank=coalesce(excluded.rank, rating_change.rank)""",
                (
                    (
                        contest_id,
                        row.handle.casefold(),
                        int(row.ratingUpdateTimeSeconds),
                        int(row.oldRating),
                        int(row.newRating),
                        int(row.rank) if has_rank and pd.notna(row.rank) else None,
                    )
                    for row in changes.itertuples(index=False)
                ),
            )
            # Any download by this version counts as rank-aware, so the rank
            # backfill only revisits contests stored before ranks were kept.
            conn.execute("UPDATE contest SET ranks_stored=1 WHERE id=?", (contest_id,))
            conn.execute("DELETE FROM rated_perf WHERE contest_id=?", (contest_id,))
            # Recompute affected handles to handle corrected timestamps and
            # out-of-order contest refreshes without regressing current ratings.
            conn.executemany(
                """INSERT OR REPLACE INTO current_rating
                   SELECT handle, new_rating, rating_update_time, contest_id
                   FROM rating_change WHERE handle=?
                   ORDER BY rating_update_time DESC, contest_id DESC LIMIT 1""",
                ((handle.casefold(),) for handle in set(changes["handle"])),
            )
            has_changes = (
                conn.execute(
                    "SELECT 1 FROM rating_change WHERE contest_id=? LIMIT 1",
                    (contest_id,),
                ).fetchone()
                is not None
            )
            conn.execute(
                """UPDATE contest SET last_checked=?, status=?,
                   retry_after=CASE WHEN end_time >= ? THEN ? ELSE ? END WHERE id=?""",
                (
                    now,
                    "available" if has_changes else "unavailable",
                    now - RECENT_WINDOW,
                    now + REFRESH_INTERVAL,
                    now + UNAVAILABLE_RETRY_INTERVAL,
                    contest_id,
                ),
            )

    def mark_inaccessible(self, contest_id: int, *, now: float | None = None) -> None:
        """Count explicit not-found responses as checked, but retry daily."""
        now = time.time() if now is None else now
        with self.connection() as conn:
            conn.execute(
                """UPDATE contest SET last_checked=?, status='inaccessible',
                   retry_after=? WHERE id=?""",
                (now, now + UNAVAILABLE_RETRY_INTERVAL, contest_id),
            )

    def mark_fetch_error(self, contest_id: int, *, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self.connection() as conn:
            conn.execute(
                "UPDATE contest SET retry_after=? WHERE id=?",
                (now + REFRESH_INTERVAL, contest_id),
            )

    def next_contest(self, *, now: float | None = None) -> int | None:
        now = time.time() if now is None else now
        with self.connection() as conn:
            row = conn.execute(
                """SELECT id FROM contest WHERE phase='FINISHED' AND retry_after <= ?
                   AND (last_checked IS NULL OR status IN ('unavailable', 'inaccessible') OR end_time >= ?)
                   ORDER BY CASE WHEN last_checked IS NOT NULL AND end_time >= ?
                       THEN 0 ELSE 1 END, start_time, id LIMIT 1""",
                (now, now - RECENT_WINDOW, now - RECENT_WINDOW),
            ).fetchone()
            return row[0] if row else None

    def published_ratings(self, contest_id: int) -> dict[str, int]:
        with self.connection() as conn:
            return dict(
                conn.execute(
                    "SELECT handle, old_rating FROM rating_change WHERE contest_id=?",
                    (contest_id,),
                )
            )

    def ratings_before(self, handles: list[str], timestamp: int) -> HistoricalRatings:
        with self.connection() as conn:
            # One read transaction keeps coverage and ratings from the same snapshot.
            conn.execute("BEGIN")
            catalog_loaded = (
                conn.execute(
                    "SELECT 1 FROM cache_state WHERE key='catalog_updated_at'"
                ).fetchone()
                is not None
            )
            missing = conn.execute(
                """SELECT 1 FROM contest WHERE phase='FINISHED' AND start_time < ?
                   AND last_checked IS NULL LIMIT 1""",
                (timestamp,),
            ).fetchone()
            ratings = {}
            for handle in {handle.casefold() for handle in handles}:
                row = conn.execute(
                    """SELECT new_rating FROM rating_change WHERE handle=?
                       AND rating_update_time < ?
                       ORDER BY rating_update_time DESC, contest_id DESC LIMIT 1""",
                    (handle, timestamp),
                ).fetchone()
                if row is not None:
                    ratings[handle] = row[0]
            return HistoricalRatings(ratings, catalog_loaded and missing is None)

    def next_rank_backfill(self, *, now: float | None = None) -> int | None:
        """Oldest downloaded contest whose rows predate the rank column."""
        now = time.time() if now is None else now
        with self.connection() as conn:
            row = conn.execute(
                """SELECT id FROM contest WHERE status='available' AND ranks_stored=0
                   AND retry_after <= ? ORDER BY start_time, id LIMIT 1""",
                (now,),
            ).fetchone()
            return row[0] if row else None

    def rank_backfill_remaining(self) -> int:
        with self.connection() as conn:
            return conn.execute(
                "SELECT count(*) FROM contest WHERE status='available' AND ranks_stored=0"
            ).fetchone()[0]

    def contest_has_ranks(self, contest_id: int, handle: str | None = None) -> bool:
        """Whether cached rows are complete enough to rank this contest's field."""
        with self.connection() as conn:
            total, missing = conn.execute(
                """SELECT count(*), count(*) - count(rank) FROM rating_change
                   WHERE contest_id=?""",
                (contest_id,),
            ).fetchone()
            if total == 0 or missing:
                return False
            if handle is None:
                return True
            return (
                conn.execute(
                    "SELECT 1 FROM rating_change WHERE contest_id=? AND handle=?",
                    (contest_id, handle.casefold()),
                ).fetchone()
                is not None
            )

    def contest_field(self, contest_id: int) -> pd.DataFrame:
        """Rated field with CF ranks and effective pre-contest ratings.

        Newcomers since the 2020 rating system (contest 1360) start with a
        shown rating of 0 and have a hidden rating of shown + NEWCOMER_OFFSETS[k]
        before their k-th rated contest. A first contest that is not shown as 0
        means an established account, which gets no offset (first_contest is
        reported as 0 for those).
        That needs complete earlier history; otherwise fall back to carrot's
        rule that a shown 0 means 1400.
        """
        with self.connection() as conn:
            conn.execute("BEGIN")
            start = conn.execute(
                "SELECT start_time FROM contest WHERE id=?", (contest_id,)
            ).fetchone()
            fix = contest_id >= NEW_RATING_SYSTEM_CONTEST and start is not None
            if fix:
                catalog = conn.execute(
                    "SELECT 1 FROM cache_state WHERE key='catalog_updated_at'"
                ).fetchone()
                missing = conn.execute(
                    """SELECT 1 FROM contest WHERE phase='FINISHED' AND start_time < ?
                       AND last_checked IS NULL LIMIT 1""",
                    (start[0],),
                ).fetchone()
                fix = catalog is not None and missing is None
            if fix:
                rows = conn.execute(
                    """SELECT rc.handle, rc.rank, rc.old_rating,
                           (SELECT count(*) FROM (
                               SELECT 1 FROM rating_change p WHERE p.handle=rc.handle
                               AND p.rating_update_time < rc.rating_update_time LIMIT 6)),
                           (SELECT f.contest_id * (f.old_rating = 0) FROM rating_change f
                               WHERE f.handle=rc.handle
                               ORDER BY f.rating_update_time, f.contest_id LIMIT 1)
                       FROM rating_change rc WHERE rc.contest_id=?""",
                    (contest_id,),
                ).fetchall()
            else:
                rows = [
                    (*row, None, None)
                    for row in conn.execute(
                        """SELECT handle, rank, old_rating FROM rating_change
                           WHERE contest_id=?""",
                        (contest_id,),
                    )
                ]
        field = pd.DataFrame(
            rows, columns=["handle", "rank", "old_rating", "prior", "first_contest"]
        )
        effective = field.old_rating.copy()
        if contest_id >= NEW_RATING_SYSTEM_CONTEST:
            effective[field.old_rating == 0] = NEWCOMER_OFFSETS[0]
        if fix and not field.empty:
            newcomer = (field.first_contest >= NEW_RATING_SYSTEM_CONTEST) & (
                field.prior < len(NEWCOMER_OFFSETS)
            )
            offsets = field.prior[newcomer].astype(int).map(
                lambda k: NEWCOMER_OFFSETS[k]
            )
            effective[newcomer] = field.old_rating[newcomer] + offsets
        field["effective_rating"] = effective.astype(int)
        return field[["handle", "rank", "old_rating", "effective_rating"]]

    def rating_before(self, handle: str, timestamp: int) -> int | None:
        return self.ratings_before([handle], timestamp).ratings.get(handle.casefold())

    def get_rated_perf(self, contest_id: int, handle: str) -> float | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT perf FROM rated_perf WHERE contest_id=? AND handle=?",
                (contest_id, handle.casefold()),
            ).fetchone()
            return row[0] if row else None

    def store_rated_perf(self, contest_id: int, handle: str, perf: float) -> None:
        with self.connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO rated_perf VALUES (?, ?, ?)",
                (contest_id, handle.casefold(), perf),
            )

    def get_virtual_perfs(self, handle: str) -> dict[tuple[int, int], dict]:
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            return {
                (row["contest_id"], row["start_time"]): dict(row)
                for row in conn.execute(
                    "SELECT * FROM virtual_perf WHERE handle=?", (handle.casefold(),)
                )
            }

    def store_virtual_perf(self, handle: str, record: dict) -> None:
        with self.connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO virtual_perf VALUES
                   (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    handle.casefold(),
                    record["contest_id"],
                    record["start_time"],
                    record["contest_name"],
                    record.get("rank"),
                    record.get("points"),
                    record.get("penalty"),
                    record.get("perf"),
                    record.get("skip_reason"),
                ),
            )

    def progress(self) -> tuple[int, int]:
        with self.connection() as conn:
            checked, total = conn.execute(
                """SELECT count(last_checked), count(*) FROM contest WHERE phase='FINISHED'"""
            ).fetchone()
            return checked, total


class RatingCacheUpdater:
    def __init__(self, cache: RatingCache) -> None:
        self.cache = cache
        self.cancel_event = threading.Event()

    def stop(self) -> None:
        self.cancel_event.set()

    async def run(self) -> None:
        """Yield between requests; checkpoints survive cancellation and restarts."""
        next_catalog_refresh = 0.0
        ready_reported = False
        while not self.cancel_event.is_set():
            try:
                await asyncio.to_thread(self.cache.initialize_current_ratings)
                if time.monotonic() >= next_catalog_refresh:
                    contests = await asyncio.to_thread(
                        cf_api.get_contests, cancel_event=self.cancel_event
                    )
                    if self.cancel_event.is_set():
                        return
                    await asyncio.to_thread(self.cache.store_contests, contests)
                    next_catalog_refresh = time.monotonic() + REFRESH_INTERVAL

                checked, total = await asyncio.to_thread(self.cache.progress)
                ready = total > 0 and checked == total
                if ready and not ready_reported:
                    logger.info("Rating cache ready: all %s contests checked; inaccessible contests skipped.", total)
                ready_reported = ready
                contest_id = await asyncio.to_thread(self.cache.next_contest)
                backfill = False
                if contest_id is None:
                    # Idle work: refetch contests cached before ranks were stored.
                    contest_id = await asyncio.to_thread(self.cache.next_rank_backfill)
                    backfill = contest_id is not None
                if contest_id is None:
                    await asyncio.sleep(60)
                    continue
                try:
                    changes = await asyncio.to_thread(
                        cf_api.get_rating_changes,
                        contest_id,
                        cancel_event=self.cancel_event,
                    )
                except cf_api.ContestNotFoundError:
                    if self.cancel_event.is_set():
                        return
                    await asyncio.to_thread(self.cache.mark_inaccessible, contest_id)
                    logger.info("Skipping inaccessible contest %s; rechecking in one day.", contest_id)
                    continue
                except cf_api.CodeforcesApiError as exc:
                    await asyncio.to_thread(self.cache.mark_fetch_error, contest_id)
                    logger.warning("Could not cache contest %s: %s", contest_id, exc)
                    continue
                if self.cancel_event.is_set():
                    return
                await asyncio.to_thread(
                    self.cache.store_rating_changes, contest_id, changes
                )
                if backfill:
                    remaining = await asyncio.to_thread(
                        self.cache.rank_backfill_remaining
                    )
                    logger.info(
                        "Rank backfill: contest %s stored; %s contests remaining.",
                        contest_id,
                        remaining,
                    )
                    continue
                checked, total = await asyncio.to_thread(self.cache.progress)
                logger.info(
                    "Rating cache: %s/%s contests checked; contest %s returned %s changes.",
                    checked,
                    total,
                    contest_id,
                    len(changes),
                )
            except cf_api.RequestCancelled:
                return
            except Exception:
                logger.exception("Rating cache update failed; retrying in one minute.")
                await asyncio.sleep(60)
