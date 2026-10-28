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
            """)

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
        with self.connection() as conn:
            # Keep members missing from a partial re-fetch; replace corrected values.
            conn.executemany(
                """INSERT INTO rating_change VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(contest_id, handle) DO UPDATE SET
                       rating_update_time=excluded.rating_update_time,
                       old_rating=excluded.old_rating, new_rating=excluded.new_rating""",
                (
                    (
                        contest_id,
                        row.handle.casefold(),
                        int(row.ratingUpdateTimeSeconds),
                        int(row.oldRating),
                        int(row.newRating),
                    )
                    for row in changes.itertuples(index=False)
                ),
            )
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
                   AND (last_checked IS NULL OR status='unavailable' OR end_time >= ?)
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

                contest_id = await asyncio.to_thread(self.cache.next_contest)
                if contest_id is None:
                    await asyncio.sleep(60)
                    continue
                try:
                    changes = await asyncio.to_thread(
                        cf_api.get_rating_changes,
                        contest_id,
                        cancel_event=self.cancel_event,
                    )
                except cf_api.CodeforcesApiError as exc:
                    await asyncio.to_thread(self.cache.mark_fetch_error, contest_id)
                    logger.warning("Could not cache contest %s: %s", contest_id, exc)
                    continue
                if self.cancel_event.is_set():
                    return
                await asyncio.to_thread(
                    self.cache.store_rating_changes, contest_id, changes
                )
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
