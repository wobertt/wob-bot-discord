import asyncio
from concurrent.futures import ThreadPoolExecutor

from wob_bot.cf_api import cf_api
from wob_bot.rating_cache import (
    RatingCache,
    RatingCacheUpdater,
    RECENT_WINDOW,
    REFRESH_INTERVAL,
    UNAVAILABLE_RETRY_INTERVAL,
)

from conftest import changes, contest


def test_history_uses_latest_change_strictly_before_start_and_survives_restart(cache):
    cache.store_contests([contest(1, 100), contest(2, 200), contest(3, 300)])
    cache.store_rating_changes(1, changes(("Alice", 0, 1200, 110)), now=400)
    cache.store_rating_changes(2, changes(("ALICE", 1200, 1600, 210)), now=400)
    cache.store_rating_changes(3, changes(("Alice", 1600, 2200, 310)), now=400)

    reopened = RatingCache(cache.path)
    snapshot = reopened.ratings_before(["aLiCe", "missing"], 310)
    assert snapshot.history_ready
    assert snapshot.ratings == {"alice": 1600}
    assert reopened.published_ratings(3) == {"alice": 1600}
    assert reopened.ratings_before(["alice"], 110).ratings == {}


def test_missing_or_failed_download_does_not_claim_complete_history(cache):
    cache.store_contests([contest(1), contest(2, 200)])
    cache.store_rating_changes(1, changes(("a", 0, 1200, 110)), now=300)
    cache.mark_fetch_error(2, now=300)
    snapshot = cache.ratings_before(["a"], 250)
    assert snapshot.ratings == {"a": 1200}
    assert not snapshot.history_ready
    assert cache.ratings_before(["a"], 150).history_ready
    assert cache.progress() == (1, 2)


def test_no_catalog_cannot_claim_complete_history(cache):
    cache.record_contest(contest())
    cache.store_rating_changes(1, changes(("a", 0, 1200, 110)))
    assert not cache.ratings_before(["a"], 200).history_ready


def test_empty_is_checkpointed_and_partial_refetches_merge_corrections(cache):
    cache.store_contests([contest(1), contest(2, 200)])
    cache.store_rating_changes(
        1, changes(("Alice", 1000, 1200, 110), ("bob", 900, 1100, 110))
    )
    cache.store_rating_changes(1, changes(("ALICE", 1000, 1250, 110)))
    cache.store_rating_changes(1, changes())
    cache.store_rating_changes(2, changes())
    assert cache.ratings_before(["alice", "bob"], 300).ratings == {
        "alice": 1250,
        "bob": 1100,
    }
    assert cache.progress() == (2, 2)
    assert cache.ratings_before(["alice"], 300).history_ready


def test_failed_transaction_keeps_previous_snapshot_and_checkpoint(cache):
    import pytest

    cache.store_contests([contest()])
    cache.store_rating_changes(1, changes(("a", 1000, 1200, 110)), now=200)
    with pytest.raises(ValueError):
        cache.store_rating_changes(
            1, changes(("a", 1000, 1500, 110), ("b", 0, "invalid", 110)), now=300
        )
    assert cache.ratings_before(["a"], 200).ratings == {"a": 1200}
    with cache.connection() as conn:
        assert conn.execute("SELECT last_checked FROM contest").fetchone()[0] == 200


def test_scheduler_resumes_and_rechecks_recent_and_unavailable_contests(cache):
    now = RECENT_WINDOW * 2
    cache.store_contests([contest(1, 100), contest(2, 200), contest(3, now - 20)])
    assert cache.next_contest(now=now) == 1
    cache.store_rating_changes(1, changes(), now=now)
    assert RatingCache(cache.path).next_contest(now=now) == 2
    cache.mark_fetch_error(2, now=now)
    assert cache.next_contest(now=now) == 3
    cache.store_rating_changes(3, changes(("a", 1200, 1300, now - 10)), now=now)
    assert cache.next_contest(now=now) is None
    # Recent verification gets priority over the remaining backfill.
    assert cache.next_contest(now=now + REFRESH_INTERVAL) == 3
    cache.store_rating_changes(3, changes(), now=now + REFRESH_INTERVAL)
    assert cache.next_contest(now=now + REFRESH_INTERVAL) == 2
    cache.store_rating_changes(
        2, changes(("a", 0, 1200, 210)), now=now + REFRESH_INTERVAL
    )
    assert cache.next_contest(now=now + UNAVAILABLE_RETRY_INTERVAL) == 3
    cache.store_rating_changes(3, changes(), now=now + UNAVAILABLE_RETRY_INTERVAL)
    assert cache.next_contest(now=now + UNAVAILABLE_RETRY_INTERVAL) == 1


def test_concurrent_connections_can_read_and_write(cache):
    cache.store_contests([contest()])
    cache.store_rating_changes(1, changes(("a", 1000, 1200, 110)))

    def operation(index):
        if index % 2:
            cache.store_rating_changes(1, changes(("a", 1000, 1200 + index, 110)))
        else:
            snapshot = cache.ratings_before(["a"], 200)
            assert snapshot.history_ready
            assert snapshot.ratings["a"] >= 1200

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(operation, range(40)))


def test_updater_automatically_populates_and_resumes(cache, monkeypatch):
    async def run():
        updater = RatingCacheUpdater(cache)
        monkeypatch.setattr(
            cf_api, "get_contests", lambda **kwargs: [contest(1), contest(2, 200)]
        )
        fetched = []

        def fetch(contest_id, **kwargs):
            fetched.append(contest_id)
            return changes(("a", 1000, 1200, 110))

        monkeypatch.setattr(cf_api, "get_rating_changes", fetch)
        task = asyncio.create_task(updater.run())
        try:
            async with asyncio.timeout(3):
                while cache.progress() != (2, 2):
                    await asyncio.sleep(0.01)
        finally:
            updater.stop()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        assert fetched == [1, 2]
        assert RatingCache(cache.path).next_contest() is None

    asyncio.run(run())


def test_updater_checkpoints_successes_and_skips_failed_contests_for_retry(
    cache, monkeypatch
):
    async def run():
        updater = RatingCacheUpdater(cache)
        monkeypatch.setattr(
            cf_api, "get_contests", lambda **kwargs: [contest(1), contest(2, 200)]
        )
        fetched = []

        def fetch(contest_id, **kwargs):
            fetched.append(contest_id)
            if contest_id == 1:
                raise cf_api.CodeforcesApiError("Temporary error")
            return changes(("a", 1000, 1200, 210))

        monkeypatch.setattr(cf_api, "get_rating_changes", fetch)
        task = asyncio.create_task(updater.run())
        try:
            async with asyncio.timeout(3):
                while cache.progress() != (1, 2):
                    await asyncio.sleep(0.01)
        finally:
            updater.stop()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        assert fetched == [1, 2]
        assert not cache.ratings_before(["a"], 300).history_ready
        assert cache.published_ratings(2) == {"a": 1000}

    asyncio.run(run())


def test_shutdown_during_fetch_does_not_save_a_partial_download(cache, monkeypatch):
    async def run():
        updater = RatingCacheUpdater(cache)
        monkeypatch.setattr(cf_api, "get_contests", lambda **kwargs: [contest()])

        def fetch(*args, **kwargs):
            updater.stop()
            return changes(("a", 1000, 1200, 110))

        monkeypatch.setattr(cf_api, "get_rating_changes", fetch)
        async with asyncio.timeout(3):
            await updater.run()
        assert cache.progress() == (0, 1)
        assert cache.published_ratings(1) == {}

    asyncio.run(run())


def test_inaccessible_counts_as_checked_and_can_recover(cache):
    now = RECENT_WINDOW * 2
    cache.store_contests([contest(1), contest(2)])
    cache.store_rating_changes(1, changes(("a", 0, 1200, 110)), now=now)
    assert not cache.ratings_before(["a"], 200).history_ready
    cache.mark_inaccessible(2, now=now)
    assert cache.progress() == (2, 2)
    assert cache.ratings_before(["a"], 200).history_ready
    reopened = RatingCache(cache.path)
    assert reopened.next_contest(now=now + 1) is None
    assert reopened.next_contest(now=now + UNAVAILABLE_RETRY_INTERVAL) == 2
    reopened.store_rating_changes(2, changes(("a", 1200, 1400, 150)))
    assert reopened.ratings_before(["a"], 200).ratings == {"a": 1400}
    reopened.mark_inaccessible(2)
    assert reopened.published_ratings(2) == {"a": 1200}


def test_updater_skips_inaccessible_and_reports_ready(cache, monkeypatch, caplog):
    import logging

    async def run():
        updater = RatingCacheUpdater(cache)
        monkeypatch.setattr(cf_api, "get_contests", lambda **kw: [contest()])
        def missing(*args, **kw):
            raise cf_api.ContestNotFoundError("Contest not found")
        monkeypatch.setattr(cf_api, "get_rating_changes", missing)
        task = asyncio.create_task(updater.run())
        try:
            async with asyncio.timeout(3):
                while "Rating cache ready" not in caplog.text:
                    await asyncio.sleep(0.01)
        finally:
            updater.stop()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        assert cache.progress() == (1, 1)
        assert "Could not cache" not in caplog.text
    with caplog.at_level(logging.INFO):
        asyncio.run(run())


def test_old_schema_is_migrated_and_queued_for_rank_backfill(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE contest (id INTEGER PRIMARY KEY, start_time INTEGER NOT NULL,
            end_time INTEGER NOT NULL, phase TEXT NOT NULL, last_checked REAL,
            status TEXT NOT NULL DEFAULT 'pending', retry_after REAL NOT NULL DEFAULT 0);
        CREATE TABLE rating_change (contest_id INTEGER NOT NULL, handle TEXT NOT NULL,
            rating_update_time INTEGER NOT NULL, old_rating INTEGER NOT NULL,
            new_rating INTEGER NOT NULL, PRIMARY KEY (contest_id, handle));
        INSERT INTO contest VALUES (1, 100, 110, 'FINISHED', 50, 'available', 0);
        INSERT INTO contest VALUES (2, 200, 210, 'FINISHED', 50, 'unavailable', 0);
        INSERT INTO rating_change VALUES (1, 'a', 110, 1000, 1200);
    """)
    conn.commit()
    conn.close()

    cache = RatingCache(path)
    assert not cache.contest_has_ranks(1)
    assert cache.next_rank_backfill(now=0) == 1
    assert cache.rank_backfill_remaining() == 1
    frame = changes(("A", 1000, 1200, 110)).assign(rank=[1])
    cache.store_rating_changes(1, frame, now=0)
    assert cache.contest_has_ranks(1, "A")
    assert cache.next_rank_backfill(now=10**12) is None
    assert cache.contest_field(1)["rank"].tolist() == [1]
    # Reopening an already-migrated cache is a no-op.
    assert RatingCache(path).contest_has_ranks(1)


def test_refetch_without_rank_keeps_stored_rank_and_clears_perf_memo(cache):
    cache.store_contests([contest(1)])
    cache.store_rating_changes(1, changes(("a", 1000, 1200, 110)).assign(rank=[3]))
    cache.store_rated_perf(1, "a", 1234.0)
    cache.store_rating_changes(1, changes(("a", 1000, 1250, 110)))
    assert cache.contest_field(1)["rank"].tolist() == [3]
    assert cache.get_rated_perf(1, "a") is None


def test_updater_backfills_ranks_when_idle(cache, monkeypatch):
    cache.store_contests([contest(1)])
    cache.store_rating_changes(1, changes(("a", 1000, 1200, 110)), now=0)
    with cache.connection() as conn:
        conn.execute("UPDATE contest SET ranks_stored=0, retry_after=0")

    async def run():
        updater = RatingCacheUpdater(cache)
        monkeypatch.setattr(cf_api, "get_contests", lambda **kwargs: [contest(1)])
        monkeypatch.setattr(
            cf_api,
            "get_rating_changes",
            lambda contest_id, **kwargs: changes(("a", 1000, 1200, 110)).assign(rank=[1]),
        )
        task = asyncio.create_task(updater.run())
        try:
            async with asyncio.timeout(3):
                while not cache.contest_has_ranks(1):
                    await asyncio.sleep(0.01)
        finally:
            updater.stop()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        assert cache.rank_backfill_remaining() == 0

    asyncio.run(run())
