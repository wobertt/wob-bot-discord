import asyncio
from unittest.mock import AsyncMock

import discord
import pytest
from discord.ext import commands

from conftest import changes
from wob_bot.cf_api import cf_api
from wob_bot.commands import register_commands
from wob_bot.fetchuser import fetchuser
from wob_bot.rating_cache import RatingCache


def seed(cache, *rows):
    cache.store_rating_changes(1, changes(*rows))
    cache.initialize_current_ratings()


def test_initialization_and_empty(cache):
    with pytest.raises(ValueError, match="warming up"):
        fetchuser(1500, cache)
    cache.initialize_current_ratings()
    with pytest.raises(ValueError, match="No rated users"):
        fetchuser(1500, cache)


def test_backfill_corrections_and_restart(cache):
    cache.store_rating_changes(1, changes(("A", 0, 1000, 10)))
    cache.store_rating_changes(2, changes(("a", 1000, 1500, 20)))
    cache.store_rating_changes(3, changes(("a", 1500, 1600, 20)))
    with cache.connection() as conn:
        conn.execute("DELETE FROM current_rating")
    cache.initialize_current_ratings()
    assert cache.nearest_users(0) == [("a", 1600)]
    cache.store_rating_changes(1, changes(("a", 0, 1100, 10)))
    assert cache.nearest_users(0) == [("a", 1600)]
    cache.store_rating_changes(3, changes(("a", 1500, 1700, 20)))
    assert cache.nearest_users(0) == [("a", 1700)]
    # Correcting a timestamp backwards must reveal the next-latest contest.
    cache.store_rating_changes(3, changes(("a", 1500, 1700, 15)))
    assert cache.nearest_users(0) == [("a", 1500)]
    reopened = RatingCache(cache.path)
    reopened.initialize_current_ratings()
    assert reopened.nearest_users(0) == [("a", 1500)]


def test_index_initialization_does_not_repeat(cache):
    cache.initialize_current_ratings()
    with cache.connection() as conn:
        conn.execute("INSERT INTO current_rating VALUES ('a', 1234, 1, 1)")
    cache.initialize_current_ratings()
    assert cache.nearest_users(1234) == [("a", 1234)]


def test_nearest_and_random_ties(cache, monkeypatch):
    seed(cache, ("a", 0, 1000, 10), ("b", 0, 1200, 10), ("c", 0, 1200, 10))
    shuffled = []
    def reverse(items):
        shuffled.append(list(items))
        items.reverse()
    monkeypatch.setattr("wob_bot.rating_cache.random.shuffle", reverse)
    assert cache.nearest_users(1200)[0][1] == 1200
    tied = cache.nearest_users(1100)
    assert len(set(tied)) == 3
    assert set(shuffled[-1]) == {("a", 1000), ("b", 1200), ("c", 1200)}
    assert cache.nearest_users(-100)[0] == ("a", 1000)
    assert cache.nearest_users(9999)[0][1] == 1200
    assert cache.nearest_users(1050)[0] == ("a", 1000)


@pytest.mark.parametrize("requested,description", [(1500, "Exact match"), (1600, "Rating difference: -100")])
def test_embed(cache, monkeypatch, requested, description):
    seed(cache, ("alice", 0, 1500, 10))
    monkeypatch.setattr(cf_api, "get_user_info", lambda handle: {
        "handle": "Alice", "rating": 1500, "rank": "specialist", "titlePhoto": "//example.org/avatar.png"
    })
    embed = fetchuser(requested, cache)
    assert embed.title == "Alice"
    assert embed.url == "https://codeforces.com/profile/Alice"
    assert embed.description == description
    assert embed.color.value == 0x03A89E
    assert [f.value for f in embed.fields] == ["Specialist", "1500", str(requested)]
    assert embed.thumbnail.url == "https://example.org/avatar.png"


def test_retry_changed_and_missing_accounts(cache, monkeypatch):
    seed(cache, ("a", 0, 1000, 10), ("b", 0, 1100, 10), ("c", 0, 1200, 10))
    seen = []
    def profile(handle):
        seen.append(handle)
        if handle == "a":
            return {"handle": handle, "rating": 2000}
        if handle == "b":
            raise cf_api.UserNotFoundError("not found")
        return {"handle": handle, "rating": 1200}
    monkeypatch.setattr(cf_api, "get_user_info", profile)
    assert fetchuser(1000, cache).title == "c"
    assert seen == ["a", "b", "c"]
    assert cache.nearest_users(1000)[0] == ("a", 1000)


def test_retry_limit_and_service_failure(cache, monkeypatch):
    seed(cache, *((str(i), 0, 1000 + i, 10) for i in range(8)))
    seen = []
    def changed(handle):
        seen.append(handle)
        return {"rating": 2000}
    monkeypatch.setattr(cf_api, "get_user_info", changed)
    with pytest.raises(ValueError, match="updating"):
        fetchuser(1000, cache)
    assert len(seen) == len(set(seen)) == 5
    def unavailable(handle):
        raise cf_api.CodeforcesApiError("Unavailable")
    monkeypatch.setattr(cf_api, "get_user_info", unavailable)
    with pytest.raises(cf_api.CodeforcesApiError, match="Unavailable"):
        fetchuser(1000, cache)


def test_command(cache, monkeypatch):
    async def run():
        bot = commands.Bot(command_prefix="!", intents=discord.Intents.none())
        register_commands(bot, cache)
        command = bot.tree.get_command("fetchuser")
        assert command.parameters[0].type == discord.AppCommandOptionType.integer
        interaction = AsyncMock()
        await command.callback(interaction, 1000)
        interaction.response.defer.assert_awaited_once_with(thinking=True)
        assert "warming up" in interaction.followup.send.call_args.args[0]
        seed(cache, ("a", 0, 1000, 10))
        monkeypatch.setattr(cf_api, "get_user_info", lambda h: {"handle": h, "rating": 1000})
        interaction = AsyncMock()
        await command.callback(interaction, 1000)
        assert interaction.followup.send.call_args.kwargs["embed"].title == "a"
        await bot.close()
    asyncio.run(run())


def test_user_info_helper(monkeypatch):
    seen = []
    def request(endpoint):
        seen.append(endpoint)
        return [{"handle": "a", "rating": 1000}]
    monkeypatch.setattr(cf_api, "make_request", request)
    assert cf_api.get_user_info("a+b")["rating"] == 1000
    assert seen == ["user.info?handles=a%2Bb"]
    monkeypatch.setattr(cf_api, "make_request", lambda _: [])
    with pytest.raises(cf_api.CodeforcesApiError, match="invalid"):
        cf_api.get_user_info("a")
