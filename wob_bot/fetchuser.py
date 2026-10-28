"""Find a nearby cached user and verify their current Codeforces profile."""

from urllib.parse import quote

import discord

from .cf_api import cf_api
from .rating_cache import RatingCache

RANK_COLORS = {
    "newbie": 0x808080,
    "pupil": 0x008000,
    "specialist": 0x03A89E,
    "expert": 0x0000FF,
    "candidate master": 0xAA00AA,
    "master": 0xFF8C00,
    "international master": 0xFF8C00,
    "grandmaster": 0xFF0000,
    "international grandmaster": 0xFF0000,
    "legendary grandmaster": 0xFF0000,
}


def fetchuser(rating: int, cache: RatingCache) -> discord.Embed:
    candidates = cache.nearest_users(rating, limit=5)
    if not candidates:
        raise ValueError("No rated users are available in the cache yet. Try again later.")
    for handle, cached_rating in candidates:
        try:
            user = cf_api.get_user_info(handle)
        except cf_api.UserNotFoundError:
            continue
        if user.get("rating") != cached_rating:
            continue
        if not isinstance(user.get("handle"), str) or not user["handle"]:
            raise cf_api.CodeforcesApiError("Codeforces returned an invalid user profile.")
        rank = user.get("rank") or "Unknown"
        difference = cached_rating - rating
        embed = discord.Embed(
            title=user["handle"],
            url=f"https://codeforces.com/profile/{quote(user['handle'], safe='')}",
            description="Exact match" if difference == 0 else f"Rating difference: {difference:+d}",
            color=RANK_COLORS.get(rank.lower(), 0x808080),
        )
        embed.add_field(name="Rank", value=rank.title())
        embed.add_field(name="Current rating", value=str(cached_rating))
        embed.add_field(name="Requested rating", value=str(rating))
        avatar = user.get("titlePhoto") or user.get("avatar")
        if isinstance(avatar, str):
            if avatar.startswith("//"):
                avatar = "https:" + avatar
            if avatar.startswith(("https://", "http://")):
                embed.set_thumbnail(url=avatar)
        embed.set_footer(text="Selected from cached ratings • Current rating verified with Codeforces")
        return embed
    raise ValueError("Cached ratings are updating or users are unavailable. Please try again later.")
