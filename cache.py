"""
cache.py — Caches full pipeline results by topic, so identical research
requests within a time window return instantly instead of re-running
the whole pipeline (saves time AND your free-tier Groq quota).

Stored in MongoDB (shared cache across all users — if two different users
research the exact same topic, both benefit from one cached result).
"""

import hashlib
from datetime import datetime, timedelta
from db import get_db

CACHE_TTL_HOURS = 24  # how long a cached result stays valid


def _cache_collection():
    return get_db()["research_cache"]


def _topic_key(topic: str) -> str:
    """Normalizes and hashes the topic so near-identical phrasing/casing still hits the cache."""
    normalized = topic.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def get_cached(topic: str) -> dict:
    """Returns the cached pipeline result dict if a fresh one exists, else None."""
    key = _topic_key(topic)
    doc = _cache_collection().find_one({"key": key})
    if not doc:
        return None

    cached_at = datetime.fromisoformat(doc["cached_at"])
    if datetime.now() - cached_at > timedelta(hours=CACHE_TTL_HOURS):
        return None  # expired

    return doc["result"]


def set_cached(topic: str, result: dict):
    """Stores a pipeline result under this topic's cache key."""
    key = _topic_key(topic)
    _cache_collection().update_one(
        {"key": key},
        {"$set": {"key": key, "topic": topic, "result": result, "cached_at": datetime.now().isoformat()}},
        upsert=True,
    )


if __name__ == "__main__":
    test_topic = "Test caching topic"
    print("Before caching:", get_cached(test_topic))
    set_cached(test_topic, {"report": "Some report content", "critic_verdict": {"verdict": "PASS"}})
    print("After caching:", get_cached(test_topic))
