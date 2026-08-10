"""
rate_limit.py — Per-user rate limiting, so one user can't burn through the
whole free-tier Groq quota. Tracks requests in MongoDB (works across
restarts, unlike an in-memory counter).
"""

from datetime import datetime, timedelta
from db import get_db

DAILY_LIMIT = 15  # research requests per user per day — adjust as needed


def _usage_collection():
    return get_db()["usage"]


def check_and_record(user_id: str, action: str = "research") -> dict:
    """
    Checks if the user is within their daily limit, and if so, records this
    request. Returns {"allowed": bool, "remaining": int, "limit": int}
    """
    usage = _usage_collection()
    today = datetime.now().strftime("%Y-%m-%d")

    doc = usage.find_one({"user_id": user_id, "date": today, "action": action})
    current_count = doc["count"] if doc else 0

    if current_count >= DAILY_LIMIT:
        return {"allowed": False, "remaining": 0, "limit": DAILY_LIMIT}

    usage.update_one(
        {"user_id": user_id, "date": today, "action": action},
        {"$inc": {"count": 1}, "$setOnInsert": {"created_at": datetime.now().isoformat()}},
        upsert=True,
    )

    return {"allowed": True, "remaining": DAILY_LIMIT - current_count - 1, "limit": DAILY_LIMIT}


def get_usage_today(user_id: str, action: str = "research") -> int:
    """Just checks current usage without incrementing — for displaying to the user."""
    today = datetime.now().strftime("%Y-%m-%d")
    doc = _usage_collection().find_one({"user_id": user_id, "date": today, "action": action})
    return doc["count"] if doc else 0


if __name__ == "__main__":
    test_user = "test_rate_limit_user"
    for i in range(3):
        result = check_and_record(test_user)
        print(f"Request {i+1}: {result}")
