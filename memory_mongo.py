"""
memory_mongo.py — MongoDB-backed research history, per user.

Same interface as the old memory.py (save_session, list_recent, get_session,
find_similar, delete_session) but every document is now scoped to a user_id,
so each user only ever sees their own research.
"""

from datetime import datetime
from bson import ObjectId
from db import get_db


def _sessions_collection():
    return get_db()["sessions"]


def save_session(user_id: str, topic: str, report: str, metadata: dict = None) -> str:
    """Saves a session for a specific user. Returns the MongoDB document ID as a string."""
    sessions = _sessions_collection()
    doc = {
        "user_id": user_id,
        "topic": topic,
        "report": report,
        "metadata": metadata or {},
        "created_at": datetime.now().isoformat(),
    }
    result = sessions.insert_one(doc)
    return str(result.inserted_id)


def list_recent(user_id: str, limit: int = 15) -> list[dict]:
    sessions = _sessions_collection()
    cursor = sessions.find(
        {"user_id": user_id},
        {"topic": 1, "created_at": 1}
    ).sort("_id", -1).limit(limit)

    return [{"session_id": str(doc["_id"]), "topic": doc["topic"], "created_at": doc["created_at"]} for doc in cursor]


def get_session(user_id: str, session_id: str) -> dict:
    """Fetches a session — only if it belongs to this user_id (prevents cross-user access)."""
    sessions = _sessions_collection()
    try:
        doc = sessions.find_one({"_id": ObjectId(session_id), "user_id": user_id})
    except Exception:
        return None  # invalid ObjectId format

    if not doc:
        return None

    return {
        "session_id": str(doc["_id"]),
        "topic": doc["topic"],
        "report": doc["report"],
        "metadata": doc.get("metadata", {}),
        "created_at": doc["created_at"],
    }


def find_similar(user_id: str, topic: str, limit: int = 5) -> list[dict]:
    """Keyword-overlap similarity, scoped to this user's own past sessions."""
    new_words = set(topic.lower().split())
    if not new_words:
        return []

    sessions = _sessions_collection()
    cursor = sessions.find({"user_id": user_id}, {"topic": 1})

    scored = []
    for doc in cursor:
        stored_words = set(doc["topic"].lower().split())
        overlap = len(new_words & stored_words)
        if overlap > 0:
            similarity = overlap / len(new_words | stored_words)
            if similarity >= 0.2:
                scored.append({
                    "session_id": str(doc["_id"]),
                    "topic": doc["topic"],
                    "similarity": round(similarity, 2),
                })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:limit]


def get_session_public(session_id: str) -> dict:
    """
    Looks up a session by ID only, with NO user_id check — used for shareable
    public links. The session_id (a MongoDB ObjectId) is unguessable enough
    to act as a share token. Only call this from an endpoint that returns
    just the report content, never anything user-identifying.
    """
    sessions = _sessions_collection()
    try:
        doc = sessions.find_one({"_id": ObjectId(session_id)})
    except Exception:
        return None
    if not doc:
        return None
    return {
        "session_id": str(doc["_id"]),
        "topic": doc["topic"],
        "report": doc["report"],
        "created_at": doc["created_at"],
    }


def delete_session(user_id: str, session_id: str) -> bool:
    """Deletes a session — only if it belongs to this user. Returns True if deleted."""
    sessions = _sessions_collection()
    try:
        result = sessions.delete_one({"_id": ObjectId(session_id), "user_id": user_id})
    except Exception:
        return False
    return result.deleted_count > 0


if __name__ == "__main__":
    test_user = "test_user_123"
    sid = save_session(test_user, "Impact of AI on jobs in 2026", "Test report content", {"confidence": 85})
    print(f"Saved: {sid}")
    print("Recent:", list_recent(test_user))
    print("Similar:", find_similar(test_user, "AI and employment"))
    print("Get:", get_session(test_user, sid))
    print("Deleted:", delete_session(test_user, sid))
