"""
db.py — MongoDB Atlas connection.

Single shared connection, reused everywhere else (auth.py, memory_mongo.py)
so we don't open a new connection per function call.
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi

load_dotenv()

_client = None
_db = None


def get_db():
    """
    Returns the MongoDB database handle, creating the connection on first call.
    Reads MONGODB_URI from .env — set this to your Atlas connection string.
    """
    global _client, _db
    if _db is not None:
        return _db

    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError(
            "MONGODB_URI not found in .env. Add a line like:\n"
            "MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/"
        )

    # tlsCAFile=certifi.where() fixes a common Windows SSL handshake error
    # (TLSV1_ALERT_INTERNAL_ERROR) caused by outdated/incomplete root certificates
    # bundled with some Python installs on Windows.
    _client = MongoClient(uri, server_api=ServerApi("1"), tlsCAFile=certifi.where())
    _db = _client["research_agent"]  # database name — created automatically on first write
    return _db


def test_connection():
    """Quick sanity check — pings the server and reports success/failure."""
    try:
        db = get_db()
        db.command("ping")
        print("MongoDB connection successful.")
        return True
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return False


if __name__ == "__main__":
    test_connection()
