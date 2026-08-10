"""
auth.py — User signup and login.

Passwords are NEVER stored in plaintext — bcrypt hashes them with a random
salt built in, so even if the database is ever exposed, raw passwords aren't.
"""

import bcrypt
from datetime import datetime
from db import get_db


def _users_collection():
    return get_db()["users"]


def signup(username: str, password: str) -> dict:
    """
    Creates a new user. Returns {"success": bool, "message": str, "user_id": str or None}
    """
    username = username.strip().lower()
    if not username or not password:
        return {"success": False, "message": "Username and password are required.", "user_id": None}
    if len(password) < 6:
        return {"success": False, "message": "Password must be at least 6 characters.", "user_id": None}

    users = _users_collection()
    if users.find_one({"username": username}):
        return {"success": False, "message": "Username already taken.", "user_id": None}

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    result = users.insert_one({
        "username": username,
        "password_hash": password_hash,
        "created_at": datetime.now().isoformat(),
    })

    return {"success": True, "message": "Account created.", "user_id": str(result.inserted_id)}


def login(username: str, password: str) -> dict:
    """
    Verifies credentials. Returns {"success": bool, "message": str, "user_id": str or None}
    """
    username = username.strip().lower()
    users = _users_collection()
    user = users.find_one({"username": username})

    if not user:
        return {"success": False, "message": "No account with that username.", "user_id": None}

    if not bcrypt.checkpw(password.encode("utf-8"), user["password_hash"]):
        return {"success": False, "message": "Incorrect password.", "user_id": None}

    return {"success": True, "message": "Login successful.", "user_id": str(user["_id"])}


if __name__ == "__main__":
    print(signup("testuser", "password123"))
    print(login("testuser", "password123"))
    print(login("testuser", "wrongpassword"))
    print(signup("testuser", "password123"))  # should fail — duplicate
