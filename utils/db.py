"""
EcoReconstruct AI — Local SQLite database layer (Temporary fallback).

Replica of the Postgres DB layer that uses Python's standard `sqlite3`
module to allow zero-config local development.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import streamlit as st

DB_FILE = "local_db.sqlite"
_lock = threading.Lock()

class _DB:
    """Context manager for SQLite connection with automatic transactions."""
    def __enter__(self) -> sqlite3.Connection:
        _lock.acquire()
        self.conn = sqlite3.connect(DB_FILE)
        self.conn.row_factory = sqlite3.Row
        # Enable foreign keys
        self.conn.execute("PRAGMA foreign_keys = ON;")
        return self.conn

    def __exit__(self, exc_type, *_):
        try:
            if exc_type:
                self.conn.rollback()
            else:
                self.conn.commit()
            self.conn.close()
        finally:
            _lock.release()


def init_db() -> None:
    """Initialize SQLite tables (idempotent)."""
    with _DB() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            TEXT PRIMARY KEY,
                username      TEXT,
                password_hash TEXT,
                google_sub    TEXT,
                email         TEXT,
                name          TEXT,
                avatar        TEXT NOT NULL DEFAULT '🌱',
                created_at    TEXT NOT NULL
            );
        """)
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username
                ON users (username);
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id                  TEXT PRIMARY KEY,
                user_id             TEXT REFERENCES users(id) ON DELETE SET NULL,
                address             TEXT NOT NULL DEFAULT '',
                green_index         REAL NOT NULL DEFAULT 0.0,
                image_original      TEXT NOT NULL DEFAULT '',
                image_generated     TEXT NOT NULL DEFAULT '',
                ai_problems         TEXT NOT NULL DEFAULT '[]',
                ai_recommendations  TEXT NOT NULL DEFAULT '[]',
                ai_priority         TEXT NOT NULL DEFAULT '',
                ai_summary          TEXT NOT NULL DEFAULT '',
                created_at          TEXT NOT NULL
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS likes (
                user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                post_id     TEXT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
                created_at  TEXT NOT NULL,
                PRIMARY KEY (user_id, post_id)
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS comments (
                id          TEXT PRIMARY KEY,
                post_id     TEXT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
                user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS follows (
                follower_id  TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                following_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at   TEXT NOT NULL,
                PRIMARY KEY (follower_id, following_id)
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_created_desc ON posts(created_at DESC);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_user ON posts(user_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_likes_user ON likes(user_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_likes_post ON likes(post_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_comments_post ON comments(post_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_follows_follower ON follows(follower_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_follows_following ON follows(following_id);")



# ---------------------------------------------------------------------------
# Password hashing (PBKDF2-HMAC-SHA256)
# ---------------------------------------------------------------------------
_PBKDF2_ITERATIONS = 200_000

def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS
    )
    return f"{salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored: str | None) -> bool:
    if not stored or "$" not in stored:
        return False
    salt_hex, digest_hex = stored.split("$", 1)
    try:
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        return False
    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS
    )
    return hmac.compare_digest(candidate.hex(), digest_hex)


class AuthError(Exception):
    """Raised for expected, user-facing auth failures."""
    pass


def _public_user(row: dict) -> dict[str, Any]:
    user = {k: v for k, v in row.items() if k != "password_hash"}
    user["id"] = str(user["id"])
    return user


def register_user(username: str, email: str | None, password: str) -> dict[str, Any]:
    username = username.strip()
    email = (email or "").strip() or None
    user_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with _DB() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM users WHERE lower(username) = lower(?)", (username,))
        if cur.fetchone() is not None:
            raise AuthError("Это имя пользователя уже занято.")
        cur.execute(
            """
            INSERT INTO users (id, username, email, name, password_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, username, email, username, _hash_password(password), created_at),
        )
        cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        return _public_user(dict(row))


def authenticate_user(login: str, password: str) -> dict[str, Any]:
    login = login.strip()
    with _DB() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM users
            WHERE lower(username) = lower(?) OR lower(email) = lower(?)
            LIMIT 1
            """,
            (login, login),
        )
        row = cur.fetchone()
    if row is None or not _verify_password(password, row["password_hash"]):
        raise AuthError("Неверное имя пользователя или пароль.")
    return _public_user(dict(row))


def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    with _DB() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
    return _public_user(dict(row)) if row else None


# ---------------------------------------------------------------------------
# Auth token
# ---------------------------------------------------------------------------
_TOKEN_TTL_SECONDS = 60 * 60 * 24 * 30

def _auth_secret() -> str:
    try:
        configured = st.secrets["app"]["secret_key"]
        if configured:
            return str(configured)
    except Exception:
        pass
    return "eco-auth::insecure-dev-secret"


def make_auth_token(user_id: str) -> str:
    expires_at = str(int(time.time()) + _TOKEN_TTL_SECONDS)
    message = f"{user_id}.{expires_at}"
    signature = hmac.new(
        _auth_secret().encode(), message.encode(), hashlib.sha256
    ).hexdigest()
    return base64.urlsafe_b64encode(f"{message}.{signature}".encode()).decode()


def verify_auth_token(token: str) -> dict[str, Any] | None:
    if not token:
        return None
    try:
        raw = base64.urlsafe_b64decode(token.encode()).decode()
        user_id, expires_at, signature = raw.rsplit(".", 2)
    except Exception:
        return None
    expected = hmac.new(
        _auth_secret().encode(), f"{user_id}.{expires_at}".encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        if int(expires_at) < int(time.time()):
            return None
    except ValueError:
        return None
    return get_user_by_id(user_id)


def update_user_avatar(user_id: str, avatar: str) -> None:
    with _DB() as conn:
        conn.execute(
            "UPDATE users SET avatar = ? WHERE id = ?",
            (avatar, user_id),
        )


# ---------------------------------------------------------------------------
# Posts
# ---------------------------------------------------------------------------
def _row_to_post(row: dict) -> dict:
    d = dict(row)
    d["id"] = str(d["id"])
    d["likes"] = int(d.get("likes", 0))
    d["comments_count"] = int(d.get("comments_count", 0))
    d["timestamp"] = d.get("created_at", "")
    d["green_index"] = float(d.get("green_index", 0.0))
    
    # JSON decoding for lists
    for k in ("ai_problems", "ai_recommendations"):
        val = d.get(k)
        if isinstance(val, str):
            try:
                d[k] = json.loads(val)
            except Exception:
                d[k] = []
        elif val is None:
            d[k] = []

    d["author"] = d.get("author") or "eco_citizen"
    d["avatar"] = d.get("avatar") or "🌱"
    return d


_POSTS_SELECT = """
    SELECT
        p.*,
        u.name   AS author,
        u.avatar AS avatar,
        (SELECT COUNT(*) FROM likes l WHERE l.post_id = p.id) AS likes,
        (SELECT COUNT(*) FROM comments c WHERE c.post_id = p.id) AS comments_count
    FROM posts p
    LEFT JOIN users u ON u.id = p.user_id
"""

def load_posts(limit: int = 20) -> list[dict]:
    with _DB() as conn:
        cur = conn.cursor()
        cur.execute(_POSTS_SELECT + " ORDER BY p.created_at DESC LIMIT ?", (limit,))
        return [_row_to_post(dict(r)) for r in cur.fetchall()]


def load_all_posts() -> list[dict]:
    with _DB() as conn:
        cur = conn.cursor()
        cur.execute(_POSTS_SELECT + " ORDER BY p.created_at DESC")
        return [_row_to_post(dict(r)) for r in cur.fetchall()]


def get_user_posts(user_id: str) -> list[dict]:
    with _DB() as conn:
        cur = conn.cursor()
        cur.execute(_POSTS_SELECT + " WHERE p.user_id = ? ORDER BY p.created_at DESC", (user_id,))
        return [_row_to_post(dict(r)) for r in cur.fetchall()]


def save_post(user_id: str, data: dict[str, Any]) -> dict:
    post_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with _DB() as conn:
        conn.execute(
            """
            INSERT INTO posts
                (id, user_id, address, green_index,
                 image_original, image_generated,
                 ai_problems, ai_recommendations, ai_priority, ai_summary, created_at)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                post_id,
                user_id,
                data.get("address", ""),
                float(data.get("green_index", 0.0)),
                data.get("image_original", ""),
                data.get("image_generated", ""),
                json.dumps(data.get("ai_problems", []), ensure_ascii=False),
                json.dumps(data.get("ai_recommendations", []), ensure_ascii=False),
                data.get("ai_priority", ""),
                data.get("ai_summary", ""),
                created_at,
            ),
        )
    return {"id": post_id}


# ---------------------------------------------------------------------------
# Likes
# ---------------------------------------------------------------------------
def toggle_like(user_id: str, post_id: str) -> tuple[bool, int]:
    with _DB() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM likes WHERE user_id = ? AND post_id = ?",
            (user_id, post_id),
        )
        already_liked = cur.fetchone() is not None

        if already_liked:
            cur.execute(
                "DELETE FROM likes WHERE user_id = ? AND post_id = ?",
                (user_id, post_id),
            )
            now_liked = False
        else:
            cur.execute(
                "INSERT OR IGNORE INTO likes (user_id, post_id, created_at) VALUES (?, ?, ?)",
                (user_id, post_id, datetime.now(timezone.utc).isoformat()),
            )
            now_liked = True

        cur.execute(
            "SELECT COUNT(*) FROM likes WHERE post_id = ?",
            (post_id,),
        )
        count = int(cur.fetchone()[0])

    return (now_liked, count)


def get_user_liked_posts(user_id: str) -> set[str]:
    with _DB() as conn:
        cur = conn.cursor()
        cur.execute("SELECT post_id FROM likes WHERE user_id = ?", (user_id,))
        return {row[0] for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# Seeding helper (for seed_demo.py)
# ---------------------------------------------------------------------------
def save_item(data: dict) -> dict:
    author = data.get("author", "eco_citizen")
    avatar = data.get("avatar", "🌱")
    
    with _DB() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = ?", (author,))
        row = cur.fetchone()
        if row:
            user_id = row["id"]
        else:
            user_id = str(uuid.uuid4())
            cur.execute(
                "INSERT INTO users (id, username, password_hash, name, avatar, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, author, _hash_password("password"), author, avatar, datetime.now(timezone.utc).isoformat())
            )
            
    post_data = {
        "address": data.get("address", ""),
        "green_index": float(data.get("green_index", 0.0)),
        "image_original": data.get("image_original", ""),
        "image_generated": data.get("image_generated", ""),
        "ai_problems": data.get("ai_problems", []),
        "ai_recommendations": data.get("ai_recommendations", []),
        "ai_priority": data.get("ai_priority", "Средний"),
        "ai_summary": data.get("ai_summary", ""),
    }
    
    saved = save_post(user_id, post_data)
    post_id = saved["id"]
    
    likes_count = int(data.get("likes", 0))
    if likes_count > 0:
        with _DB() as conn:
            cur = conn.cursor()
            for k in range(likes_count):
                dummy_username = f"liker_{post_id[:8]}_{k}"
                dummy_id = str(uuid.uuid4())
                cur.execute(
                    "INSERT INTO users (id, username, password_hash, name, avatar, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (dummy_id, dummy_username, "", dummy_username, "🌱", datetime.now(timezone.utc).isoformat())
                )
                cur.execute(
                    "INSERT INTO likes (user_id, post_id, created_at) VALUES (?, ?, ?)",
                    (dummy_id, post_id, datetime.now(timezone.utc).isoformat())
                )
    return saved


# ---------------------------------------------------------------------------
# Google Auth Helpers
# ---------------------------------------------------------------------------
def get_user_by_email(email: str) -> dict[str, Any] | None:
    with _DB() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE lower(email) = lower(?)", (email,))
        row = cur.fetchone()
    return _public_user(dict(row)) if row else None


def register_google_user(email: str, name: str) -> dict[str, Any]:
    username = email.split("@")[0]
    # Ensure username is unique in SQLite
    with _DB() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM users WHERE lower(username) = lower(?)", (username,))
        if cur.fetchone() is not None:
            # If username is taken, append a short UUID suffix
            username = f"{username}_{str(uuid.uuid4())[:4]}"
            
    user_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with _DB() as conn:
        conn.execute(
            """
            INSERT INTO users (id, username, email, name, password_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, username, email, name, "", created_at),
        )
    return get_user_by_id(user_id)


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------
def load_comments(post_id: str) -> list[dict]:
    with _DB() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT c.*, u.name AS author, u.avatar AS avatar
            FROM comments c
            LEFT JOIN users u ON u.id = c.user_id
            WHERE c.post_id = ?
            ORDER BY c.created_at ASC
            """,
            (post_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def save_comment(user_id: str, post_id: str, content: str) -> dict:
    comment_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with _DB() as conn:
        conn.execute(
            """
            INSERT INTO comments (id, post_id, user_id, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (comment_id, post_id, user_id, content.strip(), created_at),
        )
    return {"id": comment_id, "created_at": created_at}


def count_comments(post_id: str) -> int:
    with _DB() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM comments WHERE post_id = ?", (post_id,))
        row = cur.fetchone()
        return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# Follows (Followers/Following)
# ---------------------------------------------------------------------------
def toggle_follow(follower_id: str, following_id: str) -> bool:
    """Toggles follow status between follower and following. Returns True if now following, False if unfollowed."""
    if follower_id == following_id:
        return False
    with _DB() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM follows WHERE follower_id = ? AND following_id = ?",
            (follower_id, following_id),
        )
        already_following = cur.fetchone() is not None

        created_at = datetime.now(timezone.utc).isoformat()
        if already_following:
            cur.execute(
                "DELETE FROM follows WHERE follower_id = ? AND following_id = ?",
                (follower_id, following_id),
            )
            return False
        else:
            cur.execute(
                "INSERT INTO follows (follower_id, following_id, created_at) VALUES (?, ?, ?)",
                (follower_id, following_id, created_at),
            )
            return True


def is_following(follower_id: str, following_id: str) -> bool:
    with _DB() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM follows WHERE follower_id = ? AND following_id = ?",
            (follower_id, following_id),
        )
        return cur.fetchone() is not None


def get_followers_count(user_id: str) -> int:
    with _DB() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM follows WHERE following_id = ?", (user_id,))
        row = cur.fetchone()
        return int(row[0]) if row else 0


def get_following_count(user_id: str) -> int:
    with _DB() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM follows WHERE follower_id = ?", (user_id,))
        row = cur.fetchone()
        return int(row[0]) if row else 0


# Initialize DB automatically on import
init_db()

