"""
EcoReconstruct AI — Postgres database layer (Supabase).

Tables:
  users  — Google-authenticated accounts (google_sub = unique OIDC sub).
  posts  — citizen eco-audit submissions.
  likes  — per-user per-post likes; PRIMARY KEY (user_id, post_id) enforces
            the "1 account = 1 like" rule at the database level.

All public functions return plain dicts / lists of dicts with the same field
names that app.py already consumes so the UI layer needs minimal changes:
  id, address, green_index, image_original, image_generated,
  ai_problems, ai_recommendations, ai_priority, ai_summary,
  author, avatar, likes, timestamp.

Connection pool is a Streamlit @st.cache_resource singleton — created once per
deployment and shared across all concurrent user sessions.
"""

from __future__ import annotations

import json
from typing import Any

import psycopg2
import psycopg2.pool
import streamlit as st
from psycopg2.extras import RealDictCursor

# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    url: str = st.secrets["database"]["url"]
    # Ensure SSL — required by Supabase when connecting from cloud hosts.
    if "sslmode" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"
    try:
        pool = psycopg2.pool.ThreadedConnectionPool(1, 8, url)
    except psycopg2.OperationalError as exc:
        st.error(
            "⚠️ Не удалось подключиться к базе данных Supabase. "
            "Убедитесь, что в секретах Streamlit Cloud указан URL "
            "Transaction pooler (порт 6543), а не прямое подключение (5432). "
            f"Детали: {exc}"
        )
        raise
    _init_schema(pool)
    return pool


def _init_schema(pool: psycopg2.pool.ThreadedConnectionPool) -> None:
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
                    google_sub  TEXT        UNIQUE NOT NULL,
                    email       TEXT,
                    name        TEXT,
                    avatar      TEXT        NOT NULL DEFAULT '🌱',
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                CREATE TABLE IF NOT EXISTS posts (
                    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id             UUID        REFERENCES users(id) ON DELETE SET NULL,
                    address             TEXT        NOT NULL DEFAULT '',
                    green_index         FLOAT       NOT NULL DEFAULT 0.0,
                    image_original      TEXT        NOT NULL DEFAULT '',
                    image_generated     TEXT        NOT NULL DEFAULT '',
                    ai_problems         JSONB       NOT NULL DEFAULT '[]',
                    ai_recommendations  JSONB       NOT NULL DEFAULT '[]',
                    ai_priority         TEXT        NOT NULL DEFAULT '',
                    ai_summary          TEXT        NOT NULL DEFAULT '',
                    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                CREATE TABLE IF NOT EXISTS likes (
                    user_id     UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    post_id     UUID        NOT NULL REFERENCES posts(id)  ON DELETE CASCADE,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (user_id, post_id)
                );

                CREATE INDEX IF NOT EXISTS idx_posts_created_desc ON posts(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_posts_user         ON posts(user_id);
                CREATE INDEX IF NOT EXISTS idx_likes_user         ON likes(user_id);
                CREATE INDEX IF NOT EXISTS idx_likes_post         ON likes(post_id);
            """)
        conn.commit()
    finally:
        pool.putconn(conn)


class _DB:
    """Context manager that borrows / returns a connection from the pool."""

    def __enter__(self) -> psycopg2.extensions.connection:
        self._pool = _get_pool()
        self.conn = self._pool.getconn()
        return self.conn

    def __exit__(self, exc_type, *_):
        if exc_type:
            self.conn.rollback()
        else:
            self.conn.commit()
        self._pool.putconn(self.conn)


# ---------------------------------------------------------------------------
# Initialisation helper (call from app startup)
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Trigger pool creation + schema migration (idempotent)."""
    _get_pool()


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def upsert_user(google_sub: str, email: str | None, name: str | None) -> dict[str, Any]:
    """
    Insert user on first login; update email/name on subsequent logins.
    Returns the full user row as a dict.
    """
    with _DB() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO users (google_sub, email, name)
                VALUES (%(sub)s, %(email)s, %(name)s)
                ON CONFLICT (google_sub) DO UPDATE
                    SET email = EXCLUDED.email,
                        name  = COALESCE(EXCLUDED.name, users.name)
                RETURNING *
                """,
                {"sub": google_sub, "email": email, "name": name},
            )
            row = cur.fetchone()
            return {**row, "id": str(row["id"])}


def update_user_avatar(user_id: str, avatar: str) -> None:
    with _DB() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET avatar = %s WHERE id = %s",
                (avatar, user_id),
            )


# ---------------------------------------------------------------------------
# Posts
# ---------------------------------------------------------------------------

def _row_to_post(row: dict) -> dict:
    """Normalise a DB row dict to the shape app.py expects."""
    d = dict(row)
    d["id"] = str(d["id"])
    d["likes"] = int(d.get("likes", 0))
    d["timestamp"] = (
        d["created_at"].isoformat() if d.get("created_at") else ""
    )
    d["green_index"] = float(d.get("green_index", 0.0))
    d["ai_problems"] = d.get("ai_problems") or []
    d["ai_recommendations"] = d.get("ai_recommendations") or []
    d["author"] = d.get("author") or "eco_citizen"
    d["avatar"] = d.get("avatar") or "🌱"
    return d


_POSTS_SELECT = """
    SELECT
        p.*,
        u.name   AS author,
        u.avatar AS avatar,
        COUNT(l.user_id) AS likes
    FROM posts p
    LEFT JOIN users  u ON u.id = p.user_id
    LEFT JOIN likes  l ON l.post_id = p.id
    GROUP BY p.id, u.name, u.avatar
    ORDER BY p.created_at DESC
"""


def load_posts(limit: int = 20) -> list[dict]:
    """Return the most recent *limit* posts with like counts."""
    with _DB() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(_POSTS_SELECT + " LIMIT %s", (limit,))
            return [_row_to_post(r) for r in cur.fetchall()]


def load_all_posts() -> list[dict]:
    """Return all posts (used by leaderboard / analytics)."""
    with _DB() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(_POSTS_SELECT)
            return [_row_to_post(r) for r in cur.fetchall()]


def get_user_posts(user_id: str) -> list[dict]:
    """Return all posts published by *user_id* with like counts."""
    with _DB() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    p.*,
                    u.name   AS author,
                    u.avatar AS avatar,
                    COUNT(l.user_id) AS likes
                FROM posts p
                LEFT JOIN users  u ON u.id = p.user_id
                LEFT JOIN likes  l ON l.post_id = p.id
                WHERE p.user_id = %s
                GROUP BY p.id, u.name, u.avatar
                ORDER BY p.created_at DESC
                """,
                (user_id,),
            )
            return [_row_to_post(r) for r in cur.fetchall()]


def save_post(user_id: str, data: dict[str, Any]) -> dict:
    """Persist a new post and return its stored record."""
    with _DB() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO posts
                    (user_id, address, green_index,
                     image_original, image_generated,
                     ai_problems, ai_recommendations, ai_priority, ai_summary)
                VALUES
                    (%(user_id)s, %(address)s, %(green_index)s,
                     %(image_original)s, %(image_generated)s,
                     %(ai_problems)s, %(ai_recommendations)s,
                     %(ai_priority)s, %(ai_summary)s)
                RETURNING id
                """,
                {
                    "user_id":             user_id,
                    "address":             data.get("address", ""),
                    "green_index":         float(data.get("green_index", 0.0)),
                    "image_original":      data.get("image_original", ""),
                    "image_generated":     data.get("image_generated", ""),
                    "ai_problems":         json.dumps(
                        data.get("ai_problems", []), ensure_ascii=False
                    ),
                    "ai_recommendations":  json.dumps(
                        data.get("ai_recommendations", []), ensure_ascii=False
                    ),
                    "ai_priority":         data.get("ai_priority", ""),
                    "ai_summary":          data.get("ai_summary", ""),
                },
            )
            return {"id": str(cur.fetchone()["id"])}


# ---------------------------------------------------------------------------
# Likes  (1 account = 1 like, enforced by PRIMARY KEY at DB level)
# ---------------------------------------------------------------------------

def toggle_like(user_id: str, post_id: str) -> tuple[bool, int]:
    """
    Toggle like for (user_id, post_id).
    Returns (now_liked: bool, new_count: int).
    """
    with _DB() as conn:
        with conn.cursor() as cur:
            # Check current state
            cur.execute(
                "SELECT 1 FROM likes WHERE user_id = %s AND post_id = %s",
                (user_id, post_id),
            )
            already_liked = cur.fetchone() is not None

            if already_liked:
                cur.execute(
                    "DELETE FROM likes WHERE user_id = %s AND post_id = %s",
                    (user_id, post_id),
                )
                now_liked = False
            else:
                cur.execute(
                    "INSERT INTO likes (user_id, post_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (user_id, post_id),
                )
                now_liked = True

            cur.execute(
                "SELECT COUNT(*) FROM likes WHERE post_id = %s",
                (post_id,),
            )
            count = int(cur.fetchone()[0])

    return (now_liked, count)


def get_user_liked_posts(user_id: str) -> set[str]:
    """Return the set of post IDs that *user_id* has liked."""
    with _DB() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT post_id::text FROM likes WHERE user_id = %s",
                (user_id,),
            )
            return {row[0] for row in cur.fetchall()}
