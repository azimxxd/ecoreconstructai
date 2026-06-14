"""
EcoReconstruct AI — Postgres database layer (Supabase).

Tables:
  users    — accounts authenticated by username + password (PBKDF2 hash).
             Legacy column ``google_sub`` is kept (nullable) for old rows.
  posts    — citizen eco-audit submissions.
  likes    — per-user per-post likes; PRIMARY KEY (user_id, post_id) enforces
             the "1 account = 1 like" rule at the database level.
  comments — per-post comments.
  follows  — follower/following graph; PRIMARY KEY (follower_id, following_id).

All public functions return plain dicts / lists of dicts with the same field
names that app.py already consumes so the UI layer needs no changes:
  id, address, green_index, image_original, image_generated,
  ai_problems, ai_recommendations, ai_priority, ai_summary,
  author, avatar, likes, comments_count, timestamp.

Connection pool is a Streamlit @st.cache_resource singleton — created once per
deployment and shared across all concurrent user sessions. The driver is
``psycopg2-binary`` so it installs cleanly on Streamlit Community Cloud without
needing system build tools.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from datetime import datetime
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
            "Убедитесь, что в секретах указан URL Transaction pooler "
            "(порт 6543), а не прямое подключение (5432). "
            f"Детали: {exc}"
        )
        raise
    _init_schema(pool)
    return pool


def _init_schema(pool: psycopg2.pool.ThreadedConnectionPool) -> None:
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            # Fail fast if a concurrent session holds a conflicting lock instead
            # of stalling until the server statement_timeout (~2 min on Supabase).
            # On an existing deployment the schema is already applied, so a blocked
            # re-migration must never freeze the app's cold start.
            cur.execute("SET lock_timeout = '6s';")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
                    username      TEXT,
                    password_hash TEXT,
                    google_sub    TEXT,
                    email         TEXT,
                    name          TEXT,
                    avatar        TEXT        NOT NULL DEFAULT '🌱',
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                -- Migrate older deployments that still have the Google-only schema.
                ALTER TABLE users ADD COLUMN IF NOT EXISTS username      TEXT;
                ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash TEXT;
                ALTER TABLE users ALTER COLUMN google_sub DROP NOT NULL;

                CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username
                    ON users (lower(username)) WHERE username IS NOT NULL;

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

                CREATE TABLE IF NOT EXISTS comments (
                    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
                    post_id     UUID        NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
                    user_id     UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    content     TEXT        NOT NULL,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                CREATE TABLE IF NOT EXISTS follows (
                    follower_id  UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    following_id UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (follower_id, following_id)
                );

                CREATE INDEX IF NOT EXISTS idx_posts_created_desc  ON posts(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_posts_user          ON posts(user_id);
                CREATE INDEX IF NOT EXISTS idx_likes_user          ON likes(user_id);
                CREATE INDEX IF NOT EXISTS idx_likes_post          ON likes(post_id);
                CREATE INDEX IF NOT EXISTS idx_comments_post       ON comments(post_id);
                CREATE INDEX IF NOT EXISTS idx_follows_follower    ON follows(follower_id);
                CREATE INDEX IF NOT EXISTS idx_follows_following   ON follows(following_id);
            """)
        conn.commit()
    except (psycopg2.errors.LockNotAvailable, psycopg2.errors.QueryCanceled):
        # A migration statement was blocked on a lock held by another session.
        # The tables already exist on an established deployment, so roll back and
        # carry on rather than failing startup.
        conn.rollback()
    finally:
        pool.putconn(conn)


class _DB:
    """
    Context manager that borrows / returns a connection from the pool.

    Supabase's transaction pooler silently closes idle server connections, so a
    plain pool can hand back a dead one ("connection already closed"). We
    pre-ping each borrowed connection with ``SELECT 1`` and transparently
    recycle it if the ping fails, then guard commit/rollback against a
    connection that died mid-request.
    """

    def __enter__(self) -> "psycopg2.extensions.connection":
        self._pool = _get_pool()
        self.conn = self._acquire_live()
        return self.conn

    def _acquire_live(self) -> "psycopg2.extensions.connection":
        for _ in range(3):
            conn = self._pool.getconn()
            try:
                if conn.closed:
                    raise psycopg2.OperationalError("stale pooled connection")
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                conn.commit()
                return conn
            except psycopg2.Error:
                try:
                    self._pool.putconn(conn, close=True)
                except Exception:
                    pass
        # Give up pre-pinging and hand back whatever the pool yields next.
        return self._pool.getconn()

    def __exit__(self, exc_type, *_):
        try:
            if exc_type:
                self.conn.rollback()
            else:
                self.conn.commit()
        except psycopg2.Error:
            pass  # connection already broken; nothing to flush
        try:
            self._pool.putconn(self.conn, close=bool(self.conn.closed))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Initialisation helper (call from app startup)
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Trigger pool creation + schema migration (idempotent)."""
    _get_pool()


# ---------------------------------------------------------------------------
# Password hashing (PBKDF2-HMAC-SHA256, stdlib only — no extra dependency)
# ---------------------------------------------------------------------------

_PBKDF2_ITERATIONS = 200_000


def _hash_password(password: str) -> str:
    """Return a salted PBKDF2 hash encoded as ``salt_hex$hash_hex``."""
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS
    )
    return f"{salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored: str | None) -> bool:
    """Constant-time check of *password* against a stored PBKDF2 hash."""
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


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


class AuthError(Exception):
    """Raised for expected, user-facing auth failures (shown in the UI)."""


def _public_user(row: dict) -> dict[str, Any]:
    """Strip the password hash before a row enters session_state."""
    user = {k: v for k, v in row.items() if k != "password_hash"}
    user["id"] = str(user["id"])
    return user


def register_user(username: str, email: str | None, password: str) -> dict[str, Any]:
    """
    Create a new account. ``name`` defaults to the username so the rest of the
    UI (feed author, profile) keeps working unchanged.

    Raises AuthError if the username is already taken.
    """
    username = username.strip()
    email = (email or "").strip() or None
    with _DB() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT 1 FROM users WHERE lower(username) = lower(%s)",
                (username,),
            )
            if cur.fetchone() is not None:
                raise AuthError("Это имя пользователя уже занято.")
            cur.execute(
                """
                INSERT INTO users (username, email, name, password_hash)
                VALUES (%(username)s, %(email)s, %(name)s, %(pw)s)
                RETURNING *
                """,
                {
                    "username": username,
                    "email": email,
                    "name": username,
                    "pw": _hash_password(password),
                },
            )
            return _public_user(cur.fetchone())


def authenticate_user(login: str, password: str) -> dict[str, Any]:
    """
    Verify credentials (username or email + password).

    Raises AuthError on unknown user or wrong password.
    """
    login = login.strip()
    with _DB() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM users
                WHERE lower(username) = lower(%s) OR lower(email) = lower(%s)
                LIMIT 1
                """,
                (login, login),
            )
            row = cur.fetchone()
    if row is None or not _verify_password(password, row.get("password_hash")):
        raise AuthError("Неверное имя пользователя или пароль.")
    return _public_user(row)


def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    """Fetch a user by UUID (used to restore a session from a cookie token)."""
    with _DB() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
    return _public_user(row) if row else None


def get_user_by_email(email: str) -> dict[str, Any] | None:
    with _DB() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM users WHERE lower(email) = lower(%s)", (email,)
            )
            row = cur.fetchone()
    return _public_user(row) if row else None


def register_google_user(email: str, name: str) -> dict[str, Any] | None:
    """Create an account from a Google profile (legacy helper)."""
    username = email.split("@")[0]
    with _DB() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT 1 FROM users WHERE lower(username) = lower(%s)",
                (username,),
            )
            if cur.fetchone() is not None:
                import uuid as _uuid

                username = f"{username}_{str(_uuid.uuid4())[:4]}"
            cur.execute(
                """
                INSERT INTO users (username, email, name, password_hash)
                VALUES (%s, %s, %s, '')
                RETURNING id
                """,
                (username, email, name),
            )
            new_id = str(cur.fetchone()["id"])
    return get_user_by_id(new_id)


# ---------------------------------------------------------------------------
# "Stay logged in" cookie token — HMAC-signed user_id with an expiry.
# Lets the app restore the session after a full page reload (Streamlit clears
# st.session_state on reload). The token only carries the user id + expiry; it
# is verified server-side and the user is re-fetched from the DB.
# ---------------------------------------------------------------------------

_TOKEN_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days


def _auth_secret() -> str:
    """Signing key for the persistent-login cookie (kept off the client)."""
    try:
        configured = st.secrets["app"]["secret_key"]
        if configured:
            return str(configured)
    except Exception:
        pass
    # Fallback: derive a stable per-deployment key from the DB URL so tokens
    # stay valid across reruns without extra configuration.
    try:
        return "eco-auth::" + str(st.secrets["database"]["url"])
    except Exception:
        return "eco-auth::insecure-dev-secret"


def make_auth_token(user_id: str) -> str:
    """Return a signed, URL-safe token encoding ``user_id`` and an expiry."""
    expires_at = str(int(time.time()) + _TOKEN_TTL_SECONDS)
    message = f"{user_id}.{expires_at}"
    signature = hmac.new(
        _auth_secret().encode(), message.encode(), hashlib.sha256
    ).hexdigest()
    return base64.urlsafe_b64encode(f"{message}.{signature}".encode()).decode()


def verify_auth_token(token: str) -> dict[str, Any] | None:
    """Validate a cookie token and return the user dict, or None if invalid."""
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
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET avatar = %s WHERE id = %s",
                (avatar, user_id),
            )


# ---------------------------------------------------------------------------
# Posts
# ---------------------------------------------------------------------------

def _to_iso(value: Any) -> str:
    """Normalise a timestamp (datetime or str) to an ISO-8601 string."""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value else ""


def _row_to_post(row: dict) -> dict:
    """Normalise a DB row dict to the shape app.py expects."""
    d = dict(row)
    d["id"] = str(d["id"])
    d["likes"] = int(d.get("likes", 0) or 0)
    d["comments_count"] = int(d.get("comments_count", 0) or 0)
    d["timestamp"] = _to_iso(d.get("created_at"))
    d["green_index"] = float(d.get("green_index", 0.0) or 0.0)

    # JSONB columns come back as Python lists; tolerate text/None too.
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
        (SELECT COUNT(*) FROM likes    l WHERE l.post_id = p.id) AS likes,
        (SELECT COUNT(*) FROM comments c WHERE c.post_id = p.id) AS comments_count
    FROM posts p
    LEFT JOIN users u ON u.id = p.user_id
"""


def load_posts(limit: int = 20) -> list[dict]:
    """Return the most recent *limit* posts with like / comment counts."""
    with _DB() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                _POSTS_SELECT + " ORDER BY p.created_at DESC LIMIT %s", (limit,)
            )
            return [_row_to_post(r) for r in cur.fetchall()]


def load_all_posts() -> list[dict]:
    """Return all posts (used by leaderboard / analytics)."""
    with _DB() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(_POSTS_SELECT + " ORDER BY p.created_at DESC")
            return [_row_to_post(r) for r in cur.fetchall()]


def get_user_posts(user_id: str) -> list[dict]:
    """Return all posts published by *user_id* with counts."""
    with _DB() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                _POSTS_SELECT
                + " WHERE p.user_id = %s ORDER BY p.created_at DESC",
                (user_id,),
            )
            return [_row_to_post(r) for r in cur.fetchall()]


def save_post(user_id: str, data: dict[str, Any]) -> dict:
    """Persist a new post and return its stored record id."""
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
                    "INSERT INTO likes (user_id, post_id) VALUES (%s, %s) "
                    "ON CONFLICT DO NOTHING",
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


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------

def load_comments(post_id: str) -> list[dict]:
    """Return all comments for a post (oldest first) with author info."""
    with _DB() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT c.*, u.name AS author, u.avatar AS avatar
                FROM comments c
                LEFT JOIN users u ON u.id = c.user_id
                WHERE c.post_id = %s
                ORDER BY c.created_at ASC
                """,
                (post_id,),
            )
            rows = cur.fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["id"] = str(d["id"])
        d["created_at"] = _to_iso(d.get("created_at"))
        out.append(d)
    return out


def save_comment(user_id: str, post_id: str, content: str) -> dict:
    """Persist a comment and return its id + timestamp."""
    with _DB() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO comments (post_id, user_id, content)
                VALUES (%s, %s, %s)
                RETURNING id, created_at
                """,
                (post_id, user_id, content.strip()),
            )
            row = cur.fetchone()
    return {"id": str(row["id"]), "created_at": _to_iso(row.get("created_at"))}


def count_comments(post_id: str) -> int:
    with _DB() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM comments WHERE post_id = %s", (post_id,)
            )
            row = cur.fetchone()
    return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# Follows (followers / following)
# ---------------------------------------------------------------------------

def toggle_follow(follower_id: str, following_id: str) -> bool:
    """
    Toggle follow status. Returns True if now following, False if unfollowed
    (or if the two ids are the same — no self-follow).
    """
    if follower_id == following_id:
        return False
    with _DB() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM follows WHERE follower_id = %s AND following_id = %s",
                (follower_id, following_id),
            )
            already_following = cur.fetchone() is not None
            if already_following:
                cur.execute(
                    "DELETE FROM follows WHERE follower_id = %s AND following_id = %s",
                    (follower_id, following_id),
                )
                return False
            cur.execute(
                "INSERT INTO follows (follower_id, following_id) VALUES (%s, %s) "
                "ON CONFLICT DO NOTHING",
                (follower_id, following_id),
            )
            return True


def is_following(follower_id: str, following_id: str) -> bool:
    with _DB() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM follows WHERE follower_id = %s AND following_id = %s",
                (follower_id, following_id),
            )
            return cur.fetchone() is not None


def get_followers_count(user_id: str) -> int:
    with _DB() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM follows WHERE following_id = %s", (user_id,)
            )
            row = cur.fetchone()
    return int(row[0]) if row else 0


def get_following_count(user_id: str) -> int:
    with _DB() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM follows WHERE follower_id = %s", (user_id,)
            )
            row = cur.fetchone()
    return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# Seeding helper (for seed_demo.py)
# ---------------------------------------------------------------------------

def save_item(data: dict) -> dict:
    """
    Demo seeding: ensure an author user exists, insert the post, then fabricate
    *likes* dummy accounts so the leaderboard has realistic vote counts.
    """
    author = data.get("author", "eco_citizen")
    avatar = data.get("avatar", "🌱")

    with _DB() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id FROM users WHERE lower(username) = lower(%s)", (author,)
            )
            row = cur.fetchone()
            if row:
                user_id = str(row["id"])
            else:
                cur.execute(
                    """
                    INSERT INTO users (username, password_hash, name, avatar)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (author, _hash_password("password"), author, avatar),
                )
                user_id = str(cur.fetchone()["id"])

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
            with conn.cursor() as cur:
                for k in range(likes_count):
                    dummy_username = f"liker_{post_id[:8]}_{k}"
                    cur.execute(
                        """
                        INSERT INTO users (username, password_hash, name, avatar)
                        VALUES (%s, '', %s, '🌱')
                        RETURNING id
                        """,
                        (dummy_username, dummy_username),
                    )
                    dummy_id = cur.fetchone()[0]
                    cur.execute(
                        "INSERT INTO likes (user_id, post_id) VALUES (%s, %s) "
                        "ON CONFLICT DO NOTHING",
                        (dummy_id, post_id),
                    )
    return saved
