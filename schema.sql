-- EcoReconstruct AI — Supabase Postgres Schema
-- Запустить в SQL Editor Supabase (Project → SQL Editor → New query).

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
    post_id     UUID        NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, post_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_posts_created_desc ON posts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_posts_user         ON posts(user_id);
CREATE INDEX IF NOT EXISTS idx_likes_user         ON likes(user_id);
CREATE INDEX IF NOT EXISTS idx_likes_post         ON likes(post_id);
