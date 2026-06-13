"""
EcoReconstruct AI — crowdsourced urban ecology platform.

Design language: "City Scanner" — a solarpunk instrument for auditing streets.
Deep forest-black canvas with aurora glows and film grain, chlorophyll
neon gradient (teal → lime), Unbounded display type + JetBrains Mono data
chips, viewfinder brackets over the full-screen feed, floating glass dock.

Layout:
- Opens straight into a full-screen vertical feed of citizen eco-solutions
  (snap-scroll). Swiping a card horizontally reveals the AI-generated
  "green future" of the same street.
- Floating bottom dock: 👤 profile (left) · ◉ big scan button (center) ·
  🏆 monthly leaderboard (right) — plus a home button back to the feed.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import base64
import html
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from utils.free_ai import (
    free_ai_available,
    generate_appeal_text,
    generate_dashboard_report,
    generate_findings,
)
import streamlit.components.v1 as components

from utils.db import (
    AuthError,
    init_db,
    load_posts,
    load_all_posts,
    save_post,
    toggle_like,
    get_user_liked_posts,
    get_user_posts,
    register_user,
    authenticate_user,
    update_user_avatar,
    make_auth_token,
    verify_auth_token,
    load_comments,
    save_comment,
    count_comments,
    get_user_by_id,
    toggle_follow,
    is_following,
    get_followers_count,
    get_following_count,
)
from utils.models import (
    analyze_eco_status,
    eco_audit_safe,
    generate_eco_friendly_view,
)

# ===========================================================================
# Page config
# ===========================================================================
st.set_page_config(
    page_title="EcoReconstruct AI",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ===========================================================================
# Avatar options (emoji selection for profile customisation)
# ===========================================================================
AVATARS = ["🌱", "🌳", "🌻", "🦊", "🐝", "🚲", "🏙", "♻️"]


# ===========================================================================
# Auth helpers — username + password (session-based)
# ===========================================================================

def get_current_user() -> dict:
    """Return the current user's DB record stored in session_state."""
    return st.session_state.get("db_user") or {}


def get_current_user_id() -> str | None:
    """Return the current user's DB UUID string, or None."""
    uid = get_current_user().get("id")
    return str(uid) if uid else None


AUTH_COOKIE = "eco_auth"


def _login_user(user: dict) -> None:
    """Store the user in the session and queue a persistent-login cookie."""
    st.session_state["db_user"] = user
    st.session_state["_write_cookie"] = make_auth_token(user["id"])


def logout() -> None:
    """Clear the session and queue cookie removal so reloads stay logged out."""
    for key in ("db_user", "page"):
        st.session_state.pop(key, None)
    st.session_state["_clear_cookie"] = True


def _set_auth_cookie(token: str) -> None:
    """Write the auth cookie on the top-level document via a tiny JS shim."""
    max_age = 60 * 60 * 24 * 30
    components.html(
        f"""<script>
        window.parent.document.cookie =
            "{AUTH_COOKIE}={token}; path=/; max-age={max_age}; SameSite=Lax";
        </script>""",
        height=0,
    )


def _clear_auth_cookie() -> None:
    components.html(
        f"""<script>
        window.parent.document.cookie =
            "{AUTH_COOKIE}=; path=/; max-age=0; SameSite=Lax";
        </script>""",
        height=0,
    )


def _sync_auth_cookie() -> None:
    """
    Flush queued cookie writes/clears. Must run BEFORE any cookie-based session
    restore so a logout isn't immediately undone by a stale cookie.
    """
    if st.session_state.pop("_clear_cookie", False):
        _clear_auth_cookie()
        st.session_state["_skip_cookie_restore"] = True
    token = st.session_state.pop("_write_cookie", None)
    if token:
        _set_auth_cookie(token)


def _ensure_auth() -> None:
    """
    Ensure a user is in the session. If not, try to restore one from the
    persistent-login cookie; otherwise show the login/register screen and stop.
    """
    if "db_user" not in st.session_state:
        if st.session_state.pop("_skip_cookie_restore", False):
            pass  # just logged out — don't restore from the stale cookie
        else:
            try:
                token = st.context.cookies.get(AUTH_COOKIE)
            except Exception:
                token = None
            if token:
                restored = verify_auth_token(token)
                if restored:
                    st.session_state["db_user"] = restored

    if "db_user" not in st.session_state:
        _render_login_screen()
        st.stop()


def _render_login_screen() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div style="
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            text-align: center; padding: 2.4rem 1.5rem 1.2rem;
            gap: .7rem;">
            <div style="
                font-family: var(--display); font-weight: 900; font-size: 2.2rem;
                background: var(--grad);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                ECO//RE
            </div>
            <div style="font-family: var(--mono); font-size: .62rem;
                letter-spacing: .24em; text-transform: uppercase; color: var(--lime);">
                // сканер городской экологии
            </div>
            <p style="color: var(--muted); font-size: .88rem; max-width: 320px;
                line-height: 1.6; margin: .2rem auto 0;">
                Фотографируй серые улицы — ИИ покажет зелёное будущее.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_login, tab_register = st.tabs(["🔑 Вход", "✦ Регистрация"])

    with tab_login:
        with st.form("login_form", border=False):
            login_id = st.text_input(
                "Имя пользователя или почта",
                key="login_id",
                placeholder="ник или email",
            )
            login_pw = st.text_input(
                "Пароль", type="password", key="login_pw", placeholder="••••••••"
            )
            if st.form_submit_button("Войти", type="primary"):
                if not login_id.strip() or not login_pw:
                    st.warning("Введите имя пользователя и пароль.")
                else:
                    try:
                        user = authenticate_user(login_id, login_pw)
                        _login_user(user)
                        st.rerun()
                    except AuthError as exc:
                        st.error(str(exc))

    with tab_register:
        with st.form("register_form", border=False):
            reg_username = st.text_input(
                "Имя пользователя", key="reg_username", placeholder="например: eco_almaty"
            )
            reg_email = st.text_input(
                "Почта", key="reg_email", placeholder="you@example.com"
            )
            reg_pw = st.text_input(
                "Пароль", type="password", key="reg_pw", placeholder="минимум 6 символов"
            )
            reg_pw2 = st.text_input(
                "Повтор пароля", type="password", key="reg_pw2", placeholder="••••••••"
            )
            if st.form_submit_button("Создать аккаунт", type="primary"):
                username = reg_username.strip()
                if len(username) < 3:
                    st.warning("Имя пользователя — минимум 3 символа.")
                elif len(reg_pw) < 6:
                    st.warning("Пароль — минимум 6 символов.")
                elif reg_pw != reg_pw2:
                    st.warning("Пароли не совпадают.")
                else:
                    try:
                        user = register_user(username, reg_email, reg_pw)
                        _login_user(user)
                        st.session_state["flash"] = "Добро пожаловать в ECO//RE! 🌿"
                        st.rerun()
                    except AuthError as exc:
                        st.error(str(exc))

    st.markdown(
        """
        <div style="
            background: rgba(255,255,255,.04);
            border: 1px solid var(--line-soft);
            border-radius: 18px;
            padding: .9rem 1.1rem; margin-top: 1rem;
            font-size: .78rem; color: var(--muted);
            line-height: 1.55; text-align: center;">
            🔒 Аккаунт нужен, чтобы отслеживать твои публикации и голоса.
            Пароль хранится в зашифрованном виде.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# "City Scanner" design system — global CSS
# ===========================================================================
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@500;700;900&family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
    --bg: #060D09;
    --panel: #0D1A12;
    --panel-2: #101F16;
    --glass: rgba(13, 26, 18, .72);
    --line: rgba(168, 255, 96, .14);
    --line-soft: rgba(255, 255, 255, .09);
    --text: #ECF5EE;
    --muted: #8FA89A;
    --lime: #A8FF60;
    --teal: #3DF5C8;
    --mint: #74C69D;
    --coral: #FF7A5C;
    --coral-soft: #FF9B82;
    --amber: #FFC24B;
    --ink: #07130C;
    --grad: linear-gradient(135deg, var(--teal) 0%, var(--lime) 100%);
    --display: 'Unbounded', 'Manrope', sans-serif;
    --mono: 'JetBrains Mono', monospace;
    --nav-h: 86px;
    --radius: 20px;
}

html, body, [class*="css"], .stApp {
    font-family: 'Manrope', -apple-system, 'Segoe UI', sans-serif !important;
    color: var(--text);
}

/* ---- Aurora canvas + film grain ------------------------------------ */
.stApp {
    background:
        radial-gradient(900px 600px at 85% -10%, rgba(61,245,200,.10), transparent 60%),
        radial-gradient(700px 520px at -15% 28%, rgba(168,255,96,.07), transparent 55%),
        radial-gradient(800px 600px at 110% 85%, rgba(61,245,200,.05), transparent 60%),
        var(--bg);
}
.stApp::before {
    content: "";
    position: fixed; inset: 0;
    z-index: 0; pointer-events: none;
    opacity: .05;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>");
}
section[data-testid="stMain"] { position: relative; z-index: 1; }

/* ---- Hide Streamlit chrome ---------------------------------------- */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
div[data-testid="stToolbar"], div[data-testid="stDecoration"] { display: none; }

/* ---- Phone-width content column ------------------------------------ */
.block-container {
    max-width: 480px;
    padding: 1.1rem 1rem calc(var(--nav-h) + 2.2rem) 1rem !important;
}

/* Keep columns side-by-side on narrow screens (Streamlit stacks them). */
[data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; }
[data-testid="stColumn"], [data-testid="column"] { min-width: 0 !important; }

/* ---- Buttons -------------------------------------------------------- */
.stButton > button, .stFormSubmitButton > button {
    width: 100%;
    border: 1px solid var(--line-soft);
    border-radius: 16px;
    padding: .8rem 1.1rem;
    font-weight: 800;
    font-size: .92rem;
    background: rgba(255,255,255,.05);
    color: var(--text);
    box-shadow: none;
    transition: transform .15s ease, filter .15s ease, border-color .15s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    transform: translateY(-1px);
    border-color: var(--line);
    color: var(--text);
}
.stButton > button:active { transform: translateY(0); }
.stButton > button[data-testid="stBaseButton-primary"],
.stFormSubmitButton > button {
    background: var(--grad);
    border: none;
    color: var(--ink);
    box-shadow: 0 12px 30px rgba(168,255,96,.20);
}
.stButton > button[data-testid="stBaseButton-primary"]:hover,
.stFormSubmitButton > button:hover { color: var(--ink); filter: brightness(1.07); }

/* ---- Inputs --------------------------------------------------------- */
.stTextInput input, .stTextArea textarea {
    border-radius: 16px !important;
    border: 1px solid var(--line-soft) !important;
    background: rgba(255,255,255,.04) !important;
    color: var(--text) !important;
    font-size: 1rem !important;
}
.stTextInput input { padding: .8rem 1rem !important; }
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: rgba(168,255,96,.45) !important;
    box-shadow: 0 0 0 3px rgba(168,255,96,.14) !important;
}
div[data-testid="stFileUploader"] section {
    border: 1.5px dashed rgba(168,255,96,.35);
    border-radius: var(--radius);
    background: rgba(168,255,96,.04);
}
div[data-testid="stCameraInput"] { border-radius: var(--radius); overflow: hidden; }
div[data-testid="stCameraInput"] button {
    background: var(--panel-2);
    color: var(--text);
    border: none;
    font-weight: 800;
}

/* ---- Segmented control ------------------------------------------------ */
div[data-testid="stButtonGroup"] { width: 100%; gap: 6px; }
div[data-testid="stButtonGroup"] button {
    flex: 1;
    font-family: var(--mono);
    font-size: .7rem;
    letter-spacing: .04em;
    text-transform: uppercase;
    font-weight: 700;
    border-radius: 999px;
    border: 1px solid var(--line-soft);
    background: rgba(255,255,255,.04);
    color: var(--muted);
}
div[data-testid="stButtonGroup"] button p,
div[data-testid="stButtonGroup"] button div {
    overflow: visible !important;
    text-overflow: clip !important;
    font-size: inherit !important;
}
div[data-testid="stButtonGroup"] button[aria-checked="true"],
div[data-testid="stButtonGroup"] button[data-testid*="Active"] {
    background: var(--grad);
    color: var(--ink);
    border-color: transparent;
}

/* ---- Expander / misc -------------------------------------------------- */
div[data-testid="stExpander"] details {
    background: rgba(255,255,255,.03);
    border: 1px solid var(--line-soft);
    border-radius: 18px;
}
div[data-testid="stImage"] img { border-radius: 16px; }
hr { border-color: var(--line-soft); }

/* ---- Page headers ------------------------------------------------------ */
.page-head { padding: .3rem 0 .8rem; }
.page-head .kicker {
    font-family: var(--mono);
    font-size: .64rem;
    letter-spacing: .22em;
    text-transform: uppercase;
    color: var(--lime);
    margin-bottom: .45rem;
}
.page-head h1 {
    font-family: var(--display);
    font-size: 1.3rem;
    font-weight: 900;
    margin: 0;
    line-height: 1.2;
}
.page-head p { color: var(--muted); font-size: .85rem; margin: .4rem 0 0; }

/* ---- Glass cards / badges --------------------------------------------- */
.eco-card {
    background: var(--glass);
    backdrop-filter: blur(14px);
    border: 1px solid var(--line-soft);
    border-radius: var(--radius);
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
}
.section-title {
    font-weight: 800; font-size: 1rem;
    margin: .9rem 0 .45rem 0; color: var(--text);
}

.feed-badge {
    display: inline-block;
    padding: .3rem .75rem;
    border-radius: 999px;
    font-family: var(--mono);
    font-size: .62rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    border: 1px solid;
}
.badge-low  { background: rgba(255,122,92,.08); border-color: rgba(255,122,92,.4);  color: var(--coral-soft); }
.badge-mid  { background: rgba(255,194,75,.08); border-color: rgba(255,194,75,.4);  color: var(--amber); }
.badge-high { background: rgba(168,255,96,.08); border-color: rgba(168,255,96,.4);  color: var(--lime); }

.kpi-card {
    background: var(--glass);
    backdrop-filter: blur(14px);
    border: 1px solid var(--line-soft);
    border-radius: var(--radius);
    padding: .95rem .4rem .8rem;
    text-align: center;
    height: 100%;
}
.kpi-card .kpi-value {
    font-family: var(--display);
    font-size: 1.15rem; font-weight: 900; color: var(--lime);
    line-height: 1.2; word-break: break-word;
}
.kpi-card .kpi-label {
    font-family: var(--mono);
    font-size: .56rem; font-weight: 700; letter-spacing: .18em;
    text-transform: uppercase; color: var(--muted); margin-top: .35rem;
}

/* ---- Camera page: step rail + scan result ------------------------------ */
.steps { display: flex; gap: 6px; margin: .1rem 0 1rem; }
.step {
    flex: 1; text-align: center;
    font-family: var(--mono);
    font-size: .58rem; font-weight: 700;
    letter-spacing: .1em; text-transform: uppercase;
    color: var(--muted);
    padding: .5rem .2rem;
    border: 1px solid var(--line-soft);
    border-radius: 999px;
    background: rgba(255,255,255,.03);
    white-space: nowrap; overflow: hidden;
}
.step.on { color: var(--ink); background: var(--grad); border-color: transparent; }

.scan-card {
    display: flex; gap: 18px; align-items: center;
    background: linear-gradient(160deg, #0E1D13, #0A150E);
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 1.1rem 1.2rem;
    margin: .6rem 0 1rem;
}
.scan-ringwrap { position: relative; width: 92px; flex: none; }
.scan-ring {
    width: 92px; aspect-ratio: 1; border-radius: 50%;
    background: conic-gradient(var(--teal), var(--lime) calc(var(--p) * 1%), rgba(255,255,255,.09) 0);
    -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 9px), #000 calc(100% - 8px));
    mask: radial-gradient(farthest-side, transparent calc(100% - 9px), #000 calc(100% - 8px));
}
.scan-ringwrap b {
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono); font-size: 1.05rem; color: var(--lime);
}
.scan-body { min-width: 0; flex: 1; }
.scan-label {
    font-family: var(--mono); font-size: .6rem; font-weight: 700;
    letter-spacing: .2em; text-transform: uppercase; color: var(--muted);
}
.scan-verdict { font-weight: 800; font-size: .92rem; margin-top: .35rem; line-height: 1.35; }
.scan-bar {
    height: 5px; border-radius: 4px;
    background: rgba(255,255,255,.1);
    margin-top: .6rem; overflow: hidden;
}
.scan-bar > span { display: block; height: 100%; border-radius: 4px; background: var(--grad); }

/* ---- Leaderboard -------------------------------------------------------- */
.pod-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 10px;
    text-align: center;
    transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
}
.pod-card:hover {
    transform: translateY(-4px);
    border-color: rgba(168, 255, 96, 0.2);
}
.pod-card.pod-1 {
    border-color: rgba(255, 215, 0, 0.3);
    box-shadow: 0 10px 25px rgba(255, 215, 0, 0.1);
    background: linear-gradient(180deg, rgba(255, 215, 0, 0.03), rgba(255, 255, 255, 0.01));
}
.pod-card.pod-2 {
    border-color: rgba(192, 192, 192, 0.2);
    box-shadow: 0 8px 20px rgba(192, 192, 192, 0.06);
    background: linear-gradient(180deg, rgba(192, 192, 192, 0.02), rgba(255, 255, 255, 0.01));
    margin-top: 15px;
}
.pod-card.pod-3 {
    border-color: rgba(205, 127, 50, 0.15);
    box-shadow: 0 6px 15px rgba(205, 127, 50, 0.05);
    background: linear-gradient(180deg, rgba(205, 127, 50, 0.015), rgba(255, 255, 255, 0.01));
    margin-top: 25px;
}
.pod-imgwrap {
    position: relative; border-radius: 14px; overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
}
.pod-imgwrap img { width: 100%; aspect-ratio: 1; object-fit: cover; display: block; }
.pod-rank {
    position: absolute; top: 6px; left: 6px;
    width: 22px; height: 22px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono); font-weight: 700; font-size: 0.68rem;
}
.pod-1 .pod-rank { background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; }
.pod-2 .pod-rank { background: linear-gradient(135deg, #E0E0E0, #9E9E9E); color: #000; }
.pod-3 .pod-rank { background: linear-gradient(135deg, #CD7F32, #8B5A2B); color: #FFF; }

.pod-addr {
    font-weight: 700; font-size: 0.68rem; margin-top: 0.5rem;
    line-height: 1.25; color: var(--text);
    overflow: hidden; text-overflow: ellipsis; display: -webkit-box;
    -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    height: 1.7rem;
}
.pod-stats {
    display: flex; align-items: center; justify-content: space-between;
    margin-top: 0.45rem; font-family: var(--mono); font-size: 0.68rem;
}
.pod-fire { color: var(--lime); font-weight: 700; }
.pod-gvi-badge {
    padding: 1px 5px; border-radius: 5px; font-weight: 700; font-size: 0.58rem;
}
.pod-author {
    display: flex; align-items: center; justify-content: center; gap: 4px;
    margin-top: 0.45rem; border-top: 1px solid rgba(255,255,255,0.05);
    padding-top: 0.4rem; font-size: 0.62rem; color: var(--muted);
}
.pod-avatar { font-size: 0.7rem; }
.pod-username {
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 65px;
}

.lb-row {
    display: flex; align-items: center; gap: 12px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 18px;
    padding: .65rem .85rem;
    margin-bottom: .6rem;
    transition: transform 0.2s ease, border-color 0.2s ease, background-color 0.2s ease;
}
.lb-row:hover {
    transform: translateX(3px);
    border-color: rgba(168, 255, 96, 0.2);
    background: rgba(168, 255, 96, 0.015);
}
.lb-rank {
    font-family: var(--mono); font-weight: 700; font-size: 0.85rem;
    width: 24px; text-align: center; color: var(--muted); flex: none;
}
.lb-thumb { width: 48px; height: 48px; border-radius: 12px; object-fit: cover; flex: none; border: 1px solid rgba(255,255,255,0.06); }
.lb-main { flex: 1; min-width: 0; }
.lb-addr { font-weight: 800; font-size: 0.8rem; line-height: 1.25; color: var(--text); }
.lb-votebar {
    height: 4px; border-radius: 3px;
    background: rgba(255,255,255,.05);
    margin-top: 0.35rem; overflow: hidden;
}
.lb-votebar > span { display: block; height: 100%; border-radius: 3px; background: var(--grad); }
.lb-sub {
    font-family: var(--mono); color: var(--muted);
    font-size: 0.6rem; letter-spacing: 0.04em; margin-top: 0.25rem;
}
.lb-fire {
    margin-left: auto; font-family: var(--mono); font-weight: 700;
    color: var(--lime); white-space: nowrap; flex: none; font-size: 0.8rem;
}

/* ---- Profile -------------------------------------------------------------- */
.pf-ava-container {
    display: flex;
    justify-content: center;
    margin-top: 1rem;
    margin-bottom: 0.8rem;
    position: relative;
}
.pf-ava-standalone {
    width: 84px; height: 84px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 2.2rem;
    background: linear-gradient(#0D1A12, #0D1A12) padding-box, var(--grad) border-box;
    border: 2px solid transparent;
    box-shadow: 0 10px 30px rgba(0,0,0,.5);
}
.pf-name {
    text-align: center; font-family: var(--display);
    font-weight: 700; font-size: 1.02rem; margin: .6rem 0 0;
}
.pf-sub {
    text-align: center; font-family: var(--mono);
    color: var(--muted); font-size: .62rem;
    letter-spacing: .2em; text-transform: uppercase; margin-top: .3rem;
}

.pg-item {
    position: relative; border-radius: 14px; overflow: hidden;
    aspect-ratio: 3 / 4; background: var(--panel);
    border: 1px solid var(--line-soft);
}
.pg-item img { width: 100%; height: 100%; object-fit: cover; display: block; }
.pg-likes {
    position: absolute; left: 7px; bottom: 6px;
    font-family: var(--mono);
    color: #fff; font-weight: 700; font-size: .72rem;
    text-shadow: 0 1px 5px rgba(0,0,0,.85);
}

/* ---- Floating glass dock (bottom nav) ----------------------------------- */
.st-key-bottom_nav {
    position: fixed;
    bottom: 12px; left: 50%;
    transform: translateX(-50%);
    width: calc(100% - 28px); max-width: 452px;
    z-index: 1000;
    background: rgba(9, 18, 13, .82);
    backdrop-filter: blur(22px);
    border: 1px solid var(--line);
    border-radius: 28px;
    box-shadow: 0 18px 50px rgba(0,0,0,.55), inset 0 1px 0 rgba(255,255,255,.05);
    padding: 8px 12px;
    overflow: visible;
}
.st-key-bottom_nav [data-testid="stHorizontalBlock"] { gap: 4px; align-items: center; }
.st-key-bottom_nav [data-testid="stVerticalBlock"] { gap: 0 !important; }
.st-key-bottom_nav [data-testid="stElementContainer"],
.st-key-bottom_nav .stButton,
.st-key-bottom_nav .stButton > button { width: 100% !important; }
.st-key-bottom_nav .stButton > button {
    background: transparent;
    border: none;
    box-shadow: none;
    color: var(--muted);
    font-family: var(--mono);
    font-size: .56rem;
    font-weight: 700;
    letter-spacing: .02em;
    text-transform: uppercase;
    padding: .4rem 0;
    border-radius: 12px;
    white-space: pre-line !important;
    line-height: 1.35 !important;
    overflow: hidden;
}
.st-key-bottom_nav .stButton > button:hover {
    transform: none; filter: none;
    color: var(--text);
    background: rgba(255,255,255,.05);
    border: none;
}
/* Solarpunk Viewfinder Scan Button */
.st-key-bottom_nav .st-key-nav_camera button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--muted) !important;
    border-radius: 12px !important;
    padding: .4rem 0 !important;
    transition: all 0.2s ease !important;
}
.st-key-bottom_nav .st-key-nav_camera button:hover {
    color: var(--text) !important;
    background: rgba(255,255,255,.05) !important;
}

/* ---- Inbox Screen Styles ---------------------------------------------- */
.inbox-row {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    background: var(--glass);
    backdrop-filter: blur(14px);
    border: 1px solid var(--line-soft);
    border-radius: 18px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.2s ease, box-shadow 0.2s ease;
}
.inbox-row:hover {
    transform: translateY(-2px);
    border-color: rgba(168, 255, 96, 0.35);
    box-shadow: 0 8px 24px rgba(168, 255, 96, 0.08);
}
.inbox-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 42px;
    height: 42px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.04);
    font-size: 1.25rem;
    border: 1px solid var(--line-soft);
    flex-shrink: 0;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
}
.inbox-content {
    flex: 1;
    min-width: 0;
}
.inbox-title-wrap {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 0.25rem;
}
.inbox-title {
    font-weight: 700;
    font-size: 0.88rem;
    color: var(--text);
}
.inbox-time {
    font-family: var(--mono);
    font-size: 0.6rem;
    color: var(--muted);
    white-space: nowrap;
}
.inbox-text {
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.45;
}
.inbox-badge {
    display: inline-block;
    font-family: var(--mono);
    font-size: 0.58rem;
    font-weight: 700;
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.45rem;
    border: 1px solid currentColor;
}

/* ---- Comments dialog styling (normal centered) ---- */
.comments-scroll-area {
    max-height: 300px;
    overflow-y: auto !important;
    padding-right: 4px;
    margin-bottom: 1rem;
}

.comments-scroll-area::-webkit-scrollbar {
    width: 4px;
}
.comments-scroll-area::-webkit-scrollbar-track {
    background: transparent;
}
.comments-scroll-area::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.1);
    border-radius: 2px;
}

.comment-item {
    display: flex; gap: 10px; margin-bottom: 1rem; align-items: flex-start;
}
.comment-avatar {
    width: 32px; height: 32px; border-radius: 50%;
    background: rgba(255,255,255,0.04);
    display: flex; align-items: center; justify-content: center;
    border: 1px solid rgba(255,255,255,0.08);
    font-size: 1.1rem; flex-shrink: 0;
}
.comment-content { min-width: 0; flex-grow: 1; }
.comment-header { display: flex; justify-content: space-between; align-items: baseline; }
.comment-author { font-weight: 700; font-size: 0.82rem; color: var(--text); }
.comment-time { font-size: 0.65rem; color: var(--muted); font-family: var(--mono); }
.comment-text { margin: 2px 0 0 0; font-size: 0.8rem; color: #D1E2D7; line-height: 1.4; word-wrap: break-word; }

/* style the form container inside the comments dialog */
div[data-testid="stForm"]:has(input[placeholder="Добавить комментарий..."]) {
    width: 100% !important;
    background: var(--panel) !important;
    padding: 12px 0 0 0 !important;
    border-top: 1px solid rgba(255,255,255,0.06) !important;
}

.st-key-new_comment_form button,
div[data-testid="stForm"]:has(input[placeholder="Добавить комментарий..."]) button {
    height: 42px !important;
    width: 42px !important;
    min-width: 42px !important;
    padding: 0 !important;
    font-size: 1.1rem !important;
    border-radius: 50% !important;
    margin-top: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: var(--grad) !important;
    color: var(--ink) !important;
    box-shadow: 0 4px 15px rgba(168,255,96,.20) !important;
    border: none !important;
}

.st-key-new_comment_form input,
div[data-testid="stForm"]:has(input[placeholder="Добавить комментарий..."]) input {
    height: 42px !important;
    border-radius: 12px !important;
    font-size: 0.85rem !important;
}
</style>
"""

# Injected only on the feed page: full-bleed snap-scroll "scanner" layout.
FEED_CSS = """
<style>
.block-container { padding: 0 !important; max-width: 480px !important; }
section[data-testid="stMain"], div[data-testid="stAppViewContainer"] {
    scroll-snap-type: y mandatory;
}
.st-key-feedwrap [data-testid="stVerticalBlock"] { gap: 0 !important; }
.st-key-feedwrap [data-testid="stElementContainer"] { margin: 0 !important; width: 100%; }
div[class*="st-key-card_"] { position: relative; }

/* ---- One full-screen card ---- */
.tk-card {
    position: relative;
    height: 100dvh;
    scroll-snap-align: start;
    scroll-snap-stop: always;
    background: #000;
    overflow: hidden;
}
/* Horizontal swipe: slide 1 = AI green future, slide 2 = today */
.tk-swipe {
    display: flex;
    height: 100%;
    overflow-x: auto;
    scroll-snap-type: x mandatory;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: none;
}
.tk-swipe::-webkit-scrollbar { display: none; }
.tk-slide { position: relative; flex: 0 0 100%; height: 100%; scroll-snap-align: center; }
.tk-slide img {
    width: 100%; height: 100%;
    object-fit: cover; display: block;
    user-select: none; -webkit-user-drag: none;
}

/* Viewfinder corner brackets — the "scanner" signature */
.tk-frame {
    position: absolute; inset: 14px;
    z-index: 3; pointer-events: none; opacity: .4;
    background:
        linear-gradient(var(--lime), var(--lime)) left 0 top 0 / 26px 2px,
        linear-gradient(var(--lime), var(--lime)) left 0 top 0 / 2px 26px,
        linear-gradient(var(--lime), var(--lime)) right 0 top 0 / 26px 2px,
        linear-gradient(var(--lime), var(--lime)) right 0 top 0 / 2px 26px,
        linear-gradient(var(--lime), var(--lime)) left 0 bottom 0 / 26px 2px,
        linear-gradient(var(--lime), var(--lime)) left 0 bottom 0 / 2px 26px,
        linear-gradient(var(--lime), var(--lime)) right 0 bottom 0 / 26px 2px,
        linear-gradient(var(--lime), var(--lime)) right 0 bottom 0 / 2px 26px;
    background-repeat: no-repeat;
}

/* Slide tags + slide counter */
.tk-tag {
    position: absolute; top: 62px; left: 16px; z-index: 4;
    display: inline-flex; align-items: center; gap: 7px;
    padding: .38rem .85rem;
    border-radius: 999px;
    font-family: var(--mono);
    font-size: .64rem; font-weight: 700;
    letter-spacing: .14em; text-transform: uppercase;
    background: rgba(8,16,11,.6); color: #fff;
    border: 1px solid rgba(255,255,255,.14);
    backdrop-filter: blur(10px);
}
.tk-tag .rec {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--coral);
    box-shadow: 0 0 8px var(--coral);
    animation: tk-blink 1.4s ease-in-out infinite;
}
.tk-tag.future {
    background: linear-gradient(rgba(8,18,11,.85), rgba(8,18,11,.85)) padding-box,
                var(--grad) border-box;
    border: 1px solid transparent;
    color: var(--lime);
}
.tk-num {
    position: absolute; top: 62px; right: 16px; z-index: 4;
    font-family: var(--mono);
    font-size: .62rem; font-weight: 700; letter-spacing: .18em;
    color: rgba(236,245,238,.8);
    background: rgba(8,16,11,.6);
    border: 1px solid rgba(255,255,255,.12);
    padding: .38rem .6rem; border-radius: 999px;
    backdrop-filter: blur(10px);
}

.tk-grad { position: absolute; left: 0; right: 0; pointer-events: none; z-index: 2; }
.tk-grad.top { top: 0; height: 120px; background: linear-gradient(rgba(2,6,4,.62), transparent); }
.tk-grad.bottom { bottom: 0; height: 260px; background: linear-gradient(transparent, rgba(2,6,4,.78)); }

/* Bottom-left glass info panel */
.tk-info {
    position: absolute; left: 12px; right: 80px; bottom: 96px;
    z-index: 5; pointer-events: none; color: #fff;
    padding: .9rem .95rem;
    border-radius: 20px;
    background: linear-gradient(160deg, rgba(9,18,12,.78), rgba(9,18,12,.52));
    border: 1px solid rgba(255,255,255,.10);
    backdrop-filter: blur(16px);
}
.tk-addr {
    font-family: var(--display);
    font-weight: 700; font-size: .84rem; line-height: 1.35;
}
.tk-meter { margin-top: .6rem; }
.tk-meter-head {
    display: flex; justify-content: space-between; align-items: baseline;
    font-family: var(--mono);
    font-size: .58rem; font-weight: 700;
    letter-spacing: .18em; text-transform: uppercase;
    color: rgba(236,245,238,.6);
}
.tk-meter-head b { font-size: .72rem; }
.tk-meter-bar {
    height: 5px; border-radius: 4px;
    background: rgba(255,255,255,.14);
    margin-top: .35rem; overflow: hidden;
}
.tk-meter-bar > span { display: block; height: 100%; border-radius: 4px; }
.m-low  .tk-meter-bar > span { background: linear-gradient(90deg, var(--coral), var(--amber)); }
.m-mid  .tk-meter-bar > span { background: linear-gradient(90deg, var(--amber), var(--lime)); }
.m-high .tk-meter-bar > span { background: var(--grad); }
.m-low  .tk-meter-head b { color: var(--coral-soft); }
.m-mid  .tk-meter-head b { color: var(--amber); }
.m-high .tk-meter-head b { color: var(--lime); }
.tk-sum {
    margin-top: .55rem; font-size: .8rem; color: rgba(236,245,238,.85);
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden;
}
.tk-hint {
    margin-top: .6rem;
    font-family: var(--mono);
    font-size: .6rem; font-weight: 700;
    letter-spacing: .16em; text-transform: uppercase;
    color: var(--lime);
    animation: tk-pulse 2.4s ease-in-out infinite;
}
@keyframes tk-pulse { 0%, 100% { opacity: .5; } 50% { opacity: 1; } }
@keyframes tk-blink { 0%, 100% { opacity: 1; } 50% { opacity: .25; } }

/* Fixed top bar: brand + live chip */
.tk-top {
    position: fixed; top: 0; left: 50%; transform: translateX(-50%);
    width: 100%; max-width: 480px; z-index: 60;
    display: flex; justify-content: space-between; align-items: center;
    padding: .85rem 1.1rem 1.3rem;
    pointer-events: none;
    background: linear-gradient(rgba(2,6,4,.66), transparent);
}
.tk-logo {
    font-family: var(--display);
    font-weight: 900; font-size: .94rem; letter-spacing: .02em;
    background: var(--grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.tk-live {
    display: flex; align-items: center; gap: 7px;
    font-family: var(--mono);
    font-size: .6rem; font-weight: 700;
    letter-spacing: .18em; text-transform: uppercase;
    color: rgba(236,245,238,.8);
}
.tk-live i {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--lime);
    box-shadow: 0 0 10px var(--lime);
    animation: tk-blink 1.6s ease-in-out infinite;
}

/* TikTok-style profile avatar overlay */
.tk-profile-ava {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    border: 1.5px solid #fff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.35rem;
    background: #09120D;
    box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    position: relative;
    margin: 0 auto;
}
.tk-profile-plus {
    position: absolute;
    bottom: -4px;
    left: 50%;
    transform: translateX(-50%);
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--lime);
    color: #000 !important;
    font-size: 0.65rem;
    font-weight: 900;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.4);
    line-height: 1;
}

/* Action button rail (Like, Comment, Share, Avatar) */
div[class*="st-key-avatar_"] {
    position: absolute; right: 19px; bottom: 330px;
    z-index: 10; width: auto !important;
}
div[class*="st-key-avatar_"] .stButton > button {
    width: 44px; height: 44px;
    border-radius: 50% !important;
    background: #09120D;
    border: 1.5px solid #fff !important;
    color: #fff;
    font-size: 1.35rem;
    padding: 0 !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    display: flex; align-items: center; justify-content: center;
    transition: transform 0.15s ease;
}
div[class*="st-key-avatar_"] .stButton > button:hover {
    transform: scale(1.08);
}
div[class*="_unfollowed"]::after {
    content: "+";
    position: absolute;
    bottom: -4px;
    left: 50%;
    transform: translateX(-50%);
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--lime);
    color: #000 !important;
    font-family: var(--mono);
    font-size: 0.65rem;
    font-weight: 900;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.4);
    line-height: 1;
    pointer-events: none;
}

/* Flat back arrow button for profile */
div[class*="st-key-back_btn"] {
    position: absolute; left: 16px; top: 12px;
    z-index: 110; width: auto !important;
}
div[class*="st-key-back_btn"] .stButton > button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--text) !important;
    font-size: 1.8rem !important;
    padding: 0 !important;
    width: auto !important; height: auto !important;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer;
}
div[class*="st-key-back_btn"] .stButton > button:hover {
    transform: scale(1.15) !important;
    color: var(--lime) !important;
}

/* Burger menu button for profile */
div[class*="st-key-menu_btn"] {
    position: absolute; right: 16px; top: 12px;
    z-index: 110; width: auto !important;
}
div[class*="st-key-menu_btn"] .stButton > button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--text) !important;
    font-size: 1.8rem !important;
    padding: 0 !important;
    width: auto !important; height: auto !important;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer;
}
div[class*="st-key-menu_btn"] .stButton > button:hover {
    transform: scale(1.15) !important;
    color: var(--lime) !important;
}

/* Follow Button Flat & Sleek */
div[class*="st-key-follow_btn_unfollowed"] {
    width: 100% !important;
    margin-bottom: 0.5rem;
}
div[class*="st-key-follow_btn_unfollowed"] button {
    background: var(--lime) !important;
    color: var(--ink) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
    font-size: 0.88rem !important;
    padding: 0.6rem 1rem !important;
    box-shadow: 0 4px 12px rgba(168, 255, 96, 0.15) !important;
    transition: transform 0.15s ease, filter 0.15s ease !important;
}
div[class*="st-key-follow_btn_unfollowed"] button:hover {
    transform: translateY(-1px) !important;
    filter: brightness(1.08) !important;
}

div[class*="st-key-follow_btn_followed"] {
    width: 100% !important;
    margin-bottom: 0.5rem;
}
div[class*="st-key-follow_btn_followed"] button {
    background: rgba(255, 255, 255, 0.05) !important;
    color: var(--text) !important;
    border: 1px solid var(--line-soft) !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    padding: 0.6rem 1rem !important;
    box-shadow: none !important;
    transition: background 0.15s ease, color 0.15s ease !important;
}
div[class*="st-key-follow_btn_followed"] button:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(255, 255, 255, 0.15) !important;
}

div[class*="st-key-like_"] {
    position: absolute; right: 12px; bottom: 252px;
    z-index: 8; width: auto !important;
}
div[class*="st-key-like_"] .stButton > button {
    width: 58px; height: 58px;
    border-radius: 50% !important;
    background: rgba(9,18,12,.6);
    backdrop-filter: blur(12px);
    border: 1px solid var(--line);
    color: #fff;
    font-family: var(--mono);
    font-size: .7rem; font-weight: 700;
    padding: 0; line-height: 1.2;
    box-shadow: 0 8px 24px rgba(0,0,0,.45);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
div[class*="st-key-like_"] .stButton > button:hover {
    transform: scale(1.08);
    border-color: rgba(168,255,96,.6);
}

div[class*="st-key-comment_"] {
    position: absolute; right: 12px; bottom: 174px;
    z-index: 8; width: auto !important;
}
div[class*="st-key-comment_"] .stButton > button {
    width: 58px; height: 58px;
    border-radius: 50% !important;
    background: rgba(9,18,12,.6);
    backdrop-filter: blur(12px);
    border: 1px solid var(--line);
    color: #fff;
    font-family: var(--mono);
    font-size: .7rem; font-weight: 700;
    padding: 0; line-height: 1.2;
    box-shadow: 0 8px 24px rgba(0,0,0,.45);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
div[class*="st-key-comment_"] .stButton > button:hover {
    transform: scale(1.08);
    border-color: rgba(168,255,96,.6);
}

div[class*="st-key-share_"] {
    position: absolute; right: 12px; bottom: 96px;
    z-index: 8; width: auto !important;
}
div[class*="st-key-share_"] .stButton > button {
    width: 58px; height: 58px;
    border-radius: 50% !important;
    background: rgba(9,18,12,.6);
    backdrop-filter: blur(12px);
    border: 1px solid var(--line);
    color: #fff;
    font-family: var(--mono);
    font-size: 1.2rem; font-weight: 700;
    padding: 0; line-height: 1.15;
    box-shadow: 0 8px 24px rgba(0,0,0,.45);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
div[class*="st-key-share_"] .stButton > button:hover {
    transform: scale(1.08);
    border-color: rgba(168,255,96,.6);
}

/* Empty feed state */
.tk-empty {
    height: 100dvh;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: .7rem; text-align: center; padding: 0 2.2rem;
}
.tk-empty .logo {
    font-family: var(--display); font-weight: 900; font-size: 1.7rem;
    background: var(--grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.tk-empty .sub { color: var(--muted); font-size: .86rem; line-height: 1.55; }
.tk-empty .mono {
    font-family: var(--mono); font-size: .62rem;
    letter-spacing: .2em; text-transform: uppercase; color: var(--lime);
}
</style>
"""


# ===========================================================================
# Helpers
# ===========================================================================
def pil_to_base64(image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string for JSON storage."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def format_timestamp(iso_string: str) -> str:
    """Human-friendly date for compact cards."""
    try:
        return datetime.fromisoformat(iso_string).strftime("%d.%m.%Y %H:%M")
    except (ValueError, TypeError):
        return ""


def time_ago(iso_string: str) -> str:
    """Relative timestamp: 'только что', '5 мин', '3 ч', '2 дн'."""
    try:
        moment = datetime.fromisoformat(iso_string)
    except (ValueError, TypeError):
        return ""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    seconds = (datetime.now(timezone.utc) - moment).total_seconds()
    if seconds < 60:
        return "только что"
    if seconds < 3600:
        return f"{int(seconds // 60)} мин"
    if seconds < 86400:
        return f"{int(seconds // 3600)} ч"
    if seconds < 86400 * 30:
        return f"{int(seconds // 86400)} дн"
    return moment.strftime("%d.%m.%Y")


def is_current_month(iso_string: str) -> bool:
    try:
        moment = datetime.fromisoformat(iso_string)
    except (ValueError, TypeError):
        return False
    now = datetime.now(timezone.utc)
    return moment.year == now.year and moment.month == now.month


RU_MONTHS = [
    "Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
    "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь",
]


def gvi_level(green_index: float) -> tuple[str, str]:
    """(css-suffix, human label) for a Green View Index value."""
    if green_index < 0.25:
        return "low", "критично"
    if green_index < 0.5:
        return "mid", "средне"
    return "high", "зелено"


def gvi_badge(green_index: float) -> str:
    """Return an HTML badge classifying the Green View Index."""
    level, label = gvi_level(green_index)
    return (
        f'<span class="feed-badge badge-{level}">'
        f"{label} · GVI {green_index:.2f}</span>"
    )


def reset_idea_form() -> None:
    """Clear the camera-page state after a successful publish or exit."""
    for key in (
        "analysis_result",
        "generated_image",
        "original_image",
        "eco_audit",
        "ai_analysis",
        "appeal_text",
        "analyzed_signature",
        "confirmed_signature",
    ):
        st.session_state.pop(key, None)
    # Bump the key so camera_input / file_uploader widgets reset themselves.
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1


def _camera_dirty() -> bool:
    """True when the camera page holds confirmed/generated work not yet saved."""
    return bool(
        st.session_state.get("confirmed_signature")
        or st.session_state.get("generated_image")
    )


def goto(page: str) -> None:
    """Switch pages, guarding an unsaved camera session with a warning dialog."""
    if (
        st.session_state.get("page") == "camera"
        and page != "camera"
        and _camera_dirty()
    ):
        # Defer the switch — render_camera() will surface the confirm dialog.
        st.session_state["confirm_leave_to"] = page
        return
    st.session_state["page"] = page
    st.session_state.pop("view_profile_id", None)


def request_leave(target: str) -> None:
    """Exit-button handler: warn if there is unsaved work, else leave directly."""
    if _camera_dirty():
        st.session_state["confirm_leave_to"] = target
    else:
        st.session_state["page"] = target


# Data-version cache invalidation: bumping this key busts the feed/leaderboard
# caches for the current session right after a publish or vote, while other
# sessions pick up changes within the cache TTL.
def _data_version() -> int:
    return st.session_state.get("data_version", 0)


def _bump_data_version() -> None:
    st.session_state["data_version"] = st.session_state.get("data_version", 0) + 1


@st.cache_data(ttl=20, show_spinner=False)
def _feed_cached(version: int, limit: int) -> list[dict]:
    return load_posts(limit)


@st.cache_data(ttl=20, show_spinner=False)
def _all_posts_cached(version: int) -> list[dict]:
    return load_all_posts()


ENGINE_LABELS = {
    "openai": "✨ OpenAI gpt-image",
    "flux-kontext": "🟢 FLUX.1-Kontext (бесплатно)",
    "instruct-pix2pix": "🟡 InstructPix2Pix (бесплатно)",
    "flux-schnell-text": "⚠️ текстовая генерация (без фото)",
    "none": "⚠️ генерация недоступна",
}


# ===========================================================================
# Feed page — full-screen "city scanner" reels
# ===========================================================================
def feed_card_html(item: dict) -> str:
    original_b64 = item.get("image_original", "")
    generated_b64 = item.get("image_generated", "")
    author = html.escape(item.get("author") or "eco_citizen")
    avatar = html.escape(item.get("avatar") or "🌱")
    address = html.escape(item.get("address", "Без адреса"))
    when = time_ago(item.get("timestamp", ""))
    green_index = float(item.get("green_index", 0.0))
    level, label = gvi_level(green_index)
    summary = item.get("ai_summary") or next(iter(item.get("ai_problems") or []), "")
    summary_html = (
        f'<div class="tk-sum">🤖 {html.escape(summary)}</div>' if summary else ""
    )

    total = 2 if generated_b64 else 1
    slides = ""
    if generated_b64:
        slides += (
            f'<div class="tk-slide"><img src="data:image/png;base64,{generated_b64}" alt=""/>'
            f'<div class="tk-num">01/{total:02d}</div></div>'
        )
        slides += (
            f'<div class="tk-slide"><img src="data:image/png;base64,{original_b64}" alt=""/>'
            f'<div class="tk-num">02/{total:02d}</div></div>'
        )
    else:
        slides += (
            f'<div class="tk-slide"><img src="data:image/png;base64,{original_b64}" alt=""/>'
            f'<div class="tk-num">01/{total:02d}</div></div>'
        )

    return (
        f'<div class="tk-card"><div class="tk-swipe">{slides}</div>'
        '<div class="tk-grad top"></div><div class="tk-grad bottom"></div>'
        '<div class="tk-info">'
        f'<div class="tk-addr">{address}</div>'
        f'<div class="tk-meter m-{level}">'
        f'<div class="tk-meter-head"><span>индекс зелени · {label}</span>'
        f"<b>{green_index:.2f}</b></div>"
        f'<div class="tk-meter-bar"><span style="width:{green_index * 100:.0f}%"></span></div>'
        f"</div>{summary_html}"
        "</div></div>"
    )


@st.dialog("💬 Комментарии")
def _show_comments_dialog(post_id: str) -> None:
    user_id = get_current_user_id()
    comments = load_comments(post_id)
    
    comments_html = '<div class="comments-scroll-area">'
    if not comments:
        comments_html += "<p style='color: var(--muted); text-align: center; margin: 3rem 0; font-size: 0.9rem;'>Комментариев пока нет. Будьте первым!</p>"
    else:
        for c in comments:
            author_name = html.escape(c.get("author") or "user")
            avatar_char = html.escape(c.get("avatar") or "🌱")
            content_text = html.escape(c.get("content", ""))
            created_time = time_ago(c.get("created_at", ""))
            comments_html += f"""
            <div class="comment-item">
                <div class="comment-avatar">{avatar_char}</div>
                <div class="comment-content">
                    <div class="comment-header">
                        <span class="comment-author">@{author_name}</span>
                        <span class="comment-time">{created_time}</span>
                    </div>
                    <p class="comment-text">{content_text}</p>
                </div>
            </div>
            """
    comments_html += '</div>'
    st.markdown(comments_html, unsafe_allow_html=True)
    
    with st.form("new_comment_form", border=False):
        c1, c2 = st.columns([5.0, 1.0])
        with c1:
            new_content = st.text_input("Напишите комментарий...", placeholder="Добавить комментарий...", label_visibility="collapsed")
        with c2:
            if st.form_submit_button("💬", type="primary", use_container_width=True):
                if new_content.strip():
                    if user_id:
                        save_comment(user_id, post_id, new_content.strip())
                        _bump_data_version()
                        st.rerun()
                    else:
                        st.error("Пожалуйста, войдите в аккаунт.")


def render_feed() -> None:
    st.markdown(FEED_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="tk-top" style="justify-content: center;">
            <span class="tk-logo">ECO//RE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    user_id = get_current_user_id()
    feed_items = _feed_cached(_data_version(), 20)

    if not feed_items:
        st.markdown(
            """
            <div class="tk-empty">
                <div class="logo">ECO//RE</div>
                <div class="mono">// лента пуста</div>
                <div class="sub">
                    Нажми ◉ внизу — сфоткай серую улицу,
                    и ИИ нарисует её зелёное будущее
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    liked_post_ids: set[str] = (
        get_user_liked_posts(user_id) if user_id else set()
    )

    with st.container(key="feedwrap"):
        for item in feed_items:
            item_id = item["id"]
            with st.container(key=f"card_{item_id}"):
                st.markdown(feed_card_html(item), unsafe_allow_html=True)
                
                # 0. Avatar button (bottom: 330px)
                author_id = item.get("user_id")
                is_following_author = is_following(user_id, author_id) if user_id and author_id else False
                is_self = (user_id == author_id) if user_id and author_id else True
                
                avatar_key_suffix = "self" if is_self else ("followed" if is_following_author else "unfollowed")
                avatar_key = f"avatar_{item_id}_{avatar_key_suffix}"
                
                if st.button(item.get("avatar", "🌱"), key=avatar_key):
                    if author_id:
                        st.session_state["view_profile_id"] = author_id
                        st.session_state["page"] = "profile"
                        st.rerun()
                
                # 1. Like button (bottom: 252px)
                like_count = int(item.get("likes", 0))
                is_liked = item_id in liked_post_ids
                like_label = f"💚\n{like_count}" if is_liked else f"🔥\n{like_count}"
                if st.button(like_label, key=f"like_{item_id}"):
                    if user_id:
                        now_liked, new_count = toggle_like(user_id, item_id)
                        _bump_data_version()
                        st.rerun()
                
                # 2. Comment button (bottom: 174px)
                c_count = item.get("comments_count", 0)
                comment_label = f"💬\n{c_count}"
                if st.button(comment_label, key=f"comment_{item_id}"):
                    _show_comments_dialog(item_id)
                
                # 3. Share button (bottom: 96px)
                if st.button("🔗", key=f"share_{item_id}"):
                    st.session_state["flash"] = "Ссылка на пост скопирована! 🔗"
                    # Add tiny JS to copy to clipboard
                    components.html(
                        f"""<script>
                        navigator.clipboard.writeText(window.parent.location.origin + "/?post={item_id}");
                        </script>""",
                        height=0,
                    )
                    st.rerun()

    # Inject Drag-Emulation JS Script
    st.markdown("""
    <script>
    (function() {
        function initDragEmulation() {
            const containers = document.querySelectorAll('.tk-swipe');
            containers.forEach(container => {
                if (container.dataset.dragInitialized) return;
                container.dataset.dragInitialized = 'true';

                let isDown = false;
                let startX;
                let scrollLeft;

                container.addEventListener('mousedown', (e) => {
                    isDown = true;
                    startX = e.pageX - container.offsetLeft;
                    scrollLeft = container.scrollLeft;
                    container.style.scrollSnapType = 'none';
                    container.style.scrollBehavior = 'auto';
                });

                container.addEventListener('mouseleave', () => {
                    if (!isDown) return;
                    isDown = false;
                    container.style.scrollSnapType = 'x mandatory';
                });

                container.addEventListener('mouseup', () => {
                    if (!isDown) return;
                    isDown = false;
                    container.style.scrollSnapType = 'x mandatory';
                    
                    const width = container.offsetWidth;
                    const scrollPos = container.scrollLeft;
                    const slideIndex = Math.round(scrollPos / width);
                    container.style.scrollBehavior = 'smooth';
                    container.scrollTo({ left: slideIndex * width });
                    
                    // Update tag/counter manually since native scroll snap changed
                    const card = container.closest('.tk-card');
                    if (card) {
                        const num = card.querySelector('.tk-num');
                        const hint = card.querySelector('.tk-hint');
                        if (num) {
                            const totalVal = num.textContent.split('/')[1] || '02';
                            num.textContent = `${(slideIndex + 1).toString().padStart(2, '0')}/${totalVal}`;
                        }
                        if (hint) {
                            hint.innerHTML = slideIndex === 0 ? '⇆ свайп — сейчас' : '⇆ свайп — прогноз ИИ';
                        }
                    }
                });

                container.addEventListener('mousemove', (e) => {
                    if(!isDown) return;
                    e.preventDefault();
                    const x = e.pageX - container.offsetLeft;
                    const walk = (x - startX) * 1.5;
                    container.scrollLeft = scrollLeft - walk;
                });
            });
        }

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initDragEmulation);
        } else {
            initDragEmulation();
        }
        
        if (!window.hasDragInterval) {
            window.hasDragInterval = true;
            setInterval(initDragEmulation, 800);
        }
    })();
    </script>
    """, unsafe_allow_html=True)



# ===========================================================================
# Camera page — shoot on the spot or upload, then the AI pipeline
# ===========================================================================
def render_steps(active: int) -> None:
    labels = ["01 · фото", "02 · ии-аудит", "03 · публикация"]
    chips = "".join(
        f'<span class="step{" on" if i + 1 <= active else ""}">{label}</span>'
        for i, label in enumerate(labels)
    )
    st.markdown(f'<div class="steps">{chips}</div>', unsafe_allow_html=True)


@st.dialog("Выйти из режима камеры?")
def _confirm_leave_dialog() -> None:
    """Warn that unsaved camera work is lost before leaving the page."""
    target = st.session_state.get("confirm_leave_to", "feed")
    st.markdown(
        '<p style="color:var(--text);font-size:.92rem;line-height:1.6;">'
        "⚠️ <strong>Изменения не сохранятся, если выйти.</strong><br/>"
        "Сгенерированный зелёный концепт и ИИ-анализ будут потеряны, "
        "пока вы не опубликуете решение в ленту."
        "</p>",
        unsafe_allow_html=True,
    )
    col_stay, col_leave = st.columns(2)
    with col_stay:
        if st.button("← Остаться", type="primary", use_container_width=True):
            st.session_state.pop("confirm_leave_to", None)
            st.rerun()
    with col_leave:
        if st.button("Выйти без сохранения", use_container_width=True):
            reset_idea_form()
            st.session_state["page"] = target
            st.session_state.pop("confirm_leave_to", None)
            st.rerun()


# Make the live camera feel like a full-screen viewfinder.
CAMERA_CSS = """<style>
div[data-testid="stCameraInput"] > div { width: 100% !important; }
div[data-testid="stCameraInput"] video {
    width: 100% !important;
    height: 60vh !important;
    max-height: 60vh !important;
    object-fit: cover !important;
    border-radius: 18px;
    border: 1px solid var(--line);
    box-shadow: 0 0 0 1px rgba(168,255,96,.12), 0 18px 50px rgba(0,0,0,.5);
}
div[data-testid="stCameraInput"] img { border-radius: 18px; }
</style>"""


def render_camera() -> None:
    # Surface the exit-confirmation dialog when navigation was deferred.
    if st.session_state.get("confirm_leave_to"):
        _confirm_leave_dialog()

    st.markdown(CAMERA_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="page-head">
            <div class="kicker">// новое решение</div>
            <h1>Сканер улицы</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_mode = st.segmented_control(
        "Источник фото",
        options=["📷 Камера", "🖼 Галерея"],
        default="📷 Камера",
        key="source_mode",
        label_visibility="collapsed",
    )

    # --- Подсказка по качеству фото ----------------------------------------
    st.markdown(
        """
        <div class="eco-card" style="
            border-color: rgba(61,245,200,.3);
            background: rgba(61,245,200,.05);
            padding: .75rem 1rem; margin-bottom: .6rem;">
            📸 <strong>Советы по фото:</strong> снимайте
            <strong>улицы, дороги, парки и дворы — без людей</strong>
            и лишних объектов в кадре.
            Меньше людей и машин → точнее анализ ИИ и реалистичнее
            зелёный концепт.
        </div>
        """,
        unsafe_allow_html=True,
    )

    widget_generation = st.session_state.get("uploader_key", 0)
    if source_mode == "📷 Камера":
        photo_file = st.camera_input(
            "Фото локации",
            key=f"camera_{widget_generation}",
            label_visibility="collapsed",
        )
    else:
        photo_file = st.file_uploader(
            "Фото локации",
            type=["png", "jpg", "jpeg", "webp"],
            key=f"uploader_{widget_generation}",
            label_visibility="collapsed",
        )

    current_sig = (
        f"{getattr(photo_file, 'name', 'camera')}_{photo_file.size}"
        if photo_file is not None
        else None
    )
    confirmed_sig = st.session_state.get("confirmed_signature")

    # --- No photo yet --------------------------------------------------------
    if photo_file is None:
        if confirmed_sig:
            # The photo was cleared from the widget — drop the stale pipeline.
            reset_idea_form()
        render_steps(1)
        st.markdown(
            """
            <div class="eco-card" style="text-align:center;color:var(--muted);">
                ⬆️ Сделай фото на месте или выбери из галереи —
                затем подтверди кадр, и начнётся эко-аудит локации
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # --- Photo selected but NOT confirmed → preview + confirm gate -----------
    if confirmed_sig != current_sig:
        render_steps(1)
        photo_file.seek(0)
        preview_image = Image.open(photo_file).convert("RGB")
        st.image(preview_image, caption="📷 Предпросмотр кадра", width="stretch")
        st.markdown(
            """
            <div class="eco-card" style="
                border-color: rgba(61,245,200,.3);
                background: rgba(61,245,200,.05);
                text-align:center; padding:.8rem 1rem;">
                Проверь кадр. Если всё хорошо — нажми
                <strong>«Подтвердить»</strong>, и ИИ запустит анализ и
                генерацию зелёного концепта.
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_confirm, col_retake = st.columns([2, 1])
        with col_confirm:
            if st.button(
                "✓ Подтвердить и запустить ИИ",
                type="primary",
                use_container_width=True,
                key="confirm_photo",
            ):
                st.session_state["confirmed_signature"] = current_sig
                st.rerun()
        with col_retake:
            if st.button("↺ Другое фото", use_container_width=True, key="retake_photo"):
                reset_idea_form()
                st.rerun()
        return

    # --- Confirmed → run pipeline once, cached by signature ------------------
    photo_file.seek(0)
    original_image = Image.open(photo_file).convert("RGB")
    file_signature = current_sig
    if st.session_state.get("analyzed_signature") != file_signature:
        # 1) Real image analysis first: YOLOv8 + OpenCV eco-audit.
        with st.spinner("🛰 YOLOv8 анализирует улицу..."):
            eco_audit = eco_audit_safe(original_image)
            green_index = eco_audit["green_view_index"] / 100.0
            masked_image, _ = analyze_eco_status(original_image)
        # 2) Image-to-image, with a decay-driven luxury prompt from the audit.
        with st.spinner("🎨 ИИ превращает улицу в люкс-центр мегаполиса..."):
            generated_image = generate_eco_friendly_view(original_image, eco_audit)
        st.session_state["analyzed_signature"] = file_signature
        st.session_state["original_image"] = original_image
        st.session_state["eco_audit"] = eco_audit
        st.session_state["analysis_result"] = (masked_image, green_index)
        st.session_state["generated_image"] = generated_image

    masked_image, green_index = st.session_state["analysis_result"]
    generated_image = st.session_state["generated_image"]
    original_image = st.session_state["original_image"]
    eco_audit = st.session_state["eco_audit"]

    render_steps(3 if st.session_state.get("location_input", "").strip() else 2)

    # --- Scan result: GVI ring gauge -------------------------------------
    verdict = (
        "Району срочно нужно озеленение 🌵"
        if green_index < 0.25
        else "Есть зелень, но можно лучше 🌿"
        if green_index < 0.5
        else "Отличный зелёный район! 🌳"
    )
    st.markdown(
        f"""
        <div class="scan-card">
            <div class="scan-ringwrap">
                <div class="scan-ring" style="--p:{green_index * 100:.0f}"></div>
                <b>{green_index:.2f}</b>
            </div>
            <div class="scan-body">
                <div class="scan-label">green view index</div>
                <div class="scan-verdict">{verdict}</div>
                <div class="scan-bar"><span style="width:{green_index * 100:.0f}%"></span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Street Decay Index: the "how rough is this street" score ---------
    decay = float(eco_audit.get("decay_index", 0.0))
    decay_label = eco_audit.get("decay_label", "")
    decay_cls = "low" if decay >= 62 else "mid" if decay >= 42 else "high"
    st.markdown(
        f"""
        <div class="eco-card m-{decay_cls}" style="margin-bottom:1rem;">
            <div class="tk-meter-head" style="display:flex;justify-content:space-between;
                font-family:var(--mono);font-size:.6rem;font-weight:700;
                letter-spacing:.16em;text-transform:uppercase;color:var(--muted);">
                <span>индекс упадка улицы</span><b style="font-size:.8rem;">{decay:.0f}/100</b>
            </div>
            <div class="tk-meter-bar" style="height:6px;border-radius:4px;
                background:rgba(255,255,255,.12);margin-top:.45rem;overflow:hidden;">
                <span style="display:block;height:100%;width:{decay:.0f}%;
                background:linear-gradient(90deg,var(--lime),var(--amber),var(--coral));"></span>
            </div>
            <div style="margin-top:.5rem;font-size:.82rem;color:var(--text);">
                🏚 {html.escape(decay_label)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Before / analysis / future concept -------------------------------
    col_original, col_masked = st.columns(2)
    with col_original:
        st.image(original_image, caption="📷 Оригинал", width="stretch")
    with col_masked:
        st.image(masked_image, caption="🛰 Анализ озеленения", width="stretch")

    st.markdown(
        '<div class="section-title">✦ Будущий зелёный концепт</div>',
        unsafe_allow_html=True,
    )
    st.image(generated_image, caption="🎨 Future Green Concept (GenAI)", width="stretch")

    engine = st.session_state.get("last_gen_engine")
    if engine and engine in ENGINE_LABELS:
        st.caption(f"Движок генерации: {ENGINE_LABELS[engine]}")

    # --- Дисклеймер точности ИИ -------------------------------------------
    st.markdown(
        """
        <div class="eco-card" style="
            border-color: rgba(255,194,75,.35);
            background: rgba(255,194,75,.06);
            padding: .75rem 1rem; font-size: .82rem; color: var(--muted);">
            ⚠️ <strong style="color:var(--amber);">Важно:</strong>
            результаты анализа и изображения сгенерированы ИИ и могут быть
            неточными. Воспринимайте их как отправную точку — всегда
            проверяйте данные перед использованием в официальных обращениях.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Location input ----------------------------------------------------
    location_address = st.text_input(
        "📍 Адрес / Локация",
        placeholder="например: Алматы, ул. Абая 44",
        key="location_input",
    )

    # --- Claude AI analysis section ------------------------------------------
    st.markdown(
        '<div class="section-title">🤖 ИИ-анализ локации</div>',
        unsafe_allow_html=True,
    )

    if not free_ai_available():
        st.info("Добавьте HF_API_TOKEN в secrets.toml для ИИ-анализа")
    else:
        # Findings run on demand (a button), not on every keystroke — this
        # keeps the page responsive and avoids repeated LLM calls while typing.
        analysis_label = (
            "🔄 Обновить ИИ-анализ"
            if st.session_state.get("ai_analysis")
            else "🤖 Запустить ИИ-анализ локации"
        )
        if st.button(analysis_label, key="run_findings"):
            with st.spinner("🤖 ИИ формулирует выводы по данным анализа..."):
                ai_analysis = generate_findings(
                    eco_audit,
                    location_address.strip() or "адрес не указан",
                )
            if ai_analysis is None:
                st.info("Добавьте HF_API_TOKEN в secrets.toml для ИИ-анализа")
            else:
                st.session_state["ai_analysis"] = ai_analysis
                # A fresh analysis invalidates any previously drafted appeal.
                st.session_state.pop("appeal_text", None)

        ai_analysis = st.session_state.get("ai_analysis")
        if ai_analysis:
            priority = ai_analysis.get("priority", "средний").lower()
            badge_class = {
                "высокий": "badge-low",
                "средний": "badge-mid",
                "низкий": "badge-high",
            }.get(priority, "badge-mid")
            problems_html = "".join(
                f"<div>⚠️ {html.escape(p)}</div>" for p in ai_analysis["problems"]
            )
            recommendations_html = "".join(
                f"<div>🌱 {html.escape(r)}</div>"
                for r in ai_analysis["recommendations"]
            )
            st.markdown(
                f"""
                <div class="eco-card">
                    <span class="feed-badge {badge_class}">приоритет: {priority}</span>
                    <p style="margin:.55rem 0 .6rem 0;"><em>{html.escape(ai_analysis["summary"])}</em></p>
                    <div class="section-title" style="font-size:.92rem;">Проблемы</div>
                    {problems_html}
                    <div class="section-title" style="font-size:.92rem;">Рекомендации</div>
                    {recommendations_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("📄 Сгенерировать обращение в акимат"):
                with st.spinner("📄 ИИ составляет официальное обращение..."):
                    appeal_text = generate_appeal_text(
                        location_address.strip() or "адрес не указан",
                        green_index,
                        ai_analysis["problems"],
                        ai_analysis["recommendations"],
                    )
                if appeal_text:
                    st.session_state["appeal_text"] = appeal_text
                else:
                    st.info("Добавьте HF_API_TOKEN в secrets.toml для ИИ-анализа")

            if st.session_state.get("appeal_text"):
                st.caption(
                    "В тексте оставлены заполнители в [квадратных скобках] — "
                    "впишите свои данные (ФИО, телефон, дату) перед отправкой."
                )
                st.text_area(
                    "Текст обращения",
                    value=st.session_state["appeal_text"],
                    height=200,
                    disabled=True,
                    label_visibility="collapsed",
                )
                with st.expander("📋 Копировать"):
                    st.code(st.session_state["appeal_text"], language=None)

    # --- Publish ----------------------------------------------------------------
    if st.button("✦ Опубликовать в ленту", type="primary"):
        if not location_address.strip():
            st.warning("Пожалуйста, укажите адрес или название локации 📍")
        else:
            user_id = get_current_user_id()
            saved_analysis = st.session_state.get("ai_analysis") or {}
            save_post(
                user_id,
                {
                    "address": location_address.strip(),
                    "green_index": round(green_index, 3),
                    "image_original": pil_to_base64(original_image),
                    "image_generated": pil_to_base64(generated_image),
                    "ai_problems": saved_analysis.get("problems", []),
                    "ai_recommendations": saved_analysis.get("recommendations", []),
                    "ai_priority": saved_analysis.get("priority", ""),
                    "ai_summary": saved_analysis.get("summary", ""),
                },
            )
            reset_idea_form()
            _bump_data_version()
            st.session_state["flash"] = "Решение опубликовано в ленту! 🎉"
            st.session_state["page"] = "feed"
            st.rerun()


# ===========================================================================
# Leaderboard page — top voted places this month (+ akimat analytics)
# ===========================================================================
def _thumb_src(item: dict) -> str:
    encoded = item.get("image_generated") or item.get("image_original") or ""
    return f"data:image/png;base64,{encoded}"


def render_top() -> None:
    now = datetime.now(timezone.utc)
    st.markdown(
        f"""
        <div class="page-head">
            <div class="kicker">// рейтинг месяца · {RU_MONTHS[now.month - 1].lower()} {now.year}</div>
            <h1>Лидерборд</h1>
            <p>Места с самой высокой поддержкой горожан</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_items = _all_posts_cached(_data_version())
    monthly_items = [i for i in all_items if is_current_month(i.get("timestamp", ""))]
    pool = monthly_items
    if not pool and all_items:
        st.caption("В этом месяце публикаций пока нет — показываем топ за всё время.")
        pool = all_items

    ranked = sorted(pool, key=lambda i: int(i.get("likes", 0)), reverse=True)[:10]
    max_votes = max((int(i.get("likes", 0)) for i in ranked), default=1) or 1

    if not ranked:
        st.info("Пока нет публикаций. Стань первым — нажми ◉ внизу!")
    else:
        # --- Podium for the top-3 (2nd · 1st · 3rd, like real podiums) ----
        if len(ranked) >= 3:
            podium_order = [(ranked[1], 2), (ranked[0], 1), (ranked[2], 3)]
            cols = st.columns([1, 1.12, 1])
            for col, (item, rank) in zip(cols, podium_order):
                with col:
                    gvi = float(item.get("green_index", 0.0))
                    if gvi < 0.4:
                        gvi_color = "#FF7A5C"
                    elif gvi < 0.7:
                        gvi_color = "#FFC24B"
                    else:
                        gvi_color = "#A8FF60"
                        
                    author = html.escape(item.get("author") or "eco_citizen")
                    avatar = html.escape(item.get("avatar") or "🌱")
                    address = html.escape(item.get("address", "Без адреса"))
                    votes = int(item.get("likes", 0))
                    
                    st.markdown(
                        f"""
                        <div class="pod-card pod-{rank}">
                            <div class="pod-imgwrap">
                                <img src="{_thumb_src(item)}" alt=""/>
                                <div class="pod-rank">{rank}</div>
                            </div>
                            <div class="pod-addr">{address}</div>
                            <div class="pod-stats">
                                <span class="pod-fire">🔥 {votes}</span>
                                <span class="pod-gvi-badge" style="background: {gvi_color}18; color: {gvi_color}; border: 1px solid {gvi_color}35;">{gvi:.2f}</span>
                            </div>
                            <div class="pod-author">
                                <span class="pod-avatar">{avatar}</span>
                                <span class="pod-username">@{author}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            rest = ranked[3:]
            start_rank = 4
        else:
            rest = ranked
            start_rank = 1

        if rest:
            st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        rows_html = ""
        for offset, item in enumerate(rest):
            votes = int(item.get("likes", 0))
            author = html.escape(item.get("author") or "eco_citizen")
            avatar = html.escape(item.get("avatar") or "🌱")
            address = html.escape(item.get("address", "Без адреса"))
            rows_html += (
                f'<div class="lb-row"><div class="lb-rank">{start_rank + offset:02d}</div>'
                f'<img class="lb-thumb" src="{_thumb_src(item)}" alt=""/>'
                f'<div class="lb-main"><div class="lb-addr">{address}</div>'
                f'<div class="lb-votebar"><span style="width:{votes / max_votes * 100:.0f}%"></span></div>'
                f'<div class="lb-sub">{avatar} @{author} · gvi {float(item.get("green_index", 0.0)):.2f}</div></div>'
                f'<div class="lb-fire">🔥 {votes}</div></div>'
            )
        if rows_html:
            st.markdown(rows_html, unsafe_allow_html=True)


@st.dialog("⚙️ Настройки и конфиденциальность")
def _show_settings_dialog() -> None:
    current_user_id = get_current_user_id()
    db_user = get_current_user()
    profile_avatar = db_user.get("avatar") or "🌱"
    profile_email = db_user.get("email") or ""

    st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
    with st.form("dialog_profile_form", border=False):
        current_idx = AVATARS.index(profile_avatar) if profile_avatar in AVATARS else 0
        new_avatar = st.selectbox("Изменить аватар", AVATARS, index=current_idx)
        if profile_email:
            st.caption(f"📧 {profile_email}")
        if st.form_submit_button("💾 Сохранить", type="primary", use_container_width=True):
            if current_user_id:
                update_user_avatar(current_user_id, new_avatar)
                st.session_state["db_user"]["avatar"] = new_avatar
                st.session_state["flash"] = "Аватар обновлён ✅"
                st.rerun()
            
    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1.2rem 0;'>", unsafe_allow_html=True)
    if st.button("🚪 Выйти из аккаунта", type="primary", use_container_width=True):
        logout()
        st.rerun()


# ===========================================================================
# Profile page — aurora banner, gradient-ring avatar, solutions grid
# ===========================================================================
def render_profile() -> None:
    current_user_id = get_current_user_id()
    target_user_id = st.session_state.get("view_profile_id")
    
    is_own_profile = (target_user_id is None) or (target_user_id == current_user_id)
    
    if is_own_profile:
        db_user = get_current_user()
        profile_user_id = current_user_id
    else:
        db_user = get_user_by_id(target_user_id)
        if not db_user:
            st.error("Пользователь не найден.")
            if st.button("← Вернуться в ленту", use_container_width=True):
                st.session_state.pop("view_profile_id", None)
                st.session_state["page"] = "feed"
                st.rerun()
            return
        profile_user_id = target_user_id

    profile_name = db_user.get("username") or db_user.get("name") or "eco_citizen"
    profile_avatar = db_user.get("avatar") or "🌱"
    profile_email = db_user.get("email") or ""

    my_items = get_user_posts(profile_user_id) if profile_user_id else []

    total_fires = sum(int(i.get("likes", 0)) for i in my_items)
    average_gvi = (
        sum(float(i.get("green_index", 0.0)) for i in my_items) / len(my_items)
        if my_items
        else 0.0
    )

    # If it is not our own profile, show a back button
    if not is_own_profile:
        if st.button("←", key="back_btn"):
            st.session_state.pop("view_profile_id", None)
            st.session_state["page"] = "feed"
            st.rerun()

    # If it is our own profile, show the menu burger button at the top-right
    if is_own_profile:
        if st.button("☰", key="menu_btn"):
            _show_settings_dialog()

    st.markdown(
        f"""
        <div class="pf-ava-container">
            <div class="pf-ava-standalone">{html.escape(profile_avatar)}</div>
        </div>
        <div class="pf-name">@{html.escape(profile_name)}</div>
        """,
        unsafe_allow_html=True,
    )

    # Followers / Following / Likes counter (TikTok-style)
    followers_cnt = get_followers_count(profile_user_id) if profile_user_id else 0
    following_cnt = get_following_count(profile_user_id) if profile_user_id else 0
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; gap: 24px; margin: 0.9rem 0; font-family: var(--mono); font-size: 0.85rem; text-align: center;">
            <div><b style="color: var(--text); font-size: 1.05rem;">{following_cnt}</b><div style="color: var(--muted); font-size: 0.64rem; text-transform: uppercase; margin-top: 2px;">Подписок</div></div>
            <div><b style="color: var(--text); font-size: 1.05rem;">{followers_cnt}</b><div style="color: var(--muted); font-size: 0.64rem; text-transform: uppercase; margin-top: 2px;">Подписчиков</div></div>
            <div><b style="color: var(--text); font-size: 1.05rem;">{total_fires}</b><div style="color: var(--muted); font-size: 0.64rem; text-transform: uppercase; margin-top: 2px;">Лайков</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Follow button if it's someone else's profile
    if not is_own_profile and profile_user_id:
        if current_user_id:
            is_following_user = is_following(current_user_id, profile_user_id)
            if is_following_user:
                if st.button("Вы подписаны", key="follow_btn_followed", use_container_width=True):
                    toggle_follow(current_user_id, profile_user_id)
                    st.rerun()
            else:
                if st.button("Подписаться", key="follow_btn_unfollowed", use_container_width=True):
                    toggle_follow(current_user_id, profile_user_id)
                    st.rerun()
        else:
            if st.button("Подписаться", key="follow_btn_unfollowed", use_container_width=True):
                st.warning("Пожалуйста, войдите в аккаунт, чтобы подписываться.")

    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.markdown(
        f"""<div class="kpi-card"><div class="kpi-value">{len(my_items)}</div>
        <div class="kpi-label">Решений</div></div>""",
        unsafe_allow_html=True,
    )
    kpi2.markdown(
        f"""<div class="kpi-card"><div class="kpi-value">🔥 {total_fires}</div>
        <div class="kpi-label">Голосов</div></div>""",
        unsafe_allow_html=True,
    )
    kpi3.markdown(
        f"""<div class="kpi-card"><div class="kpi-value">{average_gvi:.2f}</div>
        <div class="kpi-label">Сред. GVI</div></div>""",
        unsafe_allow_html=True,
    )

    title_text = "✦ Мои решения" if is_own_profile else f"✦ Решения @{profile_name}"
    st.markdown(
        f'<div class="section-title">{title_text}</div>', unsafe_allow_html=True
    )

    if not my_items:
        no_posts_msg = (
            "Пока нет публикаций.<br/>Нажми ◉ внизу и предложи первое зелёное решение!"
            if is_own_profile
            else "Этот пользователь еще не опубликовал ни одного решения."
        )
        st.markdown(
            f"""
            <div class="eco-card" style="text-align:center;color:var(--muted);">
                {no_posts_msg}
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # 3-column grid of thumbnails with the fire count overlay.
    for row_start in range(0, len(my_items), 3):
        cols = st.columns(3)
        for col, item in zip(cols, my_items[row_start : row_start + 3]):
            with col:
                st.markdown(
                    f"""
                    <div class="pg-item">
                        <img src="{_thumb_src(item)}" alt=""/>
                        <div class="pg-likes">🔥 {int(item.get("likes", 0))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ===========================================================================
# Inbox page — mock official city updates and activity notifications
# ===========================================================================
def render_inbox() -> None:
    st.markdown(
        """
        <div class="page-head">
            <div class="kicker">// уведомления и вызовы среды</div>
            <h1>Входящие</h1>
            <p>Официальные ответы акимата, новые эко-челленджи и активность</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_all, tab_official, tab_activity = st.tabs(["⚡ Все", "🏛️ Официальные", "🔥 Активность"])

    notifications = [
        {
            "category": "official",
            "icon": "🏛️",
            "title": "Акимат г. Алматы",
            "text": "ИИ-проект озеленения по ул. Абая прошел модерацию и передан в Управление Экологии.",
            "time": "2 ч. назад",
            "badge": "В работе",
            "color": "var(--teal)",
        },
        {
            "category": "activity",
            "icon": "🔥",
            "title": "Лайк от @eco_almaty",
            "text": "Ваше решение по адресу пр. Достык, 12 получило отметку 'Огонь!' (+10 очков рейтинга).",
            "time": "5 ч. назад",
            "badge": "Рейтинг",
            "color": "var(--lime)",
        },
        {
            "category": "official",
            "icon": "🚨",
            "title": "Департамент экологии",
            "text": "Работы по ликвидации свалки по вашему обращению в Алатауском районе успешно завершены.",
            "time": "1 д. назад",
            "badge": "Выполнено",
            "color": "var(--lime)",
        },
        {
            "category": "activity",
            "icon": "💬",
            "title": "Комментарий от @green_rider",
            "text": "«Отличная идея с вертикальным озеленением! Вдоль оживленной трассы это просто необходимо.»",
            "time": "2 д. назад",
            "badge": "Обсуждение",
            "color": "var(--muted)",
        },
        {
            "category": "official",
            "icon": "⚡",
            "title": "Новый городской вызов",
            "text": "Запущен челлендж 'Зеленый Бостандыкский'. Создайте 3 эко-решения в районе и получите х2 баллов.",
            "time": "3 д. назад",
            "badge": "Челлендж",
            "color": "var(--amber)",
        },
        {
            "category": "activity",
            "icon": "🏆",
            "title": "Достижение получено!",
            "text": "Вы заработали значок 'Эко-Визионер' за первую генерацию зеленого будущего для улицы города.",
            "time": "5 д. назад",
            "badge": "Ачивка",
            "color": "var(--teal)",
        },
    ]

    def draw_list(items):
        if not items:
            st.markdown(
                """
                <div class="eco-card" style="text-align:center;color:var(--muted);padding:2rem;">
                    Здесь пока пусто. Активность появится, когда вы будете публиковать решения или участвовать в жизни города!
                </div>
                """,
                unsafe_allow_html=True,
            )
            return
        for item in items:
            st.markdown(
                f"""
                <div class="inbox-row">
                    <div class="inbox-icon">{item['icon']}</div>
                    <div class="inbox-content">
                        <div class="inbox-title-wrap">
                            <span class="inbox-title">{html.escape(item['title'])}</span>
                            <span class="inbox-time">{html.escape(item['time'])}</span>
                        </div>
                        <div class="inbox-text">{html.escape(item['text'])}</div>
                        <div class="inbox-badge" style="color: {item['color']}; border-color: {item['color']}2a; background-color: {item['color']}0f;">
                            {html.escape(item['badge'])}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_all:
        draw_list(notifications)

    with tab_official:
        draw_list([n for n in notifications if n["category"] == "official"])

    with tab_activity:
        draw_list([n for n in notifications if n["category"] == "activity"])


# ===========================================================================
# Floating glass dock — bottom nav
# ===========================================================================
def render_nav(active_page: str) -> None:
    with st.container(key="bottom_nav"):
        col_feed, col_top, col_camera, col_inbox, col_profile = st.columns([1, 1, 1, 1.1, 1])
        with col_feed:
            st.button("📺\nЛента", key="nav_feed", on_click=goto, args=("feed",), use_container_width=True)
        with col_top:
            st.button("🏆\nТоп", key="nav_top", on_click=goto, args=("top",), use_container_width=True)
        with col_camera:
            st.button("📷\nСканер", key="nav_camera", on_click=goto, args=("camera",), use_container_width=True)
        with col_inbox:
            st.button("⚡\nВходящие", key="nav_inbox", on_click=goto, args=("inbox",), use_container_width=True)
        with col_profile:
            st.button("👤\nПрофиль", key="nav_profile", on_click=goto, args=("profile",), use_container_width=True)

    if active_page == "camera":
        st.markdown(
            """<style>
            .st-key-bottom_nav .st-key-nav_camera button {
                background: rgba(168, 255, 96, 0.08) !important;
                color: var(--lime) !important;
                border: 1.5px solid var(--lime) !important;
                box-shadow: 0 0 12px rgba(168, 255, 96, 0.2) !important;
                text-shadow: 0 0 10px rgba(168, 255, 96, 0.4) !important;
                font-weight: 800 !important;
            }
            </style>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<style>.st-key-nav_{active_page} button{{
                color: #ffffff !important;
                text-shadow: 0 0 10px rgba(255,255,255,.5);
                font-weight: 800 !important;
            }}</style>""",
            unsafe_allow_html=True,
        )



# ===========================================================================
# App entry — auth check → page router
# ===========================================================================
init_db()           # one-shot pool creation + schema migration
_sync_auth_cookie() # flush queued cookie writes/clears (must precede restore)
_ensure_auth()      # restore session from cookie or show the login screen

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state["page"] = "feed"

flash_message = st.session_state.pop("flash", None)
if flash_message:
    st.toast(flash_message, icon="🌿")

current_page = st.session_state["page"]
if current_page == "camera":
    render_camera()
elif current_page == "top":
    render_top()
elif current_page == "profile":
    render_profile()
elif current_page == "inbox":
    render_inbox()
else:
    render_feed()

render_nav(current_page)
