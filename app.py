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
from utils.db import (
    init_db,
    load_posts,
    load_all_posts,
    save_post,
    toggle_like,
    get_user_liked_posts,
    get_user_posts,
    upsert_user,
    update_user_avatar,
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
# Auth helpers — Google OIDC via st.login() (Streamlit ≥ 1.42)
# ===========================================================================

def get_current_user() -> dict:
    """Return the current user's DB record stored in session_state."""
    return st.session_state.get("db_user") or {}


def get_current_user_id() -> str | None:
    """Return the current user's DB UUID string, or None."""
    uid = get_current_user().get("id")
    return str(uid) if uid else None


def _ensure_auth() -> None:
    """
    Check that the user is logged in. If not, show the login screen and stop.
    On first login (or each login) upsert the user into our DB and cache the
    record in st.session_state so we don't query the DB on every rerun.
    """
    if not st.user.is_logged_in:
        _render_login_screen()
        st.stop()

    if "db_user" not in st.session_state:
        db_user = upsert_user(
            google_sub=st.user.sub,
            email=getattr(st.user, "email", None),
            name=getattr(st.user, "name", None),
        )
        st.session_state["db_user"] = db_user


def _render_login_screen() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div style="
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            min-height: 85dvh; text-align: center; padding: 2rem 1.5rem;
            gap: 1rem;">
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
            <p style="color: var(--muted); font-size: .9rem; max-width: 320px;
                line-height: 1.6; margin-top: .5rem;">
                Фотографируй серые улицы — ИИ покажет зелёное будущее.<br/>
                Поддерживай лучшие места города голосами.
            </p>
            <div style="
                background: rgba(255,255,255,.04);
                border: 1px solid var(--line-soft);
                border-radius: 18px;
                padding: 1rem 1.2rem;
                font-size: .8rem; color: var(--muted);
                max-width: 320px; line-height: 1.55; margin-top: .3rem;">
                🔒 Аккаунт нужен, чтобы отслеживать твои публикации и голоса.
                Данные профиля не передаются третьим лицам.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        if st.button("🔑 Войти через Google", type="primary", use_container_width=True):
            st.login("google")


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
.pod-card { text-align: center; }
.pod-card.pod-2 { padding-top: 20px; }
.pod-card.pod-3 { padding-top: 30px; }
.pod-imgwrap {
    position: relative; border-radius: 20px; overflow: hidden;
    border: 1px solid var(--line-soft);
}
.pod-1 .pod-imgwrap {
    border-color: rgba(168,255,96,.55);
    box-shadow: 0 0 28px rgba(168,255,96,.20);
}
.pod-imgwrap img { width: 100%; aspect-ratio: 1; object-fit: cover; display: block; }
.pod-rank {
    position: absolute; top: 7px; left: 7px;
    width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono); font-weight: 700; font-size: .76rem;
    background: rgba(8,16,11,.72); backdrop-filter: blur(6px);
    border: 1px solid var(--line); color: var(--lime);
}
.pod-1 .pod-rank { background: var(--grad); color: var(--ink); border: none; }
.pod-addr { font-weight: 800; font-size: .7rem; margin-top: .5rem; line-height: 1.3; }
.pod-fire { font-family: var(--mono); color: var(--lime); font-size: .76rem; margin-top: .25rem; }
.pod-base {
    height: var(--h, 16px); margin-top: .55rem;
    border-radius: 10px 10px 4px 4px;
    background: linear-gradient(180deg, rgba(168,255,96,.22), rgba(168,255,96,.02));
    border: 1px solid var(--line);
}

.lb-row {
    display: flex; align-items: center; gap: 12px;
    background: var(--glass);
    backdrop-filter: blur(14px);
    border: 1px solid var(--line-soft);
    border-radius: 18px;
    padding: .6rem .75rem;
    margin-bottom: .5rem;
}
.lb-rank {
    font-family: var(--mono); font-weight: 700; font-size: .8rem;
    width: 22px; text-align: center; color: var(--lime); flex: none;
}
.lb-thumb { width: 52px; height: 52px; border-radius: 14px; object-fit: cover; flex: none; }
.lb-main { flex: 1; min-width: 0; }
.lb-addr { font-weight: 800; font-size: .84rem; line-height: 1.25; }
.lb-votebar {
    height: 4px; border-radius: 3px;
    background: rgba(255,255,255,.08);
    margin-top: .4rem; overflow: hidden;
}
.lb-votebar > span { display: block; height: 100%; border-radius: 3px; background: var(--grad); }
.lb-sub {
    font-family: var(--mono); color: var(--muted);
    font-size: .6rem; letter-spacing: .06em; margin-top: .3rem;
}
.lb-fire {
    margin-left: auto; font-family: var(--mono); font-weight: 700;
    color: var(--lime); white-space: nowrap; flex: none; font-size: .8rem;
}

/* ---- Profile -------------------------------------------------------------- */
.pf-banner {
    position: relative;
    height: 112px;
    border-radius: 24px;
    border: 1px solid var(--line);
    background:
        radial-gradient(180px 100px at 82% 0%, rgba(61,245,200,.30), transparent 70%),
        radial-gradient(240px 130px at 12% 100%, rgba(168,255,96,.24), transparent 70%),
        #0D1A12;
    margin-bottom: 52px;
}
.pf-banner::after {
    content: "ECO//RE";
    position: absolute; right: 16px; top: 14px;
    font-family: var(--display); font-weight: 900; font-size: 1.4rem;
    color: rgba(236,245,238,.07); letter-spacing: .02em;
}
.pf-ava {
    position: absolute; left: 50%; bottom: -38px; transform: translateX(-50%);
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
    font-size: .62rem;
    font-weight: 700;
    letter-spacing: .06em;
    text-transform: uppercase;
    padding: .55rem 0;
    border-radius: 14px;
    white-space: nowrap;
    overflow: hidden;
}
.st-key-bottom_nav .stButton > button:hover {
    transform: none; filter: none;
    color: var(--text);
    background: rgba(255,255,255,.05);
    border: none;
}
/* Raised gradient scan button */
.st-key-bottom_nav .st-key-nav_camera button {
    background: var(--grad) !important;
    color: var(--ink) !important;
    font-size: 1.35rem !important;
    width: 54px !important; height: 54px;
    margin: -24px auto 0;
    display: block;
    border-radius: 50% !important;
    padding: 0 !important;
    border: none !important;
    box-shadow: 0 10px 30px rgba(168,255,96,.35), 0 0 0 6px rgba(9,18,13,.9) !important;
}
.st-key-bottom_nav .st-key-nav_camera button:hover {
    transform: scale(1.06) !important;
    filter: brightness(1.06) !important;
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
/* Horizontal swipe: slide 1 = today, slide 2 = AI green future */
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
.tk-author { display: flex; align-items: center; gap: 9px; }
.tk-ava {
    width: 30px; height: 30px; border-radius: 50%; flex: none;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: .9rem;
    background: linear-gradient(#0D1A12, #0D1A12) padding-box, var(--grad) border-box;
    border: 1.5px solid transparent;
}
.tk-name { font-weight: 800; font-size: .88rem; }
.tk-time { font-family: var(--mono); color: rgba(236,245,238,.55); font-size: .62rem; letter-spacing: .06em; }
.tk-addr {
    margin-top: .55rem;
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

/* Right action rail: glass vote button */
div[class*="st-key-like_"] {
    position: absolute; right: 12px; bottom: 96px;
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
    font-size: .74rem; font-weight: 700;
    padding: 0; line-height: 1.15;
    box-shadow: 0 8px 24px rgba(0,0,0,.45);
}
div[class*="st-key-like_"] .stButton > button:hover {
    transform: scale(1.08);
    border-color: rgba(168,255,96,.6);
    box-shadow: 0 0 26px rgba(168,255,96,.3);
    background: rgba(168,255,96,.14);
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
    """Clear the camera-page state after a successful publish."""
    for key in (
        "analysis_result",
        "generated_image",
        "original_image",
        "ai_analysis",
        "appeal_text",
        "analyzed_signature",
    ):
        st.session_state.pop(key, None)
    # Bump the key so camera_input / file_uploader widgets reset themselves.
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1


def goto(page: str) -> None:
    st.session_state["page"] = page


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
    slides = (
        f'<div class="tk-slide"><img src="data:image/png;base64,{original_b64}" alt=""/>'
        f'<div class="tk-tag"><i class="rec"></i>сейчас</div>'
        f'<div class="tk-num">01/{total:02d}</div></div>'
    )
    if generated_b64:
        slides += (
            f'<div class="tk-slide"><img src="data:image/png;base64,{generated_b64}" alt=""/>'
            '<div class="tk-tag future">✦ прогноз ИИ</div>'
            f'<div class="tk-num">02/{total:02d}</div></div>'
        )
    hint = (
        '<div class="tk-hint">⇆ свайп — прогноз ИИ</div>' if generated_b64 else ""
    )

    return (
        f'<div class="tk-card"><div class="tk-swipe">{slides}</div>'
        '<div class="tk-grad top"></div><div class="tk-grad bottom"></div>'
        '<div class="tk-frame"></div>'
        '<div class="tk-info">'
        f'<div class="tk-author"><span class="tk-ava">{avatar}</span>'
        f'<span class="tk-name">@{author}</span>'
        f'<span class="tk-time">· {when}</span></div>'
        f'<div class="tk-addr">📍 {address}</div>'
        f'<div class="tk-meter m-{level}">'
        f'<div class="tk-meter-head"><span>индекс зелени · {label}</span>'
        f"<b>{green_index:.2f}</b></div>"
        f'<div class="tk-meter-bar"><span style="width:{green_index * 100:.0f}%"></span></div>'
        f"</div>{summary_html}{hint}"
        "</div></div>"
    )


def render_feed() -> None:
    st.markdown(FEED_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="tk-top">
            <span class="tk-logo">ECO//RE</span>
            <span class="tk-live"><i></i>город live</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    user_id = get_current_user_id()
    feed_items = load_posts(limit=20)

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
                like_count = int(item.get("likes", 0))
                is_liked = item_id in liked_post_ids
                btn_label = f"💚 {like_count}" if is_liked else f"🔥 {like_count}"
                if st.button(btn_label, key=f"like_{item_id}"):
                    if user_id:
                        now_liked, new_count = toggle_like(user_id, item_id)
                        if now_liked:
                            liked_post_ids.add(item_id)
                        else:
                            liked_post_ids.discard(item_id)
                        st.rerun()


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


def render_camera() -> None:
    db_user = get_current_user()
    st.markdown(
        """
        <div class="page-head">
            <div class="kicker">// новое решение</div>
            <h1>Сканер улицы</h1>
            <p>Сфоткай серую улицу — ИИ покажет её зелёное будущее</p>
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

    active_step = 1
    if photo_file is not None:
        active_step = (
            3 if st.session_state.get("location_input", "").strip() else 2
        )
    render_steps(active_step)

    if photo_file is None:
        st.markdown(
            """
            <div class="eco-card" style="text-align:center;color:var(--muted);">
                ⬆️ Сделай фото на месте или выбери из галереи —
                и начнём эко-аудит локации
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    original_image = Image.open(photo_file).convert("RGB")

    # Run the pipelines once per photo, cache in session.
    file_signature = f"{getattr(photo_file, 'name', 'camera')}_{photo_file.size}"
    if st.session_state.get("analyzed_signature") != file_signature:
        # 1) Real image analysis first: YOLOv8 + OpenCV eco-audit.
        with st.spinner("🛰 YOLOv8 анализирует улицу..."):
            eco_audit = eco_audit_safe(original_image)
            green_index = eco_audit["green_view_index"] / 100.0
            masked_image, _ = analyze_eco_status(original_image)
        # 2) Image-to-image, with a prompt built from the YOLO audit.
        with st.spinner("🎨 ИИ рисует зелёное будущее по результатам анализа..."):
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
        with st.spinner("🤖 ИИ формулирует выводы по данным анализа..."):
            ai_analysis = generate_findings(
                eco_audit,
                location_address.strip() or "адрес не указан",
            )
        if ai_analysis is None:
            st.info("Добавьте HF_API_TOKEN в secrets.toml для ИИ-анализа")
        else:
            st.session_state["ai_analysis"] = ai_analysis
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

    all_items = load_all_posts()
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
            base_heights = {1: 44, 2: 28, 3: 16}
            cols = st.columns([1, 1.18, 1])
            for col, (item, rank) in zip(cols, podium_order):
                with col:
                    st.markdown(
                        f"""
                        <div class="pod-card pod-{rank}">
                            <div class="pod-imgwrap">
                                <img src="{_thumb_src(item)}" alt=""/>
                                <div class="pod-rank">{rank}</div>
                            </div>
                            <div class="pod-addr">📍 {html.escape(item.get("address", "Без адреса"))}</div>
                            <div class="pod-fire">🔥 {int(item.get("likes", 0))}</div>
                            <div class="pod-base" style="--h:{base_heights[rank]}px"></div>
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
            st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
        rows_html = ""
        for offset, item in enumerate(rest):
            votes = int(item.get("likes", 0))
            author = html.escape(item.get("author") or "eco_citizen")
            rows_html += (
                f'<div class="lb-row"><div class="lb-rank">{start_rank + offset:02d}</div>'
                f'<img class="lb-thumb" src="{_thumb_src(item)}" alt=""/>'
                f'<div class="lb-main"><div class="lb-addr">📍 {html.escape(item.get("address", "Без адреса"))}</div>'
                f'<div class="lb-votebar"><span style="width:{votes / max_votes * 100:.0f}%"></span></div>'
                f'<div class="lb-sub">@{author} · gvi {float(item.get("green_index", 0.0)):.2f}</div></div>'
                f'<div class="lb-fire">🔥 {votes}</div></div>'
            )
        if rows_html:
            st.markdown(rows_html, unsafe_allow_html=True)

    # --- Akimat decision panel (kept from the old dashboard tab) -----------
    if all_items and st.toggle("🏛 Режим акимата — аналитика для города"):
        df = pd.DataFrame(
            [
                {
                    "Локация": item.get("address", "Без адреса"),
                    "Green Index": float(item.get("green_index", 0.0)),
                    "Голоса": int(item.get("likes", 0)),
                    "Дата": format_timestamp(item.get("timestamp", "")),
                }
                for item in all_items
            ]
        )

        total_submissions = len(df)
        average_green_index = df["Green Index"].mean()
        top_location = df.groupby("Локация")["Голоса"].sum().idxmax()

        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        kpi_col1.markdown(
            f"""<div class="kpi-card"><div class="kpi-value">{total_submissions}</div>
            <div class="kpi-label">Заявок</div></div>""",
            unsafe_allow_html=True,
        )
        kpi_col2.markdown(
            f"""<div class="kpi-card"><div class="kpi-value">{average_green_index:.2f}</div>
            <div class="kpi-label">Средний GVI</div></div>""",
            unsafe_allow_html=True,
        )
        kpi_col3.markdown(
            f"""<div class="kpi-card"><div class="kpi-value" style="font-size:.72rem;">{html.escape(top_location)}</div>
            <div class="kpi-label">Топ запрос</div></div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">📊 Экологическая срочность × Запрос горожан</div>',
            unsafe_allow_html=True,
        )
        st.caption("Левый верхний угол — критичные зоны: мало зелени, много голосов.")

        scatter_fig = px.scatter(
            df,
            x="Green Index",
            y="Голоса",
            size=df["Голоса"].clip(lower=1),
            color="Green Index",
            color_continuous_scale=["#FF7A5C", "#FFC24B", "#74C69D", "#A8FF60"],
            hover_name="Локация",
            size_max=42,
        )
        scatter_fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Manrope, sans-serif", color="#ECF5EE"),
            coloraxis_colorbar=dict(title="GVI"),
            xaxis=dict(title="Green View Index (меньше = хуже)", gridcolor="#1C2A21"),
            yaxis=dict(title="Голоса горожан", gridcolor="#1C2A21"),
        )
        st.plotly_chart(scatter_fig, width="stretch", config={"displayModeBar": False})

        st.markdown(
            '<div class="section-title">🚨 Топ-5 зон для бюджета благоустройства</div>',
            unsafe_allow_html=True,
        )
        st.caption("Приоритет = голоса × (1 − Green Index): высокий спрос при дефиците зелени.")

        critical_df = df.copy()
        critical_df["Приоритет"] = (
            critical_df["Голоса"] * (1.0 - critical_df["Green Index"])
        ).round(2)
        top_critical = (
            critical_df.sort_values("Приоритет", ascending=False)
            .head(5)
            .reset_index(drop=True)
        )
        top_critical.index += 1

        st.dataframe(
            top_critical[["Локация", "Green Index", "Голоса", "Приоритет", "Дата"]],
            width="stretch",
            column_config={
                "Green Index": st.column_config.ProgressColumn(
                    "Green Index", min_value=0.0, max_value=1.0, format="%.2f"
                ),
            },
        )

        if st.button("🤖 Сгенерировать отчёт для акимата"):
            if not free_ai_available():
                st.info("Добавьте HF_API_TOKEN в secrets.toml для ИИ-анализа")
            else:
                top_zones_list = top_critical[
                    ["Локация", "Green Index", "Голоса", "Приоритет"]
                ].to_dict("records")
                with st.spinner("🤖 ИИ готовит аналитический отчёт..."):
                    dashboard_report = generate_dashboard_report(top_zones_list)
                if dashboard_report:
                    st.session_state["dashboard_report"] = dashboard_report
                else:
                    st.info("Добавьте HF_API_TOKEN в secrets.toml для ИИ-анализа")

        if st.session_state.get("dashboard_report"):
            with st.expander("📋 Аналитический отчёт", expanded=True):
                st.markdown(st.session_state["dashboard_report"])


# ===========================================================================
# Profile page — aurora banner, gradient-ring avatar, solutions grid
# ===========================================================================
def render_profile() -> None:
    db_user = get_current_user()
    user_id = get_current_user_id()

    profile_name = db_user.get("name") or "eco_citizen"
    profile_avatar = db_user.get("avatar") or "🌱"
    profile_email = db_user.get("email") or ""

    my_items = get_user_posts(user_id) if user_id else []

    total_fires = sum(int(i.get("likes", 0)) for i in my_items)
    average_gvi = (
        sum(float(i.get("green_index", 0.0)) for i in my_items) / len(my_items)
        if my_items
        else 0.0
    )

    st.markdown(
        f"""
        <div class="pf-banner">
            <div class="pf-ava">{html.escape(profile_avatar)}</div>
        </div>
        <div class="pf-name">@{html.escape(profile_name)}</div>
        <div class="pf-sub">эко-активист // город-сканер</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
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

    with st.expander("⚙️ Настройки профиля"):
        with st.form("profile_form", border=False):
            current_idx = AVATARS.index(profile_avatar) if profile_avatar in AVATARS else 0
            new_avatar = st.selectbox("Аватар", AVATARS, index=current_idx)
            if profile_email:
                st.caption(f"📧 {profile_email}")
            if st.form_submit_button("💾 Сохранить аватар"):
                if user_id:
                    update_user_avatar(user_id, new_avatar)
                    st.session_state["db_user"]["avatar"] = new_avatar
                    st.session_state["flash"] = "Аватар обновлён ✅"
                    st.rerun()

    if st.button("🚪 Выйти из аккаунта", use_container_width=True):
        st.logout()

    st.markdown(
        '<div class="section-title">✦ Мои решения</div>', unsafe_allow_html=True
    )

    if not my_items:
        st.markdown(
            """
            <div class="eco-card" style="text-align:center;color:var(--muted);">
                Пока нет публикаций.<br/>Нажми ◉ внизу и предложи первое
                зелёное решение для своего города!
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
# Floating glass dock — bottom nav
# ===========================================================================
def render_nav(active_page: str) -> None:
    with st.container(key="bottom_nav"):
        col_feed, col_camera, col_top, col_profile = st.columns([2, 0.95, 0.92, 1.13])
        with col_feed:
            st.button("Лента", key="nav_feed", on_click=goto, args=("feed",), width="stretch")
        with col_camera:
            st.button("◉", key="nav_camera", on_click=goto, args=("camera",))
        with col_top:
            st.button("Топ", key="nav_top", on_click=goto, args=("top",), width="stretch")
        with col_profile:
            st.button("Профиль", key="nav_profile", on_click=goto, args=("profile",), width="stretch")

    if active_page != "camera":
        st.markdown(
            f"""<style>.st-key-nav_{active_page} button{{
                color: var(--lime) !important;
                text-shadow: 0 0 14px rgba(168,255,96,.55);
            }}</style>""",
            unsafe_allow_html=True,
        )


# ===========================================================================
# App entry — auth check → page router
# ===========================================================================
init_db()          # one-shot pool creation + schema migration
_ensure_auth()     # redirect to login screen if not authenticated

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
else:
    render_feed()

render_nav(current_page)
