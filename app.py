"""
EcoReconstruct AI — crowdsourced urban ecology platform (mobile-first MVP).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import base64
import io
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from utils.claude_ai import (
    analyze_photo_with_claude,
    claude_available,
    generate_appeal_text,
    generate_dashboard_report,
)
from utils.db import add_like, load_db, save_item
from utils.models import analyze_eco_status, generate_eco_friendly_view

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
# Eco-Tech design system — custom CSS injection (mobile-first)
# ===========================================================================
ECO_TECH_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --eco-forest: #1B4332;
    --eco-pine: #2D6A4F;
    --eco-leaf: #40916C;
    --eco-mint: #74C69D;
    --eco-mist: #D8F3DC;
    --eco-white: #FBFDFC;
    --eco-gray: #E9EDEB;
    --eco-text: #14281F;
    --eco-muted: #5C7268;
    --eco-danger: #E76F51;
    --eco-radius: 18px;
}

/* ---- Base canvas -------------------------------------------------- */
html, body, [class*="css"], .stApp {
    font-family: 'Manrope', -apple-system, 'Segoe UI', sans-serif !important;
    color: var(--eco-text);
}
.stApp {
    background:
        radial-gradient(1200px 500px at 80% -10%, rgba(116,198,157,.18), transparent 60%),
        radial-gradient(900px 420px at -10% 30%, rgba(216,243,220,.45), transparent 55%),
        var(--eco-white);
}

/* ---- Hide Streamlit chrome ---------------------------------------- */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
div[data-testid="stToolbar"], div[data-testid="stDecoration"] { display: none; }

/* ---- Mobile-friendly content width -------------------------------- */
.block-container {
    max-width: 640px;
    padding: 1.1rem 1rem 4rem 1rem !important;
}

/* ---- Tabs as a segmented app navbar -------------------------------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(255,255,255,.65);
    backdrop-filter: blur(12px);
    border: 1px solid var(--eco-gray);
    border-radius: 16px;
    padding: 6px;
    position: sticky;
    top: 0;
    z-index: 99;
}
.stTabs [data-baseweb="tab"] {
    flex: 1;
    justify-content: center;
    border-radius: 12px;
    padding: 10px 6px;
    font-weight: 700;
    font-size: .82rem;
    color: var(--eco-muted);
    background: transparent;
    white-space: nowrap;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--eco-pine), var(--eco-leaf)) !important;
    color: #fff !important;
    box-shadow: 0 6px 16px rgba(45,106,79,.30);
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ---- Buttons: modern app controls ---------------------------------- */
.stButton > button, .stFormSubmitButton > button {
    width: 100%;
    border: none;
    border-radius: 14px;
    padding: .85rem 1.2rem;
    font-weight: 800;
    font-size: 1rem;
    letter-spacing: .01em;
    background: linear-gradient(135deg, var(--eco-forest), var(--eco-leaf));
    color: #fff;
    box-shadow: 0 8px 20px rgba(27,67,50,.25);
    transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    transform: translateY(-2px);
    filter: brightness(1.08);
    box-shadow: 0 12px 26px rgba(27,67,50,.32);
    color: #fff;
}
.stButton > button:active { transform: translateY(0); }

/* ---- Inputs --------------------------------------------------------- */
.stTextInput input {
    border-radius: 14px !important;
    border: 1.5px solid var(--eco-gray) !important;
    padding: .8rem 1rem !important;
    background: rgba(255,255,255,.85) !important;
    font-size: 1rem !important;
}
.stTextInput input:focus {
    border-color: var(--eco-mint) !important;
    box-shadow: 0 0 0 3px rgba(116,198,157,.25) !important;
}
div[data-testid="stFileUploader"] section {
    border: 1.5px dashed var(--eco-mint);
    border-radius: var(--eco-radius);
    background: rgba(216,243,220,.30);
}

/* ---- Images & misc --------------------------------------------------- */
div[data-testid="stImage"] img { border-radius: 14px; }
hr { border-color: var(--eco-gray); }

/* ---- Custom card components ------------------------------------------ */
.eco-hero {
    text-align: center;
    padding: .4rem 0 1rem 0;
}
.eco-hero h1 {
    font-size: 1.55rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(135deg, var(--eco-forest), var(--eco-leaf));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.eco-hero p { color: var(--eco-muted); font-size: .88rem; margin: .25rem 0 0; }

.eco-card {
    background: rgba(255,255,255,.78);
    backdrop-filter: blur(10px);
    border: 1px solid var(--eco-gray);
    border-radius: var(--eco-radius);
    padding: 1.1rem 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 10px 30px rgba(27,67,50,.07);
}

.gvi-card {
    background: linear-gradient(135deg, var(--eco-forest) 0%, var(--eco-pine) 60%, var(--eco-leaf) 100%);
    border-radius: var(--eco-radius);
    padding: 1.3rem 1.4rem;
    color: #fff;
    box-shadow: 0 14px 34px rgba(27,67,50,.35);
    margin: .6rem 0 1rem 0;
}
.gvi-card .gvi-label {
    font-size: .75rem; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; opacity: .85;
}
.gvi-card .gvi-value { font-size: 2.6rem; font-weight: 800; line-height: 1.1; }
.gvi-card .gvi-note { font-size: .85rem; opacity: .9; margin-top: .2rem; }
.gvi-bar {
    height: 8px; border-radius: 6px;
    background: rgba(255,255,255,.25);
    margin-top: .8rem; overflow: hidden;
}
.gvi-bar > span {
    display: block; height: 100%;
    background: linear-gradient(90deg, #B7E4C7, #D8F3DC);
    border-radius: 6px;
}

.feed-meta { display: flex; justify-content: space-between; align-items: baseline; }
.feed-loc { font-weight: 800; font-size: 1.02rem; }
.feed-time { color: var(--eco-muted); font-size: .75rem; }
.feed-badge {
    display: inline-block;
    padding: .25rem .7rem;
    border-radius: 999px;
    font-size: .75rem;
    font-weight: 800;
    margin: .45rem 0 .3rem 0;
}
.badge-low  { background: #FDE8E2; color: var(--eco-danger); }
.badge-mid  { background: #FFF3D6; color: #B8860B; }
.badge-high { background: var(--eco-mist); color: var(--eco-pine); }

.kpi-card {
    background: rgba(255,255,255,.80);
    backdrop-filter: blur(10px);
    border: 1px solid var(--eco-gray);
    border-radius: var(--eco-radius);
    padding: .95rem .6rem;
    text-align: center;
    box-shadow: 0 8px 22px rgba(27,67,50,.06);
    height: 100%;
}
.kpi-card .kpi-value {
    font-size: 1.45rem; font-weight: 800; color: var(--eco-forest);
    line-height: 1.15; word-break: break-word;
}
.kpi-card .kpi-label {
    font-size: .68rem; font-weight: 700; letter-spacing: .06em;
    text-transform: uppercase; color: var(--eco-muted); margin-top: .2rem;
}

.section-title {
    font-weight: 800; font-size: 1.05rem;
    margin: .8rem 0 .4rem 0; color: var(--eco-forest);
}
</style>
"""
st.markdown(ECO_TECH_CSS, unsafe_allow_html=True)


# ===========================================================================
# Helpers
# ===========================================================================
def pil_to_base64(image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string for JSON storage."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_pil(encoded: str) -> Image.Image | None:
    """Decode a base64 PNG string back into a PIL image (None on failure)."""
    try:
        return Image.open(io.BytesIO(base64.b64decode(encoded)))
    except Exception:
        return None


def format_timestamp(iso_string: str) -> str:
    """Human-friendly date for feed cards."""
    try:
        return datetime.fromisoformat(iso_string).strftime("%d.%m.%Y %H:%M")
    except (ValueError, TypeError):
        return ""


def gvi_badge(green_index: float) -> str:
    """Return an HTML badge classifying the Green View Index."""
    if green_index < 0.25:
        return f'<span class="feed-badge badge-low">🟥 Критично мало зелени — GVI {green_index:.2f}</span>'
    if green_index < 0.5:
        return f'<span class="feed-badge badge-mid">🟨 Средний уровень — GVI {green_index:.2f}</span>'
    return f'<span class="feed-badge badge-high">🟩 Зелёная зона — GVI {green_index:.2f}</span>'


def reset_idea_form() -> None:
    """Clear Tab A state after a successful publish."""
    for key in (
        "analysis_result",
        "generated_image",
        "original_image",
        "ai_analysis",
        "appeal_text",
    ):
        st.session_state.pop(key, None)
    # Bump the uploader key so the file_uploader widget resets itself.
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1


# ===========================================================================
# Hero header
# ===========================================================================
st.markdown(
    """
    <div class="eco-hero">
        <h1>🌿 EcoReconstruct AI</h1>
        <p>Сфотографируй улицу — ИИ покажет её зелёное будущее</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_new_idea, tab_pulse, tab_dashboard = st.tabs(
    ["🌱 Новая идея", "🔥 Пульс Города", "🏛 Акимат Дашборд"]
)

# ===========================================================================
# Tab A — "🌱 Новая идея" (Eco-Audit & GenAI Renovation)
# ===========================================================================
with tab_new_idea:
    st.markdown(
        '<div class="section-title">📸 Загрузите фото улицы или двора</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Фото локации",
        type=["png", "jpg", "jpeg", "webp"],
        key=f"uploader_{st.session_state.get('uploader_key', 0)}",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert("RGB")

        # Run the (mock) pipelines once per uploaded file, cache in session.
        file_signature = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("analyzed_signature") != file_signature:
            with st.spinner("🛰 SegFormer анализирует озеленение..."):
                masked_image, green_index = analyze_eco_status(original_image)
            with st.spinner("🎨 Stable Diffusion рисует зелёное будущее..."):
                generated_image = generate_eco_friendly_view(
                    original_image, green_index
                )
            st.session_state["analyzed_signature"] = file_signature
            st.session_state["original_image"] = original_image
            st.session_state["analysis_result"] = (masked_image, green_index)
            st.session_state["generated_image"] = generated_image

        masked_image, green_index = st.session_state["analysis_result"]
        generated_image = st.session_state["generated_image"]
        original_image = st.session_state["original_image"]

        # --- Green View Index metric card ------------------------------
        verdict = (
            "Району срочно нужно озеленение 🌵"
            if green_index < 0.25
            else "Есть зелень, но можно лучше 🌿"
            if green_index < 0.5
            else "Отличный зелёный район! 🌳"
        )
        st.markdown(
            f"""
            <div class="gvi-card">
                <div class="gvi-label">Green View Index</div>
                <div class="gvi-value">{green_index:.2f} <span style="font-size:1.1rem;opacity:.8;">/ 1.00</span></div>
                <div class="gvi-note">{verdict}</div>
                <div class="gvi-bar"><span style="width:{green_index * 100:.0f}%"></span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Before / analysis / future concept -------------------------
        col_original, col_masked = st.columns(2)
        with col_original:
            st.image(original_image, caption="📷 Оригинал", width="stretch")
        with col_masked:
            st.image(
                masked_image,
                caption="🛰 Анализ озеленения",
                width="stretch",
            )

        st.markdown(
            '<div class="section-title">✨ Будущий зелёный концепт</div>',
            unsafe_allow_html=True,
        )
        st.image(
            generated_image,
            caption="🎨 Future Green Concept (GenAI)",
            width="stretch",
        )

        # --- Location input + publish -----------------------------------
        location_address = st.text_input(
            "📍 Адрес / Локация",
            placeholder="например: Алматы, ул. Абая 44",
            key="location_input",
        )

        # --- Claude AI analysis section ----------------------------------
        st.markdown(
            '<div class="section-title">🤖 ИИ-анализ локации</div>',
            unsafe_allow_html=True,
        )

        ai_analysis = None
        if not claude_available():
            st.info("Добавьте GOOGLE_API_KEY в secrets.toml для ИИ-анализа")
        else:
            with st.spinner("🤖 Claude изучает фотографию..."):
                ai_analysis = analyze_photo_with_claude(
                    original_image,
                    green_index,
                    location_address.strip() or "адрес не указан",
                )
            if ai_analysis is None:
                st.info("Добавьте GOOGLE_API_KEY в secrets.toml для ИИ-анализа")
            else:
                st.session_state["ai_analysis"] = ai_analysis
                priority = ai_analysis.get("priority", "средний").lower()
                badge_class = {
                    "высокий": "badge-low",
                    "средний": "badge-mid",
                    "низкий": "badge-high",
                }.get(priority, "badge-mid")
                problems_html = "".join(
                    f"<div>⚠️ {p}</div>" for p in ai_analysis["problems"]
                )
                recommendations_html = "".join(
                    f"<div>🌱 {r}</div>" for r in ai_analysis["recommendations"]
                )
                st.markdown(
                    f"""
                    <div class="eco-card">
                        <span class="feed-badge {badge_class}">Приоритет: {priority}</span>
                        <p style="margin:.4rem 0 .6rem 0;"><em>{ai_analysis["summary"]}</em></p>
                        <div class="section-title" style="font-size:.92rem;">Проблемы</div>
                        {problems_html}
                        <div class="section-title" style="font-size:.92rem;">Рекомендации</div>
                        {recommendations_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button("📄 Сгенерировать обращение в акимат"):
                    with st.spinner("📄 Claude составляет официальное обращение..."):
                        appeal_text = generate_appeal_text(
                            location_address.strip() or "адрес не указан",
                            green_index,
                            ai_analysis["problems"],
                            ai_analysis["recommendations"],
                        )
                    if appeal_text:
                        st.session_state["appeal_text"] = appeal_text
                    else:
                        st.info(
                            "Добавьте GOOGLE_API_KEY в secrets.toml для ИИ-анализа"
                        )

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

        if st.button("📢 Опубликовать в Пульс Города", type="primary"):
            if not location_address.strip():
                st.warning("Пожалуйста, укажите адрес или название локации 📍")
            else:
                saved_analysis = st.session_state.get("ai_analysis") or {}
                save_item(
                    {
                        "address": location_address.strip(),
                        "green_index": round(green_index, 3),
                        "image_original": pil_to_base64(original_image),
                        "image_generated": pil_to_base64(generated_image),
                        "ai_problems": saved_analysis.get("problems", []),
                        "ai_recommendations": saved_analysis.get(
                            "recommendations", []
                        ),
                        "ai_priority": saved_analysis.get("priority", ""),
                    }
                )
                reset_idea_form()
                st.session_state.pop("analyzed_signature", None)
                st.toast("Идея опубликована в Пульс Города! 🎉", icon="🌱")
                st.rerun()
    else:
        st.info("⬆️ Сделайте фото или выберите изображение из галереи, чтобы начать эко-аудит.")

# ===========================================================================
# Tab B — "🔥 Пульс Города" (Crowdsourced Feed)
# ===========================================================================
with tab_pulse:
    feed_items = load_db()

    if not feed_items:
        st.info("Пока нет идей. Станьте первым — опубликуйте свою во вкладке «🌱 Новая идея»!")
    else:
        st.markdown(
            f'<div class="section-title">Идеи горожан · {len(feed_items)}</div>',
            unsafe_allow_html=True,
        )

        for item in feed_items:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="feed-meta">
                        <span class="feed-loc">📍 {item.get("address", "Без адреса")}</span>
                        <span class="feed-time">{format_timestamp(item.get("timestamp", ""))}</span>
                    </div>
                    {gvi_badge(float(item.get("green_index", 0.0)))}
                    """,
                    unsafe_allow_html=True,
                )

                # "Было / Стало" switcher — thumb-friendly segmented control.
                view_choice = st.radio(
                    "Вид",
                    options=["🏚 Было", "🌳 Стало"],
                    horizontal=True,
                    key=f"view_{item['id']}",
                    label_visibility="collapsed",
                )
                shown_b64 = (
                    item.get("image_original")
                    if view_choice == "🏚 Было"
                    else item.get("image_generated")
                )
                shown_image = base64_to_pil(shown_b64) if shown_b64 else None
                if shown_image is not None:
                    st.image(shown_image, width="stretch")
                else:
                    st.caption("Изображение недоступно")

                # Large tap-friendly like button.
                like_count = int(item.get("likes", 0))
                if st.button(
                    f"🔥 Хочу такое благоустройство! ({like_count})",
                    key=f"like_{item['id']}",
                ):
                    add_like(item["id"])
                    st.rerun()

# ===========================================================================
# Tab C — "🏛 Акимат Дашборд" (B2G Decision Panel)
# ===========================================================================
with tab_dashboard:
    dashboard_items = load_db()

    if not dashboard_items:
        st.info("Данных пока нет — дашборд оживёт после первых публикаций горожан.")
    else:
        df = pd.DataFrame(
            [
                {
                    "Локация": item.get("address", "Без адреса"),
                    "Green Index": float(item.get("green_index", 0.0)),
                    "Голоса": int(item.get("likes", 0)),
                    "Дата": format_timestamp(item.get("timestamp", "")),
                }
                for item in dashboard_items
            ]
        )

        # --- KPI cards ---------------------------------------------------
        total_submissions = len(df)
        average_green_index = df["Green Index"].mean()
        top_location = (
            df.groupby("Локация")["Голоса"].sum().idxmax()
            if total_submissions
            else "—"
        )

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
            f"""<div class="kpi-card"><div class="kpi-value" style="font-size:.92rem;">{top_location}</div>
            <div class="kpi-label">Топ запрос</div></div>""",
            unsafe_allow_html=True,
        )

        # --- Urgency vs demand chart --------------------------------------
        st.markdown(
            '<div class="section-title">📊 Экологическая срочность × Запрос горожан</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Левый верхний угол — критичные зоны: мало зелени, много голосов."
        )

        scatter_fig = px.scatter(
            df,
            x="Green Index",
            y="Голоса",
            size=df["Голоса"].clip(lower=1),
            color="Green Index",
            color_continuous_scale=["#E76F51", "#F4A261", "#74C69D", "#2D6A4F"],
            hover_name="Локация",
            size_max=42,
        )
        scatter_fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Manrope, sans-serif", color="#14281F"),
            coloraxis_colorbar=dict(title="GVI"),
            xaxis=dict(title="Green View Index (меньше = хуже)", gridcolor="#E9EDEB"),
            yaxis=dict(title="Голоса горожан", gridcolor="#E9EDEB"),
        )
        st.plotly_chart(scatter_fig, width="stretch", config={"displayModeBar": False})

        # --- Top-5 critical zones table ------------------------------------
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
        top_critical.index += 1  # rank from 1

        st.dataframe(
            top_critical[["Локация", "Green Index", "Голоса", "Приоритет", "Дата"]],
            width="stretch",
            column_config={
                "Green Index": st.column_config.ProgressColumn(
                    "Green Index", min_value=0.0, max_value=1.0, format="%.2f"
                ),
            },
        )

        # --- Claude AI analytic report ------------------------------------
        if st.button("🤖 Сгенерировать отчёт для акимата"):
            if not claude_available():
                st.info("Добавьте GOOGLE_API_KEY в secrets.toml для ИИ-анализа")
            else:
                top_zones_list = top_critical[
                    ["Локация", "Green Index", "Голоса", "Приоритет"]
                ].to_dict("records")
                with st.spinner("🤖 Claude готовит аналитический отчёт..."):
                    dashboard_report = generate_dashboard_report(top_zones_list)
                if dashboard_report:
                    st.session_state["dashboard_report"] = dashboard_report
                else:
                    st.info(
                        "Добавьте GOOGLE_API_KEY в secrets.toml для ИИ-анализа"
                    )

        if st.session_state.get("dashboard_report"):
            with st.expander("📋 Аналитический отчёт", expanded=True):
                st.markdown(st.session_state["dashboard_report"])
