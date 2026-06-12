"""
EcoReconstruct AI — Gemini API integration.

Three AI functions powered by the `google-generativeai` SDK (gemini-1.5-flash):

- analyze_photo_with_claude()   : Gemini Vision eco-audit of a street photo
- generate_appeal_text()        : official appeal letter to the akimat (KZ + RU)
- generate_dashboard_report()   : analytic report over the top-5 critical zones

Design notes:
- The API key is read from st.secrets["GOOGLE_API_KEY"] with a fallback
  to the GOOGLE_API_KEY environment variable.
- Every public function degrades gracefully: if the key is missing or the
  API call fails, it returns None instead of raising, so the Streamlit app
  never crashes — the UI shows an st.info() hint instead.
- Heavy calls are wrapped in @st.cache_data(ttl=3600) keyed on the request
  payload (the image is passed as a base64 string so Streamlit can hash it).
"""

from __future__ import annotations

import base64
import io
import json
import os
import re

import streamlit as st
from PIL import Image

MODEL = "gemini-1.5-flash"

ANALYZE_SYSTEM_PROMPT = (
    "Ты эксперт по городскому экологическому планированию. Анализируй фотографии \n"
    "улиц и дворов казахстанских городов. Отвечай строго в JSON формате без \n"
    "markdown-обёртки. Язык ответа: русский."
)

ANALYZE_USER_PROMPT = """Проанализируй эту фотографию улицы/двора по адресу: {address}.
Green View Index (доля зелёных пикселей): {green_index:.2f} из 1.00.
Верни JSON строго такого формата:
{{
  "problems": ["проблема 1", "проблема 2", ...],
  "recommendations": ["рекомендация 1", "рекомендация 2", ...],
  "priority": "высокий|средний|низкий",
  "summary": "одно предложение-вывод"
}}
problems — конкретные экологические/урбанистические проблемы видимые на фото.
recommendations — конкретные действия: что посадить, что установить, что убрать.
priority — исходя из GVI и видимых проблем.
"""


# ---------------------------------------------------------------------------
# Client / availability helpers
# ---------------------------------------------------------------------------
def _get_api_key() -> str | None:
    key = None
    try:
        key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        key = None
    if not key:
        key = os.environ.get("GOOGLE_API_KEY")
    return key or None


def claude_available() -> bool:
    """True when both the SDK and an API key are present."""
    if _get_api_key() is None:
        return False
    try:
        import google.generativeai  # noqa: F401
    except ImportError:
        return False
    return True


def _get_model():
    import google.generativeai as genai

    genai.configure(api_key=_get_api_key())
    return genai.GenerativeModel(MODEL)


def _image_to_base64_jpeg(image: Image.Image, max_side: int = 1280) -> str:
    """Downscale and JPEG-encode the photo to keep the request small."""
    img = image.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _extract_json(text: str) -> dict | None:
    """Parse a JSON object from the model output, tolerating code fences."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


# ---------------------------------------------------------------------------
# 1. Photo analysis (Gemini Vision)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _analyze_cached(image_b64: str, green_index: float, address: str) -> dict | None:
    model = _get_model()
    image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
    prompt = (
        ANALYZE_SYSTEM_PROMPT
        + "\n\n"
        + ANALYZE_USER_PROMPT.format(address=address, green_index=green_index)
    )
    response = model.generate_content([prompt, image])
    return _extract_json(response.text)


def analyze_photo_with_claude(
    image: Image.Image, green_index: float, address: str
) -> dict | None:
    """
    Eco-audit of a street photo via Gemini Vision.

    Returns a dict with keys: problems (list[str]), recommendations (list[str]),
    priority ("высокий"/"средний"/"низкий"), summary (str) — or None when the
    API key is missing or the call fails.
    """
    if not claude_available():
        return None
    try:
        image_b64 = _image_to_base64_jpeg(image)
        result = _analyze_cached(image_b64, round(green_index, 2), address)
    except Exception:
        return None
    if not isinstance(result, dict):
        return None
    return {
        "problems": [str(p) for p in result.get("problems", [])],
        "recommendations": [str(r) for r in result.get("recommendations", [])],
        "priority": str(result.get("priority", "средний")),
        "summary": str(result.get("summary", "")),
    }


# ---------------------------------------------------------------------------
# 2. Appeal letter to the akimat
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _appeal_cached(
    address: str,
    green_index: float,
    problems: tuple[str, ...],
    recommendations: tuple[str, ...],
) -> str | None:
    model = _get_model()
    problems_text = "\n".join(f"- {p}" for p in problems) or "- не выявлены"
    recommendations_text = (
        "\n".join(f"- {r}" for r in recommendations) or "- требуется выезд комиссии"
    )
    prompt = (
        "Ты помощник по составлению официальных обращений граждан в акиматы "
        "Республики Казахстан. Пиши в деловом стиле, вежливо и конкретно.\n\n"
        "Составь официальное обращение в акимат по вопросу озеленения "
        "и благоустройства. Сначала текст на казахском языке, затем "
        "тот же текст на русском языке (раздели заголовками "
        "«Қазақша» и «На русском»). Объём каждой версии — деловое "
        "письмо примерно на 200 слов.\n\n"
        f"Адрес локации: {address}\n"
        f"Green View Index (доля зелени по ИИ-анализу фото): "
        f"{green_index:.2f} из 1.00\n"
        f"Выявленные проблемы:\n{problems_text}\n"
        f"Предложения:\n{recommendations_text}\n\n"
        "Обязательно включи: адрес, метрику GVI, список проблем, "
        "конкретные предложения и упоминание народной поддержки "
        "инициативы на платформе EcoReconstruct AI (голоса горожан). "
        "Верни только текст обращения без пояснений."
    )
    response = model.generate_content(prompt)
    return response.text.strip() or None


def generate_appeal_text(
    address: str,
    green_index: float,
    problems: list[str],
    recommendations: list[str],
) -> str | None:
    """
    Official appeal letter to the akimat in Kazakh and Russian (~200 words each).
    Returns None when the API key is missing or the call fails.
    """
    if not claude_available():
        return None
    try:
        return _appeal_cached(
            address,
            round(green_index, 2),
            tuple(problems),
            tuple(recommendations),
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 3. Dashboard report for city officials
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _report_cached(zones_json: str) -> str | None:
    model = _get_model()
    prompt = (
        "Ты аналитик по городской экологии, готовящий отчёты для "
        "руководства акимата. Пиши структурированно, по-деловому, "
        "на русском языке, с конкретными цифрами и приоритетами.\n\n"
        "На основе данных краудсорсинговой платформы EcoReconstruct AI "
        "подготовь аналитический отчёт для городских чиновников по "
        "топ-5 критичным зонам озеленения.\n\n"
        f"Данные (JSON):\n{zones_json}\n\n"
        "Поля: «Локация», «Green Index» (доля зелени 0–1, меньше — хуже), "
        "«Голоса» (поддержка горожан), «Приоритет» (голоса × (1 − GVI)).\n\n"
        "Структура отчёта:\n"
        "1. Краткая сводка ситуации.\n"
        "2. Рекомендации по бюджетному приоритету (какие зоны "
        "финансировать в первую очередь и почему).\n"
        "3. Ожидаемый эффект от благоустройства.\n"
        "Верни только текст отчёта в Markdown."
    )
    response = model.generate_content(prompt)
    return response.text.strip() or None


def generate_dashboard_report(top_zones: list[dict]) -> str | None:
    """
    Analytic report over the top-5 critical zones
    (fields: Локация, Green Index, Голоса, Приоритет).
    Returns None when the API key is missing or the call fails.
    """
    if not claude_available():
        return None
    try:
        zones_json = json.dumps(top_zones, ensure_ascii=False, sort_keys=True)
        return _report_cached(zones_json)
    except Exception:
        return None
