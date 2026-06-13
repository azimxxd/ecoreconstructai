"""
EcoReconstruct AI — free text-generation backend (replaces Gemini).

All "AI text" features now run on free Hugging Face router chat models
(Qwen2.5-Instruct, Llama-3.1) using the same ``HF_API_TOKEN`` that powers the
image generation. No Google / Gemini key is required.

Public functions (same shapes the UI consumed before):

- free_ai_available()        : True when an HF token is configured.
- generate_findings()        : turns the YOLO/OpenCV eco-audit metrics into
                               {problems, recommendations, priority, summary}.
- generate_appeal_text()     : official appeal letter to the akimat (KZ + RU).
- generate_dashboard_report(): analytic report over the top-5 critical zones.

Design notes:
- The vision step is done locally by YOLOv8 + OpenCV (see utils.models.
  run_eco_audit). These functions only do the *language* part on top of those
  objective metrics, so the text is grounded in real detections.
- Every function degrades gracefully: if the token is missing or every model
  fails, it falls back to a deterministic template instead of raising, so the
  Streamlit app never crashes.
- Heavy calls are cached with @st.cache_data(ttl=3600) keyed on a stable JSON
  payload.
"""

from __future__ import annotations

import json
import os
import re

import requests
import streamlit as st

# Free chat models on the HF router, tried in order. Qwen2.5-72B has the best
# Kazakh/Russian quality; the smaller ones are faster fallbacks.
CHAT_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]
_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
_CHAT_TIMEOUT = 120


# ---------------------------------------------------------------------------
# Token / availability helpers
# ---------------------------------------------------------------------------
def _get_token() -> str | None:
    """Read HF_API_TOKEN from st.secrets first, then the environment."""
    token = None
    try:
        token = st.secrets["HF_API_TOKEN"]
    except Exception:
        token = None
    return token or os.environ.get("HF_API_TOKEN") or None


def free_ai_available() -> bool:
    """True when an HF token is present (needed for the free LLM calls)."""
    return _get_token() is not None


def _chat(messages: list[dict], max_tokens: int, temperature: float = 0.6) -> str | None:
    """
    Call the HF router chat-completions API, trying each free model in turn.

    Returns the assistant text, or None if every model failed.
    """
    token = _get_token()
    if not token:
        return None

    headers = {"Authorization": f"Bearer {token}"}
    for model in CHAT_MODELS:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            resp = requests.post(
                _CHAT_URL, headers=headers, json=payload, timeout=_CHAT_TIMEOUT
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                if content and content.strip():
                    print(f"[FREE_AI] OK via {model}")
                    return content.strip()
                print(f"[FREE_AI] {model} returned empty content")
            else:
                detail = resp.text[:200]
                print(f"[FREE_AI] {model} -> {resp.status_code}: {detail}")
        except (requests.exceptions.RequestException, KeyError, ValueError) as exc:
            print(f"[FREE_AI] {model} failed: {type(exc).__name__}: {exc}")
            continue
    return None


def _extract_json(text: str) -> dict | None:
    """Parse a JSON object from model output, tolerating code fences/prose."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
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
# 1. Findings from the YOLO/OpenCV audit  (replaces Gemini Vision)
# ---------------------------------------------------------------------------
def _fallback_findings(audit: dict) -> dict:
    """Deterministic findings derived straight from the audit metrics."""
    green = float(audit.get("green_view_index", 0.0))
    asphalt = float(audit.get("asphalt_coverage", 0.0))
    cars = int(audit.get("cars_detected", 0))

    problems = list(audit.get("critical_flaws") or [])
    if not problems:
        if green < 10:
            problems.append("Очень низкий уровень озеленения территории.")
        if asphalt > 45:
            problems.append("Преобладание асфальта и бетона, эффект теплового острова.")
        if cars >= 2:
            problems.append("Пространство занято припаркованными автомобилями.")
    if not problems:
        problems.append("Серьёзных проблем не выявлено, но есть потенциал для озеленения.")

    recommendations = [
        "Высадить уличные деревья с широкой кроной вдоль тротуаров.",
        "Разбить газоны, клумбы и кустарники на свободных участках.",
    ]
    if asphalt > 45:
        recommendations.append("Сократить площадь асфальта, обустроить зелёный бульвар.")
    if cars >= 2:
        recommendations.append("Организовать пешеходную зону вместо парковки.")
    if green < 10:
        recommendations.append("Озеленить фасады вертикальными садами.")

    if green < 10 or asphalt > 45:
        priority = "высокий"
    elif green < 25:
        priority = "средний"
    else:
        priority = "низкий"

    summary = audit.get("psychological_impact") or (
        f"Индекс озеленения {green:.0f}%, асфальт {asphalt:.0f}%, "
        f"машин в кадре: {cars}."
    )
    return {
        "problems": problems,
        "recommendations": recommendations,
        "priority": priority,
        "summary": summary,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def _findings_cached(audit_json: str, address: str) -> dict:
    audit = json.loads(audit_json)
    system = (
        "Ты эксперт по городскому экологическому планированию в Казахстане. "
        "Тебе дают объективные метрики, измеренные моделью компьютерного зрения "
        "(YOLOv8 + OpenCV) по фотографии улицы. Делай выводы СТРОГО на основе "
        "этих метрик: не выдумывай объекты, которых нет в данных, не указывай "
        "конкретные названия улиц, имена людей, организаций или точные цифры "
        "бюджета. Формулируй проблемы и рекомендации конкретно, но опираясь "
        "только на приведённые показатели. Отвечай строго в JSON без "
        "markdown-обёртки, на русском языке, кратко и по делу."
    )
    user = (
        f"Адрес: {address}\n"
        f"Метрики компьютерного зрения по фото:\n"
        f"- Green View Index (доля зелени): {audit.get('green_view_index', 0):.1f}%\n"
        f"- Покрытие асфальтом/бетоном: {audit.get('asphalt_coverage', 0):.1f}%\n"
        f"- Обнаружено автомобилей: {audit.get('cars_detected', 0)}\n"
        f"- Риск теплового острова: {audit.get('urban_heat_risk', 'н/д')}\n\n"
        "Верни JSON строго такого формата:\n"
        "{\n"
        '  "problems": ["конкретная проблема 1", "проблема 2", ...],\n'
        '  "recommendations": ["конкретное действие 1", "действие 2", ...],\n'
        '  "priority": "высокий|средний|низкий",\n'
        '  "summary": "одно предложение-вывод"\n'
        "}\n"
        "problems — экологические/урбанистические проблемы, следующие из метрик. "
        "recommendations — что посадить, установить или убрать. "
        "priority — исходя из уровня зелени и асфальта."
    )
    text = _chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=700,
        temperature=0.5,
    )
    parsed = _extract_json(text) if text else None
    if not isinstance(parsed, dict) or "problems" not in parsed:
        return _fallback_findings(audit)

    return {
        "problems": [str(p) for p in parsed.get("problems", [])] or
        _fallback_findings(audit)["problems"],
        "recommendations": [str(r) for r in parsed.get("recommendations", [])] or
        _fallback_findings(audit)["recommendations"],
        "priority": str(parsed.get("priority", "средний")),
        "summary": str(parsed.get("summary", "")),
    }


def generate_findings(audit: dict, address: str) -> dict:
    """
    Turn the YOLO/OpenCV audit metrics into problems / recommendations /
    priority / summary using a free LLM, with a deterministic fallback.

    Always returns a dict (never None) so the UI can always render something.
    """
    try:
        audit_json = json.dumps(audit, ensure_ascii=False, sort_keys=True)
        return _findings_cached(audit_json, address)
    except Exception:
        return _fallback_findings(audit)


# ---------------------------------------------------------------------------
# 2. Appeal letter to the akimat  (KZ + RU)
# ---------------------------------------------------------------------------
def _fallback_appeal(
    address: str, green_index: float, problems: tuple[str, ...],
    recommendations: tuple[str, ...],
) -> str:
    """Plain template appeal when the LLM is unavailable."""
    problems_text = "\n".join(f"- {p}" for p in problems) or "- требуется выезд комиссии"
    recs_text = "\n".join(f"- {r}" for r in recommendations) or "- озеленение территории"
    return (
        "Қазақша\n\n"
        "Кімге: [әкімдіктің атауы]\n"
        "Кімнен: [Аты-жөні], [байланыс телефоны], [электрондық пошта]\n\n"
        f"Әкімдікке өтініш. Мекенжай: {address}. ЖИИ-талдау бойынша аумақтың "
        f"көгалдандыру деңгейі төмен (Green View Index {green_index:.2f}). "
        "Көгалдандыру және абаттандыру жұмыстарын жүргізуді сұраймыз.\n\n"
        "Күні: [күні]\n"
        "Қолы: ____________ / [Аты-жөні]\n\n"
        "На русском\n\n"
        "Кому: [наименование акимата]\n"
        "От кого: [ФИО заявителя], [контактный телефон], [эл. почта]\n\n"
        f"Обращение в акимат. Адрес: {address}.\n"
        f"Green View Index (доля зелени по ИИ-анализу): {green_index:.2f} из 1.00\n"
        f"Выявленные проблемы:\n{problems_text}\n"
        f"Предложения:\n{recs_text}\n\n"
        "Просим рассмотреть вопрос озеленения и благоустройства данной "
        "территории. Инициативу поддерживают горожане на платформе "
        "EcoReconstruct AI.\n\n"
        "Дата: [дата]\n"
        "Подпись: ____________ / [ФИО заявителя]"
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _appeal_cached(
    address: str, green_index: float,
    problems: tuple[str, ...], recommendations: tuple[str, ...],
) -> str:
    problems_text = "\n".join(f"- {p}" for p in problems) or "- не выявлены"
    recs_text = "\n".join(f"- {r}" for r in recommendations) or "- требуется выезд комиссии"
    system = (
        "Ты помощник по составлению официальных обращений граждан в акиматы "
        "Республики Казахстан. Пиши в деловом стиле, вежливо и конкретно. "
        "КРИТИЧЕСКИ ВАЖНО: никогда не выдумывай и не подставляй конкретные "
        "имена, ФИО, телефоны, адреса электронной почты, даты, номера и "
        "должности. Вместо любых таких данных оставляй заполнители в "
        "квадратных скобках: [ФИО заявителя], [контактный телефон], "
        "[эл. почта], [наименование акимата], [дата]. Для подписи оставь "
        "строку вида «Подпись: ____________ / [ФИО заявителя]»."
    )
    user = (
        "Составь официальное обращение в акимат по вопросу озеленения и "
        "благоустройства. Сначала текст на казахском языке (заголовок "
        "«Қазақша»), затем тот же текст на русском (заголовок «На русском»). "
        "Объём каждой версии — деловое письмо примерно на 200 слов.\n\n"
        f"Адрес локации: {address}\n"
        f"Green View Index (доля зелени по ИИ-анализу фото): {green_index:.2f} из 1.00\n"
        f"Выявленные проблемы:\n{problems_text}\n"
        f"Предложения:\n{recs_text}\n\n"
        "Структура каждой версии: блок «Кому»/«Кімге» и «От кого»/«Кімнен» с "
        "заполнителями в квадратных скобках, тело письма, затем строки даты и "
        "подписи с заполнителями.\n"
        "Обязательно включи: адрес, метрику GVI, список проблем, конкретные "
        "предложения и упоминание народной поддержки инициативы на платформе "
        "EcoReconstruct AI.\n"
        "ЗАПРЕЩЕНО подставлять вымышленные имена, телефоны, e-mail и даты — "
        "только заполнители в квадратных скобках. Верни только текст обращения "
        "без пояснений."
    )
    text = _chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=900,
        temperature=0.6,
    )
    return text or _fallback_appeal(address, green_index, problems, recommendations)


def generate_appeal_text(
    address: str, green_index: float,
    problems: list[str], recommendations: list[str],
) -> str | None:
    """Official appeal letter to the akimat in Kazakh and Russian."""
    try:
        return _appeal_cached(
            address, round(green_index, 2),
            tuple(problems), tuple(recommendations),
        )
    except Exception:
        return _fallback_appeal(
            address, round(green_index, 2),
            tuple(problems), tuple(recommendations),
        )


# ---------------------------------------------------------------------------
# 3. Dashboard report for city officials
# ---------------------------------------------------------------------------
def _fallback_report(top_zones: list[dict]) -> str:
    lines = ["## Аналитический отчёт по озеленению\n", "### Топ критичных зон\n"]
    for i, z in enumerate(top_zones, 1):
        lines.append(
            f"{i}. **{z.get('Локация', 'н/д')}** — Green Index "
            f"{z.get('Green Index', 0):.2f}, голосов: {z.get('Голоса', 0)}, "
            f"приоритет: {z.get('Приоритет', 0)}"
        )
    lines.append(
        "\n**Рекомендация:** финансировать в первую очередь зоны с высоким "
        "приоритетом (много голосов при дефиците зелени)."
    )
    return "\n".join(lines)


@st.cache_data(ttl=3600, show_spinner=False)
def _report_cached(zones_json: str) -> str:
    system = (
        "Ты аналитик по городской экологии, готовящий отчёты для руководства "
        "акимата. Пиши структурированно, по-деловому, на русском языке, с "
        "конкретными цифрами и приоритетами."
    )
    user = (
        "На основе данных краудсорсинговой платформы EcoReconstruct AI "
        "подготовь аналитический отчёт по топ-5 критичным зонам озеленения.\n\n"
        f"Данные (JSON):\n{zones_json}\n\n"
        "Поля: «Локация», «Green Index» (доля зелени 0–1, меньше — хуже), "
        "«Голоса» (поддержка горожан), «Приоритет» (голоса × (1 − GVI)).\n\n"
        "Структура отчёта:\n"
        "1. Краткая сводка ситуации.\n"
        "2. Рекомендации по бюджетному приоритету.\n"
        "3. Ожидаемый эффект от благоустройства.\n"
        "Верни только текст отчёта в Markdown."
    )
    text = _chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=1200,
        temperature=0.5,
    )
    return text or _fallback_report(json.loads(zones_json))


def generate_dashboard_report(top_zones: list[dict]) -> str | None:
    """Analytic report over the top-5 critical zones."""
    try:
        zones_json = json.dumps(top_zones, ensure_ascii=False, sort_keys=True)
        return _report_cached(zones_json)
    except Exception:
        return _fallback_report(top_zones)
