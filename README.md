# 🌿 EcoReconstruct AI

> **🌐 Live:** **https://ecoreconstructai.streamlit.app/**
> Хостинг — [Streamlit Community Cloud](https://streamlit.io/cloud). Деплой автоматический из ветки `master` этого репозитория.

Краудсорсинговая платформа городской экологии с **интерфейсом в стиле TikTok**:
горожане фотографируют серые улицы, ИИ оценивает озеленение (Green View Index)
и рисует «зелёное будущее» той же улицы, а город получает рейтинг точек,
которые жители больше всего хотят благоустроить.

Mobile-first веб-приложение на Streamlit.

---

## 🧰 Технологический стек (подробно)

| Слой | Технологии | Где в коде |
|---|---|---|
| **Фронтенд / UI** | Streamlit + кастомная дизайн-система «City Scanner» на чистом CSS (snap-scroll лента, стеклянный док, viewfinder-рамки, шрифты Unbounded / JetBrains Mono / Manrope) | [`app.py`](app.py) — `GLOBAL_CSS`, `FEED_CSS`, `CAMERA_CSS` |
| **Аутентификация** | Логин/пароль (без внешних провайдеров). Пароли — PBKDF2-HMAC-SHA256 (stdlib, 200k итераций, соль). «Оставаться в системе» — HMAC-подписанный токен в cookie `eco_auth`, чтение через `st.context.cookies` (синхронно, без мигания) | [`utils/db.py`](utils/db.py), [`app.py`](app.py) |
| **База данных** | Supabase **PostgreSQL**. Драйвер `psycopg2` с пулом соединений (`ThreadedConnectionPool`) как `@st.cache_resource`-синглтон. Схема создаётся и мигрируется автоматически при старте | [`utils/db.py`](utils/db.py), [`schema.sql`](schema.sql) |
| **Компьютерное зрение** | **YOLOv8** (`ultralytics`, веса `yolov8n.pt`) для подсчёта машин + **OpenCV** (HSV-сегментация зелени/асфальта) + **NumPy**-метрики (яркость, цветность, захламлённость) → сводный **Street Decay Index** | [`utils/models.py`](utils/models.py) — `run_eco_audit`, `compute_decay_index` |
| **Генерация изображений** | Многоуровневый каскад (лучшее → запасное): **OpenAI `gpt-image`** (основной движок) → **FLUX.1-Kontext-dev** через HF Inference Providers router → бесплатный ZeroGPU Space → **InstructPix2Pix** → **FLUX.1-schnell** (text-to-image). Промпт строится из аудита и Street Decay Index | [`utils/openai_gen.py`](utils/openai_gen.py), [`utils/models.py`](utils/models.py) — `generate_eco_friendly_view`, `build_eco_prompt` |
| **Защита сцены** | Семантическая сегментация **Cityscapes SegFormer** (`nvidia/segformer-b0-...`, бесплатный HF Inference) возвращает оригинальную дорогу / машины / людей поверх сгенерированного кадра, чтобы проезжая часть не «озеленялась» | [`utils/models.py`](utils/models.py) — `_protect_scene` |
| **Текстовый ИИ** | Бесплатные чат-модели через HF router (**Qwen2.5-72B/7B-Instruct**, **Llama-3.1-8B**): выводы по аудиту, официальное обращение в акимат (KZ+RU, с плейсхолдерами вместо имён), аналитический отчёт для города | [`utils/free_ai.py`](utils/free_ai.py) |
| **Графики / аналитика** | **Plotly** (bubble-chart срочность×спрос) + **pandas** (KPI, топ-зоны) | [`app.py`](app.py) — `render_top` |
| **Рантайм / деплой** | Python 3.12, Streamlit Community Cloud (прод). Также есть [`Dockerfile`](Dockerfile) для self-host | [`Dockerfile`](Dockerfile), [`.streamlit/config.toml`](.streamlit/config.toml) |

Полный список пакетов — в [`requirements.txt`](requirements.txt).

---

## ✨ Возможности по экранам

Приложение открывается сразу в **полноэкранную вертикальную ленту** эко-решений
(snap-scroll, как Reels): автор, адрес, бейдж GVI и ИИ-сводка поверх фото.
**Горизонтальный свайп** карточки показывает зелёное будущее той же улицы.
Кнопка 🔥 справа — голос/лайк (1 аккаунт = 1 голос, на уровне БД).

Плавающий нижний док:

- **🏠 Лента** — вернуться к ленте.
- **📷 (центральная кнопка)** — сканер улицы:
  1. Полноэкранная камера (`st.camera_input`) или загрузка из галереи.
  2. **Предпросмотр + кнопка подтверждения** — тяжёлый ИИ-пайплайн запускается
     только после подтверждения (а не на каждый кадр).
  3. YOLOv8 + OpenCV аудит → image-to-image «зелёный концепт» → ИИ-анализ
     локации **по кнопке** → черновик обращения в акимат.
  4. Кнопка **«✕ Выход»** и перехват навигации с модалкой
     «Изменения не сохранятся, если выйти».
- **🏆 Топ** — месячный лидерборд (подиум топ-3) + переключаемый
  **режим акимата**: KPI, bubble-chart «срочность × спрос горожан», топ-5
  критичных зон, ИИ-отчёт.
- **👤 Профиль** — твои решения в сетке 3×N, полученные голоса, выбор
  emoji-аватара, выход из аккаунта.

---

## 🚀 Локальный запуск

```bash
pip install -r requirements.txt
streamlit run app.py
```

Перед первым запуском создай `.streamlit/secrets.toml` на основе
[`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example).
Схема БД создаётся/мигрируется автоматически при старте (`init_db`).

---

## ⚙️ Конфигурация (`.streamlit/secrets.toml`)

```toml
# ── ВАЖНО: ключи верхнего уровня должны идти ДО любого [раздела] ──
# (иначе TOML вложит их в предыдущую секцию и приложение их не увидит)

OPENAI_API_KEY = "sk-..."   # основной движок генерации (gpt-image)
HF_API_TOKEN   = "hf_..."   # бесплатный фолбэк (FLUX) + текстовый ИИ (Qwen2.5)

[database]
# Supabase → Project Settings → Database → Connection string (Transaction pooler, порт 6543)
url = "postgresql://postgres.PROJECT_REF:PASSWORD@aws-0-REGION.pooler.supabase.com:6543/postgres"

[app]
# Ключ подписи cookie «оставаться в системе». Необязателен (по умолчанию
# выводится из URL базы). Для прода задай свой:
#   python -c "import secrets; print(secrets.token_hex(32))"
secret_key = ""
```

| Секрет | Обязателен | Назначение |
|---|---|---|
| `[database].url` | ✅ да | Подключение к Supabase Postgres |
| `OPENAI_API_KEY` | нет* | Основной движок генерации (`gpt-image`). Нужен включённый billing |
| `HF_API_TOKEN` | нет* | Бесплатная генерация FLUX + все текстовые ИИ-функции |
| `[app].secret_key` | нет | Подпись cookie постоянного входа |

\* Без обоих ключей приложение всё равно работает: генерация уходит на
анонимный InstructPix2Pix, а текстовые функции — на детерминированные шаблоны.

> **Деплой на Streamlit Cloud:** те же секреты вставляются в
> *App → Settings → Secrets*. `secrets.toml` в репозиторий **не** коммитится
> (он в `.gitignore`).

---

## 🗄️ Схема базы данных

Три таблицы (см. [`schema.sql`](schema.sql); идентичная схема создаётся
автоматически в `utils/db.py → _init_schema`):

- **`users`** — `id`, `username` (уникальный индекс по `lower(username)`),
  `password_hash` (PBKDF2), `email`, `name`, `avatar`, `created_at`.
  Колонка `google_sub` оставлена nullable для совместимости со старыми строками.
- **`posts`** — фото (base64), `green_index`, ИИ-поля (`ai_problems`,
  `ai_recommendations`, `ai_priority`, `ai_summary`), `user_id`, `created_at`.
- **`likes`** — `PRIMARY KEY (user_id, post_id)` обеспечивает правило
  «1 аккаунт = 1 голос» на уровне БД.

---

## 🧩 Структура проекта

| Путь | Назначение |
|---|---|
| [`app.py`](app.py) | Точка входа: дизайн-система, аутентификация + cookie-сессия, 4 экрана, нижний док, кэширование ленты |
| [`utils/db.py`](utils/db.py) | Postgres-слой (пул соединений), регистрация/вход (PBKDF2), токены cookie, посты, лайки |
| [`utils/models.py`](utils/models.py) | YOLOv8 + OpenCV аудит, Street Decay Index, каскад генерации, защита сцены SegFormer |
| [`utils/openai_gen.py`](utils/openai_gen.py) | Генерация через OpenAI `gpt-image` (основной движок, с graceful-фолбэком) |
| [`utils/free_ai.py`](utils/free_ai.py) | Бесплатный LLM (Qwen2.5 через HF router): выводы, письмо в акимат, отчёт города |
| [`schema.sql`](schema.sql) | SQL-схема для ручной инициализации Supabase |
| [`Dockerfile`](Dockerfile) | Образ для self-host (системные либы OpenCV/psycopg2) |
| [`requirements.txt`](requirements.txt) | Зависимости |
| [`.streamlit/config.toml`](.streamlit/config.toml) | Тёмная тема |

---

## 📝 Заметки по ИИ

- **Аудит фото** (`run_eco_audit`): YOLOv8 (`yolov8n.pt`, скачивается
  автоматически) + OpenCV. `eco_audit_safe` откатывается на чистый
  NumPy-аудит по цвету, если ML-стек (torch/opencv/ultralytics) недоступен.
- **Генерация** (`generate_eco_friendly_view`): строит structure-preserving
  промпт из аудита и Street Decay Index, прогоняет каскад движков, затем
  защищает сцену сегментацией (дорога/машины/люди берутся из оригинала).
- **Письмо в акимат и ИИ-оценка** грунтуются строго на метриках CV и
  **не выдумывают имена/телефоны/даты** — вместо них остаются заполнители
  `[ФИО]`, `[контактный телефон]`, `[дата]`.
- Все ИИ-функции **деградируют мягко**: при отсутствии ключей/ошибках
  приложение не падает, а использует фолбэки.
