# 🌿 EcoReconstruct AI

Crowdsourced urban ecology platform with a **TikTok-style UI** — citizens
photograph gray streets, AI audits the greenery (Green View Index) and renders
a "future green concept", the city sees demand-ranked renovation hotspots.

Mobile-first MVP built with Streamlit, Pillow, Plotly and NumPy.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The JSON database (`projects_db.json`) is created automatically on first run.
To populate the feed with demo submissions for a presentation:

```bash
python seed_demo.py
```

## UI (TikTok-style)

The app opens straight into a **full-screen vertical feed** of citizen
eco-solutions (snap-scroll, like Reels): author, address, GVI badge and AI
summary overlaid on the photo. **Swiping a card horizontally** reveals the
AI-generated green future of the same street. The 🔥 button on the right is
the vote/like.

Fixed bottom navbar:

- **🏠 Лента** — back to the feed.
- **👤 Профиль** — your solutions in a 3-column grid, votes received,
  editable name + emoji avatar (persisted to `profile.json`).
- **📷 (center, TikTok-style)** — shoot a photo on the spot
  (`st.camera_input`) or upload from the gallery, then: mock SegFormer
  eco-audit (Green View Index + overlay) → mock GenAI "future green concept"
  → Gemini eco-analysis + official appeal letter → publish to the feed.
- **🏆 Топ** — monthly leaderboard of places with the most votes (podium for
  the top-3) + a toggleable **акимат analytics mode** (KPIs,
  urgency-vs-demand bubble chart, top-5 critical zones, AI report).

## Project structure

| Path | Purpose |
|---|---|
| `app.py` | Main entry point: dark TikTok design system, 4 pages + bottom navbar |
| `utils/db.py` | Thread-safe, atomic JSON storage (`load_db`, `save_item`, `add_like`) |
| `utils/claude_ai.py` | Gemini integration: photo eco-audit, appeal letter, city report |
| `utils/models.py` | Mock AI pipelines + TODOs for SegFormer / Stable Diffusion |
| `requirements.txt` | Dependencies |
| `.streamlit/config.toml` | Pinned dark TikTok theme |
| `seed_demo.py` | Optional: seed 3 demo submissions |

## Wiring real AI

- Put `GOOGLE_API_KEY` in `.streamlit/secrets.toml` to enable the Gemini
  photo analysis, appeal letters and city reports.
- `utils/models.py → analyze_eco_status`: TODO block explains swapping the
  mock for `nvidia/segformer-b0-finetuned-ade-512-512` (Hugging Face).
- `utils/models.py → generate_eco_friendly_view`: paste your Replicate or
  Stability AI token into `GENAI_CONFIG`; TODO block shows both call shapes.
