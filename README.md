# 🌿 EcoReconstruct AI

Crowdsourced urban ecology platform — citizens photograph gray streets, AI
audits the greenery (Green View Index) and renders a "future green concept",
the city sees demand-ranked renovation hotspots.

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

## Project structure

| Path | Purpose |
|---|---|
| `app.py` | Main entry point: custom Eco-Tech CSS, 3 mobile tabs |
| `utils/db.py` | Thread-safe, atomic JSON storage (`load_db`, `save_item`, `add_like`) |
| `utils/models.py` | Mock AI pipelines + TODOs for SegFormer / Stable Diffusion |
| `requirements.txt` | Dependencies |
| `.streamlit/config.toml` | Pinned light Eco theme (consistent in dark mode) |
| `seed_demo.py` | Optional: seed 3 demo submissions |

## Tabs

- **🌱 Новая идея** — upload a street photo → mock SegFormer eco-audit
  (Green View Index + segmentation overlay) → mock GenAI "future green
  concept" → publish to the city feed.
- **🔥 Пульс Города** — chronological feed with "Было / Стало" switcher and
  tap-friendly like voting persisted to JSON.
- **🏛 Акимат Дашборд** — KPIs, urgency-vs-demand Plotly bubble chart, and a
  Top-5 critical-zones table (priority = votes × (1 − GVI)).

## Wiring real AI

- `utils/models.py → analyze_eco_status`: TODO block explains swapping the
  mock for `nvidia/segformer-b0-finetuned-ade-512-512` (Hugging Face).
- `utils/models.py → generate_eco_friendly_view`: paste your Replicate or
  Stability AI token into `GENAI_CONFIG`; TODO block shows both call shapes.
