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
  (`st.camera_input`) or upload from the gallery, then: YOLOv8 + OpenCV
  eco-audit (Green View Index, asphalt %, detected cars) → image-to-image
  "future green concept" (FLUX.1-Kontext) generated from a prompt built off
  that audit → free-LLM eco-analysis + official appeal letter → publish to
  the feed.
- **🏆 Топ** — monthly leaderboard of places with the most votes (podium for
  the top-3) + a toggleable **акимат analytics mode** (KPIs,
  urgency-vs-demand bubble chart, top-5 critical zones, AI report).

## Project structure

| Path | Purpose |
|---|---|
| `app.py` | Main entry point: dark TikTok design system, 4 pages + bottom navbar |
| `utils/db.py` | Thread-safe, atomic JSON storage (`load_db`, `save_item`, `add_like`) |
| `utils/free_ai.py` | Free-LLM (Qwen2.5 via HF router): findings, appeal letter, city report |
| `utils/models.py` | YOLOv8/OpenCV eco-audit + image-to-image generation (FLUX.1-Kontext → InstructPix2Pix → FLUX.1-schnell) |
| `requirements.txt` | Dependencies |
| `.streamlit/config.toml` | Pinned dark TikTok theme |
| `seed_demo.py` | Optional: seed 3 demo submissions |

## Wiring real AI

Everything runs on free models — only one credential is needed.

- Put `HF_API_TOKEN` (free Hugging Face token, role: Read) in
  `.streamlit/secrets.toml`. It unlocks both the high-quality image-to-image
  (FLUX.1-Kontext) and the free-LLM text features (Qwen2.5 via the HF router).
  Without it the app still works: image generation falls back to the
  anonymous InstructPix2Pix Space, and text features fall back to templates.
- `utils/models.py → run_eco_audit`: real YOLOv8 (`yolov8n.pt`, auto-downloaded)
  + OpenCV colour analysis. `eco_audit_safe` falls back to a pure-numpy colour
  audit if the ML stack (torch/opencv/ultralytics) is unavailable.
- `utils/models.py → generate_eco_friendly_view`: builds a structure-preserving
  edit prompt from the audit, runs the image-to-image fallback chain, then
  protects the scene — a Cityscapes SegFormer segmentation (free HF Inference)
  restores the original road / vehicles / people over the result so the
  carriageway is never turned into greenery.
- `utils/free_ai.py`: photo findings, appeal letters and city reports via the
  free HF chat models.
