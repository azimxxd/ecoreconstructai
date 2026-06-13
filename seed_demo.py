"""Throwaway: seed demo submissions so the feed and dashboard can be previewed."""
import base64
import io
import random

import numpy as np
from PIL import Image

from utils import db
from utils.models import analyze_eco_status, generate_eco_friendly_view


def fake_street_photo(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.zeros((300, 400, 3), dtype=np.uint8)
    arr[:140] = [168, 200, 222]                      # sky
    arr[140:230] = [128, 128, 130]                   # buildings/asphalt
    arr[230:] = [105, 100, 95]                       # road
    green_height = rng.integers(10, 90)              # random vegetation strip
    arr[230 - green_height:230] = [70, 150 + rng.integers(0, 40), 75]
    noise = rng.integers(-12, 12, arr.shape)
    return Image.fromarray(np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8))


def b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


addresses = [
    ("Алматы, ул. Абая 44", 14, "aigerim_eco", "🌻"),
    ("Астана, пр. Республики 12", 8, "daniyar_kz", "🚲"),
    ("Алматы, мкр. Аксай-4", 21, "green_almaty", "🌳"),
]
for i, (address, likes, author, avatar) in enumerate(addresses):
    photo = fake_street_photo(seed=i)
    masked, gvi = analyze_eco_status(photo)
    concept = generate_eco_friendly_view(photo, {"green_view_index": gvi})
    db.save_item({
        "address": address,
        "green_index": round(gvi, 3),
        "image_original": b64(photo),
        "image_generated": b64(concept),
        "likes": likes,
        "author": author,
        "avatar": avatar,
        "ai_summary": "Серый двор с дефицитом зелени — нужны деревья и тень.",
    })
print("seeded", len(db.load_all_posts()), "items")
