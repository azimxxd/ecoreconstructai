"""
EcoReconstruct AI — OpenAI image generation (single project key).

Image generation runs through one OpenAI API key configured in
``st.secrets["OPENAI_API_KEY"]`` (or the ``OPENAI_API_KEY`` env var) — the
project owner pays for it. We send the street photo plus a prompt built from
our local YOLOv8 + OpenCV audit (:func:`utils.models.build_eco_prompt`) to
OpenAI's image-edit endpoint with ``input_fidelity="high"`` so the model keeps
the street's structure (road, buildings, camera angle) while restyling it into
a premium downtown boulevard.

Configurable via secrets / env (all optional):
  - OPENAI_API_KEY      : required to enable this engine.
  - OPENAI_IMAGE_MODEL  : model id, default "gpt-image-1" (set "gpt-image-2"
                          once available on your account).
  - OPENAI_IMAGE_QUALITY: "low" | "medium" | "high", default "high".

Every function degrades gracefully: any failure returns ``None`` so the caller
can fall back to the free Hugging Face FLUX pipeline.
"""

from __future__ import annotations

import base64
import io
import os

import streamlit as st
from PIL import Image

# Default model; override with OPENAI_IMAGE_MODEL in secrets (e.g. gpt-image-2).
_DEFAULT_MODEL = "gpt-image-2"
_DEFAULT_QUALITY = "medium"

# Downscale large photos before upload to keep the request light (the model
# re-renders at the requested target size regardless).
_MAX_SIDE_PX = 1024


def _secret(name: str) -> str | None:
    """Read a config value from st.secrets first, then the environment."""
    value = None
    try:
        value = st.secrets[name]
    except Exception:
        value = None
    return value or os.environ.get(name) or None


def openai_available() -> bool:
    """True when an OpenAI API key is configured."""
    return _secret("OPENAI_API_KEY") is not None


def _target_size(image: Image.Image) -> str:
    """Pick the output size closest to the photo's aspect ratio."""
    width, height = image.size
    if width > height * 1.15:
        return "1536x1024"   # landscape street shot
    if height > width * 1.15:
        return "1024x1536"   # portrait / phone shot
    return "1024x1024"


def generate_eco_view_openai(
    image: Image.Image, prompt: str
) -> Image.Image | None:
    """
    Edit the street photo into its premium green future via the OpenAI image
    model on the project key. Returns a PIL Image, or None on any failure
    (no key, no billing, org not verified, quota exhausted, network error …)
    so the caller can fall back to the free pipeline.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("[OPENAI] openai SDK not installed — pip install openai")
        return None

    api_key = _secret("OPENAI_API_KEY")
    if not api_key:
        return None

    model = _secret("OPENAI_IMAGE_MODEL") or _DEFAULT_MODEL
    quality = (_secret("OPENAI_IMAGE_QUALITY") or _DEFAULT_QUALITY).lower()

    # Normalise: RGB + downscale, then encode as PNG for the edit endpoint.
    rgb = image.convert("RGB")
    rgb.thumbnail((_MAX_SIDE_PX, _MAX_SIDE_PX), Image.LANCZOS)
    size = _target_size(rgb)

    def _png_buffer() -> io.BytesIO:
        buf = io.BytesIO()
        rgb.save(buf, format="PNG")
        buf.name = "street.png"
        buf.seek(0)
        return buf

    base_kwargs = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "quality": quality,
    }

    try:
        client = OpenAI(api_key=api_key)
        # input_fidelity="high" keeps the original structure; retry without it
        # for older SDKs / models that don't expose the parameter.
        try:
            result = client.images.edit(
                image=_png_buffer(), input_fidelity="high", **base_kwargs
            )
        except TypeError:
            result = client.images.edit(image=_png_buffer(), **base_kwargs)

        b64 = result.data[0].b64_json
        if not b64:
            print("[OPENAI] response contained no image data")
            return None
        print(f"[OPENAI] Success ({model}, quality={quality})")
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception as exc:
        print(f"[OPENAI] failed: {type(exc).__name__}: {exc}")
        return None
