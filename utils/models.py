"""
EcoReconstruct AI — AI pipeline placeholders.

Two mocked pipelines that mirror the production architecture:

1. analyze_eco_status(image)
   -> (masked_image, green_view_index)
   Production target: semantic segmentation with
   `nvidia/segformer-b0-finetuned-ade-512-512` (ADE20K) from Hugging Face.

2. generate_eco_friendly_view(image, eco_score)
   -> generated_image
   Production target: Stable Diffusion + ControlNet (canny/depth) via a
   hosted inference API (Replicate or Stability AI).

Both functions deliberately sleep for a moment so the UI spinner behaviour
matches the latency profile of real model calls.
"""

from __future__ import annotations

import random
import time

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# ---------------------------------------------------------------------------
# Configuration — paste real credentials here when wiring up production APIs.
# ---------------------------------------------------------------------------
GENAI_CONFIG: dict = {
    # --- Replicate (https://replicate.com) -------------------------------
    "REPLICATE_API_TOKEN": "",          # e.g. "r8_xxxxxxxxxxxxxxxxxxxx"
    "REPLICATE_MODEL": "jagilley/controlnet-canny",
    # --- Stability AI (https://platform.stability.ai) ---------------------
    "STABILITY_API_KEY": "",            # e.g. "sk-xxxxxxxxxxxxxxxxxxxx"
    "STABILITY_ENGINE": "stable-diffusion-xl-1024-v1-0",
    # --- Shared generation parameters -------------------------------------
    "PROMPT_TEMPLATE": (
        "photorealistic urban renovation, lush green park, trees, flower beds, "
        "modern benches, bike lanes, clean walkways, golden hour lighting, "
        "high detail, 8k"
    ),
    "NEGATIVE_PROMPT": "cars, trash, dirt, gray concrete, low quality, blurry",
    "GUIDANCE_SCALE": 7.5,
    "NUM_INFERENCE_STEPS": 30,
    "CONTROLNET_CONDITIONING_SCALE": 0.8,
}

# Simulated latency ranges (seconds) so the demo feels like real inference.
_ANALYSIS_DELAY_RANGE = (1.2, 2.2)
_GENERATION_DELAY_RANGE = (2.0, 3.5)

# Cap working resolution to keep base64 payloads in the JSON DB lightweight.
_MAX_SIDE_PX = 1024


def _normalize(image: Image.Image) -> Image.Image:
    """Convert to RGB and downscale large photos for fast, light processing."""
    rgb_image = image.convert("RGB")
    rgb_image.thumbnail((_MAX_SIDE_PX, _MAX_SIDE_PX), Image.LANCZOS)
    return rgb_image


def analyze_eco_status(image: Image.Image) -> tuple[Image.Image, float]:
    """
    MOCK eco-audit of a street photo.

    Returns:
        masked_image     : the photo with a semi-transparent overlay —
                           green where "vegetation" was detected,
                           red where the scene lacks greenery.
        green_view_index : simulated Green View Index in [0.0, 1.0].

    -----------------------------------------------------------------------
    TODO — replace this mock with real SegFormer inference:
    -----------------------------------------------------------------------
    1. `pip install transformers torch`
    2. Load once at module level (cache with @st.cache_resource):
           from transformers import SegformerImageProcessor, \
               SegformerForSemanticSegmentation
           processor = SegformerImageProcessor.from_pretrained(
               "nvidia/segformer-b0-finetuned-ade-512-512")
           model = SegformerForSemanticSegmentation.from_pretrained(
               "nvidia/segformer-b0-finetuned-ade-512-512")
    3. Run inference:
           inputs = processor(images=image, return_tensors="pt")
           logits = model(**inputs).logits
           seg_map = logits.argmax(dim=1)[0]  # upsample to image size first
    4. ADE20K vegetation class ids: tree=4, grass=9, plant=17, field=29.
       green_view_index = (pixels in those classes) / (total pixels).
    5. Build the overlay from the real seg_map instead of the fake
       brightness heuristic below.
    -----------------------------------------------------------------------
    """
    time.sleep(random.uniform(*_ANALYSIS_DELAY_RANGE))

    base_image = _normalize(image)
    pixels = np.asarray(base_image, dtype=np.float32)

    # --- Fake "segmentation": treat greenish pixels as vegetation. --------
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    vegetation_mask = (green > red * 1.06) & (green > blue * 1.06) & (green > 50)

    green_view_index = float(np.clip(vegetation_mask.mean(), 0.0, 1.0))
    # Add slight jitter so repeat demos don't look hard-coded.
    green_view_index = float(
        np.clip(green_view_index + random.uniform(-0.03, 0.03), 0.01, 0.99)
    )

    # --- Build the semi-transparent overlay (green = good, red = barren). -
    overlay = np.zeros_like(pixels)
    overlay[vegetation_mask] = [46, 204, 113]    # mint green for vegetation
    overlay[~vegetation_mask] = [231, 76, 60]    # soft red for "needs work"

    alpha = 0.35
    blended = (pixels * (1 - alpha) + overlay * alpha).astype(np.uint8)
    masked_image = Image.fromarray(blended)

    return masked_image, green_view_index


def generate_eco_friendly_view(
    image: Image.Image, eco_score: float
) -> Image.Image:
    """
    MOCK generative "future green concept" of the same street.

    Simulates a Stable Diffusion + ControlNet render by boosting greens,
    saturation and warmth of the original photo — the lower the eco_score,
    the more dramatic the simulated transformation.

    -----------------------------------------------------------------------
    TODO — replace with a real hosted GenAI call:
    -----------------------------------------------------------------------
    Option A — Replicate (img2img + ControlNet):
        import replicate
        client = replicate.Client(api_token=GENAI_CONFIG["REPLICATE_API_TOKEN"])
        output_url = client.run(
            GENAI_CONFIG["REPLICATE_MODEL"],
            input={
                "image": <png bytes or data-URI of `image`>,
                "prompt": GENAI_CONFIG["PROMPT_TEMPLATE"],
                "negative_prompt": GENAI_CONFIG["NEGATIVE_PROMPT"],
                "guidance_scale": GENAI_CONFIG["GUIDANCE_SCALE"],
                "num_inference_steps": GENAI_CONFIG["NUM_INFERENCE_STEPS"],
            },
        )
        # download output_url -> PIL.Image

    Option B — Stability AI REST API:
        POST https://api.stability.ai/v2beta/stable-image/control/structure
        headers = {"Authorization": f"Bearer {GENAI_CONFIG['STABILITY_API_KEY']}"}
        files   = {"image": <png bytes>}
        data    = {"prompt": GENAI_CONFIG["PROMPT_TEMPLATE"],
                   "control_strength": GENAI_CONFIG["CONTROLNET_CONDITIONING_SCALE"]}
    -----------------------------------------------------------------------
    """
    time.sleep(random.uniform(*_GENERATION_DELAY_RANGE))

    base_image = _normalize(image)

    # Barren streets (low score) get a stronger simulated "renovation".
    transform_strength = 1.0 + (1.0 - float(np.clip(eco_score, 0.0, 1.0))) * 0.6

    # 1) Boost the green channel directly — the "new vegetation" illusion.
    pixels = np.asarray(base_image, dtype=np.float32)
    pixels[..., 1] = np.clip(pixels[..., 1] * (1.0 + 0.25 * transform_strength), 0, 255)
    generated = Image.fromarray(pixels.astype(np.uint8))

    # 2) Saturation, brightness and contrast for a polished concept-art look.
    generated = ImageEnhance.Color(generated).enhance(1.0 + 0.30 * transform_strength)
    generated = ImageEnhance.Brightness(generated).enhance(1.06)
    generated = ImageEnhance.Contrast(generated).enhance(1.08)

    # 3) Gentle smoothing + sharpen mimics a diffusion model's re-render.
    generated = generated.filter(ImageFilter.SMOOTH_MORE)
    generated = generated.filter(
        ImageFilter.UnsharpMask(radius=2, percent=80, threshold=2)
    )

    return generated
