"""
EcoReconstruct AI — AI pipeline placeholders.

Two mocked pipelines that mirror the production architecture:

1. analyze_eco_status(image)
   -> (masked_image, green_view_index)
   Production target: semantic segmentation with
   `nvidia/segformer-b0-finetuned-ade-512-512` (ADE20K) from Hugging Face.

2. generate_eco_friendly_view(image, eco_score)
   -> generated_image
   LIVE image-to-image: builds a scene-tailored, structure-preserving edit
   instruction and runs a 3-tier free fallback chain via the Gradio Client —
   FLUX.1-Kontext-dev (best, needs HF_API_TOKEN) → InstructPix2Pix (anonymous)
   → FLUX.1-schnell text-to-image (last resort, ignores the photo).
   An ``HF_API_TOKEN`` (st.secrets or env var) unlocks the high-quality path.

``analyze_eco_status`` deliberately sleeps for a moment so the UI spinner
behaviour matches the latency profile of a real model call.
"""

from __future__ import annotations

import base64
import io
import os
import random
import time

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageFilter

# ---------------------------------------------------------------------------
# Shared helpers and configuration.
# ---------------------------------------------------------------------------

# Simulated latency range (seconds) so the analysis step feels like real
# inference.
_ANALYSIS_DELAY_RANGE = (1.2, 2.2)

# Cap working resolution to keep base64 payloads in the JSON DB lightweight.
_MAX_SIDE_PX = 1024


def _get_secret(name: str) -> str | None:
    """Read a credential from st.secrets first, then the environment."""
    value = None
    try:
        value = st.secrets[name]
    except Exception:
        pass
    return value or os.environ.get(name) or None


def _normalize(image: Image.Image) -> Image.Image:
    """Convert to RGB and downscale large photos for fast, light processing."""
    rgb_image = image.convert("RGB")
    rgb_image.thumbnail((_MAX_SIDE_PX, _MAX_SIDE_PX), Image.LANCZOS)
    return rgb_image


# ---------------------------------------------------------------------------
# Image generation — eco-friendly "green future" of the user's photo.
#
# 3-tier chain (best → last resort), all free:
#   1. FLUX.1-Kontext-dev  (HF Space, ZeroGPU) — state-of-the-art instruction
#      editing. Preserves the building geometry / road / perspective and only
#      ADDS greenery, so it doesn't do illogical things like turning asphalt
#      into a field. Needs an HF token (free tier daily quota).
#   2. InstructPix2Pix     (HF Space, ZeroGPU) — works anonymously, lower
#      quality, used when there is no token or Kontext's quota is exhausted.
#   3. FLUX.1-schnell      (HF Inference API) — pure text-to-image, ignores
#      the photo. Only a last resort so the UI always shows something.
#
# The edit instruction is built per-photo from a quick scene analysis
# (asphalt vs greenery vs sky), so the prompt is tailored to what the picture
# actually needs instead of a fixed generic string.
# ---------------------------------------------------------------------------

# HF Spaces (image-to-image), in order of preference.
_KONTEXT_SPACE = "black-forest-labs/FLUX.1-Kontext-dev"
_PIX2PIX_SPACE = "timbrooks/instruct-pix2pix"

# Fallback: text-to-image (does NOT use the input photo).
_TXT2IMG_FALLBACK_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "black-forest-labs/FLUX.1-schnell"
)
_TXT2IMG_FALLBACK_PROMPT = (
    "A photorealistic, highly detailed architectural visualization of a "
    "sustainable modern eco-friendly city street. Lush green vertical gardens, "
    "climbing ivy on walls, a modern pocket park with young trees instead of "
    "bare asphalt, clean wooden benches, solar panels on roofs, bright sunny "
    "day, architectural digest style, 8k resolution."
)

HF_REQUEST_TIMEOUT = 120


def _pil_to_tempfile(image: Image.Image) -> str:
    """Save a PIL Image to a temporary file and return its path."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    normalized = _normalize(image)
    normalized.save(tmp, format="JPEG", quality=92)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Scene-aware prompt builder.
# ---------------------------------------------------------------------------
def _analyze_scene(image: Image.Image) -> dict[str, float]:
    """
    Cheap colour analysis of the photo (no ML) to drive the edit prompt.

    Returns rough area fractions in [0, 1]:
        greenery : existing vegetation,
        hardscape: asphalt / concrete / bare grey paving,
        sky      : open sky.
    """
    arr = np.asarray(_normalize(image), dtype=np.float32)
    red, green, blue = arr[..., 0], arr[..., 1], arr[..., 2]

    max_c = arr.max(axis=-1)
    min_c = arr.min(axis=-1)
    saturation = (max_c - min_c) / (max_c + 1e-5)

    greenery = (green > red * 1.06) & (green > blue * 1.06) & (green > 50)
    # Grey / desaturated mid-tones = asphalt, concrete, paving.
    hardscape = (saturation < 0.18) & (max_c > 45) & (max_c < 205)
    # Bright, blue-dominant, low-saturation pixels ≈ sky.
    sky = (blue > red) & (blue > green * 0.95) & (max_c > 150) & (saturation < 0.45)

    return {
        "greenery": float(greenery.mean()),
        "hardscape": float(hardscape.mean()),
        "sky": float(sky.mean()),
    }


def _scene_to_audit(image: Image.Image) -> dict:
    """
    Cheap fallback "audit" from colour analysis, in the same shape as
    run_eco_audit(), for when no real YOLO audit was supplied.
    """
    scene = _analyze_scene(image)
    return {
        "green_view_index": scene["greenery"] * 100.0,
        "asphalt_coverage": scene["hardscape"] * 100.0,
        "cars_detected": 0,
    }


def build_eco_prompt(audit: dict) -> str:
    """
    Build a tailored, structure-preserving img2img instruction from the eco
    audit (YOLOv8 + OpenCV metrics from run_eco_audit).

    The instruction always tells the model to KEEP the existing buildings,
    road and composition and only ADD green / sustainable elements — this is
    what stops the model from doing illogical edits. The specific additions
    are chosen from the measured metrics: how much asphalt and greenery the
    scene has, and how many cars YOLO detected.
    """
    green = float(audit.get("green_view_index", 0.0))      # 0–100
    asphalt = float(audit.get("asphalt_coverage", 0.0))    # 0–100

    additions: list[str] = [
        "plant rows of tall mature trees with full leafy canopies along the "
        "sidewalks and the verges between the road and the buildings, and add "
        "grass strips, flower beds, planters and bushes on the pavements"
    ]

    # Lots of bare paving → green the roadsides, NOT the carriageway itself.
    if asphalt > 40:
        additions.append(
            "line both sides of the road with dense rows of trees, add a "
            "narrow planted green median and turn any empty bare ground next "
            "to the road into a small pocket park"
        )

    # Little existing greenery → emphasise facade greening and abundance.
    if green < 15:
        additions.append(
            "cover the blank building facades and walls with lush vertical "
            "gardens and climbing ivy"
        )

    additions.append(
        "add a few solar panels on the rooftops and clean wooden benches on "
        "the sidewalks"
    )

    instruction = (
        "Add green, eco-friendly landscaping to this SAME street: "
        + "; ".join(additions) + ". "
        "VERY IMPORTANT: keep the asphalt road and driving lanes exactly as "
        "they are — do NOT place trees, grass or anything on the road surface, "
        "and do not turn the road into a park or alley. Add greenery only on "
        "the sidewalks, roadside verges, empty ground and building facades. "
        "Keep the existing buildings, road, cars, camera angle, perspective "
        "and overall composition unchanged. "
        "Photorealistic, natural soft daylight, realistic proportions, "
        "highly detailed."
    )
    return instruction


def _fetch_image_from_url(url: str) -> Image.Image | None:
    """Download an image from a URL into a PIL Image (best effort)."""
    try:
        resp = requests.get(url, timeout=HF_REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except requests.exceptions.RequestException:
        pass
    return None


def _result_to_image(result: object) -> Image.Image | None:
    """
    Extract a PIL image from whatever a Gradio endpoint returned.

    Handles the shapes the two Spaces use:
      - FLUX.1-Kontext '/infer' -> (image_path, seed)            (image first)
      - InstructPix2Pix '/generate' -> (seed, cfg, cfg, path)    (image last)
    plus bare paths, PIL images and {"path"/"name"/"url": ...} dicts. We scan
    every element (output first) and return the first openable image.
    """
    if result is None:
        return None

    if isinstance(result, (list, tuple)):
        candidates = list(result)
    else:
        candidates = [result]

    for item in candidates:
        try:
            if isinstance(item, Image.Image):
                return item.convert("RGB")
            if isinstance(item, str) and os.path.exists(item):
                return Image.open(item).convert("RGB")
            if isinstance(item, dict):
                path = item.get("path") or item.get("name")
                if path and os.path.exists(path):
                    return Image.open(path).convert("RGB")
                url = item.get("url")
                if url:
                    fetched = _fetch_image_from_url(url)
                    if fetched is not None:
                        return fetched
        except Exception:
            continue
    return None


def _try_flux_kontext(
    image: Image.Image, prompt: str, hf_token: str
) -> Image.Image | None:
    """
    Primary path — FLUX.1-Kontext-dev instruction editing via the Gradio Client.

    Highest quality and structure-preserving, but the official ZeroGPU Space
    needs an authenticated HF token (free tier has a daily GPU quota). On quota
    errors / failures this returns None so the caller can fall back.
    """
    try:
        from gradio_client import Client, handle_file
    except ImportError:
        print("[KONTEXT] gradio_client not installed — pip install gradio_client")
        return None

    img_path = _pil_to_tempfile(image)
    try:
        print(f"[KONTEXT] Connecting to '{_KONTEXT_SPACE}'...")
        client = Client(_KONTEXT_SPACE, token=hf_token, verbose=False)

        # '/infer' signature:
        #   (input_image, prompt, seed, randomize_seed, guidance_scale, steps)
        # Returns: (edited_image_path, seed)
        print("[KONTEXT] Calling '/infer'...")
        result = client.predict(
            input_image=handle_file(img_path),
            prompt=prompt,
            seed=0,
            randomize_seed=True,
            guidance_scale=2.5,
            steps=28,
            api_name="/infer",
        )
        generated = _result_to_image(result)
        if generated is not None:
            print("[KONTEXT] Success")
            return generated
        print("[KONTEXT] returned no usable image")
    except Exception as exc:
        print(f"[KONTEXT] failed: {type(exc).__name__}: {exc}")
    finally:
        try:
            os.unlink(img_path)
        except OSError:
            pass
    return None


def _try_instruct_pix2pix(image: Image.Image, prompt: str) -> Image.Image | None:
    """
    Fallback path — InstructPix2Pix via the Gradio Client.

    Works anonymously (free public ZeroGPU), lower quality than Kontext. Used
    when there is no HF token or Kontext's quota is exhausted.
    """
    try:
        from gradio_client import Client, handle_file
    except ImportError:
        print("[PIX2PIX] gradio_client not installed — pip install gradio_client")
        return None

    img_path = _pil_to_tempfile(image)
    try:
        print(f"[PIX2PIX] Connecting to '{_PIX2PIX_SPACE}'...")
        client = Client(_PIX2PIX_SPACE, verbose=False)

        # '/generate' signature (8 args):
        #   (Input Image, Edit Instruction, Steps,
        #    'Randomize Seed'|'Fix Seed', Seed,
        #    'Randomize CFG'|'Fix CFG', Text CFG, Image CFG)
        # Returns: (Seed, Text CFG, Image CFG, Edited Image path)
        print("[PIX2PIX] Calling '/generate'...")
        result = client.predict(
            handle_file(img_path),     # Input Image
            prompt,                    # Edit Instruction
            20,                        # Steps
            "Fix Seed",                # deterministic across reruns
            1371,                      # Seed
            "Fix CFG",                 # keep our CFG values
            7.5,                       # Text CFG (how strongly to edit)
            1.5,                       # Image CFG (how much to keep the photo)
            api_name="/generate",
        )
        generated = _result_to_image(result)
        if generated is not None:
            print("[PIX2PIX] Success")
            return generated
        print("[PIX2PIX] returned no usable image")
    except Exception as exc:
        print(f"[PIX2PIX] failed: {type(exc).__name__}: {exc}")
    finally:
        try:
            os.unlink(img_path)
        except OSError:
            pass
    return None


def _try_txt2img_fallback(hf_token: str) -> Image.Image | None:
    """
    Last-resort fallback: pure text-to-image with FLUX.1-schnell.

    Does NOT use the input image — generates a generic eco-city scene.
    """
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": _TXT2IMG_FALLBACK_PROMPT}

    try:
        print("[TXT2IMG] Falling back to FLUX.1-schnell text-to-image...")
        resp = requests.post(
            _TXT2IMG_FALLBACK_URL,
            headers=headers,
            json=payload,
            timeout=HF_REQUEST_TIMEOUT,
        )
        if resp.status_code == 200:
            print("[TXT2IMG] Success (text-to-image fallback)")
            return Image.open(io.BytesIO(resp.content)).convert("RGB")

        if resp.status_code == 503:
            st.warning(
                "Модель загружается на сервер Hugging Face. "
                "Подождите 20-30 секунд и повторите попытку."
            )
            return None

        try:
            detail = resp.json().get("error", resp.text)
        except ValueError:
            detail = resp.text
        print(f"[TXT2IMG] FLUX.1-schnell returned {resp.status_code}: {detail}")
    except requests.exceptions.RequestException as exc:
        print(f"[TXT2IMG] Request failed: {exc}")

    return None


# ---------------------------------------------------------------------------
# Scene protection — semantic segmentation so the road is never turned green.
#
# After the image-to-image model runs we re-paste the ORIGINAL pixels of the
# "functional" classes (road, cars, people…) back over the generated image,
# guided by a Cityscapes semantic segmentation. The model is free to add
# greenery on sidewalks, verges and facades, but the carriageway and vehicles
# are restored from the source photo — so a road never becomes an alley.
# Free, ~2s, via the HF Inference API (needs HF_API_TOKEN).
# ---------------------------------------------------------------------------
_SEG_MODEL = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
_SEG_URL = f"https://router.huggingface.co/hf-inference/models/{_SEG_MODEL}"

# Cityscapes classes whose original pixels must be preserved (restored): the
# driving surface and the dynamic objects on/next to it. Everything else
# (sidewalk, vegetation, terrain, building, wall, fence, sky, pole…) keeps the
# generated greenery.
_PROTECT_LABELS = frozenset({
    "road", "car", "truck", "bus", "motorcycle", "bicycle", "train",
    "person", "rider",
})


def _segment_protect_mask(
    image: Image.Image, hf_token: str, size: tuple[int, int]
) -> np.ndarray | None:
    """
    Ask the Cityscapes segmenter which pixels are road/vehicles/people and
    return a feathered alpha mask (float32 in [0, 1], shape (H, W)) at ``size``
    where 1.0 means "restore the original photo here". None on any failure.
    """
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "image/jpeg"}
    buf = io.BytesIO()
    _normalize(image).save(buf, format="JPEG", quality=92)
    try:
        resp = requests.post(
            _SEG_URL, headers=headers, data=buf.getvalue(), timeout=60
        )
    except requests.exceptions.RequestException as exc:
        print(f"[SEG] request failed: {exc}")
        return None
    if resp.status_code != 200:
        print(f"[SEG] {resp.status_code}: {resp.text[:160]}")
        return None

    try:
        segments = resp.json()
    except ValueError:
        return None
    if not isinstance(segments, list):
        return None

    width, height = size
    acc = np.zeros((height, width), dtype=bool)
    found = False
    for seg in segments:
        if seg.get("label") not in _PROTECT_LABELS:
            continue
        mask_b64 = seg.get("mask")
        if not mask_b64:
            continue
        try:
            mask_img = (
                Image.open(io.BytesIO(base64.b64decode(mask_b64)))
                .convert("L")
                .resize((width, height), Image.NEAREST)
            )
        except Exception:
            continue
        acc |= np.asarray(mask_img) > 127
        found = True

    if not found:
        print("[SEG] no protected classes found in scene")
        return None

    # Feather the edges so the restored road blends into the new greenery.
    alpha_img = Image.fromarray((acc * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=4)
    )
    return np.asarray(alpha_img, dtype=np.float32) / 255.0


def _protect_scene(
    original: Image.Image, generated: Image.Image, hf_token: str | None
) -> Image.Image:
    """
    Composite the original road / vehicles / people back over the generated
    image using the segmentation mask. Degrades gracefully — returns the
    generated image unchanged if there is no token or segmentation fails.
    """
    if not hf_token:
        return generated
    try:
        width, height = generated.size
        alpha = _segment_protect_mask(original, hf_token, (width, height))
        if alpha is None:
            return generated

        orig_arr = np.asarray(
            original.convert("RGB").resize((width, height), Image.LANCZOS),
            dtype=np.float32,
        )
        gen_arr = np.asarray(generated.convert("RGB"), dtype=np.float32)
        a = alpha[..., None]
        blended = gen_arr * (1.0 - a) + orig_arr * a
        print("[SEG] scene protection applied (road/vehicles restored)")
        return Image.fromarray(blended.astype(np.uint8))
    except Exception as exc:
        print(f"[SEG] protection skipped: {type(exc).__name__}: {exc}")
        return generated


def generate_eco_friendly_view(
    image: Image.Image, audit: dict | None = None
) -> Image.Image:
    """
    Generate a "green future" version of the user's photo.

    Pipeline:
      1. The edit instruction is built from the YOLOv8 + OpenCV ``audit``
         (run_eco_audit), grounded in measured greenery / asphalt / cars.
      2. A 3-tier image generation chain runs (best → last resort):
           a. FLUX.1-Kontext-dev — high-quality instruction editing. Needs token.
           b. InstructPix2Pix — anonymous free fallback, lower quality.
           c. FLUX.1-schnell text-to-image — last resort, does NOT use the photo.
      3. For the two image-to-image tiers, semantic segmentation restores the
         original road / vehicles / people so the carriageway never gets
         turned into greenery (see _protect_scene).

    Always returns a PIL Image — on total failure returns the original image
    so the UI never receives None.
    """
    if audit is None:
        audit = _scene_to_audit(image)
    prompt = build_eco_prompt(audit)
    hf_token = _get_secret("HF_API_TOKEN")

    # --- Tier 1: FLUX.1-Kontext (best, needs token) ---
    if hf_token:
        result = _try_flux_kontext(image, prompt, hf_token)
        if result is not None:
            return _protect_scene(image, result, hf_token)

    # --- Tier 2: InstructPix2Pix (anonymous, free) ---
    result = _try_instruct_pix2pix(image, prompt)
    if result is not None:
        return _protect_scene(image, result, hf_token)

    # --- Tier 3: text-to-image fallback (ignores the photo) ---
    if hf_token:
        st.info(
            "⚠️ Img2img модели недоступны (возможно, исчерпана дневная квота "
            "бесплатного GPU). Используется текстовая генерация — результат "
            "не основан на фото."
        )
        result = _try_txt2img_fallback(hf_token)
        if result is not None:
            return result

    st.error(
        "Не удалось сгенерировать изображение. "
        "Попробуйте позже — бесплатный GPU может быть перегружен."
    )
    return image



# ---------------------------------------------------------------------------
# Local Computer Vision pipeline (OpenCV + YOLOv8) configuration.
# ---------------------------------------------------------------------------
# HSV thresholds for the classic colour-segmentation indices.
_GREEN_HSV_LOWER = (35, 40, 40)
_GREEN_HSV_UPPER = (85, 255, 255)
_ASPHALT_HSV_LOWER = (0, 0, 40)
_ASPHALT_HSV_UPPER = (180, 40, 200)

# COCO class ids that count as "vehicles" for the parked-car analysis.
#   2 = car, 3 = motorcycle, 5 = bus, 7 = truck
_VEHICLE_CLASS_IDS = frozenset({2, 3, 5, 7})

_YOLO_WEIGHTS = "yolov8n.pt"

# Lazy singleton — the YOLO model is heavy, so load it once on first use.
_yolo_model = None


def _get_yolo_model():
    """Load (and cache) the YOLOv8 detector on first call."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO

        _yolo_model = YOLO(_YOLO_WEIGHTS)
    return _yolo_model


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




def run_eco_audit(pil_image: Image.Image) -> dict:
    """
    Local Computer Vision eco-audit of a street photo (no external APIs).

    Pipeline:
        1. PIL -> OpenCV BGR numpy array.
        2. BGR -> HSV colour space.
        3. Green View Index   : % of pixels inside the green HSV band.
        4. Asphalt/Concrete   : % of pixels inside the gray/dark HSV band.
        5. YOLOv8 (yolov8n.pt): count car / motorcycle / bus / truck detections.

    Returns a dictionary with the keys consumed by the UI:
        green_view_index, asphalt_coverage, cars_detected,
        urban_heat_risk, critical_flaws, psychological_impact.
    """
    import cv2

    # --- 1) PIL (RGB) -> OpenCV BGR numpy array. ---------------------------
    rgb_array = np.asarray(pil_image.convert("RGB"))
    bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    # --- 2) Convert to HSV for colour-band segmentation. ------------------
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    total_pixels = hsv_image.shape[0] * hsv_image.shape[1]

    # --- 3) Green View Index (percentage of green pixels). ----------------
    green_mask = cv2.inRange(
        hsv_image, np.array(_GREEN_HSV_LOWER), np.array(_GREEN_HSV_UPPER)
    )
    green_view_index = round(
        float(cv2.countNonZero(green_mask)) / total_pixels * 100.0, 1
    )

    # --- 4) Asphalt/Concrete Index (percentage of gray/dark pixels). ------
    asphalt_mask = cv2.inRange(
        hsv_image, np.array(_ASPHALT_HSV_LOWER), np.array(_ASPHALT_HSV_UPPER)
    )
    asphalt_coverage = round(
        float(cv2.countNonZero(asphalt_mask)) / total_pixels * 100.0, 1
    )

    # --- 5) YOLOv8 inference -> count vehicles. ---------------------------
    model = _get_yolo_model()
    results = model(bgr_image, verbose=False)

    cars_detected = 0
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for class_id in boxes.cls.tolist():
            if int(class_id) in _VEHICLE_CLASS_IDS:
                cars_detected += 1

    # --- Derived verdicts. ------------------------------------------------
    urban_heat_risk = "Критический" if asphalt_coverage > 45 else "Низкий"

    critical_flaws: list[str] = []
    if green_view_index < 10:
        critical_flaws.append(
            "Критический недостаток деревьев: полное отсутствие естественной тени."
        )
    if asphalt_coverage > 45:
        critical_flaws.append(
            "Огромные площади раскаленного бетона и асфальта (эффект теплового острова)."
        )
    if cars_detected >= 2:
        critical_flaws.append(
            "Пешеходное пространство агрессивно подавлено припаркованными авто."
        )

    psychological_impact = (
        "Атмосфера вызывает стресс: серая, агрессивная и неприветливая среда."
        if len(critical_flaws) >= 2
        else "Баланс соблюден, но есть потенциал для озеленения."
    )

    return {
        "green_view_index": green_view_index,
        "asphalt_coverage": asphalt_coverage,
        "cars_detected": cars_detected,
        "urban_heat_risk": urban_heat_risk,
        "critical_flaws": critical_flaws,
        "psychological_impact": psychological_impact,
        "engine": "yolov8",
    }


def _colour_only_audit(pil_image: Image.Image) -> dict:
    """
    Pure-numpy eco-audit (no OpenCV / YOLO) used when the ML stack is not
    available. Same shape as run_eco_audit() but cannot detect cars.
    """
    scene = _analyze_scene(pil_image)
    green_view_index = round(scene["greenery"] * 100.0, 1)
    asphalt_coverage = round(scene["hardscape"] * 100.0, 1)

    urban_heat_risk = "Критический" if asphalt_coverage > 45 else "Низкий"
    critical_flaws: list[str] = []
    if green_view_index < 10:
        critical_flaws.append(
            "Критический недостаток деревьев: полное отсутствие естественной тени."
        )
    if asphalt_coverage > 45:
        critical_flaws.append(
            "Огромные площади раскаленного бетона и асфальта (эффект теплового острова)."
        )
    psychological_impact = (
        "Атмосфера вызывает стресс: серая, агрессивная и неприветливая среда."
        if len(critical_flaws) >= 2
        else "Баланс соблюден, но есть потенциал для озеленения."
    )
    return {
        "green_view_index": green_view_index,
        "asphalt_coverage": asphalt_coverage,
        "cars_detected": 0,
        "urban_heat_risk": urban_heat_risk,
        "critical_flaws": critical_flaws,
        "psychological_impact": psychological_impact,
        "engine": "colour-fallback",
    }


def eco_audit_safe(pil_image: Image.Image) -> dict:
    """
    Run the YOLOv8 + OpenCV eco-audit, falling back to a pure-numpy colour
    audit if the ML stack (torch / opencv / ultralytics) is unavailable or
    fails to load. Never raises — always returns an audit dict.
    """
    try:
        return run_eco_audit(pil_image)
    except Exception as exc:
        print(
            f"[AUDIT] YOLO audit unavailable ({type(exc).__name__}: {exc}); "
            "using colour-only fallback."
        )
        return _colour_only_audit(pil_image)
