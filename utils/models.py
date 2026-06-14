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

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageFilter

# ---------------------------------------------------------------------------
# Shared helpers and configuration.
# ---------------------------------------------------------------------------

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

# FLUX.1-Kontext-dev model id (used by the Inference Providers router) and the
# public ZeroGPU Space id (free but quota-limited fallback).
_KONTEXT_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
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
    Build a dynamic, decay-driven image-editing instruction that turns the
    photographed street into a premium, world-class downtown boulevard while
    preserving its structure (camera angle, road layout, building footprints).

    The prompt is composed from the Street Decay Index and the specific weak
    signals measured by the audit (greenery, asphalt, vehicles, darkness,
    drabness, clutter), so the more derelict the street, the more ambitious and
    explicit the transformation — a true "ghetto → luxury" upgrade. Tuned for
    OpenAI gpt-image-* edits and FLUX.1-Kontext alike: imperative edits plus
    hard structure-preservation rules so the model restyles rather than
    inventing a new scene.
    """
    green = float(audit.get("green_view_index", 0.0))       # 0–100, higher better
    asphalt = float(audit.get("asphalt_coverage", 0.0))     # 0–100
    cars = int(audit.get("cars_detected", 0))
    brightness = float(audit.get("brightness", 150.0))      # 0–255
    colorfulness = float(audit.get("colorfulness", 40.0))   # 0–~110
    clutter = float(audit.get("clutter", 8.0))              # 0–100 edge density
    sdi = float(audit.get("decay_index", 50.0))             # 0–100, higher worse

    # --- Ambition scaled to how derelict the place is. -------------------
    # The opener deliberately talks about "this place / this exact spot"
    # rather than a "street" or "boulevard": the photo may be a courtyard, a
    # yard, an alley or a square, and we must NOT reframe it as a road. The
    # job is a realistic clean-up + greening of whatever the place already is,
    # not a redesign into a different kind of space.
    if sdi >= 62:
        opener = (
            "Renovate and beautify this rundown, neglected place into a clean, "
            "well-kept, green version of itself — a believable before→after "
            "improvement that keeps it the EXACT same place and the same kind "
            "of space"
        )
    elif sdi >= 42:
        opener = (
            "Tidy and upgrade this tired, run-down place into a clean, "
            "well-maintained, greener version of itself, keeping it the EXACT "
            "same place and the same kind of space"
        )
    elif sdi >= 24:
        opener = (
            "Refresh this ordinary, grey place into a clean, green and "
            "well-kept version of itself, keeping it the EXACT same place and "
            "the same kind of space"
        )
    else:
        opener = (
            "Gently polish this already decent place into a clean, tidy, "
            "well-landscaped version of itself, keeping it the EXACT same place "
            "and the same kind of space"
        )

    # --- Targeted fixes, switched on by the weak signals. ----------------
    fixes: list[str] = []

    if green < 25:
        fixes.append(
            "add mature trees, hedges, lawns, flower beds and planters wherever "
            "greenery would realistically fit — along the existing edges, verges "
            "and unused bare patches — without covering the existing walkable or "
            "drivable surfaces"
        )
    else:
        fixes.append(
            "enrich the existing greenery with a few extra healthy trees, "
            "trimmed hedges, flower beds and tasteful planters in the spots that "
            "already have plants or bare verges"
        )

    if asphalt > 40:
        fixes.append(
            "repair the cracked, broken paving and clean up the bare ground, "
            "resurfacing the existing ground with tidy paving of the same kind, "
            "keeping the existing surface where it already is"
        )
    else:
        fixes.append(
            "repair and clean the existing paving and tidy the curbs and edges "
            "so everything looks well-maintained"
        )

    if brightness < 120:
        fixes.append(
            "even out the lighting into clear, natural daylight while keeping "
            "the original time of day and weather plausible"
        )
    else:
        fixes.append(
            "keep the natural daylight, just cleaner and clearer"
        )

    if colorfulness < 30:
        fixes.append(
            "clean and repair the existing building facades with fresh, tidy "
            "finishes in natural, harmonious colours, keeping their original "
            "materials and character"
        )
    else:
        fixes.append(
            "clean and repair the existing building facades, keeping their "
            "original materials, colours and character"
        )

    if clutter > 12:
        fixes.append(
            "remove obvious mess — rubbish, junk, debris, peeling posters and "
            "broken signage — for a clean, orderly look, without removing "
            "functional fixtures"
        )

    if cars >= 2:
        fixes.append(
            "tidy the parked cars into neat order in their existing parking "
            "spots, without removing the parking area"
        )

    # Always-on, modest finishing touches.
    fixes.append(
        "add a few tasteful, realistic touches that suit this place: tidy "
        "benches, planters and clean surfaces"
    )

    instruction = (
        opener + ". " + "; ".join(fixes) + ". "
        "STRICT RULES — this is a realistic clean-up, NOT a redesign: do not "
        "change the type of place — a courtyard stays a courtyard, a yard stays "
        "a yard, an alley stays an alley, a road stays a road; never invent a "
        "new road, carriageway, traffic lane, median or promenade that is not "
        "already there. Keep the EXACT same camera angle, perspective, vanishing "
        "point and framing; keep all ground surfaces (roads, paths, parking, "
        "yards) in their original position, shape and purpose; keep every "
        "building in its original location, footprint, height and count — clean "
        "and repair them, do not move, remove or add buildings; do not place "
        "trees, plants or objects on top of drivable or walkable surfaces. The "
        "result must be unmistakably the SAME place from the SAME photo, only "
        "cleaner, greener and well-maintained. Photorealistic, natural, "
        "believable, true to the original scene — not a stylised render."
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


def _try_kontext_router(
    image: Image.Image, prompt: str, hf_token: str
) -> Image.Image | None:
    """
    Primary path — FLUX.1-Kontext-dev image editing via the Hugging Face
    Inference Providers router (``InferenceClient(provider="auto")``).

    This is the reliable high-quality route: unlike the public ZeroGPU Space
    (which only grants ~60s of GPU per day and is almost always exhausted), the
    router dispatches to a hosted provider (fal-ai / replicate / …) and is
    billed against the token's included monthly inference credits. Kontext is
    image-conditioned, so it actually reads the photo and preserves the road,
    buildings and perspective while landscaping the sidewalks.

    Returns None on any failure (e.g. credits exhausted) so the caller can fall
    back to the free tiers.
    """
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        print("[ROUTER] huggingface_hub not installed — pip install huggingface_hub")
        return None

    buf = io.BytesIO()
    _normalize(image).save(buf, format="JPEG", quality=92)
    try:
        print("[ROUTER] FLUX.1-Kontext via Inference Providers (provider=auto)...")
        client = InferenceClient(provider="auto", api_key=hf_token)
        generated = client.image_to_image(
            buf.getvalue(),
            prompt=prompt,
            model=_KONTEXT_MODEL,
        )
        if isinstance(generated, Image.Image):
            print("[ROUTER] Success")
            return generated.convert("RGB")
        print("[ROUTER] returned no usable image")
    except Exception as exc:
        print(f"[ROUTER] failed: {type(exc).__name__}: {exc}")
    return None


def _try_flux_kontext(
    image: Image.Image, prompt: str, hf_token: str
) -> Image.Image | None:
    """
    Secondary path — FLUX.1-Kontext-dev instruction editing via the Gradio
    Client (public ZeroGPU Space).

    Same model as the router path but free; however the official Space only
    grants ~60s of ZeroGPU per day on the free tier, so it is usually quota
    exhausted. Kept as a no-cost fallback for when the router credits run out.
    On quota errors / failures this returns None so the caller can fall back.
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

# Cityscapes classes whose original pixels must be preserved (restored):
# things that must NOT be "greened" or repainted — the driving surface, the
# dynamic objects on/next to it, and the sky (otherwise the model tints the
# sky green). Everything else (sidewalk, terrain, building, wall, fence,
# vegetation, pole…) keeps the generated improvements so it can be greened,
# repainted and tidied up.
_PROTECT_LABELS = frozenset({
    "road", "sky", "car", "truck", "bus", "motorcycle", "bicycle", "train",
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


def _set_engine(name: str) -> None:
    """Record which image engine produced the result, for the UI badge."""
    try:
        st.session_state["last_gen_engine"] = name
    except Exception:
        pass


def generate_eco_friendly_view(
    image: Image.Image, audit: dict | None = None
) -> Image.Image:
    """
    Generate a premium "green future" version of the user's photo.

    Pipeline:
      1. The edit instruction is built from the YOLOv8 + OpenCV ``audit``
         (run_eco_audit), scaled by the Street Decay Index so derelict streets
         get a full luxury rebuild.
      2. Tier 0 — OpenAI image model (gpt-image-*) on the project key, the
         primary high-quality engine when ``OPENAI_API_KEY`` is configured.
      3. Otherwise (or on failure) a 4-tier free chain runs (best → last resort):
           a. FLUX.1-Kontext-dev via the Inference Providers router — reliable,
              needs HF token + inference credits.
           b. FLUX.1-Kontext-dev via the free ZeroGPU Space — usually quota out.
           c. InstructPix2Pix — anonymous free fallback, lower quality.
           d. FLUX.1-schnell text-to-image — last resort, does NOT use the photo.
      4. For the weak image-to-image tier, semantic segmentation restores the
         original road / vehicles / people so the carriageway never gets
         turned into greenery (see _protect_scene).

    Always returns a PIL Image — on total failure returns the original image
    so the UI never receives None.
    """
    if audit is None:
        audit = _scene_to_audit(image)
    prompt = build_eco_prompt(audit)

    # --- Tier 0: OpenAI image model on the project key (primary) ---
    from utils.openai_gen import generate_eco_view_openai, openai_available

    if openai_available():
        result = generate_eco_view_openai(image, prompt)
        if result is not None:
            _set_engine("openai")
            return result

    hf_token = _get_secret("HF_API_TOKEN")

    # --- Tier 1: FLUX.1-Kontext via Inference Providers router (best) ---
    # Kontext is image-conditioned and preserves the road / geometry natively,
    # so no segmentation re-paste is needed (it only adds blending seams here).
    if hf_token:
        result = _try_kontext_router(image, prompt, hf_token)
        if result is not None:
            _set_engine("flux-kontext")
            return result

    # --- Tier 2: FLUX.1-Kontext via free ZeroGPU Space (often quota-limited) ---
    if hf_token:
        result = _try_flux_kontext(image, prompt, hf_token)
        if result is not None:
            _set_engine("flux-kontext")
            return result

    # --- Tier 3: InstructPix2Pix (anonymous, free, weaker) ---
    # This tier does NOT reliably preserve the road, so restore the original
    # carriageway / vehicles / people with the segmentation mask.
    result = _try_instruct_pix2pix(image, prompt)
    if result is not None:
        _set_engine("instruct-pix2pix")
        return _protect_scene(image, result, hf_token)

    # --- Tier 4: text-to-image fallback (ignores the photo) ---
    if hf_token:
        st.info(
            "⚠️ Img2img модели недоступны (возможно, исчерпана дневная квота "
            "бесплатного GPU). Используется текстовая генерация — результат "
            "не основан на фото."
        )
        result = _try_txt2img_fallback(hf_token)
        if result is not None:
            _set_engine("flux-schnell-text")
            return result

    st.error(
        "Не удалось сгенерировать изображение. "
        "Попробуйте позже — бесплатный GPU может быть перегружен."
    )
    _set_engine("none")
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


# ---------------------------------------------------------------------------
# Street Decay Index (SDI) — the "how rough is this street" score.
#
# A composite 0–100 score (higher = more derelict) blended from cheap, fast
# image signals — no extra ML. It drives how ambitious the beautification
# prompt is, so a genuine slum gets a full luxury rebuild while a decent street
# only gets a premium polish. Signals:
#   - greenery    (less green        -> worse)   weight 0.26
#   - asphalt     (more bare paving  -> worse)   weight 0.16
#   - vehicles    (more parked cars  -> worse)   weight 0.10
#   - darkness    (dim / gloomy      -> worse)   weight 0.16
#   - drabness    (low colourfulness -> worse)   weight 0.18
#   - clutter     (busy edges/mess   -> worse)   weight 0.14
# ---------------------------------------------------------------------------
def _image_quality_metrics(rgb_array: np.ndarray) -> dict[str, float]:
    """
    Cheap numpy perceptual metrics from an RGB uint8/float array:
        brightness   : mean luminance, 0–255 (low = gloomy).
        colorfulness : Hasler–Süsstrunk colourfulness, ~0–110 (low = drab grey).
        clutter      : gradient-edge density %, 0–100 (high = busy/messy).
    Works without OpenCV so both audit paths can share it.
    """
    arr = np.asarray(rgb_array, dtype=np.float32)
    red, green, blue = arr[..., 0], arr[..., 1], arr[..., 2]

    # Perceptual brightness (Rec. 601 luma).
    brightness = float(0.299 * red.mean() + 0.587 * green.mean() + 0.114 * blue.mean())

    # Hasler–Süsstrunk colourfulness.
    rg = red - green
    yb = 0.5 * (red + green) - blue
    colorfulness = float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2)
        + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )

    # Edge density as a clutter proxy (numpy gradient magnitude threshold).
    gray = arr.mean(axis=-1)
    gy = np.abs(np.diff(gray, axis=0))
    gx = np.abs(np.diff(gray, axis=1))
    edges = (gy[:, :-1] + gx[:-1, :]) > 36.0
    clutter = float(edges.mean() * 100.0)

    return {
        "brightness": round(brightness, 1),
        "colorfulness": round(colorfulness, 1),
        "clutter": round(clutter, 1),
    }


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def compute_decay_index(
    green: float, asphalt: float, cars: int,
    brightness: float, colorfulness: float, clutter: float,
) -> tuple[float, str, str]:
    """
    Blend the signals into the Street Decay Index.

    Returns (decay_index 0–100, tier_key, human_label_ru).
    """
    bad_green = _clamp01((30.0 - green) / 30.0)
    bad_asphalt = _clamp01((asphalt - 25.0) / 55.0)
    bad_cars = _clamp01(cars / 8.0)
    bad_dark = _clamp01((150.0 - brightness) / 110.0)
    bad_drab = _clamp01((40.0 - colorfulness) / 40.0)
    bad_clutter = _clamp01((clutter - 7.0) / 18.0)

    sdi = 100.0 * (
        0.26 * bad_green
        + 0.16 * bad_asphalt
        + 0.10 * bad_cars
        + 0.16 * bad_dark
        + 0.18 * bad_drab
        + 0.14 * bad_clutter
    )
    sdi = round(sdi, 1)

    if sdi >= 62:
        return sdi, "slum", "Трущобы — нужен полный люкс-ребилд"
    if sdi >= 42:
        return sdi, "rundown", "Запущенная улица"
    if sdi >= 24:
        return sdi, "plain", "Обычная серая улица"
    return sdi, "decent", "Приличная улица"


def _decay_flaws(
    green: float, asphalt: float, cars: int,
    brightness: float, colorfulness: float, clutter: float,
) -> list[str]:
    """Human-readable (RU) flaw list derived from the measured signals."""
    flaws: list[str] = []
    if green < 12:
        flaws.append("Почти полное отсутствие зелени и деревьев — нет тени и воздуха.")
    elif green < 25:
        flaws.append("Мало озеленения, район выглядит голым и серым.")
    if asphalt > 45:
        flaws.append("Огромные площади асфальта и бетона (эффект теплового острова).")
    if cars >= 2:
        flaws.append("Пространство задавлено припаркованными машинами.")
    if brightness < 110:
        flaws.append("Темная, мрачная, плохо освещённая среда.")
    if colorfulness < 28:
        flaws.append("Унылая серая палитра без цвета и характера.")
    if clutter > 14:
        flaws.append("Визуальный мусор: провода, хаотичные вывески, грязь.")
    return flaws


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

    # --- 6) Perceptual metrics + Street Decay Index. ----------------------
    metrics = _image_quality_metrics(rgb_array)
    decay_index, decay_tier, decay_label = compute_decay_index(
        green_view_index, asphalt_coverage, cars_detected,
        metrics["brightness"], metrics["colorfulness"], metrics["clutter"],
    )

    # --- Derived verdicts. ------------------------------------------------
    urban_heat_risk = "Критический" if asphalt_coverage > 45 else "Низкий"

    critical_flaws = _decay_flaws(
        green_view_index, asphalt_coverage, cars_detected,
        metrics["brightness"], metrics["colorfulness"], metrics["clutter"],
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
        "brightness": metrics["brightness"],
        "colorfulness": metrics["colorfulness"],
        "clutter": metrics["clutter"],
        "decay_index": decay_index,
        "decay_tier": decay_tier,
        "decay_label": decay_label,
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
    normalized = _normalize(pil_image)
    scene = _analyze_scene(pil_image)
    green_view_index = round(scene["greenery"] * 100.0, 1)
    asphalt_coverage = round(scene["hardscape"] * 100.0, 1)

    metrics = _image_quality_metrics(np.asarray(normalized))
    decay_index, decay_tier, decay_label = compute_decay_index(
        green_view_index, asphalt_coverage, 0,
        metrics["brightness"], metrics["colorfulness"], metrics["clutter"],
    )

    urban_heat_risk = "Критический" if asphalt_coverage > 45 else "Низкий"
    critical_flaws = _decay_flaws(
        green_view_index, asphalt_coverage, 0,
        metrics["brightness"], metrics["colorfulness"], metrics["clutter"],
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
        "brightness": metrics["brightness"],
        "colorfulness": metrics["colorfulness"],
        "clutter": metrics["clutter"],
        "decay_index": decay_index,
        "decay_tier": decay_tier,
        "decay_label": decay_label,
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
