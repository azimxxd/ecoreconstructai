"""
EcoReconstruct AI — Lightweight JSON-based local database.

Stores citizen submissions (eco-audit reports) in `projects_db.json`
next to the project root. Designed for Streamlit's rerun model:

- Atomic writes  : data is written to a temp file and then os.replace()'d,
                   so a crash mid-write never corrupts the database.
- Process lock   : a threading.Lock guards read-modify-write cycles
                   (Streamlit serves sessions from threads of one process).

Schema of a single item:
{
    "id":              str (uuid4),
    "address":         str,
    "green_index":     float (0.0 - 1.0),
    "image_original":  str (base64-encoded PNG),
    "image_generated": str (base64-encoded PNG),
    "likes":           int,
    "timestamp":       str (ISO 8601),
    "ai_problems":         list[str] (Claude eco-audit, may be empty),
    "ai_recommendations":  list[str] (Claude eco-audit, may be empty),
    "ai_priority":         str ("высокий"/"средний"/"низкий" or "")
}
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Database lives next to the project root, independent of the CWD quirks.
DB_PATH = Path(__file__).resolve().parent.parent / "projects_db.json"

# Guards concurrent read-modify-write cycles within the Streamlit process.
_db_lock = threading.Lock()


def _ensure_db_exists() -> None:
    """Create an empty database file on first run so the app never 404s."""
    if not DB_PATH.exists():
        _atomic_write([])


def _atomic_write(items: list[dict[str, Any]]) -> None:
    """Write the full item list atomically (temp file + os.replace)."""
    fd, tmp_path = tempfile.mkstemp(
        dir=DB_PATH.parent, prefix=".projects_db_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(items, tmp_file, ensure_ascii=False, indent=2)
        # os.replace is atomic on both POSIX and Windows, but cloud-sync
        # clients (OneDrive) and antivirus briefly lock files on Windows —
        # retry with backoff before giving up.
        for attempt in range(6):
            try:
                os.replace(tmp_path, DB_PATH)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.15 * (attempt + 1))
    except BaseException:
        # Never leave orphaned temp files behind on failure.
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load_db() -> list[dict[str, Any]]:
    """Return all submissions, newest first. Self-heals a missing/corrupt file."""
    with _db_lock:
        _ensure_db_exists()
        try:
            raw = DB_PATH.read_text(encoding="utf-8")
            items = json.loads(raw) if raw.strip() else []
        except (json.JSONDecodeError, OSError):
            # Corrupt or unreadable file — reset rather than crash the app.
            items = []
            _atomic_write(items)
    return sorted(items, key=lambda item: item.get("timestamp", ""), reverse=True)


def save_item(item: dict[str, Any]) -> dict[str, Any]:
    """
    Persist a new submission. Fills in id / likes / timestamp automatically
    if the caller did not provide them. Returns the stored item.
    """
    stored_item = {
        "id": item.get("id") or str(uuid.uuid4()),
        "address": item.get("address", "Без адреса"),
        "green_index": float(item.get("green_index", 0.0)),
        "image_original": item.get("image_original", ""),
        "image_generated": item.get("image_generated", ""),
        "likes": int(item.get("likes", 0)),
        "timestamp": item.get("timestamp")
        or datetime.now(timezone.utc).isoformat(),
        "ai_problems": list(item.get("ai_problems", [])),
        "ai_recommendations": list(item.get("ai_recommendations", [])),
        "ai_priority": str(item.get("ai_priority", "")),
    }

    with _db_lock:
        _ensure_db_exists()
        try:
            raw = DB_PATH.read_text(encoding="utf-8")
            items = json.loads(raw) if raw.strip() else []
        except (json.JSONDecodeError, OSError):
            items = []
        items.append(stored_item)
        _atomic_write(items)

    return stored_item


def add_like(item_id: str) -> int:
    """
    Increment the like counter for one submission (read-modify-write under
    the lock, then an atomic file replace). Returns the new like count,
    or -1 if the item was not found.
    """
    with _db_lock:
        _ensure_db_exists()
        try:
            raw = DB_PATH.read_text(encoding="utf-8")
            items = json.loads(raw) if raw.strip() else []
        except (json.JSONDecodeError, OSError):
            items = []

        new_count = -1
        for item in items:
            if item.get("id") == item_id:
                item["likes"] = int(item.get("likes", 0)) + 1
                new_count = item["likes"]
                break

        if new_count != -1:
            _atomic_write(items)

    return new_count
