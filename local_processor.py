#!/usr/bin/env python3
# =============================================================================
# local_processor.py  —  v1.0.0
# Unified local processing layer for the Processor Pipeline.
#
# Consolidates what was split across l_process.py and the DB-access
# portions of ai_process.py into one authoritative module.
#
# Responsibilities
# ----------------
#   • SQLite schema ownership  — single source of truth, versioned
#   • Session lifecycle        — create / resume / finalize
#   • RAW discovery            — .NEF and .ARW, Windows paths
#   • ExifTool metadata        — batch JSON read
#   • Burst grouping           — gap-based, per camera body
#   • Preview extraction       — JpgFromRaw / PreviewImage, temp-safe
#   • Checksum + verification  — sha256 / md5, spot-check with backoff
#   • AI column migration      — additive ALTER TABLE, idempotent
#   • AI result writes         — scores, captions, edge maps
#   • Engine interface         — BaseEngine ABC both engines implement
#   • Pause / cancel protocol  — consistent: cancel checked first
#   • Progress callback        — shared phase-name constants
#   • Windows-only             — no xdg-open, no udisksctl, no MPS
#
# NOT in this file
# ----------------
#   • Torch / transformers / controlnet_aux  (ai_processor.py owns those)
#   • HuggingFace network probing            (ai_processor.py owns that)
#   • Any GUI code
#
# Schema versioning
# -----------------
#   PRAGMA user_version is used as a monotone integer.
#   Each migration function checks the current version before running.
#   Current target version: 3
#     v0 → v1  initial schema  (sessions, files, bursts, previews)
#     v1 → v2  AI columns on previews + sessions
#     v2 → v3  indexes for AI columns
#
# Windows-only notes
# ------------------
#   • CREATE_NO_WINDOW flag applied to every subprocess call
#   • Path.resolve() used throughout — handles drive-letter UNC paths
#   • Safe-eject stub present but raises NotImplementedError
#     (will be wired to a Windows-specific call in a later pass)
#
# Standalone usage
# ----------------
#   python local_processor.py D:\DCIM\100NZ30
#   python local_processor.py D:\DCIM --resume-session-id 3
#   python local_processor.py --list-sessions
#   python local_processor.py --reset-ai-phase scoring --session-id 3
# =============================================================================
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterator, List, Optional,
    Sequence, Set, Tuple,
)

# ═══════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR           = Path(__file__).resolve().parent
DEFAULT_DB_PATH      = SCRIPT_DIR / "ingest.db"
DEFAULT_PREVIEW_ROOT = SCRIPT_DIR / "previews"
MODEL_PT_PATH        = SCRIPT_DIR / "model.pt"

RAW_EXTENSIONS: frozenset[str] = frozenset({".nef", ".arw"})

SUPPORTED_HASHES: frozenset[str] = frozenset({"sha256", "md5"})
DEFAULT_HASH_ALGO    = "sha256"

BURST_GAP_MS         = 200          # consecutive frames within this → same burst
DEFAULT_VERIFY_RATIO = 0.15         # fraction of previews spot-checked
VERIFY_RATIO_STEP    = 0.05         # widening step on mismatch
MAX_VERIFY_RATIO     = 0.50
MIN_PREVIEW_BYTES    = 2048         # below this → treat as extraction failure
TEMP_SUFFIX          = ".part"      # in-flight write suffix
MANIFEST_SUFFIX      = ".json"      # sidecar integrity manifest

EDGE_MAP_ENCODING    = "hex"        # "hex" | "base64"
EDGE_MAP_QUALITY     = 80           # WebP quality for stored edge maps
CAPTION_MODE         = "<MORE_DETAILED_CAPTION>"

# ── AI model identifiers (used by ai_processor.py, declared here so GUI
#    and DB layer can reference them without importing torch) ────────────
FLORENCE_MODEL_ID    = "MiaoshouAI/Florence-2-large-PromptGen-v2.0"
CLIP_MODEL_ID        = "openai/clip-vit-base-patch32"
HED_MODEL_ID         = "lllyasviel/Annotators"

SCORE_CATEGORIES: Tuple[str, ...] = (
    "overall", "quality", "composition",
    "lighting", "color", "dof", "content",
)

# ── Sibling scripts the GUI checks for existence ─────────────────────────
SIBLING_SCRIPTS: Tuple[str, ...] = (
    "local_processor.py",
    "ai_process.py",
    "processor_gui.py",
    "view_gui.py",
)

# ── Progress phase name constants ─────────────────────────────────────────
#    Both engines emit these exact strings; the GUI maps them to indicators.
class Phase:
    SCAN        = "scan"
    HASH        = "hash"
    METADATA    = "metadata"
    BURSTS      = "bursts"
    PREVIEW     = "preview"
    SCORING     = "scoring"
    CAPTIONING  = "captioning"
    EDGE_MAPS   = "edge_maps"

# ── Schema version ────────────────────────────────────────────────────────
DB_VERSION_TARGET = 3

# ═══════════════════════════════════════════════════════════════════════════
# WINDOWS SUBPROCESS HELPER
# ═══════════════════════════════════════════════════════════════════════════

def _win_flags() -> dict:
    """CREATE_NO_WINDOW for every subprocess call. Windows-only build."""
    return {"creationflags": 0x08000000}   # CREATE_NO_WINDOW


def _run(cmd: Sequence[str], timeout: int = 30) -> Optional[str]:
    """Run a command, return stdout string on success, None on failure."""
    try:
        r = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            timeout=timeout,
            **_win_flags(),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def _run_ok(cmd: Sequence[str], timeout: int = 20) -> bool:
    """Run a command, return True iff exit code is 0."""
    try:
        r = subprocess.run(
            list(cmd),
            capture_output=True,
            timeout=timeout,
            **_win_flags(),
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


# ═══════════════════════════════════════════════════════════════════════════
# PURE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _slug(value: str, max_len: int = 80) -> str:
    """Make a filesystem-safe slug from an arbitrary string."""
    out = "".join(
        c if (c.isalnum() or c in "-_.") else "_"
        for c in value
    ).strip("_")
    return (out or "file")[:max_len]


def _safe_relpath(path: Path) -> str:
    """Return POSIX relative path from SCRIPT_DIR, or absolute fallback."""
    try:
        return path.resolve().relative_to(SCRIPT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_stored(stored: str) -> Path:
    """Stored paths may be relative (to SCRIPT_DIR) or absolute."""
    p = Path(stored)
    return p if p.is_absolute() else SCRIPT_DIR / p


def checksum_file(path: Path, algorithm: str = DEFAULT_HASH_ALGO) -> str:
    if algorithm not in SUPPORTED_HASHES:
        raise ValueError(f"Unsupported hash: {algorithm!r}")
    h = hashlib.new(algorithm)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_datetime(value: Any) -> Optional[str]:
    if isinstance(value, list):
        value = value[0] if value else None
    if not value:
        return None
    s = str(value).strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).isoformat()
        except ValueError:
            continue
    return s or None


def _parse_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace("mm", "").replace("m", "").strip())
    except ValueError:
        return None


def _parse_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _format_shutter(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f <= 0:
        return str(value)
    if f >= 1:
        return f"{f:.1f}s"
    frac = Fraction(f).limit_denominator(16000)
    return f"{frac.numerator}/{frac.denominator}"


def _normalize_meta(row: dict) -> Dict[str, Any]:
    make  = str(row.get("Make")  or "").strip()
    model = str(row.get("Model") or "").strip()
    return {
        "capture_time":    _parse_datetime(row.get("DateTimeOriginal")),
        "shutter_speed":   _format_shutter(
            row.get("ExposureTime") or row.get("ShutterSpeed")),
        "iso":             _parse_int(row.get("ISO")),
        "aperture":        _parse_float(
            row.get("FNumber") or row.get("Aperture")),
        "focal_length_mm": _parse_float(row.get("FocalLength")),
        "focus_distance_m": _parse_float(
            row.get("FocusDistance")
            or row.get("SubjectDistance")
            or row.get("ApproximateFocusDistance")),
        "camera_make":     make,
        "camera_model":    model,
        "orientation":     _parse_int(row.get("Orientation")) or 1,
        "shooting_mode":   str(
            row.get("ShootingMode")
            or row.get("ReleaseMode")
            or row.get("DriveMode")
            or row.get("ReleaseMode2")
            or ""
        ).strip(),
    }


def _preview_manifest_path(preview_path: Path) -> Path:
    return preview_path.with_suffix(preview_path.suffix + MANIFEST_SUFFIX)


def _write_manifest(
    preview_path: Path,
    source_hash: str,
    preview_hash: str,
    raw_path: Path,
    algo: str,
) -> None:
    payload = {
        "schema":       1,
        "source_hash":  source_hash,
        "preview_hash": preview_hash,
        "hash_algo":    algo,
        "source_path":  str(raw_path),
        "preview_path": str(preview_path),
        "updated_at":   _now(),
    }
    manifest = _preview_manifest_path(preview_path)
    tmp = manifest.with_suffix(manifest.suffix + TEMP_SUFFIX)
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(manifest)


def _read_manifest(preview_path: Path) -> Optional[dict]:
    try:
        return json.loads(
            _preview_manifest_path(preview_path)
            .read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None


def _bucket_size(w: int, h: int) -> Tuple[int, int]:
    """Snap (w, h) to the nearest standard aspect-ratio bucket."""
    ratio = w / max(h, 1)
    candidates = [
        (768, 512), (512, 768), (512, 832),
        (576, 1024), (512, 512),
    ]
    return min(candidates, key=lambda s: abs(s[0] / s[1] - ratio))


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FileRecord:
    file_path:        str
    file_name:        str             = ""
    file_ext:         str             = ""
    file_size:        int             = 0
    camera_make:      str             = ""
    camera_model:     str             = ""
    orientation:      int             = 1
    capture_time:     Optional[str]   = None
    shutter_speed:    str             = ""
    iso:              Optional[int]   = None
    aperture:         Optional[float] = None
    focal_length_mm:  Optional[float] = None
    focus_distance_m: Optional[float] = None
    shooting_mode:    str             = ""
    source_hash:      str             = ""
    source_hash_algo: str             = DEFAULT_HASH_ALGO
    file_id:          Optional[int]   = None
    burst_id:         Optional[int]   = None


@dataclass
class BurstRecord:
    burst_index:  int
    frames:       List[FileRecord] = field(default_factory=list)
    start_time:   Optional[str]    = None
    end_time:     Optional[str]    = None
    camera_make:  str              = ""
    camera_model: str              = ""
    burst_id:     Optional[int]    = None


@dataclass
class PreviewRecord:
    file_id:          int
    preview_path:     str
    source_hash:      str
    source_hash_algo: str           = DEFAULT_HASH_ALGO
    preview_hash:     str           = ""
    preview_hash_algo:str           = DEFAULT_HASH_ALGO
    preview_verified: int           = 0
    preview_status:   str           = "pending"
    preview_id:       Optional[int] = None


@dataclass
class PreviewRow:
    """Lightweight view of a previews row, used by AI phase queries."""
    preview_id:   int
    file_id:      int
    preview_path: str
    scored:       int = 0
    captioned:    int = 0
    edge_mapped:  int = 0

    @property
    def abs_path(self) -> Path:
        return _resolve_stored(self.preview_path)

    @property
    def exists(self) -> bool:
        p = self.abs_path
        return p.exists() and p.stat().st_size >= MIN_PREVIEW_BYTES


@dataclass
class ScorePayload:
    preview_id: int
    scores:     Dict[str, float]


@dataclass
class CaptionPayload:
    preview_id: int
    caption:    str


@dataclass
class EdgePayload:
    preview_id: int
    data:       str
    encoding:   str = EDGE_MAP_ENCODING


@dataclass
class SessionSummary:
    """Returned by both engines' run() — shared result schema."""
    status:          str             # "completed" | "interrupted" | "error"
    session_id:      int
    session_name:    str
    source_folder:   str
    file_count:      int             = 0
    burst_count:     int             = 0
    preview_count:   int             = 0
    verified_count:  int             = 0
    mismatch_count:  int             = 0
    scored:          int             = 0
    captioned:       int             = 0
    edge_mapped:     int             = 0
    elapsed_s:       float           = 0.0
    device:          str             = "cpu"
    error:           Optional[str]   = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status":         self.status,
            "session_id":     self.session_id,
            "session_name":   self.session_name,
            "source_folder":  self.source_folder,
            "file_count":     self.file_count,
            "burst_count":    self.burst_count,
            "preview_count":  self.preview_count,
            "verified_count": self.verified_count,
            "mismatch_count": self.mismatch_count,
            "scored":         self.scored,
            "captioned":      self.captioned,
            "edge_mapped":    self.edge_mapped,
            "elapsed_s":      self.elapsed_s,
            "device":         self.device,
            "error":          self.error,
        }


# ═══════════════════════════════════════════════════════════════════════════
# BASE ENGINE PROTOCOL
# Both LocalProcessorEngine and AIProcessorEngine implement this.
# The GUI only ever calls methods defined here.
# ═══════════════════════════════════════════════════════════════════════════

class BaseEngine(ABC):
    """
    Minimal interface contract between any engine and the GUI.

    Implementors MUST:
      • Accept  cancel: threading.Event  in __init__
      • Set     self.cancel  to that event
      • Implement pause() / resume() using an internal threading.Event
      • Check cancel BEFORE blocking on pause  (avoids the race in audit #5)
      • Implement is_paused as a property
      • Have run() return SessionSummary
    """

    @abstractmethod
    def run(self, **kwargs) -> SessionSummary:
        ...

    @abstractmethod
    def pause(self) -> None:
        ...

    @abstractmethod
    def resume(self) -> None:
        ...

    @property
    @abstractmethod
    def is_paused(self) -> bool:
        ...

    def _check(self) -> None:
        """
        Standard pause/cancel checkpoint.

        Order is deliberate:
          1. Cancel first — if the user requests stop, honour it
             even if pause is also set. (Fixes audit #5.)
          2. Pause second — block here until resumed.
        """
        if self.cancel.is_set():          # type: ignore[attr-defined]
            raise InterruptedError("Cancelled by user")
        if not self._pause.is_set():      # type: ignore[attr-defined]
            self._pause.wait()
        # Re-check cancel after unblocking from pause
        if self.cancel.is_set():          # type: ignore[attr-defined]
            raise InterruptedError("Cancelled by user")


# ═══════════════════════════════════════════════════════════════════════════
# DATABASE — SINGLE OWNER
# ═══════════════════════════════════════════════════════════════════════════

# ── DDL ──────────────────────────────────────────────────────────────────

_DDL_V1 = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_name    TEXT    NOT NULL,
    process_date    TEXT    NOT NULL,
    source_folder   TEXT    NOT NULL,
    checksum_algo   TEXT    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'in_progress',
    file_count      INTEGER NOT NULL DEFAULT 0,
    burst_count     INTEGER NOT NULL DEFAULT 0,
    preview_count   INTEGER NOT NULL DEFAULT 0,
    verified_count  INTEGER NOT NULL DEFAULT 0,
    mismatch_count  INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS bursts (
    burst_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER NOT NULL,
    burst_index  INTEGER NOT NULL,
    frame_count  INTEGER NOT NULL,
    start_time   TEXT,
    end_time     TEXT,
    camera_make  TEXT,
    camera_model TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS files (
    file_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        INTEGER NOT NULL,
    burst_id          INTEGER,
    file_path         TEXT    NOT NULL UNIQUE,
    file_name         TEXT,
    file_ext          TEXT,
    file_size_bytes   INTEGER,
    camera_make       TEXT,
    camera_model      TEXT,
    orientation       INTEGER DEFAULT 1,
    capture_time      TEXT,
    shutter_speed     TEXT,
    iso               INTEGER,
    aperture          REAL,
    focal_length_mm   REAL,
    focus_distance_m  REAL,
    shooting_mode     TEXT,
    source_hash       TEXT    NOT NULL,
    source_hash_algo  TEXT    NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (burst_id)   REFERENCES bursts(burst_id)
);

CREATE TABLE IF NOT EXISTS previews (
    preview_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id           INTEGER NOT NULL UNIQUE,
    preview_path      TEXT    NOT NULL,
    source_hash       TEXT    NOT NULL,
    source_hash_algo  TEXT    NOT NULL,
    preview_hash      TEXT,
    preview_hash_algo TEXT,
    preview_verified  INTEGER NOT NULL DEFAULT 0,
    preview_status    TEXT    NOT NULL DEFAULT 'pending',
    updated_at        TEXT    NOT NULL,
    FOREIGN KEY (file_id) REFERENCES files(file_id)
);

CREATE INDEX IF NOT EXISTS idx_files_session    ON files(session_id);
CREATE INDEX IF NOT EXISTS idx_files_burst      ON files(burst_id);
CREATE INDEX IF NOT EXISTS idx_files_path       ON files(file_path);
CREATE INDEX IF NOT EXISTS idx_previews_file    ON previews(file_id);
CREATE INDEX IF NOT EXISTS idx_previews_status  ON previews(preview_status);
CREATE INDEX IF NOT EXISTS idx_bursts_session   ON bursts(session_id);
"""

# AI columns — added in v2 migration, idempotent via ALTER TABLE guard
_AI_PREVIEW_COLS: Dict[str, str] = {
    "scored":             "INTEGER NOT NULL DEFAULT 0",
    "score_overall":      "REAL",
    "score_quality":      "REAL",
    "score_composition":  "REAL",
    "score_lighting":     "REAL",
    "score_color":        "REAL",
    "score_dof":          "REAL",
    "score_content":      "REAL",
    "captioned":          "INTEGER NOT NULL DEFAULT 0",
    "caption":            "TEXT",
    "edge_mapped":        "INTEGER NOT NULL DEFAULT 0",
    "edge_map_data":      "TEXT",
    "edge_map_encoding":  "TEXT DEFAULT 'hex'",
    "ai_updated_at":      "TEXT",
}

_AI_SESSION_COLS: Dict[str, str] = {
    "ai_status":     "TEXT DEFAULT 'pending'",
    "ai_updated_at": "TEXT",
}

_DDL_V3_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_previews_scored    ON previews(scored);
CREATE INDEX IF NOT EXISTS idx_previews_captioned ON previews(captioned);
CREATE INDEX IF NOT EXISTS idx_previews_edge      ON previews(edge_mapped);
"""


class ProcessorDB:
    """
    Single authoritative SQLite wrapper for the entire pipeline.

    Thread-safety
    -------------
    One threading.Lock serialises all writes. Reads (SELECT) do not
    acquire the lock — WAL mode allows concurrent readers.

    Schema versioning
    -----------------
    PRAGMA user_version tracks which migrations have run.
    Migrations are additive and idempotent.

    GUI access contract
    -------------------
    The GUI must never touch self.conn or self._lock directly.
    All access goes through the public methods below.
    If a needed operation is missing, add a method here.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            str(self.path), check_same_thread=False
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._lock = threading.Lock()
        self._migrate()

    # ── migrations ────────────────────────────────────────────────────

    def _user_version(self) -> int:
        return self.conn.execute("PRAGMA user_version").fetchone()[0]

    def _set_user_version(self, v: int) -> None:
        # user_version doesn't support ? binding — safe: v is always int
        self.conn.execute(f"PRAGMA user_version = {v}")

    def _migrate(self) -> None:
        v = self._user_version()

        if v < 1:
            self.conn.executescript(_DDL_V1)
            self._set_user_version(1)
            self.conn.commit()
            v = 1

        if v < 2:
            self._add_ai_columns()
            self._set_user_version(2)
            self.conn.commit()
            v = 2

        if v < 3:
            with contextlib.suppress(sqlite3.OperationalError):
                self.conn.executescript(_DDL_V3_INDEXES)
            self._set_user_version(3)
            self.conn.commit()

    def _add_ai_columns(self) -> None:
        existing_p = self._column_names("previews")
        existing_s = self._column_names("sessions")
        for col, typedef in _AI_PREVIEW_COLS.items():
            if col not in existing_p:
                with contextlib.suppress(sqlite3.OperationalError):
                    self.conn.execute(
                        f"ALTER TABLE previews ADD COLUMN {col} {typedef}"
                    )
        for col, typedef in _AI_SESSION_COLS.items():
            if col not in existing_s:
                with contextlib.suppress(sqlite3.OperationalError):
                    self.conn.execute(
                        f"ALTER TABLE sessions ADD COLUMN {col} {typedef}"
                    )

    def _column_names(self, table: str) -> Set[str]:
        rows = self.conn.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()
        return {r[1] for r in rows}

    # ── sessions ──────────────────────────────────────────────────────

    def get_or_create_session(
        self,
        session_name: str,
        source_folder: str,
        checksum_algo: str,
    ) -> Tuple[int, bool]:
        """
        Return (session_id, is_new).
        Resumes the most recent non-completed session for this folder.
        """
        now = _now()
        with self._lock:
            row = self.conn.execute(
                "SELECT session_id FROM sessions "
                "WHERE source_folder=? AND status!='completed' "
                "ORDER BY session_id DESC LIMIT 1",
                (source_folder,),
            ).fetchone()
            if row:
                self.conn.execute(
                    "UPDATE sessions "
                    "SET updated_at=?, checksum_algo=? "
                    "WHERE session_id=?",
                    (now, checksum_algo, row[0]),
                )
                self.conn.commit()
                return int(row[0]), False

            cur = self.conn.execute(
                "INSERT INTO sessions"
                "(session_name, process_date, source_folder,"
                " checksum_algo, status, created_at, updated_at)"
                " VALUES (?,?,?,?,'in_progress',?,?)",
                (session_name, now, source_folder,
                 checksum_algo, now, now),
            )
            self.conn.commit()
            return int(cur.lastrowid), True

    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        return dict(row) if row else None

    def list_sessions(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM sessions ORDER BY session_id DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def update_session_counts(self, session_id: int) -> None:
        with self._lock:
            fc = self.conn.execute(
                "SELECT COUNT(*) FROM files WHERE session_id=?",
                (session_id,),
            ).fetchone()[0]
            bc = self.conn.execute(
                "SELECT COUNT(*) FROM bursts WHERE session_id=?",
                (session_id,),
            ).fetchone()[0]
            pc = self.conn.execute(
                "SELECT COUNT(*) FROM previews p "
                "JOIN files f ON p.file_id=f.file_id "
                "WHERE f.session_id=?",
                (session_id,),
            ).fetchone()[0]
            vc = self.conn.execute(
                "SELECT COUNT(*) FROM previews p "
                "JOIN files f ON p.file_id=f.file_id "
                "WHERE f.session_id=? AND p.preview_verified=1",
                (session_id,),
            ).fetchone()[0]
            mc = self.conn.execute(
                "SELECT COUNT(*) FROM previews p "
                "JOIN files f ON p.file_id=f.file_id "
                "WHERE f.session_id=? AND p.preview_status='mismatch'",
                (session_id,),
            ).fetchone()[0]
            self.conn.execute(
                "UPDATE sessions SET "
                "file_count=?, burst_count=?, preview_count=?,"
                "verified_count=?, mismatch_count=?, updated_at=? "
                "WHERE session_id=?",
                (fc, bc, pc, vc, mc, _now(), session_id),
            )
            self.conn.commit()

    def finalize_session(
        self,
        session_id: int,
        status: str = "completed",
    ) -> None:
        self.update_session_counts(session_id)
        with self._lock:
            self.conn.execute(
                "UPDATE sessions SET status=?, updated_at=? "
                "WHERE session_id=?",
                (status, _now(), session_id),
            )
            self.conn.commit()

    def set_ai_status(self, session_id: int, status: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE sessions SET ai_status=?, ai_updated_at=? "
                "WHERE session_id=?",
                (status, _now(), session_id),
            )
            self.conn.commit()

    # ── files ─────────────────────────────────────────────────────────

    def upsert_file(
        self,
        session_id: int,
        record: FileRecord,
    ) -> int:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO files (
                    session_id, burst_id, file_path, file_name,
                    file_ext, file_size_bytes, camera_make,
                    camera_model, orientation, capture_time,
                    shutter_speed, iso, aperture, focal_length_mm,
                    focus_distance_m, shooting_mode,
                    source_hash, source_hash_algo
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(file_path) DO UPDATE SET
                    session_id       = excluded.session_id,
                    burst_id         = excluded.burst_id,
                    file_name        = excluded.file_name,
                    file_ext         = excluded.file_ext,
                    file_size_bytes  = excluded.file_size_bytes,
                    camera_make      = excluded.camera_make,
                    camera_model     = excluded.camera_model,
                    orientation      = excluded.orientation,
                    capture_time     = excluded.capture_time,
                    shutter_speed    = excluded.shutter_speed,
                    iso              = excluded.iso,
                    aperture         = excluded.aperture,
                    focal_length_mm  = excluded.focal_length_mm,
                    focus_distance_m = excluded.focus_distance_m,
                    shooting_mode    = excluded.shooting_mode,
                    source_hash      = excluded.source_hash,
                    source_hash_algo = excluded.source_hash_algo
                """,
                (
                    session_id,
                    record.burst_id,
                    record.file_path,
                    record.file_name,
                    record.file_ext,
                    record.file_size,
                    record.camera_make,
                    record.camera_model,
                    record.orientation,
                    record.capture_time,
                    record.shutter_speed,
                    record.iso,
                    record.aperture,
                    record.focal_length_mm,
                    record.focus_distance_m,
                    record.shooting_mode,
                    record.source_hash,
                    record.source_hash_algo,
                ),
            )
            self.conn.commit()
            row = self.conn.execute(
                "SELECT file_id FROM files WHERE file_path=?",
                (record.file_path,),
            ).fetchone()
            return int(row[0])

    def get_files(self, session_id: int) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM files "
            "WHERE session_id=? "
            "ORDER BY capture_time, file_name",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def existing_file_paths(self, session_id: int) -> Set[str]:
        rows = self.conn.execute(
            "SELECT file_path FROM files WHERE session_id=?",
            (session_id,),
        ).fetchall()
        return {r[0] for r in rows}

    # ── bursts ────────────────────────────────────────────────────────

    def insert_burst(
        self,
        session_id: int,
        burst: BurstRecord,
    ) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO bursts "
                "(session_id, burst_index, frame_count,"
                " start_time, end_time, camera_make, camera_model)"
                " VALUES (?,?,?,?,?,?,?)",
                (
                    session_id,
                    burst.burst_index,
                    len(burst.frames),
                    burst.start_time,
                    burst.end_time,
                    burst.camera_make,
                    burst.camera_model,
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def link_burst(self, file_id: int, burst_id: int) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE files SET burst_id=? WHERE file_id=?",
                (burst_id, file_id),
            )
            self.conn.commit()

    # ── previews ──────────────────────────────────────────────────────

    def upsert_preview(self, record: PreviewRecord) -> int:
        now = _now()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO previews (
                    file_id, preview_path, source_hash,
                    source_hash_algo, preview_hash,
                    preview_hash_algo, preview_verified,
                    preview_status, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(file_id) DO UPDATE SET
                    preview_path      = excluded.preview_path,
                    source_hash       = excluded.source_hash,
                    source_hash_algo  = excluded.source_hash_algo,
                    preview_hash      = excluded.preview_hash,
                    preview_hash_algo = excluded.preview_hash_algo,
                    preview_verified  = excluded.preview_verified,
                    preview_status    = excluded.preview_status,
                    updated_at        = excluded.updated_at
                """,
                (
                    record.file_id,
                    record.preview_path,
                    record.source_hash,
                    record.source_hash_algo,
                    record.preview_hash,
                    record.preview_hash_algo,
                    record.preview_verified,
                    record.preview_status,
                    now,
                ),
            )
            self.conn.commit()
            row = self.conn.execute(
                "SELECT preview_id FROM previews WHERE file_id=?",
                (record.file_id,),
            ).fetchone()
            return int(row[0]) if row else 0

    def get_previews_for_session(
        self,
        session_id: int,
    ) -> List[PreviewRow]:
        rows = self.conn.execute(
            """
            SELECT
                p.preview_id,
                p.file_id,
                p.preview_path,
                COALESCE(p.scored,      0) AS scored,
                COALESCE(p.captioned,   0) AS captioned,
                COALESCE(p.edge_mapped, 0) AS edge_mapped
            FROM previews p
            JOIN files f ON p.file_id = f.file_id
            WHERE f.session_id = ?
            ORDER BY p.preview_id
            """,
            (session_id,),
        ).fetchall()
        return [
            PreviewRow(
                preview_id   = r["preview_id"],
                file_id      = r["file_id"],
                preview_path = r["preview_path"],
                scored       = r["scored"],
                captioned    = r["captioned"],
                edge_mapped  = r["edge_mapped"],
            )
            for r in rows
        ]

    def get_pending_previews(self, session_id: int) -> List[Dict[str, Any]]:
        """Files that need preview extraction (no verified preview yet)."""
        rows = self.conn.execute(
            """
            SELECT f.*
            FROM files f
            LEFT JOIN previews p ON p.file_id = f.file_id
            WHERE f.session_id = ?
              AND (p.file_id IS NULL OR p.preview_status != 'verified')
            ORDER BY f.capture_time, f.file_name
            """,
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_preview_mismatch(self, file_id: int) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE previews SET preview_status='mismatch', "
                "updated_at=? WHERE file_id=?",
                (_now(), file_id),
            )
            self.conn.commit()

    # ── AI reads ──────────────────────────────────────────────────────

    def get_unscored(self, session_id: int) -> List[PreviewRow]:
        return [
            p for p in self.get_previews_for_session(session_id)
            if not p.scored and p.exists
        ]

    def get_uncaptioned(self, session_id: int) -> List[PreviewRow]:
        return [
            p for p in self.get_previews_for_session(session_id)
            if not p.captioned and p.exists
        ]

    def get_unmapped(self, session_id: int) -> List[PreviewRow]:
        return [
            p for p in self.get_previews_for_session(session_id)
            if not p.edge_mapped and p.exists
        ]

    # ── AI writes ─────────────────────────────────────────────────────

    def batch_write_scores(self, payloads: List[ScorePayload]) -> None:
        if not payloads:
            return
        now = _now()
        with self._lock:
            for p in payloads:
                sc = p.scores
                self.conn.execute(
                    """
                    UPDATE previews SET
                        scored=1,
                        score_overall=?,     score_quality=?,
                        score_composition=?, score_lighting=?,
                        score_color=?,       score_dof=?,
                        score_content=?,     ai_updated_at=?
                    WHERE preview_id=?
                    """,
                    (
                        sc.get("overall"),
                        sc.get("quality"),
                        sc.get("composition"),
                        sc.get("lighting"),
                        sc.get("color"),
                        sc.get("dof"),
                        sc.get("content"),
                        now,
                        p.preview_id,
                    ),
                )
            self.conn.commit()

    def batch_write_captions(
        self,
        payloads: List[CaptionPayload],
    ) -> None:
        if not payloads:
            return
        now = _now()
        with self._lock:
            for p in payloads:
                self.conn.execute(
                    "UPDATE previews "
                    "SET captioned=1, caption=?, ai_updated_at=? "
                    "WHERE preview_id=?",
                    (p.caption, now, p.preview_id),
                )
            self.conn.commit()

    def batch_write_edge_maps(
        self,
        payloads: List[EdgePayload],
    ) -> None:
        if not payloads:
            return
        now = _now()
        with self._lock:
            for p in payloads:
                self.conn.execute(
                    "UPDATE previews "
                    "SET edge_mapped=1, edge_map_data=?,"
                    "    edge_map_encoding=?, ai_updated_at=? "
                    "WHERE preview_id=?",
                    (p.data, p.encoding, now, p.preview_id),
                )
            self.conn.commit()

    def reset_ai_phase(
        self,
        session_id: int,
        phase: str,
    ) -> int:
        """
        Clear completion flags for one AI phase so it re-runs.
        Returns number of rows affected.
        """
        col_map: Dict[str, str] = {
            "scoring": (
                "scored=0, score_overall=NULL, score_quality=NULL,"
                "score_composition=NULL, score_lighting=NULL,"
                "score_color=NULL, score_dof=NULL, score_content=NULL"
            ),
            "captioning": "captioned=0, caption=NULL",
            "edge_maps":  "edge_mapped=0, edge_map_data=NULL",
        }
        if phase not in col_map:
            raise ValueError(
                f"Unknown AI phase {phase!r}. "
                f"Choose from: {sorted(col_map)}"
            )
        with self._lock:
            cur = self.conn.execute(
                f"""
                UPDATE previews SET {col_map[phase]}
                WHERE file_id IN (
                    SELECT file_id FROM files WHERE session_id=?
                )
                """,
                (session_id,),
            )
            self.conn.commit()
            return cur.rowcount

    # ── GUI resume helpers ────────────────────────────────────────────

    def get_session_resume_info(
        self,
        session_id: int,
    ) -> Dict[str, Any]:
        """
        Single call that gives the GUI everything it needs for the
        resume dialog — no raw SQL outside this class.
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(
                f"Session {session_id} not found"
            )

        previews = self.get_previews_for_session(session_id)
        total    = len(previews)
        present  = sum(1 for p in previews if p.exists)
        scored   = sum(1 for p in previews if p.scored)
        cap      = sum(1 for p in previews if p.captioned)
        mapped   = sum(1 for p in previews if p.edge_mapped)

        # Find preview paths that are stored but missing on disk
        missing_paths: List[str] = [
            p.preview_path for p in previews if not p.exists
        ]

        return {
            "session_id":         session_id,
            "session_name":       session.get("session_name", ""),
            "source_folder":      session.get("source_folder",  ""),
            "status":             session.get("status",         ""),
            "ai_status":          session.get("ai_status",      "pending"),
            "file_count":         session.get("file_count",     0),
            "total_previews":     total,
            "present_on_disk":    present,
            "scored":             scored,
            "captioned":          cap,
            "edge_mapped":        mapped,
            "unscored":           present - scored,
            "uncaptioned":        present - cap,
            "unmapped":           present - mapped,
            "missing_paths":      missing_paths,
            "missing_count":      len(missing_paths),
        }

    def sessions_with_pending_work(self) -> List[Dict[str, Any]]:
        """
        All sessions that have any unfinished local or AI work.
        Returns list of dicts directly usable by the GUI picker.
        """
        rows = self.list_sessions()
        out: List[Dict[str, Any]] = []
        for s in rows:
            sid = s["session_id"]
            try:
                info = self.get_session_resume_info(sid)
            except Exception:
                continue
            has_work = (
                s["status"] != "completed"
                or info["unscored"] > 0
                or info["uncaptioned"] > 0
                or info["unmapped"] > 0
            )
            if has_work:
                out.append({
                    "session_id":    sid,
                    "name":          info["session_name"],
                    "source":        info["source_folder"],
                    "files":         info["file_count"],
                    "previews":      info["total_previews"],
                    "scored":        info["scored"],
                    "captioned":     info["captioned"],
                    "edge_mapped":   info["edge_mapped"],
                    "status":        info["status"],
                    "missing_count": info["missing_count"],
                })
        return out

    def fixup_missing_preview_paths(
        self,
        session_id: int,
        search_root: Path = DEFAULT_PREVIEW_ROOT,
    ) -> int:
        """
        For each stored preview path that is missing on disk, look for
        a file with the same name under search_root and update the DB.
        Returns count of rows fixed.
        """
        info    = self.get_session_resume_info(session_id)
        missing = info["missing_paths"]
        if not missing:
            return 0
        fixed = 0
        with self._lock:
            for stored in missing:
                bn = Path(stored).name
                candidate = search_root / bn
                if candidate.is_file():
                    new_rel = _safe_relpath(candidate)
                    self.conn.execute(
                        "UPDATE previews SET preview_path=? "
                        "WHERE preview_path=?",
                        (new_rel, stored),
                    )
                    fixed += 1
            if fixed:
                self.conn.commit()
        return fixed

    # ── close ─────────────────────────────────────────────────────────

    def close(self) -> None:
        self.conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# EXIFTOOL WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

class ExifToolManager:
    def __init__(self, exe: str = "exiftool") -> None:
        self.exe = exe

    def available(self) -> bool:
        return _run_ok([self.exe, "-ver"])

    def version(self) -> str:
        out = _run([self.exe, "-ver"])
        return out.splitlines()[0].strip() if out else "?"

    def read_json(
        self,
        files: Sequence[Path],
        tags: Sequence[str],
    ) -> List[dict]:
        if not files:
            return []
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tmp:
            argfile = Path(tmp.name)
            for f in files:
                tmp.write(str(f) + "\n")
        try:
            cmd = [
                self.exe, "-json", "-n",
                "-charset", "filename=utf8",
                *tags, "-@", str(argfile),
            ]
            raw = _run(cmd, timeout=120)
            return json.loads(raw) if raw else []
        finally:
            with contextlib.suppress(OSError):
                argfile.unlink()

    def extract_preview(
        self,
        raw_path: Path,
        out_path: Path,
    ) -> bool:
        """
        Extract embedded JPEG preview from a RAW file.
        Writes to a .part temp file, then atomically renames.
        Returns True on success.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + TEMP_SUFFIX)

        # Tag preference: ARW has PreviewImage as primary; NEF uses JpgFromRaw
        if raw_path.suffix.lower() == ".arw":
            tags = ["-PreviewImage", "-JpgFromRaw"]
        else:
            tags = ["-JpgFromRaw", "-PreviewImage"]

        for tag in tags:
            try:
                with tmp_path.open("wb") as fh:
                    r = subprocess.run(
                        [self.exe, "-b", tag, str(raw_path)],
                        stdout=fh,
                        stderr=subprocess.PIPE,
                        timeout=60,
                        **_win_flags(),
                    )
                if (
                    r.returncode == 0
                    and tmp_path.exists()
                    and tmp_path.stat().st_size >= MIN_PREVIEW_BYTES
                ):
                    self._fix_orientation(tmp_path)
                    tmp_path.replace(out_path)
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                pass
            finally:
                with contextlib.suppress(OSError):
                    if tmp_path.exists():
                        tmp_path.unlink()

        return False

    @staticmethod
    def _fix_orientation(image_path: Path) -> None:
        try:
            from PIL import Image, ImageOps
            with Image.open(image_path) as img:
                fixed = ImageOps.exif_transpose(img)
                fixed.save(
                    image_path,
                    format="JPEG",
                    quality=92,
                    optimize=True,
                )
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# BURST GROUPING
# ═══════════════════════════════════════════════════════════════════════════

def group_bursts(records: List[FileRecord]) -> List[BurstRecord]:
    """
    Group FileRecords into bursts by capture time gap and camera body.
    Records without a capture time are each placed in their own burst.
    """
    ordered = sorted(
        records,
        key=lambda r: (r.capture_time or "", r.file_path),
    )
    if not ordered:
        return []

    bursts: List[BurstRecord] = []

    def _new_burst(rec: FileRecord, index: int) -> BurstRecord:
        return BurstRecord(
            burst_index  = index,
            frames       = [rec],
            start_time   = rec.capture_time,
            end_time     = rec.capture_time,
            camera_make  = rec.camera_make,
            camera_model = rec.camera_model,
        )

    current = _new_burst(ordered[0], 0)

    for rec in ordered[1:]:
        prev         = current.frames[-1]
        same_camera  = (
            (prev.camera_make  or "").lower()
            == (rec.camera_make  or "").lower()
            and
            (prev.camera_model or "").lower()
            == (rec.camera_model or "").lower()
        )
        gap_ms: float = BURST_GAP_MS + 1   # default: force new burst

        if prev.capture_time and rec.capture_time and same_camera:
            try:
                gap_ms = (
                    datetime.fromisoformat(rec.capture_time)
                    - datetime.fromisoformat(prev.capture_time)
                ).total_seconds() * 1000
            except ValueError:
                pass

        if gap_ms <= BURST_GAP_MS:
            current.frames.append(rec)
            current.end_time = rec.capture_time
        else:
            bursts.append(current)
            current = _new_burst(rec, len(bursts))

    bursts.append(current)
    return bursts


# ═══════════════════════════════════════════════════════════════════════════
# METADATA TAG LIST  (shared constant, not buried in a method)
# ═══════════════════════════════════════════════════════════════════════════

EXIF_TAGS: Tuple[str, ...] = (
    "-DateTimeOriginal",
    "-SubSecTimeOriginal",
    "-ExposureTime",
    "-ShutterSpeed",
    "-ISO",
    "-FNumber",
    "-Aperture",
    "-FocalLength",
    "-FocusDistance",
    "-SubjectDistance",
    "-ApproximateFocusDistance",
    "-Make",
    "-Model",
    "-Orientation",
    "-Nikon:ShootingMode",
    "-Nikon:ReleaseMode",
    "-Sony:DriveMode",
    "-Sony:ReleaseMode2",
)


# ═══════════════════════════════════════════════════════════════════════════
# LOCAL PROCESSOR ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class LocalProcessorEngine(BaseEngine):
    """
    Phase 1 engine: discovery → metadata → bursts → previews.

    Constructor parameters
    ----------------------
    db_path            : path to SQLite database (created if absent)
    preview_root       : root folder for extracted JPEG previews
    checksum_algo      : "sha256" | "md5"
    verification_ratio : fraction of previews to spot-check (0.0–1.0)
    log                : callable(str) — line sink (GUI queue or print)
    progress           : callable(phase, done, total, filename)
    cancel             : threading.Event — set to abort
    confirm            : callable(str, bool) → bool — dependency prompts
    """

    def __init__(
        self,
        db_path:            Path                                   = DEFAULT_DB_PATH,
        preview_root:       Path                                   = DEFAULT_PREVIEW_ROOT,
        checksum_algo:      str                                    = DEFAULT_HASH_ALGO,
        verification_ratio: float                                  = DEFAULT_VERIFY_RATIO,
        log:                Callable[[str], None]                  = print,
        progress:           Optional[Callable[[str, int, int, str], None]] = None,
        cancel:             Optional[threading.Event]              = None,
        confirm:            Optional[Callable[[str, bool], bool]]  = None,
    ) -> None:
        if checksum_algo not in SUPPORTED_HASHES:
            raise ValueError(
                f"checksum_algo must be one of {sorted(SUPPORTED_HASHES)}"
            )
        self.db_path      = Path(db_path)
        self.preview_root = Path(preview_root)
        self.algo         = checksum_algo
        self._verify_ratio= max(0.0, min(1.0, verification_ratio))
        self._log         = log
        self._prog        = progress or (lambda *a: None)
        self.cancel       = cancel or threading.Event()
        self.confirm      = confirm or self._cli_confirm
        self._pause       = threading.Event()
        self._pause.set()
        self.etool        = ExifToolManager()
        self.db:          Optional[ProcessorDB] = None
        self._mismatch_n  = 0

    # ── BaseEngine interface ──────────────────────────────────────────

    def pause(self)  -> None: self._pause.clear()
    def resume(self) -> None: self._pause.set()

    @property
    def is_paused(self) -> bool:
        return not self._pause.is_set()

    # ── entry point ───────────────────────────────────────────────────

    def run(
        self,
        source_folder:     str,
        session_id:        Optional[int]  = None,
        session_name:      Optional[str]  = None,
    ) -> SessionSummary:
        t0 = time.monotonic()
        self._validate()
        source = Path(source_folder).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(
                f"Source folder not found: {source}"
            )

        self.preview_root.mkdir(parents=True, exist_ok=True)
        self.db = ProcessorDB(self.db_path)

        if session_id is None:
            name = session_name or datetime.now().strftime(
                "%Y-%m-%d_%H%M%S"
            )
            session_id, is_new = self.db.get_or_create_session(
                name, str(source), self.algo,
            )
            self._log(
                f"Session {session_id} "
                f"{'created' if is_new else 'resumed'}: {name}"
            )
        else:
            sess = self.db.get_session(session_id)
            if not sess:
                raise ValueError(
                    f"Session {session_id} not found in DB"
                )
            self._log(
                f"Resuming session {session_id}: "
                f"{sess['session_name']}"
            )

        self._log(f"  Source       : {source}")
        self._log(f"  Preview root : {self.preview_root}")
        self._log(f"  Hash algo    : {self.algo}")

        # ── scan ──────────────────────────────────────────────────────
        raw_files = self._scan(source)
        if not raw_files:
            raise FileNotFoundError(
                f"No .NEF or .ARW files found in {source}"
            )
        self._log(f"  RAW files    : {len(raw_files)}")

        # ── metadata ──────────────────────────────────────────────────
        meta_map = self._read_metadata(raw_files)

        # ── hash + upsert files ───────────────────────────────────────
        records: List[FileRecord] = []
        total = len(raw_files)
        for idx, raw_path in enumerate(raw_files):
            self._check()
            meta    = meta_map.get(str(raw_path.resolve()), {})
            src_hash = checksum_file(raw_path, self.algo)
            rec = FileRecord(
                file_path        = str(raw_path.resolve()),
                file_name        = raw_path.name,
                file_ext         = raw_path.suffix.lower(),
                file_size        = raw_path.stat().st_size,
                source_hash      = src_hash,
                source_hash_algo = self.algo,
            )
            rec.__dict__.update(_normalize_meta(meta))
            rec.file_id = self.db.upsert_file(session_id, rec)
            records.append(rec)
            self._prog(Phase.HASH, idx + 1, total, raw_path.name)

        # ── burst grouping ────────────────────────────────────────────
        bursts = group_bursts(records)
        for burst in bursts:
            burst.burst_id = self.db.insert_burst(session_id, burst)
            for rec in burst.frames:
                if rec.file_id is not None:
                    self.db.link_burst(rec.file_id, burst.burst_id)
        self._log(f"  Burst groups : {len(bursts)}")
        self._prog(Phase.BURSTS, len(bursts), len(bursts), "")

        # ── preview extraction ────────────────────────────────────────
        pending = self.db.get_pending_previews(session_id)
        self._log(f"  Preview queue: {len(pending)}")

        for idx, row in enumerate(pending):
            self._check()
            self._process_one_preview(
                session_id, row, idx, len(pending)
            )

        # ── finalise ──────────────────────────────────────────────────
        self.db.update_session_counts(session_id)
        sess = self.db.get_session(session_id) or {}
        self.db.finalize_session(session_id)

        elapsed = round(time.monotonic() - t0, 1)
        result  = SessionSummary(
            status         = "completed",
            session_id     = session_id,
            session_name   = sess.get("session_name", ""),
            source_folder  = str(source),
            file_count     = sess.get("file_count",    0),
            burst_count    = sess.get("burst_count",   0),
            preview_count  = sess.get("preview_count", 0),
            verified_count = sess.get("verified_count",0),
            mismatch_count = sess.get("mismatch_count",0),
            elapsed_s      = elapsed,
        )
        self._log(json.dumps(result.to_dict(), indent=2))
        self.db.close()
        return result

    # ── internals ─────────────────────────────────────────────────────

    def _validate(self) -> None:
        if not self.etool.available():
            raise RuntimeError(
                "ExifTool not found on PATH. "
                "Install it and ensure it is accessible."
            )

    def _scan(self, source: Path) -> List[Path]:
        files: List[Path] = []
        skip = self.preview_root.resolve()
        for root, dirs, names in os.walk(source):
            root_p = Path(root)
            dirs[:] = [
                d for d in dirs
                if (root_p / d).resolve() != skip
            ]
            for name in names:
                p = root_p / name
                if p.suffix.lower() in RAW_EXTENSIONS:
                    files.append(p)
        files.sort()
        return files

    def _read_metadata(
        self,
        raw_files: List[Path],
    ) -> Dict[str, dict]:
        data = self.etool.read_json(raw_files, list(EXIF_TAGS))
        out: Dict[str, dict] = {}
        for row in data:
            src = row.get("SourceFile") or ""
            if src:
                out[str(Path(src).resolve())] = row
        return out

    def _preview_path(
        self,
        session_id: int,
        source_hash: str,
        file_name: str,
    ) -> Path:
        stem   = _slug(Path(file_name).stem)
        prefix = source_hash[:16]
        return (
            self.preview_root
            / f"session_{session_id}"
            / f"{prefix}_{stem}.jpg"
        )

    def _should_verify(self, index: int, total: int) -> bool:
        if total <= 0 or self._verify_ratio <= 0:
            return False
        stride = max(1, round(1.0 / self._verify_ratio))
        return index % stride == 0 or index == total - 1

    def _process_one_preview(
        self,
        session_id: int,
        row: Dict[str, Any],
        idx: int,
        total: int,
    ) -> None:
        file_id    = int(row["file_id"])
        raw_path   = Path(row["file_path"])
        source_hash= row["source_hash"]
        file_name  = row.get("file_name") or raw_path.name
        out_path   = self._preview_path(
            session_id, source_hash, file_name
        )

        # Check existing preview via manifest
        manifest   = _read_manifest(out_path)
        if (
            manifest
            and manifest.get("source_hash") == source_hash
            and manifest.get("hash_algo")   == self.algo
            and out_path.exists()
            and out_path.stat().st_size >= MIN_PREVIEW_BYTES
        ):
            # Verify stored preview_hash matches file on disk
            disk_hash = checksum_file(out_path, self.algo)
            if disk_hash == manifest.get("preview_hash"):
                # Already good — just ensure DB is up to date
                self.db.upsert_preview(PreviewRecord(
                    file_id          = file_id,
                    preview_path     = _safe_relpath(out_path),
                    source_hash      = source_hash,
                    source_hash_algo = self.algo,
                    preview_hash     = disk_hash,
                    preview_hash_algo= self.algo,
                    preview_verified = 1,
                    preview_status   = "verified",
                ))
                self._prog(Phase.PREVIEW, idx + 1, total, file_name)
                return

        # Extract
        ok = self.etool.extract_preview(raw_path, out_path)
        if not ok:
            self._log(f"    ✗ extraction failed: {file_name}")
            self.db.upsert_preview(PreviewRecord(
                file_id          = file_id,
                preview_path     = _safe_relpath(out_path),
                source_hash      = source_hash,
                source_hash_algo = self.algo,
                preview_status   = "failed",
            ))
            self._prog(Phase.PREVIEW, idx + 1, total, file_name)
            return

        preview_hash = checksum_file(out_path, self.algo)
        verified     = True

        if self._should_verify(idx, total):
            spot = checksum_file(out_path, self.algo)
            if spot != preview_hash:
                verified = False
                self._mismatch_n += 1
                self._verify_ratio = min(
                    MAX_VERIFY_RATIO,
                    self._verify_ratio + VERIFY_RATIO_STEP,
                )
                self.db.mark_preview_mismatch(file_id)
                self._log(f"    ⚠ verify mismatch: {file_name}")

        _write_manifest(
            out_path, source_hash, preview_hash, raw_path, self.algo
        )
        self.db.upsert_preview(PreviewRecord(
            file_id           = file_id,
            preview_path      = _safe_relpath(out_path),
            source_hash       = source_hash,
            source_hash_algo  = self.algo,
            preview_hash      = preview_hash,
            preview_hash_algo = self.algo,
            preview_verified  = 1 if verified else 0,
            preview_status    = "verified" if verified else "mismatch",
        ))
        self._log(f"    ✓ {file_name}")
        self._prog(Phase.PREVIEW, idx + 1, total, file_name)

    @staticmethod
    def _cli_confirm(message: str, default_yes: bool = True) -> bool:
        suffix = "[Y/n]" if default_yes else "[y/N]"
        try:
            raw = input(f"{message} {suffix} ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return default_yes
        return not raw or raw in {"y", "yes"}


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════
