"""
processor_logic.py — Photo Processor pipeline for Nikon Z30 / Sony A7R4.

Burst detection, preview extraction with orientation fix, AI scoring,
captioning, edge mapping. Batched DB writes, resume support, pause
control, HDD SMART integration, offline-capable pipeline.

All output paths anchored to SCRIPT_DIR (where this file lives).
Previews stored as relative paths so DB + previews/ folder are portable.

Three-script architecture:
    processor_logic.py   — this file (REQUIRED)
    processor_gui.py     — tkinter front-end (REQUIRED for GUI)
    hdd_diagnostics.py   — HDD SMART monitor (OPTIONAL)
"""

import os
import io
import sys
import socket
import sqlite3
import subprocess
import json
import time
import threading
import tempfile
from datetime import datetime
from fractions import Fraction
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Tuple, Set, Any


# ════════════════════════════════════════════════════════════════════════════
# HF OFFLINE DETECTION — must run before any transformers import
# ════════════════════════════════════════════════════════════════════════════

def _check_hf_online() -> bool:
    """Quick connectivity probe to huggingface.co (3s timeout)."""
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3)
        return True
    except OSError:
        return False


_HF_ONLINE = _check_hf_online()

if not _HF_ONLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ════════════════════════════════════════════════════════════════════════════
# OPTIONAL HDD DIAGNOSTICS — guard import
# ════════════════════════════════════════════════════════════════════════════

try:
    from hdd_diagnostics import DriveMonitor
    HAS_DIAG = True
except ImportError:
    HAS_DIAG = False


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PT_PATH = os.path.join(SCRIPT_DIR, "model.pt")
DEFAULT_DB_PATH = os.path.join(SCRIPT_DIR, "ingest.db")
DEFAULT_PREVIEW_DIR = os.path.join(SCRIPT_DIR, "previews")

FLORENCE_MODEL_ID = "MiaoshouAI/Florence-2-large-PromptGen-v2.0"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
HED_MODEL_ID = "lllyasviel/Annotators"
CAPTION_MODE = "<MORE_DETAILED_CAPTION>"

BURST_GAP_MS = 200
EDGE_MAP_QUALITY = 80
EDGE_MAP_ENCODING = "hex"

CAPTION_BATCH_SIZE = 20
EDGE_MAP_BATCH_SIZE = 20
SCORE_BATCH_SIZE = 50

SUPPORTED_RAW_EXTENSIONS = {".nef", ".arw"}

METADATA_TAGS = [
    "-DateTimeOriginal", "-SubSecTimeOriginal", "-ExposureTime",
    "-ShutterSpeed", "-ISO", "-FNumber", "-Aperture", "-FocalLength",
    "-FocusDistance", "-SubjectDistance", "-ApproximateFocusDistance",
    "-Make", "-Model", "-Orientation", "-Nikon:ShootingMode",
    "-Nikon:ReleaseMode", "-Sony:DriveMode", "-Sony:ReleaseMode2",
]

SCORE_CATEGORIES = [
    "overall", "quality", "composition", "lighting",
    "color", "dof", "content",
]

SIBLING_SCRIPTS = (
    "processor_logic.py", "processor_gui.py", "hdd_diagnostics.py"
)


# ════════════════════════════════════════════════════════════════════════════
# PATH HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _norm(p: str) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


def _store_preview_path(abs_path: str) -> str:
    """Convert absolute path to SCRIPT_DIR-relative for DB storage."""
    try:
        return os.path.relpath(abs_path, SCRIPT_DIR)
    except ValueError:
        return abs_path


def _resolve_preview_path(stored_path: str) -> str:
    """Resolve a DB-stored preview path to absolute for file I/O.
    Handles both legacy absolute paths and new relative paths."""
    if os.path.isabs(stored_path):
        return stored_path
    return os.path.join(SCRIPT_DIR, stored_path)


# ════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class FileRecord:
    file_path: str
    file_name: str = ""
    file_size: int = 0
    file_ext: str = ""
    camera_make: str = ""
    camera_model: str = ""
    orientation: int = 1
    capture_time: Optional[datetime] = None
    capture_time_str: str = ""
    shutter_speed: Optional[str] = None
    iso: Optional[int] = None
    aperture: Optional[float] = None
    focal_length_mm: Optional[float] = None
    focus_distance_m: Optional[float] = None
    shooting_mode: str = ""
    burst_id: Optional[int] = None
    file_id: Optional[int] = None


@dataclass
class BurstInfo:
    frames: List[FileRecord] = field(default_factory=list)
    burst_index: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    burst_id: Optional[int] = None
    camera_make: str = ""


@dataclass
class PreviewRecord:
    preview_path: str          # relative to SCRIPT_DIR
    file_id: int
    burst_id: Optional[int] = None
    preview_id: Optional[int] = None


@dataclass
class PipelineConfig:
    skip_scoring: bool = False
    skip_captioning: bool = False
    skip_edge_maps: bool = False
    skip_hdd_diag: bool = False
    offline_mode: bool = False


@dataclass
class DepStatus:
    name: str
    category: str          # required | ai | optional
    available: bool
    install_hint: str = ""
    affects: str = ""


# ════════════════════════════════════════════════════════════════════════════
# DEPENDENCY CHECKER
# ════════════════════════════════════════════════════════════════════════════

def _try_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _try_cmd(*args) -> bool:
    try:
        subprocess.run(list(args), capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _torch_cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _hf_model_cached(model_id: str) -> bool:
    """Check whether a HuggingFace model exists in the local cache."""
    cache_root = os.environ.get(
        "HF_HOME",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    sanitized = model_id.replace("/", "--")
    model_dir = os.path.join(cache_root, "hub", f"models--{sanitized}")
    if not os.path.isdir(model_dir):
        return False
    snapshots = os.path.join(model_dir, "snapshots")
    if os.path.isdir(snapshots) and os.listdir(snapshots):
        return True
    return False


class DependencyChecker:

    @staticmethod
    def all_scripts_present() -> bool:
        return all(
            os.path.isfile(os.path.join(SCRIPT_DIR, s))
            for s in SIBLING_SCRIPTS)

    @staticmethod
    def check_all(model_path: str = MODEL_PT_PATH) -> List[DepStatus]:
        deps = [
            DepStatus("exiftool", "required",
                      _try_cmd("exiftool", "-ver"),
                      "sudo apt install libimage-exiftool-perl",
                      "metadata extraction and preview export"),
            DepStatus("Pillow", "required",
                      _try_import("PIL"),
                      "pip install Pillow",
                      "all image operations"),
            DepStatus("torch+CUDA", "ai",
                      _torch_cuda_ok(),
                      "pip install torch torchvision "
                      "--index-url https://download.pytorch.org/whl/cu121",
                      "scoring, captioning, edge maps"),
            DepStatus("transformers", "ai",
                      _try_import("transformers"),
                      "pip install transformers accelerate safetensors",
                      "scoring and captioning"),
            DepStatus("model.pt", "ai",
                      os.path.isfile(model_path),
                      f"place scorer weights at {model_path}",
                      "aesthetic scoring only"),
            DepStatus("controlnet_aux", "ai",
                      _try_import("controlnet_aux"),
                      "pip install controlnet-aux",
                      "edge map generation only"),
            DepStatus("mogrify", "optional",
                      _try_cmd("mogrify", "-version"),
                      "sudo apt install imagemagick",
                      "lossless JPEG orientation fix"),
            DepStatus("smartmontools", "optional",
                      _try_cmd("smartctl", "--version"),
                      "sudo apt install smartmontools",
                      "HDD SMART attribute reading"),
        ]
        # HF cache checks (only warn if offline)
        if not _HF_ONLINE:
            if not _hf_model_cached(CLIP_MODEL_ID):
                deps.append(DepStatus(
                    "CLIP cache", "ai", False,
                    f"Connect to internet and run once, or:\n"
                    f"       huggingface-cli download {CLIP_MODEL_ID}",
                    "aesthetic scoring only"))
            if not _hf_model_cached(FLORENCE_MODEL_ID):
                deps.append(DepStatus(
                    "Florence-2 cache", "ai", False,
                    f"Connect to internet and run once, or:\n"
                    f"       huggingface-cli download {FLORENCE_MODEL_ID}",
                    "captioning only"))
            if not _hf_model_cached(HED_MODEL_ID):
                deps.append(DepStatus(
                    "HED cache", "ai", False,
                    f"Connect to internet and run once, or:\n"
                    f"       huggingface-cli download {HED_MODEL_ID}",
                    "edge map generation only"))
        return deps

    @staticmethod
    def evaluate(deps: List[DepStatus], all_scripts: bool,
                 log: Callable = print,
                 confirm: Callable = None) -> PipelineConfig:
        cfg = PipelineConfig()
        required_ok = True
        need_interaction = False

        for d in deps:
            icon = "✓" if d.available else "✗"
            log(f"    {icon}  {d.name:<22s} ({d.category})")
            if not d.available:
                log(f"       → {d.install_hint}")
                if d.category == "required":
                    required_ok = False
                else:
                    need_interaction = True

        if not required_ok:
            missing = [d.name for d in deps
                       if d.category == "required" and not d.available]
            raise RuntimeError(
                "Missing required: " + ", ".join(missing))

        if all_scripts and not need_interaction:
            log("    All dependencies satisfied")
            return cfg

        if confirm is None:
            confirm = DependencyChecker._cli_confirm

        for d in deps:
            if d.available or d.category == "required":
                continue
            if d.category == "ai":
                msg = (f"{d.name} not found ({d.affects}). "
                       f"Skip affected phases?")
                if confirm(msg, True):
                    DependencyChecker._apply_skip(cfg, d)
                else:
                    raise RuntimeError(f"User declined skip: {d.name}")
            elif d.category == "optional":
                msg = (f"{d.name} not found ({d.affects}). "
                       f"Continue without?")
                if not confirm(msg, True):
                    raise RuntimeError(f"User declined: {d.name}")
                DependencyChecker._apply_skip(cfg, d)
        return cfg

    @staticmethod
    def _apply_skip(cfg: PipelineConfig, dep: DepStatus):
        a = dep.affects.lower()
        n = dep.name.lower()
        # torch/CUDA missing → skip all AI
        if "scoring" in a and "captioning" in a and "edge" in a:
            cfg.skip_scoring = True
            cfg.skip_captioning = True
            cfg.skip_edge_maps = True
            return
        if "scoring" in a:
            cfg.skip_scoring = True
        if "captioning" in a:
            cfg.skip_captioning = True
        if "edge" in a:
            cfg.skip_edge_maps = True
        if "smart" in n or "hdd" in n:
            cfg.skip_hdd_diag = True

    @staticmethod
    def _cli_confirm(message: str, default_yes: bool = True) -> bool:
        suffix = "[Y/n]" if default_yes else "[y/N]"
        try:
            resp = input(f"  {message} {suffix} ").strip().lower()
            if not resp:
                return default_yes
            return resp in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return default_yes


# ════════════════════════════════════════════════════════════════════════════
# DATABASE
# ════════════════════════════════════════════════════════════════════════════

class ProcessorDB:
    _DDL = """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        session_name   TEXT NOT NULL,
        process_date   TEXT NOT NULL,
        source_folder  TEXT NOT NULL,
        file_count     INTEGER DEFAULT 0,
        burst_count    INTEGER DEFAULT 0,
        preview_count  INTEGER DEFAULT 0,
        status         TEXT DEFAULT 'in_progress'
    );
    CREATE TABLE IF NOT EXISTS bursts (
        burst_id       INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id     INTEGER NOT NULL,
        burst_index    INTEGER NOT NULL,
        frame_count    INTEGER NOT NULL,
        start_time     TEXT,
        end_time       TEXT,
        camera_make    TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    );
    CREATE TABLE IF NOT EXISTS files (
        file_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id     INTEGER NOT NULL,
        burst_id       INTEGER,
        file_path      TEXT NOT NULL UNIQUE,
        file_name      TEXT,
        file_ext       TEXT,
        file_size_bytes INTEGER,
        camera_make    TEXT,
        camera_model   TEXT,
        orientation    INTEGER DEFAULT 1,
        capture_time   TEXT,
        shutter_speed  TEXT,
        iso            INTEGER,
        aperture       REAL,
        focal_length_mm REAL,
        focus_distance_m REAL,
        shooting_mode  TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id),
        FOREIGN KEY (burst_id)   REFERENCES bursts(burst_id)
    );
    CREATE TABLE IF NOT EXISTS previews (
        preview_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id        INTEGER NOT NULL UNIQUE,
        burst_id       INTEGER,
        preview_path   TEXT NOT NULL,
        score_overall  REAL,
        score_quality  REAL,
        score_composition REAL,
        score_lighting REAL,
        score_color    REAL,
        score_dof      REAL,
        score_content  REAL,
        caption        TEXT,
        edge_map_data  TEXT,
        edge_map_encoding TEXT DEFAULT 'hex',
        scored         INTEGER DEFAULT 0,
        captioned      INTEGER DEFAULT 0,
        edge_mapped    INTEGER DEFAULT 0,
        FOREIGN KEY (file_id)  REFERENCES files(file_id),
        FOREIGN KEY (burst_id) REFERENCES bursts(burst_id)
    );
    CREATE TABLE IF NOT EXISTS drive_info (
        drive_id       INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id     INTEGER NOT NULL,
        device_path    TEXT,
        serial_number  TEXT,
        model_name     TEXT,
        firmware       TEXT,
        drive_type     TEXT,
        capacity_bytes INTEGER,
        smart_start    TEXT,
        smart_end      TEXT,
        scan_start_ts  TEXT,
        scan_end_ts    TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    );
    CREATE TABLE IF NOT EXISTS drive_read_log (
        read_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        drive_id       INTEGER NOT NULL,
        session_id     INTEGER NOT NULL,
        file_path      TEXT,
        file_size_bytes INTEGER,
        read_start_ts  TEXT,
        read_end_ts    TEXT,
        read_time_ms   REAL,
        expected_ms    REAL,
        flag           TEXT,
        detail         TEXT,
        FOREIGN KEY (drive_id)   REFERENCES drive_info(drive_id),
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    );
    CREATE TABLE IF NOT EXISTS drive_smart_deltas (
        delta_id       INTEGER PRIMARY KEY AUTOINCREMENT,
        drive_id       INTEGER NOT NULL,
        attribute_id   INTEGER,
        attribute_name TEXT,
        value_start    INTEGER,
        value_end      INTEGER,
        raw_start      INTEGER,
        raw_end        INTEGER,
        delta          INTEGER,
        FOREIGN KEY (drive_id) REFERENCES drive_info(drive_id)
    );
    CREATE INDEX IF NOT EXISTS idx_files_session      ON files(session_id);
    CREATE INDEX IF NOT EXISTS idx_files_burst         ON files(burst_id);
    CREATE INDEX IF NOT EXISTS idx_files_time          ON files(capture_time);
    CREATE INDEX IF NOT EXISTS idx_files_path          ON files(file_path);
    CREATE INDEX IF NOT EXISTS idx_previews_file       ON previews(file_id);
    CREATE INDEX IF NOT EXISTS idx_previews_burst      ON previews(burst_id);
    CREATE INDEX IF NOT EXISTS idx_previews_scored     ON previews(scored);
    CREATE INDEX IF NOT EXISTS idx_previews_captioned  ON previews(captioned);
    CREATE INDEX IF NOT EXISTS idx_previews_edge       ON previews(edge_mapped);
    CREATE INDEX IF NOT EXISTS idx_bursts_session      ON bursts(session_id);
    CREATE INDEX IF NOT EXISTS idx_readlog_session     ON drive_read_log(session_id);
    CREATE INDEX IF NOT EXISTS idx_readlog_flag        ON drive_read_log(flag);
    CREATE INDEX IF NOT EXISTS idx_smartdelta_drv      ON drive_smart_deltas(drive_id);
    """

    def __init__(self, db_path: str):
        self.path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(self._DDL)
        self.conn.commit()
        self._lock = threading.Lock()
        self._migrate()

    def _migrate(self):
        """Add columns missing from older DB versions."""
        tables = {}
        for tbl in ("drive_info", "drive_read_log", "previews", "files"):
            rows = self.conn.execute(
                f"PRAGMA table_info({tbl})").fetchall()
            tables[tbl] = {r[1] for r in rows}

        migrations = [
            ("drive_info", "filesystem", "TEXT"),
            ("drive_info", "smart_mode", "TEXT"),
            ("drive_info", "throughput_json", "TEXT"),
            ("drive_read_log", "read_type", "TEXT"),
        ]
        changed = False
        for tbl, col, typ in migrations:
            if tbl in tables and col not in tables[tbl]:
                self.conn.execute(
                    f"ALTER TABLE {tbl} ADD COLUMN {col} {typ}")
                changed = True
        if changed:
            self.conn.commit()

    # ── session ──

    def get_or_create_session(self, name: str,
                              folder: str) -> Tuple[int, bool]:
        with self._lock:
            row = self.conn.execute(
                "SELECT session_id FROM sessions "
                "WHERE source_folder=? AND status!='completed'",
                (folder,)).fetchone()
            if row:
                return row["session_id"], False
            cur = self.conn.execute(
                "INSERT INTO sessions"
                "(session_name,process_date,source_folder,status) "
                "VALUES(?,?,?,'in_progress')",
                (name, datetime.now().isoformat(), folder))
            self.conn.commit()
            return cur.lastrowid, True

    def list_resumable_sessions(self) -> List[Dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM sessions WHERE status!='completed' "
                "ORDER BY session_id DESC").fetchall()
            return [dict(r) for r in rows]

    def get_session_info(self, session_id: int) -> Dict:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM sessions WHERE session_id=?",
                (session_id,)).fetchone()
            return dict(row) if row else {}

    def get_resume_state(self, session_id: int) -> Dict:
        """What is done and what is pending for this session."""
        with self._lock:
            total_files = self.conn.execute(
                "SELECT COUNT(*) FROM files "
                "WHERE session_id=?", (session_id,)).fetchone()[0]
            total_prev = self.conn.execute(
                "SELECT COUNT(*) FROM previews p "
                "JOIN files f ON p.file_id=f.file_id "
                "WHERE f.session_id=?", (session_id,)).fetchone()[0]
            scored = self.conn.execute(
                "SELECT COUNT(*) FROM previews p "
                "JOIN files f ON p.file_id=f.file_id "
                "WHERE f.session_id=? AND p.scored=1",
                (session_id,)).fetchone()[0]
            captioned = self.conn.execute(
                "SELECT COUNT(*) FROM previews p "
                "JOIN files f ON p.file_id=f.file_id "
                "WHERE f.session_id=? AND p.captioned=1",
                (session_id,)).fetchone()[0]
            edge_mapped = self.conn.execute(
                "SELECT COUNT(*) FROM previews p "
                "JOIN files f ON p.file_id=f.file_id "
                "WHERE f.session_id=? AND p.edge_mapped=1",
                (session_id,)).fetchone()[0]
            paths = self.conn.execute(
                "SELECT preview_path FROM previews p "
                "JOIN files f ON p.file_id=f.file_id "
                "WHERE f.session_id=?",
                (session_id,)).fetchall()

        missing = [r["preview_path"] for r in paths
                   if not os.path.isfile(
                       _resolve_preview_path(r["preview_path"]))]

        return {
            "total_files":    total_files,
            "total_previews": total_prev,
            "scored":         scored,
            "captioned":      captioned,
            "edge_mapped":    edge_mapped,
            "unscored":       total_prev - scored,
            "uncaptioned":    total_prev - captioned,
            "unmapped":       total_prev - edge_mapped,
            "missing_preview_files": missing,
            "previews_complete": (
                len(missing) == 0
                and total_prev > 0
                and total_prev == total_files),
        }

    def get_existing_file_paths(self, session_id: int) -> Set[str]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT file_path FROM files WHERE session_id=?",
                (session_id,)).fetchall()
            return {r["file_path"] for r in rows}

    # ── insert ──

    def insert_file(self, session_id: int, rec: FileRecord) -> int:
        with self._lock:
            try:
                cur = self.conn.execute(
                    "INSERT INTO files(session_id,file_path,file_name,"
                    "file_ext,file_size_bytes,camera_make,camera_model,"
                    "orientation,capture_time,shutter_speed,iso,"
                    "aperture,focal_length_mm,focus_distance_m,"
                    "shooting_mode) VALUES"
                    "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (session_id, rec.file_path, rec.file_name,
                     rec.file_ext, rec.file_size, rec.camera_make,
                     rec.camera_model, rec.orientation,
                     rec.capture_time_str, rec.shutter_speed,
                     rec.iso, rec.aperture, rec.focal_length_mm,
                     rec.focus_distance_m, rec.shooting_mode))
                self.conn.commit()
                return cur.lastrowid
            except sqlite3.IntegrityError:
                row = self.conn.execute(
                    "SELECT file_id FROM files WHERE file_path=?",
                    (rec.file_path,)).fetchone()
                return row["file_id"] if row else 0

    def insert_burst(self, session_id: int, b: BurstInfo) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO bursts(session_id,burst_index,"
                "frame_count,start_time,end_time,camera_make) "
                "VALUES(?,?,?,?,?,?)",
                (session_id, b.burst_index, len(b.frames),
                 b.start_time.isoformat() if b.start_time else None,
                 b.end_time.isoformat() if b.end_time else None,
                 b.camera_make))
            self.conn.commit()
            return cur.lastrowid

    def update_file_burst(self, file_id: int, burst_id: int):
        with self._lock:
            self.conn.execute(
                "UPDATE files SET burst_id=? WHERE file_id=?",
                (burst_id, file_id))
            self.conn.commit()

    def insert_preview(self, p: PreviewRecord) -> int:
        with self._lock:
            try:
                cur = self.conn.execute(
                    "INSERT INTO previews(file_id,burst_id,"
                    "preview_path) VALUES(?,?,?)",
                    (p.file_id, p.burst_id, p.preview_path))
                self.conn.commit()
                return cur.lastrowid
            except sqlite3.IntegrityError:
                row = self.conn.execute(
                    "SELECT preview_id FROM previews "
                    "WHERE file_id=?", (p.file_id,)).fetchone()
                return row["preview_id"] if row else 0

    # ── AI phase queries ──

    def get_unscored_previews(self) -> List[Dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT preview_id, file_id, preview_path "
                "FROM previews WHERE scored=0").fetchall()
            return [dict(r) for r in rows]

    def get_uncaptioned_previews(self) -> List[Dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT preview_id, file_id, preview_path "
                "FROM previews WHERE captioned=0").fetchall()
            return [dict(r) for r in rows]

    def get_unmapped_previews(self) -> List[Dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT preview_id, file_id, preview_path "
                "FROM previews WHERE edge_mapped=0").fetchall()
            return [dict(r) for r in rows]

    # ── batch writes ──

    def batch_update_scores(self,
                            updates: List[Tuple[int, Dict[str, float]]]):
        with self._lock:
            for pid, sc in updates:
                self.conn.execute(
                    "UPDATE previews SET "
                    "score_overall=?,score_quality=?,"
                    "score_composition=?,score_lighting=?,"
                    "score_color=?,score_dof=?,score_content=?,"
                    "scored=1 WHERE preview_id=?",
                    (sc.get('overall'), sc.get('quality'),
                     sc.get('composition'), sc.get('lighting'),
                     sc.get('color'), sc.get('dof'),
                     sc.get('content'), pid))
            self.conn.commit()

    def batch_update_captions(self,
                              updates: List[Tuple[int, str]]):
        with self._lock:
            for pid, cap in updates:
                self.conn.execute(
                    "UPDATE previews SET caption=?,captioned=1 "
                    "WHERE preview_id=?", (cap, pid))
            self.conn.commit()

    def batch_update_edge_maps(self,
                               updates: List[Tuple[int, str, str]]):
        with self._lock:
            for pid, data, enc in updates:
                self.conn.execute(
                    "UPDATE previews SET edge_map_data=?,"
                    "edge_map_encoding=?,edge_mapped=1 "
                    "WHERE preview_id=?", (data, enc, pid))
            self.conn.commit()

    # ── stats / finalize ──

    def update_session_stats(self, session_id: int):
        with self._lock:
            fc = self.conn.execute(
                "SELECT COUNT(*) FROM files "
                "WHERE session_id=?", (session_id,)).fetchone()[0]
            bc = self.conn.execute(
                "SELECT COUNT(*) FROM bursts "
                "WHERE session_id=?", (session_id,)).fetchone()[0]
            pc = self.conn.execute(
                "SELECT COUNT(*) FROM previews p "
                "JOIN files f ON p.file_id=f.file_id "
                "WHERE f.session_id=?", (session_id,)).fetchone()[0]
            self.conn.execute(
                "UPDATE sessions SET file_count=?,"
                "burst_count=?,preview_count=? "
                "WHERE session_id=?",
                (fc, bc, pc, session_id))
            self.conn.commit()

    def finalize_session(self, session_id: int,
                         status: str = "completed"):
        self.update_session_stats(session_id)
        with self._lock:
            self.conn.execute(
                "UPDATE sessions SET status=? "
                "WHERE session_id=?", (status, session_id))
            self.conn.commit()

    def has_wal_files(self) -> bool:
        """Check if WAL/SHM sidecar files exist (incomplete state)."""
        wal = self.path + "-wal"
        shm = self.path + "-shm"
        return os.path.exists(wal) or os.path.exists(shm)

    def close(self):
        self.conn.close()


# ════════════════════════════════════════════════════════════════════════════
# EXIFTOOL MANAGER
# ════════════════════════════════════════════════════════════════════════════

class ExifToolManager:
    def __init__(self):
        self.exe = "exiftool"
        self._mogrify = None
        self._jpegtran = None

    def available(self) -> bool:
        try:
            return subprocess.run(
                [self.exe, "-ver"],
                capture_output=True, text=True, timeout=10
            ).returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def get_version(self) -> str:
        try:
            r = subprocess.run(
                [self.exe, "-ver"],
                capture_output=True, text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else "?"
        except Exception:
            return "?"

    def check_mogrify(self) -> bool:
        if self._mogrify is None:
            try:
                self._mogrify = subprocess.run(
                    ["mogrify", "-version"],
                    capture_output=True, timeout=5
                ).returncode == 0
            except (FileNotFoundError, subprocess.TimeoutExpired):
                self._mogrify = False
        return self._mogrify

    def check_jpegtran(self) -> bool:
        if self._jpegtran is None:
            try:
                subprocess.run(
                    ["jpegtran", "-h"],
                    capture_output=True, timeout=5)
                self._jpegtran = True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                self._jpegtran = False
        return self._jpegtran

    def run_json(self, extra_args: List[str],
                 timeout: int = 600) -> List[dict]:
        cmd = [self.exe, "-json", "-n", "-charset",
               "filename=utf8"] + extra_args
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0 and not proc.stdout.strip():
            raise RuntimeError(
                f"ExifTool error: {proc.stderr[:400]}")
        return json.loads(proc.stdout) if proc.stdout.strip() else []

    def extract_preview(self, raw_path: str, out_path: str,
                        fix_orientation: bool = True) -> bool:
        ext = os.path.splitext(raw_path)[1].lower()
        tags = (["-PreviewImage", "-JpgFromRaw"] if ext == ".arw"
                else ["-JpgFromRaw", "-PreviewImage"])
        extracted = False
        for tag in tags:
            try:
                proc = subprocess.run(
                    [self.exe, "-b", tag, raw_path],
                    capture_output=True, timeout=30)
                if proc.returncode == 0 and len(proc.stdout) > 1000:
                    with open(out_path, "wb") as f:
                        f.write(proc.stdout)
                    extracted = True
                    break
            except Exception:
                continue
        if not extracted:
            return False
        if fix_orientation:
            self._fix_orientation(raw_path, out_path)
        return True

    def _fix_orientation(self, raw_path: str, preview: str):
        try:
            subprocess.run(
                [self.exe, "-overwrite_original",
                 "-TagsFromFile", raw_path,
                 "-Orientation", preview],
                capture_output=True, timeout=15)
            if self.check_mogrify():
                subprocess.run(
                    ["mogrify", "-auto-orient", preview],
                    capture_output=True, timeout=30)
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════════════════
# METADATA PARSING HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _fmt_shutter(val) -> Optional[str]:
    if val is None:
        return None
    try:
        s = float(val)
    except (ValueError, TypeError):
        return str(val)
    if s <= 0:
        return str(s)
    if s >= 1:
        return f"{s:.1f}"
    fr = Fraction(s).limit_denominator(16000)
    return (f"1/{fr.denominator}" if fr.numerator == 1
            else f"{fr.numerator}/{fr.denominator}")


def _parse_time(d: dict) -> Optional[datetime]:
    dto = d.get("DateTimeOriginal")
    if not dto or not isinstance(dto, str):
        return None
    dt = None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(dto, fmt)
            break
        except ValueError:
            continue
    if dt is None:
        return None
    sub = d.get("SubSecTimeOriginal")
    if sub is not None:
        s = str(sub).strip()
        if s:
            try:
                cs = int(s[:2]) if len(s) >= 2 else int(s)
                dt = dt.replace(microsecond=cs * 10000)
            except ValueError:
                pass
    return dt


def _parse_focus(d: dict) -> Optional[float]:
    for tag in ("FocusDistance", "SubjectDistance",
                "ApproximateFocusDistance"):
        v = d.get(tag)
        if v is None:
            continue
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
        s = str(v).strip().lower()
        if s in ("inf", "infinity", "undef", "unknown", ""):
            continue
        s = s.replace("m", "").strip()
        try:
            f = float(s)
            if f > 0:
                return f
        except ValueError:
            continue
    return None


def _parse_meta(d: dict) -> dict:
    ct = _parse_time(d)
    exp = d.get("ExposureTime") or d.get("ShutterSpeed")
    iso = d.get("ISO")
    if isinstance(iso, (list, tuple)):
        iso = iso[0] if iso else None
    try:
        iso = int(iso) if iso is not None else None
    except (ValueError, TypeError):
        iso = None
    ap = d.get("FNumber") or d.get("Aperture")
    try:
        ap = float(ap) if ap is not None else None
    except (ValueError, TypeError):
        ap = None
    fl = d.get("FocalLength")
    if isinstance(fl, str):
        fl = fl.lower().replace("mm", "").strip()
    try:
        fl = float(fl) if fl is not None else None
    except (ValueError, TypeError):
        fl = None
    make = d.get("Make", "").strip()
    model = d.get("Model", "").strip()
    ori = d.get("Orientation", 1)
    if isinstance(ori, str):
        try:
            ori = int(ori)
        except ValueError:
            ori = 1
    sm = ""
    if "Nikon" in make or "NIKON" in make:
        sm = d.get("ShootingMode", "") or d.get("ReleaseMode", "")
    elif "Sony" in make or "SONY" in make:
        sm = d.get("DriveMode", "") or d.get("ReleaseMode2", "")
    if isinstance(sm, (list, tuple)):
        sm = sm[0] if sm else ""
    return dict(
        capture_time=ct,
        capture_time_str=ct.isoformat() if ct else "",
        shutter_speed=_fmt_shutter(exp),
        iso=iso, aperture=ap, focal_length_mm=fl,
        focus_distance_m=_parse_focus(d),
        camera_make=make, camera_model=model,
        orientation=ori, shooting_mode=str(sm))


def _detect_bursts(recs: List[FileRecord]) -> List[BurstInfo]:
    ordered = sorted(
        recs,
        key=lambda r: (r.capture_time or datetime.max,
                       os.path.basename(r.file_path)))
    if not ordered:
        return []
    groups: List[List[FileRecord]] = [[ordered[0]]]
    for i in range(1, len(ordered)):
        p, c = ordered[i - 1], ordered[i]
        if p.capture_time is None or c.capture_time is None:
            groups.append([c])
            continue
        gap = (c.capture_time - p.capture_time).total_seconds() * 1000
        if gap > BURST_GAP_MS:
            groups.append([c])
        else:
            groups[-1].append(c)
    bursts = []
    for idx, frames in enumerate(groups):
        makes = [f.camera_make for f in frames if f.camera_make]
        bursts.append(BurstInfo(
            frames=frames, burst_index=idx,
            start_time=frames[0].capture_time,
            end_time=frames[-1].capture_time,
            camera_make=makes[0] if makes else ""))
    return bursts


def _session_name() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _bucket_size(w: int, h: int) -> Tuple[int, int]:
    ratio = w / h
    cands = [(768, 512), (512, 768), (512, 832),
             (576, 1024), (512, 512)]
    return min(cands, key=lambda s: abs((s[0] / s[1]) - ratio))


# ════════════════════════════════════════════════════════════════════════════
# AI MODELS — offline-safe loading
# ════════════════════════════════════════════════════════════════════════════

class AestheticScorer:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self, log: Callable = print) -> bool:
        if self._loaded:
            return True
        if not os.path.isfile(self.model_path):
            log(f"  Scorer model not found: {self.model_path}")
            return False
        try:
            import torch
            import torch.nn as nn
            from transformers import CLIPVisionModel, CLIPProcessor

            class _M(nn.Module):
                def __init__(self_m):
                    super().__init__()
                    self_m.backbone = CLIPVisionModel.from_pretrained(
                        CLIP_MODEL_ID,
                        local_files_only=not _HF_ONLINE)
                    for name in SCORE_CATEGORIES:
                        setattr(self_m, f"{name}_head",
                                nn.Sequential(nn.Linear(768, 1)))

                def forward(self_m, pixel_values):
                    f = self_m.backbone(
                        pixel_values=pixel_values).pooler_output
                    return torch.cat([
                        getattr(self_m, f"{n}_head")(f)
                        for n in SCORE_CATEGORIES], dim=-1)

            log("  Loading CLIP scorer...")
            if not _HF_ONLINE:
                log("    (offline — using local cache)")
            raw_sd = torch.load(
                self.model_path, map_location="cpu",
                weights_only=False)
            remapped = {}
            for k, v in raw_sd.items():
                if k.startswith("backbone."):
                    remapped["backbone.vision_model."
                             + k[9:]] = v
                else:
                    remapped[k] = v
            self.model = _M()
            self.model.load_state_dict(remapped, strict=False)
            self.model.to(self.device).eval()
            self.processor = CLIPProcessor.from_pretrained(
                CLIP_MODEL_ID,
                local_files_only=not _HF_ONLINE)
            self._loaded = True
            log("  Scorer ready")
            return True
        except OSError as e:
            if "offline" in str(e).lower():
                log(f"  CLIP not cached locally. Run online once or:")
                log(f"    huggingface-cli download {CLIP_MODEL_ID}")
            else:
                log(f"  ERROR loading scorer: {e}")
            return False
        except Exception as e:
            log(f"  ERROR loading scorer: {e}")
            return False

    def score(self, image) -> Dict[str, float]:
        if not self._loaded:
            return {}
        import torch
        inp = self.processor(
            images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            s = self.model(inp["pixel_values"]).squeeze().cpu()
        return {c: s[i].item()
                for i, c in enumerate(SCORE_CATEGORIES)}

    def unload(self, log: Callable = print):
        if self._loaded:
            import torch
            self.model = self.model.to("cpu")
            del self.model
            del self.processor
            self.model = self.processor = None
            self._loaded = False
            torch.cuda.empty_cache()
            log("  Scorer unloaded")


class FlorenceCaptioner:
    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self, log: Callable = print) -> bool:
        if self._loaded:
            return True
        try:
            import torch
            from transformers import (AutoModelForCausalLM,
                                      AutoProcessor)
            import transformers.utils
            if not hasattr(transformers.utils,
                           'is_flash_attn_greater_or_equal_2_10'):
                transformers.utils \
                    .is_flash_attn_greater_or_equal_2_10 = (
                        lambda *a, **k: False)

            log(f"  Loading Florence-2...")
            if not _HF_ONLINE:
                log("    (offline — using local cache)")

            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    local_files_only=not _HF_ONLINE)
            except OSError:
                log(f"  Florence-2 not cached locally.")
                log(f"  Connect to internet and run once, or:")
                log(f"    huggingface-cli download {self.model_id}")
                return False

            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                local_files_only=not _HF_ONLINE)
            self._loaded = True
            log("  Florence-2 ready")
            return True
        except Exception as e:
            log(f"  ERROR loading Florence-2: {e}")
            return False

    def caption(self, image) -> str:
        if not self._loaded:
            return ""
        import torch
        self.model = self.model.to(self.device)
        inp = self.processor(
            text=CAPTION_MODE, images=image,
            return_tensors="pt"
        ).to(self.device, torch.float32)
        with torch.no_grad():
            ids = self.model.generate(
                input_ids=inp["input_ids"],
                pixel_values=inp["pixel_values"],
                max_new_tokens=512,
                do_sample=False, num_beams=3)
        result = self.processor.post_process_generation(
            self.processor.batch_decode(
                ids, skip_special_tokens=False)[0],
            task=CAPTION_MODE,
            image_size=(image.width, image.height))
        self.model = self.model.to("cpu")
        import torch as _t
        _t.cuda.empty_cache()
        return result[CAPTION_MODE].strip()

    def unload(self, log: Callable = print):
        if self._loaded:
            import torch
            if self.model is not None:
                self.model = self.model.to("cpu")
                del self.model
            if self.processor is not None:
                del self.processor
            self.model = self.processor = None
            self._loaded = False
            torch.cuda.empty_cache()
            log("  Florence-2 unloaded")


class EdgeMapper:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.hed = None
        self._loaded = False

    def load(self, log: Callable = print) -> bool:
        if self._loaded:
            return True
        try:
            from controlnet_aux import HEDdetector
            log("  Loading HED detector...")
            if not _HF_ONLINE:
                log("    (offline — using local cache)")
            self.hed = HEDdetector.from_pretrained(HED_MODEL_ID)
            self._loaded = True
            log("  HED ready")
            return True
        except Exception as e:
            if "offline" in str(e).lower():
                log(f"  HED not cached locally.")
                log(f"    huggingface-cli download {HED_MODEL_ID}")
            else:
                log(f"  ERROR loading HED: {e}")
            return False

    def detect(self, image,
               target_size: Tuple[int, int] = None) -> str:
        if not self._loaded:
            return ""
        if target_size is None:
            target_size = _bucket_size(image.width, image.height)
        edges = self.hed(
            image, detect_resolution=512,
            image_resolution=max(target_size), safe=True)
        edges_r = edges.resize(target_size)
        buf = io.BytesIO()
        edges_r.save(buf, format="WEBP", quality=EDGE_MAP_QUALITY)
        if EDGE_MAP_ENCODING == "base64":
            import base64
            return base64.b64encode(buf.getvalue()).decode('ascii')
        return buf.getvalue().hex()

    def unload(self, log: Callable = print):
        if self._loaded:
            import torch
            del self.hed
            self.hed = None
            self._loaded = False
            torch.cuda.empty_cache()
            log("  HED unloaded")


# ════════════════════════════════════════════════════════════════════════════
# PROCESSING ENGINE
# ════════════════════════════════════════════════════════════════════════════

class ProcessorEngine:
    def __init__(self, db_path: str = DEFAULT_DB_PATH,
                 model_path: str = MODEL_PT_PATH,
                 log: Callable = print,
                 cancel: threading.Event = None,
                 confirm: Callable = None):
        self.db_path = db_path
        self.model_path = model_path
        self._log = log
        self.cancel = cancel or threading.Event()
        self._confirm = confirm or DependencyChecker._cli_confirm

        self._pause = threading.Event()
        self._pause.set()

        try:
            import torch
            self.device = ("cuda" if torch.cuda.is_available()
                           else "cpu")
        except ImportError:
            self.device = "cpu"

        self.etool = ExifToolManager()
        self.scorer = AestheticScorer(model_path, self.device)
        self.captioner = FlorenceCaptioner(
            FLORENCE_MODEL_ID, self.device)
        self.edge_mapper = EdgeMapper(self.device)

        self.db: Optional[ProcessorDB] = None
        self._monitor: Optional[Any] = None
        self._cfg = PipelineConfig()

    # ── controls ──

    def pause(self):
        self._pause.clear()

    def resume(self):
        self._pause.set()

    @property
    def is_paused(self) -> bool:
        return not self._pause.is_set()

    def _chk(self):
        if self.cancel.is_set():
            raise InterruptedError("Cancelled")

    def _wait(self):
        if not self._pause.is_set():
            self._log("  ⏸  Paused — model loaded in VRAM")
        self._pause.wait()
        self._chk()

    # ════════════════════════════════════════════════════════════════════
    # RUN
    # ════════════════════════════════════════════════════════════════════

    def run(self, source_folder: str = None,
            resume_session_id: int = None,
            progress: Callable = None,
            extract_all: bool = True,
            custom_filter: str = "") -> dict:

        prog = progress or (lambda *a: None)
        preview_dir = DEFAULT_PREVIEW_DIR
        nef_count = arw_count = 0

        self._log("=" * 60)
        self._log(f"SCRIPT_DIR : {SCRIPT_DIR}")
        self._log(f"Previews   : {preview_dir}")
        self._log(f"Database   : {self.db_path}")
        self._log(f"Network    : {'online' if _HF_ONLINE else 'OFFLINE'}")
        self._log(f"Device     : {self.device}")
        self._log("=" * 60)

        # ── Phase 0 · Dependencies ─────────────────────────────────
        self._log("\nPhase 0 · Dependencies")
        self._cfg = self._validate_deps()
        self._chk()

        # ── Phase 1 · Tools ────────────────────────────────────────
        self._log("\nPhase 1 · Tools")
        if not self.etool.available():
            raise RuntimeError(
                "ExifTool not found. "
                "Install: sudo apt install libimage-exiftool-perl")
        self._log(f"  ExifTool v{self.etool.get_version()}")
        mg = "✓" if self.etool.check_mogrify() else "✗"
        self._log(f"  mogrify: {mg}")

        # ── Database ───────────────────────────────────────────────
        self.db = ProcessorDB(self.db_path)

        # ── Session resolution ─────────────────────────────────────
        self._log("\nPhase 2 · Session")

        if resume_session_id is not None:
            session_id = resume_session_id
            info = self.db.get_session_info(session_id)
            if not info:
                raise ValueError(
                    f"Session {session_id} not found in DB")
            if source_folder is None:
                source_folder = info["source_folder"]
            self._log(f"  Resuming session {session_id}: "
                      f"{info.get('session_name', '?')}")

        elif source_folder is not None:
            source_folder = os.path.abspath(source_folder)
            session_id, is_new = self.db.get_or_create_session(
                _session_name(), source_folder)
            self._log(
                f"  {'New' if is_new else 'Resumed'} session "
                f"{session_id}")
        else:
            raise ValueError(
                "Need source_folder or resume_session_id")

        # ── Resume state ───────────────────────────────────────────
        state = self.db.get_resume_state(session_id)
        source_online = (source_folder is not None
                         and os.path.isdir(source_folder))
        previews_ok = state["previews_complete"]

        self._log(f"  Source: {'online' if source_online else 'OFFLINE'}")
        self._log(f"  Files in DB: {state['total_files']}")
        self._log(f"  Previews: {state['total_previews']} "
                  f"({'all present' if previews_ok else str(len(state['missing_preview_files'])) + ' missing'})")
        self._log(f"  Scored: {state['scored']}  "
                  f"Captioned: {state['captioned']}  "
                  f"Edge-mapped: {state['edge_mapped']}")

        # ── Online phases (2-6) ────────────────────────────────────
        if not previews_ok:
            if not source_online:
                miss = len(state["missing_preview_files"])
                raise RuntimeError(
                    f"Source offline, {miss} previews missing.\n"
                    f"Connect: {source_folder}")
            nef_count, arw_count = self._online_phases(
                source_folder, preview_dir,
                session_id, state, prog)
        else:
            self._log("\n  All previews present locally — "
                      "skipping online phases (3-6)")
            self._cfg.offline_mode = True

        # ── Phase 7 · Scoring ──────────────────────────────────────
        self._run_scoring(prog)

        # ── Phase 8 · Captioning ───────────────────────────────────
        self._run_captioning(prog)

        # ── Phase 9 · Edge maps ────────────────────────────────────
        self._run_edge_maps(prog)

        # ── Phase 10 · Finalize ────────────────────────────────────
        self._log("\nPhase 10 · Finalizing")

        if self._monitor and self._monitor.is_active:
            self._monitor.end_session()

        self.db.finalize_session(session_id)
        info = self.db.get_session_info(session_id)
        self.db.close()

        dt = ""
        if self._monitor and self._monitor.drive_info:
            dt = self._monitor.drive_info.drive_type

        summary = {
            "status": "completed",
            "session_name": info.get("session_name", ""),
            "source_folder": source_folder or "",
            "file_count": info.get("file_count", 0),
            "burst_count": info.get("burst_count", 0),
            "preview_count": info.get("preview_count", 0),
            "db_path": self.db_path,
            "nef_count": nef_count,
            "arw_count": arw_count,
            "offline_mode": self._cfg.offline_mode,
            "drive_type": dt,
        }

        self._log("\n" + "=" * 60)
        self._log("COMPLETE")
        for k, v in summary.items():
            if k != "status":
                self._log(f"  {k}: {v}")
        self._log("=" * 60)
        return summary

    # ════════════════════════════════════════════════════════════════
    # ONLINE PHASES  3-6
    # ════════════════════════════════════════════════════════════════

    def _online_phases(self, source_folder, preview_dir,
                       session_id, state, prog
                       ) -> Tuple[int, int]:

        # Phase 3 · Scan
        self._log("\nPhase 3 · Scanning...")
        self._chk()
        raw_files = self._scan(source_folder)
        if not raw_files:
            raise FileNotFoundError(
                f"No NEF/ARW in {source_folder}")

        nef_c = sum(1 for f in raw_files
                    if f.lower().endswith('.nef'))
        arw_c = sum(1 for f in raw_files
                    if f.lower().endswith('.arw'))
        total_gb = sum(os.path.getsize(f)
                       for f in raw_files) / 2**30
        self._log(f"  {len(raw_files)} files: "
                  f"{nef_c} NEF, {arw_c} ARW ({total_gb:.2f} GB)")
        prog("scan", len(raw_files), len(raw_files), "done")

        # HDD monitoring
        if HAS_DIAG and not self._cfg.skip_hdd_diag:
            self._log("\n  Drive classification...")
            self._monitor = DriveMonitor(self._log)
            self._monitor.start_session(
                self.db.conn, session_id, source_folder)

        # Phase 4 · Metadata
        self._log("\nPhase 4 · Metadata...")
        self._chk()
        existing = self.db.get_existing_file_paths(session_id)
        new_files = [f for f in raw_files if f not in existing]
        self._log(f"  New: {len(new_files)}  "
                  f"Existing: {len(existing)}")

        if new_files:
            t0 = time.monotonic()
            meta_map = self._exif_batch(new_files)
            batch_ms = (time.monotonic() - t0) * 1000

            if self._monitor and self._monitor.is_active:
                bs = sum(os.path.getsize(f) for f in new_files)
                now = datetime.now().isoformat()
                self._monitor.log_file_read(
                    "[batch_metadata]", bs, now, now, batch_ms)

            for i, fp in enumerate(new_files):
                self._chk()
                rec = FileRecord(
                    file_path=fp,
                    file_name=os.path.basename(fp),
                    file_ext=os.path.splitext(fp)[1].lower(),
                    file_size=os.path.getsize(fp))
                meta = meta_map.get(_norm(fp))
                if meta:
                    p = _parse_meta(meta)
                    for k, v in p.items():
                        setattr(rec, k, v)
                rec.file_id = self.db.insert_file(session_id, rec)
                prog("metadata", i + 1, len(new_files),
                     rec.file_name)

        all_recs = self._load_all_records(session_id)

        # Phase 5 · Bursts
        self._log("\nPhase 5 · Bursts...")
        self._chk()
        bursts = _detect_bursts(all_recs)
        self._log(f"  {len(bursts)} burst(s)")
        for b in bursts:
            b.burst_id = self.db.insert_burst(session_id, b)
            for fr in b.frames:
                self.db.update_file_burst(fr.file_id, b.burst_id)
        prog("bursts", len(bursts), len(bursts), "done")

        # Phase 6 · Extract previews
        self._log(f"\nPhase 6 · Previews → {preview_dir}")
        self._chk()
        os.makedirs(preview_dir, exist_ok=True)

        to_extract = []
        for rec in all_recs:
            pname = (os.path.splitext(rec.file_name)[0]
                     + "_preview.jpg")
            ppath = os.path.join(preview_dir, pname)
            if not os.path.exists(ppath):
                to_extract.append((rec, ppath))

        self._log(f"  To extract: {len(to_extract)}  "
                  f"Existing: {len(all_recs) - len(to_extract)}")

        for i, (rec, ppath) in enumerate(to_extract):
            self._chk()
            try:
                if self._monitor and self._monitor.is_active:
                    ok = self._monitor.timed_read(
                        lambda r=rec.file_path, o=ppath:
                            self.etool.extract_preview(r, o),
                        rec.file_path, rec.file_size)
                else:
                    ok = self.etool.extract_preview(
                        rec.file_path, ppath)
                if ok:
                    rel = _store_preview_path(ppath)
                    pr = PreviewRecord(
                        preview_path=rel,
                        file_id=rec.file_id,
                        burst_id=rec.burst_id)
                    self.db.insert_preview(pr)
                else:
                    self._log(f"    ✗ {rec.file_name}")
            except OSError as e:
                self._log(f"    ⚠️  I/O ERROR: {e}")
                self._log("    Source lost — saving progress")
                break
            prog("previews", i + 1, len(to_extract),
                 rec.file_name)

        # ensure DB rows for pre-existing previews
        for rec in all_recs:
            pname = (os.path.splitext(rec.file_name)[0]
                     + "_preview.jpg")
            ppath = os.path.join(preview_dir, pname)
            if os.path.exists(ppath):
                rel = _store_preview_path(ppath)
                self.db.insert_preview(PreviewRecord(
                    preview_path=rel,
                    file_id=rec.file_id,
                    burst_id=rec.burst_id))

        self.db.update_session_stats(session_id)
        return nef_c, arw_c

    # ════════════════════════════════════════════════════════════════
    # AI PHASES  7-9
    # ════════════════════════════════════════════════════════════════

    def _run_scoring(self, prog):
        self._log("\nPhase 7 · Scoring...")
        self._chk()
        if self._cfg.skip_scoring:
            self._log("  SKIP (dependency)")
            return
        unscored = self.db.get_unscored_previews()
        self._log(f"  Pending: {len(unscored)}")
        if not unscored:
            return
        if not self.scorer.load(self._log):
            return
        from PIL import Image
        batch = []
        try:
            for i, prev in enumerate(unscored):
                self._wait()
                try:
                    pp = _resolve_preview_path(prev["preview_path"])
                    img = Image.open(pp).convert("RGB")
                    sc = self.scorer.score(img)
                    batch.append((prev["preview_id"], sc))
                    s = " ".join(f"{k[:3]}={v:.2f}"
                                for k, v in sc.items())
                    self._log(f"    {os.path.basename(pp)}: {s}")
                except Exception as e:
                    self._log(f"    ERR {prev['preview_path']}: {e}")
                if len(batch) >= SCORE_BATCH_SIZE:
                    self.db.batch_update_scores(batch)
                    self._log(f"    → {len(batch)} scores written")
                    batch = []
                prog("scoring", i + 1, len(unscored),
                     os.path.basename(prev["preview_path"]))
        finally:
            if batch:
                self.db.batch_update_scores(batch)
                self._log(f"    → {len(batch)} scores written")
        self.scorer.unload(self._log)

    def _run_captioning(self, prog):
        self._log("\nPhase 8 · Captioning...")
        self._chk()
        if self._cfg.skip_captioning:
            self._log("  SKIP (dependency)")
            return
        uncap = self.db.get_uncaptioned_previews()
        self._log(f"  Pending: {len(uncap)}")
        if not uncap:
            return
        if not self.captioner.load(self._log):
            return
        from PIL import Image
        batch = []
        try:
            for i, prev in enumerate(uncap):
                self._wait()
                try:
                    pp = _resolve_preview_path(prev["preview_path"])
                    img = Image.open(pp).convert("RGB")
                    cap = self.captioner.caption(img)
                    batch.append((prev["preview_id"], cap))
                    wc = len(cap.split())
                    bn = os.path.basename(pp)
                    self._log(f"    {bn} ({wc}w): {cap[:120]}")
                except Exception as e:
                    self._log(
                        f"    ERR {prev['preview_path']}: {e}")
                if len(batch) >= CAPTION_BATCH_SIZE:
                    self.db.batch_update_captions(batch)
                    self._log(f"    → {len(batch)} captions written")
                    batch = []
                prog("captioning", i + 1, len(uncap),
                     os.path.basename(prev["preview_path"]))
        finally:
            if batch:
                self.db.batch_update_captions(batch)
                self._log(f"    → {len(batch)} captions written")
        self.captioner.unload(self._log)

    def _run_edge_maps(self, prog):
        self._log("\nPhase 9 · Edge maps...")
        self._chk()
        if self._cfg.skip_edge_maps:
            self._log("  SKIP (dependency)")
            return
        unmapped = self.db.get_unmapped_previews()
        self._log(f"  Pending: {len(unmapped)}")
        if not unmapped:
            return
        if not self.edge_mapper.load(self._log):
            return
        from PIL import Image
        batch = []
        try:
            for i, prev in enumerate(unmapped):
                self._wait()
                try:
                    pp = _resolve_preview_path(prev["preview_path"])
                    img = Image.open(pp).convert("RGB")
                    ed = self.edge_mapper.detect(img)
                    batch.append((
                        prev["preview_id"], ed, EDGE_MAP_ENCODING))
                    if EDGE_MAP_ENCODING == "hex":
                        kb = len(ed) / 2 / 1024
                    else:
                        kb = len(ed) * 3 / 4 / 1024
                    self._log(f"    {os.path.basename(pp)}: "
                              f"{kb:.1f}KB")
                except Exception as e:
                    self._log(
                        f"    ERR {prev['preview_path']}: {e}")
                if len(batch) >= EDGE_MAP_BATCH_SIZE:
                    self.db.batch_update_edge_maps(batch)
                    self._log(
                        f"    → {len(batch)} edge maps written")
                    batch = []
                prog("edge_map", i + 1, len(unmapped),
                     os.path.basename(prev["preview_path"]))
        finally:
            if batch:
                self.db.batch_update_edge_maps(batch)
                self._log(f"    → {len(batch)} edge maps written")
        self.edge_mapper.unload(self._log)

    # ════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ════════════════════════════════════════════════════════════════

    def _validate_deps(self) -> PipelineConfig:
        deps = DependencyChecker.check_all(self.model_path)
        all_s = DependencyChecker.all_scripts_present()
        if all_s:
            self._log("  Three-script suite detected")
        present = [s for s in SIBLING_SCRIPTS
                   if os.path.isfile(
                       os.path.join(SCRIPT_DIR, s))]
        if not all_s:
            self._log(f"  Scripts: {', '.join(present)}")
        if not HAS_DIAG:
            self._log("  hdd_diagnostics: not loaded")
        return DependencyChecker.evaluate(
            deps, all_s, self._log, self._confirm)

    def _scan(self, folder: str) -> List[str]:
        out = []
        for root, dirs, files in os.walk(folder):
            if "previews" in dirs:
                dirs.remove("previews")
            for f in files:
                if os.path.splitext(f)[1].lower() \
                        in SUPPORTED_RAW_EXTENSIONS:
                    out.append(os.path.join(root, f))
        out.sort()
        return out

    def _exif_batch(self, paths: List[str]) -> Dict[str, dict]:
        if not paths:
            return {}
        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.txt', delete=False) as f:
            argfile = f.name
            for p in paths:
                f.write(p + "\n")
        try:
            data = self.etool.run_json(
                METADATA_TAGS + ["-@", argfile])
            return {_norm(d.get("SourceFile", "")): d
                    for d in data}
        finally:
            try:
                os.remove(argfile)
            except OSError:
                pass

    def _load_all_records(self,
                          session_id: int) -> List[FileRecord]:
        with self.db._lock:
            rows = self.db.conn.execute(
                "SELECT * FROM files WHERE session_id=? "
                "ORDER BY capture_time",
                (session_id,)).fetchall()
        recs = []
        for r in rows:
            rec = FileRecord(
                file_path=r["file_path"],
                file_name=(r["file_name"]
                           or os.path.basename(r["file_path"])),
                file_ext=r["file_ext"] or "",
                file_size=r["file_size_bytes"] or 0,
                camera_make=r["camera_make"] or "",
                camera_model=r["camera_model"] or "",
                orientation=r["orientation"] or 1,
                shutter_speed=r["shutter_speed"],
                iso=r["iso"],
                aperture=r["aperture"],
                focal_length_mm=r["focal_length_mm"],
                focus_distance_m=r["focus_distance_m"],
                shooting_mode=r["shooting_mode"] or "",
                file_id=r["file_id"],
                burst_id=r["burst_id"])
            ct = r["capture_time"]
            if ct:
                try:
                    rec.capture_time = datetime.fromisoformat(ct)
                    rec.capture_time_str = ct
                except ValueError:
                    pass
            recs.append(rec)
        return recs