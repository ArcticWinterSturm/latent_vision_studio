#!/usr/bin/env python3
"""
gui.py — Photo Database Viewer with On-Demand Image Generation

Entry point. Reads SQLite databases produced by the ingestion pipeline.
Displays metadata, quality scores, captions, and edge maps.
Generates images via view_backend.py (local SD) and optionally refines
via view_api.py (Google Gemini Flash).

Usage:
    python gui.py [working_directory]
    python gui.py [path/to/database.db]
    Default: current directory

Dependencies:
    Required : tkinter, Pillow
    Optional : view_backend.py  → local SD generation
    Optional : view_api.py      → API refinement (needs view_backend.py)
"""

import os
import sys
import io
import re
import glob
import base64
import queue
import random
import sqlite3
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Set
from dataclasses import dataclass, field
from PIL import Image, ImageTk


# ════════════════════════════════════════════════════════════════════════════
# FEATURE DETECTION
# ════════════════════════════════════════════════════════════════════════════

BACKEND_AVAILABLE = False
GENERATION_AVAILABLE = False

try:
    from view_backend import (
        ImageGenerator,
        GenerationCache,
        HesitancyFilter,
        decode_edge_map,
        compute_bucket,
        GENERATION_AVAILABLE as _GENAV,
        DEFAULT_CCS,
        DEFAULT_CFG,
        DEFAULT_STEPS,
        DEFAULT_SEED,
        DEFAULT_NEGATIVE_PROMPT,
        CCS_PRESETS,
    )

    BACKEND_AVAILABLE = True
    GENERATION_AVAILABLE = _GENAV
except ImportError:
    # Inline defaults so the GUI still works stand-alone
    DEFAULT_CCS = 1.0
    DEFAULT_CFG = 1.5
    DEFAULT_STEPS = 4
    DEFAULT_SEED = 42
    DEFAULT_NEGATIVE_PROMPT = (
        "blurry, low quality, cartoon, painting, "
        "text, watermark, duplicate"
    )
    CCS_PRESETS = [
        {"ccs": 0.6, "cfg": 1.2},
        {"ccs": 0.8, "cfg": 1.5},
        {"ccs": 1.0, "cfg": 1.5},
        {"ccs": 1.0, "cfg": 1.8},
        {"ccs": 1.1, "cfg": 2.0},
    ]

    def decode_edge_map(data, encoding="hex"):
        if not data:
            return None
        raw = None
        for enc in [encoding, "hex", "base64"]:
            if raw is not None:
                break
            try:
                raw = (
                    base64.b64decode(data) if enc == "base64"
                    else bytes.fromhex(data)
                )
            except Exception:
                continue
        if raw is None:
            return None
        try:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return None

API_AVAILABLE = False
try:
    from view_api import GeminiRefiner
    API_AVAILABLE = True
except ImportError:
    pass


# ════════════════════════════════════════════════════════════════════════════
# FALLBACK STUBS  (when view_backend.py is absent)
# ════════════════════════════════════════════════════════════════════════════

GENERATED_DIR = "generated"


class _FallbackCache:
    """Read-only cache that can still list previously-generated files."""

    def __init__(self, base_dir: str):
        self.dir = os.path.join(base_dir, GENERATED_DIR)
        self._stems: Dict[str, Set[str]] = {}
        self._scanned = False

    def _ensure(self):
        if self._scanned:
            return
        self._scanned = True
        if not os.path.isdir(self.dir):
            return
        for f in os.listdir(self.dir):
            m = re.match(r"(.+)_(gen|gem)\d+\.jpg$", f)
            if m:
                self._stems.setdefault(m.group(1), set()).add(m.group(2))

    def has_generation(self, pp, prefix="gen"):
        self._ensure()
        if not pp:
            return False
        stem = os.path.splitext(os.path.basename(pp))[0]
        return prefix in self._stems.get(stem, set())

    def has_any_generation(self, pp):
        return self.has_generation(pp, "gen") or self.has_generation(pp, "gem")

    def get_latest(self, pp, prefix="gen"):
        if not pp:
            return None
        stem = os.path.splitext(os.path.basename(pp))[0]
        m = sorted(glob.glob(os.path.join(self.dir, f"{stem}_{prefix}*.jpg")))
        return m[-1] if m else None

    def get_all_versions(self, pp, prefix="gen"):
        if not pp:
            return []
        stem = os.path.splitext(os.path.basename(pp))[0]
        return sorted(glob.glob(os.path.join(self.dir, f"{stem}_{prefix}*.jpg")))

    def get_version_count(self, pp, prefix="gen"):
        return len(self.get_all_versions(pp, prefix))

    def count_generated(self, prefix="gen"):
        self._ensure()
        return sum(1 for p in self._stems.values() if prefix in p)

    def count_all_generated(self):
        self._ensure()
        return len(self._stems)

    def save_new(self, *_a, **_kw):
        return ""


class _FallbackHesitancy:
    def filter(self, text):
        return (text or "").strip()


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

SCORE_MAX = 5.0

SCORE_CATEGORIES = [
    "overall", "quality", "composition",
    "lighting", "color", "dof", "content",
]

MAX_PHOTO_REFS = 24
TREE_BATCH = 500

DEFAULT_PREPROMPT = (
    "Regenerate this image as a photorealistic photograph. "
    "Fix any artifacts, unnatural textures, distorted features, "
    "or synthetic-looking elements. Maintain the exact same "
    "composition, framing, and subject matter."
)

# ── Theme palette ─────────────────────────────────────────────────────────
C_BG       = "#1e1e1e"
C_SURFACE  = "#252526"
C_SURFACE2 = "#2d2d2d"
C_SURFACE3 = "#333333"
C_HOVER    = "#3c3c3c"
C_TEXT     = "#d4d4d4"
C_DIM      = "#808080"
C_ACCENT   = "#007acc"
C_GREEN    = "#4ec9b0"
C_YELLOW   = "#cca700"
C_ORANGE   = "#ce9178"
C_RED      = "#f14c4c"
C_SELECT   = "#094771"
C_CANVAS   = "#141414"


# ════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class FileListItem:
    file_id: int
    preview_id: int
    file_name: str
    preview_path: str
    capture_time: str = ""
    score_overall: Optional[float] = None
    burst_id: Optional[int] = None
    burst_index: Optional[int] = None
    burst_frame_count: Optional[int] = None


@dataclass
class FileDetails:
    file_id: int
    preview_id: int
    file_path: str
    file_name: str
    preview_path: str
    camera_make: str = ""
    camera_model: str = ""
    capture_time: str = ""
    shutter_speed: str = ""
    iso: Optional[int] = None
    aperture: Optional[float] = None
    focal_length_mm: Optional[float] = None
    focus_distance_m: Optional[float] = None
    shooting_mode: str = ""
    burst_id: Optional[int] = None
    burst_index: Optional[int] = None
    burst_frame_count: Optional[int] = None
    session_id: Optional[int] = None
    caption: str = ""
    edge_map_data: str = ""
    edge_map_encoding: str = "hex"
    scores: Dict[str, float] = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════════════
# DATABASE ACCESS (Read-Only)
# ════════════════════════════════════════════════════════════════════════════

class ViewerDB:

    def __init__(self, db_path: str):
        self.path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._table_cache: Optional[Set[str]] = None
        self._col_cache: Dict[str, Set[str]] = {}
        self._open()

    def _open(self):
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"Database not found: {self.path}")
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        try:
            self.conn.execute("PRAGMA query_only = ON")
        except Exception:
            pass

    def _tables(self) -> Set[str]:
        if self._table_cache is None:
            rows = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            self._table_cache = {r["name"] for r in rows}
        return self._table_cache

    def _has(self, t: str) -> bool:
        return t in self._tables()

    def _cols(self, t: str) -> Set[str]:
        if t not in self._col_cache:
            rows = self.conn.execute(f"PRAGMA table_info({t})").fetchall()
            self._col_cache[t] = {r["name"] for r in rows}
        return self._col_cache[t]

    @staticmethod
    def _pick(col, alias, available, default="NULL"):
        return f"{alias}.{col}" if col in available else f"{default} as {col}"

    # ── queries ───────────────────────────────────────────────────────────

    def get_sessions(self) -> List[Dict]:
        if not self._has("sessions"):
            return []
        with self._lock:
            rows = self.conn.execute(
                "SELECT session_id, session_name, source_folder, "
                "file_count, burst_count, status "
                "FROM sessions ORDER BY session_id DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_file_list(
        self,
        session_id: Optional[int] = None,
        sort_by: str = "capture_time",
        sort_desc: bool = False,
    ) -> List[FileListItem]:
        """Light-weight list — never touches blob columns."""
        if not self._has("files"):
            return []

        fc = self._cols("files")
        hp = self._has("previews")
        hb = self._has("bursts")
        pc = self._cols("previews") if hp else set()
        bc = self._cols("bursts") if hb else set()

        sel = [
            "f.file_id", "f.file_name", "f.capture_time",
            self._pick("burst_id", "f", fc),
        ]
        joins: List[str] = []

        if hp:
            sel += [
                "p.preview_id" if "preview_id" in pc else "NULL as preview_id",
                self._pick("preview_path", "p", pc, "''"),
                self._pick("score_overall", "p", pc),
            ]
            joins.append("LEFT JOIN previews p ON f.file_id = p.file_id")
        else:
            sel += ["NULL as preview_id", "'' as preview_path", "NULL as score_overall"]

        if hb and "burst_id" in fc:
            sel += [
                self._pick("burst_index", "b", bc),
                "b.frame_count as burst_frame_count" if "frame_count" in bc
                else "NULL as burst_frame_count",
            ]
            jc = "file_id" if "file_id" in bc else "burst_id"
            joins.append(f"LEFT JOIN bursts b ON f.{jc} = b.{jc}")
        else:
            sel += ["NULL as burst_index", "NULL as burst_frame_count"]

        omap = {
            "capture_time": "f.capture_time",
            "score": ("p.score_overall" if hp else "NULL"),
            "name": "f.file_name",
            "burst": "f.burst_id",
        }
        oc = omap.get(sort_by, "f.capture_time")
        od = "DESC" if sort_desc else "ASC"
        if sort_by == "score":
            order = f"ORDER BY CASE WHEN {oc} IS NULL THEN 1 ELSE 0 END, {oc} {od}"
        else:
            order = f"ORDER BY {oc} {od}"

        where, params = "", ()
        if session_id is not None and "session_id" in fc:
            where = "WHERE f.session_id = ?"
            params = (session_id,)

        sql = f"SELECT {', '.join(sel)} FROM files f {' '.join(joins)} {where} {order}"

        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()

        return [
            FileListItem(
                file_id=r["file_id"],
                preview_id=r["preview_id"] or 0,
                file_name=r["file_name"] or "",
                preview_path=r["preview_path"] or "",
                capture_time=r["capture_time"] or "",
                score_overall=r["score_overall"],
                burst_id=r["burst_id"],
                burst_index=r["burst_index"],
                burst_frame_count=r["burst_frame_count"],
            )
            for r in rows
        ]

    def get_file_details(self, file_id: int) -> Optional[FileDetails]:
        """Full detail row including blobs — call for one file at a time."""
        if not self._has("files"):
            return None

        fc = self._cols("files")
        hp = self._has("previews")
        hb = self._has("bursts")
        pc = self._cols("previews") if hp else set()
        bc = self._cols("bursts") if hb else set()

        sel = ["f.file_id"]
        for c, d in [
            ("file_path", "''"), ("file_name", "''"),
            ("camera_make", "''"), ("camera_model", "''"),
            ("capture_time", "''"), ("shutter_speed", "''"),
            ("iso", "NULL"), ("aperture", "NULL"),
            ("focal_length_mm", "NULL"), ("focus_distance_m", "NULL"),
            ("shooting_mode", "''"), ("burst_id", "NULL"),
            ("session_id", "NULL"),
        ]:
            sel.append(self._pick(c, "f", fc, d))

        joins: List[str] = []

        if hp:
            for c, d in [
                ("preview_id", "NULL"), ("preview_path", "''"),
                ("caption", "''"), ("edge_map_data", "''"),
                ("edge_map_encoding", "'hex'"),
                ("score_overall", "NULL"), ("score_quality", "NULL"),
                ("score_composition", "NULL"),
                ("score_lighting", "NULL"), ("score_color", "NULL"),
                ("score_dof", "NULL"), ("score_content", "NULL"),
            ]:
                sel.append(self._pick(c, "p", pc, d))
            joins.append("LEFT JOIN previews p ON f.file_id = p.file_id")
        else:
            sel += [
                "NULL as preview_id", "'' as preview_path",
                "'' as caption", "'' as edge_map_data",
                "'hex' as edge_map_encoding",
                "NULL as score_overall", "NULL as score_quality",
                "NULL as score_composition", "NULL as score_lighting",
                "NULL as score_color", "NULL as score_dof",
                "NULL as score_content",
            ]

        if hb and "burst_id" in fc:
            sel.append(self._pick("burst_index", "b", bc))
            sel.append(
                "b.frame_count as burst_frame_count" if "frame_count" in bc
                else "NULL as burst_frame_count"
            )
            jc = "file_id" if "file_id" in bc else "burst_id"
            joins.append(f"LEFT JOIN bursts b ON f.{jc} = b.{jc}")
        else:
            sel += ["NULL as burst_index", "NULL as burst_frame_count"]

        sql = (
            f"SELECT {', '.join(sel)} FROM files f "
            f"{' '.join(joins)} WHERE f.file_id = ? LIMIT 1"
        )

        with self._lock:
            row = self.conn.execute(sql, (file_id,)).fetchone()
        if not row:
            return None

        scores: Dict[str, float] = {}
        for cat in SCORE_CATEGORIES:
            try:
                v = row[f"score_{cat}"]
                if v is not None:
                    scores[cat] = float(v)
            except (IndexError, KeyError):
                pass

        return FileDetails(
            file_id=row["file_id"],
            preview_id=row["preview_id"] or 0,
            file_path=row["file_path"] or "",
            file_name=row["file_name"] or "",
            preview_path=row["preview_path"] or "",
            camera_make=row["camera_make"] or "",
            camera_model=row["camera_model"] or "",
            capture_time=row["capture_time"] or "",
            shutter_speed=row["shutter_speed"] or "",
            iso=row["iso"],
            aperture=row["aperture"],
            focal_length_mm=row["focal_length_mm"],
            focus_distance_m=row["focus_distance_m"],
            shooting_mode=row["shooting_mode"] or "",
            burst_id=row["burst_id"],
            burst_index=row["burst_index"],
            burst_frame_count=row["burst_frame_count"],
            session_id=row["session_id"],
            caption=row["caption"] or "",
            edge_map_data=row["edge_map_data"] or "",
            edge_map_encoding=row["edge_map_encoding"] or "hex",
            scores=scores,
        )

    def get_stats(self) -> Dict[str, int]:
        s: Dict[str, int] = {}
        with self._lock:
            s["files"] = (
                self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
                if self._has("files") else 0
            )
            if self._has("previews"):
                pc = self._cols("previews")
                s["previews"] = self.conn.execute(
                    "SELECT COUNT(*) FROM previews"
                ).fetchone()[0]
                if "captioned" in pc:
                    s["captioned"] = self.conn.execute(
                        "SELECT COUNT(*) FROM previews WHERE captioned=1"
                    ).fetchone()[0]
                elif "caption" in pc:
                    s["captioned"] = self.conn.execute(
                        "SELECT COUNT(*) FROM previews "
                        "WHERE caption IS NOT NULL AND caption!=''"
                    ).fetchone()[0]
                else:
                    s["captioned"] = 0
                if "edge_mapped" in pc:
                    s["edge_mapped"] = self.conn.execute(
                        "SELECT COUNT(*) FROM previews WHERE edge_mapped=1"
                    ).fetchone()[0]
                elif "edge_map_data" in pc:
                    s["edge_mapped"] = self.conn.execute(
                        "SELECT COUNT(*) FROM previews "
                        "WHERE edge_map_data IS NOT NULL AND edge_map_data!=''"
                    ).fetchone()[0]
                else:
                    s["edge_mapped"] = 0
            else:
                s.update(previews=0, captioned=0, edge_mapped=0)
        return s

    def close(self):
        if self.conn:
            with self._lock:
                self.conn.close()
                self.conn = None


# ════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ════════════════════════════════════════════════════════════════════════════

class ViewerApp:

    # ──────────────────────────────────────────────────────────────────────
    # Init
    # ──────────────────────────────────────────────────────────────────────

    def __init__(self, root: tk.Tk, working_dir: str):
        self.root = root
        self.working_dir = os.path.abspath(working_dir)
        self.root.title(f"Photo Viewer — {os.path.basename(self.working_dir)}")
        self.root.geometry("1440x920")
        self.root.minsize(1020, 700)

        # State
        self.db: Optional[ViewerDB] = None
        self.file_list: List[FileListItem] = []
        self.current_details: Optional[FileDetails] = None
        self.current_index: int = 0
        self.current_session_id: Optional[int] = None
        self._photo_refs: List[ImageTk.PhotoImage] = []
        self._view_mode: str = "generated"       # generated | edge | api
        self._generating: bool = False
        self._generating_file_id: Optional[int] = None
        self._resize_after_id: Optional[str] = None
        self._progress_dots: int = 0
        self._db_paths: List[str] = []
        self._sessions: List[Optional[int]] = [None]
        self._edge_expanded: bool = False
        self._api_expanded: bool = False
        self._api_key_visible: bool = False
        self._load_gen: int = 0       # async-load generation counter
        self._batch_items: List[FileListItem] = []
        self._batch_offset: int = 0
        self._queue: queue.Queue = queue.Queue()

        # Backend components (real or fallback)
        if BACKEND_AVAILABLE:
            self.cache = GenerationCache(self.working_dir)
            self.generator = ImageGenerator(log=self._log)
            self.hesitancy = HesitancyFilter(
                os.path.join(self.working_dir, "hesitancy.txt"),
                log=self._log,
            )
        else:
            self.cache = _FallbackCache(self.working_dir)
            self.generator = None
            self.hesitancy = _FallbackHesitancy()

        # Phase 0 — build everything, paint immediately
        self._build_menu()
        self._build_ui()
        self._apply_theme()
        self._bind_keys()
        self.root.update_idletasks()

        # Phase 1 — discover databases (sync, < 10 ms)
        self._find_database()

        # Start polling + cleanup handler
        self._poll_queue()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ──────────────────────────────────────────────────────────────────────
    # Menu
    # ──────────────────────────────────────────────────────────────────────

    def _build_menu(self):
        bar = tk.Menu(self.root)
        self.root.config(menu=bar)

        fm = tk.Menu(bar, tearoff=0)
        fm.add_command(label="Unload Pipeline", command=self._unload_pipeline,
                       accelerator="Ctrl+U")
        fm.add_separator()
        fm.add_command(label="Quit", command=self._on_close,
                       accelerator="Ctrl+Q")
        bar.add_cascade(label="File", menu=fm)

        hm = tk.Menu(bar, tearoff=0)
        hm.add_command(label="About", command=self._show_about)
        bar.add_cascade(label="Help", menu=hm)

    def _unload_pipeline(self):
        if self.generator and self.generator.is_loaded():
            self.generator.unload()
            self._pipeline_lbl.config(text="Pipeline: Unloaded")
            self._status_var.set("Pipeline unloaded")

    def _show_about(self):
        caps = []
        if BACKEND_AVAILABLE:
            caps.append("view_backend ✓")
        else:
            caps.append("view_backend ✗")
        if GENERATION_AVAILABLE:
            caps.append("torch/diffusers ✓")
        else:
            caps.append("torch/diffusers ✗")
        if API_AVAILABLE:
            caps.append("view_api ✓")
        else:
            caps.append("view_api ✗")

        messagebox.showinfo(
            "About",
            "Photo Database Viewer\n\n"
            f"Capabilities: {', '.join(caps)}\n\n"
            "Keyboard shortcuts:\n"
            "  ←/→         Navigate files\n"
            "  Home/End     First / last file\n"
            "  G            Generate (local SD)\n"
            "  R            Regenerate new version\n"
            "  A            API regenerate\n"
            "  Space        Cycle view mode\n"
            "  1-5          CCS/CFG presets\n"
            "  Ctrl+U       Unload pipeline\n"
            "  Ctrl+Q       Quit",
        )

    # ──────────────────────────────────────────────────────────────────────
    # UI Construction
    # ──────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ── top bar ──
        top = ttk.Frame(main)
        top.pack(fill=tk.X, pady=(0, 4))

        ttk.Label(top, text="Database:").pack(side=tk.LEFT, padx=(0, 4))
        self._db_var = tk.StringVar()
        self._db_combo = ttk.Combobox(
            top, textvariable=self._db_var, state="readonly", width=30,
        )
        self._db_combo.pack(side=tk.LEFT, padx=(0, 10))
        self._db_combo.bind("<<ComboboxSelected>>", self._on_db_select)

        self._session_lbl = ttk.Label(top, text="Session: All")
        self._session_lbl.pack(side=tk.LEFT, padx=(10, 0))

        self._file_count_lbl = ttk.Label(top, text="")
        self._file_count_lbl.pack(side=tk.RIGHT)

        # ── three-panel pane ──
        self._pane = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        self._pane.pack(fill=tk.BOTH, expand=True)
        self._build_left()
        self._build_center()
        self._build_right()

        # ── status bar ──
        self._build_status()

    # ── left panel ────────────────────────────────────────────────────────

    def _build_left(self):
        left = ttk.Frame(self._pane, width=280)
        self._pane.add(left, weight=0)

        # Sessions
        sf = ttk.LabelFrame(left, text="Sessions", padding=4)
        sf.pack(fill=tk.X, pady=(0, 4))
        self._sess_list = tk.Listbox(sf, height=4, exportselection=False)
        self._sess_list.pack(fill=tk.X)
        self._sess_list.bind("<<ListboxSelect>>", self._on_session_select)

        # Sort
        sof = ttk.LabelFrame(left, text="Sort", padding=4)
        sof.pack(fill=tk.X, pady=(0, 4))
        self._sort_var = tk.StringVar(value="capture_time")
        row = ttk.Frame(sof)
        row.pack(fill=tk.X)
        for txt, val in [("Time", "capture_time"), ("Score ↓", "score"),
                         ("Name", "name"), ("Burst", "burst")]:
            ttk.Radiobutton(
                row, text=txt, variable=self._sort_var, value=val,
                command=self._refresh_file_list,
            ).pack(side=tk.LEFT, padx=2)

        # File tree
        ff = ttk.LabelFrame(left, text="Files", padding=4)
        ff.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        cols = ("name", "time", "score", "g")
        self._tree = ttk.Treeview(
            ff, columns=cols, show="headings", selectmode="browse", height=20,
        )
        self._tree.heading("name", text="File",
                           command=lambda: self._set_sort("name"))
        self._tree.heading("time", text="Time",
                           command=lambda: self._set_sort("capture_time"))
        self._tree.heading("score", text="Score",
                           command=lambda: self._set_sort("score"))
        self._tree.heading("g", text="✓")
        self._tree.column("name", width=130, minwidth=60)
        self._tree.column("time", width=65, minwidth=40)
        self._tree.column("score", width=42, minwidth=30)
        self._tree.column("g", width=22, minwidth=22)

        vsb = ttk.Scrollbar(ff, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.bind("<<TreeviewSelect>>", self._on_file_select)
        self._tree.tag_configure("odd", background="#2a2a2a")
        self._tree.tag_configure("even", background=C_SURFACE)

        # Nav
        nf = ttk.Frame(left)
        nf.pack(fill=tk.X)
        ttk.Button(nf, text="◄ Prev", command=self._prev).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(nf, text="Next ►", command=self._next).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))
        self._nav_lbl = ttk.Label(left, text="—", anchor=tk.CENTER)
        self._nav_lbl.pack(fill=tk.X, pady=(4, 0))

    # ── center panel ──────────────────────────────────────────────────────

    def _build_center(self):
        center = ttk.Frame(self._pane)
        self._pane.add(center, weight=1)

        self._img_frame = ttk.LabelFrame(center, text="Image", padding=4)
        self._img_frame.pack(fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(
            self._img_frame, bg=C_CANVAS, highlightthickness=0,
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas.bind("<Configure>", self._on_canvas_cfg)

        bot = ttk.Frame(center)
        bot.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(bot, text="◄", width=3, command=self._prev).pack(side=tk.LEFT)
        self._cnav_lbl = ttk.Label(bot, text="—")
        self._cnav_lbl.pack(side=tk.LEFT, padx=8)
        ttk.Button(bot, text="►", width=3, command=self._next).pack(side=tk.LEFT)

        self._mode_lbl = ttk.Label(bot, text="[Generated]", foreground=C_GREEN)
        self._mode_lbl.pack(side=tk.LEFT, padx=(20, 0))

        self._img_status = ttk.Label(bot, text="", foreground=C_DIM)
        self._img_status.pack(side=tk.RIGHT)

    # ── right panel (scrollable) ──────────────────────────────────────────

    def _build_right(self):
        right = ttk.Frame(self._pane, width=320)
        self._pane.add(right, weight=0)

        canvas = tk.Canvas(right, highlightthickness=0, width=310, bg=C_BG)
        vsb = ttk.Scrollbar(right, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>",
                    lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor=tk.NW)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._right_canvas = canvas

        # Scroll bindings on canvas & inner only (not global)
        def _mw(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind("<MouseWheel>", _mw)
        inner.bind("<MouseWheel>", _mw)
        canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))
        inner.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        inner.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        # ── File Info ──
        inf = ttk.LabelFrame(inner, text="File Info", padding=6)
        inf.pack(fill=tk.X, pady=(0, 6))
        self._info: Dict[str, ttk.Label] = {}
        for k in ["file", "camera", "settings", "time", "burst"]:
            l = ttk.Label(inf, text="—", wraplength=290)
            l.pack(anchor=tk.W, pady=1)
            self._info[k] = l

        # ── Scores ──
        scf = ttk.LabelFrame(inner, text="Scores  (0 – 5)", padding=6)
        scf.pack(fill=tk.X, pady=(0, 6))
        self._bars: Dict[str, Tuple[tk.Canvas, ttk.Label]] = {}
        for cat in SCORE_CATEGORIES:
            r = ttk.Frame(scf)
            r.pack(fill=tk.X, pady=1)
            ttk.Label(r, text=cat[:7].title(), width=8).pack(side=tk.LEFT)
            bf = ttk.Frame(r, width=120, height=12)
            bf.pack(side=tk.LEFT, padx=4)
            bf.pack_propagate(False)
            b = tk.Canvas(bf, bg=C_SURFACE3, highlightthickness=0)
            b.pack(fill=tk.BOTH, expand=True)
            vl = ttk.Label(r, text="—", width=5)
            vl.pack(side=tk.LEFT)
            self._bars[cat] = (b, vl)

        # ── Caption (original) ──
        cf = ttk.LabelFrame(inner, text="Caption (Original)", padding=6)
        cf.pack(fill=tk.X, pady=(0, 6))
        self._cap_text = tk.Text(cf, height=4, wrap=tk.WORD, state=tk.DISABLED,
                                 font=("Consolas", 9))
        self._cap_text.pack(fill=tk.X)

        # ── Filtered prompt ──
        ff = ttk.LabelFrame(inner, text="Filtered Prompt (≤ 77 words)", padding=6)
        ff.pack(fill=tk.X, pady=(0, 6))
        self._filt_text = tk.Text(ff, height=3, wrap=tk.WORD, state=tk.DISABLED,
                                  font=("Consolas", 9))
        self._filt_text.pack(fill=tk.X)

        # ── Edge Map (collapsible) ──
        self._edge_frame = ttk.LabelFrame(inner, text="Edge Map ▶", padding=6)
        self._edge_frame.pack(fill=tk.X, pady=(0, 6))
        self._edge_hdr = ttk.Label(self._edge_frame, text="Click to expand",
                                   cursor="hand2", foreground=C_DIM)
        self._edge_hdr.pack(anchor=tk.W)
        self._edge_hdr.bind("<Button-1>", self._toggle_edge)
        self._edge_body = ttk.Frame(self._edge_frame)
        self._edge_img_lbl = ttk.Label(self._edge_body, text="No edge map")
        self._edge_img_lbl.pack()
        self._edge_info_lbl = ttk.Label(self._edge_body, text="", foreground=C_DIM)
        self._edge_info_lbl.pack()

        # ── Local Generate ──
        gf = ttk.LabelFrame(inner, text="Local Generate", padding=6)
        gf.pack(fill=tk.X, pady=(0, 6))
        self._build_gen_controls(gf)

        # ── API Refine (collapsible) ──
        api_title = "API Refine ▶"
        if not (API_AVAILABLE and BACKEND_AVAILABLE):
            api_title += "  (unavailable)"
        self._api_frame = ttk.LabelFrame(inner, text=api_title, padding=6)
        self._api_frame.pack(fill=tk.X, pady=(0, 6))
        self._api_hdr = ttk.Label(self._api_frame, text="Click to expand",
                                  cursor="hand2", foreground=C_DIM)
        self._api_hdr.pack(anchor=tk.W)
        self._api_hdr.bind("<Button-1>", self._toggle_api)
        self._api_body = ttk.Frame(self._api_frame)
        self._build_api_controls(self._api_body)

    # ── generation controls (inside right panel) ──

    def _build_gen_controls(self, parent):
        # CCS
        r = ttk.Frame(parent); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="CCS", width=5).pack(side=tk.LEFT)
        self._ccs_var = tk.DoubleVar(value=DEFAULT_CCS)
        ttk.Scale(r, from_=0.5, to=1.5, variable=self._ccs_var,
                  orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self._ccs_lbl = ttk.Label(r, text=f"{DEFAULT_CCS:.2f}", width=5)
        self._ccs_lbl.pack(side=tk.LEFT)
        self._ccs_var.trace_add("write", self._slider_update)

        # CFG
        r = ttk.Frame(parent); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="CFG", width=5).pack(side=tk.LEFT)
        self._cfg_var = tk.DoubleVar(value=DEFAULT_CFG)
        ttk.Scale(r, from_=1.0, to=2.5, variable=self._cfg_var,
                  orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self._cfg_lbl = ttk.Label(r, text=f"{DEFAULT_CFG:.2f}", width=5)
        self._cfg_lbl.pack(side=tk.LEFT)
        self._cfg_var.trace_add("write", self._slider_update)

        # Steps
        r = ttk.Frame(parent); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="Steps", width=5).pack(side=tk.LEFT)
        self._steps_var = tk.IntVar(value=DEFAULT_STEPS)
        ttk.Combobox(r, textvariable=self._steps_var, values=[4, 6, 8],
                     state="readonly", width=5).pack(side=tk.LEFT, padx=4)

        # Seed
        r = ttk.Frame(parent); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="Seed", width=5).pack(side=tk.LEFT)
        self._seed_var = tk.StringVar(value=str(DEFAULT_SEED))
        ttk.Entry(r, textvariable=self._seed_var, width=10).pack(side=tk.LEFT, padx=4)
        self._rand_seed = tk.BooleanVar(value=False)
        ttk.Checkbutton(r, text="Random", variable=self._rand_seed).pack(side=tk.LEFT)

        # Negative prompt
        nf = ttk.LabelFrame(parent, text="Negative Prompt", padding=4)
        nf.pack(fill=tk.X, pady=(6, 4))
        self._neg_var = tk.StringVar(value=DEFAULT_NEGATIVE_PROMPT)
        ttk.Entry(nf, textvariable=self._neg_var).pack(fill=tk.X)

        # Buttons
        br = ttk.Frame(parent); br.pack(fill=tk.X, pady=(8, 2))
        self._gen_btn = ttk.Button(br, text="⚡ Generate", command=self._do_generate)
        self._gen_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        self._regen_btn = ttk.Button(br, text="↻ Regenerate", command=self._do_regenerate)
        self._regen_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        if not (BACKEND_AVAILABLE and GENERATION_AVAILABLE):
            self._gen_btn.config(state=tk.DISABLED)
            self._regen_btn.config(state=tk.DISABLED)

        # Presets
        pr = ttk.Frame(parent); pr.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(pr, text="Presets:", width=7).pack(side=tk.LEFT)
        for i, p in enumerate(CCS_PRESETS):
            ttk.Button(pr, text=str(i + 1), width=3,
                       command=lambda idx=i: self._apply_preset(idx)).pack(
                side=tk.LEFT, padx=1)

        # Version info
        self._ver_lbl = ttk.Label(parent, text="No generations", foreground=C_DIM)
        self._ver_lbl.pack(anchor=tk.W, pady=(4, 0))

    # ── API controls (inside right panel) ──

    def _build_api_controls(self, parent):
        can_api = API_AVAILABLE and BACKEND_AVAILABLE
        entry_state = "normal" if can_api else "disabled"

        # API Key
        r = ttk.Frame(parent); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="Key", width=5).pack(side=tk.LEFT)
        self._api_key_var = tk.StringVar()
        self._api_key_entry = ttk.Entry(
            r, textvariable=self._api_key_var, show="•", width=24,
            state=entry_state,
        )
        self._api_key_entry.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self._eye_btn = ttk.Button(
            r, text="👁", width=3, command=self._toggle_key_vis,
            state=entry_state,
        )
        self._eye_btn.pack(side=tk.LEFT)

        # Preprompt
        ttk.Label(parent, text="Preprompt:").pack(anchor=tk.W, pady=(4, 0))
        self._preprompt = tk.Text(
            parent, height=3, wrap=tk.WORD, font=("Consolas", 9),
        )
        self._preprompt.pack(fill=tk.X, pady=2)
        self._preprompt.insert("1.0", DEFAULT_PREPROMPT)
        if not can_api:
            self._preprompt.config(state=tk.DISABLED)

        # API button
        self._api_btn = ttk.Button(
            parent, text="🌐 API Regenerate", command=self._do_api,
            state="normal" if can_api else "disabled",
        )
        self._api_btn.pack(fill=tk.X, pady=(4, 2))

        # Status
        status_txt = "Ready" if can_api else "Requires view_backend.py + view_api.py"
        self._api_status = ttk.Label(parent, text=status_txt, foreground=C_DIM)
        self._api_status.pack(anchor=tk.W, pady=1)

        self._api_ver_lbl = ttk.Label(parent, text="API versions: 0", foreground=C_DIM)
        self._api_ver_lbl.pack(anchor=tk.W)

    # ── status bar ────────────────────────────────────────────────────────

    def _build_status(self):
        sf = ttk.Frame(self.root)
        sf.pack(fill=tk.X, side=tk.BOTTOM, padx=4, pady=2)

        self._status_var = tk.StringVar(value="Starting…")
        ttk.Label(sf, textvariable=self._status_var).pack(side=tk.LEFT)

        pl_text = ("Pipeline: Not loaded" if GENERATION_AVAILABLE
                   else "Pipeline: No torch/diffusers" if BACKEND_AVAILABLE
                   else "Pipeline: No view_backend.py")
        pl_fg = C_DIM if GENERATION_AVAILABLE else C_RED
        self._pipeline_lbl = ttk.Label(sf, text=pl_text, foreground=pl_fg)
        self._pipeline_lbl.pack(side=tk.LEFT, padx=(20, 0))

        self._gen_count_lbl = ttk.Label(sf, text="")
        self._gen_count_lbl.pack(side=tk.RIGHT)

    # ──────────────────────────────────────────────────────────────────────
    # Theme
    # ──────────────────────────────────────────────────────────────────────

    def _apply_theme(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except Exception:
            pass

        self.root.configure(bg=C_BG)

        s.configure(".", background=C_BG, foreground=C_TEXT,
                    fieldbackground=C_SURFACE2)
        s.configure("TFrame", background=C_BG)
        s.configure("TLabel", background=C_BG, foreground=C_TEXT)
        s.configure("TLabelframe", background=C_BG, foreground=C_TEXT)
        s.configure("TLabelframe.Label", background=C_BG, foreground=C_TEXT)
        s.configure("TButton", background=C_HOVER, foreground=C_TEXT)
        s.configure("TEntry", fieldbackground=C_SURFACE2, foreground=C_TEXT)
        s.configure("TCombobox", fieldbackground=C_SURFACE2, foreground=C_TEXT)
        s.configure("TScale", background=C_BG, troughcolor=C_HOVER)
        s.configure("TCheckbutton", background=C_BG, foreground=C_TEXT)
        s.configure("TRadiobutton", background=C_BG, foreground=C_TEXT)
        s.map("TButton", background=[("active", "#4a4a4a")])
        s.configure("Treeview", background=C_SURFACE2, foreground=C_TEXT,
                    fieldbackground=C_SURFACE2)
        s.configure("Treeview.Heading", background=C_HOVER, foreground=C_TEXT)
        s.map("Treeview", background=[("selected", C_SELECT)],
              foreground=[("selected", C_TEXT)])

        for w in [self._sess_list]:
            w.configure(bg=C_SURFACE2, fg=C_TEXT, selectbackground=C_SELECT,
                        selectforeground=C_TEXT, highlightthickness=0)
        for w in [self._cap_text, self._filt_text]:
            w.configure(bg=C_SURFACE2, fg=C_TEXT, insertbackground=C_TEXT)
        self._filt_text.configure(fg=C_ORANGE)
        self._preprompt.configure(bg=C_SURFACE2, fg=C_GREEN,
                                  insertbackground=C_TEXT)

    # ──────────────────────────────────────────────────────────────────────
    # Key Bindings
    # ──────────────────────────────────────────────────────────────────────

    def _bind_keys(self):
        self.root.bind("<Left>", lambda e: self._prev())
        self.root.bind("<Right>", lambda e: self._next())
        self.root.bind("<Home>", lambda e: self._goto(0))
        self.root.bind("<End>", lambda e: self._goto(len(self.file_list) - 1))
        self.root.bind("g", lambda e: self._do_generate())
        self.root.bind("G", lambda e: self._do_generate())
        self.root.bind("r", lambda e: self._do_regenerate())
        self.root.bind("R", lambda e: self._do_regenerate())
        self.root.bind("a", lambda e: self._do_api())
        self.root.bind("A", lambda e: self._do_api())
        self.root.bind("<space>", lambda e: self._cycle_view())
        self.root.bind("<Control-q>", lambda e: self._on_close())
        self.root.bind("<Control-Q>", lambda e: self._on_close())
        self.root.bind("<Control-u>", lambda e: self._unload_pipeline())
        self.root.bind("<Control-U>", lambda e: self._unload_pipeline())
        for i in range(5):
            self.root.bind(str(i + 1), lambda e, x=i: self._apply_preset(x))

    # ──────────────────────────────────────────────────────────────────────
    # Logging / Queue
    # ──────────────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")
        try:
            self._queue.put_nowait(("status", msg))
        except Exception:
            pass

    def _poll_queue(self):
        for _ in range(60):
            try:
                kind, data = self._queue.get_nowait()
            except queue.Empty:
                break
            self._handle_msg(kind, data)
        self.root.after(80, self._poll_queue)

    def _handle_msg(self, kind, data):
        if kind == "status":
            self._status_var.set(str(data)[:120])

        elif kind == "pipeline_status":
            self._pipeline_lbl.config(text=f"Pipeline: {data}")

        elif kind == "db_opened":
            gen, db = data
            if gen != self._load_gen:
                db.close()
                return
            if self.db:
                self.db.close()
            self.db = db

        elif kind == "sessions_loaded":
            gen, sessions = data
            if gen != self._load_gen:
                return
            self._populate_sessions(sessions)

        elif kind == "stats_loaded":
            gen, stats = data
            if gen != self._load_gen:
                return
            self._status_var.set(
                f"{stats['files']} files, "
                f"{stats.get('captioned', 0)} captioned, "
                f"{stats.get('edge_mapped', 0)} edge mapped"
            )

        elif kind == "file_list_loaded":
            gen, items = data
            if gen != self._load_gen:
                return
            self.file_list = items
            self._file_count_lbl.config(text=f"{len(items)} files")
            self._start_batch(items)

        elif kind == "generated":
            self._on_gen_done(data)

        elif kind == "api_generated":
            self._on_api_done(data)

        elif kind == "error":
            self._generating = False
            self._generating_file_id = None
            self._refresh_buttons()
            messagebox.showerror("Error", str(data))

        elif kind == "progress":
            self._status_var.set(str(data))

    # ──────────────────────────────────────────────────────────────────────
    # Database Discovery & Loading
    # ──────────────────────────────────────────────────────────────────────

    def _find_database(self):
        # Handle direct .db file argument
        if os.path.isfile(self.working_dir) and self.working_dir.endswith(".db"):
            self._db_paths = [self.working_dir]
            self.working_dir = os.path.dirname(self.working_dir) or "."
            if BACKEND_AVAILABLE:
                self.cache = GenerationCache(self.working_dir)
            else:
                self.cache = _FallbackCache(self.working_dir)
        else:
            db_files = []
            try:
                for e in os.listdir(self.working_dir):
                    if e.lower().endswith(".db"):
                        full = os.path.join(self.working_dir, e)
                        if os.path.isfile(full):
                            db_files.append(full)
            except OSError as ex:
                self._log(f"Cannot list directory: {ex}")
            db_files.sort()
            self._db_paths = db_files

        if not self._db_paths:
            self._status_var.set(f"No .db files in {self.working_dir}")
            return

        self._db_combo["values"] = [os.path.basename(f) for f in self._db_paths]

        # Prefer ingest.db
        idx = 0
        for i, f in enumerate(self._db_paths):
            if os.path.basename(f).lower() == "ingest.db":
                idx = i
                break
        self._db_combo.current(idx)
        self._open_db_async(self._db_paths[idx])

    def _on_db_select(self, _e=None):
        idx = self._db_combo.current()
        if 0 <= idx < len(self._db_paths):
            self.current_session_id = None
            self._open_db_async(self._db_paths[idx])

    def _open_db_async(self, path: str):
        """Phase 2: background thread opens DB and queries lightweight data."""
        self._load_gen += 1
        gen = self._load_gen

        # Clear current display immediately
        self._tree.delete(*self._tree.get_children())
        self.file_list = []
        self.current_details = None
        self._clear_display()
        self._status_var.set(f"Opening {os.path.basename(path)}…")

        sort_by = self._sort_var.get()
        sort_desc = sort_by == "score"
        session_id = self.current_session_id

        def worker():
            try:
                db = ViewerDB(path)
                if gen != self._load_gen:
                    db.close()
                    return
                self._queue.put(("db_opened", (gen, db)))

                sessions = db.get_sessions()
                if gen != self._load_gen:
                    return
                self._queue.put(("sessions_loaded", (gen, sessions)))

                stats = db.get_stats()
                if gen != self._load_gen:
                    return
                self._queue.put(("stats_loaded", (gen, stats)))

                items = db.get_file_list(
                    session_id=session_id,
                    sort_by=sort_by,
                    sort_desc=sort_desc,
                )
                if gen != self._load_gen:
                    return
                self._queue.put(("file_list_loaded", (gen, items)))
            except Exception as ex:
                if gen == self._load_gen:
                    self._queue.put(("error", f"Failed to open: {ex}"))

        threading.Thread(target=worker, daemon=True).start()

    def _populate_sessions(self, sessions: List[Dict]):
        self._sess_list.delete(0, tk.END)
        self._sess_list.insert(tk.END, "All Sessions")
        self._sessions = [None]
        for s in sessions:
            name = s.get("session_name", f"Session {s['session_id']}")
            cnt = s.get("file_count", "?")
            self._sess_list.insert(tk.END, f"{name} ({cnt})")
            self._sessions.append(s["session_id"])
        self._sess_list.selection_set(0)

    def _on_session_select(self, _e=None):
        sel = self._sess_list.curselection()
        if not sel:
            return
        idx = sel[0]
        self.current_session_id = (
            self._sessions[idx] if idx < len(self._sessions) else None
        )
        label = "All" if self.current_session_id is None else self._sess_list.get(idx)
        self._session_lbl.config(text=f"Session: {label}")
        self._refresh_file_list()

    # ──────────────────────────────────────────────────────────────────────
    # File List (batched insert)
    # ──────────────────────────────────────────────────────────────────────

    def _refresh_file_list(self):
        """Re-query file list async with current sort/session."""
        if not self.db:
            return
        self._load_gen += 1
        gen = self._load_gen

        self._tree.delete(*self._tree.get_children())
        self.file_list = []
        self.current_details = None
        self._clear_display()

        sort_by = self._sort_var.get()
        sort_desc = sort_by == "score"
        sid = self.current_session_id
        db = self.db

        def worker():
            try:
                items = db.get_file_list(session_id=sid,
                                         sort_by=sort_by, sort_desc=sort_desc)
                if gen == self._load_gen:
                    self._queue.put(("file_list_loaded", (gen, items)))
            except Exception as ex:
                if gen == self._load_gen:
                    self._queue.put(("error", f"Query failed: {ex}"))

        threading.Thread(target=worker, daemon=True).start()

    def _set_sort(self, col):
        self._sort_var.set(col)
        self._refresh_file_list()

    def _start_batch(self, items: List[FileListItem]):
        """Begin batched treeview insert (Phase 3)."""
        self._tree.delete(*self._tree.get_children())
        self._batch_items = items
        self._batch_offset = 0
        if items:
            self._status_var.set(f"Loading 0/{len(items)}…")
        self._do_batch()

    def _do_batch(self):
        items = self._batch_items
        off = self._batch_offset
        chunk = items[off:off + TREE_BATCH]

        if not chunk:
            n = len(items)
            self._file_count_lbl.config(text=f"{n} files")
            self._status_var.set(f"Loaded {n} files")
            self._update_gen_count()
            if items:
                self.current_index = 0
                self._tree.selection_set("0")
                self._tree.see("0")
                self._load_current()
            return

        for i, it in enumerate(chunk, start=off):
            sc = f"{it.score_overall:.1f}" if it.score_overall is not None else "—"
            g = ""
            if it.preview_path:
                if self.cache.has_generation(it.preview_path, "gen"):
                    g = "✓"
                elif self.cache.has_generation(it.preview_path, "gem"):
                    g = "◆"
            ts = self._time_short(it.capture_time)
            tag = "odd" if i % 2 else "even"
            self._tree.insert("", tk.END, iid=str(i),
                              values=(it.file_name, ts, sc, g), tags=(tag,))

        self._batch_offset = off + len(chunk)
        self._status_var.set(f"Loading {self._batch_offset}/{len(items)}…")
        self.root.after(1, self._do_batch)

    @staticmethod
    def _time_short(ct: str) -> str:
        if not ct:
            return ""
        try:
            return datetime.fromisoformat(
                ct.replace("Z", "+00:00")).strftime("%H:%M:%S")
        except Exception:
            return ct[:8]

    @staticmethod
    def _time_full(ct: str) -> str:
        if not ct:
            return "—"
        try:
            return datetime.fromisoformat(
                ct.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return ct

    # ──────────────────────────────────────────────────────────────────────
    # Navigation
    # ──────────────────────────────────────────────────────────────────────

    def _on_file_select(self, _e=None):
        sel = self._tree.selection()
        if not sel:
            return
        try:
            idx = int(sel[0])
        except (ValueError, IndexError):
            return
        if idx != self.current_index and 0 <= idx < len(self.file_list):
            self.current_index = idx
            self._load_current()

    def _prev(self):
        if self.file_list and self.current_index > 0:
            self._goto(self.current_index - 1)

    def _next(self):
        if self.file_list and self.current_index < len(self.file_list) - 1:
            self._goto(self.current_index + 1)

    def _goto(self, idx):
        if 0 <= idx < len(self.file_list):
            self.current_index = idx
            iid = str(idx)
            self._tree.selection_set(iid)
            self._tree.see(iid)
            self._load_current()

    def _load_current(self):
        if not self.file_list or self.current_index >= len(self.file_list):
            return
        item = self.file_list[self.current_index]
        if self.db:
            try:
                self.current_details = self.db.get_file_details(item.file_id)
            except Exception as ex:
                self._log(f"Detail load error: {ex}")
                self.current_details = None
                return
        if self.current_details:
            self._display()

    # ──────────────────────────────────────────────────────────────────────
    # Display
    # ──────────────────────────────────────────────────────────────────────

    def _clear_display(self):
        self._nav_lbl.config(text="—")
        self._cnav_lbl.config(text="—")
        for l in self._info.values():
            l.config(text="—")
        for cat in SCORE_CATEGORIES:
            b, l = self._bars[cat]
            b.delete("all"); l.config(text="—")
        for w in [self._cap_text, self._filt_text]:
            w.config(state=tk.NORMAL)
            w.delete("1.0", tk.END)
            w.config(state=tk.DISABLED)
        self._canvas.delete("all")
        self._ver_lbl.config(text="No generations")
        self._img_status.config(text="")
        self._img_frame.config(text="Image")

    def _display(self):
        fd = self.current_details
        if not fd:
            return
        nav = f"{self.current_index + 1} / {len(self.file_list)}"
        self._nav_lbl.config(text=nav)
        self._cnav_lbl.config(text=nav)

        self._show_info(fd)
        self._show_scores(fd)
        self._show_caption(fd)
        if self._edge_expanded:
            self._show_edge(fd)
        self._show_versions(fd)
        self._refresh_buttons()
        self._show_image()

    def _show_info(self, fd):
        self._info["file"].config(text=fd.file_name or "—")
        cam = f"{fd.camera_make} {fd.camera_model}".strip()
        self._info["camera"].config(text=cam or "Unknown")

        parts = []
        if fd.iso:
            parts.append(f"ISO {fd.iso}")
        if fd.aperture:
            parts.append(f"f/{fd.aperture}")
        if fd.shutter_speed:
            parts.append(fd.shutter_speed)
        if fd.focal_length_mm:
            parts.append(f"{fd.focal_length_mm}mm")
        self._info["settings"].config(text=" │ ".join(parts) if parts else "—")
        self._info["time"].config(text=self._time_full(fd.capture_time))

        burst = ""
        if fd.burst_id is not None:
            if fd.burst_index is not None and fd.burst_frame_count is not None:
                burst = f"Burst #{fd.burst_index} ({fd.burst_frame_count} frames)"
            else:
                burst = f"Burst #{fd.burst_id}"
        else:
            burst = "No burst"
        if fd.focus_distance_m:
            burst += f" │ Focus: {fd.focus_distance_m}m"
        self._info["burst"].config(text=burst)

    def _show_scores(self, fd):
        self.root.after(10, lambda: self._draw_bars(fd))

    def _draw_bars(self, fd):
        for cat in SCORE_CATEGORIES:
            bar, lbl = self._bars[cat]
            bar.delete("all")
            val = fd.scores.get(cat)
            if val is None:
                lbl.config(text="—")
                continue
            norm = max(0.0, min(1.0, val / SCORE_MAX))
            # Green-to-red gradient
            if norm < 0.5:
                r, g = 255, int(255 * norm * 2)
            else:
                r, g = int(255 * (1 - norm) * 2), 255
            r, g = max(0, min(255, r)), max(0, min(255, g))
            color = f"#{r:02x}{g:02x}00"
            bar.update_idletasks()
            w = bar.winfo_width()
            if w <= 1:
                w = 100
            bar.create_rectangle(0, 0, w * norm, 20, fill=color, outline="")
            lbl.config(text=f"{val:.2f}")

    def _show_caption(self, fd):
        self._cap_text.config(state=tk.NORMAL)
        self._cap_text.delete("1.0", tk.END)
        self._cap_text.insert("1.0", fd.caption or "No caption")
        self._cap_text.config(state=tk.DISABLED)

        filtered = self.hesitancy.filter(fd.caption) if fd.caption else ""
        wc = len(filtered.split()) if filtered else 0
        self._filt_text.config(state=tk.NORMAL)
        self._filt_text.delete("1.0", tk.END)
        if filtered:
            self._filt_text.insert("1.0", f"{filtered}\n[{wc} words]")
        else:
            self._filt_text.insert("1.0", "No caption")
        self._filt_text.config(state=tk.DISABLED)

    # ── edge map ──

    def _toggle_edge(self, _e=None):
        self._edge_expanded = not self._edge_expanded
        if self._edge_expanded:
            self._edge_frame.config(text="Edge Map ▼")
            self._edge_body.pack(fill=tk.X, pady=(4, 0))
            self._edge_hdr.config(text="Click to collapse")
            if self.current_details:
                self._show_edge(self.current_details)
        else:
            self._edge_frame.config(text="Edge Map ▶")
            self._edge_body.pack_forget()
            self._edge_hdr.config(text="Click to expand")

    def _show_edge(self, fd):
        if not fd.edge_map_data:
            self._edge_img_lbl.config(image="", text="No edge map")
            self._edge_info_lbl.config(text="")
            return
        img = decode_edge_map(fd.edge_map_data, fd.edge_map_encoding)
        if img is None:
            self._edge_img_lbl.config(image="", text="Decode failed")
            return
        t = img.copy()
        t.thumbnail((200, 200), Image.LANCZOS)
        photo = ImageTk.PhotoImage(t)
        self._edge_img_lbl.config(image=photo, text="")
        self._keep_ref(photo)
        if fd.edge_map_encoding == "base64":
            sz = len(fd.edge_map_data) * 3 // 4
        else:
            sz = len(fd.edge_map_data) // 2
        self._edge_info_lbl.config(
            text=f"{img.width}×{img.height}  •  {sz / 1024:.1f} KB")

    # ── API panel toggle ──

    def _toggle_api(self, _e=None):
        self._api_expanded = not self._api_expanded
        title_base = "API Refine"
        if not (API_AVAILABLE and BACKEND_AVAILABLE):
            title_base += "  (unavailable)"
        if self._api_expanded:
            self._api_frame.config(text=f"{title_base} ▼")
            self._api_body.pack(fill=tk.X, pady=(4, 0))
            self._api_hdr.config(text="Click to collapse")
            if self.current_details:
                self._refresh_api_info()
        else:
            self._api_frame.config(text=f"{title_base} ▶")
            self._api_body.pack_forget()
            self._api_hdr.config(text="Click to expand")

    def _toggle_key_vis(self):
        self._api_key_visible = not self._api_key_visible
        self._api_key_entry.config(show="" if self._api_key_visible else "•")
        self._eye_btn.config(text="🔒" if self._api_key_visible else "👁")

    def _refresh_api_info(self):
        if not self.current_details:
            return
        fd = self.current_details
        cnt = self.cache.get_version_count(fd.preview_path, "gem")
        self._api_ver_lbl.config(text=f"API versions: {cnt}")

    # ── version info ──

    def _show_versions(self, fd):
        if not fd.preview_path:
            self._ver_lbl.config(text="No preview path")
            return
        gen_n = self.cache.get_version_count(fd.preview_path, "gen")
        gem_n = self.cache.get_version_count(fd.preview_path, "gem")
        parts = []
        if gen_n:
            parts.append(f"Local: {gen_n}")
        if gem_n:
            parts.append(f"API: {gem_n}")
        self._ver_lbl.config(text=" │ ".join(parts) if parts else "No generations")
        if self._api_expanded:
            self._api_ver_lbl.config(text=f"API versions: {gem_n}")

    # ── button states ──

    def _refresh_buttons(self):
        can_gen = BACKEND_AVAILABLE and GENERATION_AVAILABLE
        fd = self.current_details

        if self._generating or not fd:
            self._gen_btn.config(state=tk.DISABLED)
            self._regen_btn.config(state=tk.DISABLED)
            self._api_btn.config(state=tk.DISABLED)
            return

        has_data = bool(fd.edge_map_data and fd.caption)
        has_local = (bool(fd.preview_path) and
                     self.cache.has_generation(fd.preview_path, "gen"))

        # Local buttons
        if can_gen and has_data:
            if has_local:
                self._gen_btn.config(state=tk.DISABLED)
                self._regen_btn.config(state=tk.NORMAL)
            else:
                self._gen_btn.config(state=tk.NORMAL)
                self._regen_btn.config(state=tk.DISABLED)
        else:
            self._gen_btn.config(state=tk.DISABLED)
            self._regen_btn.config(state=tk.DISABLED)

        # API button
        can_api = (API_AVAILABLE and BACKEND_AVAILABLE and has_local and
                   bool(self._api_key_var.get().strip()))
        self._api_btn.config(state=tk.NORMAL if can_api else tk.DISABLED)

    # ── image display ──

    def _keep_ref(self, photo):
        self._photo_refs.append(photo)
        if len(self._photo_refs) > MAX_PHOTO_REFS:
            self._photo_refs = self._photo_refs[-MAX_PHOTO_REFS:]

    def _show_image(self):
        fd = self.current_details
        if not fd:
            return

        if self._view_mode == "generated":
            path = self.cache.get_latest(fd.preview_path, "gen") if fd.preview_path else None
            if path and os.path.isfile(path):
                try:
                    self._render(Image.open(path))
                    self._img_status.config(text=os.path.basename(path),
                                            foreground=C_GREEN)
                    self._img_frame.config(text="Generated Image")
                    return
                except Exception:
                    pass
            # Fallback to edge
            if fd.edge_map_data:
                img = decode_edge_map(fd.edge_map_data, fd.edge_map_encoding)
                if img:
                    self._render(img)
                    self._img_status.config(text="Edge map (not generated)",
                                            foreground=C_YELLOW)
                    self._img_frame.config(text="Edge Map  — press G to generate")
                    return

        elif self._view_mode == "api":
            path = self.cache.get_latest(fd.preview_path, "gem") if fd.preview_path else None
            if path and os.path.isfile(path):
                try:
                    self._render(Image.open(path))
                    self._img_status.config(text=os.path.basename(path),
                                            foreground=C_ACCENT)
                    self._img_frame.config(text="API Refined")
                    return
                except Exception:
                    pass
            # Fallback: show local gen or edge
            gen_path = self.cache.get_latest(fd.preview_path, "gen") if fd.preview_path else None
            if gen_path and os.path.isfile(gen_path):
                try:
                    self._render(Image.open(gen_path))
                    self._img_status.config(text="Local gen (no API version)",
                                            foreground=C_YELLOW)
                    self._img_frame.config(text="Local Gen  — press A for API")
                    return
                except Exception:
                    pass

        elif self._view_mode == "edge":
            if fd.edge_map_data:
                img = decode_edge_map(fd.edge_map_data, fd.edge_map_encoding)
                if img:
                    self._render(img)
                    self._img_status.config(
                        text=f"Edge map {img.width}×{img.height}",
                        foreground=C_YELLOW)
                    self._img_frame.config(text="Edge Map")
                    return

        # Nothing
        self._canvas.delete("all")
        cw = max(self._canvas.winfo_width(), 100)
        ch = max(self._canvas.winfo_height(), 100)
        self._canvas.create_text(cw // 2, ch // 2, text="No image data",
                                 fill=C_DIM, font=("Sans", 14))
        self._img_status.config(text="No data", foreground=C_DIM)
        self._img_frame.config(text="No Image")

    def _render(self, img: Image.Image):
        self._canvas.update_idletasks()
        cw = max(self._canvas.winfo_width(), 100)
        ch = max(self._canvas.winfo_height(), 100)
        iw, ih = img.size
        if iw <= 0 or ih <= 0:
            return
        scale = min(cw / iw, ch / ih, 1.0)
        if scale < 1.0:
            img = img.resize((max(int(iw * scale), 1),
                              max(int(ih * scale), 1)), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self._canvas.delete("all")
        self._canvas.create_image(cw // 2, ch // 2, image=photo, anchor=tk.CENTER)
        self._keep_ref(photo)

    def _on_canvas_cfg(self, _e):
        if self._resize_after_id:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(150, self._do_resize)

    def _do_resize(self):
        self._resize_after_id = None
        if self.current_details:
            self._show_image()

    def _cycle_view(self):
        """Space: cycle through view modes, skipping those with no data."""
        modes = ["generated", "edge", "api"]
        try:
            ci = modes.index(self._view_mode)
        except ValueError:
            ci = 0
        for offset in range(1, len(modes) + 1):
            candidate = modes[(ci + offset) % len(modes)]
            if self._mode_available(candidate):
                self._view_mode = candidate
                break
        self._update_mode_label()
        self._show_image()

    def _mode_available(self, mode: str) -> bool:
        fd = self.current_details
        if not fd:
            return False
        if mode == "generated":
            return (bool(fd.preview_path) and
                    self.cache.has_generation(fd.preview_path, "gen"))
        if mode == "edge":
            return bool(fd.edge_map_data)
        if mode == "api":
            return (bool(fd.preview_path) and
                    self.cache.has_generation(fd.preview_path, "gem"))
        return False

    def _update_mode_label(self):
        labels = {
            "generated": ("[Generated]", C_GREEN),
            "edge":      ("[Edge Map]",  C_YELLOW),
            "api":       ("[API Refined]", C_ACCENT),
        }
        txt, clr = labels.get(self._view_mode, ("[?]", C_DIM))
        self._mode_lbl.config(text=txt, foreground=clr)

    def _update_gen_count(self):
        local = self.cache.count_generated("gen")
        api = self.cache.count_generated("gem")
        total = len(self.file_list)
        parts = [f"Local: {local}/{total}"]
        if api:
            parts.append(f"API: {api}")
        self._gen_count_lbl.config(text="  │  ".join(parts))

    # ──────────────────────────────────────────────────────────────────────
    # Slider & Preset helpers
    # ──────────────────────────────────────────────────────────────────────

    def _slider_update(self, *_):
        try:
            self._ccs_lbl.config(text=f"{self._ccs_var.get():.2f}")
        except Exception:
            pass
        try:
            self._cfg_lbl.config(text=f"{self._cfg_var.get():.2f}")
        except Exception:
            pass

    def _apply_preset(self, idx):
        if 0 <= idx < len(CCS_PRESETS):
            p = CCS_PRESETS[idx]
            self._ccs_var.set(p["ccs"])
            self._cfg_var.set(p["cfg"])
            self._log(f"Preset {idx + 1}: CCS={p['ccs']}, CFG={p['cfg']}")

    # ──────────────────────────────────────────────────────────────────────
    # Local Generation
    # ──────────────────────────────────────────────────────────────────────

    def _do_generate(self):
        if self._generating or not (BACKEND_AVAILABLE and GENERATION_AVAILABLE):
            return
        fd = self.current_details
        if not fd:
            return
        if not fd.edge_map_data:
            messagebox.showwarning("Missing", "No edge map data.")
            return
        if not fd.caption:
            messagebox.showwarning("Missing", "No caption.")
            return
        if fd.preview_path and self.cache.has_generation(fd.preview_path, "gen"):
            self._log("Already generated — use Regenerate (R)")
            return
        self._start_local_gen(fd)

    def _do_regenerate(self):
        if self._generating or not (BACKEND_AVAILABLE and GENERATION_AVAILABLE):
            return
        fd = self.current_details
        if not fd or not fd.edge_map_data or not fd.caption:
            messagebox.showwarning("Missing", "Edge map or caption missing.")
            return
        self._start_local_gen(fd)

    def _start_local_gen(self, fd):
        ccs = self._ccs_var.get()
        cfg = self._cfg_var.get()
        steps = self._steps_var.get()
        neg = self._neg_var.get().strip() or DEFAULT_NEGATIVE_PROMPT
        if self._rand_seed.get():
            seed = random.randint(0, 2**32 - 1)
            self._seed_var.set(str(seed))
        else:
            try:
                seed = int(self._seed_var.get())
            except ValueError:
                seed = DEFAULT_SEED
                self._seed_var.set(str(seed))

        prompt = self.hesitancy.filter(fd.caption)

        edge_img = decode_edge_map(fd.edge_map_data, fd.edge_map_encoding)
        if edge_img is None:
            messagebox.showerror("Error", "Failed to decode edge map.")
            return
        edge_img = edge_img.copy()

        self._generating = True
        self._generating_file_id = fd.file_id
        self._refresh_buttons()

        pp = fd.preview_path
        fid = fd.file_id

        self._progress_dots = 0
        self._animate("Generating")

        def worker():
            try:
                if not self.generator.is_loaded():
                    self._queue.put(("pipeline_status", "Loading…"))
                    self._queue.put(("status", "Loading pipeline (first time)…"))
                    if not self.generator.load():
                        self._queue.put(("error", "Pipeline load failed"))
                        return
                    self._queue.put(("pipeline_status", "Ready"))

                self._queue.put(("status",
                    f"Generating CCS={ccs:.2f} CFG={cfg:.2f} seed={seed}…"))

                result = self.generator.generate(
                    prompt=prompt, negative_prompt=neg, edge_image=edge_img,
                    ccs=ccs, cfg=cfg, steps=steps, seed=seed,
                )
                if result is None:
                    self._queue.put(("error", "Generation returned None"))
                    return

                out = self.cache.save_new(pp, result, prefix="gen")
                self._queue.put(("generated", (pp, out, fid)))
                self._queue.put(("status", f"Saved: {os.path.basename(out)}"))
            except Exception as ex:
                self._queue.put(("error", str(ex)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_gen_done(self, data):
        pp, out_path, fid = data
        self._generating = False
        self._generating_file_id = None

        # Update tree checkmark
        for i, it in enumerate(self.file_list):
            if it.preview_path == pp:
                try:
                    self._tree.set(str(i), "g", "✓")
                except Exception:
                    pass
                break

        if self.current_details and self.current_details.file_id == fid:
            self._show_versions(self.current_details)
            self._refresh_buttons()
            self._view_mode = "generated"
            self._update_mode_label()
            self._show_image()

        self._update_gen_count()

    # ──────────────────────────────────────────────────────────────────────
    # API Refinement
    # ──────────────────────────────────────────────────────────────────────

    def _do_api(self):
        if self._generating or not (API_AVAILABLE and BACKEND_AVAILABLE):
            return
        fd = self.current_details
        if not fd:
            return
        if not fd.preview_path:
            return

        api_key = self._api_key_var.get().strip()
        if not api_key:
            messagebox.showwarning("API Key", "Enter your Gemini API key first.")
            if not self._api_expanded:
                self._toggle_api()
            return

        gen_path = self.cache.get_latest(fd.preview_path, "gen")
        if not gen_path or not os.path.isfile(gen_path):
            messagebox.showwarning("No Local Gen",
                                   "Generate locally first (G), then use API.")
            return

        try:
            sd_image = Image.open(gen_path).copy()
        except Exception as ex:
            messagebox.showerror("Error", f"Can't load generated image: {ex}")
            return

        prompt = self.hesitancy.filter(fd.caption) if fd.caption else ""
        preprompt = self._preprompt.get("1.0", tk.END).strip()

        self._generating = True
        self._generating_file_id = fd.file_id
        self._refresh_buttons()

        pp = fd.preview_path
        fid = fd.file_id

        self._progress_dots = 0
        self._animate("API calling")

        def worker():
            try:
                refiner = GeminiRefiner(log=self._log)
                result = refiner.refine(
                    api_key=api_key,
                    preprompt=preprompt,
                    generation_prompt=prompt,
                    source_image=sd_image,
                )
                if result is None:
                    self._queue.put(("error", "API returned no image"))
                    return

                out = self.cache.save_new(pp, result, prefix="gem")
                self._queue.put(("api_generated", (pp, out, fid)))
                self._queue.put(("status", f"API saved: {os.path.basename(out)}"))
            except Exception as ex:
                self._queue.put(("error", f"API failed: {ex}"))

        threading.Thread(target=worker, daemon=True).start()

    def _on_api_done(self, data):
        pp, out_path, fid = data
        self._generating = False
        self._generating_file_id = None

        # Update tree mark
        for i, it in enumerate(self.file_list):
            if it.preview_path == pp:
                try:
                    cur = self._tree.set(str(i), "g")
                    if "✓" not in cur:
                        self._tree.set(str(i), "g", "◆")
                except Exception:
                    pass
                break

        if self.current_details and self.current_details.file_id == fid:
            self._show_versions(self.current_details)
            self._refresh_buttons()
            self._view_mode = "api"
            self._update_mode_label()
            self._show_image()

        self._update_gen_count()

    # ── progress animation ──

    def _animate(self, prefix="Working"):
        if self._generating:
            self._progress_dots = (self._progress_dots + 1) % 4
            self._queue.put_nowait(("progress", f"{prefix}{'.' * self._progress_dots}"))
            self.root.after(300, self._animate, prefix)

    # ──────────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────────

    def _on_close(self):
        if self.generator and hasattr(self.generator, 'is_loading') and self.generator.is_loading():
            if not messagebox.askyesno("Loading", "Pipeline loading. Exit?"):
                return
        if self.db:
            self.db.close()
            self.db = None
        if self.generator and hasattr(self.generator, 'is_loaded') and self.generator.is_loaded():
            self.generator.unload()
        self.root.destroy()


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    working_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

    if os.path.isfile(working_dir) and working_dir.lower().endswith(".db"):
        pass  # handled by _find_database
    elif not os.path.isdir(working_dir):
        print(f"Error: Not found: {working_dir}")
        print(f"Usage: python {sys.argv[0]} [working_directory]")
        print(f"       python {sys.argv[0]} [path/to/database.db]")
        sys.exit(1)

    root = tk.Tk()
    _app = ViewerApp(root, working_dir)
    root.mainloop()


if __name__ == "__main__":
    main()