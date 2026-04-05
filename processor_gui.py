#!/usr/bin/env python3
"""
processor_gui.py — GUI front-end for photo processor pipeline.

Features:
  • new session from source folder  
  • resume session from existing DB (picks up where it left off)
  • pause / resume during AI phases (model stays in VRAM)
  • preview path fixup for portability (absolute → relative)
  • drive type + network status indicators
  • dependency check via dialog prompts
  • elapsed timer, phase indicators, keyboard shortcuts
  • log export

All paths anchored to SCRIPT_DIR (where processor_logic.py lives).
"""

import os
import sys
import time
import queue
import sqlite3
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from datetime import datetime
from typing import Optional, List, Dict

from processor_logic import (
    ProcessorEngine, ProcessorDB,
    MODEL_PT_PATH, DEFAULT_DB_PATH, DEFAULT_PREVIEW_DIR, SCRIPT_DIR,
    SIBLING_SCRIPTS, HAS_DIAG, _HF_ONLINE, _resolve_preview_path,
)

try:
    from hdd_diagnostics import DriveMonitor
except ImportError:
    DriveMonitor = None

WINDOW_TITLE = "Photo Processor — Nikon Z30 / Sony A7R4"
WINDOW_GEOM = "1080x880"
WINDOW_MIN = (840, 640)

PHASE_NAMES = [
    "Deps", "Scan", "Metadata", "Bursts",
    "Previews", "Scoring", "Captioning", "EdgeMap",
]
PHASE_MAP = {
    "deps": "deps", "scan": "scan", "metadata": "metadata",
    "bursts": "bursts", "previews": "previews",
    "scoring": "scoring", "captioning": "captioning",
    "edge_map": "edgemap", "offline_check": "scan",
}


# ════════════════════════════════════════════════════════════════════════
# SESSION PICKER DIALOG
# ════════════════════════════════════════════════════════════════════════

class SessionPickerDialog(tk.Toplevel):
    """Modal dialog listing sessions with pending work."""

    def __init__(self, parent, sessions: List[Dict]):
        super().__init__(parent)
        self.result: Optional[int] = None
        self.sessions = sessions
        self.title("Resume Session")
        self.geometry("780x380")
        self.minsize(620, 260)
        self.transient(parent)
        self.grab_set()

        self._build()

        self.update_idletasks()
        px = parent.winfo_x() + (parent.winfo_width()
                                  - self.winfo_width()) // 2
        py = parent.winfo_y() + (parent.winfo_height()
                                  - self.winfo_height()) // 2
        self.geometry(f"+{max(px,0)}+{max(py,0)}")
        self.bind("<Escape>", lambda e: self.destroy())
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _build(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            top, text="Sessions with pending work:",
            font=("Sans", 10, "bold"),
        ).pack(anchor=tk.W, pady=(0, 6))

        cols = ("id", "name", "files", "scored",
                "captioned", "edgemap", "status")
        headings = {
            "id": ("ID", 40), "name": ("Session", 160),
            "files": ("Files", 55), "scored": ("Scored", 75),
            "captioned": ("Captioned", 80),
            "edgemap": ("EdgeMap", 75), "status": ("Status", 80),
        }

        tree_frame = ttk.Frame(top)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(
            tree_frame, columns=cols,
            show="headings", height=8, selectmode="browse")
        for c in cols:
            txt, w = headings[c]
            self.tree.heading(c, text=txt)
            self.tree.column(c, width=w, minwidth=35)
        vs = ttk.Scrollbar(
            tree_frame, orient=tk.VERTICAL,
            command=self.tree.yview)
        self.tree.configure(yscrollcommand=vs.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vs.pack(side=tk.RIGHT, fill=tk.Y)

        for s in self.sessions:
            p = s["previews"] or 1
            self.tree.insert("", tk.END, values=(
                s["session_id"], s["name"], s["files"],
                f"{s['scored']}/{p}",
                f"{s['captioned']}/{p}",
                f"{s['edge_mapped']}/{p}",
                s["status"]))

        # detail label
        self._detail_var = tk.StringVar()
        ttk.Label(
            top, textvariable=self._detail_var,
            font=("Monospace", 8), foreground="gray",
        ).pack(anchor=tk.W, pady=(6, 0))

        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # buttons
        btn = ttk.Frame(self, padding=(10, 6))
        btn.pack(fill=tk.X)
        ttk.Button(
            btn, text="Resume Selected",
            command=self._do_select,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            btn, text="Cancel",
            command=self.destroy,
        ).pack(side=tk.RIGHT)

        # auto-select first
        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children[0])
            self._on_select(None)

    def _on_select(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        sid = int(self.tree.item(sel[0], "values")[0])
        s = next((x for x in self.sessions
                  if x["session_id"] == sid), None)
        if s:
            miss = s.get("missing_count", 0)
            src = s.get("source", "?")
            if len(src) > 70:
                src = "…" + src[-67:]
            self._detail_var.set(
                f"Source: {src}   "
                f"Previews missing: {miss}")

    def _do_select(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning(
                "Select", "Select a session first.",
                parent=self)
            return
        self.result = int(self.tree.item(sel[0], "values")[0])
        self.destroy()


# ════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ════════════════════════════════════════════════════════════════════════

class ProcessorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_GEOM)
        self.root.minsize(*WINDOW_MIN)

        self._q: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._engine: Optional[ProcessorEngine] = None

        self._db_path = DEFAULT_DB_PATH
        self._timer_running = False
        self._start_time = 0.0

        self._build_ui()
        self._bind_keys()
        self._poll_queue()
        self._startup()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ────────────────────────────────────────────────────────────────
    # UI CONSTRUCTION
    # ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        m = ttk.Frame(self.root, padding=10)
        m.pack(fill=tk.BOTH, expand=True)

        # row 1 — folder
        f1 = ttk.LabelFrame(m, text="Source Folder (NEF / ARW)",
                             padding=8)
        f1.pack(fill=tk.X, pady=(0, 6))
        self._folder_var = tk.StringVar()
        ttk.Entry(f1, textvariable=self._folder_var,
                  font=("Monospace", 10),
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True,
                         padx=(0, 8))
        ttk.Button(f1, text="Browse…",
                   command=self._browse_folder).pack(side=tk.RIGHT)

        # row 2 — options
        f2 = ttk.LabelFrame(m, text="Options", padding=8)
        f2.pack(fill=tk.X, pady=(0, 6))
        self._extract_all_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            f2, text="Extract ALL previews",
            variable=self._extract_all_var,
            command=self._on_extract_toggle,
        ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(f2, text="Filter:").pack(side=tk.LEFT, padx=(0, 4))
        self._filter_var = tk.StringVar()
        self._filter_entry = ttk.Entry(
            f2, textvariable=self._filter_var,
            width=24, state=tk.DISABLED)
        self._filter_entry.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(f2, text="(future)",
                  foreground="gray").pack(side=tk.LEFT)

        # row 3 — info bar
        f3 = ttk.Frame(m)
        f3.pack(fill=tk.X, pady=(0, 6))

        self._camera_var = tk.StringVar(
            value="Nikon Z30 (.NEF) · Sony A7R4 (.ARW)")
        ttk.Label(f3, textvariable=self._camera_var,
                  font=("Sans", 9)).pack(side=tk.LEFT)

        self._elapsed_var = tk.StringVar(value="")
        ttk.Label(f3, textvariable=self._elapsed_var,
                  font=("Monospace", 9),
                  foreground="#569cd6").pack(side=tk.RIGHT, padx=(12, 0))

        self._net_var = tk.StringVar(
            value=("🌐 Online" if _HF_ONLINE else "📴 Offline"))
        net_fg = "#4ec9b0" if _HF_ONLINE else "#cca700"
        ttk.Label(f3, textvariable=self._net_var,
                  font=("Sans", 9),
                  foreground=net_fg).pack(side=tk.RIGHT, padx=(12, 0))

        self._drive_var = tk.StringVar(value="")
        self._drive_lbl = ttk.Label(
            f3, textvariable=self._drive_var,
            font=("Sans", 9), foreground="gray")
        self._drive_lbl.pack(side=tk.RIGHT, padx=(12, 0))

        # row 4 — action buttons
        b = ttk.Frame(m)
        b.pack(fill=tk.X, pady=(0, 6))

        self._run_btn = ttk.Button(
            b, text="▶ Run All (F5)", width=14,
            command=self._start_new)
        self._run_btn.pack(side=tk.LEFT, padx=(0, 5))

        self._pause_btn = ttk.Button(
            b, text="⏸ Pause (F6)", width=14,
            command=self._toggle_pause, state=tk.DISABLED)
        self._pause_btn.pack(side=tk.LEFT, padx=(0, 5))

        self._cancel_btn = ttk.Button(
            b, text="⏹ Cancel (Esc)", width=14,
            command=self._do_cancel, state=tk.DISABLED)
        self._cancel_btn.pack(side=tk.LEFT, padx=(0, 5))

        self._resume_btn = ttk.Button(
            b, text="↻ Resume (F7)", width=14,
            command=self._do_resume)
        self._resume_btn.pack(side=tk.LEFT, padx=(0, 5))

        ttk.Separator(b, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8)

        self._open_src_btn = ttk.Button(
            b, text="📁 Source", command=self._open_source,
            state=tk.DISABLED)
        self._open_src_btn.pack(side=tk.LEFT, padx=(0, 5))

        self._open_prev_btn = ttk.Button(
            b, text="🖼️ Previews", command=self._open_previews,
            state=tk.DISABLED)
        self._open_prev_btn.pack(side=tk.LEFT, padx=(0, 5))

        self._export_btn = ttk.Button(
            b, text="💾 Log", command=self._export_log)
        self._export_btn.pack(side=tk.RIGHT, padx=(5, 0))

        self._open_db_btn = ttk.Button(
            b, text="🗄️ DB", command=self._open_db_loc)
        self._open_db_btn.pack(side=tk.RIGHT, padx=(5, 0))

        # row 5 — progress
        pf = ttk.Frame(m)
        pf.pack(fill=tk.X, pady=(0, 5))
        self._status_var = tk.StringVar(
            value="Ready — select a folder or resume a session")
        ttk.Label(pf, textvariable=self._status_var,
                  font=("Sans", 9), width=50,
                  anchor=tk.W).pack(side=tk.LEFT)
        self._progress_var = tk.DoubleVar()
        ttk.Progressbar(
            pf, variable=self._progress_var,
            maximum=100, length=280,
        ).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))

        # row 6 — phase indicators
        ph = ttk.Frame(m)
        ph.pack(fill=tk.X, pady=(0, 5))
        self._phase_lbls: Dict[str, ttk.Label] = {}
        for name in PHASE_NAMES:
            lbl = ttk.Label(ph, text=f"○ {name}",
                            foreground="gray")
            lbl.pack(side=tk.LEFT, padx=(0, 10))
            self._phase_lbls[name.lower()] = lbl

        # row 7 — script indicators
        sf = ttk.Frame(m)
        sf.pack(fill=tk.X, pady=(0, 6))
        self._script_lbls: Dict[str, ttk.Label] = {}
        for sn in SIBLING_SCRIPTS:
            lbl = ttk.Label(sf, text=f"? {sn}",
                            font=("Monospace", 8),
                            foreground="gray")
            lbl.pack(side=tk.LEFT, padx=(0, 14))
            self._script_lbls[sn] = lbl

        # row 8 — log
        lf = ttk.LabelFrame(m, text="Log", padding=5)
        lf.pack(fill=tk.BOTH, expand=True)
        self._log_w = scrolledtext.ScrolledText(
            lf, height=24, font=("Monospace", 9),
            state=tk.DISABLED, wrap=tk.WORD,
            bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="#fff")
        self._log_w.pack(fill=tk.BOTH, expand=True)

        for tag, fg, bold in [
            ("error",   "#f14c4c", False),
            ("warning", "#cca700", False),
            ("success", "#4ec9b0", False),
            ("phase",   "#569cd6", True),
            ("caption", "#ce9178", False),
            ("dbwrite", "#b5cea8", False),
            ("paused",  "#dcdcaa", False),
            ("drive",   "#c586c0", False),
        ]:
            kw = {"foreground": fg}
            if bold:
                kw["font"] = ("Monospace", 9, "bold")
            self._log_w.tag_configure(tag, **kw)

    def _bind_keys(self):
        self.root.bind("<F5>", lambda e: self._start_new())
        self.root.bind("<F6>", lambda e: self._toggle_pause())
        self.root.bind("<F7>", lambda e: self._do_resume())
        self.root.bind("<Escape>", lambda e: self._do_cancel())
        self.root.bind("<Control-o>", lambda e: self._browse_folder())
        self.root.bind("<Control-l>", lambda e: self._export_log())

    def _on_extract_toggle(self):
        st = tk.DISABLED if self._extract_all_var.get() else tk.NORMAL
        self._filter_entry.config(state=st)

    # ────────────────────────────────────────────────────────────────
    # LOGGING
    # ────────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._q.put(f"[{ts}] {msg}")

    def _poll_queue(self):
        while True:
            try:
                msg = self._q.get_nowait()
            except queue.Empty:
                break
            self._log_w.config(state=tk.NORMAL)
            tag = self._pick_tag(msg)
            self._log_w.insert(
                tk.END, msg + "\n",
                (tag,) if tag else ())
            self._log_w.see(tk.END)
            self._log_w.config(state=tk.DISABLED)

        # live pause-button text sync
        if self._engine is not None:
            if self._engine.is_paused:
                self._pause_btn.config(text="▶ Resume (F6)")
            else:
                self._pause_btn.config(text="⏸ Pause (F6)")

        self.root.after(100, self._poll_queue)

    @staticmethod
    def _pick_tag(msg: str) -> Optional[str]:
        if "ERROR" in msg or "Error" in msg:
            return "error"
        if "WARN" in msg or "⚠️" in msg:
            return "warning"
        if "Phase" in msg:
            return "phase"
        if "COMPLETE" in msg or "ready" in msg.lower():
            return "success"
        if "→" in msg and "written" in msg:
            return "dbwrite"
        if "Paused" in msg or "⏸" in msg:
            return "paused"
        if "Drive" in msg or "HDD" in msg or "SMART" in msg:
            return "drive"
        if "words" in msg or "w):" in msg:
            return "caption"
        return None

    # ────────────────────────────────────────────────────────────────
    # STARTUP
    # ────────────────────────────────────────────────────────────────

    def _startup(self):
        self._log("Photo Processor initialised")
        self._log(f"  SCRIPT_DIR : {SCRIPT_DIR}")
        self._log(f"  Database   : {self._db_path}")
        m_ok = "✓" if os.path.isfile(MODEL_PT_PATH) else "✗"
        self._log(f"  model.pt   : {m_ok}")
        self._log(f"  Previews   : {DEFAULT_PREVIEW_DIR}")
        self._log(f"  Network    : {'online' if _HF_ONLINE else 'OFFLINE'}")
        self._log(f"  HDD diag   : {'loaded' if HAS_DIAG else 'not present'}")
        self._log("")

        # script indicators
        for sn in SIBLING_SCRIPTS:
            path = os.path.join(SCRIPT_DIR, sn)
            ok = os.path.isfile(path)
            self._script_lbls[sn].config(
                text=f"{'✓' if ok else '✗'} {sn}",
                foreground="#4ec9b0" if ok else "#f14c4c")

        # check for existing DB / resumable sessions
        self._check_existing_db()

        # check for local previews
        if os.path.isdir(DEFAULT_PREVIEW_DIR):
            pc = len([f for f in os.listdir(DEFAULT_PREVIEW_DIR)
                      if f.endswith(".jpg")])
            if pc:
                self._log(f"  Local previews found: {pc}")
                self._open_prev_btn.config(state=tk.NORMAL)

    def _check_existing_db(self):
        if not os.path.isfile(self._db_path):
            return
        try:
            db = ProcessorDB(self._db_path)
            sessions = self._sessions_with_work(db)
            db.close()
        except Exception:
            self._log("  ⚠️  Database exists but could not be read")
            return

        if os.path.exists(self._db_path + "-wal"):
            self._log("  ⚠️  WAL sidecar present — "
                      "previous run may not have finished")

        if sessions:
            n = len(sessions)
            self._log(f"  {n} session(s) with pending work in DB")
            self._resume_btn.config(
                text=f"↻ Resume ({n})" if n < 10
                else "↻ Resume (F7)")
        else:
            self._log("  Database present — no pending work")

    # ────────────────────────────────────────────────────────────────
    # FOLDER SELECTION
    # ────────────────────────────────────────────────────────────────

    def _browse_folder(self):
        folder = filedialog.askdirectory(
            title="Select folder with RAW files",
            mustexist=True)
        if not folder:
            return
        self._folder_var.set(folder)
        self._open_src_btn.config(state=tk.NORMAL)
        self._log(f"Selected: {folder}")

        nef = arw = 0
        for root, dirs, files in os.walk(folder):
            if "previews" in dirs:
                dirs.remove("previews")
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext == ".nef":
                    nef += 1
                elif ext == ".arw":
                    arw += 1

        if nef or arw:
            self._log(f"  {nef} NEF, {arw} ARW")
            self._camera_var.set(
                f"{nef} NEF (Nikon) · {arw} ARW (Sony)")

        # drive classification
        if DriveMonitor is not None:
            try:
                info = DriveMonitor.classify_drive(folder)
                model = info.model_name or "?"
                self._drive_var.set(
                    f"Drive: {info.drive_type} ({model})")
                fg = ("#4ec9b0" if info.drive_type != "HDD"
                      else "#cca700")
                self._drive_lbl.config(foreground=fg)
                self._log(f"  Drive: {info.parent_device} → "
                          f"{info.drive_type} ({model})")
            except Exception:
                self._drive_var.set("Drive: unknown")

    # ────────────────────────────────────────────────────────────────
    # START NEW SESSION
    # ────────────────────────────────────────────────────────────────

    def _start_new(self):
        if self._thread and self._thread.is_alive():
            return
        folder = self._folder_var.get().strip()
        if not folder:
            messagebox.showerror("Error",
                                 "Select a source folder first.")
            return
        if not os.path.isdir(folder):
            messagebox.showerror("Error",
                                 f"Folder not found:\n{folder}")
            return
        self._launch_engine(source_folder=folder)

    # ────────────────────────────────────────────────────────────────
    # RESUME SESSION
    # ────────────────────────────────────────────────────────────────

    def _do_resume(self):
        if self._thread and self._thread.is_alive():
            return

        db_path = self._db_path
        if not os.path.isfile(db_path):
            db_path = filedialog.askopenfilename(
                title="Select Database",
                filetypes=[("SQLite DB", "*.db"),
                           ("All files", "*.*")],
                initialdir=SCRIPT_DIR)
            if not db_path:
                return

        if not self._validate_db(db_path):
            messagebox.showerror(
                "Error", "Not a valid processor database.")
            return

        try:
            db = ProcessorDB(db_path)
            sessions = self._sessions_with_work(db)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open DB:\n{e}")
            return

        if not sessions:
            messagebox.showinfo(
                "Info", "No sessions with pending work.")
            db.close()
            return

        session_id = None

        if len(sessions) == 1:
            s = sessions[0]
            p = s["previews"] or 0
            msg = (f"Session: {s['name']}\n"
                   f"Files: {s['files']}\n"
                   f"Scored: {s['scored']}/{p}\n"
                   f"Captioned: {s['captioned']}/{p}\n"
                   f"Edge-mapped: {s['edge_mapped']}/{p}\n"
                   f"Missing previews: {s['missing_count']}\n\n"
                   f"Resume?")
            if messagebox.askyesno("Resume Session", msg):
                session_id = s["session_id"]
        else:
            picker = SessionPickerDialog(self.root, sessions)
            self.root.wait_window(picker)
            session_id = picker.result

        if session_id is None:
            db.close()
            return

        # preview path fixup
        fixed = self._fixup_paths(db, session_id)
        if fixed:
            self._log(f"  Fixed {fixed} preview path(s)")

        # populate source folder from session if not set
        info = db.get_session_info(session_id)
        if not self._folder_var.get().strip():
            self._folder_var.set(info.get("source_folder", ""))

        db.close()
        self._db_path = db_path
        self._launch_engine(
            source_folder=self._folder_var.get().strip() or None,
            resume_session_id=session_id)

    def _sessions_with_work(self, db: ProcessorDB) -> List[Dict]:
        """All sessions that have unfinished AI work or aren't completed."""
        try:
            rows = db.conn.execute(
                "SELECT session_id, session_name, source_folder,"
                " file_count, preview_count, status "
                "FROM sessions ORDER BY session_id DESC"
            ).fetchall()
        except Exception:
            return []

        out = []
        for r in rows:
            sid = r["session_id"]
            try:
                state = db.get_resume_state(sid)
            except Exception:
                continue
            has_work = (
                state["unscored"] > 0
                or state["uncaptioned"] > 0
                or state["unmapped"] > 0
                or r["status"] != "completed")
            if has_work:
                out.append({
                    "session_id":      sid,
                    "name":            r["session_name"],
                    "source":          r["source_folder"],
                    "files":           r["file_count"],
                    "previews":        state["total_previews"],
                    "scored":          state["scored"],
                    "captioned":       state["captioned"],
                    "edge_mapped":     state["edge_mapped"],
                    "status":          r["status"],
                    "previews_complete": state["previews_complete"],
                    "missing_count":   len(
                        state["missing_preview_files"]),
                })
        return out

    def _fixup_paths(self, db: ProcessorDB,
                     session_id: int) -> int:
        """Re-map broken preview paths to SCRIPT_DIR/previews/."""
        state = db.get_resume_state(session_id)
        missing = state["missing_preview_files"]
        if not missing:
            return 0
        fixed = 0
        with db._lock:
            for stored in missing:
                bn = os.path.basename(stored)
                candidate = os.path.join(
                    DEFAULT_PREVIEW_DIR, bn)
                if os.path.isfile(candidate):
                    try:
                        new_rel = os.path.relpath(
                            candidate, SCRIPT_DIR)
                    except ValueError:
                        new_rel = candidate
                    db.conn.execute(
                        "UPDATE previews SET preview_path=? "
                        "WHERE preview_path=?",
                        (new_rel, stored))
                    fixed += 1
            if fixed:
                db.conn.commit()
        return fixed

    @staticmethod
    def _validate_db(path: str) -> bool:
        try:
            conn = sqlite3.connect(path)
            conn.execute("SELECT COUNT(*) FROM sessions")
            conn.close()
            return True
        except Exception:
            return False

    # ────────────────────────────────────────────────────────────────
    # ENGINE LAUNCH (shared by new + resume)
    # ────────────────────────────────────────────────────────────────

    def _launch_engine(self, source_folder: str = None,
                       resume_session_id: int = None):
        self._set_ui_running()
        self._cancel.clear()
        self._progress_var.set(0)
        self._reset_phases()
        self._start_timer()

        ea = self._extract_all_var.get()
        cf = self._filter_var.get().strip()
        db_path = self._db_path

        def worker():
            try:
                self._engine = ProcessorEngine(
                    db_path=db_path,
                    model_path=MODEL_PT_PATH,
                    log=self._log,
                    cancel=self._cancel,
                    confirm=self._gui_confirm)

                summary = self._engine.run(
                    source_folder=source_folder,
                    resume_session_id=resume_session_id,
                    progress=self._on_progress,
                    extract_all=ea,
                    custom_filter=cf)

                self.root.after(
                    0, lambda s=summary: self._on_complete(s))

            except InterruptedError:
                self._log("\nCancelled by user.")
                self.root.after(
                    0, lambda: self._on_complete(
                        {"status": "cancelled"}))

            except (FileNotFoundError, ValueError,
                    RuntimeError) as e:
                self._log(f"\nERROR: {e}")
                self.root.after(
                    0, lambda err=str(e): self._on_complete(
                        {"status": "error", "error": err}))

            except Exception as e:
                self._log(f"\nERROR: {e}")
                import traceback
                self._log(traceback.format_exc())
                self.root.after(
                    0, lambda err=str(e): self._on_complete(
                        {"status": "error", "error": err}))

        self._thread = threading.Thread(
            target=worker, daemon=True)
        self._thread.start()
        self._log("\n" + "─" * 60)
        self._log("Pipeline started")

    # ────────────────────────────────────────────────────────────────
    # DEPENDENCY CONFIRM (cross-thread safe)
    # ────────────────────────────────────────────────────────────────

    def _gui_confirm(self, message: str,
                     default_yes: bool = True) -> bool:
        if threading.current_thread() is threading.main_thread():
            return messagebox.askyesno(
                "Dependency Check", message,
                default=(messagebox.YES if default_yes
                         else messagebox.NO))

        result = [default_yes]
        done = threading.Event()

        def ask():
            try:
                result[0] = messagebox.askyesno(
                    "Dependency Check", message,
                    default=(messagebox.YES if default_yes
                             else messagebox.NO))
            except Exception:
                result[0] = default_yes
            finally:
                done.set()

        self.root.after(0, ask)
        done.wait(timeout=120)
        return result[0]

    # ────────────────────────────────────────────────────────────────
    # PAUSE / CANCEL
    # ────────────────────────────────────────────────────────────────

    def _toggle_pause(self):
        if self._engine is None:
            return
        if self._engine.is_paused:
            self._engine.resume()
            self._log("  ▶ Resumed")
            self._status_var.set("Resumed")
            # un-pause phase indicators
            for key, lbl in self._phase_lbls.items():
                if lbl.cget("text").startswith("⏸"):
                    self._set_phase(key, "active")
        else:
            self._engine.pause()
            self._log("  ⏸ Paused — model stays in VRAM")
            self._status_var.set(
                "⏸ Paused — model loaded in VRAM")
            for key, lbl in self._phase_lbls.items():
                if lbl.cget("text").startswith("●"):
                    self._set_phase(key, "paused")

    def _do_cancel(self):
        if not (self._thread and self._thread.is_alive()):
            return
        self._cancel.set()
        if self._engine is not None:
            self._engine.resume()          # unblock if paused
        self._cancel_btn.config(state=tk.DISABLED)
        self._pause_btn.config(state=tk.DISABLED)
        self._status_var.set("Cancelling…")
        self._log("Cancel requested — finishing current item…")

    # ────────────────────────────────────────────────────────────────
    # PROGRESS
    # ────────────────────────────────────────────────────────────────

    def _on_progress(self, phase: str, current: int,
                     total: int, msg: str):
        pct = (current / total * 100) if total else 0

        def ui():
            self._progress_var.set(pct)
            self._status_var.set(
                f"{phase}: {current}/{total} — {msg}")
            self.root.title(
                f"[{pct:.0f}%] {phase} {current}/{total}"
                f" — {WINDOW_TITLE}")
            mapped = PHASE_MAP.get(phase)
            if mapped:
                self._set_phase(mapped, "active")
                if current >= total:
                    self._set_phase(mapped, "done")

        self.root.after(0, ui)

    # ────────────────────────────────────────────────────────────────
    # COMPLETION
    # ────────────────────────────────────────────────────────────────

    def _on_complete(self, summary: dict):
        self._stop_timer()
        self._set_ui_idle()
        self.root.title(WINDOW_TITLE)
        self._engine = None
        status = summary.get("status")

        # refresh resume button
        self._check_existing_db()

        if status == "completed":
            self._progress_var.set(100)
            self._status_var.set("✓ Complete")
            self._open_prev_btn.config(state=tk.NORMAL)
            self._open_src_btn.config(state=tk.NORMAL)

            off = " (offline)" if summary.get("offline_mode") else ""
            dt = summary.get("drive_type", "")
            drv = f"\nDrive: {dt}" if dt else ""

            messagebox.showinfo(
                "Complete",
                f"Session: {summary.get('session_name','?')}\n"
                f"Files: {summary.get('file_count',0)}\n"
                f"Bursts: {summary.get('burst_count',0)}\n"
                f"Previews: {summary.get('preview_count',0)}"
                f"{off}{drv}\n\n"
                f"DB: {summary.get('db_path','?')}")

        elif status == "cancelled":
            self._progress_var.set(0)
            self._status_var.set("Cancelled (resumable)")
            messagebox.showwarning(
                "Cancelled",
                "Progress saved.\n"
                "Click Resume to continue.")

        else:
            self._progress_var.set(0)
            self._status_var.set("✗ Error")
            messagebox.showerror(
                "Error",
                summary.get("error", "Unknown error"))

    # ────────────────────────────────────────────────────────────────
    # PHASE INDICATORS
    # ────────────────────────────────────────────────────────────────

    def _set_phase(self, key: str, state: str):
        key = key.lower()
        if key not in self._phase_lbls:
            return
        lbl = self._phase_lbls[key]
        display = key.capitalize()
        if key == "edgemap":
            display = "EdgeMap"
        icons = {
            "pending": ("○", "gray"),
            "active":  ("●", "#569cd6"),
            "done":    ("✓", "#4ec9b0"),
            "skipped": ("–", "#cca700"),
            "paused":  ("⏸", "#dcdcaa"),
        }
        icon, fg = icons.get(state, ("○", "gray"))
        lbl.config(text=f"{icon} {display}", foreground=fg)

    def _reset_phases(self):
        for k in self._phase_lbls:
            self._set_phase(k, "pending")

    # ────────────────────────────────────────────────────────────────
    # TIMER
    # ────────────────────────────────────────────────────────────────

    def _start_timer(self):
        self._start_time = time.monotonic()
        self._timer_running = True
        self._tick_timer()

    def _tick_timer(self):
        if not self._timer_running:
            return
        elapsed = time.monotonic() - self._start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        self._elapsed_var.set(f"⏱ {h:02d}:{m:02d}:{s:02d}")
        self.root.after(1000, self._tick_timer)

    def _stop_timer(self):
        self._timer_running = False

    # ────────────────────────────────────────────────────────────────
    # UI STATE HELPERS
    # ────────────────────────────────────────────────────────────────

    def _set_ui_running(self):
        self._run_btn.config(state=tk.DISABLED)
        self._resume_btn.config(state=tk.DISABLED)
        self._cancel_btn.config(state=tk.NORMAL)
        self._pause_btn.config(state=tk.NORMAL,
                               text="⏸ Pause (F6)")

    def _set_ui_idle(self):
        self._run_btn.config(state=tk.NORMAL)
        self._resume_btn.config(state=tk.NORMAL)
        self._cancel_btn.config(state=tk.DISABLED)
        self._pause_btn.config(state=tk.DISABLED,
                               text="⏸ Pause (F6)")

    # ────────────────────────────────────────────────────────────────
    # UTILITY BUTTONS
    # ────────────────────────────────────────────────────────────────

    def _open_source(self):
        f = self._folder_var.get().strip()
        if f and os.path.isdir(f):
            subprocess.Popen(["xdg-open", f])

    def _open_previews(self):
        if os.path.isdir(DEFAULT_PREVIEW_DIR):
            subprocess.Popen(["xdg-open", DEFAULT_PREVIEW_DIR])
        else:
            messagebox.showinfo(
                "Info", "No previews folder yet.")

    def _open_db_loc(self):
        d = os.path.dirname(self._db_path) or SCRIPT_DIR
        if os.path.isdir(d):
            subprocess.Popen(["xdg-open", d])

    def _export_log(self):
        path = filedialog.asksaveasfilename(
            title="Export Log",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All", "*.*")],
            initialdir=SCRIPT_DIR,
            initialfile=f"log_{datetime.now():%Y%m%d_%H%M%S}.txt")
        if not path:
            return
        self._log_w.config(state=tk.NORMAL)
        content = self._log_w.get("1.0", tk.END)
        self._log_w.config(state=tk.DISABLED)
        try:
            with open(path, "w") as f:
                f.write(content)
            self._log(f"  Log exported → {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{e}")

    # ────────────────────────────────────────────────────────────────
    # CLOSE
    # ────────────────────────────────────────────────────────────────

    def _on_close(self):
        if self._thread and self._thread.is_alive():
            if messagebox.askyesno(
                    "Confirm Exit",
                    "Processing is running.\n\n"
                    "Progress is saved and can be resumed.\n"
                    "Cancel and exit?"):
                self._cancel.set()
                if self._engine is not None:
                    self._engine.resume()
                self.root.after(800, self.root.destroy)
            return
        self.root.destroy()


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        for pref in ("clam", "alt", "default"):
            if pref in style.theme_names():
                style.theme_use(pref)
                break
    except Exception:
        pass

    ProcessorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()