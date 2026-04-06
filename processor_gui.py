#!/usr/bin/env python3
# =============================================================================
# processor_gui.py  —  v1.0.0
# Tkinter front-end for the Processor Pipeline.
# Entry point:  python processor_gui.py
#
# Orchestrates:
#   local_processor.py   (Phase 1: scan / hash / metadata / bursts / preview)
#   ai_process.py        (Phase 2: scoring / captioning / edge maps)
#   hdd_diagnostics.py   (optional drive info)
#
# Threading model
# ---------------
#   Main thread    :  Tkinter event loop — ALL widget writes happen here.
#   Worker thread  :  Both engine.run() calls; one thread, sequential.
#   Progress path  :  worker → root.after(0, _update_progress)
#   Log path       :  worker → queue.put → root.after(100) drain loop
#   Confirm path   :  worker blocks on threading.Event; main shows dialog
#
# Pause / cancel protocol
# -----------------------
#   self._cancel (threading.Event) is shared between both engines.
#   _EngineHolder holds a reference to whichever engine is currently live;
#   GUI pause/resume/cancel delegates through it.
#   Cancel MUST call holder.resume() first — BaseEngine._check() may be
#   blocking on self._pause.wait() and won't re-check cancel until woken.
#
# Tri-script DB protocol
# ----------------------
#   GUI opens ProcessorDB for pre-flight reads (session picker, resume info).
#   GUI closes ProcessorDB before the worker thread starts.
#   Each engine opens its own connection internally:
#     LocalProcessorEngine  → ProcessorDB
#     AIProcessorEngine     → AIDB  (separate class, same WAL file)
#   Both are valid SQLite3 connections in WAL mode; concurrent writes are
#   serialised by SQLite itself.  No cross-process coordination needed.
#   After worker completes, GUI may reopen ProcessorDB for result display.
# =============================================================================
from __future__ import annotations

import os
import queue
import sys
import threading
from pathlib import Path
from typing import Any, Callable, List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ════════════════════════════════════════════════════════════════════════════
# PIPELINE IMPORTS — graceful degradation if a sibling is absent
# ════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent

try:
    from local_processor import (
        LocalProcessorEngine, ProcessorDB, SessionSummary,
        DEFAULT_DB_PATH, DEFAULT_PREVIEW_ROOT, MODEL_PT_PATH,
        DEFAULT_HASH_ALGO, DEFAULT_VERIFY_RATIO,
    )
    _HAVE_LOCAL = True
    _LOCAL_ERR  = ""
except Exception as exc:
    _HAVE_LOCAL = False
    _LOCAL_ERR  = str(exc)
    # Stubs so the rest of the module survives import
    DEFAULT_DB_PATH      = SCRIPT_DIR / "ingest.db"
    DEFAULT_PREVIEW_ROOT = SCRIPT_DIR / "previews"
    MODEL_PT_PATH        = SCRIPT_DIR / "model.pt"
    DEFAULT_HASH_ALGO    = "sha256"
    DEFAULT_VERIFY_RATIO = 0.15

try:
    from ai_process import (
        AIProcessorEngine, AIConfig, _pick_device,
        DEFAULT_SCORE_BATCH, DEFAULT_CAPTION_BATCH, DEFAULT_EDGE_BATCH,
    )
    _HAVE_AI = True
    _AI_ERR  = ""
except Exception as exc:
    _HAVE_AI = False
    _AI_ERR  = str(exc)
    DEFAULT_SCORE_BATCH    = 32
    DEFAULT_CAPTION_BATCH  = 8
    DEFAULT_EDGE_BATCH     = 16

    def _pick_device(requested: str) -> str:  # type: ignore[misc]
        return "cpu"

try:
    from hdd_diagnostics import DriveMonitor
    _HAVE_DIAG = True
except Exception:
    DriveMonitor = None  # type: ignore[assignment]
    _HAVE_DIAG = False

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

# Correct filenames — local_processor.SIBLING_SCRIPTS has "ai_processor.py"
# (typo); we use the real names here.
_SIBLINGS = ["local_processor.py", "ai_process.py", "processor_gui.py"]

# Ordered phases exactly as emitted by both engines' progress callbacks.
PHASE_ORDER  = [
    "scan", "hash", "metadata", "bursts", "preview",   # local engine
    "scoring", "captioning", "edge_maps",               # AI engine
]
PHASE_LABELS = [
    "Scan", "Hash", "Meta", "Bursts", "Preview",
    "Score", "Caption", "Edge",
]
AI_PHASE_START = 5   # first AI phase index in PHASE_ORDER

C: dict[str, str] = {
    "bg":      "#1e1e1e",
    "panel":   "#252526",
    "border":  "#3e3e42",
    "fg":      "#d4d4d4",
    "fg_dim":  "#888888",
    "accent":  "#569cd6",
    "ok":      "#4ec9b0",
    "err":     "#f44747",
    "warn":    "#dcdcaa",
    "pending": "#444444",
    "entry":   "#3c3c3c",
}

_DOT_FILL = {
    "pending": C["pending"],
    "active":  C["accent"],
    "done":    C["ok"],
    "error":   C["err"],
    "skip":    C["fg_dim"],
}
_DOT_R        = 8
_DOT_SPACING  = 80    # px between dot centres


# ════════════════════════════════════════════════════════════════════════════
# SESSION PICKER DIALOG
# ════════════════════════════════════════════════════════════════════════════

class SessionPickerDialog(tk.Toplevel):
    """List resumable sessions; user picks one or starts fresh.

    result == None     →  dialog was closed without a choice
    result == -1       →  user chose "Start Fresh"
    result == <int>    →  session_id to resume
    """

    def __init__(self, parent: tk.Tk, sessions: List[dict]) -> None:
        super().__init__(parent)
        self.title("Resume Session")
        self.configure(bg=C["bg"])
        self.minsize(720, 280)
        self.resizable(True, True)
        self.grab_set()
        self.transient(parent)

        self.result: Optional[int] = None
        self._build(sessions)

        self.update_idletasks()
        x = parent.winfo_rootx() + 40
        y = parent.winfo_rooty() + 40
        self.geometry(f"+{x}+{y}")
        self.wait_visibility()

    def _build(self, sessions: List[dict]) -> None:
        tk.Label(
            self,
            text=f"  {len(sessions)} session(s) with pending work:",
            bg=C["bg"], fg=C["fg_dim"],
            font=("Segoe UI", 9),
            anchor="w", padx=4, pady=6,
        ).pack(fill="x")

        tree_frame = tk.Frame(self, bg=C["bg"])
        tree_frame.pack(fill="both", expand=True, padx=8, pady=(0, 4))

        cols   = ("id", "name", "source", "files", "prev", "pending", "ai", "status")
        hdrs   = ("#",  "Name", "Source", "Files", "Prev", "Pending AI", "AI Status", "Status")
        widths = ( 36,   150,    240,      52,      52,     84,           80,           80)

        self._tree = ttk.Treeview(
            tree_frame, columns=cols, show="headings",
            selectmode="browse",
        )
        for col, hdr, w in zip(cols, hdrs, widths):
            self._tree.heading(col, text=hdr, anchor="w")
            self._tree.column(col, width=w, minwidth=w,
                              stretch=(col == "source"))

        sb = ttk.Scrollbar(tree_frame, orient="vertical",
                           command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        for s in sessions:
            src = s.get("source", "")
            if len(src) > 38:
                src = "…" + src[-35:]
            p   = s.get("previews", 0)
            sc  = s.get("scored",   0)
            cap = s.get("captioned", 0)
            edg = s.get("edge_mapped", 0)
            pending = (p - sc) + (p - cap) + (p - edg)
            self._tree.insert(
                "", "end",
                values=(
                    s.get("session_id", ""),
                    s.get("name", ""),
                    src,
                    s.get("files", 0),
                    p,
                    pending,
                    s.get("ai_status", "—") if "ai_status" in s else "—",
                    s.get("status", ""),
                ),
            )

        self._tree.bind("<Double-1>", lambda _: self._on_resume())

        btn_f = tk.Frame(self, bg=C["bg"])
        btn_f.pack(fill="x", padx=8, pady=8)

        self._btn(btn_f, "↩  Resume Selected", self._on_resume,
                  bg=C["accent"], fg="white").pack(side="left", padx=(0, 6))
        self._btn(btn_f, "+  Start Fresh", self._on_fresh,
                  bg=C["panel"], fg=C["fg"]).pack(side="left")
        self._btn(btn_f, "Close", self.destroy,
                  bg=C["panel"], fg=C["fg_dim"]).pack(side="right")

    def _btn(self, parent, text, cmd, bg, fg):
        return tk.Button(
            parent, text=text, command=cmd,
            bg=bg, fg=fg, relief="flat",
            padx=10, pady=4,
            activebackground=C["border"], activeforeground=C["fg"],
        )

    def _selected_sid(self) -> Optional[int]:
        sel = self._tree.selection()
        if not sel:
            return None
        return int(self._tree.item(sel[0])["values"][0])

    def _on_resume(self) -> None:
        sid = self._selected_sid()
        if sid is None:
            messagebox.showwarning("No selection",
                                   "Select a session to resume.", parent=self)
            return
        self.result = sid
        self.destroy()

    def _on_fresh(self) -> None:
        self.result = -1
        self.destroy()


# ════════════════════════════════════════════════════════════════════════════
# ENGINE HOLDER — thread-safe pause / resume / cancel delegation
# ════════════════════════════════════════════════════════════════════════════

class _EngineHolder:
    def __init__(self) -> None:
        self._engine: Optional[Any] = None
        self._lock   = threading.Lock()

    def set(self, engine: Any) -> None:
        with self._lock:
            self._engine = engine

    def clear(self) -> None:
        with self._lock:
            self._engine = None

    def pause(self) -> None:
        with self._lock:
            if self._engine is not None:
                self._engine.pause()

    def resume(self) -> None:
        with self._lock:
            if self._engine is not None:
                self._engine.resume()

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return (self._engine.is_paused
                    if self._engine is not None else False)


# ════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ════════════════════════════════════════════════════════════════════════════

class ProcessorGUI:

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Processor Pipeline")
        root.configure(bg=C["bg"])
        root.minsize(800, 580)

        # State
        self._source_var    = tk.StringVar()
        self._status_var    = tk.StringVar(value="Ready.")
        self._device_var    = tk.StringVar(value="auto")
        self._skip_score    = tk.BooleanVar(value=False)
        self._skip_caption  = tk.BooleanVar(value=False)
        self._skip_edge     = tk.BooleanVar(value=False)

        self._log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._cancel        = threading.Event()
        self._holder        = _EngineHolder()
        self._running       = False
        self._resume_sid: Optional[int] = None

        # Phase dot canvas items  {phase_str: canvas_oval_id}
        self._dots: dict[str, int]   = {}
        self._phase_idx               = {p: i for i, p in enumerate(PHASE_ORDER)}
        self._dot_canvas: Optional[tk.Canvas] = None

        self._apply_style()
        self._build_ui()
        self._drain_log()
        root.after(150, self._startup_check)

    # ── style ──────────────────────────────────────────────────────────────

    def _apply_style(self) -> None:
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except Exception:
            pass
        s.configure(".", background=C["bg"], foreground=C["fg"],
                     fieldbackground=C["entry"], borderwidth=0)
        s.configure("TFrame",       background=C["bg"])
        s.configure("TLabel",       background=C["bg"], foreground=C["fg"])
        s.configure("TCheckbutton", background=C["bg"], foreground=C["fg"])
        s.configure("TCombobox",    fieldbackground=C["entry"],
                    background=C["entry"], foreground=C["fg"],
                    selectbackground=C["accent"])
        s.map("TCombobox",
              fieldbackground=[("disabled", C["panel"])],
              foreground=[("disabled", C["fg_dim"])])
        s.configure("TProgressbar", troughcolor=C["panel"],
                    background=C["accent"], borderwidth=0, relief="flat")
        s.configure("Treeview",     background=C["panel"],
                    foreground=C["fg"], fieldbackground=C["panel"],
                    rowheight=22)
        s.configure("Treeview.Heading", background=C["border"],
                    foreground=C["fg"], relief="flat")
        s.map("Treeview",
              background=[("selected", C["accent"])],
              foreground=[("selected", "#ffffff")])

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self._build_header()
        self._build_source_row()
        self._build_controls_row()
        self._build_phase_strip()

        self._pbar = ttk.Progressbar(
            self.root, orient="horizontal", mode="determinate", maximum=100,
        )
        self._pbar.pack(fill="x", padx=8, pady=(2, 0))
        self._pbar_lbl = ttk.Label(
            self.root, text="", foreground=C["fg_dim"], font=("Segoe UI", 8),
        )
        self._pbar_lbl.pack(anchor="w", padx=8)

        self._build_log()

        tk.Frame(self.root, height=1, bg=C["border"]).pack(fill="x")
        ttk.Label(
            self.root, textvariable=self._status_var,
            foreground=C["fg_dim"], font=("Segoe UI", 8),
        ).pack(anchor="w", padx=8, pady=2)

    def _build_header(self) -> None:
        h = tk.Frame(self.root, bg=C["panel"])
        h.pack(fill="x")

        tk.Label(
            h, text="Processor Pipeline",
            bg=C["panel"], fg=C["fg"],
            font=("Segoe UI", 11, "bold"),
            padx=12, pady=6,
        ).pack(side="left")

        for name in _SIBLINGS:
            exists = (SCRIPT_DIR / name).exists()
            tk.Label(
                h, text=f"{'✓' if exists else '✗'} {name}",
                bg=C["panel"],
                fg=C["ok"] if exists else C["err"],
                font=("Consolas", 8),
                padx=6,
            ).pack(side="left")

        if _HAVE_DIAG:
            tk.Label(
                h, text="⬤ diag",
                bg=C["panel"], fg=C["fg_dim"],
                font=("Consolas", 8), padx=6,
            ).pack(side="right")

    def _build_source_row(self) -> None:
        f = tk.Frame(self.root, bg=C["bg"])
        f.pack(fill="x", padx=8, pady=(6, 2))

        ttk.Label(f, text="Source:").pack(side="left")
        self._source_entry = tk.Entry(
            f, textvariable=self._source_var,
            bg=C["entry"], fg=C["fg"],
            insertbackground=C["fg"],
            relief="flat", bd=4, width=55,
        )
        self._source_entry.pack(side="left", padx=(4, 2), fill="x", expand=True)

        self._browse_btn = tk.Button(
            f, text="Browse…",
            command=self._pick_source,
            bg=C["panel"], fg=C["fg"],
            relief="flat", padx=8, pady=2,
            activebackground=C["border"], activeforeground=C["fg"],
        )
        self._browse_btn.pack(side="left")

        self._drive_lbl = ttk.Label(f, text="", foreground=C["fg_dim"],
                                    font=("Segoe UI", 8))
        self._drive_lbl.pack(side="left", padx=(10, 0))

    def _build_controls_row(self) -> None:
        f = tk.Frame(self.root, bg=C["bg"])
        f.pack(fill="x", padx=8, pady=(0, 4))

        def _btn(text, cmd, bg, fg, **kw):
            return tk.Button(
                f, text=text, command=cmd,
                bg=bg, fg=fg, relief="flat",
                padx=10, pady=4,
                activebackground=C["border"], activeforeground=C["fg"],
                **kw,
            )

        self._start_btn = _btn(
            "▶  Start", self._start_run,
            bg=C["ok"], fg="#000000",
            font=("Segoe UI", 9, "bold"),
        )
        self._start_btn.pack(side="left", padx=(0, 4))

        self._pause_btn = _btn("⏸  Pause", self._pause_resume,
                                bg=C["panel"], fg=C["fg"])
        self._pause_btn.pack(side="left", padx=(0, 4))
        self._pause_btn.configure(state="disabled")

        self._cancel_btn = _btn("✕  Cancel", self._cancel_run,
                                 bg=C["panel"], fg=C["err"])
        self._cancel_btn.pack(side="left", padx=(0, 16))
        self._cancel_btn.configure(state="disabled")

        ttk.Label(f, text="Device:").pack(side="left")
        self._device_cb = ttk.Combobox(
            f, textvariable=self._device_var,
            values=["auto", "cuda", "cpu"],
            width=6, state="readonly",
        )
        self._device_cb.pack(side="left", padx=(2, 14))

        ttk.Label(f, text="Skip AI:").pack(side="left", padx=(0, 2))
        ttk.Checkbutton(f, text="Score",
                         variable=self._skip_score).pack(side="left")
        ttk.Checkbutton(f, text="Caption",
                         variable=self._skip_caption).pack(side="left")
        ttk.Checkbutton(f, text="Edge",
                         variable=self._skip_edge).pack(side="left")

        if not _HAVE_AI:
            ttk.Label(f, text="  ⚠ ai_process unavailable",
                      foreground=C["warn"]).pack(side="left", padx=8)

    def _build_phase_strip(self) -> None:
        f = tk.Frame(self.root, bg=C["bg"])
        f.pack(fill="x", padx=8, pady=(0, 2))
        ttk.Label(f, text="Phases:", foreground=C["fg_dim"],
                  font=("Segoe UI", 8)).pack(anchor="w")

        n = len(PHASE_ORDER)
        canvas_w = _DOT_SPACING * n + 20
        canvas_h = 48
        c = tk.Canvas(f, width=canvas_w, height=canvas_h,
                       bg=C["bg"], highlightthickness=0)
        c.pack(anchor="w")
        self._dot_canvas = c

        for i, (phase, label) in enumerate(zip(PHASE_ORDER, PHASE_LABELS)):
            x = 16 + i * _DOT_SPACING
            if i == AI_PHASE_START:
                c.create_line(x - 10, 2, x - 10, canvas_h - 4,
                              fill=C["border"], width=1, dash=(3, 4))
                c.create_text(x - 10, canvas_h - 4,
                              text="AI", anchor="s",
                              fill=C["fg_dim"], font=("Segoe UI", 7))
            oid = c.create_oval(
                x - _DOT_R, 12 - _DOT_R,
                x + _DOT_R, 12 + _DOT_R,
                fill=_DOT_FILL["pending"], outline="",
            )
            self._dots[phase] = oid
            c.create_text(x, 30, text=label,
                          fill=C["fg_dim"], font=("Segoe UI", 7))

    def _build_log(self) -> None:
        f = tk.Frame(self.root, bg=C["bg"])
        f.pack(fill="both", expand=True, padx=8, pady=(2, 4))

        ttk.Label(f, text="Log", foreground=C["fg_dim"],
                  font=("Segoe UI", 8)).pack(anchor="w")

        self._log_text = tk.Text(
            f,
            bg=C["panel"], fg=C["fg"],
            font=("Consolas", 9),
            relief="flat", state="disabled",
            wrap="none",
        )
        sb_y = ttk.Scrollbar(f, orient="vertical",
                             command=self._log_text.yview)
        sb_x = ttk.Scrollbar(f, orient="horizontal",
                             command=self._log_text.xview)
        self._log_text.configure(
            yscrollcommand=sb_y.set,
            xscrollcommand=sb_x.set,
        )
        sb_y.pack(side="right",  fill="y")
        sb_x.pack(side="bottom", fill="x")
        self._log_text.pack(fill="both", expand=True)

        self._log_text.tag_configure("err",  foreground=C["err"])
        self._log_text.tag_configure("warn", foreground=C["warn"])
        self._log_text.tag_configure("ok",   foreground=C["ok"])
        self._log_text.tag_configure("dim",  foreground=C["fg_dim"])

    # ── startup ────────────────────────────────────────────────────────────

    def _startup_check(self) -> None:
        if not _HAVE_LOCAL:
            messagebox.showerror(
                "Import Error",
                f"Cannot import local_processor.py:\n\n{_LOCAL_ERR}\n\n"
                "All scripts must be in the same directory.",
            )
            self.root.quit()
            return

        if not _HAVE_AI:
            self._log_append(
                f"[WARN] ai_process.py not importable — AI phases disabled.\n"
                f"       {_AI_ERR}\n",
                tag="warn",
            )

        try:
            db       = ProcessorDB(DEFAULT_DB_PATH)
            sessions = db.sessions_with_pending_work()
            db.close()
        except Exception:
            sessions = []

        if sessions:
            self.root.after(50, lambda: self._offer_resume(sessions))

    def _offer_resume(self, sessions: List[dict]) -> None:
        dlg = SessionPickerDialog(self.root, sessions)
        self.root.wait_window(dlg)

        if dlg.result is None:
            return   # closed without choice

        if dlg.result == -1:
            self._resume_sid = None   # start fresh
            return

        sid = dlg.result
        try:
            db   = ProcessorDB(DEFAULT_DB_PATH)
            info = db.get_session_resume_info(sid)
            db.close()
        except Exception as exc:
            messagebox.showerror("Resume error", str(exc))
            return

        self._resume_sid = sid
        src = info.get("source_folder", "")
        self._source_var.set(src)
        self._update_drive_label(src)

        n_miss = info.get("missing_count", 0)
        miss_note = f", {n_miss} previews missing on disk" if n_miss else ""
        self._status_var.set(
            f"Session {sid} · {info.get('session_name', '')} · "
            f"{info.get('unscored', 0)} unscored, "
            f"{info.get('uncaptioned', 0)} uncaptioned, "
            f"{info.get('unmapped', 0)} unmapped"
            f"{miss_note}"
        )

        # Attempt path fixup for missing previews
        if n_miss:
            try:
                db    = ProcessorDB(DEFAULT_DB_PATH)
                fixed = db.fixup_missing_preview_paths(sid)
                db.close()
                if fixed:
                    self._log_append(
                        f"[INFO] Re-linked {fixed} missing preview paths.\n",
                        tag="ok",
                    )
            except Exception:
                pass

    # ── source picker ──────────────────────────────────────────────────────

    def _pick_source(self) -> None:
        folder = filedialog.askdirectory(
            title="Select RAW source folder",
            initialdir=self._source_var.get() or str(Path.home()),
            mustexist=True,
        )
        if not folder:
            return
        self._source_var.set(folder)
        self._resume_sid = None
        self._update_drive_label(folder)

    def _update_drive_label(self, path: str) -> None:
        if not _HAVE_DIAG or DriveMonitor is None or not path:
            self._drive_lbl.configure(text="")
            return
        try:
            info = DriveMonitor.classify_drive(path)
            self._drive_lbl.configure(
                text=f"  {info.drive_type} · {info.model_name}"
            )
        except Exception:
            self._drive_lbl.configure(text="")

    # ── start / pause / cancel ─────────────────────────────────────────────

    def _start_run(self) -> None:
        if self._running:
            return

        source = self._source_var.get().strip()

        # If resuming and source entry is populated (set by session picker),
        # use what's there.  If entry is empty with a resume_sid, we'll
        # pull source_folder from the DB inside the worker.
        if not source and self._resume_sid is None:
            messagebox.showwarning("No source", "Select a source folder first.")
            return
        if source and not Path(source).is_dir():
            messagebox.showerror("Not found", f"Folder not found:\n{source}")
            return

        device = _pick_device(self._device_var.get())

        self._cancel.clear()
        self._running = True
        self._reset_dots()
        self._log_clear()
        self._set_controls(running=True)

        threading.Thread(
            target=self._worker,
            args=(source, self._resume_sid, device),
            daemon=True,
            name="pipeline-worker",
        ).start()

    def _pause_resume(self) -> None:
        if self._holder.is_paused:
            self._holder.resume()
            self._pause_btn.configure(text="⏸  Pause")
            self._status_var.set("Running…")
        else:
            self._holder.pause()
            self._pause_btn.configure(text="▶  Resume")
            self._status_var.set("Paused.")

    def _cancel_run(self) -> None:
        if not self._running:
            return
        self._cancel.set()
        self._holder.resume()   # unblock a paused engine before it checks cancel
        self._status_var.set("Cancelling…")

    # ── worker thread ──────────────────────────────────────────────────────

    def _worker(
        self,
        source_folder: str,
        resume_sid: Optional[int],
        device: str,
    ) -> None:
        session_id: Optional[int] = resume_sid

        # If resuming but source entry was blank, pull from DB
        if not source_folder and resume_sid is not None:
            try:
                _db   = ProcessorDB(DEFAULT_DB_PATH)
                _sess = _db.get_session(resume_sid)
                _db.close()
                source_folder = (_sess or {}).get("source_folder", "") if _sess else ""
            except Exception as exc:
                self._on_log(f"Could not load session source: {exc}", tag="err")
                self._finish(error=str(exc))
                return

        # ── Phase 1: Local ────────────────────────────────────────────────
        if _HAVE_LOCAL:
            engine = LocalProcessorEngine(
                db_path            = DEFAULT_DB_PATH,
                preview_root       = DEFAULT_PREVIEW_ROOT,
                checksum_algo      = DEFAULT_HASH_ALGO,
                verification_ratio = DEFAULT_VERIFY_RATIO,
                log                = self._on_log,
                progress           = self._on_progress,
                cancel             = self._cancel,
                confirm            = self._confirm_callback,
            )
            self._holder.set(engine)
            try:
                result: SessionSummary = engine.run(
                    source_folder = source_folder,
                    session_id    = resume_sid,
                )
                session_id = result.session_id
                self._on_log(
                    f"✓ Local complete — session {session_id} · "
                    f"{result.file_count} files · "
                    f"{result.preview_count} previews · "
                    f"{result.elapsed_s}s",
                    tag="ok",
                )
            except InterruptedError:
                self._on_log("✋ Local phase cancelled — session is resumable.")
                self._finish(cancelled=True)
                return
            except Exception as exc:
                self._on_log(f"✗ Local phase error: {exc}", tag="err")
                self._finish(error=str(exc))
                return
            finally:
                self._holder.clear()

        if self._cancel.is_set():
            self._finish(cancelled=True)
            return

        # ── Phase 2: AI ───────────────────────────────────────────────────
        if _HAVE_AI and session_id is not None:
            cfg = AIConfig(
                skip_scoring       = self._skip_score.get(),
                skip_captioning    = self._skip_caption.get(),
                skip_edge_maps     = self._skip_edge.get(),
                score_batch_size   = DEFAULT_SCORE_BATCH,
                caption_batch_size = DEFAULT_CAPTION_BATCH,
                edge_batch_size    = DEFAULT_EDGE_BATCH,
                device             = device,
                model_path         = MODEL_PT_PATH,
                db_path            = DEFAULT_DB_PATH,
            )
            ai_engine = AIProcessorEngine(
                config   = cfg,
                log      = self._on_log,
                progress = self._on_progress,
                cancel   = self._cancel,
            )
            self._holder.set(ai_engine)
            try:
                ai_r = ai_engine.run(session_id)
                if ai_r.get("status") == "completed":
                    self._on_log(
                        f"✓ AI complete — scored={ai_r.get('scored', 0)} · "
                        f"captioned={ai_r.get('captioned', 0)} · "
                        f"edge={ai_r.get('edge_mapped', 0)} · "
                        f"device={ai_r.get('device', '?')} · "
                        f"{ai_r.get('elapsed_s', 0)}s",
                        tag="ok",
                    )
                elif ai_r.get("status") == "interrupted":
                    self._finish(cancelled=True)
                    return
            except InterruptedError:
                self._on_log("✋ AI phase cancelled — session is resumable.")
                self._finish(cancelled=True)
                return
            except Exception as exc:
                self._on_log(f"✗ AI phase error: {exc}", tag="err")
                self._finish(error=str(exc))
                return
            finally:
                self._holder.clear()

        self._finish()

    def _finish(
        self,
        cancelled: bool = False,
        error: Optional[str] = None,
    ) -> None:
        self.root.after(0, self._on_finish, cancelled, error)

    def _on_finish(self, cancelled: bool, error: Optional[str]) -> None:
        self._running = False
        self._set_controls(running=False)

        if cancelled:
            self._status_var.set("Cancelled — session is resumable.")
        elif error:
            self._status_var.set(f"Error: {error}")
            for phase in PHASE_ORDER:
                if self._dot_state(phase) == "active":
                    self._set_dot(phase, "error")
        else:
            self._status_var.set("Pipeline complete ✓")
            for phase in PHASE_ORDER:
                self._set_dot(phase, "done")
            self.root.title("Processor Pipeline — Done ✓")

    # ── progress / log ─────────────────────────────────────────────────────

    def _on_progress(
        self, phase: str, done: int, total: int, filename: str,
    ) -> None:
        """Called from worker thread."""
        self.root.after(0, self._update_progress, phase, done, total, filename)

    def _update_progress(
        self, phase: str, done: int, total: int, filename: str,
    ) -> None:
        """Runs in main thread only."""
        if phase in self._phase_idx:
            cur_idx = self._phase_idx[phase]
            # Set all earlier phases to done
            for p, i in self._phase_idx.items():
                if i < cur_idx and self._dot_state(p) != "done":
                    self._set_dot(p, "done")
            self._set_dot(phase, "done" if done >= total else "active")

        pct = int(done / max(total, 1) * 100)
        self._pbar["value"] = pct

        short = Path(filename).name if filename else ""
        self._pbar_lbl.configure(
            text=f"{phase}  {done}/{total}  {short}"
        )
        self.root.title(f"Processor Pipeline — {phase} {pct}%")

    def _on_log(self, msg: str, tag: str = "") -> None:
        """Called from any thread."""
        self._log_queue.put((msg, tag))

    def _drain_log(self) -> None:
        try:
            while True:
                msg, tag = self._log_queue.get_nowait()
                self._log_append(msg, tag=tag)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log)

    def _log_append(self, msg: str, tag: str = "") -> None:
        if not tag:
            low = msg.lower()
            if "error" in low or "✗" in low:
                tag = "err"
            elif "warn" in low or "⚠" in low:
                tag = "warn"
            elif "✓" in low or "complete" in low:
                tag = "ok"
            elif msg.startswith("  ") or msg.startswith("["):
                tag = "dim"

        self._log_text.configure(state="normal")
        self._log_text.insert("end", msg if msg.endswith("\n") else msg + "\n",
                               tag if tag else ())
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _log_clear(self) -> None:
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")

    # ── confirm callback (cross-thread dialog) ─────────────────────────────

    def _confirm_callback(self, message: str, default_yes: bool) -> bool:
        result  = [default_yes]
        event   = threading.Event()

        def _ask() -> None:
            result[0] = messagebox.askyesno(
                "Confirm", message, parent=self.root
            )
            event.set()

        self.root.after(0, _ask)
        event.wait(timeout=120)   # engine falls back to default_yes on timeout
        return result[0]

    # ── phase dot helpers ──────────────────────────────────────────────────

    def _set_dot(self, phase: str, state: str) -> None:
        if self._dot_canvas is None or phase not in self._dots:
            return
        self._dot_canvas.itemconfigure(
            self._dots[phase], fill=_DOT_FILL.get(state, _DOT_FILL["pending"])
        )
        # Store state as a tag on the item for _dot_state() to read
        self._dot_canvas.dtag(self._dots[phase], "state_*")
        self._dot_canvas.addtag_withtag(f"state_{state}", self._dots[phase])

    def _dot_state(self, phase: str) -> str:
        if self._dot_canvas is None or phase not in self._dots:
            return "pending"
        tags = self._dot_canvas.gettags(self._dots[phase])
        for t in tags:
            if t.startswith("state_"):
                return t[6:]
        return "pending"

    def _reset_dots(self) -> None:
        for phase in PHASE_ORDER:
            self._set_dot(phase, "pending")
        self._pbar["value"] = 0
        self._pbar_lbl.configure(text="")
        self.root.title("Processor Pipeline")

    # ── controls state ─────────────────────────────────────────────────────

    def _set_controls(self, running: bool) -> None:
        if running:
            self._start_btn.configure(state="disabled")
            self._pause_btn.configure(state="normal",  text="⏸  Pause")
            self._cancel_btn.configure(state="normal")
            self._browse_btn.configure(state="disabled")
            self._device_cb.configure(state="disabled")
        else:
            self._start_btn.configure(state="normal")
            self._pause_btn.configure(state="disabled", text="⏸  Pause")
            self._cancel_btn.configure(state="disabled")
            self._browse_btn.configure(state="normal")
            self._device_cb.configure(state="readonly")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    root = tk.Tk()
    ProcessorGUI(root)
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()


if __name__ == "__main__":
    main()
