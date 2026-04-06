#!/usr/bin/env python3
# =============================================================================
# ai_process.py  —  v2.3.0
# AI inference layer for the Processor Pipeline.
#
# Florence-2 on Windows without flash_attn — what actually happens
# ----------------------------------------------------------------
# transformers.dynamic_module_utils.check_imports() does a TEXT SCAN of
# the remote-code modeling file before any Python executes.  It finds
# "from flash_attn import" on three lines (63, 685, 686) and raises:
#
#   ImportError: This modeling file requires the following packages that
#   were not found in your environment: flash_attn.
#
# The is_flash_attn_2_available() guards around those imports are irrelevant
# to this check — check_imports() uses regex/grep, not execution.
#
# Fix: _patch_florence_cached_file() comments out those three lines in
# the cached snapshot.  Since is_flash_attn_2_available() returns False on
# any machine without flash_attn, this has zero runtime effect.  The patch
# is idempotent (marked with a sentinel comment so it never double-patches).
#
# Additionally:
#   - attn_implementation="eager" passed to from_pretrained() so the config
#     can never select flash_attention_2 even if the checkpoint config has it
#   - forced_bos/eos_token_id injected on language sub-config (v2.1 fix)
#   - language_model.generate() bound from MRO if absent (v2.2 fix)
#   - _manual_generate() fallback via model.forward() + KV cache (v2.2 fix)
# =============================================================================
from __future__ import annotations

import argparse
import contextlib
import gc
import io
import os
import re
import shutil
import socket
import sqlite3
import sys
import threading
import time
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR         = Path(__file__).resolve().parent
DEFAULT_DB_PATH    = SCRIPT_DIR / "ingest.db"
DEFAULT_MODEL_PATH = SCRIPT_DIR / "model.pt"
FLORENCE_MODEL_ID  = "MiaoshouAI/Florence-2-large-PromptGen-v2.0"
CLIP_MODEL_ID      = "openai/clip-vit-base-patch32"
HED_MODEL_ID       = "lllyasviel/Annotators"
CAPTION_MODE       = "<MORE_DETAILED_CAPTION>"

SCORE_CATEGORIES       = ["overall", "quality", "composition",
                           "lighting", "color", "dof", "content"]
DEFAULT_SCORE_BATCH    = 32
DEFAULT_CAPTION_BATCH  = 8
DEFAULT_EDGE_BATCH     = 16
EDGE_MAP_QUALITY       = 80
EDGE_MAP_ENCODING      = "hex"
MIN_PREVIEW_BYTES      = 2048
HF_CONNECT_TIMEOUT     = 3
CAPTION_MAX_DIM        = 1920
CAPTION_MAX_NEW_TOKENS = 512
CAPTION_NUM_BEAMS      = 3

_GEN_COMPAT_ATTRS = ("forced_bos_token_id", "forced_eos_token_id")

# Sentinel written into patched files so we never double-patch
_PATCH_SENTINEL = "# PATCHED_BY_AI_PROCESS_NO_FLASH_ATTN"

# ════════════════════════════════════════════════════════════════════════════
# HF OFFLINE PROBE
# ════════════════════════════════════════════════════════════════════════════

_HF_ONLINE_CACHE: Optional[bool] = None
_HF_PROBE_LOCK   = threading.Lock()


def _hf_online() -> bool:
    global _HF_ONLINE_CACHE
    with _HF_PROBE_LOCK:
        if _HF_ONLINE_CACHE is not None:
            return _HF_ONLINE_CACHE
        try:
            socket.create_connection(("huggingface.co", 443),
                                     timeout=HF_CONNECT_TIMEOUT)
            _HF_ONLINE_CACHE = True
        except OSError:
            _HF_ONLINE_CACHE = False
        return _HF_ONLINE_CACHE


def _hf_cached(model_id: str) -> bool:
    cache = Path(os.environ.get("HF_HOME",
                                Path.home() / ".cache" / "huggingface"))
    slug = model_id.replace("/", "--")
    snap = cache / "hub" / f"models--{slug}" / "snapshots"
    return snap.is_dir() and any(snap.iterdir())


def _hf_snapshot_dir(model_id: str) -> Optional[Path]:
    """Return the most-recent snapshot directory for a cached model, or None."""
    cache = Path(os.environ.get("HF_HOME",
                                Path.home() / ".cache" / "huggingface"))
    slug  = model_id.replace("/", "--")
    snaps = cache / "hub" / f"models--{slug}" / "snapshots"
    if not snaps.is_dir():
        return None
    # Pick the most recently modified snapshot (mtime)
    dirs = [d for d in snaps.iterdir() if d.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)


# ════════════════════════════════════════════════════════════════════════════
# FLORENCE CACHED-FILE PATCH
# Removes the flash_attn import lines that transformers' check_imports()
# text-scanner trips on.  Safe to call multiple times (idempotent).
# ════════════════════════════════════════════════════════════════════════════

# Exact patterns check_imports() triggers on (leading whitespace varies,
# so we match with a simple 'from flash_attn' substring check)
_FLASH_IMPORT_PATTERNS = (
    "from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input",
    "from flash_attn import flash_attn_func, flash_attn_varlen_func",
)


def _patch_florence_cached_file(log: Callable[[str], None]) -> bool:
    """
    Comment out flash_attn import lines in the cached modeling_florence2.py.

    Returns True if the file was patched (or was already patched).
    Returns False if the file could not be found or written.

    Why this is safe:
      Every occurrence of "from flash_attn import ..." in the file is already
      inside an "if is_flash_attn_2_available():" block.  is_flash_attn_2_available()
      returns False on any machine without flash_attn installed, so these branches
      never execute.  Commenting them out is a no-op at runtime.

    Why this is necessary:
      transformers.dynamic_module_utils.check_imports() text-scans the file
      for import statements and raises ImportError if those packages are absent,
      before any Python in the file is evaluated.  The is_flash_attn_2_available()
      guard is invisible to this scan.
    """
    snap = _hf_snapshot_dir(FLORENCE_MODEL_ID)
    if snap is None:
        log("  [patch] Florence-2 snapshot not found in cache — cannot patch")
        return False

    target = snap / "modeling_florence2.py"
    if not target.exists():
        log(f"  [patch] modeling_florence2.py not found at {target}")
        return False

    content = target.read_text(encoding="utf-8")

    if _PATCH_SENTINEL in content:
        log(f"  [patch] modeling_florence2.py already patched — skipping")
        return True

    # Backup original before any modification
    backup = target.with_suffix(".py.orig")
    if not backup.exists():
        shutil.copy2(target, backup)
        log(f"  [patch] Backup written: {backup.name}")

    lines    = content.splitlines(keepends=True)
    patched  = 0
    out      = []

    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(pat) or stripped == pat.strip()
               for pat in _FLASH_IMPORT_PATTERNS):
            # Comment out the line, preserving indent
            indent = len(line) - len(line.lstrip())
            out.append(" " * indent + "# " + line.lstrip())
            patched += 1
        else:
            out.append(line)

    if patched == 0:
        log("  [patch] No flash_attn import lines found — file may differ from expected")
        log(f"         Snapshot: {snap}")
        log("         Captioning may still fail; check modeling_florence2.py manually")
        # Still write sentinel so we don't re-attempt on every run
        out.append(f"\n{_PATCH_SENTINEL}\n")
    else:
        out.append(f"\n{_PATCH_SENTINEL}\n")
        log(f"  [patch] Commented out {patched} flash_attn import line(s) in "
            f"modeling_florence2.py")

    target.write_text("".join(out), encoding="utf-8")
    return True


# ════════════════════════════════════════════════════════════════════════════
# RUNTIME COMPAT PATCHES (applied after model load)
# ════════════════════════════════════════════════════════════════════════════

def _configs_to_patch(model: Any) -> List[Any]:
    cfgs: List[Any] = [model.config]
    lang = getattr(model, "language_model", None)
    if lang is not None and hasattr(lang, "config"):
        cfgs.append(lang.config)
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        cfgs.append(gen_cfg)
    return cfgs


def _patch_generation_compat(model: Any) -> None:
    """Inject forced_bos/eos_token_id=None on sub-configs that lack them."""
    for cfg in _configs_to_patch(model):
        for attr in _GEN_COMPAT_ATTRS:
            if not hasattr(cfg, attr):
                setattr(cfg, attr, None)


def _patch_florence_lang_generate(model: Any) -> None:
    """
    Bind generate() onto language_model if absent.

    Florence2ForConditionalGeneration.generate() delegates to
    self.language_model.generate() in transformers >= 4.43.
    Florence2LanguageForConditionalGeneration does not expose generate()
    in newer checkpoints.  Walk MRO to find it in GenerationMixin and
    bind it as an instance method.
    """
    lang = getattr(model, "language_model", None)
    if lang is None:
        return
    if hasattr(lang, "generate") and callable(getattr(lang, "generate")):
        return

    found_fn = None
    for cls in type(lang).__mro__[1:]:
        fn = cls.__dict__.get("generate")
        if fn is not None:
            found_fn = fn
            break

    if found_fn is None:
        try:
            from transformers.generation import GenerationMixin
            found_fn = GenerationMixin.generate
        except ImportError:
            return

    lang.generate = types.MethodType(found_fn, lang)
    if not getattr(lang, "can_generate", lambda: False)():
        lang.can_generate = lambda: True


def _stub_missing_transformers_utils() -> None:
    """
    Stub helpers that older transformers versions may not export but that
    newer Florence-2 remote code imports from transformers.utils.
    """
    try:
        import transformers.utils as tu
        if not hasattr(tu, "is_flash_attn_greater_or_equal_2_10"):
            tu.is_flash_attn_greater_or_equal_2_10 = lambda *a, **kw: False
        if not hasattr(tu, "is_flash_attn_2_available"):
            tu.is_flash_attn_2_available = lambda *a, **kw: False
    except ImportError:
        pass


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _resolve_stored(stored: str) -> Path:
    p = Path(stored)
    return p if p.is_absolute() else SCRIPT_DIR / p


def _bucket_size(w: int, h: int) -> Tuple[int, int]:
    ratio = w / max(h, 1)
    candidates = [(768, 512), (512, 768), (512, 832), (576, 1024), (512, 512)]
    return min(candidates, key=lambda s: abs(s[0] / s[1] - ratio))


def _encode_image_bytes(buf: bytes) -> str:
    if EDGE_MAP_ENCODING == "base64":
        import base64
        return base64.b64encode(buf).decode("ascii")
    return buf.hex()


def _restore_env(key: str, original: Optional[str]) -> None:
    if original is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = original


# ════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PreviewRow:
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
    encoding:   str


@dataclass
class AIConfig:
    skip_scoring:       bool  = False
    skip_captioning:    bool  = False
    skip_edge_maps:     bool  = False
    score_batch_size:   int   = DEFAULT_SCORE_BATCH
    caption_batch_size: int   = DEFAULT_CAPTION_BATCH
    edge_batch_size:    int   = DEFAULT_EDGE_BATCH
    device:             str   = "cpu"
    model_path:         Path  = DEFAULT_MODEL_PATH
    db_path:            Path  = DEFAULT_DB_PATH


# ════════════════════════════════════════════════════════════════════════════
# DATABASE
# ════════════════════════════════════════════════════════════════════════════

_NEW_PREVIEW_COLS: Dict[str, str] = {
    "scored":            "INTEGER NOT NULL DEFAULT 0",
    "score_overall":     "REAL",
    "score_quality":     "REAL",
    "score_composition": "REAL",
    "score_lighting":    "REAL",
    "score_color":       "REAL",
    "score_dof":         "REAL",
    "score_content":     "REAL",
    "captioned":         "INTEGER NOT NULL DEFAULT 0",
    "caption":           "TEXT",
    "edge_mapped":       "INTEGER NOT NULL DEFAULT 0",
    "edge_map_data":     "TEXT",
    "edge_map_encoding": "TEXT DEFAULT 'hex'",
    "ai_updated_at":     "TEXT",
}
_NEW_SESSION_COLS: Dict[str, str] = {
    "ai_status":     "TEXT DEFAULT 'pending'",
    "ai_updated_at": "TEXT",
}
_AI_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_previews_scored    ON previews(scored);
CREATE INDEX IF NOT EXISTS idx_previews_captioned ON previews(captioned);
CREATE INDEX IF NOT EXISTS idx_previews_edge      ON previews(edge_mapped);
"""
_REQUIRED_COLS = frozenset(_NEW_PREVIEW_COLS) | {
    "preview_id", "file_id", "preview_path"
}


class AIDB:
    def __init__(self, db_path: Path) -> None:
        if not db_path.exists():
            raise FileNotFoundError(
                f"Database not found: {db_path}\n"
                f"Run local_processor.py first."
            )
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._lock = threading.Lock()
        self._migrate()
        self._verify()

    def _migrate(self) -> None:
        ep = self._cols("previews")
        es = self._cols("sessions")
        with self._lock:
            for col, td in _NEW_PREVIEW_COLS.items():
                if col not in ep:
                    with contextlib.suppress(sqlite3.OperationalError):
                        self.conn.execute(
                            f"ALTER TABLE previews ADD COLUMN {col} {td}"
                        )
            for col, td in _NEW_SESSION_COLS.items():
                if col not in es:
                    with contextlib.suppress(sqlite3.OperationalError):
                        self.conn.execute(
                            f"ALTER TABLE sessions ADD COLUMN {col} {td}"
                        )
            with contextlib.suppress(sqlite3.OperationalError):
                self.conn.executescript(_AI_INDEXES)
            self.conn.commit()

    def _verify(self) -> None:
        missing = _REQUIRED_COLS - self._cols("previews")
        if missing:
            raise RuntimeError(
                f"Schema incomplete — missing columns: {sorted(missing)}"
            )

    def _cols(self, table: str) -> set:
        return {r[1] for r in
                self.conn.execute(f"PRAGMA table_info({table})").fetchall()}

    def list_sessions(self) -> List[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM sessions ORDER BY session_id DESC"
        ).fetchall()]

    def get_session(self, sid: int) -> Optional[dict]:
        r = self.conn.execute(
            "SELECT * FROM sessions WHERE session_id=?", (sid,)
        ).fetchone()
        return dict(r) if r else None

    def set_ai_status(self, sid: int, status: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE sessions SET ai_status=?, ai_updated_at=? "
                "WHERE session_id=?",
                (status, _now(), sid),
            )
            self.conn.commit()

    def get_previews(self, sid: int) -> List[PreviewRow]:
        rows = self.conn.execute(
            """
            SELECT p.preview_id, p.file_id, p.preview_path,
                   COALESCE(p.scored,      0) AS scored,
                   COALESCE(p.captioned,   0) AS captioned,
                   COALESCE(p.edge_mapped, 0) AS edge_mapped
            FROM previews p
            JOIN files f ON p.file_id = f.file_id
            WHERE f.session_id = ?
            ORDER BY p.preview_id
            """,
            (sid,),
        ).fetchall()
        return [
            PreviewRow(r["preview_id"], r["file_id"], r["preview_path"],
                       r["scored"], r["captioned"], r["edge_mapped"])
            for r in rows
        ]

    def get_unscored(self, sid: int) -> List[PreviewRow]:
        return [p for p in self.get_previews(sid) if not p.scored    and p.exists]

    def get_uncaptioned(self, sid: int) -> List[PreviewRow]:
        return [p for p in self.get_previews(sid) if not p.captioned and p.exists]

    def get_unmapped(self, sid: int) -> List[PreviewRow]:
        return [p for p in self.get_previews(sid) if not p.edge_mapped and p.exists]

    def batch_write_scores(self, payloads: List[ScorePayload]) -> None:
        if not payloads:
            return
        now = _now()
        with self._lock:
            for p in payloads:
                sc = p.scores
                self.conn.execute(
                    """UPDATE previews SET scored=1,
                        score_overall=?,     score_quality=?,
                        score_composition=?, score_lighting=?,
                        score_color=?,       score_dof=?,
                        score_content=?,     ai_updated_at=?
                       WHERE preview_id=?""",
                    (sc.get("overall"),     sc.get("quality"),
                     sc.get("composition"), sc.get("lighting"),
                     sc.get("color"),       sc.get("dof"),
                     sc.get("content"),     now,
                     p.preview_id),
                )
            self.conn.commit()

    def batch_write_captions(self, payloads: List[CaptionPayload]) -> None:
        if not payloads:
            return
        now = _now()
        with self._lock:
            for p in payloads:
                self.conn.execute(
                    "UPDATE previews SET captioned=1, caption=?, "
                    "ai_updated_at=? WHERE preview_id=?",
                    (p.caption, now, p.preview_id),
                )
            self.conn.commit()

    def batch_write_edge_maps(self, payloads: List[EdgePayload]) -> None:
        if not payloads:
            return
        now = _now()
        with self._lock:
            for p in payloads:
                self.conn.execute(
                    "UPDATE previews SET edge_mapped=1, edge_map_data=?, "
                    "edge_map_encoding=?, ai_updated_at=? WHERE preview_id=?",
                    (p.data, p.encoding, now, p.preview_id),
                )
            self.conn.commit()

    def reset_phase(self, sid: int, phase: str) -> int:
        col_map = {
            "scoring":    ("scored=0, score_overall=NULL, score_quality=NULL, "
                           "score_composition=NULL, score_lighting=NULL, "
                           "score_color=NULL, score_dof=NULL, score_content=NULL"),
            "captioning": "captioned=0, caption=NULL",
            "edge_maps":  "edge_mapped=0, edge_map_data=NULL",
        }
        if phase not in col_map:
            raise ValueError(f"Unknown phase {phase!r}")
        with self._lock:
            cur = self.conn.execute(
                f"UPDATE previews SET {col_map[phase]} "
                f"WHERE file_id IN (SELECT file_id FROM files WHERE session_id=?)",
                (sid,),
            )
            self.conn.commit()
            return cur.rowcount

    def ai_summary(self, sid: int) -> dict:
        rows    = self.get_previews(sid)
        total   = len(rows)
        present = sum(1 for r in rows if r.exists)
        scored  = sum(1 for r in rows if r.scored)
        cap     = sum(1 for r in rows if r.captioned)
        mapped  = sum(1 for r in rows if r.edge_mapped)
        return {
            "total_previews":  total,
            "present_on_disk": present,
            "scored":          scored,
            "captioned":       cap,
            "edge_mapped":     mapped,
            "unscored":        present - scored,
            "uncaptioned":     present - cap,
            "unmapped":        present - mapped,
        }

    def close(self) -> None:
        self.conn.close()


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ════════════════════════════════════════════════════════════════════════════

class AestheticScorer:
    def __init__(self, model_path: Path, device: str) -> None:
        self.model_path = model_path
        self.device     = device
        self._model     = None
        self._processor = None
        self._ready     = False

    def load(self, log: Callable[[str], None]) -> bool:
        if self._ready:
            return True
        if not self.model_path.exists():
            log(f"  [scorer] model.pt not found: {self.model_path}")
            return False
        try:
            import torch
            import torch.nn as nn
            from transformers import CLIPVisionModel, CLIPProcessor

            online     = _hf_online()
            local_only = not online and not _hf_cached(CLIP_MODEL_ID)

            class _ScorerNet(nn.Module):
                def __init__(si) -> None:
                    super().__init__()
                    si.backbone = CLIPVisionModel.from_pretrained(
                        CLIP_MODEL_ID, local_files_only=local_only
                    )
                    for cat in SCORE_CATEGORIES:
                        setattr(si, f"{cat}_head",
                                nn.Sequential(nn.Linear(768, 1)))

                def forward(si, pixel_values):
                    feat = si.backbone(
                        pixel_values=pixel_values
                    ).pooler_output
                    return torch.cat(
                        [getattr(si, f"{c}_head")(feat)
                         for c in SCORE_CATEGORIES],
                        dim=-1,
                    )

            log("  [scorer] Loading CLIP scorer …")
            orig_hf = os.environ.get("HF_HUB_OFFLINE")
            orig_tr = os.environ.get("TRANSFORMERS_OFFLINE")
            if local_only:
                os.environ["HF_HUB_OFFLINE"]      = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
            try:
                raw_sd = torch.load(self.model_path, map_location="cpu",
                                    weights_only=False)
                remapped: Dict[str, Any] = {}
                for k, v in raw_sd.items():
                    if (k.startswith("backbone.")
                            and not k.startswith("backbone.vision_model.")):
                        remapped["backbone.vision_model." + k[9:]] = v
                    else:
                        remapped[k] = v
                net = _ScorerNet()
                missing, _ = net.load_state_dict(remapped, strict=False)
                if missing:
                    log(f"  [scorer] {len(missing)} missing keys (expected)")
                net.to(self.device).eval()
                self._model     = net
                self._processor = CLIPProcessor.from_pretrained(
                    CLIP_MODEL_ID, local_files_only=local_only
                )
                self._ready = True
                log(f"  [scorer] Ready on {self.device}")
                return True
            finally:
                _restore_env("HF_HUB_OFFLINE",      orig_hf)
                _restore_env("TRANSFORMERS_OFFLINE", orig_tr)
        except Exception as exc:
            log(f"  [scorer] load error: {exc}")
            return False

    def score(self, image) -> Dict[str, float]:
        import torch
        inp = self._processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model(inp["pixel_values"]).squeeze().cpu()
        return {c: float(out[i]) for i, c in enumerate(SCORE_CATEGORIES)}

    def unload(self, log: Callable[[str], None]) -> None:
        if not self._ready:
            return
        import torch
        if self._model is not None:
            self._model.cpu()
            del self._model
        if self._processor is not None:
            del self._processor
        self._model = self._processor = None
        self._ready = False
        torch.cuda.empty_cache()
        gc.collect()
        log("  [scorer] Unloaded")


class FlorenceCaptioner:
    """
    Florence-2-large-PromptGen-v2.0 captioner.

    Load sequence
    -------------
    1. Stub any missing transformers.utils helpers (is_flash_attn_*)
    2. Patch the cached modeling_florence2.py to comment out flash_attn
       import lines that transformers' text-scanner trips on
    3. from_pretrained() with attn_implementation="eager" to force CPU-safe
       attention; this overrides any flash_attention_2 in the checkpoint config
    4. Runtime patches: forced_bos/eos_token_id, language_model.generate()

    Generation sequence (three-layer fallback)
    -------------------------------------------
    1. model.generate() with beam search
    2. model.generate() greedy (num_beams=1, different internal path)
    3. _manual_generate() — greedy via model.forward() + KV cache, completely
       bypasses generate() delegation chain
    """

    def __init__(self, device: str) -> None:
        self.device     = device
        self._model     = None
        self._processor = None
        self._dtype     = None
        self._ready     = False

    def load(self, log: Callable[[str], None]) -> bool:
        if self._ready:
            return True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            online     = _hf_online()
            local_only = not online and not _hf_cached(FLORENCE_MODEL_ID)

            # Step 1: stub missing transformers.utils helpers
            _stub_missing_transformers_utils()

            log("  [captioner] Loading Florence-2 …")

            if local_only and not _hf_cached(FLORENCE_MODEL_ID):
                log(f"  [captioner] not cached locally.")
                log(f"    huggingface-cli download {FLORENCE_MODEL_ID}")
                return False

            # Step 2: patch cached modeling_florence2.py
            # This must happen before from_pretrained() calls check_imports()
            _patch_florence_cached_file(log)

            dtype = torch.float16 if self.device == "cuda" else torch.float32

            orig_hf = os.environ.get("HF_HUB_OFFLINE")
            orig_tr = os.environ.get("TRANSFORMERS_OFFLINE")
            if local_only:
                os.environ["HF_HUB_OFFLINE"]      = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
            try:
                # Step 3: load with eager attention
                # attn_implementation="eager" overrides any flash_attention_2
                # setting in the checkpoint config.  Works on all hardware.
                self._model = AutoModelForCausalLM.from_pretrained(
                    FLORENCE_MODEL_ID,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    local_files_only=local_only,
                    attn_implementation="eager",
                ).eval()

                # Step 4a: forced_bos/eos_token_id on sub-configs
                _patch_generation_compat(self._model)

                # Step 4b: language_model.generate() from MRO if absent
                _patch_florence_lang_generate(self._model)

                self._processor = AutoProcessor.from_pretrained(
                    FLORENCE_MODEL_ID,
                    trust_remote_code=True,
                    local_files_only=local_only,
                )
                self._dtype = dtype
                self._ready = True
                log(f"  [captioner] Ready "
                    f"(device={self.device}, dtype={dtype}, attn=eager)")
                return True
            finally:
                _restore_env("HF_HUB_OFFLINE",      orig_hf)
                _restore_env("TRANSFORMERS_OFFLINE", orig_tr)

        except OSError as exc:
            log(f"  [captioner] OSError: {exc}")
            log(f"    huggingface-cli download {FLORENCE_MODEL_ID}")
            return False
        except Exception as exc:
            log(f"  [captioner] load error: {exc}")
            return False

    def caption(self, image) -> str:
        import torch

        if max(image.width, image.height) > CAPTION_MAX_DIM:
            image = image.copy()
            image.thumbnail((CAPTION_MAX_DIM, CAPTION_MAX_DIM))

        self._model.to(self.device)
        try:
            inp = self._processor(
                text=CAPTION_MODE,
                images=image,
                return_tensors="pt",
            ).to(self.device, self._dtype)

            with torch.no_grad():
                generated_ids = self._generate(inp)

            raw = self._processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]
            result = self._processor.post_process_generation(
                raw,
                task=CAPTION_MODE,
                image_size=(image.width, image.height),
            )
            return result[CAPTION_MODE].strip()
        finally:
            self._model.cpu()
            torch.cuda.empty_cache()

    def _generate(self, inp: Any) -> Any:
        import torch
        input_ids    = inp["input_ids"]
        pixel_values = inp["pixel_values"]

        # Strategy 1: beam search
        try:
            return self._model.generate(
                input_ids      = input_ids,
                pixel_values   = pixel_values,
                max_new_tokens = CAPTION_MAX_NEW_TOKENS,
                do_sample      = False,
                num_beams      = CAPTION_NUM_BEAMS,
            )
        except (AttributeError, TypeError) as exc:
            if "generate" not in str(exc).lower() and "attribute" not in str(exc).lower():
                raise

        # Strategy 2: greedy
        try:
            return self._model.generate(
                input_ids      = input_ids,
                pixel_values   = pixel_values,
                max_new_tokens = CAPTION_MAX_NEW_TOKENS,
                do_sample      = False,
                num_beams      = 1,
            )
        except (AttributeError, TypeError):
            pass

        # Strategy 3: manual forward loop
        return self._manual_generate(input_ids, pixel_values)

    def _manual_generate(self, input_ids: Any, pixel_values: Any) -> Any:
        """
        Greedy decode via model.forward() + KV cache.
        Bypasses generate() delegation entirely.
        model.forward() works even when generate() doesn't.
        """
        import torch

        tok    = self._processor.tokenizer
        eos_id = getattr(tok, "eos_token_id", None)
        if eos_id is None:
            eos_id = tok.convert_tokens_to_ids("</s>")
        if isinstance(eos_id, list):
            eos_id = eos_id[0]
        if eos_id is None:
            eos_id = 2

        generated = input_ids.clone()
        past_kv   = None

        for _ in range(CAPTION_MAX_NEW_TOKENS):
            cur_ids = generated if past_kv is None else generated[:, -1:]
            cur_pv  = pixel_values if past_kv is None else None

            with torch.no_grad():
                out = self._model(
                    input_ids       = cur_ids,
                    pixel_values    = cur_pv,
                    past_key_values = past_kv,
                    use_cache       = True,
                    return_dict     = True,
                )

            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_kv = out.past_key_values
            generated = torch.cat([generated, next_id], dim=-1)

            if next_id.item() == eos_id:
                break

        return generated

    def unload(self, log: Callable[[str], None]) -> None:
        if not self._ready:
            return
        import torch
        if self._model is not None:
            self._model.cpu()
            del self._model
        if self._processor is not None:
            del self._processor
        self._model = self._processor = self._dtype = None
        self._ready = False
        torch.cuda.empty_cache()
        gc.collect()
        log("  [captioner] Unloaded")


class EdgeMapper:
    def __init__(self, device: str) -> None:
        self.device = device
        self._hed   = None
        self._ready = False

    def load(self, log: Callable[[str], None]) -> bool:
        if self._ready:
            return True
        try:
            from controlnet_aux import HEDdetector
            online     = _hf_online()
            local_only = not online and not _hf_cached(HED_MODEL_ID)
            log("  [edge] Loading HED detector …")
            if local_only and not _hf_cached(HED_MODEL_ID):
                log(f"  [edge] not cached — huggingface-cli download {HED_MODEL_ID}")
                return False
            orig_hf = os.environ.get("HF_HUB_OFFLINE")
            if local_only:
                os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                self._hed   = HEDdetector.from_pretrained(HED_MODEL_ID)
                self._ready = True
                log("  [edge] Ready")
                return True
            finally:
                _restore_env("HF_HUB_OFFLINE", orig_hf)
        except ImportError:
            log("  [edge] controlnet_aux not installed — pip install controlnet-aux")
            return False
        except Exception as exc:
            log(f"  [edge] load error: {exc}")
            return False

    def detect(self, image) -> str:
        target = _bucket_size(image.width, image.height)
        edges  = self._hed(
            image, detect_resolution=512,
            image_resolution=max(target), safe=True,
        )
        buf = io.BytesIO()
        edges.resize(target).save(buf, format="WEBP", quality=EDGE_MAP_QUALITY)
        return _encode_image_bytes(buf.getvalue())

    def unload(self, log: Callable[[str], None]) -> None:
        if not self._ready:
            return
        import torch
        del self._hed
        self._hed   = None
        self._ready = False
        torch.cuda.empty_cache()
        gc.collect()
        log("  [edge] Unloaded")


# ════════════════════════════════════════════════════════════════════════════
# AI ENGINE
# ════════════════════════════════════════════════════════════════════════════

class AIProcessorEngine:
    def __init__(
        self,
        config:   AIConfig,
        log:      Callable[[str], None]                           = print,
        progress: Optional[Callable[[str, int, int, str], None]] = None,
        cancel:   Optional[threading.Event]                       = None,
    ) -> None:
        self.config    = config
        self._log      = log
        self._prog     = progress or (lambda *a: None)
        self.cancel    = cancel or threading.Event()
        self._pause    = threading.Event()
        self._pause.set()
        self.db:       Optional[AIDB] = None
        self.scorer    = AestheticScorer(config.model_path, config.device)
        self.captioner = FlorenceCaptioner(config.device)
        self.edger     = EdgeMapper(config.device)

    def pause(self)  -> None: self._pause.clear()
    def resume(self) -> None: self._pause.set()

    @property
    def is_paused(self) -> bool:
        return not self._pause.is_set()

    def _check(self) -> None:
        if not self._pause.is_set():
            self._log("  ⏸  Paused (VRAM held)")
            self._pause.wait()
        if self.cancel.is_set():
            raise InterruptedError("Cancelled")

    def run(self, session_id: int) -> dict:
        self._log("=" * 60)
        self._log(f"ai_process.py  —  session {session_id}")
        self._log(f"  DB      : {self.config.db_path}")
        self._log(f"  Device  : {self.config.device}")
        self._log(f"  Network : {'online' if _hf_online() else 'OFFLINE'}")
        self._log(
            f"  Score/Caption/Edge : "
            f"{'skip' if self.config.skip_scoring   else 'on'} / "
            f"{'skip' if self.config.skip_captioning else 'on'} / "
            f"{'skip' if self.config.skip_edge_maps  else 'on'}"
        )
        self._log("=" * 60)

        self.db  = AIDB(self.config.db_path)
        sess     = self.db.get_session(session_id)
        if not sess:
            raise ValueError(f"Session {session_id} not found")

        self._log(f"  Session : {sess.get('session_name', '?')}")
        self._log(f"  Source  : {sess.get('source_folder', '?')}")
        sm = self.db.ai_summary(session_id)
        self._log(
            f"  Previews: {sm['total_previews']} total, "
            f"{sm['present_on_disk']} on disk"
        )
        self._log(
            f"  Pending : scored={sm['unscored']}  "
            f"captioned={sm['uncaptioned']}  edge={sm['unmapped']}"
        )
        if sm["present_on_disk"] == 0:
            raise RuntimeError(
                "No previews on disk. Run local_processor.py first."
            )

        self.db.set_ai_status(session_id, "running")
        t0 = time.monotonic()
        try:
            self._phase_scoring(session_id)
            self._phase_captioning(session_id)
            self._phase_edge_maps(session_id)
        except InterruptedError:
            self.db.set_ai_status(session_id, "interrupted")
            self._log("\n✋ Interrupted — progress saved to DB")
            self.db.close()
            return {"status": "interrupted", "session_id": session_id}
        except Exception as exc:
            self.db.set_ai_status(session_id, "error")
            self._log(f"\n✗ Error: {exc}")
            self.db.close()
            raise

        elapsed = time.monotonic() - t0
        final   = self.db.ai_summary(session_id)
        self.db.set_ai_status(session_id, "completed")
        self.db.close()

        result = {
            "status":       "completed",
            "session_id":   session_id,
            "session_name": sess.get("session_name", ""),
            "scored":       final["scored"],
            "captioned":    final["captioned"],
            "edge_mapped":  final["edge_mapped"],
            "elapsed_s":    round(elapsed, 1),
            "device":       self.config.device,
        }
        self._log("\n" + "=" * 60)
        self._log("AI COMPLETE")
        for k, v in result.items():
            if k != "status":
                self._log(f"  {k}: {v}")
        self._log("=" * 60)
        return result

    def _phase_scoring(self, sid: int) -> None:
        self._log("\nPhase A · Aesthetic Scoring")
        if self.config.skip_scoring:
            self._log("  SKIP (--skip-scoring)"); return
        pending = self.db.get_unscored(sid)
        self._log(f"  Pending: {len(pending)}")
        if not pending: self._log("  Nothing to do"); return
        if not self.scorer.load(self._log):
            self._log("  Scoring unavailable — skipping"); return
        from PIL import Image
        buf: List[ScorePayload] = []
        bs = self.config.score_batch_size
        try:
            for i, row in enumerate(pending):
                self._check()
                try:
                    img    = Image.open(row.abs_path).convert("RGB")
                    scores = self.scorer.score(img)
                    buf.append(ScorePayload(row.preview_id, scores))
                    self._log(
                        f"    [{i+1}/{len(pending)}] {row.abs_path.name}: "
                        + "  ".join(f"{k[:3]}={v:+.2f}"
                                    for k, v in scores.items())
                    )
                except Exception as exc:
                    self._log(f"    ✗ {row.abs_path.name}: {exc}")
                if len(buf) >= bs:
                    self.db.batch_write_scores(buf)
                    self._log(f"    → {len(buf)} scores written")
                    buf = []
                self._prog("scoring", i + 1, len(pending), row.abs_path.name)
        finally:
            if buf:
                self.db.batch_write_scores(buf)
                self._log(f"    → {len(buf)} scores written (flush)")
            self.scorer.unload(self._log)

    def _phase_captioning(self, sid: int) -> None:
        self._log("\nPhase B · Captioning")
        if self.config.skip_captioning:
            self._log("  SKIP (--skip-captioning)"); return
        pending = self.db.get_uncaptioned(sid)
        self._log(f"  Pending: {len(pending)}")
        if not pending: self._log("  Nothing to do"); return
        if not self.captioner.load(self._log):
            self._log("  Captioning unavailable — skipping"); return
        from PIL import Image
        buf: List[CaptionPayload] = []
        bs = self.config.caption_batch_size
        try:
            for i, row in enumerate(pending):
                self._check()
                try:
                    img = Image.open(row.abs_path).convert("RGB")
                    cap = self.captioner.caption(img)
                    buf.append(CaptionPayload(row.preview_id, cap))
                    wc  = len(cap.split())
                    self._log(
                        f"    [{i+1}/{len(pending)}] {row.abs_path.name} "
                        f"({wc}w): {cap[:90]}"
                        + ("…" if len(cap) > 90 else "")
                    )
                except Exception as exc:
                    self._log(f"    ✗ {row.abs_path.name}: {exc}")
                if len(buf) >= bs:
                    self.db.batch_write_captions(buf)
                    self._log(f"    → {len(buf)} captions written")
                    buf = []
                self._prog("captioning", i + 1, len(pending), row.abs_path.name)
        finally:
            if buf:
                self.db.batch_write_captions(buf)
                self._log(f"    → {len(buf)} captions written (flush)")
            self.captioner.unload(self._log)

    def _phase_edge_maps(self, sid: int) -> None:
        self._log("\nPhase C · Edge Maps")
        if self.config.skip_edge_maps:
            self._log("  SKIP (--skip-edge-maps)"); return
        pending = self.db.get_unmapped(sid)
        self._log(f"  Pending: {len(pending)}")
        if not pending: self._log("  Nothing to do"); return
        if not self.edger.load(self._log):
            self._log("  Edge mapping unavailable — skipping"); return
        from PIL import Image
        buf: List[EdgePayload] = []
        bs = self.config.edge_batch_size
        try:
            for i, row in enumerate(pending):
                self._check()
                try:
                    img  = Image.open(row.abs_path).convert("RGB")
                    data = self.edger.detect(img)
                    kb   = (len(data) / 2 / 1024 if EDGE_MAP_ENCODING == "hex"
                            else len(data) * 3 / 4 / 1024)
                    buf.append(EdgePayload(row.preview_id, data, EDGE_MAP_ENCODING))
                    self._log(
                        f"    [{i+1}/{len(pending)}] "
                        f"{row.abs_path.name}: {kb:.1f} KB"
                    )
                except Exception as exc:
                    self._log(f"    ✗ {row.abs_path.name}: {exc}")
                if len(buf) >= bs:
                    self.db.batch_write_edge_maps(buf)
                    self._log(f"    → {len(buf)} edge maps written")
                    buf = []
                self._prog("edge_maps", i + 1, len(pending), row.abs_path.name)
        finally:
            if buf:
                self.db.batch_write_edge_maps(buf)
                self._log(f"    → {len(buf)} edge maps written (flush)")
            self.edger.unload(self._log)


# ════════════════════════════════════════════════════════════════════════════
# DEVICE / CLI
# ════════════════════════════════════════════════════════════════════════════

def _pick_device(requested: str) -> str:
    if requested not in ("auto", "cuda", "mps", "cpu"):
        return "cpu"
    try:
        import torch
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if (hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available()):
                return "mps"
            return "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if requested == "mps":
            if (hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available()):
                return "mps"
            return "cpu"
        return requested
    except ImportError:
        return "cpu"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AI inference layer for the Processor Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ai_process.py --session-id 1
  python ai_process.py --list-sessions
  python ai_process.py --session-id 1 --skip-scoring --device cuda
  python ai_process.py --session-id 1 --reset-phase captioning
  python ai_process.py --session-id 1 --skip-edge-maps --batch-size 4
""",
    )
    p.add_argument("--session-id",    type=int,   default=None)
    p.add_argument("--list-sessions", action="store_true")
    p.add_argument("--db",            default=str(DEFAULT_DB_PATH))
    p.add_argument("--model-pt",      default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--device",        default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--batch-size",    type=int,   default=None)
    p.add_argument("--skip-scoring",    action="store_true")
    p.add_argument("--skip-captioning", action="store_true")
    p.add_argument("--skip-edge-maps",  action="store_true")
    p.add_argument("--reset-phase",
                   choices=["scoring", "captioning", "edge_maps"],
                   default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args    = _build_parser().parse_args(argv)
    db_path = Path(args.db)

    if args.list_sessions:
        try:
            db   = AIDB(db_path)
            sess = db.list_sessions()
            db.close()
            if not sess:
                print("No sessions found.")
                return 0
            print(f"\n{'ID':>4}  {'Name':<28}  {'Status':<12}  "
                  f"{'Files':>6}  {'AI':<12}  Source")
            print("-" * 90)
            for s in sess:
                print(
                    f"{s['session_id']:>4}  "
                    f"{str(s.get('session_name','')):<28}  "
                    f"{str(s.get('status','')):<12}  "
                    f"{s.get('file_count',0):>6}  "
                    f"{str(s.get('ai_status','pending')):<12}  "
                    f"{s.get('source_folder','')}"
                )
            print()
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.session_id is None:
        print("ERROR: --session-id required.", file=sys.stderr)
        return 1

    if args.reset_phase:
        try:
            db = AIDB(db_path)
            n  = db.reset_phase(args.session_id, args.reset_phase)
            db.close()
            print(f"Reset {args.reset_phase}: {n} rows cleared.")
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.skip_scoring and args.skip_captioning and args.skip_edge_maps:
            return 0
        print("Re-running with reset phase …")

    device = _pick_device(args.device)
    bs     = args.batch_size
    cfg    = AIConfig(
        skip_scoring       = args.skip_scoring,
        skip_captioning    = args.skip_captioning,
        skip_edge_maps     = args.skip_edge_maps,
        score_batch_size   = bs or DEFAULT_SCORE_BATCH,
        caption_batch_size = bs or DEFAULT_CAPTION_BATCH,
        edge_batch_size    = bs or DEFAULT_EDGE_BATCH,
        device             = device,
        model_path         = Path(args.model_pt),
        db_path            = db_path,
    )
    engine = AIProcessorEngine(config=cfg, log=print)
    try:
        result = engine.run(args.session_id)
        return 0 if result.get("status") == "completed" else 1
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
