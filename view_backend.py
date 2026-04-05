"""
view_backend.py — Local Generation Backend for Photo Viewer

Provides:
  - ImageGenerator: SD 1.5 + ControlNet + Hyper-SD 4-step pipeline
  - GenerationCache: Generated image file management
  - HesitancyFilter: Caption cleanup for SD prompts
  - decode_edge_map(): Edge map blob → PIL Image
  - compute_bucket(): Aspect-ratio-aware resolution selection

No Tk imports. Communicates via return values and a log callable.
"""

import os
import io
import re
import glob
import base64
from typing import Optional, List, Callable, Tuple, Set

from PIL import Image

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

CLIP_MAX_WORDS = 77
GENERATED_DIR = "generated"

DEFAULT_CCS = 1.0
DEFAULT_CFG = 1.5
DEFAULT_STEPS = 4
DEFAULT_SEED = 42

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, cartoon, painting, text, watermark, duplicate"
)

CCS_PRESETS = [
    {"ccs": 0.6, "cfg": 1.2},
    {"ccs": 0.8, "cfg": 1.5},
    {"ccs": 1.0, "cfg": 1.5},
    {"ccs": 1.0, "cfg": 1.8},
    {"ccs": 1.1, "cfg": 2.0},
]

BUCKET_CANDIDATES = [
    (768, 512), (512, 768), (512, 832),
    (832, 512), (576, 768), (768, 576),
    (512, 512), (640, 640),
]

# Probe for torch/diffusers at import time
GENERATION_AVAILABLE = False
try:
    import torch
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        EulerDiscreteScheduler,
    )
    GENERATION_AVAILABLE = True
except ImportError:
    pass


# ════════════════════════════════════════════════════════════════════════════
# EDGE MAP UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def decode_edge_map(
    data: str, encoding: str = "hex"
) -> Optional[Image.Image]:
    """Decode edge map from hex or base64 string.
    Tries primary encoding first, falls back to the other."""
    if not data:
        return None

    raw_bytes = None

    if encoding == "base64":
        try:
            raw_bytes = base64.b64decode(data)
        except Exception:
            pass
    else:
        try:
            raw_bytes = bytes.fromhex(data)
        except Exception:
            pass

    # Fallback
    if raw_bytes is None:
        if encoding == "base64":
            try:
                raw_bytes = bytes.fromhex(data)
            except Exception:
                pass
        else:
            try:
                raw_bytes = base64.b64decode(data)
            except Exception:
                pass

    if raw_bytes is None:
        return None

    try:
        return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        return None


def compute_bucket(w: int, h: int) -> Tuple[int, int]:
    """Compute optimal bucket size for generation."""
    if w <= 0 or h <= 0:
        return (512, 512)
    ratio = w / h
    return min(BUCKET_CANDIDATES, key=lambda s: abs((s[0] / s[1]) - ratio))


# ════════════════════════════════════════════════════════════════════════════
# HESITANCY FILTER
# ════════════════════════════════════════════════════════════════════════════

class HesitancyFilter:
    """Removes hedging words/phrases from prompts before SD generation.
    Supports live reload when the source file changes on disk."""

    def __init__(self, filepath: str, log: Callable[[str], None] = print):
        self.filepath = filepath
        self.phrases: List[str] = []
        self._log = log
        self._last_mtime: float = 0
        self._load()

    def _load(self):
        if not os.path.isfile(self.filepath):
            self._log("No hesitancy.txt found — skipping filter")
            self._last_mtime = 0
            self.phrases = []
            return

        try:
            mtime = os.path.getmtime(self.filepath)
            with open(self.filepath, "r", encoding="utf-8") as f:
                lines = [
                    line.strip().lower()
                    for line in f
                    if line.strip() and not line.strip().startswith("#")
                ]
            self.phrases = sorted(lines, key=len, reverse=True)
            self._last_mtime = mtime
            self._log(
                f"Loaded {len(self.phrases)} hesitancy filters "
                f"from hesitancy.txt"
            )
        except Exception as e:
            self._log(f"Error loading hesitancy.txt: {e}")

    def _check_reload(self):
        if not os.path.isfile(self.filepath):
            if self.phrases:
                self.phrases = []
                self._last_mtime = 0
            return
        try:
            mtime = os.path.getmtime(self.filepath)
            if mtime > self._last_mtime:
                self._log("hesitancy.txt changed, reloading...")
                self._load()
        except Exception:
            pass

    def filter(self, text: str) -> str:
        """Remove hesitancy phrases and truncate to CLIP token limit."""
        if not text:
            return ""

        self._check_reload()

        result = text.lower()
        for phrase in self.phrases:
            result = result.replace(phrase, " ")

        result = re.sub(r"\s+", " ", result).strip()

        words = result.split()
        if len(words) > CLIP_MAX_WORDS:
            result = " ".join(words[:CLIP_MAX_WORDS])

        return result


# ════════════════════════════════════════════════════════════════════════════
# GENERATION CACHE
# ════════════════════════════════════════════════════════════════════════════

class GenerationCache:
    """Manages generated images in a flat folder.

    Naming convention:
        Local SD:    {stem}_gen001.jpg, _gen002.jpg, ...
        API refine:  {stem}_gem001.jpg, _gem002.jpg, ...

    The `prefix` parameter controls which family of files is addressed.
    """

    def __init__(self, base_dir: str):
        self.dir = os.path.join(base_dir, GENERATED_DIR)
        os.makedirs(self.dir, exist_ok=True)
        # stem → set of prefixes that exist ("gen", "gem")
        self._stem_prefixes: dict[str, Set[str]] = {}
        self._scan_existing()

    # ── internal ──────────────────────────────────────────────────────────

    def _scan_existing(self):
        """One-time startup scan of generated/ folder."""
        pattern = os.path.join(self.dir, "*_ge[nm]*.jpg")
        for f in glob.glob(pattern):
            basename = os.path.basename(f)
            match = re.match(r"(.+)_(gen|gem)\d+\.jpg$", basename)
            if match:
                stem = match.group(1)
                prefix = match.group(2)
                if stem not in self._stem_prefixes:
                    self._stem_prefixes[stem] = set()
                self._stem_prefixes[stem].add(prefix)

    def _stem_from_preview(self, preview_path: str) -> str:
        if not preview_path:
            return "unknown"
        basename = os.path.basename(preview_path)
        stem = os.path.splitext(basename)[0]
        return stem or "unknown"

    # ── queries ───────────────────────────────────────────────────────────

    def has_generation(
        self, preview_path: str, prefix: str = "gen"
    ) -> bool:
        """Fast check using cached stem set."""
        if not preview_path:
            return False
        stem = self._stem_from_preview(preview_path)
        return prefix in self._stem_prefixes.get(stem, set())

    def has_any_generation(self, preview_path: str) -> bool:
        """Check if any type of generation exists."""
        if not preview_path:
            return False
        stem = self._stem_from_preview(preview_path)
        return stem in self._stem_prefixes

    def get_latest(
        self, preview_path: str, prefix: str = "gen"
    ) -> Optional[str]:
        """Get path to latest generated image of given type."""
        matches = self.get_all_versions(preview_path, prefix)
        return matches[-1] if matches else None

    def get_all_versions(
        self, preview_path: str, prefix: str = "gen"
    ) -> List[str]:
        """Get all versions for a file and prefix, sorted."""
        if not preview_path:
            return []
        stem = self._stem_from_preview(preview_path)
        pattern = os.path.join(self.dir, f"{stem}_{prefix}*.jpg")
        return sorted(glob.glob(pattern))

    def get_version_count(
        self, preview_path: str, prefix: str = "gen"
    ) -> int:
        return len(self.get_all_versions(preview_path, prefix))

    def count_generated(self, prefix: str = "gen") -> int:
        """Count distinct stems that have at least one generation."""
        return sum(
            1 for prefixes in self._stem_prefixes.values()
            if prefix in prefixes
        )

    def count_all_generated(self) -> int:
        """Count distinct stems that have any generation type."""
        return len(self._stem_prefixes)

    # ── save ──────────────────────────────────────────────────────────────

    def save_new(
        self,
        preview_path: str,
        image: Image.Image,
        prefix: str = "gen",
        quality: int = 90,
    ) -> str:
        """Save new generation, returns output path."""
        stem = self._stem_from_preview(preview_path)
        existing = self.get_all_versions(preview_path, prefix)

        next_num = 1
        if existing:
            last = os.path.basename(existing[-1])
            match = re.search(rf"_{prefix}(\d+)\.jpg$", last)
            if match:
                next_num = int(match.group(1)) + 1

        filename = f"{stem}_{prefix}{next_num:03d}.jpg"
        path = os.path.join(self.dir, filename)
        image.save(path, quality=quality)

        # Update cache
        if stem not in self._stem_prefixes:
            self._stem_prefixes[stem] = set()
        self._stem_prefixes[stem].add(prefix)

        return path


# ════════════════════════════════════════════════════════════════════════════
# IMAGE GENERATOR (Lazy-Loading SD Pipeline)
# ════════════════════════════════════════════════════════════════════════════

class ImageGenerator:
    """SD 1.5 + ControlNet (HED softedge) + Hyper-SD 4-step Lightning.
    Pipeline is loaded lazily on first generate call."""

    def __init__(self, log: Callable[[str], None] = print):
        self._log = log
        self.pipe = None
        self.device = "cpu"
        self._loaded = False
        self._loading = False

    def is_loaded(self) -> bool:
        return self._loaded

    def is_loading(self) -> bool:
        return self._loading

    def load(self) -> bool:
        """Load the full pipeline. Returns True on success."""
        if self._loaded:
            return True
        if self._loading:
            return False
        if not GENERATION_AVAILABLE:
            self._log(
                "Generation unavailable (torch/diffusers not installed)"
            )
            return False

        self._loading = True

        try:
            import torch as _torch
            from diffusers import (
                ControlNetModel as _CN,
                StableDiffusionControlNetPipeline as _Pipe,
                EulerDiscreteScheduler as _Sched,
            )

            self.device = (
                "cuda" if _torch.cuda.is_available() else "cpu"
            )
            dtype = (
                _torch.float16
                if self.device == "cuda"
                else _torch.float32
            )

            self._log("Loading ControlNet (softedge)...")
            controlnet = _CN.from_pretrained(
                "lllyasviel/control_v11p_sd15_softedge",
                torch_dtype=dtype,
            )

            self._log("Loading SD 1.5 pipeline...")
            self.pipe = _Pipe.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=dtype,
                safety_checker=None,
            )

            self._log("Loading Hyper-SD 4-step LoRA...")
            self.pipe.load_lora_weights(
                "ByteDance/Hyper-SD",
                weight_name="Hyper-SD15-4steps-lora.safetensors",
            )

            try:
                self.pipe.fuse_lora()
                self._log("LoRA fused")
            except Exception as e:
                self._log(f"LoRA fusion skipped: {e}")

            self.pipe.scheduler = _Sched.from_config(
                self.pipe.scheduler.config,
                timestep_spacing="trailing",
            )

            self._log("Applying optimizations...")
            if self.device == "cuda":
                self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()

            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                self._log("xformers enabled")
            except Exception:
                self.pipe.enable_attention_slicing()
                self._log(
                    "Attention slicing enabled (xformers unavailable)"
                )

            self._loaded = True
            self._loading = False
            self._log("Pipeline ready")
            return True

        except Exception as e:
            self._log(f"Pipeline load failed: {e}")
            self._loading = False
            return False

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        edge_image: Image.Image,
        ccs: float = DEFAULT_CCS,
        cfg: float = DEFAULT_CFG,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,
    ) -> Optional[Image.Image]:
        """Generate image from prompt + edge map. Returns PIL Image or None."""
        if not self._loaded:
            self._log("Pipeline not loaded")
            return None

        import torch as _torch

        bucket = compute_bucket(edge_image.width, edge_image.height)
        edge_resized = edge_image.resize(bucket, Image.LANCZOS)

        try:
            gen_device = "cpu" if self.device == "cpu" else self.device
            generator = _torch.Generator(
                device=gen_device
            ).manual_seed(seed)

            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=edge_resized,
                num_inference_steps=steps,
                guidance_scale=cfg,
                controlnet_conditioning_scale=ccs,
                width=bucket[0],
                height=bucket[1],
                generator=generator,
            ).images[0]

            return result

        except Exception as e:
            self._log(f"Generation failed: {e}")
            return None

    def unload(self):
        """Unload pipeline to free VRAM/RAM."""
        if self._loaded:
            del self.pipe
            self.pipe = None
            self._loaded = False
            if GENERATION_AVAILABLE:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            self._log("Pipeline unloaded")