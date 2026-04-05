# Photo Processor Pipeline
A resumable, multi-stage RAW photo ingestion pipeline with AI-powered aesthetic scoring, captioning, and edge detection. Built for Nikon Z30 and Sony Alpha series (A7R2, A7III, A7R4).
```
┌─────────────────────────────────────────────────────────────┐
│   processor_gui.py          [Presentation Layer]            │
│   ─────────────────────────────────────────────────────────  │
│   Tkinter GUI • Threading bridge • Session management       │
└──────────────────────────┬─���────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│   processor_logic.py        [Business Logic Layer]          │
│   ─────────────────────────────────────────────────────────  │
│   ProcessorEngine • ProcessorDB • AI modules                │
│   AestheticScorer • FlorenceCaptioner • EdgeMapper          │
└──────────────────────────┬──────────────────────────────────┘
                           │ optional
┌──────────────────────────▼──────────────────────────────────┐
│   hdd_diagnostics.py        [Observability Layer]           │
│   ─────────────────────────────────────────────────────────  │
│   DriveMonitor • SMART attributes • Kernel I/O stats        │
└─────────────────────────────────────────────────────────────┘
```
---
## Status
| Branch | Platform | Status | Notes |
|--------|----------|--------|-------|
| `main` | Linux (Xubuntu) | ✅ Stable | Fully tested on ~20,000 photos |
| `cross-platform` | Linux + Windows | ⚠️ Beta | Core works, some bugs remain |
| `mac-arm` | macOS Apple Silicon | 🔴 TODO | Planned |
---
## Features
### Core Pipeline
- **RAW Ingestion** — NEF (Nikon) and ARW (Sony) support
- **Preview Extraction** — Embedded JPEG extraction with auto-orientation fix
- **Burst Detection** — Groups consecutive shots (configurable gap threshold)
- **Metadata Extraction** — Full EXIF via ExifTool (shutter, ISO, aperture, focal length, focus distance)
### AI Processing
- **Aesthetic Scoring** — CLIP-based model scores 7 categories (overall, quality, composition, lighting, color, dof, content)
- **Auto Captioning** — Florence-2-large generates detailed image descriptions
- **Edge Mapping** — HED (Holistically-Nested Edge Detection) for structure analysis
### Pipeline Architecture
- **Resumable** — SQLite-backed state, crash recovery, per-file completion tracking
- **Pause/Resume** — AI models stay loaded in VRAM during pause (no 30-60s reload penalty)
- **Batch Writes** — Progress saved every N items, never lose work
- **Offline Capable** — HuggingFace models cached locally, works without internet
### HDD Diagnostics (Linux)
- **SMART Monitoring** — Tracks attribute changes during heavy I/O
- **Kernel I/O Stats** — Read throughput, latency via `/sys/block`
- **Throughput Calibration** — EMA-based expected time, flags slow/retry reads
- **Drive Classification** — Auto-detects HDD vs SSD vs NVMe
---
## Tested Cameras
| Camera | Format | Status |
|--------|--------|--------|
| Nikon Z30 | NEF | ✅ Primary |
| Sony A7R IV | ARW | ✅ Primary |
| Sony A7R II | ARW | ✅ Verified |
| Sony A7 III | ARW | ✅ Verified |
*Should work with any Nikon NEF or Sony ARW camera.*
---
## Requirements
### Minimum
- Python 3.9+
- 8 GB RAM (4 GB for models, 4 GB for OS)
- 5 GB disk (PyTorch + Florence-2 + model.pt)
### Recommended
- NVIDIA GPU with CUDA (25× speedup on AI phases)
- 16 GB RAM
- SSD for preview storage
### Dependencies
```
# Core
torch torchvision
transformers accelerate safetensors
Pillow opencv-python
# AI Models
controlnet-aux  # HED edge detection
# External Tools
exiftool        # metadata extraction
```
---
## Installation
### Linux (Xubuntu) — Manual Install
Tested and stable. No setup script needed.
```bash
# 1. System packages
sudo apt update
sudo apt install python3-venv python3-pip exiftool
# Optional: HDD diagnostics
sudo apt install smartmontools
# Optional: JPEG orientation fix
sudo apt install imagemagick
# 2. Clone and create venv
git clone https://github.com/YOUR_USERNAME/photo-processor.git
cd photo-processor
python3 -m venv venv
source venv/bin/activate
# 3. Install Python packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate safetensors
pip install Pillow opencv-python controlnet-aux
# 4. Download AI model weights (first run needs internet)
# Models will cache to ~/.cache/huggingface/
# 5. Place your aesthetic scorer
cp /path/to/model.pt .
# 6. Run
python processor_gui.py
```
### Cross-Platform (Linux + Windows) — Beta
Use `processor_setup.py` for automated bootstrap:
```bash
python processor_setup.py
```
This script:
- Detects OS/distro
- Installs system packages via apt/dnf/pacman/brew
- Downloads ExifTool binary (Windows)
- Installs Python packages
- Pre-caches HuggingFace models
- Optionally downloads `model.pt` from URL
**Known Issues (Cross-Platform Branch):**
- Preview path handling needs testing on Windows
- HDD diagnostics Windows adapter incomplete
- Some edge cases in resumability
---
## Project Structure
```
photo-processor/
├── processor_gui.py          # Tkinter GUI (entry point)
├── processor_logic.py        # Core engine + DB + AI modules
├── hdd_diagnostics.py        # Drive health monitor (optional)
├── processor_setup.py        # Cross-platform bootstrap
│
├── model.pt                  # Aesthetic scorer weights (user-supplied)
├── ingest.db                 # SQLite database (auto-created)
├── previews/                 # Extracted JPEG previews
│   └── *.jpg
│
├── skins/                    # GUI themes (optional)
│   └── default.json
│
└── README.md
```
### Modular Architecture (Experimental)
New split for better separation of concerns:
```
├── l_process.py              # Local processing (scan, extract, verify)
├── ai_process.py             # AI processing (score, caption, edge)
├── view_gui.py               # Standalone viewer (in development)
```
---
## Usage
### Starting a New Session
1. Click **Browse…** to select a folder with RAW files
2. Click **Run All (F5)** to start the pipeline
3. Monitor progress in the log pane
### Resuming a Session
1. Click **Resume (F7)** to see sessions with pending work
2. Select a session to continue from last checkpoint
3. Already-processed files are skipped automatically
### During Processing
| Action | Shortcut | Notes |
|--------|----------|-------|
| Pause | F6 | Models stay in VRAM, resume is instant |
| Resume | F6 | Continues where paused |
| Cancel | Esc | Saves progress, can resume later |
### AI Phases
The pipeline runs in order:
```
Phase 0  → Dependency check
Phase 1  → Tool verification (ExifTool)
Phase 2  → Session creation/resume
Phase 3  → RAW file scan
Phase 4  → Metadata extraction (batch)
Phase 5  → Burst detection
Phase 6  → Preview extraction
Phase 7  → Aesthetic scoring (CLIP)
Phase 8  → Captioning (Florence-2)
Phase 9  → Edge mapping (HED)
Phase 10 → Finalize
```
Each phase is resumable. If phase 8 crashes at file 500, resume continues from file 501.
---
## Database Schema
### Sessions
```sql
session_id     INTEGER PRIMARY KEY
session_name   TEXT
source_folder  TEXT
file_count     INTEGER
status         TEXT  -- 'in_progress', 'completed', 'error'
```
### Files
```sql
file_id        INTEGER PRIMARY KEY
session_id     INTEGER
file_path      TEXT UNIQUE
camera_make    TEXT
camera_model   TEXT
capture_time   TEXT
shutter_speed  TEXT
iso            INTEGER
aperture       REAL
focal_length   REAL
burst_id       INTEGER
```
### Previews
```sql
preview_id     INTEGER PRIMARY KEY
file_id        INTEGER UNIQUE
preview_path   TEXT
score_overall  REAL
caption        TEXT
edge_map_data  TEXT
scored         BOOLEAN
captioned      BOOLEAN
edge_mapped    BOOLEAN
```
---
## Viewer (In Development)
`view_gui.py` provides a standalone gallery viewer:
```bash
python view_gui.py
```
**Current Status:**
- Basic grid view implemented
- API integration is a **dummy placeholder**
- Active development in progress
---
## Configuration
### Environment Variables
| Variable | Purpose |
|----------|---------|
| `HF_HUB_OFFLINE=1` | Force offline mode for HuggingFace |
| `HF_TOKEN` | Authenticated HF downloads (avoids rate limits) |
| `CUDA_VISIBLE_DEVICES` | Select GPU |
### GPU Selection
```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python processor_gui.py
# Force CPU (slow)
CUDA_VISIBLE_DEVICES="" python processor_gui.py
```
---
## Troubleshooting
| Symptom | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError: model.pt` | Scorer weights missing | Place `model.pt` in script directory |
| `ExifTool not found` | Binary not in PATH | Install via package manager or run `processor_setup.py` |
| GUI freezes | Long op in main thread | Ensure all phases run in worker thread |
| CUDA out of memory | Batch size too large | Reduce `SCORE_BATCH_SIZE` in code |
| Florence download fails | HF rate limit | Set `HF_TOKEN` env var |
| Database is locked | Concurrent access | Only one engine per DB at a time |
| SMART permission denied | NTFS/udisks2 on Linux | Use `sudo smartctl -a /dev/sdX` |
---
## License
```
AGPLv3 — GNU Affero General Public License v3
Copyright (C) 2026 ArcticWinter
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.
Network use notice: If you modify this program and provide it as a
service over a network, you must make the modified source code
available to users of that service.
```
---
## Roadmap
### v1.1 (Planned)
- [ ] Merged release (unified Linux + Windows codebase)
- [ ] macOS Apple Silicon support
- [ ] Built-in gallery viewer
- [ ] Export to CSV/JSON
### v1.2 (Future)
- [ ] Multi-folder batch mode
- [ ] Live preview during scoring
- [ ] DNG support
- [ ] Video file support (MOV/MP4)
---
## Acknowledgments
- **ExifTool** by Phil Harvey — Metadata extraction
- **Florence-2** by Microsoft — Vision-language model
- **CLIP** by OpenAI — Image embeddings
- **HED** via controlnet-aux — Edge detection
---
## Author
**ArcticWinter**
---
## Contributing
Issues and pull requests welcome. For major changes, please open an issue first to discuss what you would like to change.
When contributing code, ensure:
1. AGPLv3 license header on all Python files
2. Type hints on function signatures
3. Docstrings on public methods
4. Test with `--dry-run` on small folder first
