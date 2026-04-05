#!/usr/bin/env python3
"""
hdd_diagnostics.py — Drive health monitor for photo processor pipeline.

Real-world Linux: NTFS via udisks2 on Xubuntu, permission-denied smartctl,
USB-attached externals with bridge controllers that hide serial numbers.

Three monitoring tiers:
  TIER 1: SMART + kernel I/O + timing   (smartctl accessible)
  TIER 2: Kernel I/O + timing           (no smartctl — DEFAULT for NTFS)
  TIER 3: Skip                          (SSD / NVMe / unclassifiable)

Kernel I/O from /sys/block/sdX/stat — no root needed, gives:
  - sectors actually read by the kernel
  - time the kernel spent in I/O (ms)
  - lets you separate "drive was slow" from "exiftool was slow"

Standalone:
    python hdd_diagnostics.py /media/user/drive
    python hdd_diagnostics.py /media/user/drive --smart
    python hdd_diagnostics.py /media/user/drive --benchmark
    python hdd_diagnostics.py /media/user/drive --json

No pip dependencies. Uses: lsblk, findmnt (util-linux).
Optional: smartctl (smartmontools), udevadm (udev).
"""

__version__ = "3.0.0"

import os
import re
import sys
import json
import time
import sqlite3
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable, Any
from datetime import datetime


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

WATCHED_SMART_ATTRS = {
    1:   "Raw_Read_Error_Rate",
    5:   "Reallocated_Sector_Ct",
    7:   "Seek_Error_Rate",
    10:  "Spin_Retry_Count",
    187: "Reported_Uncorrectable",
    188: "Command_Timeout",
    195: "Hardware_ECC_Recovered",
    197: "Current_Pending_Sector",
    198: "Offline_Uncorrectable",
    199: "UDMA_CRC_Error_Count",
}

WARMUP_READS         = 12
SLOW_SEEK_MULTIPLIER = 3.0
RETRY_MULTIPLIER     = 8.0
TIMEOUT_ABS_MS       = 30_000
EMA_ALPHA            = 0.08
READ_LOG_BATCH_SIZE  = 50
SECTOR_BYTES         = 512
MAX_DM_DEPTH         = 10


# ════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class DriveInfo:
    device_path:      str  = ""
    parent_device:    str  = ""
    drive_type:       str  = "unknown"
    model_name:       str  = ""
    serial_number:    str  = ""
    firmware:         str  = ""
    capacity_bytes:   int  = 0
    filesystem:       str  = ""
    mount_point:      str  = ""
    rotational:       Optional[bool] = None
    smart_accessible: bool = False
    smart_mode:       str  = "none"
    has_diskstats:    bool = False


@dataclass
class DiskIOSnapshot:
    """Parsed from /sys/block/sdX/stat — 11 fields, all integers."""
    read_ios:        int = 0
    read_merges:     int = 0
    read_sectors:    int = 0
    read_time_ms:    int = 0
    write_ios:       int = 0
    write_merges:    int = 0
    write_sectors:   int = 0
    write_time_ms:   int = 0
    in_flight:       int = 0
    io_time_ms:      int = 0
    weighted_ms:     int = 0
    timestamp:       str = ""

    @property
    def read_bytes(self) -> int:
        return self.read_sectors * SECTOR_BYTES

    def delta(self, earlier: "DiskIOSnapshot") -> "DiskIOSnapshot":
        return DiskIOSnapshot(
            read_ios=self.read_ios - earlier.read_ios,
            read_merges=self.read_merges - earlier.read_merges,
            read_sectors=self.read_sectors - earlier.read_sectors,
            read_time_ms=self.read_time_ms - earlier.read_time_ms,
            write_ios=self.write_ios - earlier.write_ios,
            write_merges=self.write_merges - earlier.write_merges,
            write_sectors=self.write_sectors - earlier.write_sectors,
            write_time_ms=self.write_time_ms - earlier.write_time_ms,
            in_flight=self.in_flight,
            io_time_ms=self.io_time_ms - earlier.io_time_ms,
            weighted_ms=self.weighted_ms - earlier.weighted_ms,
            timestamp=self.timestamp)

    def to_dict(self) -> dict:
        return {
            "read_ios": self.read_ios,
            "read_sectors": self.read_sectors,
            "read_bytes": self.read_bytes,
            "read_time_ms": self.read_time_ms,
            "write_ios": self.write_ios,
            "write_sectors": self.write_sectors,
            "io_time_ms": self.io_time_ms,
            "timestamp": self.timestamp,
            "kernel_read_mbps": round(
                self.read_bytes / self.read_time_ms / 1000, 2)
            if self.read_time_ms > 0 else 0,
        }


@dataclass
class ReadEvent:
    file_path:       str
    file_size_bytes: int
    read_start_ts:   str
    read_end_ts:     str
    read_time_ms:    float
    expected_ms:     float
    flag:            Optional[str]  = None
    detail:          Optional[str]  = None
    read_type:       str            = "single"
    kernel_read_ms:  Optional[float] = None
    kernel_sectors:  Optional[int]   = None


@dataclass
class ThroughputStats:
    count:       int   = 0
    total_bytes: int   = 0
    total_ms:    float = 0.0
    min_bpms:    float = float('inf')
    max_bpms:    float = 0.0
    ema_bpms:    float = 0.0    # starts at 0 — first read sets it
    _m2:         float = 0.0
    _mean:       float = 0.0

    def update(self, size_bytes: int, time_ms: float):
        if time_ms <= 0 or size_bytes <= 0:
            return
        bpms = size_bytes / time_ms
        self.count += 1
        self.total_bytes += size_bytes
        self.total_ms += time_ms
        if bpms < self.min_bpms:
            self.min_bpms = bpms
        if bpms > self.max_bpms:
            self.max_bpms = bpms
        if self.count == 1:
            self.ema_bpms = bpms
        else:
            self.ema_bpms = EMA_ALPHA * bpms + (1 - EMA_ALPHA) * self.ema_bpms
        delta = bpms - self._mean
        self._mean += delta / self.count
        delta2 = bpms - self._mean
        self._m2 += delta * delta2

    @property
    def calibrated(self) -> bool:
        return self.count >= 1

    @property
    def avg_mbps(self) -> float:
        return (self.total_bytes / self.total_ms / 1000) if self.total_ms > 0 else 0

    @property
    def ema_mbps(self) -> float:
        return self.ema_bpms / 1000 if self.ema_bpms > 0 else 0

    @property
    def stddev_mbps(self) -> float:
        if self.count < 2:
            return 0
        return (self._m2 / (self.count - 1)) ** 0.5 / 1000

    @property
    def min_mbps(self) -> float:
        return self.min_bpms / 1000 if self.min_bpms != float('inf') else 0

    @property
    def max_mbps(self) -> float:
        return self.max_bpms / 1000

    def summary_dict(self) -> dict:
        return {
            "reads":     self.count,
            "total_mb":  round(self.total_bytes / (1024 * 1024), 1),
            "total_sec": round(self.total_ms / 1000, 1),
            "avg_mbps":  round(self.avg_mbps, 2),
            "ema_mbps":  round(self.ema_mbps, 2),
            "stddev":    round(self.stddev_mbps, 2),
            "min_mbps":  round(self.min_mbps, 2),
            "max_mbps":  round(self.max_mbps, 2),
        }


# ════════════════════════════════════════════════════════════════════════════
# DDL — inert tables + migration
# ════════════════════════════════════════════════════════════════════════════

SMART_DDL = """
CREATE TABLE IF NOT EXISTS drive_info (
    drive_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id         INTEGER NOT NULL,
    device_path        TEXT,
    serial_number      TEXT,
    model_name         TEXT,
    firmware           TEXT,
    drive_type         TEXT,
    filesystem         TEXT,
    capacity_bytes     INTEGER,
    smart_mode         TEXT,
    smart_start        TEXT,
    smart_end          TEXT,
    throughput_json    TEXT,
    kernel_io_start    TEXT,
    kernel_io_end      TEXT,
    scan_start_ts      TEXT,
    scan_end_ts        TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS drive_read_log (
    read_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    drive_id           INTEGER NOT NULL,
    session_id         INTEGER NOT NULL,
    file_path          TEXT,
    file_size_bytes    INTEGER,
    read_start_ts      TEXT,
    read_end_ts        TEXT,
    read_time_ms       REAL,
    expected_ms        REAL,
    flag               TEXT,
    detail             TEXT,
    read_type          TEXT DEFAULT 'single',
    kernel_read_ms     REAL,
    kernel_sectors     INTEGER,
    FOREIGN KEY (drive_id)   REFERENCES drive_info(drive_id),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS drive_smart_deltas (
    delta_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    drive_id           INTEGER NOT NULL,
    attribute_id       INTEGER,
    attribute_name     TEXT,
    value_start        INTEGER,
    value_end          INTEGER,
    raw_start          INTEGER,
    raw_end            INTEGER,
    delta              INTEGER,
    FOREIGN KEY (drive_id) REFERENCES drive_info(drive_id)
);

CREATE INDEX IF NOT EXISTS idx_read_log_session  ON drive_read_log(session_id);
CREATE INDEX IF NOT EXISTS idx_read_log_flag     ON drive_read_log(flag);
CREATE INDEX IF NOT EXISTS idx_read_log_type     ON drive_read_log(read_type);
CREATE INDEX IF NOT EXISTS idx_smart_deltas_drv  ON drive_smart_deltas(drive_id);
"""

# columns that may be missing in older databases
_MIGRATIONS = [
    ("drive_info",     "filesystem",      "TEXT"),
    ("drive_info",     "smart_mode",      "TEXT"),
    ("drive_info",     "throughput_json",  "TEXT"),
    ("drive_info",     "kernel_io_start",  "TEXT"),
    ("drive_info",     "kernel_io_end",    "TEXT"),
    ("drive_read_log", "read_type",        "TEXT DEFAULT 'single'"),
    ("drive_read_log", "kernel_read_ms",   "REAL"),
    ("drive_read_log", "kernel_sectors",   "INTEGER"),
]


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

_OCTAL_RE = re.compile(r"\\([0-7]{3})")


def _unescape(s: str) -> str:
    return _OCTAL_RE.sub(lambda m: chr(int(m.group(1), 8)), s)


def _run(cmd: list, timeout: int = 5) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _read_sysfs(path: str) -> Optional[str]:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except (IOError, PermissionError, FileNotFoundError):
        return None


def _capture_diskstats(parent_dev: str) -> Optional[DiskIOSnapshot]:
    """Read /sys/block/<dev>/stat — works without root on all Linux."""
    base = os.path.basename(parent_dev)
    stat_path = f"/sys/block/{base}/stat"
    raw = _read_sysfs(stat_path)
    if raw is None:
        return None
    parts = raw.split()
    if len(parts) < 11:
        return None
    try:
        nums = [int(p) for p in parts[:11]]
    except ValueError:
        return None
    return DiskIOSnapshot(
        read_ios=nums[0], read_merges=nums[1],
        read_sectors=nums[2], read_time_ms=nums[3],
        write_ios=nums[4], write_merges=nums[5],
        write_sectors=nums[6], write_time_ms=nums[7],
        in_flight=nums[8], io_time_ms=nums[9],
        weighted_ms=nums[10],
        timestamp=datetime.now().isoformat())


# ════════════════════════════════════════════════════════════════════════════
# DRIVE MONITOR
# ════════════════════════════════════════════════════════════════════════════

class DriveMonitor:

    def __init__(self, log: Callable[..., Any] = print):
        self._log = log
        self._db: Optional[sqlite3.Connection] = None
        self._session_id: Optional[int] = None
        self._drive_id: Optional[int] = None
        self._drive_info: Optional[DriveInfo] = None
        self._smart_start: Optional[dict] = None
        self._io_start: Optional[DiskIOSnapshot] = None
        self._active: bool = False
        self._lock = threading.Lock()
        self._stats = ThroughputStats()
        self._read_count = 0
        self._read_buffer: List[ReadEvent] = []
        self._flag_counts = {"slow_seek": 0, "retry": 0, "timeout": 0}

    # ────────────────────────────────────────────────────────────────────
    # CLASSIFICATION
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def classify_drive(path: str) -> DriveInfo:
        info = DriveInfo()
        try:
            path = os.path.realpath(path)
            device = DriveMonitor._resolve_device(path)
            if not device:
                return info
            info.device_path = device

            parent = DriveMonitor._resolve_parent(device)
            info.parent_device = parent or device
            base = os.path.basename(info.parent_device)

            if base.startswith("nvme"):
                info.drive_type = "NVMe"
                info.rotational = False
                DriveMonitor._fill_identity(info)
                return info

            rot = _read_sysfs(f"/sys/block/{base}/queue/rotational")
            if rot is not None:
                info.rotational = rot == "1"
                info.drive_type = "HDD" if info.rotational else "SSD"

            DriveMonitor._fill_identity(info)

            fs = _run(["lsblk", "-no", "FSTYPE", device])
            if fs:
                info.filesystem = fs.split("\n")[0].strip()

            mp = _run(["findmnt", "-n", "-o", "TARGET", "--source", device])
            if mp:
                info.mount_point = mp.split("\n")[0].strip()

            info.has_diskstats = _capture_diskstats(info.parent_device) is not None

        except Exception:
            info.drive_type = "unknown"
        return info

    @staticmethod
    def _resolve_device(path: str) -> Optional[str]:
        out = _run(["findmnt", "-n", "-o", "SOURCE", "--target", path])
        if out:
            dev = out.split("[")[0].strip()
            if dev.startswith("/dev/"):
                return dev
        try:
            best_dev, best_mnt = "", ""
            with open("/proc/mounts") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    dev = parts[0]
                    mnt = _unescape(parts[1])
                    if not dev.startswith("/dev/"):
                        continue
                    if path == mnt or path.startswith(mnt + "/"):
                        if len(mnt) > len(best_mnt):
                            best_dev, best_mnt = dev, mnt
            if best_dev:
                return best_dev
        except Exception:
            pass
        return None

    @staticmethod
    def _resolve_parent(device: str) -> Optional[str]:
        """Resolve partition/LVM/DM to physical whole-disk device."""
        if not device:
            return None

        visited = set()
        current = device

        for _ in range(MAX_DM_DEPTH):
            real = os.path.realpath(current)
            if real in visited:
                break
            visited.add(real)
            base = os.path.basename(real)

            # if /sys/block/<base>/queue exists, it IS a block device
            if os.path.isdir(f"/sys/block/{base}"):
                # check if it has a rotational file — means it's a real device
                if os.path.isfile(f"/sys/block/{base}/queue/rotational"):
                    return real

            # lsblk PKNAME
            out = _run(["lsblk", "-no", "PKNAME", current])
            if out:
                pk = out.strip().splitlines()[0].strip()
                if pk and pk != base:
                    current = f"/dev/{pk}"
                    continue

            # sysfs slaves (DM / LVM / MD)
            slaves_dir = f"/sys/block/{base}/slaves"
            if os.path.isdir(slaves_dir):
                children = os.listdir(slaves_dir)
                if children:
                    current = f"/dev/{children[0]}"
                    continue

            # heuristic strip: sda1→sda, nvme0n1p2→nvme0n1
            if "nvme" in base:
                m = re.match(r"^(nvme\d+n\d+)p\d+$", base)
                if m:
                    c = f"/dev/{m.group(1)}"
                    if os.path.exists(c):
                        return c
            else:
                m = re.match(r"^((?:sd|hd|vd)[a-z]+)\d+$", base)
                if m:
                    c = f"/dev/{m.group(1)}"
                    if os.path.exists(c):
                        return c
            break

        return current

    @staticmethod
    def _fill_identity(info: DriveInfo):
        """Serial, model, capacity via udevadm → lsblk → sysfs fallback."""
        dev = info.parent_device
        if not dev:
            return

        # ── udevadm first (best for USB-attached drives) ──
        udev_out = _run(["udevadm", "info", "--query=property",
                         "--name=" + dev])
        if udev_out:
            props = {}
            for line in udev_out.splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    props[k.strip()] = v.strip()
            # ID_SERIAL_SHORT is the actual drive serial
            # ID_SERIAL is often "VENDOR_MODEL_SERIAL" combined
            if not info.serial_number:
                info.serial_number = (props.get("ID_SERIAL_SHORT")
                                      or props.get("ID_SERIAL", ""))
            if not info.model_name:
                info.model_name = props.get("ID_MODEL", "")
            if not info.firmware:
                info.firmware = props.get("ID_REVISION", "")

        # ── lsblk fallback ──
        if not info.model_name:
            out = _run(["lsblk", "-dno", "MODEL", dev])
            if out:
                info.model_name = out.strip()
        if not info.serial_number:
            out = _run(["lsblk", "-dno", "SERIAL", dev])
            if out:
                info.serial_number = out.strip()

        # ── capacity ──
        out = _run(["lsblk", "-dno", "SIZE", "--bytes", dev])
        if out:
            try:
                info.capacity_bytes = int(out.strip())
            except ValueError:
                pass

        # ── firmware from sysfs ──
        if not info.firmware:
            base = os.path.basename(os.path.realpath(dev))
            for fw_file in [f"/sys/block/{base}/device/firmware_rev",
                            f"/sys/block/{base}/device/rev"]:
                val = _read_sysfs(fw_file)
                if val:
                    info.firmware = val
                    break

    # ────────────────────────────────────────────────────────────────────
    # SMART
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _try_smartctl(device: str) -> Tuple[Optional[dict], str]:
        if _run(["smartctl", "--version"]) is None:
            return None, "unavailable"

        for sudo in (False, True):
            cmd = (["sudo", "-n"] if sudo else []) + [
                "smartctl", "-j", "-a", device]
            try:
                r = subprocess.run(cmd, capture_output=True,
                                   text=True, timeout=15)
                if not r.stdout.strip():
                    continue
                data = json.loads(r.stdout)
                msgs = data.get("smartctl", {}).get("messages", [])
                perm_denied = any(
                    "permission denied" in m.get("string", "").lower()
                    for m in msgs)
                exit_status = data.get("smartctl", {}).get("exit_status", -1)

                if perm_denied:
                    if sudo:
                        return data, "permission_denied"
                    continue
                if exit_status in (0, 4):
                    return data, "full"
                if "ata_smart_attributes" in json.dumps(data):
                    return data, "full"
                return data, "permission_denied"
            except (json.JSONDecodeError, FileNotFoundError,
                    subprocess.TimeoutExpired, OSError):
                continue
        return None, "unavailable"

    @staticmethod
    def extract_watched_attrs(smart_json: dict) -> Dict[int, dict]:
        out = {}
        for entry in smart_json.get(
                "ata_smart_attributes", {}).get("table", []):
            aid = entry.get("id")
            if aid in WATCHED_SMART_ATTRS:
                out[aid] = {
                    "id":         aid,
                    "name":       entry.get("name",
                                            WATCHED_SMART_ATTRS[aid]),
                    "value":      entry.get("value"),
                    "worst":      entry.get("worst"),
                    "thresh":     entry.get("thresh"),
                    "raw_value":  entry.get("raw", {}).get("value", 0),
                    "raw_string": entry.get("raw", {}).get("string", ""),
                }
        return out

    @staticmethod
    def diff_smart(before: dict, after: dict) -> List[dict]:
        if not before or not after:
            return []
        ab = DriveMonitor.extract_watched_attrs(before)
        aa = DriveMonitor.extract_watched_attrs(after)
        if not ab and not aa:
            return []
        rows = []
        for aid in sorted(set(ab) | set(aa)):
            b, a = ab.get(aid, {}), aa.get(aid, {})
            rb = b.get("raw_value", 0) or 0
            ra = a.get("raw_value", 0) or 0
            rows.append({
                "attribute_id":   aid,
                "attribute_name": (a or b).get("name",
                                               WATCHED_SMART_ATTRS.get(aid, "?")),
                "value_start":    b.get("value", 0) or 0,
                "value_end":      a.get("value", 0) or 0,
                "raw_start":      rb,
                "raw_end":        ra,
                "delta":          ra - rb,
            })
        return rows

    # ────────────────────────────────────────────────────────────────────
    # TABLE MIGRATION
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _migrate(conn: sqlite3.Connection):
        """Add missing columns to existing tables without dropping data."""
        for table, col, coltype in _MIGRATIONS:
            try:
                existing = {row[1] for row in
                            conn.execute(f"PRAGMA table_info({table})")}
                if col not in existing:
                    conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
            except sqlite3.OperationalError:
                pass
        conn.commit()

    def ensure_tables(self, db_conn: sqlite3.Connection):
        db_conn.executescript(SMART_DDL)
        db_conn.commit()
        self._migrate(db_conn)

    # ────────────────────────────────────────────────────────────────────
    # SESSION LIFECYCLE
    # ────────────────────────────────────────────────────────────────────

    def start_session(self, db_conn: sqlite3.Connection,
                      session_id: int,
                      source_path: str) -> Optional[int]:
        self._db = db_conn
        self._session_id = session_id
        self._read_buffer = []
        self._read_count = 0
        self._stats = ThroughputStats()
        self._flag_counts = {"slow_seek": 0, "retry": 0, "timeout": 0}

        self.ensure_tables(db_conn)

        info = self.classify_drive(source_path)
        self._drive_info = info

        # identity
        parts = [info.parent_device or info.device_path or "?",
                 f"→ {info.drive_type}"]
        if info.model_name:
            parts.append(f"({info.model_name})")
        if info.serial_number:
            parts.append(f"[S/N: {info.serial_number}]")
        if info.filesystem:
            parts.append(f"fs={info.filesystem}")
        self._log(f"  Drive: {' '.join(parts)}")

        if info.drive_type != "HDD":
            self._log(f"  Source is {info.drive_type} — monitoring skipped")
            self._active = False
            return None

        cap_gb = info.capacity_bytes / (1024 ** 3) if info.capacity_bytes else 0
        self._log(f"  Capacity: {cap_gb:.0f} GB")

        # kernel I/O baseline
        self._io_start = _capture_diskstats(info.parent_device)
        if self._io_start:
            info.has_diskstats = True
            self._log(f"  Kernel I/O stats: available (/sys/block/"
                      f"{os.path.basename(info.parent_device)}/stat)")
        else:
            self._log(f"  Kernel I/O stats: not available")

        # SMART
        smart_data, smart_mode = self._try_smartctl(info.parent_device)
        self._smart_start = smart_data
        info.smart_mode = smart_mode

        if smart_mode == "full":
            info.smart_accessible = True
            passed = smart_data.get("smart_status", {}).get("passed")
            tag = ("PASSED" if passed
                   else ("FAILED ⚠️" if passed is False else "unknown"))
            self._log(f"  SMART health: {tag}")
            attrs = self.extract_watched_attrs(smart_data)
            for aid in (5, 197, 198):
                a = attrs.get(aid)
                if a and (a["raw_value"] or 0) > 0:
                    self._log(f"    ⚠️  {a['name']}: {a['raw_value']}")
        elif smart_mode == "permission_denied":
            self._log("  SMART: permission denied (typical for NTFS/udisks2)")
            self._log("    sudo smartctl -a "
                      f"{info.parent_device}  ← for full report")
        else:
            self._log("  SMART: unavailable (smartmontools not installed)")

        tier = ("SMART + kernel I/O + timing" if smart_mode == "full"
                and info.has_diskstats
                else "kernel I/O + timing" if info.has_diskstats
                else "timing only")
        self._log(f"  Monitoring tier: {tier}")
        self._log(f"  Warmup: {WARMUP_READS} reads "
                  f"(calibration, no flagging)")
        self._log(f"  EMA starts from first actual read (no preset bias)")

        # DB row
        now = datetime.now().isoformat()
        smart_blob = json.dumps(smart_data) if smart_data else None
        io_blob = (json.dumps(self._io_start.to_dict())
                   if self._io_start else None)

        with self._lock:
            cur = db_conn.execute(
                "INSERT INTO drive_info"
                "(session_id, device_path, serial_number, model_name,"
                " firmware, drive_type, filesystem, capacity_bytes,"
                " smart_mode, smart_start, kernel_io_start, scan_start_ts)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (session_id, info.parent_device, info.serial_number,
                 info.model_name, info.firmware, info.drive_type,
                 info.filesystem, info.capacity_bytes,
                 smart_mode, smart_blob, io_blob, now))
            db_conn.commit()
            self._drive_id = cur.lastrowid

        self._active = True
        return self._drive_id

    def end_session(self):
        if not self._active:
            return

        self._flush_read_buffer()

        # kernel I/O end snapshot + delta
        io_end = None
        io_delta = None
        if self._io_start and self._drive_info:
            io_end = _capture_diskstats(self._drive_info.parent_device)
            if io_end:
                io_delta = io_end.delta(self._io_start)
                d = io_delta
                kernel_mb = d.read_bytes / (1024 * 1024)
                kernel_mbps = (d.read_bytes / d.read_time_ms / 1000
                               if d.read_time_ms > 0 else 0)
                app_mbps = self._stats.avg_mbps
                overhead = ((self._stats.total_ms - d.read_time_ms)
                            if d.read_time_ms > 0 else 0)
                self._log(f"  ── Kernel I/O delta ──")
                self._log(f"    Kernel reads:      {d.read_ios} ops, "
                          f"{kernel_mb:.0f} MB")
                self._log(f"    Kernel read time:  {d.read_time_ms} ms")
                self._log(f"    Kernel throughput: {kernel_mbps:.1f} MB/s")
                self._log(f"    App wall time:     {self._stats.total_ms:.0f} ms")
                self._log(f"    App throughput:    {app_mbps:.1f} MB/s")
                if overhead > 0:
                    self._log(f"    Overhead (app−kernel): "
                              f"{overhead:.0f} ms "
                              f"({overhead / self._stats.total_ms * 100:.1f}%)")

        # final SMART
        smart_end = None
        if (self._drive_info and self._drive_info.smart_accessible
                and self._drive_info.smart_mode == "full"):
            smart_end_data, _ = self._try_smartctl(
                self._drive_info.parent_device)
            smart_end = smart_end_data

        # persist
        now = datetime.now().isoformat()
        end_blob = json.dumps(smart_end) if smart_end else None
        tp_blob = json.dumps(self._stats.summary_dict())
        io_end_blob = json.dumps(io_end.to_dict()) if io_end else None

        if self._db and self._drive_id:
            with self._lock:
                self._db.execute(
                    "UPDATE drive_info SET smart_end=?, throughput_json=?, "
                    "kernel_io_end=?, scan_end_ts=? WHERE drive_id=?",
                    (end_blob, tp_blob, io_end_blob, now, self._drive_id))
                self._db.commit()

        # SMART deltas
        if self._smart_start and smart_end:
            deltas = self.diff_smart(self._smart_start, smart_end)
            changed = [d for d in deltas if d["delta"] != 0]
            if changed:
                self._log("  SMART deltas:")
                for d in changed:
                    self._log(f"    {d['attribute_name']}: "
                              f"{d['raw_start']} → {d['raw_end']} "
                              f"(Δ{d['delta']:+d})")
            else:
                self._log("  No SMART changes during session")
            if self._db and self._drive_id:
                with self._lock:
                    for d in deltas:
                        self._db.execute(
                            "INSERT INTO drive_smart_deltas"
                            "(drive_id, attribute_id, attribute_name,"
                            " value_start, value_end, raw_start,"
                            " raw_end, delta)"
                            " VALUES(?,?,?,?,?,?,?,?)",
                            (self._drive_id, d["attribute_id"],
                             d["attribute_name"],
                             d["value_start"], d["value_end"],
                             d["raw_start"], d["raw_end"], d["delta"]))
                    self._db.commit()

        # throughput summary
        s = self._stats
        self._log(f"  ── App-level throughput ──")
        self._log(f"    Reads:     {s.count}")
        self._log(f"    Total:     {s.total_bytes / (1024**2):.0f} MB "
                  f"in {s.total_ms / 1000:.1f} s")
        self._log(f"    Average:   {s.avg_mbps:.1f} MB/s")
        self._log(f"    Smoothed:  {s.ema_mbps:.1f} MB/s (EMA)")
        self._log(f"    Range:     {s.min_mbps:.1f} – {s.max_mbps:.1f} MB/s")
        self._log(f"    Std dev:   {s.stddev_mbps:.1f} MB/s")
        flagged = sum(self._flag_counts.values())
        self._log(f"    Flagged:   {flagged} "
                  f"(slow={self._flag_counts['slow_seek']} "
                  f"retry={self._flag_counts['retry']} "
                  f"timeout={self._flag_counts['timeout']})")

        self._active = False
        self._smart_start = None
        self._io_start = None

    # ────────────────────────────────────────────────────────────────────
    # PER-FILE READ LOGGING
    # ────────────────────────────────────────────────────────────────────

    def log_file_read(self, file_path: str, file_size_bytes: int,
                      read_start_ts: str, read_end_ts: str,
                      read_time_ms: float,
                      batch: bool = False):
        if not self._active:
            return

        self._read_count += 1
        is_warmup = self._read_count <= WARMUP_READS
        read_type = "batch" if batch else "single"

        # expected time from EMA (0 until first read calibrates)
        if self._stats.calibrated and not batch:
            expected_ms = (file_size_bytes / self._stats.ema_bpms
                           if self._stats.ema_bpms > 0 else 0)
        else:
            expected_ms = 0

        # kernel I/O snapshot delta for this read (single files only)
        kern_ms = None
        kern_sectors = None
        # per-read kernel stats are only meaningful for single file reads
        # batch reads overlap with exiftool processing so kernel delta
        # would include unrelated I/O — skip for batch

        # flag classification
        flag = None
        if not is_warmup and not batch and expected_ms > 0:
            if read_time_ms > TIMEOUT_ABS_MS:
                flag = "timeout"
            else:
                ratio = read_time_ms / expected_ms
                if ratio > RETRY_MULTIPLIER:
                    flag = "retry"
                elif ratio > SLOW_SEEK_MULTIPLIER:
                    flag = "slow_seek"

        # update throughput EMA (single clean reads only)
        if not batch:
            self._stats.update(file_size_bytes, read_time_ms)

        if flag:
            self._flag_counts[flag] += 1

        # detail
        actual_mbps = (file_size_bytes / read_time_ms / 1000
                       if read_time_ms > 0 else 0)
        detail = None
        if flag or batch:
            detail = json.dumps({
                "ratio":       round(read_time_ms / expected_ms, 2)
                               if expected_ms > 0 else None,
                "actual_mbps": round(actual_mbps, 2),
                "ema_mbps":    round(self._stats.ema_mbps, 2),
                "read_num":    self._read_count,
                "warmup":      is_warmup,
                "batch":       batch,
            })

        if flag:
            bn = os.path.basename(file_path)
            self._log(
                f"    ⚠️  {flag.upper()}: {bn}  "
                f"({read_time_ms:.0f} ms, "
                f"expected {expected_ms:.0f} ms, "
                f"{actual_mbps:.1f} MB/s)")
        elif is_warmup and not batch and self._read_count % 4 == 0:
            self._log(
                f"    calibrating [{self._read_count}/{WARMUP_READS}]  "
                f"{actual_mbps:.1f} MB/s  "
                f"(EMA → {self._stats.ema_mbps:.1f})")

        self._read_buffer.append(ReadEvent(
            file_path=file_path, file_size_bytes=file_size_bytes,
            read_start_ts=read_start_ts, read_end_ts=read_end_ts,
            read_time_ms=read_time_ms, expected_ms=expected_ms,
            flag=flag, detail=detail, read_type=read_type,
            kernel_read_ms=kern_ms, kernel_sectors=kern_sectors))

        if len(self._read_buffer) >= READ_LOG_BATCH_SIZE:
            self._flush_read_buffer()

    def _flush_read_buffer(self):
        if not self._read_buffer or not self._db or not self._drive_id:
            return
        with self._lock:
            for ev in self._read_buffer:
                self._db.execute(
                    "INSERT INTO drive_read_log"
                    "(drive_id, session_id, file_path, file_size_bytes,"
                    " read_start_ts, read_end_ts, read_time_ms,"
                    " expected_ms, flag, detail, read_type,"
                    " kernel_read_ms, kernel_sectors)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (self._drive_id, self._session_id, ev.file_path,
                     ev.file_size_bytes, ev.read_start_ts, ev.read_end_ts,
                     ev.read_time_ms, ev.expected_ms, ev.flag, ev.detail,
                     ev.read_type, ev.kernel_read_ms, ev.kernel_sectors))
            self._db.commit()
        self._read_buffer = []

    # ────────────────────────────────────────────────────────────────────
    # TIMED READ WRAPPER
    # ────────────────────────────────────────────────────────────────────

    def timed_read(self, func: Callable, file_path: str,
                   file_size_bytes: int,
                   batch: bool = False) -> Any:
        if not self._active:
            return func()

        t0 = time.monotonic()
        ts0 = datetime.now().isoformat()
        result = func()
        ts1 = datetime.now().isoformat()
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        self.log_file_read(file_path, file_size_bytes,
                           ts0, ts1, elapsed_ms, batch=batch)
        return result

    # ────────────────────────────────────────────────────────────────────
    # PROPERTIES
    # ────────────────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def drive_info(self) -> Optional[DriveInfo]:
        return self._drive_info

    @property
    def avg_throughput_mbps(self) -> float:
        return self._stats.avg_mbps

    @property
    def throughput_stats(self) -> ThroughputStats:
        return self._stats


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _cli():
    import argparse
    p = argparse.ArgumentParser(
        description="Drive classifier + SMART + kernel I/O + benchmark")
    p.add_argument("path", help="Folder or mount point")
    p.add_argument("--smart", action="store_true",
                   help="Attempt SMART read (may need sudo)")
    p.add_argument("--benchmark", action="store_true",
                   help="Time-read files and report throughput")
    p.add_argument("--json", action="store_true",
                   help="JSON output")
    p.add_argument("--count", type=int, default=50,
                   help="Files to benchmark (default 50)")
    args = p.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: {args.path} not found", file=sys.stderr)
        sys.exit(1)

    mon = DriveMonitor()
    info = DriveMonitor.classify_drive(args.path)

    if args.json:
        blob = {
            "device":     info.device_path,
            "parent":     info.parent_device,
            "type":       info.drive_type,
            "model":      info.model_name,
            "serial":     info.serial_number,
            "firmware":   info.firmware,
            "filesystem": info.filesystem,
            "mount":      info.mount_point,
            "capacity":   info.capacity_bytes,
            "rotational": info.rotational,
            "diskstats":  info.has_diskstats,
        }
        if args.smart and info.drive_type == "HDD":
            data, mode = mon._try_smartctl(info.parent_device)
            blob["smart_mode"] = mode
            if data and mode == "full":
                blob["smart_passed"] = data.get(
                    "smart_status", {}).get("passed")
                blob["smart_attrs"] = mon.extract_watched_attrs(data)
            elif data:
                msgs = data.get("smartctl", {}).get("messages", [])
                blob["smart_error"] = (msgs[0].get("string", "")
                                       if msgs else "")
        if args.benchmark and info.has_diskstats:
            snap = _capture_diskstats(info.parent_device)
            if snap:
                blob["diskstats_snapshot"] = snap.to_dict()
        print(json.dumps(blob, indent=2))
        return

    # human
    cap_gb = info.capacity_bytes / (1024 ** 3) if info.capacity_bytes else 0
    print(f"Path:        {args.path}")
    print(f"Device:      {info.device_path}")
    print(f"Parent:      {info.parent_device}")
    print(f"Type:        {info.drive_type}")
    print(f"Model:       {info.model_name or '—'}")
    print(f"Serial:      {info.serial_number or '—'}")
    print(f"Firmware:    {info.firmware or '—'}")
    print(f"Filesystem:  {info.filesystem or '—'}")
    print(f"Mount:       {info.mount_point or '—'}")
    print(f"Capacity:    {cap_gb:.1f} GB")
    print(f"Rotational:  {info.rotational}")
    print(f"Kernel stat: {'yes' if info.has_diskstats else 'no'}")

    if info.has_diskstats:
        snap = _capture_diskstats(info.parent_device)
        if snap:
            print(f"\nKernel I/O (cumulative since boot):")
            print(f"  Read I/Os:  {snap.read_ios}")
            print(f"  Read data:  "
                  f"{snap.read_bytes / (1024**3):.1f} GB")
            print(f"  Read time:  {snap.read_time_ms / 1000:.1f} s")
            if snap.read_time_ms > 0:
                print(f"  Read avg:   "
                      f"{snap.read_bytes / snap.read_time_ms / 1000:.1f}"
                      f" MB/s")

    if args.smart:
        print()
        if info.drive_type != "HDD":
            print(f"SMART skipped — {info.drive_type}")
        else:
            data, mode = mon._try_smartctl(info.parent_device)
            if mode == "full":
                passed = data.get("smart_status", {}).get("passed")
                tag = ("PASSED" if passed
                       else ("FAILED ⚠️" if passed is False else "?"))
                print(f"SMART health: {tag}")
                attrs = mon.extract_watched_attrs(data)
                if attrs:
                    print(f"\n {'ID':>3}  {'Attribute':<30}  "
                          f"{'Val':>5}  {'Raw':>12}")
                    print(f" {'─'*3}  {'─'*30}  {'─'*5}  {'─'*12}")
                    for aid in sorted(attrs):
                        a = attrs[aid]
                        raw = a["raw_value"] or 0
                        w = (" ⚠️" if aid in (5, 197, 198)
                             and raw > 0 else "")
                        print(f" {aid:>3}  {a['name']:<30}  "
                              f"{a.get('value',''):>5}  {raw:>12}{w}")
            elif mode == "permission_denied":
                msgs = (data.get("smartctl", {}).get("messages", [])
                        if data else [])
                err = (msgs[0].get("string", "permission denied")
                       if msgs else "permission denied")
                print(f"SMART: {err}")
                print(f"  sudo smartctl -a {info.parent_device}")
            else:
                print("SMART: unavailable (install smartmontools)")

    if args.benchmark:
        print()
        if not os.path.isdir(args.path):
            print("--benchmark needs a directory")
            sys.exit(1)

        files = []
        for root, _dirs, names in os.walk(args.path):
            for n in sorted(names):
                fp = os.path.join(root, n)
                try:
                    sz = os.path.getsize(fp)
                except OSError:
                    continue
                if sz > 100_000:
                    files.append((fp, sz))
                if len(files) >= args.count:
                    break
            if len(files) >= args.count:
                break

        if not files:
            print("No files > 100 KB")
            return

        total_bytes = sum(sz for _, sz in files)
        print(f"Benchmarking {len(files)} files "
              f"({total_bytes / (1024**2):.0f} MB)...")

        # kernel snapshot before
        io_before = (_capture_diskstats(info.parent_device)
                     if info.has_diskstats else None)

        stats = ThroughputStats()
        for i, (fp, sz) in enumerate(files):
            t0 = time.monotonic()
            try:
                with open(fp, "rb") as fh:
                    _ = fh.read()
            except (IOError, PermissionError) as e:
                print(f"  skip: {os.path.basename(fp)}: {e}")
                continue
            ms = (time.monotonic() - t0) * 1000.0
            stats.update(sz, ms)
            if (i + 1) <= WARMUP_READS and (i + 1) % 4 == 0:
                mbps = sz / ms / 1000 if ms > 0 else 0
                print(f"  calibrating [{i+1}/{WARMUP_READS}]  "
                      f"{mbps:.1f} MB/s  (EMA → {stats.ema_mbps:.1f})")

        # kernel snapshot after
        io_after = (_capture_diskstats(info.parent_device)
                    if info.has_diskstats else None)

        print(f"\n── App-level results ──")
        print(f"  Reads:     {stats.count}")
        print(f"  Total:     {stats.total_bytes / (1024**2):.0f} MB "
              f"in {stats.total_ms / 1000:.1f} s")
        print(f"  Average:   {stats.avg_mbps:.1f} MB/s")
        print(f"  Smoothed:  {stats.ema_mbps:.1f} MB/s (EMA)")
        print(f"  Range:     {stats.min_mbps:.1f} – "
              f"{stats.max_mbps:.1f} MB/s")
        print(f"  Std dev:   {stats.stddev_mbps:.1f} MB/s")

        if io_before and io_after:
            d = io_after.delta(io_before)
            k_mb = d.read_bytes / (1024 * 1024)
            k_mbps = (d.read_bytes / d.read_time_ms / 1000
                      if d.read_time_ms > 0 else 0)
            overhead_ms = stats.total_ms - d.read_time_ms
            print(f"\n── Kernel-level ──")
            print(f"  Kernel reads:  {d.read_ios} ops, {k_mb:.0f} MB")
            print(f"  Kernel time:   {d.read_time_ms} ms")
            print(f"  Kernel speed:  {k_mbps:.1f} MB/s")
            print(f"  App overhead:  {overhead_ms:.0f} ms "
                  f"({overhead_ms / stats.total_ms * 100:.1f}%)"
                  if stats.total_ms > 0 else "")


if __name__ == "__main__":
    _cli()