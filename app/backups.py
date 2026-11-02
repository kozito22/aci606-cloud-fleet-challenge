"""Logical and physical backups with a fixed schedule.

- Logical backup: the rows of the `telemetry` table exported to a gzip CSV.
  It is independent of the engine (SQLite or PostgreSQL), human readable and
  can be restored into another database (e.g. moving local data to Neon).
- Physical backup: a binary page-level copy of the whole database file
  (SQLite online backup API, consistent while the API keeps writing). With
  PostgreSQL on Neon the physical layer is managed by the provider (WAL,
  point-in-time restore and branches), so the app does not copy files there.
- Off-site copy: every physical backup is also copied to BACKUP_OFFSITE_DIR
  (another disk, a synced folder...) so a disk failure does not take the
  database and its backup at the same time.

Schedule (environment variables, minutes):
    BACKUP_LOGICAL_MINUTES   default 30
    BACKUP_PHYSICAL_MINUTES  default 360 (6 h)
    BACKUP_KEEP              files kept per kind, default 10

CLI:
    python -m app.backups logical
    python -m app.backups physical
    python -m app.backups restore backups/logical/telemetry_YYYYmmdd_HHMMSS.csv.gz
"""
from __future__ import annotations

import csv
import gzip
import os
import shutil
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import insert, select

from app import db

BACKUP_DIR = Path(os.environ.get("BACKUP_DIR", "/tmp/backups" if os.environ.get("VERCEL") else db.ROOT / "backups"))
OFFSITE_DIR = os.environ.get("BACKUP_OFFSITE_DIR", "").strip()
LOGICAL_MINUTES = float(os.environ.get("BACKUP_LOGICAL_MINUTES", "30"))
PHYSICAL_MINUTES = float(os.environ.get("BACKUP_PHYSICAL_MINUTES", "360"))
KEEP = int(os.environ.get("BACKUP_KEEP", "10"))
# Serverless platforms (Vercel) have no persistent disk nor background threads:
# there the schedule runs outside the API (.github/workflows/backup.yml).
SCHEDULER_ENABLED = os.environ.get("BACKUPS_ENABLED", "0" if os.environ.get("VERCEL") else "1") == "1"
SCHEDULER = "API (hilo interno)" if SCHEDULER_ENABLED else os.environ.get("BACKUP_SCHEDULER", "GitHub Actions")

_run_lock = threading.Lock()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _rotate(folder: Path, pattern: str) -> None:
    files = sorted(folder.glob(pattern))
    for old in files[:-KEEP]:
        old.unlink(missing_ok=True)


def _log(kind: str, started: float, status: str, **fields) -> dict:
    row = {"kind": kind, "status": status, "started_at": started, "finished_at": time.time(), **fields}
    with db.engine.begin() as conn:
        conn.execute(insert(db.backup_log).values(**row))
    return row


def logical_backup() -> dict:
    db.init_db()
    started = time.time()
    folder = BACKUP_DIR / "logical"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"telemetry_{_stamp()}.csv.gz"
    columns = [c.name for c in db.telemetry.columns]
    rows = 0
    with gzip.open(target, "wt", newline="", encoding="utf-8") as fh, db.engine.connect() as conn:
        writer = csv.writer(fh)
        writer.writerow(columns)
        result = conn.execution_options(stream_results=True).execute(
            select(db.telemetry).order_by(db.telemetry.c.id))
        for row in result:
            writer.writerow(row)
            rows += 1
    _rotate(folder, "telemetry_*.csv.gz")
    return _log("logical", started, "OK", rows=rows, size_bytes=target.stat().st_size,
                location=str(target.relative_to(db.ROOT) if target.is_relative_to(db.ROOT) else target))


def physical_backup() -> dict:
    db.init_db()
    started = time.time()
    if not db.IS_SQLITE:
        return _log("physical", started, "MANAGED",
                    detail="PostgreSQL en Neon: respaldo fisico continuo del proveedor (WAL, "
                           "restauracion a un punto en el tiempo y ramas).")
    source_path = Path(db.engine.url.database)
    folder = BACKUP_DIR / "physical"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"fleet_{_stamp()}.db"
    # Online backup API: consistent page-by-page copy even while writing.
    with sqlite3.connect(source_path) as src, sqlite3.connect(target) as dst:
        src.backup(dst)
    _rotate(folder, "fleet_*.db")

    offsite = None
    if OFFSITE_DIR:
        offsite_folder = Path(OFFSITE_DIR)
        offsite_folder.mkdir(parents=True, exist_ok=True)
        offsite = str(shutil.copy2(target, offsite_folder / target.name))
        _rotate(offsite_folder, "fleet_*.db")
    return _log("physical", started, "OK", size_bytes=target.stat().st_size,
                location=str(target.relative_to(db.ROOT) if target.is_relative_to(db.ROOT) else target),
                offsite_copy=offsite)


def run(kind: str) -> dict:
    with _run_lock:
        try:
            return logical_backup() if kind == "logical" else physical_backup()
        except Exception as exc:
            return _log(kind, time.time(), "ERROR", detail=str(exc)[:500])


def _last(kind: str) -> dict | None:
    query = (select(db.backup_log).where(db.backup_log.c.kind == kind)
             .order_by(db.backup_log.c.id.desc()).limit(1))
    with db.engine.connect() as conn:
        row = conn.execute(query).first()
    return dict(row._mapping) if row else None


def status() -> dict:
    db.init_db()
    now = time.time()
    summary = {}
    for kind, minutes in (("logical", LOGICAL_MINUTES), ("physical", PHYSICAL_MINUTES)):
        last = _last(kind)
        next_due = (last["finished_at"] if last else now) + minutes * 60
        summary[kind] = {
            "every_minutes": minutes,
            "last": last,
            "next_due_at": next_due,
            "seconds_since_last": round(now - last["finished_at"], 1) if last else None,
        }
    with db.engine.connect() as conn:
        history = [dict(r._mapping) for r in conn.execute(
            select(db.backup_log).order_by(db.backup_log.c.id.desc()).limit(20))]
    return {
        "scheduler_enabled": SCHEDULER_ENABLED,
        "scheduler": SCHEDULER,
        "retention_files": KEEP,
        "backup_dir": str(BACKUP_DIR),
        "offsite_dir": OFFSITE_DIR or None,
        "database": db.backend_name(),
        **summary,
        "history": history,
    }


def _due(kind: str, minutes: float) -> bool:
    last = _last(kind)
    return last is None or time.time() - last["finished_at"] >= minutes * 60


def _scheduler(stop: threading.Event) -> None:
    while not stop.wait(30):
        for kind, minutes in (("logical", LOGICAL_MINUTES), ("physical", PHYSICAL_MINUTES)):
            try:
                if _due(kind, minutes):
                    run(kind)
            except Exception as exc:  # database asleep, etc.: retry next tick
                print(f"backup scheduler: {kind}: {exc}")


def start_scheduler() -> threading.Event | None:
    if not SCHEDULER_ENABLED:
        return None
    stop = threading.Event()
    threading.Thread(target=_scheduler, args=(stop,), daemon=True, name="backup-scheduler").start()
    return stop


def restore(path: Path) -> int:
    """Load a logical backup into the current database (works across engines,
    e.g. local SQLite -> Neon). Rows already present (same robot_id and
    timestamp) are skipped, and ids are assigned by the target database."""
    db.init_db()
    with db.engine.connect() as conn:
        existing = {(r[0], r[1]) for r in conn.execute(select(db.telemetry.c.robot_id, db.telemetry.c.timestamp))}
    inserted = 0
    batch = []
    float_cols = {c.name for c in db.telemetry.columns if c.name not in ("id", "robot_id", "predicted_status")}
    with gzip.open(path, "rt", encoding="utf-8") as fh, db.engine.begin() as conn:
        for row in csv.DictReader(fh):
            if (row["robot_id"], float(row["timestamp"])) in existing:
                continue
            record = {"robot_id": row["robot_id"],
                      "predicted_status": row.get("predicted_status") or None}
            for col in float_cols:
                record[col] = float(row[col]) if row.get(col) not in (None, "") else None
            batch.append(record)
            if len(batch) == 1000:
                conn.execute(insert(db.telemetry), batch)
                inserted += len(batch)
                batch = []
        if batch:
            conn.execute(insert(db.telemetry), batch)
            inserted += len(batch)
    return inserted


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command in ("logical", "physical"):
        print(run(command))
    elif command == "restore" and len(sys.argv) == 3:
        print(f"restored rows: {restore(Path(sys.argv[2]))}")
    else:
        raise SystemExit(__doc__)
