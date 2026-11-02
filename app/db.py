"""Persistence layer (SQLAlchemy Core).

- In the cloud: PostgreSQL managed by Neon, read from the DATABASE_URL env var.
- Locally (no DATABASE_URL): SQLite file at data/fleet.db.

The same code and the same SQL work on both engines, so the API does not care
where the data lives. Credentials never appear in the code.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from sqlalchemy import (
    BigInteger,
    Column,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    cast,
    create_engine,
    event,
    func,
    inspect,
    select,
    text,
)

ROOT = Path(__file__).resolve().parents[1]

TELEMETRY_FIELDS = [
    "robot_id",
    "timestamp",
    "x",
    "y",
    "speed",
    "battery",
    "motor_temp",
    "vibration",
    "motor_current",
    "obstacle_distance",
]


def database_url() -> str:
    # DATABASE_URL is the standard name (Render, Neon). On Vercel the Neon
    # integration prefixes its variables (e.g. DATABASE_POSTGRES_URL with the
    # prefix "DATABASE", STORAGE_URL by default) and legacy Postgres used POSTGRES_URL.
    names = ("DATABASE_URL", "DATABASE_POSTGRES_URL", "POSTGRES_URL", "STORAGE_URL")
    url = next((os.environ[k] for k in names if os.environ.get(k)), "").strip()
    if not url:
        # Serverless file systems are read-only except /tmp (data there is
        # temporary: production must always define DATABASE_URL).
        folder = Path("/tmp") if os.environ.get("VERCEL") else ROOT / "data"
        folder.mkdir(exist_ok=True)
        return f"sqlite:///{(folder / 'fleet.db').as_posix()}"
    # Neon/Heroku style URLs -> SQLAlchemy driver for psycopg 3.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


URL = database_url()
IS_SQLITE = URL.startswith("sqlite")

if IS_SQLITE:
    engine = create_engine(URL, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")  # readers do not block the writer
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()
else:
    # pool_pre_ping: Neon suspends idle databases after ~5 min; the ping
    # detects dead connections and reconnects transparently.
    # prepare_threshold=None: Neon's pooled endpoint (PgBouncer, transaction
    # mode) does not keep server-side prepared statements between requests.
    engine = create_engine(URL, pool_pre_ping=True, pool_recycle=300, pool_size=5, max_overflow=5,
                           connect_args={"prepare_threshold": None})

# SQLite only autoincrements INTEGER PRIMARY KEY; PostgreSQL gets BIGSERIAL.
PK = BigInteger().with_variant(Integer, "sqlite")

metadata = MetaData()

telemetry = Table(
    "telemetry",
    metadata,
    Column("id", PK, primary_key=True, autoincrement=True),
    Column("robot_id", String(20), nullable=False),
    Column("timestamp", Float, nullable=False),
    Column("x", Float, nullable=False),
    Column("y", Float, nullable=False),
    Column("speed", Float, nullable=False),
    Column("battery", Float, nullable=False),
    Column("motor_temp", Float, nullable=False),
    Column("vibration", Float, nullable=False),
    Column("motor_current", Float, nullable=False),
    Column("obstacle_distance", Float, nullable=False),
    Column("received_at", Float, nullable=False),
    # Prediction computed by the API at ingestion time (dashboard/history).
    Column("predicted_status", String(10)),
    Column("confidence", Float),
    Index("ix_telemetry_robot_id_id", "robot_id", "id"),
)

# One row per backup run (logical/physical), used for scheduling and the dashboard.
backup_log = Table(
    "backup_log",
    metadata,
    Column("id", PK, primary_key=True, autoincrement=True),
    Column("kind", String(10), nullable=False),
    Column("status", String(10), nullable=False),
    Column("started_at", Float, nullable=False),
    Column("finished_at", Float),
    Column("rows", BigInteger),
    Column("size_bytes", BigInteger),
    Column("location", String(500)),
    Column("offsite_copy", String(500)),
    Column("detail", String(500)),
)

_init_lock = threading.Lock()
_initialized = False


def init_db() -> None:
    """Create tables once (idempotent). Called lazily so a cold start never
    fails just because the database was still waking up."""
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if not _initialized:
            metadata.create_all(engine)
            _add_missing_columns()
            _initialized = True


def _add_missing_columns() -> None:
    """Minimal migration: add new nullable columns to an existing table."""
    existing = {c["name"] for c in inspect(engine).get_columns("telemetry")}
    with engine.begin() as conn:
        for column in telemetry.columns:
            if column.name not in existing and column.nullable:
                ddl = column.type.compile(dialect=engine.dialect)
                conn.execute(text(f'ALTER TABLE telemetry ADD COLUMN "{column.name}" {ddl}'))


def _telemetry_dict(row) -> dict:
    return {field: row._mapping[field] for field in TELEMETRY_FIELDS}


def _with_prediction(row) -> dict:
    data = _telemetry_dict(row)
    data["predicted_status"] = row._mapping["predicted_status"]
    data["confidence"] = row._mapping["confidence"]
    data["received_at"] = row._mapping["received_at"]
    return data


def insert_telemetry(sample: dict, received_at: float) -> None:
    init_db()
    with engine.begin() as conn:
        conn.execute(telemetry.insert().values(**sample, received_at=received_at))


def latest_per_robot(with_prediction: bool = False) -> list[dict]:
    """Most recent sample of every robot (uses the (robot_id, id) index)."""
    init_db()
    last_ids = select(func.max(telemetry.c.id)).group_by(telemetry.c.robot_id)
    query = select(telemetry).where(telemetry.c.id.in_(last_ids)).order_by(telemetry.c.robot_id)
    convert = _with_prediction if with_prediction else _telemetry_dict
    with engine.connect() as conn:
        return [convert(r) for r in conn.execute(query)]


def robot_history(robot_id: str, limit: int | None = None, with_prediction: bool = False) -> list[dict] | None:
    """Complete history in chronological order (None if the robot is unknown).
    `limit` optionally returns only the last N samples."""
    init_db()
    convert = _with_prediction if with_prediction else _telemetry_dict
    query = select(telemetry).where(telemetry.c.robot_id == robot_id)
    if limit:
        query = query.order_by(telemetry.c.id.desc()).limit(limit)
    else:
        query = query.order_by(telemetry.c.id)
    with engine.connect() as conn:
        rows = [convert(r) for r in conn.execute(query)]
    if not rows:
        return None
    return rows[::-1] if limit else rows


def _floor(expr):
    """floor() that behaves the same on SQLite (truncating CAST, values >= 0)
    and PostgreSQL (FLOOR)."""
    return cast(expr, Integer) if IS_SQLITE else cast(func.floor(expr), Integer)


def trend(minutes: int = 15, now: float | None = None) -> list[dict]:
    """Samples per minute and predicted status for the last `minutes` minutes,
    aggregated in SQL so only a few rows travel from the database."""
    init_db()
    now = now or time.time()
    first = int(now // 60) - minutes + 1
    bucket = _floor(telemetry.c.received_at / 60).label("bucket")
    query = (select(bucket, telemetry.c.predicted_status, func.count())
             .where(telemetry.c.received_at >= first * 60)
             .group_by(bucket, telemetry.c.predicted_status))
    buckets = {first + i: {"minute": (first + i) * 60, "total": 0, "NORMAL": 0, "WARNING": 0, "CRITICAL": 0}
               for i in range(minutes)}
    with engine.connect() as conn:
        for b, status, n in conn.execute(query):
            if b in buckets:
                buckets[b]["total"] += n
                if status in ("NORMAL", "WARNING", "CRITICAL"):
                    buckets[b][status] += n
    return list(buckets.values())


def visited_cells(since: float, cell_m: float = 1.0) -> list[tuple[int, int, int]]:
    """(col, row, samples) of every grid cell visited since `since`."""
    init_db()
    col = _floor(telemetry.c.x / cell_m).label("col")
    row = _floor(telemetry.c.y / cell_m).label("row")
    query = select(col, row, func.count()).where(telemetry.c.received_at >= since).group_by(col, row)
    with engine.connect() as conn:
        return [(int(c), int(r), int(n)) for c, r, n in conn.execute(query)]


def counts() -> dict:
    init_db()
    with engine.connect() as conn:
        total = conn.execute(select(func.count()).select_from(telemetry)).scalar_one()
        robots = conn.execute(select(func.count(func.distinct(telemetry.c.robot_id)))).scalar_one()
    return {"telemetry_rows": total, "robots": robots}


def backend_name() -> str:
    return "SQLITE" if IS_SQLITE else "POSTGRESQL"


def ping() -> bool:
    try:
        init_db()
        with engine.connect() as conn:
            conn.execute(select(1))
        return True
    except Exception:
        return False
