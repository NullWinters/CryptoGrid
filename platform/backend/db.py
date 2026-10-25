"""SQLite 访问层：建表与读写助手。

库文件位于 platform/data/strategy.db，运行时目录不纳入版本控制。
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DB_DIR / "strategy.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS strategy (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    kind           TEXT    NOT NULL CHECK (kind IN ('buy_hold')),
    env            TEXT    NOT NULL CHECK (env IN ('mainnet','testnet')),
    benchmark      TEXT    NOT NULL CHECK (benchmark IN ('btc','eth','paxg','earn')),
    status         TEXT    NOT NULL CHECK (status IN ('running','closed','open_failed','mismatch')),
    account        TEXT    NOT NULL,
    total_invested REAL    NOT NULL CHECK (total_invested > 0),
    bench_start_px REAL,
    created_at     INTEGER NOT NULL,
    closed_at      INTEGER,
    note           TEXT
);

CREATE TABLE IF NOT EXISTS position (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id    INTEGER NOT NULL REFERENCES strategy(id),
    coin           TEXT    NOT NULL,
    amount_usdc    REAL    NOT NULL CHECK (amount_usdc > 0),
    leverage       INTEGER NOT NULL CHECK (leverage >= 1),
    notional       REAL,
    sz             REAL,
    entry_px       REAL,
    margin_used    REAL,
    liquidation_px REAL,
    opened         INTEGER NOT NULL DEFAULT 0,
    UNIQUE (strategy_id, coin)
);

CREATE TABLE IF NOT EXISTS fill (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL REFERENCES strategy(id),
    coin        TEXT    NOT NULL,
    side        TEXT    NOT NULL CHECK (side IN ('buy','sell')),
    px          REAL    NOT NULL,
    sz          REAL    NOT NULL,
    fee         REAL    NOT NULL DEFAULT 0,
    cloid       TEXT    UNIQUE,
    time        INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS equity_point (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id     INTEGER NOT NULL REFERENCES strategy(id),
    time            INTEGER NOT NULL,
    equity          REAL    NOT NULL,
    benchmark_value REAL,
    backfilled      INTEGER NOT NULL DEFAULT 0,
    UNIQUE (strategy_id, time)
);

CREATE TABLE IF NOT EXISTS log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER,
    time        INTEGER NOT NULL,
    level       TEXT    NOT NULL,
    message     TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_position_strategy ON position(strategy_id, coin);
CREATE INDEX IF NOT EXISTS idx_fill_strategy     ON fill(strategy_id, coin);
CREATE INDEX IF NOT EXISTS idx_equity_strategy   ON equity_point(strategy_id, time);
"""

ACTIVE_STATUSES = ("running", "mismatch")


def now_ms() -> int:
    return int(time.time() * 1000)


@contextmanager
def connect():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
        # 旧库补列，重复执行无副作用
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(strategy)").fetchall()}
        if "bench_start_px" not in cols:
            conn.execute("ALTER TABLE strategy ADD COLUMN bench_start_px REAL")


# --- 策略 ---

def create_strategy(env: str, benchmark: str, account: str, total_invested: float,
                    bench_start_px: float | None = None,
                    kind: str = "buy_hold", status: str = "running") -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO strategy (kind, env, benchmark, status, account, total_invested,"
            " bench_start_px, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (kind, env, benchmark, status, account, total_invested, bench_start_px, now_ms()),
        )
        return int(cur.lastrowid)


def set_status(strategy_id: int, status: str, note: str | None = None,
               closed_at: int | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE strategy SET status=?, note=COALESCE(?, note), closed_at=COALESCE(?, closed_at)"
            " WHERE id=?",
            (status, note, closed_at, strategy_id),
        )


def get_strategy(strategy_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM strategy WHERE id=?", (strategy_id,)).fetchone()
        return dict(row) if row else None


def get_active_strategy() -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM strategy WHERE status IN ('running','mismatch') ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def has_active_strategy() -> bool:
    return get_active_strategy() is not None


# --- 持仓 ---

def create_position(strategy_id: int, coin: str, amount_usdc: float, leverage: int,
                    notional: float | None = None) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO position (strategy_id, coin, amount_usdc, leverage, notional)"
            " VALUES (?,?,?,?,?)",
            (strategy_id, coin, amount_usdc, leverage, notional),
        )
        return int(cur.lastrowid)


def mark_position_opened(strategy_id: int, coin: str, sz: float, entry_px: float,
                         margin_used: float | None = None,
                         liquidation_px: float | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE position SET sz=?, entry_px=?, opened=1,"
            " margin_used=COALESCE(?, margin_used),"
            " liquidation_px=COALESCE(?, liquidation_px)"
            " WHERE strategy_id=? AND coin=?",
            (sz, entry_px, margin_used, liquidation_px, strategy_id, coin),
        )


def update_position_mark(strategy_id: int, coin: str, margin_used: float | None,
                         liquidation_px: float | None) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE position SET margin_used=COALESCE(?, margin_used),"
            " liquidation_px=COALESCE(?, liquidation_px) WHERE strategy_id=? AND coin=?",
            (margin_used, liquidation_px, strategy_id, coin),
        )


def list_positions(strategy_id: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM position WHERE strategy_id=? ORDER BY id", (strategy_id,)
        ).fetchall()
        return [dict(r) for r in rows]


# --- 成交 ---

def add_fill(strategy_id: int, coin: str, side: str, px: float, sz: float,
             fee: float = 0.0, cloid: str | None = None, ts: int | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO fill (strategy_id, coin, side, px, sz, fee, cloid, time)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (strategy_id, coin, side, px, sz, fee, cloid, ts or now_ms()),
        )


def list_fills(strategy_id: int, limit: int = 500) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT coin, side, px, sz, fee, time FROM fill WHERE strategy_id=?"
            " ORDER BY time ASC, id ASC LIMIT ?",
            (strategy_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def set_last_fill_fee(strategy_id: int, coin: str, fee: float) -> None:
    """下单返回体不含手续费，操作结束后按标的回填最近一笔。"""
    with connect() as conn:
        row = conn.execute(
            "SELECT id FROM fill WHERE strategy_id=? AND coin=? ORDER BY id DESC LIMIT 1",
            (strategy_id, coin),
        ).fetchone()
        if row:
            conn.execute("UPDATE fill SET fee=? WHERE id=?", (fee, row["id"]))


def fill_by_cloid(cloid: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM fill WHERE cloid=?", (cloid,)).fetchone()
        return dict(row) if row else None


# --- 净值点 ---

def add_equity_point(strategy_id: int, ts: int, equity: float,
                     benchmark_value: float | None = None, backfilled: int = 0) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO equity_point"
            " (strategy_id, time, equity, benchmark_value, backfilled) VALUES (?,?,?,?,?)",
            (strategy_id, ts, equity, benchmark_value, backfilled),
        )


def last_equity_point(strategy_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM equity_point WHERE strategy_id=? ORDER BY time DESC LIMIT 1",
            (strategy_id,),
        ).fetchone()
        return dict(row) if row else None


def list_equity_points(strategy_id: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT time, equity, benchmark_value, backfilled FROM equity_point"
            " WHERE strategy_id=? ORDER BY time ASC",
            (strategy_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# --- 日志 ---

def add_log(level: str, message: str, strategy_id: int | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO log (strategy_id, time, level, message) VALUES (?,?,?,?)",
            (strategy_id, now_ms(), level, message),
        )


def list_logs(limit: int = 100) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT time, level, message FROM log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
