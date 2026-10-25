"""策略交易 REST 接口。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import db
import strategy_engine as engine
from trading import get_account

router = APIRouter(prefix="/api/strategy")


class PositionIn(BaseModel):
    coin: str
    amount_usdc: float
    leverage: int


class StartIn(BaseModel):
    env: str
    benchmark: str
    positions: list[PositionIn] = Field(default_factory=list)


class RetryIn(BaseModel):
    coin: str


class ResolveIn(BaseModel):
    action: str


def _fail(exc: Exception) -> HTTPException:
    if isinstance(exc, ValueError):
        arg = exc.args[0] if exc.args else str(exc)
        detail = arg.get("errors") if isinstance(arg, dict) else str(arg)
        return HTTPException(status_code=400, detail=detail)
    arg = exc.args[0] if exc.args else str(exc)
    detail = arg.get("errors") if isinstance(arg, dict) else str(arg)
    return HTTPException(status_code=502, detail=detail)


@router.get("/account")
def account(env: str):
    try:
        acct = get_account(env)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    try:
        withdrawable = acct.withdrawable()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"无法读取余额: {exc}")
    return {"account": acct.address, "withdrawable": withdrawable, "env": env}


@router.get("/current")
def current():
    return engine.current()


@router.get("/progress")
def progress():
    return engine.progress_snapshot()


@router.post("/start")
async def start(body: StartIn):
    try:
        return await engine.start(body.env, body.benchmark,
                                  [p.model_dump() for p in body.positions])
    except Exception as exc:
        raise _fail(exc)


@router.post("/retry")
async def retry(body: RetryIn):
    try:
        return await engine.retry(body.coin)
    except Exception as exc:
        raise _fail(exc)


@router.post("/stop")
async def stop():
    try:
        return await engine.close()
    except Exception as exc:
        raise _fail(exc)


@router.post("/resolve")
async def resolve(body: ResolveIn):
    try:
        return await engine.resolve(body.action)
    except Exception as exc:
        raise _fail(exc)


@router.get("/equity")
def equity():
    row = db.get_active_strategy()
    if not row:
        return {"points": [], "total_invested": 0.0, "max_drawdown": 0.0, "offline_ranges": []}
    points = [
        {"t": p["time"], "equity": p["equity"], "benchmark": p["benchmark_value"],
         "backfilled": p["backfilled"]}
        for p in db.list_equity_points(row["id"])
    ]

    peak = None
    max_dd = 0.0
    for p in points:
        eq = p["equity"]
        if peak is None or eq > peak:
            peak = eq
        if peak and peak > 0:
            max_dd = max(max_dd, (peak - eq) / peak)

    ranges: list[list[int]] = []
    run_start = None
    prev_t = None
    for p in points:
        if p["backfilled"]:
            if run_start is None:
                run_start = p["t"]
            prev_t = p["t"]
        elif run_start is not None:
            ranges.append([run_start, prev_t])
            run_start = None
    if run_start is not None:
        ranges.append([run_start, prev_t])

    return {"points": points, "total_invested": row["total_invested"],
            "max_drawdown": max_dd, "offline_ranges": ranges}


@router.get("/fills")
def fills(strategy_id: int | None = None):
    if strategy_id is None:
        row = db.get_active_strategy()
        if not row:
            return []
        strategy_id = row["id"]
    elif db.get_strategy(strategy_id) is None:
        raise HTTPException(status_code=404, detail="策略不存在")
    return db.list_fills(strategy_id)


@router.get("/logs")
def logs():
    return db.list_logs()
