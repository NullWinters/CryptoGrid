"""看板后端：Hyperliquid 行情的薄代理与元数据透传。

启动（仓库根目录）：
    uv run uvicorn --app-dir platform/backend main:app --host 127.0.0.1 --port 8000
用 --app-dir 而非包路径，避免包名 platform 遮蔽 Python 标准库同名模块。
"""
from __future__ import annotations

import asyncio
import time

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from hyperliquid.info import Info

import db
import strategy_engine as engine
import strategy_api
from hl_proxy import Proxy

REST_URLS = {
    "mainnet": "https://api.hyperliquid.xyz",
    "testnet": "https://api.hyperliquid-testnet.xyz",
}

VALID_INTERVALS = {
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "8h", "12h",
    "1d", "3d", "1w", "1M",
}

app = FastAPI(title="Realtime Market Dashboard")
app.include_router(strategy_api.router)


@app.on_event("startup")
async def _startup() -> None:
    db.init_db()
    asyncio.create_task(engine.recover())

# Info 客户端按环境缓存，避免每次请求重建
_info: dict[str, Info] = {}


def _get_info(env: str) -> Info:
    if env not in REST_URLS:
        raise HTTPException(status_code=400, detail=f"unknown env {env}")
    if env not in _info:
        _info[env] = Info(REST_URLS[env], skip_ws=True)
    return _info[env]


@app.get("/api/instruments")
def instruments(env: str):
    info = _get_info(env)
    universe = info.meta()["universe"]
    return [{"name": c["name"], "szDecimals": c["szDecimals"]} for c in universe]


@app.get("/api/candles")
def candles(env: str, coin: str, interval: str):
    if interval not in VALID_INTERVALS:
        raise HTTPException(status_code=400, detail=f"unknown interval {interval}")
    info = _get_info(env)
    end = int(time.time() * 1000)
    # 取足够长的区间后裁剪末尾 200 根，避免长周期按天估算起点导致根数偏差。
    # 以最长周期 1M 近似 31 天计，300 根的跨度足以覆盖任意周期的最近 200 根。
    start = end - 300 * 31 * 24 * 60 * 60 * 1000
    rows = info.candles_snapshot(coin, interval, start, end)
    return rows[-200:]


@app.websocket("/api/stream")
async def stream(ws: WebSocket):
    await ws.accept()

    async def send(msg: dict) -> None:
        await ws.send_json(msg)

    proxy = Proxy(send)
    strategy_task: asyncio.Task | None = None

    async def push_strategy() -> None:
        while True:
            try:
                await ws.send_json({"channel": "strategy", "data": engine.current()})
            except Exception:
                return
            await asyncio.sleep(3)

    try:
        while True:
            req = await ws.receive_json()
            action = req.get("action")
            if action == "subscribe":
                await proxy.subscribe(req["env"], req["coin"], req["interval"])
            elif action == "watch_strategy" and strategy_task is None:
                strategy_task = asyncio.create_task(push_strategy())
    except WebSocketDisconnect:
        pass
    finally:
        if strategy_task:
            strategy_task.cancel()
        await proxy.close()
