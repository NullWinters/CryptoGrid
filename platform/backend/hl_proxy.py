"""前端 WebSocket 与 Hyperliquid WebSocket 之间的薄代理。

每个浏览器连接对应一条到上游 Hyperliquid 的连接，只转发订阅与推送，不缓存行情状态。
"""
from __future__ import annotations

import asyncio
import json

import websockets

WS_URLS = {
    "mainnet": "wss://api.hyperliquid.xyz/ws",
    "testnet": "wss://api.hyperliquid-testnet.xyz/ws",
}

# 单个标的订阅的三个频道，candle 需带 interval
def _subscriptions(coin: str, interval: str) -> list[dict]:
    return [
        {"type": "candle", "coin": coin, "interval": interval},
        {"type": "l2Book", "coin": coin},
        {"type": "trades", "coin": coin},
    ]


# 上游频道名到回传给前端的 channel 名的映射
_CHANNEL_MAP = {"candle": "candle", "l2Book": "l2Book", "trades": "trades"}


class Proxy:
    """管理一条上游连接，按前端订阅转发消息。"""

    def __init__(self, send_to_client):
        self._send = send_to_client
        self._upstream: websockets.WebSocketClientProtocol | None = None
        self._pump_task: asyncio.Task | None = None
        self._current: tuple[str, str, str] | None = None  # (env, coin, interval)
        self._switching = False  # 主动切换环境导致的关闭，不应通报断线

    async def subscribe(self, env: str, coin: str, interval: str) -> None:
        if env not in WS_URLS:
            await self._send({"channel": "status", "data": {"state": "error", "message": f"unknown env {env}"}})
            return
        prev = self._current
        self._current = (env, coin, interval)
        # env 变化需重建上游连接；否则取消旧订阅再订新的
        if self._upstream is None or prev is None or prev[0] != env:
            await self._reconnect()
        else:
            await self._resubscribe(prev[1], prev[2], coin, interval)

    async def _reconnect(self) -> None:
        if self._upstream is not None:
            self._switching = True
        await self._close_upstream()
        if self._pump_task is None or self._pump_task.done():
            self._pump_task = asyncio.create_task(self._run())

    async def _resubscribe(self, old_coin: str, old_interval: str, coin: str, interval: str) -> None:
        if self._upstream is None:
            await self._reconnect()
            return
        try:
            for sub in _subscriptions(old_coin, old_interval):
                await self._upstream.send(json.dumps({"method": "unsubscribe", "subscription": sub}))
            for sub in _subscriptions(coin, interval):
                await self._upstream.send(json.dumps({"method": "subscribe", "subscription": sub}))
        except websockets.ConnectionClosed:
            await self._reconnect()

    async def _run(self) -> None:
        """维护上游连接，断线按退避重连，期间向前端通报状态。"""
        backoff = 1
        while self._current is not None:
            env, coin, interval = self._current
            url = WS_URLS[env]
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    self._upstream = ws
                    backoff = 1
                    for sub in _subscriptions(coin, interval):
                        await ws.send(json.dumps({"method": "subscribe", "subscription": sub}))
                    await self._send({"channel": "status", "data": {"state": "connected"}})
                    async for raw in ws:
                        await self._forward(raw)
            except (websockets.ConnectionClosed, OSError):
                pass
            finally:
                self._upstream = None
            if self._current is None:
                break
            if self._switching:
                self._switching = False
                continue
            await self._send({"channel": "status", "data": {"state": "disconnected"}})
            await self._send({"channel": "status", "data": {"state": "reconnecting"}})
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 8)

    async def _forward(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        channel = _CHANNEL_MAP.get(msg.get("channel"))
        if channel is None:
            return  # 忽略 subscriptionResponse、pong 等
        await self._send({"channel": channel, "data": msg.get("data")})

    async def _close_upstream(self) -> None:
        if self._upstream is not None:
            try:
                await self._upstream.close()
            except Exception:
                pass
            self._upstream = None

    async def close(self) -> None:
        self._current = None
        await self._close_upstream()
        if self._pump_task is not None:
            self._pump_task.cancel()
