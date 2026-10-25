"""Buy & Hold 策略引擎：建仓、净值采样、持续对账、离线补齐与恢复。"""
from __future__ import annotations

import asyncio
import json
import time

import websockets

import db
from trading import Account, get_account

MIN_NOTIONAL = 10.0
SAMPLE_INTERVAL_MS = 5_000
FUNDING_REFRESH_MS = 60_000
WS_URLS = {
    "mainnet": "wss://api.hyperliquid.xyz/ws",
    "testnet": "wss://api.hyperliquid-testnet.xyz/ws",
}
# 补齐时按此顺序挑周期，保证所需根数不超过数据源的 5000 根上限
BACKFILL_INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"]
INTERVAL_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000,
    "4h": 14_400_000, "1d": 86_400_000,
}

_last_earn_apy: float | None = None
_lock = asyncio.Lock()
_tasks: dict[int, asyncio.Task] = {}
_closing: set[int] = set()
_funding_cache: dict[int, float] = {}

# ---------------- 执行进度（仅存内存，不落库） ----------------

_progress: dict = {
    "op": None, "strategy_id": None, "env": None,
    "started_at": None, "finished": True, "items": [],
}


def progress_snapshot() -> dict:
    return {
        "op": _progress["op"],
        "strategy_id": _progress["strategy_id"],
        "env": _progress["env"],
        "started_at": _progress["started_at"],
        "finished": _progress["finished"],
        "items": [dict(i) for i in _progress["items"]],
    }


def _progress_reset(op: str, env: str, strategy_id: int | None = None) -> None:
    """每次操作开始时无条件重置，避免残留上一次的条目。"""
    _progress.update(op=op, strategy_id=strategy_id, env=env,
                     started_at=db.now_ms(), finished=False, items=[])


def _progress_set_items(pairs: list[tuple[str, str]]) -> None:
    _progress["items"] = [
        {"coin": c, "action": a, "status": "pending", "sz": None, "px": None, "error": None}
        for c, a in pairs
    ]


def _progress_item(coin: str, **fields) -> None:
    for it in _progress["items"]:
        if it["coin"] == coin:
            it.update(fields)
            return


def _progress_finish() -> None:
    _progress["finished"] = True


# ---------------- 公共视图 ----------------

def _public(row: dict) -> dict:
    strategy_id = row["id"]
    positions = db.list_positions(strategy_id)
    try:
        acct = get_account(row["env"])
        marks = acct.mids()
    except Exception:
        marks = {}

    out_positions = []
    for p in positions:
        mark = marks.get(p["coin"])
        pnl = None
        if p["opened"] and p["sz"] and p["entry_px"] and mark:
            pnl = (mark - p["entry_px"]) * p["sz"]
        out_positions.append({
            "coin": p["coin"],
            "amount_usdc": p["amount_usdc"],
            "leverage": p["leverage"],
            "notional": p["notional"],
            "sz": p["sz"],
            "entry_px": p["entry_px"],
            "opened": p["opened"],
            "mark_px": mark,
            "unrealized_pnl": pnl,
            "margin_used": p["margin_used"],
            "liquidation_px": p["liquidation_px"],
        })

    return {
        "id": strategy_id,
        "kind": row["kind"],
        "env": row["env"],
        "benchmark": row["benchmark"],
        "status": row["status"],
        "total_invested": row["total_invested"],
        "created_at": row["created_at"],
        "closed_at": row["closed_at"],
        "note": row["note"],
        "positions": out_positions,
    }


def current() -> dict | None:
    row = db.get_active_strategy()
    return _public(row) if row else None


# ---------------- 校验与建仓 ----------------

def validate(env: str, positions: list[dict]) -> tuple[Account | None, list[dict], list[str]]:
    errors: list[str] = []
    if not positions:
        return None, [], ["至少需要添加一个标的"]
    coins = [p.get("coin") for p in positions]
    if len(coins) != len(set(coins)):
        errors.append("同一标的不能重复添加")

    try:
        acct = get_account(env)
    except Exception as exc:  # 缺密钥或地址异常
        return None, [], [f"无法连接账户: {exc}"]

    try:
        meta = acct.meta()
        withdrawable = acct.withdrawable()
    except Exception as exc:
        return None, [], [f"无法读取账户信息: {exc}"]

    if db.has_active_strategy():
        errors.append("已有策略在运行，请先终止")
        return acct, [], errors

    total = 0.0
    clean: list[dict] = []
    for p in positions:
        coin = p.get("coin")
        amount = float(p.get("amount_usdc") or 0)
        leverage = int(p.get("leverage") or 0)
        if coin not in meta:
            errors.append(f"{coin}: 标的不在当前环境")
            continue
        if amount <= 0:
            errors.append(f"{coin}: 金额必须大于 0")
            continue
        max_lev = int(meta[coin].get("maxLeverage") or 1)
        if leverage < 1 or leverage > max_lev:
            errors.append(f"{coin}: 杠杆须在 1 到 {max_lev} 之间")
            continue
        notional = amount * leverage
        if notional < MIN_NOTIONAL:
            errors.append(f"{coin}: 金额过小，名义价值不足 {MIN_NOTIONAL:.0f} USDC，按当前杠杆至少需 {MIN_NOTIONAL / leverage:.2f} USDC")
            continue
        total += amount
        clean.append({"coin": coin, "amount_usdc": amount, "leverage": leverage,
                      "notional": notional})

    if total > withdrawable + 1e-9:
        errors.append(f"投入合计 {total:.2f} 超过可用余额 {withdrawable:.2f}")
    return acct, clean, errors


def _benchmark_start_price(acct: Account, benchmark: str) -> float | None:
    if benchmark == "earn":
        return None
    coin = {"btc": "BTC", "eth": "ETH", "paxg": "PAXG"}[benchmark]
    return acct.mids().get(coin)


async def start(env: str, benchmark: str, positions: list[dict]) -> dict:
    async with _lock:
        _progress_reset("start", env)
        # 先按请求里的标的占位，弹窗一出现就有行可看
        _progress_set_items([(p.get("coin"), "open") for p in positions if p.get("coin")])
        acct, clean, errors = await asyncio.to_thread(validate, env, positions)
        if errors:
            _progress_finish()
            raise ValueError({"errors": errors})
        assert acct is not None
        total = sum(p["amount_usdc"] for p in clean)
        bench_px = await asyncio.to_thread(_benchmark_start_price, acct, benchmark)
        sid = db.create_strategy(env, benchmark, acct.address, total, bench_px)
        for p in clean:
            db.create_position(sid, p["coin"], p["amount_usdc"], p["leverage"], p["notional"])

        _progress_reset("start", env, sid)
        _progress_set_items([(p["coin"], "open") for p in clean])

        failed: list[str] = []
        op_started = db.now_ms()
        marks = await asyncio.to_thread(acct.mids)
        try:
            for p in clean:
                coin = p["coin"]
                _progress_item(coin, status="submitting")
                try:
                    await asyncio.to_thread(acct.set_isolated_leverage, coin, p["leverage"])
                    res = await asyncio.to_thread(acct.open_market, coin, p["notional"],
                                                  marks.get(coin))
                except Exception as exc:
                    failed.append(f"{coin}: {exc}")
                    _progress_item(coin, status="failed", error=str(exc))
                    continue
                if res["ok"]:
                    db.mark_position_opened(sid, coin, res["total_sz"], res["avg_px"])
                    db.add_fill(sid, coin, "buy", res["avg_px"], res["total_sz"], 0.0, res["cloid"])
                    _progress_item(coin, status="filled", sz=res["total_sz"], px=res["avg_px"])
                else:
                    failed.append(f"{coin}: {res['error']}")
                    _progress_item(coin, status="failed", error=res["error"])
        finally:
            _progress_finish()

        # 操作结束后一次性补手续费与保证金信息，避免逐单多轮往返
        await asyncio.to_thread(_enrich_after_orders, acct, sid, op_started)

        opened = [p for p in db.list_positions(sid) if p["opened"]]
        if not opened:
            db.set_status(sid, "open_failed", note="; ".join(failed) or "全部标的建仓失败")
            db.add_log("error", f"策略 {sid} 全部建仓失败: {failed}", sid)
            raise RuntimeError({"errors": failed or ["全部标的建仓失败"]})

        if failed:
            note = "部分标的建仓失败: " + "; ".join(failed)
            db.set_status(sid, "running", note=note)
            db.add_log("warn", note, sid)
        _start_tasks(env, sid)
        return _public(db.get_strategy(sid))


def _enrich_after_orders(acct: Account, sid: int, since_ms: int) -> None:
    """一次性回填手续费与保证金占用，减少下单路径上的往返次数。"""
    try:
        for coin, fee in acct.recent_fees(since_ms).items():
            db.set_last_fill_fee(sid, coin, fee)
    except Exception:
        pass
    try:
        live = acct.positions()
        for p in db.list_positions(sid):
            info = live.get(p["coin"])
            if info:
                db.update_position_mark(sid, p["coin"], info.get("margin_used"),
                                        info.get("liquidation_px"))
    except Exception:
        pass


async def retry(coin: str) -> dict:
    async with _lock:
        row = db.get_active_strategy()
        if not row or row["status"] != "running":
            raise ValueError("当前没有运行中的策略")
        target = next((p for p in db.list_positions(row["id"]) if p["coin"] == coin), None)
        if target is None:
            raise ValueError(f"{coin} 不在本策略中")
        if target["opened"]:
            raise ValueError(f"{coin} 已建仓，无需重试")

        _progress_reset("retry", row["env"], row["id"])
        op_started = db.now_ms()
        _progress_set_items([(coin, "open")])
        _progress_item(coin, status="submitting")
        try:
            acct = get_account(row["env"])
            await asyncio.to_thread(acct.set_isolated_leverage, coin, target["leverage"])
            res = await asyncio.to_thread(acct.open_market, coin, target["notional"])
        except Exception as exc:
            _progress_item(coin, status="failed", error=str(exc))
            _progress_finish()
            db.add_log("error", f"{coin} 重试建仓失败: {exc}", row["id"])
            raise
        if not res["ok"]:
            _progress_item(coin, status="failed", error=res["error"])
            _progress_finish()
            db.add_log("error", f"{coin} 重试建仓失败: {res['error']}", row["id"])
            raise RuntimeError(res["error"])
        db.mark_position_opened(row["id"], coin, res["total_sz"], res["avg_px"])
        db.add_fill(row["id"], coin, "buy", res["avg_px"], res["total_sz"], 0.0, res["cloid"])
        db.add_log("info", f"{coin} 重试建仓成功", row["id"])
        _progress_item(coin, status="filled", sz=res["total_sz"], px=res["avg_px"])
        _progress_finish()
        await asyncio.to_thread(_enrich_after_orders, acct, row["id"], op_started)
        return _public(db.get_strategy(row["id"]))


async def close() -> dict:
    async with _lock:
        row = db.get_active_strategy()
        if not row or row["status"] != "running":
            raise ValueError("当前没有运行中的策略")
        return await _close_row(row)


async def _close_row(row: dict) -> dict:
    sid = row["id"]
    acct = get_account(row["env"])
    failed: list[str] = []
    _progress_reset("stop", row["env"], sid)
    op_started = db.now_ms()
    _closing.add(sid)
    # 保护标记一直保留到写状态结束，否则回填期间的对账推送会把策略误判为不一致
    try:
        # 先用库中已建仓的标的占位，弹窗一出现就有行可看
        _progress_set_items([(p["coin"], "close") for p in db.list_positions(sid) if p["opened"]])
        actual = await asyncio.to_thread(acct.positions)
        targets = []
        for p in db.list_positions(sid):
            if not p["opened"]:
                continue
            live_sz = actual.get(p["coin"], {}).get("szi")
            if live_sz:
                targets.append((p["coin"], abs(live_sz)))
        _progress_set_items([(c, "close") for c, _ in targets])

        for coin, live_sz in targets:
            _progress_item(coin, status="submitting")
            try:
                res = await asyncio.to_thread(acct.close_market, coin, live_sz)
            except Exception as exc:
                failed.append(f"{coin}: {exc}")
                _progress_item(coin, status="failed", error=str(exc))
                continue
            if res["ok"]:
                db.add_fill(sid, coin, "sell", res["avg_px"], res["total_sz"], res["fee"], res["cloid"])
                _progress_item(coin, status="filled", sz=res["total_sz"], px=res["avg_px"])
            else:
                failed.append(f"{coin}: {res['error']}")
                _progress_item(coin, status="failed", error=res["error"])

        await asyncio.to_thread(_enrich_after_orders, acct, sid, op_started)

        if failed:
            note = "平仓失败: " + "; ".join(failed)
            db.set_status(sid, "running", note=note)
            db.add_log("error", note, sid)
            raise RuntimeError(note)

        # 先停对账任务再写状态，避免中间态被推送出去
        _stop_tasks(sid)
        db.set_status(sid, "closed", note="", closed_at=db.now_ms())
        db.add_log("info", "策略已终止", sid)
        return _public(db.get_strategy(sid))
    finally:
        _progress_finish()
        _closing.discard(sid)


async def resolve(action: str) -> dict:
    async with _lock:
        row = db.get_active_strategy()
        if not row or row["status"] != "mismatch":
            raise ValueError("当前没有待确认的异常策略")
        sid = row["id"]
        if action == "resume":
            acct = get_account(row["env"])
            for p in db.list_positions(sid):
                if p["opened"]:
                    actual = acct.positions().get(p["coin"], {}).get("szi")
                    if actual is not None:
                        db.mark_position_opened(sid, p["coin"], abs(actual), p["entry_px"] or 0.0)
            db.set_status(sid, "running", note="已按实际持仓恢复")
            _start_tasks(row["env"], sid)
            db.add_log("info", "异常策略已恢复运行", sid)
            return _public(db.get_strategy(sid))
        if action == "close":
            db.set_status(sid, "running", note="按用户选择平仓")
            return await _close_row(db.get_strategy(sid))
        raise ValueError("action 只能是 resume 或 close")


# ---------------- 净值与基准 ----------------

def _earn_apy(acct: Account) -> float:
    global _last_earn_apy
    try:
        r = acct.info.post("/info", {"type": "borrowLendReserveState", "token": 0})
        _last_earn_apy = float(r["supplyYearlyRate"])
    except Exception:
        if _last_earn_apy is None:
            _last_earn_apy = 0.0
    return _last_earn_apy or 0.0


def _funding_total(acct: Account, sid: int, since_ms: int) -> float:
    now = db.now_ms()
    cached = _funding_cache.get(sid)
    if cached is not None and now - _last_funding_refresh.get(sid, 0) < FUNDING_REFRESH_MS:
        return cached
    total = 0.0
    for row in acct.funding_since(since_ms):
        delta = row.get("delta") or {}
        total += float(delta.get("usdc", 0) or 0)
    _funding_cache[sid] = total
    _last_funding_refresh[sid] = now
    return total


_last_funding_refresh: dict[int, int] = {}


def _equity(acct: Account, row: dict, marks: dict[str, float], positions: list[dict]) -> float:
    equity = row["total_invested"]
    for p in positions:
        if not p["opened"] or not p["sz"] or not p["entry_px"]:
            continue
        mark = marks.get(p["coin"])
        if mark:
            equity += (mark - p["entry_px"]) * p["sz"]
    fees = sum(f["fee"] for f in db.list_fills(row["id"]))
    equity -= fees
    equity += _funding_total(acct, row["id"], row["created_at"])
    return equity


def _benchmark_value(acct: Account, row: dict, marks: dict[str, float], ts_ms: int) -> float | None:
    base = row["total_invested"]
    bench = row["benchmark"]
    if bench == "earn":
        apy = _earn_apy(acct)
        days = max(0.0, (ts_ms - row["created_at"]) / 86_400_000)
        return base * (1 + apy) ** days
    start = row.get("bench_start_px")
    coin = {"btc": "BTC", "eth": "ETH", "paxg": "PAXG"}.get(bench)
    mark = marks.get(coin) if coin else None
    if not start or not mark:
        return None
    return base * mark / start


def _sample_active() -> None:
    row = db.get_active_strategy()
    if not row or row["status"] != "running":
        return
    acct = get_account(row["env"])
    marks = acct.mids()
    positions = db.list_positions(row["id"])
    ts = db.now_ms()
    equity = _equity(acct, row, marks, positions)
    bench = _benchmark_value(acct, row, marks, ts)
    db.add_equity_point(row["id"], ts, equity, bench)
    for p in positions:
        if p["opened"]:
            actual = acct.positions().get(p["coin"], {})
            if actual:
                db.update_position_mark(row["id"], p["coin"], actual.get("margin_used"),
                                        actual.get("liquidation_px"))


# ---------------- 后台任务 ----------------

async def _sampler(env: str, sid: int) -> None:
    while True:
        try:
            await asyncio.to_thread(_sample_active)
        except Exception as exc:
            db.add_log("warn", f"净值采样失败: {exc}", sid)
        await asyncio.sleep(SAMPLE_INTERVAL_MS / 1000)


async def _reconciler(env: str, sid: int) -> None:
    """订阅账户状态推送，逐标的比对；订阅不可用时退回 30 秒轮询。"""
    url = WS_URLS[env]
    acct = get_account(env)
    attempt = 0
    while True:
        try:
            async with websockets.connect(url) as ws:
                await ws.send(json.dumps({"method": "subscribe",
                                          "subscription": {"type": "clearinghouseState", "user": acct.address}}))
                attempt = 0
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("channel") == "subscriptionResponse":
                        continue
                    data = msg.get("data")
                    if isinstance(data, dict):
                        await asyncio.to_thread(_check_against, sid, data)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            db.add_log("warn", f"账户状态订阅异常，转轮询: {exc}", sid)
            await asyncio.sleep(30)
            try:
                state = await asyncio.to_thread(acct.info.user_state, acct.address)
                await asyncio.to_thread(_check_against, sid, state)
            except Exception as exc2:
                db.add_log("warn", f"轮询对账失败: {exc2}", sid)
            attempt += 1
            await asyncio.sleep(min(30, 2 ** min(attempt, 4)))


def _check_against(sid: int, state: dict) -> None:
    row = db.get_strategy(sid)
    if not row or row["status"] != "running" or sid in _closing:
        return
    # 推送的形如 {dex, user, clearinghouseState:{...}}，REST 的形如 {assetPositions,...}
    inner = state.get("clearinghouseState") if isinstance(state.get("clearinghouseState"), dict) else state
    actual = {}
    for ap in inner.get("assetPositions", []):
        pos = ap.get("position", ap)
        if float(pos.get("szi", 0)) != 0:
            actual[pos["coin"]] = abs(float(pos["szi"]))

    diffs = []
    for p in db.list_positions(sid):
        if not p["opened"]:
            continue
        want = float(p["sz"] or 0)
        got = actual.get(p["coin"], 0.0)
        if abs(got - want) > max(1e-9, want * 1e-6):
            diffs.append(f"{p['coin']} 记录 {want} 实际 {got}")

    if diffs:
        note = "对账不一致: " + "; ".join(diffs)
        db.set_status(sid, "mismatch", note=note)
        db.add_log("error", note, sid)
        _stop_tasks(sid)


def _start_tasks(env: str, sid: int) -> None:
    _stop_tasks(sid)
    _tasks[sid] = asyncio.create_task(_sampler(env, sid))
    _tasks[sid * 1000] = asyncio.create_task(_reconciler(env, sid))


def _stop_tasks(sid: int) -> None:
    for key in (sid, sid * 1000):
        task = _tasks.pop(key, None)
        if task and not task.done():
            task.cancel()


# ---------------- 离线补齐与恢复 ----------------

def _pick_interval(gap_ms: int) -> str:
    for iv in BACKFILL_INTERVALS:
        if gap_ms / INTERVAL_MS[iv] <= 5000:
            return iv
    return BACKFILL_INTERVALS[-1]


def backfill(row: dict) -> list[tuple[int, int]]:
    last = db.last_equity_point(row["id"])
    if not last:
        return []
    start = int(last["time"])
    end = db.now_ms()
    gap = end - start
    if gap <= SAMPLE_INTERVAL_MS:
        return []

    acct = get_account(row["env"])
    positions = [p for p in db.list_positions(row["id"]) if p["opened"] and p["sz"]]
    if not positions:
        return []

    interval = _pick_interval(gap)
    series: dict[str, list] = {}
    for p in positions:
        rows = acct.info.candles_snapshot(p["coin"], interval, start, end)
        series[p["coin"]] = rows

    fees = sum(f["fee"] for f in db.list_fills(row["id"]))
    stamps = sorted({int(r["t"]) for rows in series.values() for r in rows})
    for ts in stamps:
        if ts <= start:
            continue
        equity = row["total_invested"] - fees
        marks: dict[str, float] = {}
        for p in positions:
            bar = next((r for r in series[p["coin"]] if int(r["t"]) == ts), None)
            if not bar:
                continue
            close = float(bar["c"])
            marks[p["coin"]] = close
            equity += (close - p["entry_px"]) * p["sz"]
        bench = _benchmark_value(acct, row, marks, ts)
        db.add_equity_point(row["id"], ts, equity, bench, backfilled=1)
    return [(start, end)]


async def recover() -> None:
    """启动时恢复运行中的策略：订单追查、对账、补齐。"""
    db.init_db()
    row = db.get_active_strategy()
    if not row or row["status"] != "running":
        return
    sid = row["id"]
    try:
        acct = get_account(row["env"])
        state = acct.info.user_state(acct.address)
        _check_against(sid, state)
        row = db.get_strategy(sid)
        if row["status"] != "running":
            db.add_log("warn", "启动对账发现仓位不一致，已暂停等待确认", sid)
            return
        ranges = await asyncio.to_thread(backfill, row)
        if ranges:
            db.add_log("info", f"已补齐离线净值 {ranges}", sid)
        _start_tasks(row["env"], sid)
        db.add_log("info", "策略已恢复运行", sid)
    except Exception as exc:
        db.add_log("error", f"策略恢复失败: {exc}", sid)
