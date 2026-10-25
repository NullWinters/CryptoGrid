"""Hyperliquid 账户与下单封装。

账户地址从 .env 的 HL_ACC_ADD 读取，若为代理钱包则经 userRole 解析出主账户；
私钥从 HL_SECRET_KEY 读取，仅用于签名，不打印、不返回。
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import eth_account
from dotenv import load_dotenv

from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils.types import Cloid

REST_URLS = {
    "mainnet": "https://api.hyperliquid.xyz",
    "testnet": "https://api.hyperliquid-testnet.xyz",
}

ROOT = Path(__file__).resolve().parent.parent.parent


def _load_env() -> None:
    load_dotenv(ROOT / ".env")


class Account:
    """一个环境下已解析主账户的读写下单封装。"""

    def __init__(self, env: str):
        if env not in REST_URLS:
            raise ValueError(f"unknown env {env}")
        _load_env()
        raw = os.environ.get("HL_ACC_ADD", "").strip()
        secret = os.environ.get("HL_SECRET_KEY", "").strip()
        if not raw or not secret:
            raise RuntimeError(".env 缺少 HL_ACC_ADD 或 HL_SECRET_KEY")

        self.env = env
        self.base_url = REST_URLS[env]
        self.info = Info(self.base_url, skip_ws=True)
        self._meta: dict[str, dict] | None = None
        self.signer = eth_account.Account.from_key(secret)
        self.address = self._resolve_master(raw)
        self.exchange = Exchange(self.signer, self.base_url, account_address=self.address)

    def _resolve_master(self, raw: str) -> str:
        role = self.info.post("/info", {"type": "userRole", "user": raw})
        if isinstance(role, dict) and role.get("role") == "agent":
            return role["data"]["user"]
        return raw

    # --- 查询 ---

    def withdrawable(self) -> float:
        """统一账户下永续可用余额常为 0，实际资金在现货 USDC 里，两者相加才是可用额度。"""
        perp = float(self.info.user_state(self.address)["withdrawable"])
        spot = 0.0
        try:
            for b in self.info.spot_user_state(self.address)["balances"]:
                if b["coin"] == "USDC":
                    spot = float(b["total"]) - float(b.get("hold", 0) or 0)
                    break
        except Exception:
            pass
        return perp + spot

    def meta(self) -> dict[str, dict]:
        if self._meta is None:
            uni = self.info.meta()["universe"]
            self._meta = {c["name"]: c for c in uni}
        return self._meta

    def mids(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.info.all_mids().items()}

    def positions(self) -> dict[str, dict]:
        """按标的返回持仓，值为 szi、entry_px、margin_used、liquidation_px。"""
        state = self.info.user_state(self.address)
        out: dict[str, dict] = {}
        for ap in state.get("assetPositions", []):
            p = ap["position"]
            if float(p["szi"]) == 0:
                continue
            out[p["coin"]] = {
                "szi": float(p["szi"]),
                "entry_px": float(p["entryPx"]) if p.get("entryPx") else None,
                "margin_used": float(p["marginUsed"]) if p.get("marginUsed") else None,
                "liquidation_px": float(p["liquidationPx"]) if p.get("liquidationPx") else None,
            }
        return out

    def funding_since(self, start_ms: int) -> list[dict]:
        try:
            rows = self.info.post("/info", {"type": "userFunding", "user": self.address,
                                            "startTime": start_ms})
            return rows if isinstance(rows, list) else []
        except Exception:
            return []

    def order_by_cloid(self, cloid: str) -> dict | None:
        try:
            res = self.info.query_order_by_cloid(self.address, cloid)
        except Exception:
            return None
        if not isinstance(res, dict) or res.get("status") != "order":
            return None
        return res

    # --- 交易 ---

    def set_isolated_leverage(self, coin: str, leverage: int) -> None:
        self.exchange.update_leverage(leverage, coin, is_cross=False)

    def size_for_notional(self, coin: str, notional: float, price: float) -> float:
        sz_decimals = int(self.meta()[coin]["szDecimals"])
        return round(notional / price, sz_decimals)

    def open_market(self, coin: str, notional: float, price: float | None = None,
                    slippage: float = 0.01) -> dict:
        if price is None:
            price = self.mids()[coin]
        sz = self.size_for_notional(coin, notional, price)
        cloid = Cloid("0x" + uuid.uuid4().hex)
        res = self.exchange.market_open(coin, True, sz, None, slippage, cloid=cloid)
        return self._normalize(res, cloid)

    def close_market(self, coin: str, sz: float, slippage: float = 0.01) -> dict:
        cloid = Cloid("0x" + uuid.uuid4().hex)
        res = self.exchange.market_close(coin, sz, None, slippage, cloid=cloid)
        return self._normalize(res, cloid)

    def recent_fees(self, since_ms: int) -> dict[str, float]:
        """一次取回区间内各标的最近一笔成交的手续费，避免逐单查询。"""
        out: dict[str, float] = {}
        try:
            rows = self.info.user_fills_by_time(self.address, since_ms)
        except Exception:
            return out
        for f in rows:
            coin = f.get("coin")
            if coin and coin not in out:
                out[coin] = float(f.get("fee", 0) or 0)
        return out

    @staticmethod
    def _normalize(res: dict, cloid: Cloid) -> dict:
        """统一为 {ok, cloid, avg_px, total_sz, fee, error}。"""
        out = {"ok": False, "cloid": str(cloid), "avg_px": None, "total_sz": None,
               "fee": 0.0, "error": None}
        if not isinstance(res, dict) or res.get("status") != "ok":
            out["error"] = str(res)
            return out
        statuses = (res.get("response") or {}).get("data", {}).get("statuses", [])
        if not statuses:
            out["error"] = "empty statuses"
            return out
        st = statuses[0]
        if "filled" in st:
            f = st["filled"]
            out.update(ok=True, avg_px=float(f["avgPx"]), total_sz=float(f["totalSz"]),
                       fee=float(f.get("fee", 0.0) or 0.0))
        elif "error" in st:
            out["error"] = st["error"]
        else:
            out["error"] = str(st)
        return out


_cache: dict[str, Account] = {}


def get_account(env: str) -> Account:
    if env not in _cache:
        _cache[env] = Account(env)
    return _cache[env]
