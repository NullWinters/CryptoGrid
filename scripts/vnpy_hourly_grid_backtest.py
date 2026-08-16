"""vnpy 小时频动态带网格回测：验证 DQN 门控动作序列与 Gym 模拟器的一致性。

说明
----
- vnpy.alpha.AlphaLab 仅支持 DAILY / MINUTE 两种 bar 频率；BacktestingEngine 本身只按
  bar.datetime 逐根回放，不限制频率。因此本脚本把小时 bar 以 Interval.MINUTE 写入
  独立实验室 data/vnpy_lab_btc_1h，引擎 interval 同样设为 MINUTE，实现小时频回放。
- 动态上下界采用方案 A：日频等价 mid=MA(60)、half=4×ATR(14)，shift(1)+ffill 映射到
  小时频（与 scripts/dqn_gate_env.build_bounds(band_mode="daily") 完全一致）。
- 门控：--mode rl 时查 DQN 动作映射表；--mode ma 时 close < mid 撤单；--mode no 永不撤单。

用法
----
    uv run python scripts/vnpy_hourly_grid_backtest.py --mode rl --print-json
    uv run python scripts/vnpy_hourly_grid_backtest.py --mode all
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- vnpy.alpha 可选依赖兼容处理（与 dynamic_grid_backtest.py 一致） ----
import types

_STUB_ATTRS = {
    "alphalens": {},
    "alphalens.utils": {"get_clean_factor_and_forward_returns": lambda *a, **k: None},
    "alphalens.tears": {"create_full_tear_sheet": lambda *a, **k: None},
    "talib": {},
    "lightgbm": {},
}
for _name, _attrs in _STUB_ATTRS.items():
    if _name in sys.modules:
        continue
    _mod = types.ModuleType(_name)
    for _k, _v in _attrs.items():
        setattr(_mod, _k, _v)
    sys.modules[_name] = _mod

import bisect
from collections import deque
from loguru import logger as _loguru
from vnpy.alpha import AlphaLab, AlphaStrategy, BacktestingEngine
from vnpy.trader.constant import Direction, Exchange, Interval
from vnpy.trader.object import BarData

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.dqn_gate_env as g  # noqa: E402
from stable_baselines3 import DQN  # noqa: E402


# ============================================================
# 查表构造
# ============================================================
def build_hourly_bounds_map(df: pd.DataFrame, cfg: dict) -> dict[datetime, tuple[float, float, int]]:
    """小时频动态带查表 {bar.datetime: (mid, half, steps)}。

    与 scripts.dqn_gate_env.build_bounds(band_mode="daily") 同一口径。
    """
    bounds = g.build_bounds(
        df,
        ma_window=cfg["ma_window"],
        atr_window=cfg["atr_window"],
        k_width=cfg["k_width"],
        target_ret=cfg["target_ret"],
        min_steps=cfg["min_steps"],
        max_steps=cfg["max_steps"],
        pricetick=cfg["pricetick"],
        band_mode="daily",
    )
    out: dict[datetime, tuple[float, float, int]] = {}
    for ts, mid, half, stp in zip(df["timestamp"], bounds["mid"], bounds["half"], bounds["steps"]):
        out[pd.Timestamp(ts).to_pydatetime()] = (float(mid), float(half), int(stp))
    return out


def build_action_map(test_df: pd.DataFrame, action_csv: Path) -> dict[datetime, int]:
    """DQN 动作查表 {bar.datetime: 0/1}，与测试集小时 bar 一一对应。"""
    actions = pd.read_csv(action_csv)["action"].astype(int).to_list()
    if len(actions) != len(test_df):
        raise ValueError(f"动作序列长度 {len(actions)} 与测试集 bar 数 {len(test_df)} 不一致")
    out: dict[datetime, int] = {}
    for ts, a in zip(test_df["timestamp"], actions):
        out[pd.Timestamp(ts).to_pydatetime()] = int(a)
    return out


# ============================================================
# 策略
# ============================================================
_CFG: dict = {}
_BOUNDS_MAP: dict[datetime, tuple[float, float, int]] = {}
_ACTION_MAP: dict[datetime, int] = {}
_GATE_MODE: str = "rl"  # rl / ma / no

# 在线推理相关
_USE_ONLINE: bool = False
_MODEL = None
_FEATURES_DF: pd.DataFrame | None = None
_FEATURE_MEAN: np.ndarray | None = None
_FEATURE_STD: np.ndarray | None = None
_TS_TO_IDX: dict[datetime, int] = {}
_START_IDX: int = 0
_END_IDX: int = 0
_N_STACK: int = 4
_DECISION_INTERVAL: int = 1
_ACCOUNT_FEATURES: list[str] = []


def build_grid_prices(lower: float, upper: float, steps: int, ratio: float = 1.0) -> list[float]:
    if ratio >= 1:
        step = (upper - lower) / steps
        return [lower + step * i for i in range(steps + 1)]
    d1 = (upper - lower) * (1 - ratio) / (1 - ratio ** steps)
    prices = [lower + d1 * (1 - ratio ** k) / (1 - ratio) for k in range(steps)]
    prices.append(upper)
    return prices


class HourlyDynamicBandGridStrategy(AlphaStrategy):
    """小时频动态带 + 动态层数 + 门控（门控来源由 _GATE_MODE 决定）。"""

    def on_init(self) -> None:
        self.volume_per_grid = _CFG["volume_per_grid"]
        self.grid_prices = build_grid_prices(
            _CFG["grid_lower"], _CFG["grid_upper"], _CFG["grid_steps"], 1.0
        )
        self.standing_sells: dict[float, float] = {}
        self.standing_buys: dict[float, float] = {}
        self.order_level: dict[str, float] = {}
        self.roll_to_buy: list[float] = []
        self.roll_to_sell: list[float] = []
        self.base_pos_done: bool = False
        self.pending_base_vol: float = 0.0
        self.pending_base_cost: float = 0.0
        self.state = "GRID"
        self.last_reanchor_dt: datetime | None = None
        self.cur_bounds: tuple[float, float, int] | None = None
        self.reanchor_count: int = 0

        # 在线推理状态
        self.last_action: int = 0
        self.nav_peak: float = _CFG["capital"]
        self.position_cost: float = 0.0
        self.obs_dim: int = len(g.FEATURE_COLS) + 1             + (1 if _DECISION_INTERVAL > 1 else 0)             + len(_ACCOUNT_FEATURES)
        self.obs_buffer: deque = deque([np.zeros(self.obs_dim, dtype=np.float32) for _ in range(_N_STACK)], maxlen=_N_STACK)

    def floor_grid(self, price: float) -> float:
        idx = bisect.bisect_right(self.grid_prices, price) - 1
        return self.grid_prices[max(idx, 0)]

    def sell_capacity(self) -> float:
        pending = self.pending_base_vol if self.pending_base_cost > 0 else 0.0
        return self.get_pos(_CFG["vt_symbol"]) + pending

    def buy_cash(self) -> float:
        pending = self.pending_base_cost if self.pending_base_cost > 0 else 0.0
        return self.get_cash_available() - pending

    def place_sell_at(self, price: float, vol: float) -> float:
        committed = sum(self.standing_sells.values())
        avail = max(0.0, self.sell_capacity() - committed)
        vol = min(vol, avail)
        if vol > 0:
            oids = self.sell(_CFG["vt_symbol"], price, vol)
            self.standing_sells[price] = self.standing_sells.get(price, 0.0) + vol
            for oid in oids:
                self.order_level[oid] = price
        return vol

    def place_buy_at(self, price: float, vol: float) -> float:
        committed = sum(p * v for p, v in self.standing_buys.items())
        avail_cash = max(0.0, self.buy_cash() - committed)
        vol = min(vol, avail_cash / (price * (1 + _CFG["fee_rate"])))
        if vol > 0:
            oids = self.buy(_CFG["vt_symbol"], price, vol)
            self.standing_buys[price] = self.standing_buys.get(price, 0.0) + vol
            for oid in oids:
                self.order_level[oid] = price
        return vol

    def _bounds_on(self, bar) -> tuple[float, float, int]:
        return _BOUNDS_MAP.get(bar.datetime, (bar.close_price, 0.0, _CFG["min_steps"]))

    def _bear_market(self, bar) -> bool:
        if _GATE_MODE == "no":
            return False
        if _GATE_MODE == "ma":
            mid, _, _ = self._bounds_on(bar)
            return bar.close_price < mid
        # rl: 查 DQN 动作表，1=撤单观望
        return _ACTION_MAP.get(bar.datetime, 0) == 1

    # ---------- 在线 DQN 推理 ----------
    def _gate_is_bear(self, bar, first_bar: bool = False) -> bool:
        if _GATE_MODE == "rl" and _USE_ONLINE:
            return self._online_is_bear(bar, first_bar)
        return self._bear_market(bar)

    def _current_nav(self, close: float, first_bar: bool = False) -> float:
        if first_bar:
            pos = self.pending_base_vol
            cash = self.get_cash_available() - self.pending_base_cost
        else:
            pos = self.get_pos(_CFG["vt_symbol"])
            cash = self.get_cash_available()
        return cash + pos * close

    def _price_grid_pos(self, close: float) -> float:
        cur = self.cur_bounds
        if cur is not None:
            mid, half, _ = cur
        else:
            key = self.strategy_engine.datetime
            mid, half, _ = _BOUNDS_MAP.get(key, (close, 0.0, _CFG["min_steps"]))
        lower, upper = mid - half, mid + half
        width = upper - lower
        if width <= 0:
            return 0.5
        return float(np.clip((close - lower) / width, 0.0, 1.0))

    def _account_obs(self, close: float, first_bar: bool) -> np.ndarray:
        if not _ACCOUNT_FEATURES:
            return np.asarray([], dtype=np.float32)
        if first_bar:
            pos = self.pending_base_vol
            cash = self.get_cash_available() - self.pending_base_cost
            cost = self.pending_base_cost * (1 + _CFG["fee_rate"])
        else:
            pos = self.get_pos(_CFG["vt_symbol"])
            cash = self.get_cash_available()
            cost = self.position_cost
        nav = max(cash + pos * close, 1e-9)
        vals = []
        for name in _ACCOUNT_FEATURES:
            if name == "position_ratio":
                vals.append(float(np.clip(pos * close / nav, 0.0, 1.0)))
            elif name == "cash_ratio":
                vals.append(float(np.clip(cash / nav, 0.0, 1.0)))
            elif name == "nav_drawdown":
                vals.append(float(np.clip(nav / max(self.nav_peak, 1e-9) - 1.0, -1.0, 0.0)))
            elif name == "upnl_ratio":
                upnl = pos * close - cost
                vals.append(float(np.clip(upnl / nav, -1.0, 1.0)))
            elif name == "grid_state_flag":
                vals.append(1.0 if self.state == "GRID" else 0.0)
            elif name == "price_grid_pos":
                vals.append(self._price_grid_pos(close))
            else:
                vals.append(0.0)
        return np.asarray(vals, dtype=np.float32)

    def _online_obs(self, bar, first_bar: bool) -> np.ndarray:
        idx = _TS_TO_IDX.get(bar.datetime, _START_IDX)
        feat = (_FEATURES_DF.iloc[idx].to_numpy() - _FEATURE_MEAN) / _FEATURE_STD
        feat = np.clip(feat, -5.0, 5.0)
        out = np.concatenate([feat, [np.float32(self.last_action)]])
        if _DECISION_INTERVAL > 1:
            is_decision = 1.0 if (idx - _START_IDX) % _DECISION_INTERVAL == 0 else 0.0
            out = np.concatenate([out, [np.float32(is_decision)]])
        if _ACCOUNT_FEATURES:
            out = np.concatenate([out, self._account_obs(bar.close_price, first_bar)])
        return out.astype(np.float32)

    def _online_is_bear(self, bar, first_bar: bool) -> bool:
        close = bar.close_price
        nav = self._current_nav(close, first_bar)
        self.nav_peak = max(self.nav_peak, nav)

        obs = self._online_obs(bar, first_bar)
        self.obs_buffer.append(obs)
        flat = np.concatenate(list(self.obs_buffer)).reshape(1, -1)
        a_arr, _ = _MODEL.predict(flat, deterministic=True)
        action = int(a_arr[0])
        self.last_action = action
        return action == 1

    def _should_reanchor(self, bar, close: float) -> bool:
        if self.cur_bounds is None:
            return True
        if self.last_reanchor_dt is not None:
            gap_hours = (bar.datetime - self.last_reanchor_dt).total_seconds() / 3600
            if gap_hours < _CFG["reanchor_min_gap"]:
                return False
        mid, half, _ = self.cur_bounds
        lower, upper = mid - half, mid + half
        out_of_range = not (lower <= close <= upper)
        dev = abs(close - mid) / mid if mid > 0 else 0.0
        return out_of_range or dev > _CFG["reanchor_dev"]

    def _cancel_grid_orders(self) -> None:
        """只撤网格挂单，不撤底仓订单（与 Gym 环境底仓必成交的口径一致）。"""
        for oid in list(self.order_level.keys()):
            self.cancel_order(oid)

    def _rebuild_grid(self, close: float, bar) -> None:
        mid, half, steps = self._bounds_on(bar)
        lower, upper = mid - half, mid + half
        self.cur_bounds = (mid, half, steps)
        self.last_reanchor_dt = bar.datetime
        self.reanchor_count += 1

        self._cancel_grid_orders()
        self.standing_sells.clear()
        self.standing_buys.clear()
        self.order_level.clear()
        self.roll_to_buy.clear()
        self.roll_to_sell.clear()

        self.grid_prices = build_grid_prices(lower, upper, int(steps), 1.0)
        anchor = self.floor_grid(close)
        for p in self.grid_prices:
            if p > anchor:
                self.place_sell_at(p, self.volume_per_grid)
        for p in self.grid_prices:
            if p < anchor:
                self.place_buy_at(p, self.volume_per_grid)

    def on_bars(self, bars: dict) -> None:
        bar = bars.get(_CFG["vt_symbol"])
        if bar is None:
            return
        close = bar.close_price

        if not self.base_pos_done:
            base_vol = min(
                _CFG["capital"] * _CFG["initial_base_ratio"] / close,
                self.get_cash_available() / close,
            )
            if base_vol > 0:
                self.buy(_CFG["vt_symbol"], close, base_vol)
                self.pending_base_vol = base_vol
                self.pending_base_cost = base_vol * close
            self.base_pos_done = True
            self._rebuild_grid(close, bar)
            if self._gate_is_bear(bar, first_bar=True):
                self._cancel_grid_orders()
                self.standing_sells.clear()
                self.standing_buys.clear()
                self.order_level.clear()
                self.roll_to_buy.clear()
                self.roll_to_sell.clear()
                self.pending_base_vol = 0.0
                self.pending_base_cost = 0.0
                self.state = "SUSPENDED"
            return

        is_bear = self._gate_is_bear(bar, first_bar=False)

        if is_bear and self.state == "GRID":
            self._cancel_grid_orders()
            self.standing_sells.clear()
            self.standing_buys.clear()
            self.order_level.clear()
            self.roll_to_buy.clear()
            self.roll_to_sell.clear()
            self.state = "SUSPENDED"
            return

        if not is_bear and self.state == "SUSPENDED":
            self.state = "GRID"
            self._rebuild_grid(close, bar)
            return

        if self.state == "SUSPENDED":
            return

        if self._should_reanchor(bar, close):
            self._rebuild_grid(close, bar)
            return

        for p in self.roll_to_buy:
            self.place_buy_at(p, self.volume_per_grid)
        self.roll_to_buy.clear()
        for p in self.roll_to_sell:
            self.place_sell_at(p, self.volume_per_grid)
        self.roll_to_sell.clear()

    def on_trade(self, trade) -> None:
        level = self.order_level.pop(trade.vt_orderid, None)
        if level is None:
            # 底仓成交：成本记录 + 释放待成交占用
            self.position_cost += trade.volume * trade.price * (1 + _CFG["fee_rate"])
            self.pending_base_vol = 0.0
            self.pending_base_cost = 0.0
            return
        if trade.direction == Direction.LONG:
            self.position_cost += trade.volume * trade.price * (1 + _CFG["fee_rate"])
            self.standing_buys[level] -= trade.volume
            if self.standing_buys[level] <= 1e-9:
                del self.standing_buys[level]
            self.roll_to_sell.append(level)
        else:
            if self.get_pos(_CFG["vt_symbol"]) > 0:
                avg_cost = self.position_cost / self.get_pos(_CFG["vt_symbol"])
                self.position_cost = max(0.0, self.position_cost - trade.volume * avg_cost)
            self.standing_sells[level] -= trade.volume
            if self.standing_sells[level] <= 1e-9:
                del self.standing_sells[level]
            self.roll_to_buy.append(level)


# ============================================================
# 回测运行
# ============================================================
def run_backtest(df: pd.DataFrame, cfg: dict, mode: str = "rl") -> dict:
    """对给定数据与参数运行一次 vnpy 小时频回测。"""
    global _GATE_MODE, _BOUNDS_MAP, _ACTION_MAP
    global _USE_ONLINE, _MODEL, _FEATURES_DF, _FEATURE_MEAN, _FEATURE_STD
    global _TS_TO_IDX, _START_IDX, _END_IDX, _DECISION_INTERVAL, _ACCOUNT_FEATURES

    _GATE_MODE = mode
    _BOUNDS_MAP = build_hourly_bounds_map(df, cfg)

    start, end = cfg["start"], cfg["end"]
    vt_symbol = cfg["vt_symbol"]
    symbol = vt_symbol.split(".")[0]

    in_range = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    bdf = df[in_range].reset_index(drop=True)

    global _N_STACK
    _N_STACK = int(cfg.get("n_stack", 4))
    _USE_ONLINE = mode == "rl" and cfg.get("model_path") is not None
    if _USE_ONLINE:
        _MODEL = DQN.load(str(cfg["model_path"]))
        _FEATURE_MEAN, _FEATURE_STD = g.get_feature_stats(
            df, cfg["train_start"], cfg["train_end"],
            ma_window=cfg["ma_window"], atr_window=cfg["atr_window"],
            k_width=cfg["k_width"], target_ret=cfg["target_ret"],
            min_steps=cfg["min_steps"], max_steps=cfg["max_steps"],
            pricetick=cfg["pricetick"], band_mode="daily",
        )
        feat, _ = g.prepare_data(
            df, ma_window=cfg["ma_window"], atr_window=cfg["atr_window"],
            k_width=cfg["k_width"], target_ret=cfg["target_ret"],
            min_steps=cfg["min_steps"], max_steps=cfg["max_steps"],
            pricetick=cfg["pricetick"], band_mode="daily",
        )
        _FEATURES_DF = feat[g.FEATURE_COLS].astype(np.float32)
        _TS_TO_IDX = {pd.Timestamp(ts).to_pydatetime(): i for i, ts in enumerate(df["timestamp"])}
        _START_IDX = int(df.index[df["timestamp"] >= start][0])
        _END_IDX = int(df.index[df["timestamp"] <= end][-1])
        _DECISION_INTERVAL = int(cfg.get("decision_interval", 1))
        _ACCOUNT_FEATURES = list(cfg.get("account_features", []))
        _ACTION_MAP = {}
    else:
        _ACTION_MAP = build_action_map(bdf, cfg["action_csv"]) if mode == "rl" else {}

    # 独立实验室：每次重建，避免旧数据污染
    lab_path = Path(cfg["lab"])
    if lab_path.exists():
        shutil.rmtree(lab_path)

    lab = AlphaLab(str(lab_path))
    bars = []
    for r in bdf.itertuples():
        bars.append(
            BarData(
                symbol=symbol,
                exchange=Exchange("GLOBAL"),
                datetime=r.timestamp.to_pydatetime(),
                interval=Interval.MINUTE,  # vnpy AlphaLab 不支持 HOUR，借用 MINUTE 通道存小时 bar
                open_price=r.open,
                high_price=r.high,
                low_price=r.low,
                close_price=r.close,
                volume=r.volume,
                gateway_name="DB",
            )
        )
    lab.save_bar_data(bars)
    lab.add_contract_setting(
        vt_symbol,
        long_rate=cfg["fee_rate"],
        short_rate=cfg["fee_rate"],
        size=1,
        pricetick=cfg["pricetick"],
    )

    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=[vt_symbol],
        interval=Interval.MINUTE,
        start=pd.Timestamp(start).to_pydatetime(),
        end=pd.Timestamp(end).to_pydatetime(),
        capital=cfg["capital"],
        risk_free=0,
        annual_days=365,
    )
    engine.add_strategy(HourlyDynamicBandGridStrategy, {}, pd.DataFrame())
    _loguru.disable("vnpy.alpha")
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    st = engine.calculate_statistics()
    trades = engine.get_all_trades()
    _loguru.enable("vnpy.alpha")

    buy_cnt = sum(1 for t in trades if t.direction == Direction.LONG)
    sell_cnt = len(trades) - buy_cnt
    reanchor = getattr(engine.strategy, "reanchor_count", 0)

    # 用 vnpy daily_df 的累计 net_pnl + capital 构造日频净值
    balance = None
    if engine.daily_df is not None:
        net_pnl = engine.daily_df["net_pnl"].to_list()
        balance = np.cumsum(net_pnl) + cfg["capital"]

    # 用日频 balance 计算与 Gym 可比的指标
    metrics = {}
    if balance is not None and len(balance) > 0:
        bal = pd.Series(balance)
        total_return = bal.iloc[-1] / cfg["capital"] - 1
        max_dd = float((bal / bal.cummax() - 1).min())
        n_days = len(bal)
        annual_return = (bal.iloc[-1] / cfg["capital"]) ** (365 / n_days) - 1 if n_days > 0 else 0.0
        daily_ret = bal.pct_change().dropna()
        return_std = float(daily_ret.std() * np.sqrt(365)) if len(daily_ret) > 0 else 0.0
        sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(365)) if daily_ret.std() > 0 else 0.0
        calmar = float(annual_return / abs(max_dd)) if max_dd < 0 else 0.0
        metrics = {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "return_std": return_std,
            "max_dd": max_dd,
            "sharpe": sharpe,
            "calmar": calmar,
            "end_nav": float(bal.iloc[-1]),
        }

    return {
        "mode": mode,
        "start": start,
        "end": end,
        "total_return": round(metrics.get("total_return", float(st.get("total_return", 0)) / 100), 4),
        "annual_return": round(metrics.get("annual_return", 0.0), 4),
        "return_std": round(metrics.get("return_std", 0.0), 4),
        "max_dd": round(metrics.get("max_dd", float(st.get("max_ddpercent", 0)) / 100), 4),
        "sharpe": round(metrics.get("sharpe", 0.0), 4),
        "calmar": round(metrics.get("calmar", 0.0), 4),
        "end_nav": round(metrics.get("end_nav", float(st.get("end_balance", 0))), 2),
        "trades": len(trades),
        "buy": buy_cnt,
        "sell": sell_cnt,
        "reanchor": reanchor,
        "balance": balance,
        "bdf": bdf,
    }


# ============================================================
# Gym 对比
# ============================================================
def _gym_stats_for_actions(df: pd.DataFrame, cfg: dict, actions=None, mode="custom") -> dict:
    """运行 Gym 模拟器，返回与 vnpy 可比的指标。"""
    res = g.simulate(
        df, cfg["start"], cfg["end"], actions=actions, mode=mode,
        ma_window=cfg["ma_window"], atr_window=cfg["atr_window"], k_width=cfg["k_width"],
        target_ret=cfg["target_ret"], min_steps=cfg["min_steps"], max_steps=cfg["max_steps"],
        reanchor_min_gap=cfg["reanchor_min_gap"], reanchor_dev=cfg["reanchor_dev"],
        volume_per_grid=cfg["volume_per_grid"], fee_rate=cfg["fee_rate"],
        initial_base_ratio=cfg["initial_base_ratio"], pricetick=cfg["pricetick"],
        band_mode="daily",
    )
    st = res["stats"]
    # 日频口径最大回撤，便于和 vnpy 的 daily balance 对比
    daily_nav = res["nav"].resample("1D").last().dropna()
    daily_max_dd = float((daily_nav / daily_nav.cummax() - 1).min()) if len(daily_nav) > 0 else 0.0
    return {
        "total_return": round(st["total_return"], 4),
        "annual_return": round(st["annual_return"], 4),
        "max_dd_hourly": round(st["max_dd"], 4),
        "max_dd_daily": round(daily_max_dd, 4),
        "end_nav": round(st["end_nav"], 2),
        "trades": res["n_trades"],
        "reanchor": res["reanchor_count"],
    }


def compare_vnpy_gym(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """跑三个策略的 vnpy 与 Gym 对比。"""
    actions = pd.read_csv(cfg["action_csv"])["action"].astype(int).to_list()
    rows = []
    for mode, label in [("rl", "RL门控(DQN)"), ("ma", "MA门控动态带"), ("no", "永不撤单动态带")]:
        v = run_backtest(df, cfg, mode)
        if mode == "rl":
            gym = _gym_stats_for_actions(df, cfg, actions=actions, mode="custom")
        elif mode == "ma":
            gym = _gym_stats_for_actions(df, cfg, mode="ma_gate")
        else:
            gym = _gym_stats_for_actions(df, cfg, mode="no_gate")

        rows.append({
            "策略": label,
            "回测": "vnpy",
            "总收益率%": round(v["total_return"] * 100, 2),
            "最大回撤%": round(v["max_dd"] * 100, 2),
            "期末NAV": round(v["end_nav"], 0),
            "成交笔数": v["trades"],
            "重锚定": v["reanchor"],
        })
        rows.append({
            "策略": label,
            "回测": "Gym",
            "总收益率%": round(gym["total_return"] * 100, 2),
            "最大回撤%": round(gym["max_dd_hourly"] * 100, 2),
            "期末NAV": round(gym["end_nav"], 0),
            "成交笔数": gym["trades"],
            "重锚定": gym["reanchor"],
        })
    return pd.DataFrame(rows)


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vnpy 小时频动态带网格回测：DQN 门控一致性验证")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "btc_usdt_1h.csv"), help="小时频 OHLCV CSV")
    parser.add_argument("--actions", default=str(PROJECT_ROOT / "data" / "dqn_gate_test_actions.csv"), help="DQN 测试集动作 CSV")
    parser.add_argument("--start", default="2026-01-01", help="回测起始时间")
    parser.add_argument("--end", default="2026-07-31 23:00:00", help="回测结束时间")
    parser.add_argument("--vt-symbol", default="BTCUSDT.GLOBAL", help="vnpy 合约代码")
    parser.add_argument("--lab", default=str(PROJECT_ROOT / "data" / "vnpy_lab_btc_1h"), help="vnpy 数据实验室目录")

    parser.add_argument("--ma-window", type=int, default=60, help="日频中枢均线周期")
    parser.add_argument("--atr-window", type=int, default=14, help="日频 ATR 周期")
    parser.add_argument("--k-width", type=float, default=4.0, help="通道半宽 = K × ATR")
    parser.add_argument("--target-ret", type=float, default=0.015, help="单格目标收益率")
    parser.add_argument("--min-steps", type=int, default=5, help="动态层数下限")
    parser.add_argument("--max-steps", type=int, default=200, help="动态层数上限")
    parser.add_argument("--reanchor-min-gap", type=int, default=168, help="重锚定冷却期（小时）")
    parser.add_argument("--reanchor-dev", type=float, default=0.25, help="偏离中枢触发重锚定阈值")

    parser.add_argument("--capital", type=float, default=100_000, help="初始资金")
    parser.add_argument("--fee-rate", type=float, default=0.001, help="单边手续费率")
    parser.add_argument("--volume-per-grid", type=float, default=0.01, help="每格成交数量（BTC）")
    parser.add_argument("--initial-base-ratio", type=float, default=0.5, help="初始底仓占资金比例")
    parser.add_argument("--pricetick", type=float, default=0.01, help="最小价格变动")

    parser.add_argument("--model", default="", help="DQN 模型路径；提供时 rl 模式启用 vnpy 在线推理")
    parser.add_argument("--train-start", default="2022-01-01", help="特征归一化统计量所用训练集起点")
    parser.add_argument("--train-end", default="2025-12-31 23:00:00", help="特征归一化统计量所用训练集终点")
    parser.add_argument("--account-features", default="", help="在线推理账户特征，逗号分隔；如 position_ratio,cash_ratio")
    parser.add_argument("--decision-interval", type=int, default=1, help="在线推理决策间隔（小时）")
    parser.add_argument("--n-stack", type=int, default=4, help="在线推理观测帧堆叠数")

    parser.add_argument("--mode", choices=["rl", "ma", "no", "all"], default="all", help="回测模式")
    parser.add_argument("--print-json", action="store_true", help="输出单行 JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    global _CFG, _BOUNDS_MAP, _ACTION_MAP, _GATE_MODE
    account_features = [x.strip() for x in args.account_features.split(",") if x.strip()] if args.account_features else []
    cfg = {
        "start": args.start, "end": args.end, "vt_symbol": args.vt_symbol, "lab": args.lab,
        "ma_window": args.ma_window, "atr_window": args.atr_window,
        "k_width": args.k_width, "target_ret": args.target_ret,
        "min_steps": args.min_steps, "max_steps": args.max_steps,
        "reanchor_min_gap": args.reanchor_min_gap, "reanchor_dev": args.reanchor_dev,
        "capital": args.capital, "fee_rate": args.fee_rate,
        "volume_per_grid": args.volume_per_grid, "initial_base_ratio": args.initial_base_ratio,
        "pricetick": args.pricetick,
        "action_csv": Path(args.actions),
        "model_path": Path(args.model) if args.model else None,
        "train_start": args.train_start, "train_end": args.train_end,
        "account_features": account_features,
        "decision_interval": args.decision_interval,
        "n_stack": args.n_stack,
        "grid_lower": 0.0, "grid_upper": 1e9, "grid_steps": 50,  # 初始占位，实际由 _rebuild_grid 重建
    }

    _CFG = cfg
    if args.mode == "all":
        cmp = compare_vnpy_gym(df, cfg)
        print(cmp.to_string(index=False))
        if args.print_json:
            print(json.dumps(cmp.to_dict(orient="records"), ensure_ascii=False))
        return 0

    _CFG = cfg
    result = run_backtest(df, cfg, args.mode)
    if args.print_json:
        print(json.dumps({k: v for k, v in result.items() if k not in ("balance", "bdf")}, ensure_ascii=False))
    else:
        print("=== vnpy 小时频动态带网格回测 ===")
        print(f"模式: {args.mode} | 区间: {result['start']} ~ {result['end']}")
        print(f"总收益率: {result['total_return']*100:.2f}%")
        print(f"最大回撤: {result['max_dd']*100:.2f}%")
        print(f"期末NAV: {result['end_nav']:,.2f}")
        print(f"成交笔数: {result['trades']} (买 {result['buy']} / 卖 {result['sell']})")
        print(f"重锚定次数: {result['reanchor']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
