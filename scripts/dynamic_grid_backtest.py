"""日频加密货币动态网格（动态上下界 + 动态层数 + 均线门控）回测脚本。

核心设计：**无前视偏差**
- 全量 CSV 加载后，门控查表与动态上下界查表仅在 [0, t] 的滚动窗口上计算
  （`rolling` 天然只依赖 t 及以前数据），并只做 ffill、不做 bfill
- 校验 START 前预热天数 >= max(ma_window, gate_window, atr_window)，不足则报错退出
- vnpy 回测仅回放 [START, END]，策略按日期查表取滚动值

策略机制（与 notebook/dynamic_grid_backtest_eth_dynamic_band.ipynb 一致）：
- 上下界：mid = MA(ma_window)，half = K × ATR(atr_window)，区间 [mid-half, mid+half]
- 层数：steps = clip(ceil(2·half / max(mid×target_ret, pricetick)), min_steps, max_steps)
- 门控：close < MA(gate_window) -> 撤单观望；行情回暖 -> 以当前价重锚定
- 重锚定择时：①初始建仓 ②价格出界 ③偏离中枢 > reanchor_dev（冷却期 reanchor_gap 天内跳过）④门控恢复

用法：
    uv run python scripts/dynamic_grid_backtest.py
    uv run python scripts/dynamic_grid_backtest.py --data data/btc_usdt_1d.csv --end 2024-12-31 --print-json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# 绘图（无头后端，仅 --plot 时启用）
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- vnpy.alpha 可选依赖兼容处理（脚本无头运行，注入空实现避免导入失败） ----
import types

_STUB_ATTRS = {
    "alphalens": {},
    "alphalens.utils": {"get_clean_factor_and_forward_returns": lambda *a, **k: None},
    "alphalens.tears": {"create_full_tear_sheet": lambda *a, **k: None},
    "talib": {},
    "torch": {},
    "torch.nn": {},
    "torch.optim": {},
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
from loguru import logger as _loguru
from vnpy.alpha import AlphaLab, AlphaStrategy, BacktestingEngine
from vnpy.trader.constant import Direction, Exchange, Interval
from vnpy.trader.object import BarData

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# 网格构造
# ============================================================
def build_grid_prices(lower: float, upper: float, steps: int, ratio: float = 1.0) -> list[float]:
    """构造网格价：ratio=1 时等距切分。"""
    if ratio >= 1:
        step = (upper - lower) / steps
        return [lower + step * i for i in range(steps + 1)]
    d1 = (upper - lower) * (1 - ratio) / (1 - ratio ** steps)
    prices = [lower + d1 * (1 - ratio ** k) / (1 - ratio) for k in range(steps)]
    prices.append(upper)
    return prices


# ============================================================
# 预计算查表（无前视：仅滚动窗口 + ffill）
# ============================================================
def build_ma_gate_map(df: pd.DataFrame, gate_window: int) -> dict[str, bool]:
    """门控查表 {日期iso: close<MA(gate_window) 是否下跌行情}。"""
    t = df.copy()
    t["ma"] = t["close"].rolling(gate_window).mean()
    t["bear"] = t["close"] < t["ma"]
    return {pd.Timestamp(d).date().isoformat(): bool(b) for d, b in zip(t["timestamp"], t["bear"])}


def build_bounds_map(
    df: pd.DataFrame,
    ma_window: int,
    atr_window: int,
    k_width: float,
    target_ret: float,
    min_steps: int,
    max_steps: int,
    pricetick: float,
) -> dict[str, tuple[float, float, int]]:
    """动态上下界查表 {日期iso: (mid, half, steps)}，仅用滚动窗口 + ffill（无 bfill，无前视）。"""
    t = df.copy()
    tr = pd.concat(
        [
            t["high"] - t["low"],
            (t["high"] - t["close"].shift()).abs(),
            (t["low"] - t["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    t["mid"] = t["close"].rolling(ma_window).mean().ffill()
    t["half"] = (k_width * tr.rolling(atr_window).mean()).ffill().fillna(0.0)
    t["spacing"] = (t["mid"] * target_ret).clip(lower=pricetick)
    # 数据开头（START 之前，不会被回测查询）滚动指标为 NaN，fillna 兜底避免 astype 报错
    t["steps"] = np.clip(np.ceil(2 * t["half"] / t["spacing"]), min_steps, max_steps).fillna(min_steps).astype(int)
    out = {}
    for d, mid, half, stp in zip(t["timestamp"], t["mid"], t["half"], t["steps"]):
        out[pd.Timestamp(d).date().isoformat()] = (float(mid), float(half), int(stp))
    return out


# ============================================================
# 策略
# ============================================================
class DynamicBandMAGatedGridStrategy(AlphaStrategy):
    """动态上下界 + 动态层数 + 均线门控动态网格。

    - 网格区间 = [mid-half, mid+half] = [MA(ma_window) - K×ATR(atr_window),
      MA(ma_window) + K×ATR(atr_window)]（查表，无前视）
    - 层数 = clip(ceil(2·half/(mid×target_ret)), min_steps, max_steps)
    - 门控：close < MA(gate_window) -> 撤单观望；回暖 -> 以当前价重锚定
    """

    # 运行期由 globals 注入的策略参数（在 run_backtest 中统一赋值）
    pass


# 策略运行时依赖的外部状态（通过闭包全局在 run 前赋值）
_GATE_MAP: dict[str, bool] = {}
_BOUNDS_MAP: dict[str, tuple[float, float, int]] = {}


class DynamicGridStrategy(AlphaStrategy):
    """无门控动态网格（静态网格对照基准）。"""

    grid_spacing_ratio: float = 1.0

    def on_init(self) -> None:
        self.volume_per_grid = _CFG["volume_per_grid"]
        self.grid_prices = build_grid_prices(
            _CFG["grid_lower"], _CFG["grid_upper"], _CFG["grid_steps"], self.grid_spacing_ratio
        )
        self.standing_sells: dict[float, float] = {}
        self.standing_buys: dict[float, float] = {}
        self.order_level: dict[str, float] = {}
        self.roll_to_buy: list[float] = []
        self.roll_to_sell: list[float] = []
        self.base_pos_done: bool = False
        self.pending_base_vol: float = 0.0
        self.pending_base_cost: float = 0.0

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
            anchor = self.floor_grid(close)
            for p in self.grid_prices:
                if p > anchor:
                    self.place_sell_at(p, self.volume_per_grid)
            for p in self.grid_prices:
                if p < anchor:
                    self.place_buy_at(p, self.volume_per_grid)
            return

        if not (_CFG["grid_lower"] <= close <= _CFG["grid_upper"]):
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
            self.pending_base_vol = 0.0
            self.pending_base_cost = 0.0
            return
        if trade.direction == Direction.LONG:
            self.standing_buys[level] -= trade.volume
            if self.standing_buys[level] <= 1e-9:
                del self.standing_buys[level]
            self.roll_to_sell.append(level)
        else:
            self.standing_sells[level] -= trade.volume
            if self.standing_sells[level] <= 1e-9:
                del self.standing_sells[level]
            self.roll_to_buy.append(level)


class MAGatedGridStrategy(DynamicGridStrategy):
    """静态 MA 门控动态网格（对照基准，静态区间 + 门控撤单）。"""

    def on_init(self) -> None:
        super().on_init()
        self.state = "GRID"

    def _bear_market(self, bar) -> bool:
        key = bar.datetime.date().isoformat()
        return _GATE_MAP.get(key, False)

    def _rebuild_grid(self, close: float) -> None:
        self.cancel_all()
        self.standing_sells.clear()
        self.standing_buys.clear()
        self.order_level.clear()
        self.roll_to_buy.clear()
        self.roll_to_sell.clear()
        anchor = self.floor_grid(close)
        for p in self.grid_prices:
            if p > anchor:
                self.place_sell_at(p, self.volume_per_grid)
            if p < anchor:
                self.place_buy_at(p, self.volume_per_grid)

    def on_bars(self, bars: dict) -> None:
        bar = bars.get(_CFG["vt_symbol"])
        if bar is None:
            return
        if not self.base_pos_done:
            super().on_bars(bars)
            if self._bear_market(bar):
                self.cancel_all()
                self.standing_sells.clear()
                self.standing_buys.clear()
                self.order_level.clear()
                self.roll_to_buy.clear()
                self.roll_to_sell.clear()
                self.pending_base_vol = 0.0
                self.pending_base_cost = 0.0
                self.state = "SUSPENDED"
            return
        is_bear = self._bear_market(bar)
        if is_bear and self.state == "GRID":
            self.cancel_all()
            self.standing_sells.clear()
            self.standing_buys.clear()
            self.order_level.clear()
            self.roll_to_buy.clear()
            self.roll_to_sell.clear()
            self.state = "SUSPENDED"
            return
        if not is_bear and self.state == "SUSPENDED":
            self.state = "GRID"
            self._rebuild_grid(bar.close_price)
            return
        if self.state == "SUSPENDED":
            return
        super().on_bars(bars)


class DynamicBandMAGatedGridStrategy(MAGatedGridStrategy):
    """动态上下界 + 动态层数 + MA 门控（本次核心策略）。"""

    def on_init(self) -> None:
        super().on_init()
        self.last_reanchor_date: pd.Timestamp | None = None
        self.cur_bounds: tuple[float, float, int] | None = None
        self.reanchor_count: int = 0

    def _bounds_on(self, bar) -> tuple[float, float, int]:
        key = bar.datetime.date().isoformat()
        return _BOUNDS_MAP.get(key, (bar.close_price, 0.0, _CFG["min_steps"]))

    def _should_reanchor(self, bar, close: float) -> bool:
        if self.cur_bounds is None:
            return True
        if self.last_reanchor_date is not None:
            gap = (bar.datetime.date() - self.last_reanchor_date).days
            if gap < _CFG["reanchor_gap"]:
                return False
        mid, half, _ = self.cur_bounds
        lower, upper = mid - half, mid + half
        out_of_range = not (lower <= close <= upper)
        dev = abs(close - mid) / mid if mid > 0 else 0.0
        return out_of_range or dev > _CFG["reanchor_dev"]

    def _rebuild_grid(self, close: float, bar) -> None:
        mid, half, steps = self._bounds_on(bar)
        lower, upper = mid - half, mid + half
        self.cur_bounds = (mid, half, steps)
        self.last_reanchor_date = bar.datetime.date()
        self.reanchor_count += 1
        self.cancel_all()
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
            if self._bear_market(bar):
                self.cancel_all()
                self.standing_sells.clear()
                self.standing_buys.clear()
                self.order_level.clear()
                self.roll_to_buy.clear()
                self.roll_to_sell.clear()
                self.pending_base_vol = 0.0
                self.pending_base_cost = 0.0
                self.state = "SUSPENDED"
            return

        is_bear = self._bear_market(bar)
        if is_bear and self.state == "GRID":
            self.cancel_all()
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


# ============================================================
# 回测运行
# ============================================================
def run_backtest(df: pd.DataFrame, cfg: dict, strategy_class) -> dict:
    """对给定数据与参数运行一次回测，返回指标 dict。"""
    start, end = cfg["start"], cfg["end"]
    vt_symbol = cfg["vt_symbol"]
    symbol = vt_symbol.split(".")[0]

    # 无前视校验：START 前预热天数必须足够
    in_range = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    bdf = df[in_range].reset_index(drop=True)
    pre_start = df[df["timestamp"] < start]
    min_warm = max(cfg["ma_window"], cfg["gate_window"], cfg["atr_window"])
    if len(pre_start) < min_warm:
        raise ValueError(
            f"START={start} 前仅 {len(pre_start)} 天数据，不足预热 min(max(ma,gate,atr))={min_warm} 天，"
            f"请提前 START 或缩短窗口，否则引入 bfill 前视偏差"
        )

    # vnpy 数据准备（仅写 [start, end] 区间）
    lab = AlphaLab(cfg["lab"])
    bars = []
    for r in bdf.itertuples():
        bars.append(
            BarData(
                symbol=symbol,
                exchange=Exchange("GLOBAL"),
                datetime=r.timestamp.to_pydatetime(),
                interval=Interval.DAILY,
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
        interval=Interval.DAILY,
        start=pd.Timestamp(start).to_pydatetime(),
        end=pd.Timestamp(end).to_pydatetime(),
        capital=cfg["capital"],
        risk_free=0,
        annual_days=365,
    )
    engine.add_strategy(strategy_class, {}, pl.DataFrame())
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

    # 配对胜率：按时间序买卖配对（同量），卖价>买价计赢
    win, total_pair = 0, 0
    buy_vol = 0.0
    buy_px = 0.0
    for t in sorted(trades, key=lambda x: x.datetime or datetime.min):
        if t.direction == Direction.LONG:
            buy_vol += t.volume
            buy_px = (buy_px * max(buy_vol - t.volume, 0) + t.price * t.volume) / max(buy_vol, 1e-9) if buy_vol > 0 else t.price
            buy_vol = max(buy_vol, 0.0)
        elif buy_vol > 0:
            match_vol = min(t.volume, buy_vol)
            total_pair += 1
            if t.price > buy_px:
                win += 1
            buy_vol = max(buy_vol - t.volume, 0.0)

    # 绘图数据：逐日余额（cumsum(net_pnl)+capital）
    balance = None
    if engine.daily_df is not None:
        net_pnl = engine.daily_df["net_pnl"].to_list()
        balance = np.cumsum(net_pnl) + cfg["capital"]

    return {
        "start": start,
        "end": end,
        "total_return": round(st["total_return"] / 100, 4),
        "annual_return": round(st["annual_return"] / 100, 4),
        "return_std": round((st["return_std"] or 0) * np.sqrt(365) / 100, 4),
        "max_dd": round(st["max_ddpercent"] / 100, 4),
        "max_dd_duration": int(st["max_drawdown_duration"]),
        "sharpe": round(st["sharpe_ratio"], 4),
        "end_balance": round(st["end_balance"], 2),
        "trades": len(trades),
        "buy": buy_cnt,
        "sell": sell_cnt,
        "win_rate": round(win / total_pair, 4) if total_pair else None,
        "turnover_ratio": round(st["total_turnover"] / cfg["capital"], 4),
        "profit_days": int(st["profit_days"]),
        "loss_days": int(st["loss_days"]),
        "reanchor": reanchor,
        "balance": balance,
        "bdf": bdf,
    }


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="日频动态网格（动态上下界+动态层数+均线门控）回测")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "eth_usdt_1d.csv"), help="日频 OHLCV CSV")
    parser.add_argument("--start", default="auto", help="回测起始日期，或 auto=数据起始后第 max(ma,gate,atr) 个交易日")
    parser.add_argument("--end", default="2025-12-31", help="回测结束日期")
    parser.add_argument("--vt-symbol", default="ETHUSDT.GLOBAL", help="vnpy 合约代码")
    parser.add_argument("--lab", default=str(PROJECT_ROOT / "data" / "vnpy_lab"), help="vnpy 数据实验室目录")

    parser.add_argument("--ma-window", type=int, default=60, help="中枢均线周期（动态上下界 mid）")
    parser.add_argument("--gate-window", type=int, default=60, help="门控均线周期：close<MA(gate) 撤单观望")
    parser.add_argument("--atr-window", type=int, default=14, help="ATR 周期")
    parser.add_argument("--k-width", type=float, default=4.0, help="通道半宽 = K × ATR")
    parser.add_argument("--target-ret", type=float, default=0.015, help="单格目标收益率（间距=mid×TARGET_RET）")
    parser.add_argument("--min-steps", type=int, default=5, help="动态层数下限")
    parser.add_argument("--max-steps", type=int, default=200, help="动态层数上限")
    parser.add_argument("--reanchor-gap", type=int, default=7, help="重锚定冷却期（天）")
    parser.add_argument("--reanchor-dev", type=float, default=0.25, help="偏离中枢触发重锚定阈值")

    parser.add_argument("--capital", type=float, default=100_000, help="初始资金")
    parser.add_argument("--fee-rate", type=float, default=0.001, help="单边手续费率")
    parser.add_argument("--volume-per-grid", type=float, default=1.0, help="每格成交数量")
    parser.add_argument("--initial-base-ratio", type=float, default=0.5, help="初始底仓占资金比例")
    parser.add_argument("--pricetick", type=float, default=0.01, help="最小价格变动（也作单格间距下限）")

    parser.add_argument("--strategy", choices=["dynamic", "static_gate", "plain"], default="dynamic",
                        help="回测策略：dynamic=动态带+门控（默认），static_gate=静态MA门控，plain=无门控")
    parser.add_argument("--print-json", action="store_true", help="输出单行 JSON（供参数搜索解析）")
    parser.add_argument("--plot", default="", help="输出目录（如 pic/），非空时生成净值/区间/回撤 3 张图")
    parser.add_argument("--plot-prefix", default="", help="图片文件名前缀，自动加币种（如 TRXUSDT+train_ -> trx_train_）")
    return parser.parse_args()


_CFG: dict = {}


def _json_default(o):
    """JSON 序列化兜底：numpy 标量 -> 原生类型，NaN/Inf -> None。"""
    try:
        import math
        if isinstance(o, float) and not math.isfinite(o):
            return None
        if hasattr(o, "item"):
            return o.item()
    except Exception:
        pass
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def plot_report(result: dict, cfg: dict, outdir: Path, prefix: str) -> None:
    """生成报告图片：净值曲线 / 动态区间 / 回撤阴影。

    Args:
        result: run_backtest 返回值（含 balance / bdf）
        cfg: 回测配置
        outdir: 输出目录（如 pic/）
        prefix: 文件名前缀（train_ / val_）
    """
    plt.rcParams.update({"font.family": "Noto Sans CJK SC", "axes.unicode_minus": False})
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bdf = result["bdf"]
    balance = result["balance"]
    dates = bdf["timestamp"]
    capital = cfg["capital"]
    start, end = result["start"], result["end"]

    # 基准：简单持有（期初全仓）
    bh = capital * bdf["close"] / bdf["close"].iloc[0]

    # 动态区间（从 BOUNDS_MAP 提取区间内每日 lower/upper/mid）
    lo, hi, mid = [], [], []
    for d in bdf["timestamp"].dt.date.astype(str):
        b = _BOUNDS_MAP.get(d)
        if b:
            lo.append(b[0] - b[1]); hi.append(b[0] + b[1]); mid.append(b[0])
        else:
            lo.append(np.nan); hi.append(np.nan); mid.append(np.nan)

    # --- 1. 净值曲线 ---
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(dates, balance, lw=1.4, color="#1f77b4", label=f"策略净值 (期末 {result['end_balance']:,.0f})")
    ax.plot(dates, bh, lw=1.2, color="#2ca02c", ls="--", label="Buy&Hold")
    ax.axhline(capital, color="#333", ls=":", lw=0.8)
    ax.set_title(f"净值曲线：策略 vs Buy&Hold（{start} ~ {end}）")
    ax.set_ylabel("净值 (USDT)")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(outdir / f"{prefix}grid_nav.png", dpi=110)
    plt.close(fig)

    # --- 2. 动态区间 ---
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(dates, bdf["close"], lw=0.8, color="#333333", label="Close", zorder=3)
    ax.fill_between(dates, lo, hi, color="#1f77b4", alpha=0.15, label="动态区间 [mid±K×ATR]", zorder=2)
    ax.plot(dates, mid, color="#1f77b4", lw=1.0, ls=":", label="中枢 mid")
    ax.set_title(f"价格与动态网格上下界（ma={cfg['ma_window']} K={cfg['k_width']} atr={cfg['atr_window']}）")
    ax.set_ylabel("价格 (USDT)")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    fig.savefig(outdir / f"{prefix}grid_band.png", dpi=110)
    plt.close(fig)

    # --- 3. 回撤（双轴：左净值、右回撤%）
    bal_series = pd.Series(balance, index=dates)
    dd_pct = (bal_series / bal_series.cummax() - 1) * 100
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(dates, bal_series, lw=1.3, color="#1f77b4", label="净值")
    ax.set_ylabel("净值 (USDT)")
    ax2 = ax.twinx()
    ax2.fill_between(dates, dd_pct, 0, where=(dd_pct < 0), color="#d62728", alpha=0.35, label="回撤 %")
    ax2.plot(dates, dd_pct, lw=0.6, color="#d62728", alpha=0.6)
    ax2.set_ylabel("回撤 (%)")
    ax.set_title(f"净值与回撤（最大回撤 {result['max_dd']*100:.2f}%，持续 {result['max_dd_duration']} 天）")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)
    plt.tight_layout()
    fig.savefig(outdir / f"{prefix}grid_drawdown.png", dpi=110)
    plt.close(fig)
    print(f"图片已保存: {outdir}/{prefix}grid_nav.png, grid_band.png, grid_drawdown.png")


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    global _GATE_MAP, _BOUNDS_MAP
    _GATE_MAP = build_ma_gate_map(df, args.gate_window)
    _BOUNDS_MAP = build_bounds_map(
        df, args.ma_window, args.atr_window, args.k_width, args.target_ret,
        args.min_steps, args.max_steps, args.pricetick,
    )

    # auto 起点：数据起始日后第 max(ma,gate,atr) 个交易日（保证足够预热，杜绝前视/失真）
    start = args.start
    if start == "auto":
        warm = max(args.ma_window, args.gate_window, args.atr_window)
        if len(df) <= warm:
            raise ValueError(f"数据仅 {len(df)} 天，无法满足预热 {warm} 天")
        start = str(df["timestamp"].iloc[warm].date())

    cfg = {
        "start": start, "end": args.end, "vt_symbol": args.vt_symbol, "lab": args.lab,
        "ma_window": args.ma_window, "gate_window": args.gate_window, "atr_window": args.atr_window,
        "k_width": args.k_width, "target_ret": args.target_ret,
        "min_steps": args.min_steps, "max_steps": args.max_steps,
        "reanchor_gap": args.reanchor_gap, "reanchor_dev": args.reanchor_dev,
        "capital": args.capital, "fee_rate": args.fee_rate,
        "volume_per_grid": args.volume_per_grid, "initial_base_ratio": args.initial_base_ratio,
        "pricetick": args.pricetick,
        "grid_lower": 1200.0, "grid_upper": 4300.0, "grid_steps": 50,
    }
    global _CFG
    _CFG = cfg

    if args.strategy == "dynamic":
        cls = DynamicBandMAGatedGridStrategy
    elif args.strategy == "static_gate":
        cls = MAGatedGridStrategy
    else:
        cls = DynamicGridStrategy

    result = run_backtest(df, cfg, cls)

    if args.plot:
        coin = args.vt_symbol.split(".")[0].replace("USDT", "").lower()
        prefix = f"{coin}_{args.plot_prefix}" if args.plot_prefix else coin + "_"
        plot_report(result, cfg, args.plot, prefix)

    if args.print_json:
        json_result = {k: v for k, v in result.items() if k not in ("balance", "bdf")}
        print(json.dumps(json_result, ensure_ascii=False, default=_json_default))
    else:
        print("=== 动态网格回测结果 ===")
        print(f"区间: {result['start']} ~ {result['end']} | 币种: {args.vt_symbol}")
        print(f"策略: {args.strategy} | ma={args.ma_window} gate={args.gate_window} "
              f"atr={args.atr_window} K={args.k_width} target_ret={args.target_ret}")
        print(f"总收益率: {result['total_return']*100:.2f}%")
        print(f"年化收益: {result['annual_return']*100:.2f}%")
        print(f"年化波动: {result['return_std']*100:.2f}%")
        print(f"最大回撤: {result['max_dd']*100:.2f}% (持续 {result['max_dd_duration']} 天)")
        print(f"Sharpe:   {result['sharpe']:.2f}")
        print(f"期末余额: {result['end_balance']:,.2f}")
        print(f"成交笔数: {result['trades']} (买 {result['buy']} / 卖 {result['sell']})")
        if result["win_rate"] is not None:
            print(f"配对胜率: {result['win_rate']*100:.2f}%  换手率: {result['turnover_ratio']:.2f}")
        print(f"盈利/亏损天数: {result['profit_days']}/{result['loss_days']}")
        print(f"重锚定次数: {result['reanchor']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
