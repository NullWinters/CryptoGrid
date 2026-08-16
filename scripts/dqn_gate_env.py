"""小时频 DQN 门控环境：动态上下界 + 动态层数网格，RL 仅替换 MA 门控。

本模块提供：
- build_features: 构造小时频滚动特征（严格无前视）
- build_bounds_map: 动态 ATR 通道上下界与层数（严格无前视）
- DynamicBandGrid: 与 notebook/dynamic_grid_backtest_eth_dynamic_band.ipynb 一致的
  动态带网格撮合模拟器（底仓 + 限价单 + 同价反手挂单 + 事件驱动重锚定）
- GridGateEnv: gym.Env，动作 0=正常挂单(GRID)，1=撤单观望(SUSPENDED)
- simulate: 按动作序列回测，返回 NAV 曲线与指标

用法见 notebook/dqn_gate_btc_1h.ipynb。
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# ============================================================
# 特征工程（仅用 t 及以前数据，rolling/shift/diff，无 shift(-k)）
# ============================================================
FEATURE_COLS = [
    "ret1", "ret6", "ret24", "ret168",
    "price_vs_mid",        # close / MA(MA_WINDOW) - 1
    "price_atr_pos",       # (close - mid) / half，截断
    "ma_dist168", "ma_dist720", "ma_dist1440",
    "slope20", "slope60",
    "er24", "er168",
    "adx14", "di_ratio",
    "vol24", "vol168", "vol_ratio",
    "atr_pct",
    "range_pos60",
    "vol_intensity24", "vol_zscore168",
]


# 可选账户状态特征（use_account_state=True 时启用）
ACCOUNT_FEATURES = [
    "position_ratio",       # 仓位市值 / NAV
    "cash_ratio",           # 现金 / NAV
    "nav_drawdown",         # NAV / nav_peak - 1
    "upnl_ratio",           # 持仓未实现盈亏 / NAV
    "grid_state_flag",      # 当前是否 GRID
    "price_grid_pos",       # 价格在当前网格区间中的位置 0~1
]


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> tuple[pd.Series, pd.Series]:
    """返回 (ADX, +DI/-DI 比值，已截断)。"""
    up = high.diff()
    dn = -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)
    trv = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = trv.rolling(n).mean()
    plus_di = 100 * plus_dm.rolling(n).mean() / atr
    minus_di = 100 * minus_dm.rolling(n).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(n).mean()
    di_ratio = (plus_di / minus_di).replace([np.inf, -np.inf], np.nan).clip(lower=1e-3, upper=1e3)
    return adx, np.log(di_ratio)  # 对数比，量纲更稳


def build_features(df: pd.DataFrame, ma_window: int = 1440) -> pd.DataFrame:
    """构造小时频滚动特征，返回与 df 等长的 DataFrame。"""
    t = df.copy()
    t["logret"] = np.log(t["close"]).diff()

    out: dict[str, pd.Series] = {}
    for n in (1, 6, 24, 168):
        out[f"ret{n}"] = t["close"].pct_change(n)

    mid = t["close"].rolling(ma_window).mean()
    half = t["close"].rolling(ma_window).std()  # 占位，真实 half 在 build_bounds_map 后覆盖

    out["price_vs_mid"] = t["close"] / mid - 1
    out["price_atr_pos"] = ((t["close"] - mid) / half.replace(0, np.nan)).clip(-3, 3)

    for n, name in ((168, "ma_dist168"), (720, "ma_dist720"), (1440, "ma_dist1440")):
        out[name] = t["close"] / t["close"].rolling(n).mean() - 1

    sma20 = t["close"].rolling(20).mean()
    sma60 = t["close"].rolling(60).mean()
    out["slope20"] = sma20 / sma20.shift(24) - 1
    out["slope60"] = sma60 / sma60.shift(168) - 1

    for n in (24, 168):
        abs_ret = t["close"].diff().abs()
        out[f"er{n}"] = (t["close"].diff(n).abs() / abs_ret.rolling(n).sum()).clip(0, 1)

    adx, di_log = _adx(t["high"], t["low"], t["close"])
    out["adx14"] = adx
    out["di_ratio"] = di_log

    lr = t["logret"]
    out["vol24"] = lr.rolling(24).std()
    out["vol168"] = lr.rolling(168).std()
    out["vol_ratio"] = out["vol24"] / out["vol168"]

    tr = pd.concat(
        [
            t["high"] - t["low"],
            (t["high"] - t["close"].shift()).abs(),
            (t["low"] - t["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_pct"] = tr.rolling(336).mean() / t["close"]

    low60 = t["low"].rolling(60).min()
    high60 = t["high"].rolling(60).max()
    out["range_pos60"] = ((t["close"] - low60) / (high60 - low60).replace(0, np.nan)).clip(-0.1, 1.1)

    vol_sma24 = t["volume"].rolling(24).mean()
    out["vol_intensity24"] = t["volume"] / vol_sma24
    vol_sma168 = t["volume"].rolling(168).mean()
    vol_std168 = t["volume"].rolling(168).std()
    out["vol_zscore168"] = (t["volume"] - vol_sma168) / vol_std168

    out_df = pd.DataFrame(out)
    out_df = out_df.replace([np.inf, -np.inf], np.nan)
    # 极端值截断，避免离群值主导归一化
    out_df["ret1"] = out_df["ret1"].clip(-0.2, 0.2)
    out_df["ret6"] = out_df["ret6"].clip(-0.3, 0.3)
    out_df["ret24"] = out_df["ret24"].clip(-0.5, 0.5)
    out_df["ret168"] = out_df["ret168"].clip(-1.0, 1.0)
    out_df["vol_intensity24"] = out_df["vol_intensity24"].clip(0, 20)
    out_df["vol_zscore168"] = out_df["vol_zscore168"].clip(-5, 10)
    out_df["di_ratio"] = out_df["di_ratio"].clip(-7, 7)
    return out_df.fillna(0.0)


def prepare_data(
    df: pd.DataFrame,
    ma_window: int = 60,
    atr_window: int = 14,
    k_width: float = 4.0,
    target_ret: float = 0.015,
    min_steps: int = 5,
    max_steps: int = 200,
    pricetick: float = 0.01,
    feature_ma_window: int = 1440,
    band_mode: str = "daily",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """一次性计算特征与动态 bounds。

    band_mode="daily" 时：mid/half/steps 在日频上计算（原日频策略口径），
    再 shift(1)+ffill 映射到小时频，避免日内使用当日日线数据（无前视）。
    band_mode="hourly" 时：直接在小时频上滚动计算。
    """
    feat = build_features(df, ma_window=feature_ma_window)
    bounds = build_bounds(
        df, ma_window=ma_window, atr_window=atr_window, k_width=k_width,
        target_ret=target_ret, min_steps=min_steps, max_steps=max_steps,
        pricetick=pricetick, band_mode=band_mode,
    )
    mid = bounds["mid"].to_numpy(float)
    half = bounds["half"].to_numpy(float)
    feat["price_vs_mid"] = (df["close"].to_numpy(float) / np.maximum(mid, 1e-9) - 1.0).clip(-1, 1)
    feat["price_atr_pos"] = ((df["close"].to_numpy(float) - mid) / np.maximum(half, 1e-9)).clip(-3, 3)
    return feat.fillna(0.0).astype(np.float32), bounds


def build_bounds(
    df: pd.DataFrame,
    ma_window: int = 60,
    atr_window: int = 14,
    k_width: float = 4.0,
    target_ret: float = 0.015,
    min_steps: int = 5,
    max_steps: int = 200,
    pricetick: float = 0.01,
    band_mode: str = "daily",
) -> pd.DataFrame:
    """动态 ATR 通道查表：mid/half/steps/lower/upper（无 bfill，无前视）。

    band_mode="daily"：日频计算原版动态带，shift(1)+ffill 映射小时频。
    band_mode="hourly"：直接在小时频上滚动计算（旧口径，会低估半宽）。
    """
    if band_mode == "daily":
        return _build_bounds_daily(
            df, ma_window=ma_window, atr_window=atr_window, k_width=k_width,
            target_ret=target_ret, min_steps=min_steps, max_steps=max_steps, pricetick=pricetick,
        )
    t = df.copy()
    tr = pd.concat(
        [
            t["high"] - t["low"],
            (t["high"] - t["close"].shift()).abs(),
            (t["low"] - t["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    t["mid"] = t["close"].rolling(ma_window).mean()
    t["half"] = (k_width * tr.rolling(atr_window).mean()).fillna(0.0)
    t["spacing"] = (t["mid"] * target_ret).clip(lower=pricetick)
    t["steps"] = np.clip(np.ceil(2 * t["half"] / t["spacing"]), min_steps, max_steps).fillna(min_steps).astype(int)
    t["lower"] = t["mid"] - t["half"]
    t["upper"] = t["mid"] + t["half"]
    return t[["mid", "half", "steps", "lower", "upper"]]


def _build_bounds_daily(
    df: pd.DataFrame,
    ma_window: int = 60,
    atr_window: int = 14,
    k_width: float = 4.0,
    target_ret: float = 0.015,
    min_steps: int = 5,
    max_steps: int = 200,
    pricetick: float = 0.01,
) -> pd.DataFrame:
    """日频等价动态带映射到小时频。

    关键点：日线值在当日收盘后才可知，因此对日频结果 shift(1)，
    今日 24 个小时均使用昨日收盘计算的 mid/half/steps。
    """
    daily = (
        df.set_index("timestamp")
        .resample("1D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    tr = pd.concat(
        [
            daily["high"] - daily["low"],
            (daily["high"] - daily["close"].shift()).abs(),
            (daily["low"] - daily["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    daily["mid"] = daily["close"].rolling(ma_window).mean()
    daily["half"] = (k_width * tr.rolling(atr_window).mean()).fillna(0.0)
    daily["spacing"] = (daily["mid"] * target_ret).clip(lower=pricetick)
    daily["steps"] = (
        np.clip(np.ceil(2 * daily["half"] / daily["spacing"]), min_steps, max_steps)
        .fillna(min_steps)
        .astype(int)
    )
    daily["lower"] = daily["mid"] - daily["half"]
    daily["upper"] = daily["mid"] + daily["half"]

    # 昨日收盘可得的日频 band -> ffill 到今日小时频
    out = daily[["mid", "half", "steps", "lower", "upper"]].shift(1)
    ts_index = df.set_index("timestamp").index
    hourly = out.reindex(ts_index, method="ffill")
    close_s = df.set_index("timestamp")["close"]
    hourly["mid"] = hourly["mid"].fillna(close_s)
    hourly["half"] = hourly["half"].fillna(0.0)
    hourly["steps"] = hourly["steps"].fillna(min_steps).astype(int)
    hourly["lower"] = hourly["mid"] - hourly["half"]
    hourly["upper"] = hourly["mid"] + hourly["half"]
    return hourly.reset_index(drop=True)


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
# 网格撮合模拟器
# ============================================================
@dataclass
class Fill:
    timestamp: pd.Timestamp
    side: str  # "buy" / "sell"
    price: float
    volume: float


class DynamicBandGrid:
    """动态带网格组合模拟器。

    复刻 notebook 的 DynamicBandMAGatedGridStrategy 核心逻辑：
    - 底仓：首根 bar 收盘建仓，initial_base_ratio 比例
    - 网格：mid ± K×ATR，steps 动态，等距
    - 成交：买限价 low<=p 成交，价 min(open,p)；卖限价 high>=p 成交，价 max(open,p)
    - 反手：买成交 -> 同价挂卖；卖成交 -> 同价挂买（与原 notebook 一致）
    - 重锚定：首仓、出界、偏离中枢超阈值、门控恢复；冷却期 reanchor_min_gap
    """

    def __init__(
        self,
        capital: float = 100_000.0,
        fee_rate: float = 0.001,
        initial_base_ratio: float = 0.5,
        volume_per_grid: float = 0.01,
        reanchor_min_gap: int = 168,
        reanchor_dev: float = 0.25,
        min_steps: int = 5,
        max_steps: int = 200,
        pricetick: float = 0.01,
    ):
        self.capital = capital
        self.fee_rate = fee_rate
        self.initial_base_ratio = initial_base_ratio
        self.volume_per_grid = volume_per_grid
        self.reanchor_min_gap = reanchor_min_gap
        self.reanchor_dev = reanchor_dev
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.pricetick = pricetick
        self.reset()

    def reset(self) -> None:
        self.cash = self.capital
        self.position = 0.0
        self.standing_buys: dict[float, float] = {}
        self.standing_sells: dict[float, float] = {}
        self.roll_to_buy: list[float] = []
        self.roll_to_sell: list[float] = []
        self.state = "GRID"
        self.base_pos_done = False
        self.cur_bounds: tuple[float, float, int] | None = None
        self.last_reanchor_idx = -10**9
        self.reanchor_count = 0
        self.grid_prices: list[float] = []
        self.fills: list[Fill] = []
        self.initial_base_cost = 0.0
        self.position_cost = 0.0  # 含手续费的持仓成本，用于未实现盈亏

    # ---- 基础工具 ----
    def floor_grid(self, price: float) -> float:
        idx = bisect.bisect_right(self.grid_prices, price) - 1
        return self.grid_prices[max(idx, 0)]

    def sell_capacity(self) -> float:
        committed = sum(self.standing_sells.values())
        return max(0.0, self.position - committed)

    def buy_cash(self) -> float:
        committed = sum(p * v for p, v in self.standing_buys.items())
        return max(0.0, self.cash - committed)

    # ---- 挂单 ----
    def place_sell_at(self, price: float, vol: float) -> float:
        vol = min(vol, self.sell_capacity())
        if vol > 1e-12:
            self.standing_sells[price] = self.standing_sells.get(price, 0.0) + vol
        return vol

    def place_buy_at(self, price: float, vol: float) -> float:
        avail_cash = self.buy_cash()
        vol = min(vol, avail_cash / (price * (1 + self.fee_rate)))
        if vol > 1e-12:
            self.standing_buys[price] = self.standing_buys.get(price, 0.0) + vol
        return vol

    def cancel_all(self) -> None:
        self.standing_sells.clear()
        self.standing_buys.clear()
        self.roll_to_buy.clear()
        self.roll_to_sell.clear()

    def rebuild_grid(self, close: float, bar_idx: int, mid: float, half: float, steps: int) -> None:
        lower, upper = mid - half, mid + half
        self.cur_bounds = (mid, half, int(steps))
        self.last_reanchor_idx = bar_idx
        self.reanchor_count += 1
        self.cancel_all()
        self.grid_prices = build_grid_prices(lower, upper, int(steps), 1.0)
        anchor = self.floor_grid(close)
        for p in self.grid_prices:
            if p > anchor:
                self.place_sell_at(p, self.volume_per_grid)
        for p in self.grid_prices:
            if p < anchor:
                self.place_buy_at(p, self.volume_per_grid)

    def build_base(self, close: float, bar_idx: int, mid: float, half: float, steps: int) -> None:
        base_vol = min(self.capital * self.initial_base_ratio / close, self.cash / close)
        if base_vol > 0:
            self.cash -= base_vol * close * (1 + self.fee_rate)
            self.position += base_vol
            self.initial_base_cost = base_vol * close
            self.position_cost += base_vol * close * (1 + self.fee_rate)
        self.base_pos_done = True
        self.rebuild_grid(close, bar_idx, mid, half, steps)

    def should_reanchor(self, close: float, bar_idx: int) -> bool:
        if self.cur_bounds is None:
            return True
        if bar_idx - self.last_reanchor_idx < self.reanchor_min_gap:
            return False
        mid, half, _ = self.cur_bounds
        lower, upper = mid - half, mid + half
        out_of_range = not (lower <= close <= upper)
        dev = abs(close - mid) / mid if mid > 0 else 0.0
        return out_of_range or dev > self.reanchor_dev

    def apply_action(self, action: int, close: float, bar_idx: int, mid: float, half: float, steps: int,
                     first_bar: bool = False) -> None:
        """在 bar_idx 收盘应用门控动作。first_bar=True 时只允许撤单，不做重锚定。"""
        if first_bar:
            if action == 1 and self.state == "GRID":
                self.cancel_all()
                self.state = "SUSPENDED"
            return
        if action == 1 and self.state == "GRID":
            self.cancel_all()
            self.state = "SUSPENDED"
        elif action == 0 and self.state == "SUSPENDED":
            self.state = "GRID"
            self.rebuild_grid(close, bar_idx, mid, half, steps)
        elif action == 0 and self.state == "GRID":
            if self.should_reanchor(close, bar_idx):
                self.rebuild_grid(close, bar_idx, mid, half, steps)
            else:
                self.place_roll_orders()
        # action==1 and SUSPENDED: 什么都不做

    def place_roll_orders(self) -> None:
        for p in self.roll_to_buy:
            self.place_buy_at(p, self.volume_per_grid)
        self.roll_to_buy.clear()
        for p in self.roll_to_sell:
            self.place_sell_at(p, self.volume_per_grid)
        self.roll_to_sell.clear()

    def process_fills(self, bar: pd.Series) -> None:
        """用当前 bar 的 high/low/open 撮合 standing 限价单。"""
        ts = bar["timestamp"]
        # 卖单：high >= price 成交，成交价 = max(open, price)
        for p in sorted(self.standing_sells.keys()):
            v = self.standing_sells[p]
            if bar["high"] >= p:
                fill_price = max(bar["open"], p)
                if self.position > 0:
                    avg_cost = self.position_cost / self.position
                    self.position_cost = max(0.0, self.position_cost - v * avg_cost)
                self.cash += v * fill_price * (1 - self.fee_rate)
                self.position -= v
                self.fills.append(Fill(ts, "sell", fill_price, v))
                del self.standing_sells[p]
                self.roll_to_buy.append(p)
        # 买单：low <= price 成交，成交价 = min(open, price)
        for p in sorted(self.standing_buys.keys(), reverse=True):
            v = self.standing_buys[p]
            if bar["low"] <= p:
                fill_price = min(bar["open"], p)
                self.cash -= v * fill_price * (1 + self.fee_rate)
                self.position += v
                self.position_cost += v * fill_price * (1 + self.fee_rate)
                self.fills.append(Fill(ts, "buy", fill_price, v))
                del self.standing_buys[p]
                self.roll_to_sell.append(p)

    def nav(self, close: float) -> float:
        return self.cash + self.position * close

    def upnl_ratio(self, close: float) -> float:
        nav = self.nav(close)
        if nav <= 0:
            return 0.0
        upnl = self.position * close - self.position_cost
        return float(np.clip(upnl / nav, -1.0, 1.0))


# ============================================================
# Gym 环境
# ============================================================
class GridGateEnv(gym.Env):
    """DQN 门控环境。

    obs_t: 市场特征(bar t 收盘) + prev_action
    action: 0=正常挂单(GRID)，1=撤单观望(SUSPENDED)
    时序:
        reset(): 在 start bar 收盘初始化底仓与网格，返回 obs_start
        step(a_t): 在 bar t 收盘应用 a_t -> 模拟 bar t+1 成交 -> 返回 obs_{t+1}
                   reward 为 bar t+1 的 RL 组合相对永不撤单基准的超额收益。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        capital: float = 100_000.0,
        fee_rate: float = 0.001,
        initial_base_ratio: float = 0.5,
        volume_per_grid: float = 0.01,
        ma_window: int = 60,
        atr_window: int = 14,
        k_width: float = 4.0,
        target_ret: float = 0.015,
        min_steps: int = 5,
        max_steps: int = 200,
        reanchor_min_gap: int = 168,
        reanchor_dev: float = 0.25,
        pricetick: float = 0.01,
        feature_ma_window: int = 1440,
        band_mode: str = "daily",
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        switch_penalty: float = 0.05,
        decision_interval: int = 1,
        asym_neg: float = 1.0,
        lambda_down: float = 0.0,
        lambda_dd: float = 0.0,
        dd_threshold: float = 0.08,
        asymmetric_switch: bool = False,
        switch_penalty_off: float = 0.02,
        switch_penalty_on: float = 0.10,
        use_account_state: bool = False,
        account_features: list[str] | None = None,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.capital = capital
        self.fee_rate = fee_rate
        self.initial_base_ratio = initial_base_ratio
        self.volume_per_grid = volume_per_grid
        self.ma_window = ma_window
        self.atr_window = atr_window
        self.feature_ma_window = feature_ma_window
        self.band_mode = band_mode
        self.k_width = k_width
        self.target_ret = target_ret
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.reanchor_min_gap = reanchor_min_gap
        self.reanchor_dev = reanchor_dev
        self.pricetick = pricetick
        self.switch_penalty = switch_penalty
        self.decision_interval = int(decision_interval)
        self.asym_neg = asym_neg
        self.lambda_down = lambda_down
        self.lambda_dd = lambda_dd
        self.dd_threshold = dd_threshold
        self.asymmetric_switch = asymmetric_switch
        self.switch_penalty_off = switch_penalty_off
        self.switch_penalty_on = switch_penalty_on
        self.use_account_state = use_account_state
        if account_features is not None:
            self.account_features = list(account_features)
        else:
            self.account_features = list(ACCOUNT_FEATURES) if use_account_state else []

        # 预处理：特征与 bounds
        feat, bounds = prepare_data(
            self.df, ma_window=ma_window, atr_window=atr_window, k_width=k_width,
            target_ret=target_ret, min_steps=min_steps, max_steps=max_steps, pricetick=pricetick,
            feature_ma_window=feature_ma_window, band_mode=band_mode,
        )
        self.features = feat[FEATURE_COLS].astype(np.float32)
        self.bounds_mid = bounds["mid"].to_numpy(float)
        self.bounds_half = bounds["half"].to_numpy(float)
        self.bounds_steps = bounds["steps"].to_numpy(int)

        # 默认统计：用 [start, end] 切片计算，调用方应传 train 统计
        if feature_mean is None:
            mask = (self.df["timestamp"] >= self.start) & (self.df["timestamp"] <= self.end)
            feature_mean = np.nanmean(self.features[mask.to_numpy()], axis=0)
            feature_std = np.nanstd(self.features[mask.to_numpy()], axis=0)
        self.feature_mean = feature_mean.astype(np.float32)
        self.feature_std = np.maximum(feature_std.astype(np.float32), 1e-6)

        self.n_features = len(FEATURE_COLS)
        self.extra_obs = (
            1
            + (1 if self.decision_interval > 1 else 0)
            + len(self.account_features)
        )  # prev_action + 是否决策点 + 账户状态
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_features + self.extra_obs,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        # 确定 start/end 索引
        self.start_idx = int(self.df.index[self.df["timestamp"] >= self.start][0])
        end_mask = self.df["timestamp"] <= self.end
        self.end_idx = int(self.df.index[end_mask][-1])

        self._prev_action = 0
        self._cur_idx = self.start_idx
        self._first_bar = True
        self._prev_nav_rl = self.capital
        self._prev_nav_bench = self.capital

    def _obs(self, idx: int) -> np.ndarray:
        feat = (self.features.iloc[idx].to_numpy() - self.feature_mean) / self.feature_std
        feat = np.clip(feat, -5.0, 5.0)
        out = np.concatenate([feat, [np.float32(self._prev_action)]])
        if self.decision_interval > 1:
            is_decision = 1.0 if (idx - self.start_idx) % self.decision_interval == 0 else 0.0
            out = np.concatenate([out, [np.float32(is_decision)]])
        if self.account_features:
            out = np.concatenate([out, self._account_obs(idx)])
        return out.astype(np.float32)

    def _account_obs(self, idx: int) -> np.ndarray:
        """账户状态观测，全部为比例并 clip，不参与 z-score。"""
        close = float(self.df["close"].iloc[idx])
        nav = max(self.grid.nav(close), 1e-9)
        vals: list[float] = []
        for name in self.account_features:
            if name == "position_ratio":
                vals.append(float(np.clip(self.grid.position * close / nav, 0.0, 1.0)))
            elif name == "cash_ratio":
                vals.append(float(np.clip(self.grid.cash / nav, 0.0, 1.0)))
            elif name == "nav_drawdown":
                vals.append(float(np.clip(nav / max(self.nav_peak, 1e-9) - 1.0, -1.0, 0.0)))
            elif name == "upnl_ratio":
                vals.append(self.grid.upnl_ratio(close))
            elif name == "grid_state_flag":
                vals.append(1.0 if self.grid.state == "GRID" else 0.0)
            elif name == "price_grid_pos":
                vals.append(self._price_grid_pos(idx, close))
            elif name == "standing_imbalance":
                bv = sum(self.grid.standing_buys.values())
                sv = sum(self.grid.standing_sells.values())
                vals.append(float(np.clip((sv - bv) / (sv + bv + 1e-9), -1.0, 1.0)))
            elif name == "standing_levels_ratio":
                levels = len(self.grid.standing_buys) + len(self.grid.standing_sells)
                vals.append(float(np.clip(levels / max(self.max_steps, 1), 0.0, 1.0)))
            else:
                vals.append(0.0)
        return np.asarray(vals, dtype=np.float32)

    def _price_grid_pos(self, idx: int, close: float) -> float:
        """价格在当前网格区间中的相对位置 0~1。优先用策略实际 cur_bounds，否则用理论日频等价 bounds。"""
        cur = self.grid.cur_bounds
        if cur is not None:
            mid, half, _ = cur
        else:
            mid = float(self.bounds_mid[idx])
            half = float(self.bounds_half[idx])
        lower, upper = mid - half, mid + half
        width = upper - lower
        if width <= 0:
            return 0.5
        return float(np.clip((close - lower) / width, 0.0, 1.0))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._cur_idx = self.start_idx
        self._first_bar = True
        self._prev_action = 0

        close0 = float(self.df["close"].iloc[self.start_idx])
        mid0 = float(self.bounds_mid[self.start_idx])
        half0 = float(self.bounds_half[self.start_idx])
        steps0 = int(self.bounds_steps[self.start_idx])

        self.grid = DynamicBandGrid(
            self.capital, self.fee_rate, self.initial_base_ratio, self.volume_per_grid,
            self.reanchor_min_gap, self.reanchor_dev, self.min_steps, self.max_steps, self.pricetick,
        )
        self.bench = DynamicBandGrid(
            self.capital, self.fee_rate, self.initial_base_ratio, self.volume_per_grid,
            self.reanchor_min_gap, self.reanchor_dev, self.min_steps, self.max_steps, self.pricetick,
        )
        self.grid.build_base(close0, self.start_idx, mid0, half0, steps0)
        self.bench.build_base(close0, self.start_idx, mid0, half0, steps0)

        self._prev_nav_rl = self.grid.nav(close0)
        self._prev_nav_bench = self.bench.nav(close0)
        self.nav_peak = self._prev_nav_rl
        self._prev_close = close0
        self._prev_nav = self._prev_nav_rl

        return self._obs(self.start_idx), {}

    def step(self, action: int):
        i = self._cur_idx
        close_i = float(self.df["close"].iloc[i])
        mid_i = float(self.bounds_mid[i])
        half_i = float(self.bounds_half[i])
        steps_i = int(self.bounds_steps[i])

        # 1) bar i 收盘：应用 RL 动作与基准动作（基准永远 GRID）
        action_int = int(action)
        is_decision = True
        if self.decision_interval > 1:
            bar_pos = i - self.start_idx
            is_decision = (bar_pos % self.decision_interval == 0)
            if not is_decision:
                action_int = self._prev_action  # 非决策点：沿用上一动作

        prev_action_before = self._prev_action
        self.grid.apply_action(action_int, close_i, i, mid_i, half_i, steps_i, first_bar=self._first_bar)
        if not self._first_bar:
            self.bench.apply_action(0, close_i, i, mid_i, half_i, steps_i, first_bar=False)
        self._prev_action = action_int

        # 2) 模拟 bar i+1
        j = i + 1
        if j > self.end_idx:
            # 最后一步没有下一根 bar，无成交、无奖励
            self._cur_idx = i
            self._first_bar = False
            return self._obs(i), 0.0, True, False, {}

        bar_j = self.df.iloc[j]
        close_j = float(bar_j["close"])
        self.grid.process_fills(bar_j)
        self.bench.process_fills(bar_j)

        nav_rl = self.grid.nav(close_j)
        nav_bench = self.bench.nav(close_j)

        ret_rl = nav_rl / self._prev_nav_rl - 1.0
        ret_bench = nav_bench / self._prev_nav_bench - 1.0

        # 差分收益，负差分（错误偏离基准）可加大惩罚
        diff_ret = ret_rl - ret_bench
        reward_raw = 100.0 * diff_ret
        if diff_ret < 0:
            reward_raw *= self.asym_neg

        # 组合自身亏损惩罚
        if self.lambda_down > 0:
            reward_raw -= self.lambda_down * 100.0 * max(0.0, -ret_rl)

        # 回撤阈值惩罚
        self.nav_peak = max(self.nav_peak, nav_rl)
        dd = nav_rl / self.nav_peak - 1.0
        if self.lambda_dd > 0 and dd < -self.dd_threshold:
            reward_raw -= self.lambda_dd * 100.0 * (abs(dd) - self.dd_threshold)

        # 切换惩罚：对称或非对称
        if action_int != prev_action_before:
            if self.asymmetric_switch:
                if action_int == 1 and prev_action_before == 0:
                    switch_pen = self.switch_penalty_off
                else:
                    switch_pen = self.switch_penalty_on
            else:
                switch_pen = self.switch_penalty
        else:
            switch_pen = 0.0

        reward = float(np.clip(reward_raw - switch_pen, -1.0, 1.0))

        self._prev_nav_rl = nav_rl
        self._prev_nav_bench = nav_bench
        self._prev_close = close_j
        self._prev_nav = nav_rl
        self._cur_idx = j
        self._first_bar = False

        info = {
            "nav_rl": nav_rl,
            "nav_bench": nav_bench,
            "ret_rl": ret_rl,
            "ret_bench": ret_bench,
            "reward_raw": reward_raw,
            "state": self.grid.state,
            "is_decision": is_decision,
        }
        done = j >= self.end_idx
        return self._obs(j), reward, done, False, info


# ============================================================
# 批量回测
# ============================================================
def simulate(
    df: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    actions: Sequence[int] | None = None,
    *,
    mode: str = "custom",  # custom=按 actions 序列, no_gate=全 0, ma_gate=close<MA
    capital: float = 100_000.0,
    fee_rate: float = 0.001,
    initial_base_ratio: float = 0.5,
    volume_per_grid: float = 0.01,
    ma_window: int = 60,
    atr_window: int = 14,
    k_width: float = 4.0,
    target_ret: float = 0.015,
    min_steps: int = 5,
    max_steps: int = 200,
    reanchor_min_gap: int = 168,
    reanchor_dev: float = 0.25,
    pricetick: float = 0.01,
    band_mode: str = "daily",
) -> dict:
    """按动作序列回测动态带网格，返回 NAV 曲线与统计。

    返回 dict:
        timestamps, nav, bench_nav, buyhold_nav, fills, reanchor_count,
        total_return, annual_return, return_std, max_dd, sharpe, calmar,
        n_trades, n_buys, n_sells, switch_count, state_history
    """
    df = df.reset_index(drop=True)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    start_idx = int(df.index[df["timestamp"] >= start_ts][0])
    end_idx = int(df.index[df["timestamp"] <= end_ts][-1])

    bounds = build_bounds(
        df, ma_window=ma_window, atr_window=atr_window, k_width=k_width,
        target_ret=target_ret, min_steps=min_steps, max_steps=max_steps, pricetick=pricetick,
        band_mode=band_mode,
    )
    mid = bounds["mid"].to_numpy(float)
    half = bounds["half"].to_numpy(float)
    steps = bounds["steps"].to_numpy(int)

    # 动作序列
    n_bars = end_idx - start_idx + 1
    if mode == "no_gate":
        actions = [0] * n_bars
    elif mode == "ma_gate":
        closes = df["close"].to_numpy(float)
        actions = [1 if closes[start_idx + k] < mid[start_idx + k] else 0 for k in range(n_bars)]
    elif actions is None:
        raise ValueError("mode='custom' 时必须提供 actions")
    actions = list(actions)
    if len(actions) < n_bars:
        actions = actions + [actions[-1]] * (n_bars - len(actions))

    rl = DynamicBandGrid(capital, fee_rate, initial_base_ratio, volume_per_grid,
                         reanchor_min_gap, reanchor_dev, min_steps, max_steps, pricetick)
    bench = DynamicBandGrid(capital, fee_rate, initial_base_ratio, volume_per_grid,
                            reanchor_min_gap, reanchor_dev, min_steps, max_steps, pricetick)

    close_arr = df["close"].to_numpy(float)
    ts_arr = df["timestamp"].to_numpy()

    # 首根 bar 收盘初始化
    close0 = float(close_arr[start_idx])
    rl.build_base(close0, start_idx, float(mid[start_idx]), float(half[start_idx]), int(steps[start_idx]))
    bench.build_base(close0, start_idx, float(mid[start_idx]), float(half[start_idx]), int(steps[start_idx]))

    navs, bench_navs, bhs, timestamps = [], [], [], []
    state_history = []

    nav_prev_rl = rl.nav(close0)
    nav_prev_bench = bench.nav(close0)
    timestamps.append(ts_arr[start_idx])
    navs.append(nav_prev_rl)
    bench_navs.append(nav_prev_bench)
    bh0 = capital * close0 / close0
    bhs.append(bh0)
    state_history.append(rl.state)

    for k in range(n_bars):
        i = start_idx + k
        close_i = float(close_arr[i])
        rl.apply_action(actions[k], close_i, i, float(mid[i]), float(half[i]), int(steps[i]), first_bar=(k == 0))
        if k > 0:
            bench.apply_action(0, close_i, i, float(mid[i]), float(half[i]), int(steps[i]), first_bar=False)
        j = i + 1
        if j > end_idx:
            break
        bar_j = df.iloc[j]
        close_j = float(close_arr[j])
        rl.process_fills(bar_j)
        bench.process_fills(bar_j)
        nav_rl = rl.nav(close_j)
        nav_bench = bench.nav(close_j)
        timestamps.append(ts_arr[j])
        navs.append(nav_rl)
        bench_navs.append(nav_bench)
        bhs.append(capital * close_j / close0)
        state_history.append(rl.state)
        nav_prev_rl, nav_prev_bench = nav_rl, nav_bench

    nav_series = pd.Series(navs, index=pd.to_datetime(timestamps))
    bench_series = pd.Series(bench_navs, index=pd.to_datetime(timestamps))
    bh_series = pd.Series(bhs, index=pd.to_datetime(timestamps))

    stats = _nav_stats(nav_series, capital)
    bench_stats = _nav_stats(bench_series, capital)
    bh_stats = _nav_stats(bh_series, capital)

    n_buys = sum(1 for f in rl.fills if f.side == "buy")
    n_sells = len(rl.fills) - n_buys
    switch = int(sum(1 for k in range(1, len(actions)) if actions[k] != actions[k - 1]))

    return {
        "timestamps": pd.to_datetime(timestamps),
        "nav": nav_series,
        "bench_nav": bench_series,
        "buyhold_nav": bh_series,
        "fills": rl.fills,
        "reanchor_count": rl.reanchor_count,
        "bench_reanchor_count": bench.reanchor_count,
        "state_history": state_history,
        "actions": actions,
        "stats": stats,
        "bench_stats": bench_stats,
        "bh_stats": bh_stats,
        "n_trades": len(rl.fills),
        "n_buys": n_buys,
        "n_sells": n_sells,
        "switch_count": switch,
    }


def _nav_stats(nav: pd.Series, capital: float) -> dict:
    nav = nav.dropna()
    if len(nav) < 2:
        return {}
    ret = nav.pct_change().dropna()
    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    hours = len(nav) - 1
    annual_return = (nav.iloc[-1] / nav.iloc[0]) ** (8760 / hours) - 1 if hours > 0 else 0.0
    return_std = float(ret.std() * np.sqrt(8760))
    sharpe = float(ret.mean() / ret.std() * np.sqrt(8760)) if ret.std() > 0 else 0.0
    dd = nav / nav.cummax() - 1
    max_dd = float(dd.min())
    calmar = float(annual_return / abs(max_dd)) if max_dd < 0 else 0.0
    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "return_std": return_std,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "calmar": calmar,
        "end_nav": float(nav.iloc[-1]),
    }


def get_feature_stats(
    df: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    ma_window: int = 60,
    atr_window: int = 14,
    k_width: float = 4.0,
    target_ret: float = 0.015,
    min_steps: int = 5,
    max_steps: int = 200,
    pricetick: float = 0.01,
    feature_ma_window: int = 1440,
    band_mode: str = "daily",
):
    """在 [start, end] 上计算特征均值/标准差，供 env 归一化复用。"""
    feat, _ = prepare_data(
        df, ma_window=ma_window, atr_window=atr_window, k_width=k_width,
        target_ret=target_ret, min_steps=min_steps, max_steps=max_steps, pricetick=pricetick,
        feature_ma_window=feature_ma_window, band_mode=band_mode,
    )
    df = df.reset_index(drop=True)
    mask = (df["timestamp"] >= pd.Timestamp(start)) & (df["timestamp"] <= pd.Timestamp(end))
    mask = mask.to_numpy()
    mean = np.nanmean(feat[FEATURE_COLS].to_numpy()[mask], axis=0)
    std = np.nanstd(feat[FEATURE_COLS].to_numpy()[mask], axis=0)
    return mean.astype(np.float32), np.maximum(std.astype(np.float32), 1e-6)
