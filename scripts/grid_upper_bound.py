"""静态网格收益上界实验引擎。

在理想无摩擦（fee_rate=0）、等距静态网格下，用经典"低买高卖"网格撮合，
测量网格套利的已实现收益，用于验证：

- 实验 A（固定每格交易量 q，纯多）：收益随间距 δ→0 逼近可实现上界；
- 实验 A2（对称多空）：允许做空，上下两腿都兑现，收益高于纯多；
- 实验 B（固定总资金 M）：每档均分现金 M/N，收益随 δ→0 反而趋于 0。

撮合逻辑镜像 scripts/dqn_gate_env.py 的 DynamicBandGrid：
- 卖单：bar.high >= price 成交，成交价 = max(bar.open, price)
- 买单：bar.low  <= price 成交，成交价 = min(bar.open, price)
- 同一 bar 内先卖后买，只做单次触发（不模拟 bar 内路径穿越）
- 当根新挂的单只从下一根 bar 起可成交

挂单逻辑：
- 纯多（allow_short=False）：起始价下方挂买单，不挂卖单，禁止做空；
- 多空（allow_short=True）：起始价下方挂买单、上方挂卖单，持仓可正可负。
- 买单成交于档位 i 后，在档位 i+1 挂等量卖单；卖单成交于档位 j 后，在档位 j-1 挂等量买单。

利润口径：
- 纯多：按"档位配对"结转（卖单与其来源买单作差），避免把浮动盈亏计入网格收益；
- 多空：用 long_lots / short_lots 两套 FIFO 队列配对结转，浮动盈亏另由 final_nav 体现。

用法：
    uv run python scripts/grid_upper_bound.py     # 运行内置自检
"""

from __future__ import annotations

import heapq
from collections import deque
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class StaticGridEngine:
    """等距静态网格撮合引擎。"""

    def __init__(self, lower, upper, spacing, initial_cash, q_fn, fee_rate=0.0,
                 fill_at_level=True, allow_short=False):
        self.lower = float(lower)
        self.upper = float(upper)
        self.spacing = float(spacing)
        self.initial_cash = float(initial_cash)
        self.q_fn = q_fn
        self.fee_rate = float(fee_rate)
        # True : 成交价取网格档位价（理想连续模型，每次往返恰好赚 δ）
        # False: 成交价取 min/max(open, price)（现有 DynamicBandGrid 约定，
        #        会在 OHLC 粒度下把跳空也算作利润，收益可超过 TV，非理想上界）
        self.fill_at_level = bool(fill_at_level)
        # True : 对称挂单多空网格，持仓可正可负，FIFO 配对
        # False: 纯现金做多网格
        self.allow_short = bool(allow_short)
        self.n_levels = int(round((self.upper - self.lower) / self.spacing)) + 1
        self.prices = [self.lower + i * self.spacing for i in range(self.n_levels)]
        self._seq = 0

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def run(self, df: pd.DataFrame) -> dict:
        start_close = float(df.iloc[0]["close"])

        sell_heap: list = []  # (price, seq, level, vol, cost) 小顶堆，cost=产生该卖单的买价
        buy_heap: list = []   # (-price, seq, level, vol) 大顶堆
        for i, p in enumerate(self.prices):
            if p < start_close:
                heapq.heappush(buy_heap, (-p, self._next_seq(), i, self.q_fn(p)))
            elif self.allow_short and p > start_close:
                heapq.heappush(sell_heap, (p, self._next_seq(), i, self.q_fn(p), 0.0))

        cash = self.initial_cash
        position = 0.0
        realized = 0.0
        long_lots: deque[list[float]] = deque()   # [price, qty]，多空模式 FIFO
        short_lots: deque[list[float]] = deque()
        peak_long = 0.0
        peak_short = 0.0
        min_cash = cash
        navs: list[float] = []
        buys = sells = 0
        exhausted = False

        def close_fifo(lots: deque[list[float]], price: float, vol: float, *, is_long: bool) -> float:
            """按 FIFO 平掉 lots，返回未配对余量；已实现利润累加进 realized。"""
            nonlocal realized
            rem = vol
            while rem > 1e-12 and lots:
                lot = lots[0]
                m = min(rem, lot[1])
                realized += m * ((price - lot[0]) if is_long else (lot[0] - price))
                rem -= m
                if lot[1] - m > 1e-12:
                    lot[1] -= m
                else:
                    lots.popleft()
            return rem

        for t in range(1, len(df)):
            bar = df.iloc[t]
            o, h, l, c = (
                float(bar["open"]),
                float(bar["high"]),
                float(bar["low"]),
                float(bar["close"]),
            )
            new_buys: list[tuple[int, float]] = []
            new_sells: list[tuple[int, float, float]] = []

            while sell_heap and sell_heap[0][0] <= h:
                price, seq, lvl, vol, cost = heapq.heappop(sell_heap)
                fp = price if self.fill_at_level else max(o, price)
                if self.allow_short:
                    rem = close_fifo(long_lots, fp, vol, is_long=True)
                    if rem > 1e-12:
                        short_lots.append([fp, rem])
                    position -= vol
                    cash += vol * fp * (1 - self.fee_rate)
                    sells += 1
                else:
                    if position + 1e-9 < vol:
                        heapq.heappush(sell_heap, (price, seq, lvl, vol, cost))
                        break
                    realized += vol * (fp - cost)
                    position -= vol
                    cash += vol * fp * (1 - self.fee_rate)
                    sells += 1
                if lvl - 1 >= 0:
                    new_buys.append((lvl - 1, vol))

            peak_short = max(peak_short, -position)

            while buy_heap and -buy_heap[0][0] >= l:
                neg, seq, lvl, vol = heapq.heappop(buy_heap)
                fp = -neg if self.fill_at_level else min(o, -neg)
                if self.allow_short:
                    rem = close_fifo(short_lots, fp, vol, is_long=False)
                    if rem > 1e-12:
                        long_lots.append([fp, rem])
                    position += vol
                    cash -= vol * fp * (1 + self.fee_rate)
                    buys += 1
                else:
                    if cash + 1e-9 < vol * fp * (1 + self.fee_rate):
                        exhausted = True
                        heapq.heappush(buy_heap, (neg, seq, lvl, vol))
                        break
                    cash -= vol * fp * (1 + self.fee_rate)
                    position += vol
                    buys += 1
                if lvl + 1 < self.n_levels:
                    new_sells.append((lvl + 1, vol, fp))

            peak_long = max(peak_long, position)

            for lvl, vol in new_buys:
                p = self.prices[lvl]
                heapq.heappush(buy_heap, (-p, self._next_seq(), lvl, vol))
            for lvl, vol, cost in new_sells:
                p = self.prices[lvl]
                heapq.heappush(sell_heap, (p, self._next_seq(), lvl, vol, cost))

            min_cash = min(min_cash, cash)
            navs.append(cash + position * c)

        last_close = float(df.iloc[-1]["close"])
        metrics = {
            "spacing": self.spacing,
            "n_levels": self.n_levels,
            "buys": buys,
            "sells": sells,
            "realized_profit": realized,
            "final_position": position,
            "final_cash": cash,
            "final_nav": cash + position * last_close,
            "peak_position": peak_long if not self.allow_short else max(peak_long, peak_short),
            "peak_long": peak_long,
            "peak_short": peak_short,
            "max_cash_deployed": self.initial_cash - min_cash,
            "capital_exhausted": exhausted,
            "initial_cash": self.initial_cash,
        }
        return {"metrics": metrics, "navs": navs}


def run_experiment(df, *, lower, upper, spacing, mode, initial_cash, q_unit=1.0, fee_rate=0.0,
                   fill_at_level=True, allow_short=False):
    """运行单组实验。

    mode="fixed_q"       : 每档固定数量 q_unit（实验 A / A2）
    mode="fixed_capital" : 每档均分现金 initial_cash/N（实验 B）
    """
    n_levels = int(round((upper - lower) / spacing)) + 1
    if mode == "fixed_q":
        q_fn = lambda p: q_unit  # noqa: E731
    elif mode == "fixed_capital":
        cash_per_level = initial_cash / n_levels
        q_fn = lambda p: cash_per_level / p  # noqa: E731
    else:
        raise ValueError(f"unknown mode: {mode}")

    engine = StaticGridEngine(lower, upper, spacing, initial_cash, q_fn, fee_rate,
                              fill_at_level, allow_short)
    res = engine.run(df)
    res["metrics"]["mode"] = mode
    res["metrics"]["allow_short"] = allow_short
    return res


def _demo() -> None:
    """纯多自检：收盘路径 100 → 110 → 90 → 100，δ=10，q=1。"""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2022-01-01", periods=4, freq="h"),
            "open": [100.0, 100.0, 110.0, 90.0],
            "high": [100.0, 110.0, 110.0, 100.0],
            "low": [100.0, 100.0, 90.0, 90.0],
            "close": [100.0, 110.0, 90.0, 100.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
        }
    )
    res = run_experiment(
        df, lower=90.0, upper=110.0, spacing=10.0, mode="fixed_q", initial_cash=1000.0
    )
    m = res["metrics"]
    assert m["n_levels"] == 3, m
    assert m["buys"] == 1 and m["sells"] == 1, m
    assert abs(m["realized_profit"] - 10.0) < 1e-9, m
    assert abs(m["final_position"]) < 1e-9, m
    assert abs(m["final_nav"] - 1010.0) < 1e-9, m
    print("demo 纯多 ok:", {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})


def _demo_long_short() -> None:
    """多空自检：同一条路径对称挂单，应上下两腿各赚一个 δ，realized=20=TV/2。

    收盘路径 100→110→90→100，TV=40，δ=10，q=1：
      上穿 110 开空 -> 下穿 100 平空赚 10；下穿 90 开多 -> 上穿 100 平多赚 10。
    """
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2022-01-01", periods=4, freq="h"),
            "open": [100.0, 100.0, 110.0, 90.0],
            "high": [100.0, 110.0, 110.0, 100.0],
            "low": [100.0, 100.0, 90.0, 90.0],
            "close": [100.0, 110.0, 90.0, 100.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
        }
    )
    res = run_experiment(
        df, lower=90.0, upper=110.0, spacing=10.0, mode="fixed_q",
        initial_cash=1000.0, allow_short=True,
    )
    m = res["metrics"]
    assert m["buys"] == 2 and m["sells"] == 2, m
    assert abs(m["realized_profit"] - 20.0) < 1e-9, m
    assert abs(m["final_position"]) < 1e-9, m
    assert abs(m["final_nav"] - 1020.0) < 1e-9, m
    assert abs(m["peak_long"] - 1.0) < 1e-9 and abs(m["peak_short"] - 1.0) < 1e-9, m
    print("demo 多空 ok:", {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})


if __name__ == "__main__":
    _demo()
    _demo_long_short()