"""分析"时频策略"（七币 DQN 门控网格 + 季度风险平价组合）从 2026-08-01 起运行的可能收益与交易摩擦。

口径（与 portfolio_optimize_7coins.py 保持一致）
- 币种：BTC/ETH/BNB/PAXG/SOL/TRX/XLM，1h 决策，A1 账户状态，模型来自 models/
- 权重：2026-06-30 季度风险平价调仓所得目标权重（即 2026Q3 在管权重，无前视）
- 初始资金 100,000；每格名义 = 段初资金 × 1%
- 手续费默认 0.1%/单边；摩擦 = 净收益(0.1%)与毛收益(0%)之差
- 区间：2026-08-01 00:00 ~ 最后一个已收盘小时 bar
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.dqn_gate_env as g  # noqa: E402
from stable_baselines3 import DQN  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack  # noqa: E402

COINS = [
    ("BTC", 0.01),
    ("ETH", 0.01),
    ("BNB", 0.01),
    ("PAXG", 0.01),
    ("SOL", 0.001),
    ("TRX", 0.0001),
    ("XLM", 0.0001),
]
COIN_NAMES = [c for c, _ in COINS]

LAUNCH = "2026-08-01 00:00:00"
CAPITAL = 100_000.0
PER_GRID_RATIO = 0.01
FEE = 0.001
N_STACK = 4

TRAIN_START = "2022-01-01"
TRAIN_END = "2025-12-31 23:00:00"

MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


def load_data() -> dict:
    out = {}
    for coin, tick in COINS:
        df = pd.read_csv(DATA_DIR / f"{coin.lower()}_usdt_1h.csv", parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        out[coin] = {"df": df, "pricetick": tick}
    return out


def standing_weights(data: dict) -> dict:
    """2026-06-30 风险平价调仓后的目标权重（Q3 在管，无前视）。"""
    tr = pd.read_csv(DATA_DIR / "portfolio_7coins_trades.csv", parse_dates=["date"])
    row = tr[tr["date"] == pd.Timestamp("2026-06-30 23:00:00")].iloc[-1]
    w = {c: float(row[f"w_new_{c}"]) for c in COIN_NAMES}
    assert abs(sum(w.values()) - 1.0) < 1e-6, "权重和必须为 1"
    return w


def latest_closed_hour() -> pd.Timestamp:
    """返回最后一个已收盘（可交易）的小时 bar 时间戳（naive UTC）。"""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return (pd.Timestamp(now).floor("h") - pd.Timedelta(hours=1))


def build_env(df, start, end, feature_mean, feature_std, pricetick, capital, fee_rate):
    return g.GridGateEnv(
        df, start, end,
        feature_mean=feature_mean, feature_std=feature_std,
        ma_window=60, atr_window=14, k_width=4.0, target_ret=0.015,
        min_steps=5, max_steps=200, reanchor_min_gap=168, reanchor_dev=0.25,
        volume_per_grid=capital * PER_GRID_RATIO / float(df[df["timestamp"] >= pd.Timestamp(start)]["close"].iloc[0]),
        fee_rate=fee_rate, initial_base_ratio=0.5,
        pricetick=pricetick, band_mode="daily",
        switch_penalty=0.0, decision_interval=1,
        use_account_state=True,
        account_features=["position_ratio", "cash_ratio"],
        capital=capital,
    )


def run_coin(coin, tick, data, model, mean, std, start, end, capital, fee_rate):
    """在 [start,end] 用给定费率跑一次 DQN 门控网格，返回 nav / fills 等。"""
    df = data["df"]
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    seg_start_close = float(df[df["timestamp"] >= start_ts]["close"].iloc[0])

    def make_env():
        return g.GridGateEnv(
            df, start, end,
            feature_mean=mean, feature_std=std,
            ma_window=60, atr_window=14, k_width=4.0, target_ret=0.015,
            min_steps=5, max_steps=200, reanchor_min_gap=168, reanchor_dev=0.25,
            volume_per_grid=capital * PER_GRID_RATIO / seg_start_close,
            fee_rate=FEE, initial_base_ratio=0.5,
            pricetick=tick, band_mode="daily",
            switch_penalty=0.0, decision_interval=1,
            use_account_state=True,
            account_features=["position_ratio", "cash_ratio"],
            capital=capital,
        )

    n_bars = int(((df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)).sum())
    ev = DummyVecEnv([make_env]); ev = VecFrameStack(ev, n_stack=N_STACK)
    obs = ev.reset()
    actions = []
    for _ in range(n_bars):
        a_arr, _ = model.predict(obs, deterministic=True)
        a = int(a_arr[0]); actions.append(a)
        obs, r, done, info = ev.step([a])

    res = g.simulate(
        df, start, end, actions=actions,
        ma_window=60, atr_window=14, k_width=4.0, target_ret=0.015,
        min_steps=5, max_steps=200, reanchor_min_gap=168, reanchor_dev=0.25,
        volume_per_grid=capital * PER_GRID_RATIO / seg_start_close,
        fee_rate=fee_rate, initial_base_ratio=0.5,
        pricetick=tick, band_mode="daily", capital=capital,
    )
    return res


def nav_stats(nav: pd.Series) -> dict:
    nav = nav.dropna()
    if len(nav) < 2:
        return {}
    ret = nav.pct_change().dropna()
    total = nav.iloc[-1] / nav.iloc[0] - 1
    hours = len(nav) - 1
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (8760 / hours) - 1 if hours > 0 else 0.0
    vol = float(ret.std() * np.sqrt(8760))
    dd = float((nav / nav.cummax() - 1).min())
    sharpe = float(ret.mean() / ret.std() * np.sqrt(8760)) if ret.std() > 0 else 0.0
    calmar = float(ann / abs(dd)) if dd < 0 else 0.0
    return {
        "total_return": float(total), "annual_return": float(ann),
        "return_std": vol, "max_dd": dd, "sharpe": sharpe, "calmar": calmar,
        "end_nav": float(nav.iloc[-1]),
    }


def buyhold_nav(data, weights, start, end):
    """按风险平价权重买入持有基准。"""
    parts = []
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    for coin in COIN_NAMES:
        df = data[coin]["df"]
        df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
        px = df.set_index("timestamp")["close"]
        c0 = px.iloc[0]
        parts.append(CAPITAL * weights[coin] / c0 * px)
    return pd.concat(parts, axis=1).sum(axis=1)


def main():
    end = latest_closed_hour()
    # 与各币数据对齐，取每个币最后一个 <= end 的 bar
    data = load_data()
    per_coin_last = {c: pd.Timestamp(d["df"]["timestamp"].max()) for c, d in data.items()}
    closed = min(pd.Timestamp(end), *per_coin_last.values())
    end_s = closed.strftime("%Y-%m-%d %H:%M:%S")
    print(f"区间: {LAUNCH} ~ {end_s}  ({closed - pd.Timestamp(LAUNCH)})", flush=True)
    print("每币最后bar:", {c: str(t) for c, t in per_coin_last.items()}, flush=True)

    weights = standing_weights(data)
    print("风险平价权重(Q3):", {c: round(w, 5) for c, w in weights.items()}, flush=True)

    models, feature_stats = {}, {}
    for coin, tick in COINS:
        models[coin] = DQN.load(str(MODEL_DIR / f"dqn_gate_{coin.lower()}_a1_1h.zip"))
        feature_stats[coin] = g.get_feature_stats(
            data[coin]["df"], TRAIN_START, TRAIN_END,
            ma_window=60, atr_window=14, k_width=4.0, target_ret=0.015,
            min_steps=5, max_steps=200, pricetick=tick, band_mode="daily",
        )

    # 每币：先按 0.1% 环境推理动作（实盘口径），再以 0.1% / 0% 费率各撮合一次
    nav_net, nav_gross = {}, {}
    rows = []
    t0 = time.time()
    for coin, tick in COINS:
        cap = CAPITAL * weights[coin]
        st = time.time()
        res_net = run_coin(coin, tick, data[coin], models[coin],
                           feature_stats[coin][0], feature_stats[coin][1],
                           LAUNCH, end_s, cap, FEE)
        res_gross = run_coin(coin, tick, data[coin], models[coin],
                             feature_stats[coin][0], feature_stats[coin][1],
                             LAUNCH, end_s, cap, 0.0)
        nav_net[coin] = res_net["nav"]
        nav_gross[coin] = res_gross["nav"]

        n_buy = sum(1 for f in res_net["fills"] if f.side == "buy")
        notional = sum(f.price * f.volume for f in res_net["fills"])
        drag_usd = float(res_gross["stats"]["end_nav"]) - float(res_net["stats"]["end_nav"])
        rows.append({
            "coin": coin, "cap_start": cap,
            "n_fills": len(res_net["fills"]), "n_buy": n_buy,
            "n_sell": len(res_net["fills"]) - n_buy,
            "notional_one_way": notional,
            "grid_fee_est": notional * FEE,
            "drag_usd": drag_usd,
            "gross_ret": float(res_gross["stats"]["total_return"]),
            "net_ret": float(res_net["stats"]["total_return"]),
            "end_nav_net": float(res_net["stats"]["end_nav"]),
            "end_nav_gross": float(res_gross["stats"]["end_nav"]),
            "reanchor": res_net["reanchor_count"],
            "switch": res_net["switch_count"],
        })
        print(f"{coin}: cap={cap:,.0f} 成交{len(res_net['fills'])}笔 "
              f"净{res_net['stats']['total_return']*100:.2f}% 毛{res_gross['stats']['total_return']*100:.2f}% "
              f"({time.time()-st:.0f}s)", flush=True)

    # 组合净值
    idx = sorted(set.intersection(*[set(s.index) for s in nav_net.values()]))
    net_df = pd.DataFrame({c: nav_net[c].reindex(idx).ffill() for c in COIN_NAMES})
    gross_df = pd.DataFrame({c: nav_gross[c].reindex(idx).ffill() for c in COIN_NAMES})
    port_net = net_df.sum(axis=1)
    port_gross = gross_df.sum(axis=1)

    # 基准：同权重买入持有
    bh = buyhold_nav(data, weights, LAUNCH, end_s)
    bh = bh.reindex(idx).ffill()

    friction_usd = float(port_gross.iloc[-1] - port_net.iloc[-1])
    days = (idx[-1] - idx[0]).total_seconds() / 86400

    metrics_all = {}
    for label, nav in [("portfolio_net", port_net), ("portfolio_gross", port_gross), ("buyhold", bh)]:
        metrics_all[label] = nav_stats(nav)

    per = pd.DataFrame(rows)
    per["drag_bps_cap"] = per["drag_usd"] / per["cap_start"] * 1e4
    per["notional_turnover"] = per["notional_one_way"] / per["cap_start"]
    per["fills_per_day"] = per["n_fills"] / days

    print("\n--------------- 收益 ---------------")
    print(f"{'':16s} {'总收益%':>9s} {'年化%':>9s} {'年化波动%':>9s} {'最大回撤%':>9s} {'Sharpe':>7s} {'期末NAV':>11s}")
    for label, nav in [("组合(0.1%费率)", port_net), ("组合(无费率)", port_gross), ("买入持有(同权)", bh)]:
        m = nav_stats(nav)
        print(f"{label:16s} {m['total_return']*100:8.2f}% {m['annual_return']*100:8.2f}% "
              f"{m['return_std']*100:8.2f}% {m['max_dd']*100:8.2f}% {m['sharpe']:7.2f} {m['end_nav']:11,.0f}")

    print("\n8月日历月(08-01~08-31):")
    for label, nav in [("组合(0.1%费率)", port_net[port_net.index <= pd.Timestamp("2026-08-31 23:00:00")]),
                       ("买入持有", bh[bh.index <= pd.Timestamp("2026-08-31 23:00:00")])]:
        m = nav_stats(nav)
        print(f"  {label:18s} 总收益 {m['total_return']*100:6.2f}%  期末NAV {m['end_nav']:,.0f}")

    print("\n--------------- 交易摩擦 ---------------")
    print(f"窗口天数 {days:.1f} | 组合摩擦成本(毛-净): {friction_usd:,.0f} USDT "
          f"≈ {friction_usd/port_net.iloc[0]*1e4:.0f} bp (按期初资金)")
    print(f"\n{'币':6s} {'期初资金':>10s} {'成交笔':>6s} {'买/卖':>8s} {'单边成交额':>12s} {'单边换手':>8s} "
          f"{'净收益%':>8s} {'毛收益%':>8s} {'摩擦USDT':>10s} {'摩擦bp':>6s} {'笔/日':>6s}")
    for r in per.itertuples(index=False):
        print(f"{r.coin:6s} {r.cap_start:10,.0f} {r.n_fills:6d} {r.n_buy:>4d}/{r.n_sell:<4d} "
              f"{r.notional_one_way:12,.0f} {r.notional_turnover:7.2f}x "
              f"{r.net_ret*100:8.2f}% {r.gross_ret*100:8.2f}% "
              f"{r.drag_usd:10,.0f} {r.drag_bps_cap:6.0f} {r.fills_per_day:6.2f}")

    print("\n费率敏感度(摩擦近似按费率线性缩放):")
    for fr, lbl in [(0.0002, "0.02%/边(maker优惠)"), (0.0005, "0.05%/边"), (0.001, "0.1%/边(默认)")]:
        est = friction_usd * (fr / FEE)
        print(f"  费率 {lbl:20s} 摩擦≈{est:,.0f} USDT ≈ {est/port_net.iloc[0]*1e4:.0f} bp "
              f"-> 期间净收益≈{(port_gross.iloc[-1]-est)/port_net.iloc[0]*100-100:.2f}%")

    # 保存
    out = pd.DataFrame({"timestamp": idx, "portfolio_net": port_net,
                        "portfolio_gross": port_gross, "buyhold": bh})
    out.to_csv(DATA_DIR / "aug_run_nav.csv", index=False)
    per.to_csv(DATA_DIR / "aug_run_friction.csv", index=False)
    with open(DATA_DIR / "aug_run_metrics.json", "w") as f:
        json.dump(metrics_all, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: data/aug_run_nav.csv, data/aug_run_friction.csv, data/aug_run_metrics.json | 耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
