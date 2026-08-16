"""七币种 DQN 门控策略的季度风险平价组合回测。

- 资产：BTC/ETH/BNB/PAXG/SOL/TRX/XLM，每个资产用 DQN 门控动态带网格策略
  （模型来自 models/dqn_gate_{coin}_a1_1h.zip，A1 账户状态，1h 决策）
- 组合：每季度末做风险平价（ERC）优化，优化器使用 cvxopt
- 权重输入：七币种原生价格日收益率的 EWMA 协方差（无前视，只用 ≤ 调仓日数据）
- 运行区间：2026-01-01 ~ 2026-07-31，季度调仓 03-31 / 06-30
- 费用：调仓 0.1% 单边，按 |Δ资金| 计
- 保存运行数据到 data/portfolio_7coins_*.csv / json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.dqn_gate_env as g  # noqa: E402
from stable_baselines3 import DQN  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack  # noqa: E402

solvers.options["show_progress"] = False

# ============================================================
# 参数
# ============================================================
COINS = [
    ("BTC", 0.01),
    ("ETH", 0.01),
    ("BNB", 0.01),
    ("PAXG", 0.01),
    ("SOL", 0.001),
    ("TRX", 0.0001),
    ("XLM", 0.0001),
]
CAPITAL = 100_000.0
FEE = 0.001
PER_GRID_RATIO = 0.01      # 每格名义金额 / 段初资金
LAMBDA = 0.98              # EWMA 衰减
W_UPPER = 0.40             # 单资产权重上限
W_LOWER = 0.0
N_STACK = 4

TRAIN_START = "2022-01-01"
TRAIN_END = "2025-12-31 23:00:00"
VAL_START = "2026-01-01"
VAL_END = "2026-07-31 23:00:00"

MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# ============================================================
# 数据与模型加载
# ============================================================
def load_data() -> dict:
    out = {}
    for coin, tick in COINS:
        df = pd.read_csv(DATA_DIR / f"{coin.lower()}_usdt_1h.csv", parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        out[coin] = {"df": df, "pricetick": tick}
    return out


def daily_returns(data: dict, end: str) -> pd.DataFrame:
    """七币种原生价格日收益率（≤ end 的自然日数据，无前视）。"""
    closes = {}
    for coin, d in data.items():
        df = d["df"]
        df = df[df["timestamp"] <= end]
        daily = df.set_index("timestamp").resample("1D").agg({"close": "last"}).dropna()
        closes[coin] = daily["close"]
    px = pd.DataFrame(closes)
    return px.pct_change().dropna()


def ewma_covariance(returns: pd.DataFrame, lam: float) -> np.ndarray:
    """EWMA 协方差（从 returns 起点递推到末尾）。"""
    n = returns.shape[1]
    cov = np.zeros((n, n))
    for _, r in returns.iterrows():
        rv = r.values.astype(float)
        cov = lam * cov + (1 - lam) * np.outer(rv, rv)
    return cov


# ============================================================
# cvxopt 风险平价
# ============================================================
def _solve_erc_unconstrained(cov: np.ndarray) -> np.ndarray:
    """cvxopt 求解无约束 ERC 对数障碍问题并归一化。

    min  0.5 w'Σw − Σ b_i ln(w_i),  b_i = 1/n
    KKT 条件等价于 w_i(Σw)_i = b_i σ²(w)，即等风险贡献。
    """
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2 + np.eye(cov.shape[0]) * 1e-10
    n = cov.shape[0]
    b = np.ones(n) / n
    S = matrix(cov)

    inv_vol = 1.0 / np.sqrt(np.maximum(np.diag(cov), 1e-12))
    x0 = inv_vol / inv_vol.sum()

    def F(x=None, z=None):
        if x is None:
            return 0, matrix(x0)
        w = np.array(x).flatten()
        if np.any(w <= 0):
            return None
        f0 = 0.5 * w @ cov @ w - float(np.sum(b * np.log(w)))
        Df = (cov @ w - b / w).reshape(1, -1)
        f = matrix([f0])
        Df = matrix(Df)
        if z is None:
            return f, Df
        H = cov + np.diag(b / (w * w))
        return f, Df, matrix(H)

    sol = solvers.cp(F)
    w = np.array(sol["x"]).flatten()
    if np.any(np.isnan(w)) or np.any(w <= 0):
        raise RuntimeError("cvxopt ERC 求解失败")
    return w / w.sum()


def risk_parity_weights_cvxopt(cov: np.ndarray, upper: float = W_UPPER,
                               lower: float = W_LOWER) -> np.ndarray:
    """等风险贡献权重，cvxopt 求解；权重上限采用递归封顶法。

    1. 用 cvxopt 求无约束 ERC；
    2. 对超过 upper 的资产封顶，并把剩余风险预算在其余资产上重求 ERC；
    3. 重复直到无超限。
    """
    cov = np.asarray(cov, dtype=float)
    w = _solve_erc_unconstrained(cov)
    for _ in range(10):
        over = w > upper
        if not over.any():
            break
        fixed = float(upper * over.sum())
        w[over] = upper
        remain = ~over
        if remain.sum() == 0 or fixed >= 1.0:
            break
        budget = 1.0 - fixed
        w_sub = _solve_erc_unconstrained(cov[np.ix_(remain, remain)]) * budget
        w[remain] = w_sub
    w = np.clip(w, lower, upper)
    return w / w.sum()


# ============================================================
# 单资产策略段
# ============================================================
def run_strategy_segment(data: dict, coin: str, model, feature_mean, feature_std,
                         start: str, end: str, capital: float) -> pd.Series:
    """在 [start, end] 段内，以 capital 运行 DQN 门控网格策略，返回逐小时净值。"""
    df = data[coin]["df"]
    tick = data[coin]["pricetick"]
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    seg_start_close = float(df[df["timestamp"] >= start_ts]["close"].iloc[0])
    volume_per_grid = capital * PER_GRID_RATIO / seg_start_close

    def make_env():
        return g.GridGateEnv(
            df, start, end,
            feature_mean=feature_mean, feature_std=feature_std,
            ma_window=60, atr_window=14, k_width=4.0, target_ret=0.015,
            min_steps=5, max_steps=200, reanchor_min_gap=168, reanchor_dev=0.25,
            volume_per_grid=volume_per_grid, fee_rate=FEE, initial_base_ratio=0.5,
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
        volume_per_grid=volume_per_grid, fee_rate=FEE, initial_base_ratio=0.5,
        pricetick=tick, band_mode="daily",
        capital=capital,
    )
    return res["nav"]


# ============================================================
# 组合回放
# ============================================================
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


def run_portfolio(data: dict, models: dict, feature_stats: dict,
                  start: str, end: str) -> dict:
    """季度风险平价组合回放。"""
    quarters = []
    seg_start = pd.Timestamp(start)
    for qe in [pd.Timestamp("2026-03-31 23:00:00"), pd.Timestamp("2026-06-30 23:00:00")]:
        if seg_start < qe:
            quarters.append((seg_start, qe))
            seg_start = qe + pd.Timedelta(hours=1)
    if seg_start <= pd.Timestamp(end):
        quarters.append((seg_start, pd.Timestamp(end)))

    # 初始权重：EWMA 协方差 ≤ 2025-12-31
    ret_df = daily_returns(data, TRAIN_END)
    cov = ewma_covariance(ret_df, LAMBDA)
    w = risk_parity_weights_cvxopt(cov)

    capital = CAPITAL
    nav_total = []
    nav_by_coin = []
    weights_log = []
    trades_log = []
    segment_log = []

    for si, (s0, s1) in enumerate(quarters):
        s0s, s1s = s0.strftime("%Y-%m-%d %H:%M:%S"), s1.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n--- 段 {si+1}: {s0s} ~ {s1s} ---", flush=True)
        print("权重:", {c: round(w[i], 4) for i, c in enumerate(COINS_NAMES)}, flush=True)

        seg_cap = {c: capital * w[i] for i, c in enumerate(COINS_NAMES)}
        seg_navs = {}
        for c in COINS_NAMES:
            t0 = time.time()
            nav = run_strategy_segment(data, c, models[c], feature_stats[c][0], feature_stats[c][1],
                                       s0s, s1s, seg_cap[c])
            seg_navs[c] = nav
            print(f"  {c}: cap={seg_cap[c]:,.0f} end_nav={nav.iloc[-1]:,.0f} ({time.time()-t0:.1f}s)", flush=True)

        # 合并为组合净值
        idx = sorted(set.intersection(*[set(s.index) for s in seg_navs.values()]))
        combo = pd.DataFrame({c: seg_navs[c].reindex(idx).ffill() for c in COINS_NAMES})
        total = combo.sum(axis=1)

        for d in idx:
            row = {"timestamp": d, "nav": total.loc[d]}
            for c in COINS_NAMES:
                row[c] = combo[c].loc[d]
            nav_by_coin.append(row)
        nav_total.append(total)

        # 段末记录
        v_end = combo.iloc[-1]
        V_total = float(v_end.sum())
        weights_log.append({
            "date": s1s, **{c: float(w[i]) for i, c in enumerate(COINS_NAMES)}
        })
        segment_log.append({
            "segment": si + 1, "start": s0s, "end": s1s,
            "capital_start": capital,
            **{f"end_nav_{c}": float(v_end[c]) for c in COINS_NAMES},
            "end_nav_total": V_total,
        })

        if si < len(quarters) - 1:
            # 调仓：EWMA 协方差 ≤ 段末
            ret_df = daily_returns(data, s1s)
            cov = ewma_covariance(ret_df, LAMBDA)
            w_new = risk_parity_weights_cvxopt(cov)
            delta = V_total * w_new - v_end.values
            fee = FEE * float(np.abs(delta).sum())
            next_capital = V_total - fee
            trades_log.append({
                "date": s1s, "v_total": V_total, "fee": fee,
                "next_capital": next_capital,
                **{f"w_old_{c}": float(w[i]) for i, c in enumerate(COINS_NAMES)},
                **{f"w_new_{c}": float(w_new[i]) for i, c in enumerate(COINS_NAMES)},
                **{f"delta_{c}": float(delta[i]) for i, c in enumerate(COINS_NAMES)},
            })
            print(f"  调仓: V={V_total:,.0f} fee={fee:,.0f} next_cap={next_capital:,.0f}", flush=True)
            w = w_new
            capital = next_capital

    nav = pd.concat(nav_total).sort_index()
    nav = nav[~nav.index.duplicated(keep="last")]
    return {
        "nav": nav,
        "nav_by_coin": pd.DataFrame(nav_by_coin).set_index("timestamp").sort_index(),
        "weights": pd.DataFrame(weights_log),
        "trades": pd.DataFrame(trades_log),
        "segments": pd.DataFrame(segment_log),
    }


# ============================================================
# 基准
# ============================================================
def equal_weight_dqn(data, models, feature_stats, start, end):
    """等权七策略，季度再平衡。"""
    w = np.ones(len(COINS)) / len(COINS)
    capital = CAPITAL
    quarters = []
    seg_start = pd.Timestamp(start)
    for qe in [pd.Timestamp("2026-03-31 23:00:00"), pd.Timestamp("2026-06-30 23:00:00")]:
        if seg_start < qe:
            quarters.append((seg_start, qe))
            seg_start = qe + pd.Timedelta(hours=1)
    if seg_start <= pd.Timestamp(end):
        quarters.append((seg_start, pd.Timestamp(end)))

    nav_parts = []
    for s0, s1 in quarters:
        s0s = s0.strftime("%Y-%m-%d %H:%M:%S"); s1s = s1.strftime("%Y-%m-%d %H:%M:%S")
        seg_cap = {c: capital * w[i] for i, c in enumerate(COINS_NAMES)}
        seg_navs = {}
        for c in COINS_NAMES:
            seg_navs[c] = run_strategy_segment(data, c, models[c], feature_stats[c][0], feature_stats[c][1],
                                               s0s, s1s, seg_cap[c])
        idx = sorted(set.intersection(*[set(s.index) for s in seg_navs.values()]))
        combo = pd.DataFrame({c: seg_navs[c].reindex(idx).ffill() for c in COINS_NAMES})
        total = combo.sum(axis=1)
        nav_parts.append(total)
        if s1 < pd.Timestamp(end):
            V_total = float(total.iloc[-1])
            delta = V_total * w - combo.iloc[-1].values
            fee = FEE * float(np.abs(delta).sum())
            capital = V_total - fee
    return pd.concat(nav_parts).sort_index()


def equal_weight_buyhold(data, start, end):
    """期初等权买入七币持有。"""
    closes = {}
    for c in COINS_NAMES:
        df = data[c]["df"]
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
        closes[c] = df.set_index("timestamp")["close"]
    px = pd.DataFrame(closes).sort_index().ffill().dropna()
    c0 = {c: float(px[c].iloc[0]) for c in COINS_NAMES}
    shares = {c: (CAPITAL / len(COINS)) / c0[c] for c in COINS_NAMES}
    nav = sum(px[c] * shares[c] for c in COINS_NAMES)
    return nav


# ============================================================
# 主流程
# ============================================================
def main():
    global COINS_NAMES
    COINS_NAMES = [c for c, _ in COINS]

    t0 = time.time()
    data = load_data()
    print("数据加载完成", flush=True)

    models = {}
    feature_stats = {}
    for coin, tick in COINS:
        models[coin] = DQN.load(str(MODEL_DIR / f"dqn_gate_{coin.lower()}_a1_1h.zip"))
        feature_stats[coin] = g.get_feature_stats(
            data[coin]["df"], TRAIN_START, TRAIN_END,
            ma_window=60, atr_window=14, k_width=4.0, target_ret=0.015,
            min_steps=5, max_steps=200, pricetick=tick, band_mode="daily",
        )
        print(f"{coin} 模型/特征就绪", flush=True)

    print("\n===== 风险平价组合回放 =====", flush=True)
    rp = run_portfolio(data, models, feature_stats, VAL_START, VAL_END)

    print("\n===== 等权 DQN 策略组合（季度再平衡） =====", flush=True)
    eq_nav = equal_weight_dqn(data, models, feature_stats, VAL_START, VAL_END)

    print("\n===== 等权 Buy&Hold =====", flush=True)
    bh_nav = equal_weight_buyhold(data, VAL_START, VAL_END)

    # 保存数据
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rp["nav"].to_csv(DATA_DIR / "portfolio_7coins_nav.csv", index_label="timestamp")
    rp["nav_by_coin"].to_csv(DATA_DIR / "portfolio_7coins_nav_by_coin.csv", index_label="timestamp")
    rp["weights"].to_csv(DATA_DIR / "portfolio_7coins_weights.csv", index=False)
    rp["trades"].to_csv(DATA_DIR / "portfolio_7coins_trades.csv", index=False)
    rp["segments"].to_csv(DATA_DIR / "portfolio_7coins_segments.csv", index=False)
    eq_nav.to_csv(DATA_DIR / "portfolio_7coins_equalweight_dqn_nav.csv", index_label="timestamp")
    bh_nav.to_csv(DATA_DIR / "portfolio_7coins_equalweight_bh_nav.csv", index_label="timestamp")

    metrics = {
        "risk_parity": nav_stats(rp["nav"]),
        "equal_weight_dqn": nav_stats(eq_nav),
        "equal_weight_buyhold": nav_stats(bh_nav),
    }
    with open(DATA_DIR / "portfolio_7coins_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n===== 指标 =====")
    for name, m in metrics.items():
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
    print("\n权重演变:")
    print(rp["weights"].to_string(index=False))
    print("\n调仓记录:")
    print(rp["trades"].to_string(index=False))
    print(f"\n总耗时 {time.time()-t0:.0f}s，数据已保存至 {DATA_DIR}")


if __name__ == "__main__":
    sys.exit(main())
