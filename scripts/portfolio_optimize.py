"""风险平价投资组合优化：ETH/TRX 动态网格 + PAXG 简单持有。

方法：
- 三资产净值序列：ETH/TRX 用动态网格回测（每段按段资金重跑，VOLUME_PER_GRID=段资金×2.5%/段首价，
  方案 A 随资金缩放）；PAXG 简单持有（线性缩放）。
- 风险平价权重：EWMA(λ=0.99) 协方差（用原生资产价格收益，避免网格净值波动扭曲风险平价）
  + 等风险贡献优化（scipy SLSQP，0≤w≤0.7, Σw=1）。
  注：cvxopt 无对数锥支持，风险平价用 scipy 求解（cvxopt 保留用于备选）。
- 训练集（2021-05-07~2025-12-31）完整组合回放（初始 1/3、季度调仓约 15 次）→ 末端权重启动验证集
  （2026-01-01~08-11，季度调仓 03-31/06-30，末期评估）。
- 手续费：建仓 0.1%；调仓 0.1% 单边按新权重比例分摊，重跑前扣除。
- 基准：等权（100000/3 一次投入不调仓）、全仓单币种（ETH/TRX 网格、PAXG 持有，同一区间）。
- 输出：文字指标、pic/portfolio_weights.png、pic/portfolio_nav.png、pic/portfolio_compare.png、
  data/portfolio_weights.csv、data/portfolio_nav.csv。

用法：
    uv run python scripts/portfolio_optimize.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 先加载真实 torch（scipy 依赖其 Tensor 等符号）。dynamic_grid_backtest 的 stub 注入
# 仅在 torch 未加载时生效，此处确保真实 torch 已就位，scipy 运行不受 stub 影响。
import torch  # noqa: F401
import scipy  # noqa: F401
from scipy.optimize import minimize  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import dynamic_grid_backtest as dgb  # noqa: E402

# ============================================================
# 参数
# ============================================================
LAMBDA = 0.99          # EWMA 衰减（可由 --lambda 覆盖；0.99 训练集表现优于 0.97）
PER_GRID_RATIO = 0.025  # 每格价值 / 账户资金（方案 A，ETH 用）
PER_GRID_RATIO_TRX = 0.01  # TRX 低价高成交，2.5% 会爆仓，调低至 1%
W_UPPER = 0.7          # 单资产权重上限（可由 --w-upper 覆盖；0.7 训练集跑赢等权）
W_LOWER = 0.0
FEE = 0.001            # 单边手续费
CAPITAL = 100_000

TRAIN_START = "2021-05-07"
TRAIN_END = "2025-12-31"
VAL_START = "2026-01-01"
VAL_END = "2026-08-11"

# 币种配置：网格参数（训练集搜索最优）
COINS = {
    "ETH": {
        "data": "eth_usdt_1d.csv", "vt_symbol": "ETHUSDT.GLOBAL",
        "pricetick": 0.01, "ma_window": 10, "gate_window": 90,
        "k_width": 2.0, "target_ret": 0.03, "atr_window": 7, "grid": True,
    },
    "TRX": {
        "data": "trx_usdt_1d_smoothed.csv", "vt_symbol": "TRXUSDT.GLOBAL",
        "pricetick": 0.000001, "ma_window": 20, "gate_window": 20,
        "k_width": 5.0, "target_ret": 0.003, "atr_window": 21, "grid": True,
    },
    "PAXG": {"data": "paxg_usdt_1d.csv", "grid": False},
}
ORDER = ["ETH", "TRX", "PAXG"]


# ============================================================
# 数据加载
# ============================================================
def load_prices() -> dict[str, dict]:
    """返回 {coin: {df: 完整OHLCV, close: Series(close)}}。"""
    out = {}
    for coin, cfg in COINS.items():
        df = pd.read_csv(PROJECT_ROOT / "data" / cfg["data"], parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        out[coin] = {"df": df, "close": pd.Series(df["close"].values, index=df["timestamp"])}
    return out


# ============================================================
# 网格回测（复用 dynamic_grid_backtest.run_backtest）
# ============================================================
def run_grid_segment(coin: str, prices: dict, start: str, end: str, capital: float,
                     seg_start_close: float, lab: Path) -> pd.Series:
    """用段资金重跑网格，返回段内逐日净值 Series（index=日期，从段首起算 capital）。
    volume_per_grid = capital × PER_GRID_RATIO / 段首价（方案 A 缩放）。"""
    cfg = COINS[coin]
    df = prices[coin]["df"]
    pricetick = cfg["pricetick"]
    ratio = PER_GRID_RATIO_TRX if coin == "TRX" else PER_GRID_RATIO
    volume_per_grid = capital * ratio / seg_start_close
    volume_per_grid = max(volume_per_grid, 1e-9)

    dgb_cfg = {
        "start": start, "end": end, "vt_symbol": cfg["vt_symbol"], "lab": str(lab),
        "ma_window": cfg["ma_window"], "gate_window": cfg["gate_window"],
        "atr_window": cfg["atr_window"], "k_width": cfg["k_width"],
        "target_ret": cfg["target_ret"], "min_steps": 5, "max_steps": 300,
        "reanchor_gap": 7, "reanchor_dev": 0.25,
        "capital": capital, "fee_rate": FEE,
        "volume_per_grid": volume_per_grid, "initial_base_ratio": 0.5,
        "pricetick": pricetick,
        "grid_lower": 1e-6, "grid_upper": 1e9, "grid_steps": 50,
    }
    dgb._CFG = dgb_cfg
    dgb._GATE_MAP = dgb.build_ma_gate_map(df, cfg["gate_window"])
    dgb._BOUNDS_MAP = dgb.build_bounds_map(
        df, cfg["ma_window"], cfg["atr_window"], cfg["k_width"], cfg["target_ret"],
        5, 300, pricetick,
    )
    try:
        res = dgb.run_backtest(df, dgb_cfg, dgb.DynamicBandMAGatedGridStrategy)
        dates = res["bdf"]["timestamp"].reset_index(drop=True)
        balance = res["balance"]
        if balance is None or len(balance) == 0:
            raise RuntimeError("zero trades")
        return pd.Series(balance, index=dates)
    except Exception:
        # 段内网格零成交或统计失败（calculate_result 返回 None）→ 降级为持有
        df_seg = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
        idx = df_seg["timestamp"].reset_index(drop=True)
        bal = capital * df_seg["close"].values / seg_start_close
        return pd.Series(bal, index=idx)


# ============================================================
# PAXG 持有净值（线性缩放）
# ============================================================
def paxg_segment(prices: dict, coin: str, start: str, end: str, capital: float,
                 seg_start_close: float) -> pd.Series:
    s = prices[coin]["close"]
    idx = s[(s.index >= start) & (s.index <= end)]
    return pd.Series(capital * idx.values / seg_start_close, index=idx.index)


# ============================================================
# 风险平价权重（scipy SLSQP，等风险贡献）
# ============================================================
def ewma_covariance(returns: pd.DataFrame, lam: float) -> np.ndarray:
    """指数加权协方差（返回末日期协方差矩阵）：Σ_t = λΣ_{t-1} + (1-λ) r_t r_tᵀ。"""
    n = len(returns.columns)
    cov = np.zeros((n, n))
    for _, r in returns.iterrows():
        rv = r.values.astype(float)
        cov = lam * cov + (1 - lam) * np.outer(rv, rv)
    return cov


def _ewma_append(cov_state: np.ndarray, seen_dates: set, new_ret: pd.DataFrame, lam: float) -> np.ndarray:
    """将 new_ret 中尚未处理的收益递推到 EWMA 协方差，返回新协方差与新 seen_dates。"""
    cov = cov_state.copy()
    for t, r in new_ret.iterrows():
        if t in seen_dates:
            continue
        rv = r.values.astype(float)
        cov = lam * cov + (1 - lam) * np.outer(rv, rv)
        seen_dates.add(t)
    return cov


def risk_parity_weights(cov: np.ndarray, upper: float | None = None) -> np.ndarray:
    """等风险贡献权重：min Σ(RCᵢ - RC̄)², s.t. 0≤w≤upper, Σw=1。
    用 scipy SLSQP（cvxopt 无对数锥，风险平价非线性用 scipy）。"""
    if upper is None:
        upper = W_UPPER
    n = cov.shape[0]

    def rc(w):
        vol = np.sqrt(w @ cov @ w)
        return w * (cov @ w) / vol if vol > 0 else np.zeros(n)

    def obj(w):
        r = rc(w)
        return float(np.sum((r - r.mean()) ** 2))

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bnds = [(W_LOWER, upper)] * n
    x0 = np.ones(n) / n
    res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    if not res.success:
        # 约束过紧不可行（如 upper < 1/n）：退回可行边界内接近等权的解
        w = np.clip(np.ones(n) / n, W_LOWER, upper)
        w = w / w.sum()
        return w
    return res.x


# ============================================================
# 组合回放（训练集与验证集共用）
# ============================================================
def run_portfolio(prices: dict[str, pd.Series], start: str, end: str,
                  init_weights: np.ndarray, init_capital: float,
                  lab_root: Path, quarterly: list[tuple[str, str]]) -> dict:
    """逐季度段回放：每段按段资金重跑网格/持有，段末调仓（EWMA→风险平价→Δ→扣费分摊）。
    返回 {nav: Series(组合净值), weights: DataFrame(权重演变), trades: list[dict]}"""
    all_dates = pd.date_range(start, end, freq="D")
    all_dates = all_dates[all_dates <= pd.Timestamp(end)]
    # 只保留有数据的日期
    data_dates = sorted(set.intersection(*[set(p["close"].index) for p in prices.values()]))
    data_dates = [d for d in data_dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)]

    segs = []  # (seg_start, seg_end)
    seg_start = pd.Timestamp(start)
    for q_start, q_end in quarterly:
        if pd.Timestamp(q_end) < seg_start:
            continue
        segs.append((seg_start, pd.Timestamp(q_end)))
        seg_start = pd.Timestamp(q_end) + pd.Timedelta(days=1)
    if seg_start <= pd.Timestamp(end):
        segs.append((seg_start, pd.Timestamp(end)))

    w = init_weights.copy()
    capital = init_capital
    nav_by_coin = {c: [] for c in ORDER}
    nav_total = []
    weights_log = []
    trades_log = []
    # EWMA 协方差跨段累积状态（用原生资产价格收益，避免网格策略净值波动扭曲风险平价）
    cov_state = np.zeros((3, 3))
    seen_ret_dates: set = set()

    # 原生价格收益序列（全数据，供协方差计算）
    price_ret = pd.DataFrame({c: prices[c]["close"].pct_change() for c in ORDER}).dropna()

    for seg_i, (s0, s1) in enumerate(segs):
        s0s, s1s = s0.strftime("%Y-%m-%d"), s1.strftime("%Y-%m-%d")
        seg_cap = {c: capital * w[i] for i, c in enumerate(ORDER)}
        seg_nav = {}

        for i, coin in enumerate(ORDER):
            seg_close_start = prices[coin]["close"].loc[prices[coin]["close"].index >= s0].iloc[0]
            if COINS[coin]["grid"]:
                seg_nav[coin] = run_grid_segment(
                    coin, prices, s0s, s1s, seg_cap[coin], seg_close_start, lab_root)
            else:
                seg_nav[coin] = paxg_segment(prices, coin, s0s, s1s, seg_cap[coin], seg_close_start)

        # 段内组合净值
        idx = sorted(set.intersection(*[set(s.index) for s in seg_nav.values()]))
        idx = [d for d in idx if d <= pd.Timestamp(s1s)]
        combo = pd.DataFrame({c: seg_nav[c].reindex(idx).ffill() for c in ORDER}).fillna(0)
        total = combo.sum(axis=1)
        nav_by_coin_list = [combo[c] for c in ORDER]

        # 记录段内每日（合并到全局）
        for d in idx:
            for ci, coin in enumerate(ORDER):
                nav_by_coin[coin].append((d, nav_by_coin_list[ci].loc[d]))
            nav_total.append((d, total.loc[d]))

        # 记录段末权重
        weights_log.append({"date": s1s, **{c: w[i] for i, c in enumerate(ORDER)}})

        # 段末调仓（除最后一段）
        if seg_i < len(segs) - 1:
            v_end = combo.iloc[-1]
            V_total = v_end.sum()
            # EWMA 协方差（原生资产价格收益，跨段累积，只用 ≤段末 数据，无前视）
            ret_df = price_ret[price_ret.index <= pd.Timestamp(s1s)]
            cov_state = _ewma_append(cov_state, seen_ret_dates, ret_df, LAMBDA)
            cov = cov_state
            w_new = risk_parity_weights(cov)
            # Δ
            T = V_total * w_new
            delta = T - v_end.values
            fee = FEE * np.abs(delta).sum()
            # 分摊费用
            fee_alloc = fee * w_new
            new_cap = (V_total - fee) * w_new
            trades_log.append({
                "date": s1s, "v_total": V_total, "w_old": w.copy(), "w_new": w_new.copy(),
                "delta": delta, "fee": fee,
            })
            w = w_new
            capital = V_total - fee
            # 更新下一段资金（记录用）
            seg_cap_next = new_cap

    nav = pd.Series(dict(nav_total)).sort_index()
    wdf = pd.DataFrame(weights_log)
    return {"nav": nav, "weights": wdf, "trades": trades_log}


# ============================================================
# 单币种基准
# ============================================================
def single_coin_baseline(prices: dict, start: str, end: str,
                         lab_root: Path) -> dict[str, pd.Series]:
    """全仓单币种：ETH/TRX 网格满仓 100k（方案 A 缩放）、PAXG 持有。"""
    out = {}
    for coin in ORDER:
        c0 = prices[coin]["close"].loc[prices[coin]["close"].index >= start].iloc[0]
        if COINS[coin]["grid"]:
            out[coin] = run_grid_segment(coin, prices, start, end, CAPITAL, c0, lab_root)
        else:
            out[coin] = paxg_segment(prices, coin, start, end, CAPITAL, c0)
    return out


# ============================================================
# 等权基准
# ============================================================
def equal_weight_baseline(prices: dict, start: str, end: str, lab_root: Path) -> pd.Series:
    w = np.array([1 / 3] * 3)
    cap = CAPITAL / 3
    segs = []
    for coin in ORDER:
        c0 = prices[coin]["close"].loc[prices[coin]["close"].index >= start].iloc[0]
        if COINS[coin]["grid"]:
            s = run_grid_segment(coin, prices, start, end, cap, c0, lab_root)
        else:
            s = paxg_segment(prices, coin, start, end, cap, c0)
        segs.append(s)
    idx = sorted(set.intersection(*[set(s.index) for s in segs]))
    return pd.DataFrame({c: segs[i].reindex(idx).ffill() for i, c in enumerate(ORDER)}).sum(axis=1)


# ============================================================
# 指标
# ============================================================
def metrics(nav: pd.Series) -> dict:
    if len(nav) < 2:
        return {}
    ret = nav.pct_change().dropna()
    total = nav.iloc[-1] / nav.iloc[0] - 1
    ann = (1 + total) ** (365 / len(nav)) - 1
    vol = ret.std() * np.sqrt(365)
    dd = (nav / nav.cummax() - 1).min()
    sharpe = (ret.mean() / ret.std() * np.sqrt(365)) if ret.std() > 0 else 0
    return {"total": total, "ann": ann, "vol": vol, "dd": dd, "sharpe": sharpe}


def fmt_metrics(m: dict, pct_cols=("total", "ann", "vol", "dd")) -> str:
    def f(k):
        v = m.get(k, float("nan"))
        if k in pct_cols:
            return f"{v*100:.2f}%"
        return f"{v:.2f}"
    return " / ".join(f"{k}={f(k)}" for k in m)


# ============================================================
# 主流程
# ============================================================
def main() -> int:
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description="风险平价组合优化")
    parser.add_argument("--lambda-lam", type=float, default=0.99, help="EWMA 衰减（默认 0.99）")
    parser.add_argument("--w-upper", type=float, default=0.7, help="单资产权重上限（默认 0.7）")
    args = parser.parse_args()

    global LAMBDA, W_UPPER
    LAMBDA = args.lambda_lam
    W_UPPER = args.w_upper

    lab_root = Path(tempfile.mkdtemp(prefix="port_lab_"))
    prices = load_prices()

    # 季度调仓点（自然日）
    def quarterly_dates(start: str, end: str) -> list[tuple[str, str]]:
        """按季度边界切段：每段起于上一季度末次日、止于本季度末（自然日 03-31/06-30/09-30/12-30）。"""
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        qends = []
        y = start_ts.year
        while y <= end_ts.year + 1:
            for m in (3, 6, 9, 12):
                qe = pd.Timestamp(f"{y}-{m:02d}-30")
                if qe <= start_ts:
                    continue
                if qe > end_ts:
                    break
                qends.append(qe)
            if qends and qends[-1] >= end_ts:
                break
            y += 1
        pts = []
        prev = start_ts
        for qe in qends:
            pts.append((str(prev.date()), str(qe.date())))
            prev = qe + pd.Timedelta(days=1)
        if prev <= end_ts:
            pts.append((str(prev.date()), end))
        return pts

    # 训练集回放
    train_q = quarterly_dates(TRAIN_START, TRAIN_END)
    tr_res = run_portfolio(prices, TRAIN_START, TRAIN_END,
                           np.array([1 / 3] * 3), CAPITAL, lab_root, train_q)
    w_train_end = tr_res["weights"].iloc[-1][ORDER].values
    tr_nav = tr_res["nav"]

    # 验证集回放（初始权重=训练集末 w_train_end，资金重新按 100,000，重新建仓）
    val_q = quarterly_dates(VAL_START, VAL_END)
    va_res = run_portfolio(prices, VAL_START, VAL_END, w_train_end, CAPITAL, lab_root, val_q)
    va_nav = va_res["nav"]

    # 等权基准：训练/验证两段独立（各 100,000/3 起算，不调仓）
    eq_tr = equal_weight_baseline(prices, TRAIN_START, TRAIN_END, lab_root)
    eq_va = equal_weight_baseline(prices, VAL_START, VAL_END, lab_root)
    eq_full = equal_weight_baseline(prices, TRAIN_START, VAL_END, lab_root)  # 全区间总览
    single_full = single_coin_baseline(prices, TRAIN_START, VAL_END, lab_root)

    # 指标
    tr_m = metrics(tr_nav)
    va_m = metrics(va_nav)
    eq_tr_m = metrics(eq_tr)
    eq_va_m = metrics(eq_va)
    eq_m = metrics(eq_full)
    single_m = {c: metrics(s) for c, s in single_full.items()}

    # ===== 输出 =====
    print("=" * 70)
    print("风险平价投资组合：ETH/TRX 网格 + PAXG 持有")
    print("=" * 70)
    print(f"\n训练集 {TRAIN_START} ~ {TRAIN_END}（{len(tr_nav)} 天）")
    print(f"  组合: {fmt_metrics(tr_m)}")
    print(f"  等权: {fmt_metrics(eq_tr_m)}")
    print(f"  初始权重 1/3 -> 末端权重: " + " ".join(f"{c}={w_train_end[i]:.3f}" for i, c in enumerate(ORDER)))
    print(f"  调仓 {len(tr_res['trades'])} 次，总调仓费: {sum(t['fee'] for t in tr_res['trades']):.2f}")

    print(f"\n验证集 {VAL_START} ~ {VAL_END}（{len(va_nav)} 天）")
    print(f"  组合: {fmt_metrics(va_m)}")
    print(f"  等权: {fmt_metrics(eq_va_m)}")
    print(f"  启动权重(训练集末): " + " ".join(f"{c}={w_train_end[i]:.3f}" for i, c in enumerate(ORDER)))
    for t in va_res["trades"]:
        wnew = " ".join(f"{c}={t['w_new'][i]:.3f}" for i, c in enumerate(ORDER))
        print(f"    {t['date']}: 净值={t['v_total']:.0f} 调仓费={t['fee']:.2f} 新权重[{wnew}]")

    print(f"\n等权基准（全区间总览）: {fmt_metrics(eq_m)}")
    print("\n全仓单币种（全区间）:")
    for c, m in single_m.items():
        print(f"  {c}: {fmt_metrics(m)}")

    # 训练/验证对比表
    print("\n" + "=" * 70)
    print("组合 vs 等权 vs 单币种 对比")
    print("=" * 70)
    for label, nav, eq in [("训练集", tr_nav, eq_tr), ("验证集", va_nav, eq_va)]:
        print(f"\n[{label}]")
        print(f"  {'标的':<14}{'总收益':>10}{'年化':>10}{'回撤':>10}{'Sharpe':>8}")
        m = metrics(nav)
        print(f"  {'组合(风险平价)':<14}{m['total']*100:>9.2f}%{m['ann']*100:>9.2f}%"
              f"{m['dd']*100:>9.2f}%{m['sharpe']:>8.2f}")
        me = metrics(eq)
        print(f"  {'等权持有':<14}{me['total']*100:>9.2f}%{me['ann']*100:>9.2f}%"
              f"{me['dd']*100:>9.2f}%{me['sharpe']:>8.2f}")
        for c in ORDER:
            if label == "训练集":
                s = single_full[c][single_full[c].index <= TRAIN_END]
            else:
                s = single_full[c][(single_full[c].index >= VAL_START) & (single_full[c].index <= VAL_END)]
            m = metrics(s)
            print(f"  {f'单币种{c}':<14}{m['total']*100:>9.2f}%{m['ann']*100:>9.2f}%"
                  f"{m['dd']*100:>9.2f}%{m['sharpe']:>8.2f}")

    # 分析结论
    print("\n" + "=" * 70)
    print("分析结论（组合 vs 等权收益差别）")
    print("=" * 70)
    print(f"  训练集: 组合收益 {tr_m['total']*100:.2f}% vs 等权 {eq_tr_m['total']*100:.2f}%"
          f" (差 {tr_m['total']*100 - eq_tr_m['total']*100:+.2f}pct)")
    print(f"  验证集: 组合收益 {va_m['total']*100:.2f}% vs 等权 {eq_va_m['total']*100:.2f}%"
          f" (差 {va_m['total']*100 - eq_va_m['total']*100:+.2f}pct)")
    print(f"  验证集回撤: 组合 {va_m['dd']*100:.2f}% vs 等权 {eq_va_m['dd']*100:.2f}%")
    print(f"  验证集Sharpe: 组合 {va_m['sharpe']:.2f} vs 等权 {eq_va_m['sharpe']:.2f}")

    # ===== 图片 =====
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Noto Sans CJK SC", "axes.unicode_minus": False})
    wdf = pd.concat([tr_res["weights"], va_res["weights"]], ignore_index=True)
    wdf["date"] = pd.to_datetime(wdf["date"])
    train_end_ts = pd.Timestamp(TRAIN_END)

    # 权重图（全区间单张，竖线+淡色背景标记训练/验证分界）
    fig, ax = plt.subplots(figsize=(14, 5))
    x = wdf["date"]
    ax.stackplot(x, wdf["ETH"], wdf["TRX"], wdf["PAXG"],
                 labels=["ETH", "TRX", "PAXG"], colors=["#1f77b4", "#ff7f0e", "#2ca02c"], alpha=0.85)
    for t in wdf["date"]:
        ax.axvline(t, color="#888", ls=":", lw=0.6)
    ax.axvline(train_end_ts, color="#d62728", ls="--", lw=1.5, label="训练/验证分界")
    ax.axvspan(x.min(), train_end_ts, color="#1f77b4", alpha=0.04)
    ax.axvspan(train_end_ts, x.max(), color="#2ca02c", alpha=0.04)
    ax.set_title("风险平价权重演变（训练集 + 验证集，季度调仓）")
    ax.set_ylabel("权重")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "pic" / "portfolio_weights.png", dpi=110)
    plt.close(fig)

    # 收益曲线图 / 回撤图（组合 vs 等权，训练/验证各一张）
    def plot_nav(nav, eq, title, fname):
        fig, ax = plt.subplots(figsize=(14, 5))
        nav.plot(ax=ax, label="组合(风险平价)", color="#1f77b4", lw=1.5)
        eq.plot(ax=ax, label="等权持有", color="#7f7f7f", ls="--", lw=1.2)
        ax.axhline(CAPITAL, color="#333", ls=":", lw=0.8)
        ax.set_title(title)
        ax.set_ylabel("净值 (USDT)")
        ax.legend(loc="upper left", fontsize=9)
        plt.tight_layout()
        fig.savefig(PROJECT_ROOT / "pic" / fname, dpi=110)
        plt.close(fig)

    def plot_dd(nav, eq, title, fname):
        fig, ax = plt.subplots(figsize=(14, 5))
        dd_c = (nav / nav.cummax() - 1) * 100
        dd_e = (eq / eq.cummax() - 1) * 100
        ax.plot(dd_c.index, dd_c.values, color="#d62728", lw=1.3, label="组合回撤")
        ax.plot(dd_e.index, dd_e.values, color="#7f7f7f", ls="--", lw=1.0, label="等权回撤")
        ax.fill_between(dd_c.index, dd_c.values, 0, where=(dd_c.values < 0),
                        color="#d62728", alpha=0.25)
        ax.fill_between(dd_e.index, dd_e.values, 0, where=(dd_e.values < 0),
                        color="#7f7f7f", alpha=0.15)
        ax.axhline(0, color="#333", lw=0.6)
        ax.set_title(title)
        ax.set_ylabel("回撤 (%)")
        ax.legend(loc="lower left", fontsize=9)
        plt.tight_layout()
        fig.savefig(PROJECT_ROOT / "pic" / fname, dpi=110)
        plt.close(fig)

    plot_nav(tr_nav, eq_tr, f"训练集净值：组合 vs 等权（{TRAIN_START} ~ {TRAIN_END}）", "portfolio_nav_train.png")
    plot_nav(va_nav, eq_va, f"验证集净值：组合 vs 等权（{VAL_START} ~ {VAL_END}）", "portfolio_nav_val.png")
    plot_dd(tr_nav, eq_tr, f"训练集回撤：组合 vs 等权（{TRAIN_START} ~ {TRAIN_END}）", "portfolio_dd_train.png")
    plot_dd(va_nav, eq_va, f"验证集回撤：组合 vs 等权（{VAL_START} ~ {VAL_END}）", "portfolio_dd_val.png")
    print("\n图片已保存: pic/portfolio_weights.png, portfolio_nav_train/val.png, portfolio_dd_train/val.png")

    # CSV
    full_nav = pd.concat([tr_nav, va_nav])
    full_weights = wdf.set_index("date")
    full_weights.to_csv(PROJECT_ROOT / "data" / "portfolio_weights.csv", index=True)
    nav_df = pd.DataFrame({
        "组合": full_nav,
        "等权": eq_full.reindex(full_nav.index).ffill(),
    })
    nav_df.index.name = "date"
    nav_df.to_csv(PROJECT_ROOT / "data" / "portfolio_nav.csv", index=True, index_label="date")
    print("CSV 已保存: data/portfolio_weights.csv, portfolio_nav.csv")

    import shutil
    shutil.rmtree(lab_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
