"""使用 XGBoost 预测未来 N 天市场处于「震荡」还是「单边」行情（二分类），支持多币种。

方法概述
--------
对每个交易日 t：
  - 特征：仅用 t 及以前的历史数据计算 16 个滚动特征（动量 / 波动 / K线形态 / 量能 / 位置 / 趋势强度），
    其中最长窗口为 60 日，因此最早的有效样本从数据第 61 天开始（冷启动窗口）。
  - 标签：取未来窗口 [t+1, t+30] 的累计对数收益与年化波动率，定义
        trend_score = |fut_logret| / fut_vol
    该指标在全部历史上取第 QUANTILE 分位数作为阈值，trend_score >= 阈值 → 单边(1)，否则 → 震荡(0)。
    标签依赖未来数据，特征只依赖历史，二者错开对齐，避免前视偏差。

验证方式：两种模式（--mode）：
  - walkforward：扩张窗口 walk-forward（详见下方「Fold 含义」），预测只覆盖测试折
  - rolling：细粒度滚动重训，每个预测区间前用「截至该区间起点的全部历史」重训模型，
    预测覆盖全时段（严格无泄漏），代价是每个区间都要重训一次

用法
----
    uv run python scripts/predict_regime.py                 # 默认 all（两模式都跑）
    uv run python scripts/predict_regime.py --mode rolling --step 30
    uv run python scripts/predict_regime.py --coins btc,sol --horizon 30 --n-recent 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# 仅作为脚本运行时强制无界面后端；被 notebook 导入时保留调用方的后端（如 inline）
if __name__ == "__main__":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 中文字体（脚本以命令行方式运行，需在导入 pyplot 前设置后端与字体）
plt.rcParams.update(
    {
        "font.family": "Noto Sans CJK SC",
        "axes.unicode_minus": False,
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# ============================================================
# Fold 含义（扩张窗口 walk-forward）
# ------------------------------------------------------------
# 数据按时间切成 N_SPLITS 段（本脚本为 4 段），产生 N_SPLITS-2 个测试折。
# 第 i 折：训练集 = [t0, cuts[i])，测试集 = [cuts[i], cuts[i+1])。
# 训练集随折数扩张（使用到测试起点前的全部历史），测试集永远晚于训练集，
# 模拟"只用过去数据预测未来"的真实部署情形。
#
#   Fold1: train=[t0→c1)   test=[c1→c2)
#   Fold2: train=[t0→c2)   test=[c2→c3)
#
# 每折独立报告 AUC 等指标：若多折均明显 >0.5 说明信号稳定；仅单折有效多为行情段偶然。
# ============================================================
N_SPLITS = 4  # 时间切分数，产生 N_SPLITS-2 = 2 个测试折


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """构造 16 个滚动特征（仅使用 t 时刻及以前的数据）。"""
    df = df.copy()
    df["logret"] = np.log(df["close"]).diff()

    out: dict[str, pd.Series] = {}
    for n in (5, 10, 20, 60):
        out[f"ret{n}"] = df["close"].pct_change(n)

    out["vol20"] = df["logret"].rolling(20).std() * np.sqrt(365)
    out["vol60"] = df["logret"].rolling(60).std() * np.sqrt(365)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr14"] = tr.rolling(14).mean()
    out["atr_pct"] = tr.rolling(14).mean() / df["close"]

    hl = (df["high"] - df["low"]).replace(0, np.nan)
    out["body_prop"] = (df["close"] - df["open"]) / hl
    out["range_pos"] = (df["close"] - df["low"]) / hl

    vol_sma20 = df["volume"].rolling(20).mean()
    out["vol_intensity"] = df["volume"] / vol_sma20
    out["vol_zscore"] = (df["volume"] - vol_sma20) / df["volume"].rolling(20).std()

    out["close_sma20"] = df["close"] / df["close"].rolling(20).mean() - 1
    rng60 = (df["high"].rolling(60).max() - df["low"].rolling(60).min()).replace(0, np.nan)
    out["range60_pos"] = (df["close"] - df["low"].rolling(60).min()) / rng60

    def slope20(x: pd.Series) -> float:
        x = x.dropna().values
        if len(x) < 5:
            return np.nan
        return np.polyfit(np.arange(len(x)), x, 1)[0]

    out["slope20"] = df["close"].rolling(20).apply(slope20, raw=False) / df["close"]

    out["adx14"] = _adx(df["high"], df["low"], df["close"])

    return pd.DataFrame(out)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """ADX(14) 平均趋向指数。"""
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    trv = np.maximum(
        np.maximum(high - low, (high - close.shift()).abs()),
        (low - close.shift()).abs(),
    )
    atr = pd.Series(trv).rolling(n).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(n).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(n).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(n).mean()


FEATURE_COLS = [
    "ret5", "ret10", "ret20", "ret60", "vol20", "vol60", "atr14", "atr_pct",
    "body_prop", "range_pos", "vol_intensity", "vol_zscore",
    "close_sma20", "range60_pos", "slope20", "adx14",
]


def build_labels(df: pd.DataFrame, horizon: int, quantile: float) -> tuple[pd.Series, float]:
    """构造未来 horizon 天的二分类标签（震荡=0 / 单边=1）。

    标签使用未来窗口数据（shift(-horizon) 对齐），特征只用历史，二者错开防泄漏。
    """
    logret = np.log(df["close"]).diff()
    fut_logret = logret.shift(-horizon).rolling(horizon).sum()
    fut_vol = logret.shift(-horizon).rolling(horizon).std() * np.sqrt(365)
    trend_score = fut_logret.abs() / fut_vol
    threshold = trend_score.dropna().quantile(quantile)
    labels = (trend_score >= threshold).astype(int).rename("y")
    return labels, threshold


def load_data(coin: str) -> pd.DataFrame:
    """读取 data/{coin}_usdt_1d.csv 并按时间排序。"""
    path = PROJECT_ROOT / "data" / f"{coin}_usdt_1d.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def walk_forward_metrics(
    FR: pd.DataFrame, horizon: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """扩张窗口 walk-forward 评估。

    返回:
        metrics_df: 每折的 AUC / PR-AUC / 最优阈值下 F1、精确率、召回率 / 混淆矩阵元素
        preds_df:   每折的 (日期, 真实, 预测概率, 最优阈值下预测类别)
        last_thr:   最后一折使用的最优阈值（用于图片标注）
    """
    cuts = np.linspace(0, len(FR), N_SPLITS).astype(int)
    test_folds = [(cuts[i], cuts[i + 1]) for i in range(1, N_SPLITS - 1)]

    metrics, preds_all = [], []
    last_thr = 0.5

    for fold_i, (a, b) in enumerate(test_folds, 1):
        tr_end = cuts[fold_i]
        Xtr, ytr = FR[FEATURE_COLS].iloc[:tr_end], FR["y"].iloc[:tr_end]
        Xte, yte = FR[FEATURE_COLS].iloc[a:b], FR["y"].iloc[a:b]

        model = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.7,
            scale_pos_weight=(1 - ytr.mean()) / ytr.mean(),
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(Xtr, ytr)
        prob = model.predict_proba(Xte)[:, 1]

        auc = roc_auc_score(yte, prob)
        ap = average_precision_score(yte, prob)

        prec, rec, ths = precision_recall_curve(yte, prob)
        f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        k = int(np.argmax(f1s))
        thr = ths[k] if k < len(ths) else 0.5
        pred = (prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()

        metrics.append(
            dict(
                折=fold_i,
                样本=len(yte),
                单边占比=round(yte.mean(), 3),
                AUC=round(auc, 3),
                PR_AUC=round(ap, 3),
                F1=round(f1s[k], 3),
                精确率=round(prec[k], 3),
                召回率=round(rec[k], 3),
                阈值=round(thr, 3),
                TN=tn, FP=fp, FN=fn, TP=tp,
            )
        )
        preds_all.append(
            pd.DataFrame(
                {
                    "日期": FR["timestamp"].iloc[a:b],
                    "真实": yte.values,
                    "P(单边)": prob,
                    "预测": pred,
                    "折": fold_i,
                }
            )
        )
        last_thr = thr

    return pd.DataFrame(metrics), pd.concat(preds_all, ignore_index=True), last_thr


def rolling_metrics(
    FR: pd.DataFrame, seed: int, step: int = 30, min_train: int = 500
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """细粒度滚动重训评估（rolling refit）。

    与 walk-forward 不同，这里对每个预测区间都先用「截至该区间起点的全部历史」重训模型，
    因此预测覆盖从 min_train 起到数据末尾的每一天（全时段、严格无泄漏）。

      refit_idx = min_train
      while refit_idx < n:
          train = FR[0:refit_idx]                      # 扩张窗口，只用历史
          fit XGBoost 并预测 FR[refit_idx:refit_idx+step]   # 每 step 天重训一次
          refit_idx += step

    返回:
        metrics_df: 整体 AUC / PR-AUC 与样本规模
        preds_df:   全时段 (日期, 真实, P(单边))
        best_thr:   训练期最优阈值（用于图片标注）
    """
    n = len(FR)
    rows, last_thr = [], 0.5
    refit_idx = min_train

    while refit_idx < n - 1:
        end = min(refit_idx + step, n)
        Xtr, ytr = FR[FEATURE_COLS].iloc[:refit_idx], FR["y"].iloc[:refit_idx]

        model = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.7,
            scale_pos_weight=(1 - ytr.mean()) / ytr.mean(),
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(Xtr, ytr)
        prob = model.predict_proba(FR[FEATURE_COLS].iloc[refit_idx:end])[:, 1]

        # 训练期最优阈值（与 walk_forward 的阈值选择方式对称）
        prec, rec, ths = precision_recall_curve(ytr, model.predict_proba(Xtr)[:, 1])
        f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        k = int(np.argmax(f1s))
        last_thr = ths[k] if k < len(ths) else 0.5

        rows.append(
            pd.DataFrame(
                {
                    "日期": FR["timestamp"].iloc[refit_idx:end].values,
                    "真实": FR["y"].iloc[refit_idx:end].values,
                    "P(单边)": prob,
                }
            )
        )
        refit_idx = end

    preds = pd.concat(rows, ignore_index=True)
    auc = roc_auc_score(preds["真实"], preds["P(单边)"])
    ap = average_precision_score(preds["真实"], preds["P(单边)"])
    metrics = pd.DataFrame(
        [dict(样本=len(preds), 单边占比=round(preds["真实"].mean(), 3), AUC=round(auc, 3), PR_AUC=round(ap, 3))]
    )
    return metrics, preds, last_thr


def print_recent(preds: pd.DataFrame, n_recent: int, coin: str) -> None:
    """打印最近 n_recent 天的预测明细。"""
    recent = preds.sort_values("日期").tail(n_recent).reset_index(drop=True)
    print(f"\n--- {coin.upper()} 最近 {len(recent)} 天预测明细 ---")
    for r in recent.itertuples():
        real = "单边" if r.真实 == 1 else "震荡"
        pred = "单边" if r.预测 == 1 else "震荡"
        mark = "  <-- 预测单边" if r.预测 == 1 else ""
        print(
            f"  {r.日期:%Y-%m-%d}  真实={real}  预测={pred}  P(单边)={r._3:.3f}{mark}"
        )


def plot_coin(df: pd.DataFrame, labels: pd.Series, preds: pd.DataFrame, thr: float, coin: str, outdir: Path, suffix: str = "") -> None:
    """保存每币种的预测可视化：上=价格+真实单边着色，下=P(单边)概率+预测标记。"""
    df = df.copy()
    df["label"] = labels.values

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    axes[0].plot(df["timestamp"], df["close"], lw=0.8, color="#333333", zorder=3)
    for ts, lab in zip(df["timestamp"], df["label"]):
        if lab == 1:
            axes[0].axvspan(ts, ts + pd.Timedelta(days=1), color="#d62728", alpha=0.15, linewidth=0)
    axes[0].set_title(f"{coin.upper()} 收盘价（红底=真实未来单边）")
    axes[0].set_ylabel("价格")

    axes[1].plot(preds["日期"], preds["P(单边)"], color="#1f77b4", lw=1, label="P(单边)")
    axes[1].axhline(thr, color="#ff7f0e", ls="--", lw=1, label=f"最优阈值 {thr:.2f}")
    for r in preds.itertuples():
        if r.预测 == 1:
            axes[1].plot(r.日期, 1.02, marker="^", color="#d62728", markersize=4, linestyle="none", alpha=0.7)
    axes[1].set_ylim(0, 1.08)
    axes[1].set_title("模型预测 P(单边)（▲=预测单边）")
    axes[1].legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    out_path = outdir / f"regime_pred_{coin}{suffix}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  图片已保存: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost 预测未来市场震荡/单边（多币种，二分类）")
    parser.add_argument("--coins", default="btc,eth,sol", help="币种列表，逗号分隔（对应 data/{coin}_usdt_1d.csv）")
    parser.add_argument("--horizon", type=int, default=30, help="预测未来 N 天（默认 30）")
    parser.add_argument("--quantile", type=float, default=0.60, help="单边标签阈值分位数（默认 0.60）")
    parser.add_argument("--n-recent", type=int, default=30, help="打印最近 N 天预测明细（默认 30）")
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "pic"), help="图片输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--mode", choices=["walkforward", "rolling", "all"], default="all",
        help="评估模式（默认 all：两种都跑）",
    )
    parser.add_argument("--step", type=int, default=30, help="滚动重训间隔天数（默认 30）")
    parser.add_argument("--min-train", type=int, default=500, help="滚动模式首个预测点之前的训练天数（默认 500）")
    args = parser.parse_args()

    coins = [c.strip().lower() for c in args.coins.split(",") if c.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = []
    for coin in coins:
        df = load_data(coin)
        features = build_features(df)
        labels, threshold = build_labels(df, args.horizon, args.quantile)

        FR = pd.concat([df["timestamp"], features, labels], axis=1).dropna().reset_index(drop=True)

        print(f"\n{'=' * 64}")
        print(f"{coin.upper()} | 数据 {df['timestamp'].iloc[0].date()} ~ {df['timestamp'].iloc[-1].date()} "
              f"({len(df)} 根日线)")
        print(f"对齐样本 {len(FR)} | 单边占比 {FR['y'].mean():.1%} | 单边阈值 trend_score ≥ {threshold:.3f} "
              f"(分位数 {args.quantile:.0%})")
        print("=" * 64)

        row = {"币种": coin.upper(), "数据天数": len(df), "单边占比": f"{FR['y'].mean():.1%}"}

        if args.mode in ("walkforward", "all"):
            metrics, preds, last_thr = walk_forward_metrics(FR, args.horizon, args.seed)
            print("walk-forward 各折指标:")
            print(metrics.to_string(index=False))
            print_recent(preds, args.n_recent, coin)
            plot_coin(df, labels, preds, last_thr, coin, outdir)
            row.update(
                {
                    "WF_F1_AUC": metrics["AUC"].iloc[0],
                    "WF_F2_AUC": metrics["AUC"].iloc[1],
                    "WF_F1_F1": metrics["F1"].iloc[0],
                    "WF_F2_F1": metrics["F1"].iloc[1],
                }
            )

        if args.mode in ("rolling", "all"):
            r_metrics, r_preds, r_thr = rolling_metrics(
                FR, args.seed, step=args.step, min_train=args.min_train
            )
            r_preds["预测"] = (r_preds["P(单边)"] >= r_thr).astype(int)
            r_preds["折"] = 0
            print(f"\n滚动重训指标（step={args.step} 天, min_train={args.min_train} 天）:")
            print(r_metrics.to_string(index=False))
            print(f"  预测覆盖 {r_preds['日期'].iloc[0]:%Y-%m-%d} ~ {r_preds['日期'].iloc[-1]:%Y-%m-%d} "
                  f"（{len(r_preds)} 天）")
            print_recent(r_preds, args.n_recent, coin)
            plot_coin(df, labels, r_preds, r_thr, coin, outdir, suffix="_rolling")
            row.update({"Roll_AUC": r_metrics["AUC"].iloc[0], "Roll_PR_AUC": r_metrics["PR_AUC"].iloc[0]})

        summary.append(pd.DataFrame([row]))

    print("\n" + "=" * 64)
    print("多币种汇总对比")
    print("=" * 64)
    print(pd.concat(summary, ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
