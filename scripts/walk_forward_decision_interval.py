"""Walk-forward 验证 4h vs 1h DQN 门控稳健性。

扩张窗口 + 6 个月 OOS，共 4 折：
  Fold 1: train 2022-01-01 ~ 2023-12-31, test 2024-01-01 ~ 2024-06-30
  Fold 2: train 2022-01-01 ~ 2024-06-30, test 2024-07-01 ~ 2024-12-31
  Fold 3: train 2022-01-01 ~ 2024-12-31, test 2025-01-01 ~ 2025-06-30
  Fold 4: train 2022-01-01 ~ 2025-06-30, test 2025-07-01 ~ 2025-12-31

每个 fold 训练 decision_interval=1 和 4 两个 DQN，并回测 5 个策略。
结果增量写入 data/walk_forward_4h_results.csv，完成后打印汇总表。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.dqn_gate_env as g  # noqa: E402
from stable_baselines3 import DQN  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack  # noqa: E402

FOLDS = [
    ("2022-01-01", "2023-12-31 23:00:00", "2024-01-01", "2024-06-30 23:00:00"),
    ("2022-01-01", "2024-06-30 23:00:00", "2024-07-01", "2024-12-31 23:00:00"),
    ("2022-01-01", "2024-12-31 23:00:00", "2025-01-01", "2025-06-30 23:00:00"),
    ("2022-01-01", "2025-06-30 23:00:00", "2025-07-01", "2025-12-31 23:00:00"),
]

RESULT_CSV = PROJECT_ROOT / "data" / "walk_forward_4h_results.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward 验证 4h vs 1h DQN 门控")
    p.add_argument("--data", default=str(PROJECT_ROOT / "data" / "btc_usdt_1h.csv"))
    p.add_argument("--fold", type=int, default=0, help="只跑指定 fold（1~4），0=全部")
    p.add_argument("--timesteps", type=int, default=100_000, help="每个 DQN 的训练步数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--switch-penalty", type=float, default=0.05)
    p.add_argument("--intervals", default="1,4", help="逗号分隔的决策间隔，默认 1,4")
    p.add_argument("--result-csv", default=str(RESULT_CSV), help="结果 CSV 路径")
    return p.parse_args()


def simulate_actions(df: pd.DataFrame, start: str, end: str, actions: list[int]) -> dict:
    return g.simulate(
        df, start, end, actions=actions,
        ma_window=60, atr_window=14, k_width=4.0, target_ret=0.015,
        min_steps=5, max_steps=200, reanchor_min_gap=168, reanchor_dev=0.25,
        volume_per_grid=0.01, fee_rate=0.001, initial_base_ratio=0.5,
        pricetick=0.01, band_mode="daily",
    )


def simulate_rule(df: pd.DataFrame, start: str, end: str, mode: str) -> dict:
    return g.simulate(
        df, start, end, mode=mode,
        ma_window=60, atr_window=14, k_width=4.0, target_ret=0.015,
        min_steps=5, max_steps=200, reanchor_min_gap=168, reanchor_dev=0.25,
        volume_per_grid=0.01, fee_rate=0.001, initial_base_ratio=0.5,
        pricetick=0.01, band_mode="daily",
    )


def train_dqn(df, train_start, train_end, feature_mean, feature_std, interval, timesteps, seed):
    def make_env():
        return g.GridGateEnv(
            df, train_start, train_end,
            feature_mean=feature_mean, feature_std=feature_std,
            ma_window=60, atr_window=14, k_width=4.0, target_ret=0.015,
            min_steps=5, max_steps=200, reanchor_min_gap=168, reanchor_dev=0.25,
            volume_per_grid=0.01, fee_rate=0.001, initial_base_ratio=0.5,
            pricetick=0.01, band_mode="daily",
            switch_penalty=SWITCH_PENALTY, decision_interval=interval,
        )

    vec_env = DummyVecEnv([make_env])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    model = DQN(
        "MlpPolicy", vec_env,
        learning_rate=1e-4, buffer_size=100_000, batch_size=128, gamma=0.99,
        train_freq=4, target_update_interval=2000,
        exploration_fraction=0.1, exploration_final_eps=0.02,
        seed=seed, verbose=0,
    )
    model.learn(total_timesteps=timesteps)
    return model


def generate_actions(df, model, test_start, test_end, feature_mean, feature_std, interval):
    n_test = int(((df["timestamp"] >= test_start) & (df["timestamp"] <= test_end)).sum())

    def make_env():
        return g.GridGateEnv(
            df, test_start, test_end,
            feature_mean=feature_mean, feature_std=feature_std,
            ma_window=60, atr_window=14, k_width=4.0, target_ret=0.015,
            min_steps=5, max_steps=200, reanchor_min_gap=168, reanchor_dev=0.25,
            volume_per_grid=0.01, fee_rate=0.001, initial_base_ratio=0.5,
            pricetick=0.01, band_mode="daily",
            switch_penalty=0.0, decision_interval=interval,
        )

    ev = DummyVecEnv([make_env])
    ev = VecFrameStack(ev, n_stack=4)
    obs = ev.reset()
    actions: list[int] = []
    last_a = 0
    for k in range(n_test):
        if k % interval == 0:
            a_arr, _ = model.predict(obs, deterministic=True)
            last_a = int(a_arr[0])
        actions.append(last_a)
        obs, r, done, info = ev.step([last_a])
    return actions


def run_fold(fold_idx: int, df: pd.DataFrame, args) -> list[dict]:
    train_start, train_end, test_start, test_end = FOLDS[fold_idx - 1]
    print(f"\n===== Fold {fold_idx}: train {train_start} ~ {train_end} | test {test_start} ~ {test_end} =====",
          flush=True)

    feature_mean, feature_std = g.get_feature_stats(
        df, train_start, train_end, ma_window=60, atr_window=14,
        k_width=4.0, target_ret=0.015, min_steps=5, max_steps=200,
        pricetick=0.01, band_mode="daily",
    )

    rows: list[dict] = []
    intervals = [int(x) for x in args.intervals.split(",")]

    # 规则基准
    for mode, label in [("no_gate", "永不撤单动态带"), ("ma_gate", "MA门控动态带")]:
        res = simulate_rule(df, test_start, test_end, mode)
        st = res["stats"]
        rows.append({
            "fold": fold_idx, "test_start": test_start, "test_end": test_end,
            "strategy": label, "interval": 0,
            "total_return": st["total_return"], "max_dd": st["max_dd"],
            "sharpe": st["sharpe"], "calmar": st["calmar"], "end_nav": st["end_nav"],
            "trades": res["n_trades"], "reanchor": res["reanchor_count"],
            "switch": res["switch_count"],
        })

    bh = simulate_rule(df, test_start, test_end, "no_gate")["bh_stats"]
    rows.append({
        "fold": fold_idx, "test_start": test_start, "test_end": test_end,
        "strategy": "Buy&Hold", "interval": 0,
        "total_return": bh["total_return"], "max_dd": bh["max_dd"],
        "sharpe": bh["sharpe"], "calmar": bh["calmar"], "end_nav": bh["end_nav"],
        "trades": 1, "reanchor": 0, "switch": 0,
    })

    # DQN 门控
    for interval in intervals:
        t0 = time.time()
        model = train_dqn(df, train_start, train_end, feature_mean, feature_std,
                          interval, args.timesteps, args.seed)
        train_t = time.time() - t0

        actions = generate_actions(df, model, test_start, test_end, feature_mean, feature_std, interval)
        res = simulate_actions(df, test_start, test_end, actions)
        st = res["stats"]
        n_actions = pd.Series(actions)
        rows.append({
            "fold": fold_idx, "test_start": test_start, "test_end": test_end,
            "strategy": f"DQN门控({interval}h)", "interval": interval,
            "total_return": st["total_return"], "max_dd": st["max_dd"],
            "sharpe": st["sharpe"], "calmar": st["calmar"], "end_nav": st["end_nav"],
            "trades": res["n_trades"], "reanchor": res["reanchor_count"],
            "switch": res["switch_count"], "action0_ratio": float(n_actions.eq(0).mean()),
            "train_seconds": round(train_t, 1),
        })
        print(f"  DQN {interval}h: train {train_t:.0f}s | ret {st['total_return']*100:.2f}% | "
              f"dd {st['max_dd']*100:.2f}% | sharpe {st['sharpe']:.3f} | switch {res['switch_count']}",
              flush=True)

    # 增量写入 CSV
    new_df = pd.DataFrame(rows)
    if RESULT_CSV.exists():
        old = pd.read_csv(RESULT_CSV)
        old = old[old["fold"] != fold_idx]
        new_df = pd.concat([old, new_df], ignore_index=True)
    new_df.to_csv(RESULT_CSV, index=False)
    return rows


def summarize(df: pd.DataFrame) -> None:
    df = df.copy()
    df["ret_pct"] = df["total_return"] * 100
    df["dd_pct"] = df["max_dd"] * 100
    print("\n===== 每折明细 =====")
    print(df.pivot_table(index=["fold", "strategy"],
                         values=["ret_pct", "dd_pct", "sharpe", "trades", "switch"]).round(3).to_string())

    dqn4 = df[df["strategy"] == "DQN门控(4h)"].sort_values("fold")
    dqn1 = df[df["strategy"] == "DQN门控(1h)"].sort_values("fold")
    ma = df[df["strategy"] == "MA门控动态带"].sort_values("fold")
    no = df[df["strategy"] == "永不撤单动态带"].sort_values("fold")

    def diff(a, b):
        d_ret = a["total_return"].values - b["total_return"].values
        d_dd = a["max_dd"].values - b["max_dd"].values
        d_sharpe = a["sharpe"].values - b["sharpe"].values
        return {
            "ΔReturn(pp)": float(np.mean(d_ret) * 100),
            "ΔMaxDD(pp)": float(np.mean(d_dd) * 100),
            "ΔSharpe": float(np.mean(d_sharpe)),
            "收益胜率": f"{int((d_ret > 0).sum())}/4",
            "回撤胜率": f"{int((d_dd > 0).sum())}/4",
            "Sharpe胜率": f"{int((d_sharpe > 0).sum())}/4",
        }

    print("\n===== 4h DQN vs 基准（4 折汇总） =====")
    for name, base in [("MA门控", ma), ("永不撤单", no), ("1h DQN", dqn1)]:
        d = diff(dqn4, base)
        d["总收益4h均值%"] = float(dqn4["total_return"].mean() * 100)
        d["总收益基准均值%"] = float(base["total_return"].mean() * 100)
        print(f"  vs {name}: {d}")

    print("\n===== 4 折收益（%） =====")
    for label, sub in [("4h DQN", dqn4), ("1h DQN", dqn1), ("MA门控", ma), ("永不撤单", no)]:
        vals = sub["total_return"].values * 100
        print(f"  {label}: {np.round(vals, 2)}  mean={vals.mean():.2f}  min={vals.min():.2f}  max={vals.max():.2f}")

    # 合成 OOS 净值
    for label, sub in [("4h DQN", dqn4), ("1h DQN", dqn1), ("MA门控", ma), ("永不撤单", no)]:
        comp = (1 + sub["total_return"].values).prod() - 1
        print(f"  {label} 合成OOS收益: {comp*100:.2f}%")


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.data, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    global SWITCH_PENALTY, RESULT_CSV
    SWITCH_PENALTY = args.switch_penalty
    RESULT_CSV = Path(args.result_csv)

    if args.fold > 0:
        run_fold(args.fold, df, args)
    else:
        for f in range(1, 5):
            run_fold(f, df, args)

    if RESULT_CSV.exists():
        summarize(pd.read_csv(RESULT_CSV))
    return 0


if __name__ == "__main__":
    sys.exit(main())
