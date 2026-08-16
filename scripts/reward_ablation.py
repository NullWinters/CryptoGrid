"""奖励函数消融实验：R0~R4。

用法：
    python scripts/reward_ablation.py --variant R1          # 只跑 R1
    python scripts/reward_ablation.py --variant all         # 跑全部（耗时较长）

每个变体：4 折 walk-forward，训练 1h DQN（100k 步，seed 42），测试段回测。
结果增量写入 data/reward_ablation_results.csv。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.dqn_gate_env as g
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

FOLDS = [
    ("2022-01-01", "2023-12-31 23:00:00", "2024-01-01", "2024-06-30 23:00:00"),
    ("2022-01-01", "2024-06-30 23:00:00", "2024-07-01", "2024-12-31 23:00:00"),
    ("2022-01-01", "2024-12-31 23:00:00", "2025-01-01", "2025-06-30 23:00:00"),
    ("2022-01-01", "2025-06-30 23:00:00", "2025-07-01", "2025-12-31 23:00:00"),
]

RESULT_CSV = PROJECT_ROOT / "data" / "reward_ablation_results.csv"

VARIANTS = {
    # 变体: (asym_neg, lambda_down, lambda_dd, dd_threshold, switch_penalty, asymmetric_switch, switch_off, switch_on)
    "R0": (1.0, 0.0, 0.0, 0.08, 0.05, False, 0.02, 0.10),  # 当前基线：对称差分 + 对称切换 0.05
    "R1": (2.0, 0.0, 0.0, 0.08, 0.00, False, 0.02, 0.10),  # 负差分放大 2 倍，无切换惩罚
    "R2": (2.0, 0.1, 0.0, 0.08, 0.00, False, 0.02, 0.10),  # R1 + 组合亏损惩罚
    "R3": (2.0, 0.0, 0.0, 0.08, 0.00, True, 0.02, 0.10),   # R1 + 非对称切换
    "R4": (2.0, 0.0, 0.5, 0.08, 0.00, False, 0.02, 0.10),  # R1 + 回撤阈值惩罚
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(PROJECT_ROOT / "data" / "btc_usdt_1h.csv"))
    p.add_argument("--variant", default="all", help="R0/R1/R2/R3/R4/all")
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--result-csv", default=str(RESULT_CSV))
    p.add_argument("--intervals", default="1", help="决策间隔，默认 1（1h）")
    return p.parse_args()


def simulate_rule(df, start, end, mode):
    return g.simulate(
        df, start, end, mode=mode,
        ma_window=60, atr_window=14, k_width=4.0, target_ret=0.015,
        min_steps=5, max_steps=200, reanchor_min_gap=168, reanchor_dev=0.25,
        volume_per_grid=0.01, fee_rate=0.001, initial_base_ratio=0.5,
        pricetick=0.01, band_mode="daily",
    )


def make_env(df, start, end, feature_mean, feature_std, reward_params, decision_interval):
    asym_neg, lambda_down, lambda_dd, dd_threshold, switch_penalty, asym_switch, sw_off, sw_on = reward_params
    return g.GridGateEnv(
        df, start, end,
        feature_mean=feature_mean, feature_std=feature_std,
        ma_window=60, atr_window=14, k_width=4.0, target_ret=0.015,
        min_steps=5, max_steps=200, reanchor_min_gap=168, reanchor_dev=0.25,
        volume_per_grid=0.01, fee_rate=0.001, initial_base_ratio=0.5,
        pricetick=0.01, band_mode="daily",
        switch_penalty=switch_penalty, decision_interval=decision_interval,
        asym_neg=asym_neg, lambda_down=lambda_down, lambda_dd=lambda_dd,
        dd_threshold=dd_threshold, asymmetric_switch=asym_switch,
        switch_penalty_off=sw_off, switch_penalty_on=sw_on,
    )


def train_and_eval(df, train_start, train_end, test_start, test_end,
                   feature_mean, feature_std, reward_params, timesteps, seed, interval):
    model = None
    # 训练
    def make_train():
        return make_env(df, train_start, train_end, feature_mean, feature_std, reward_params, interval)
    vec = DummyVecEnv([make_train]); vec = VecFrameStack(vec, n_stack=4)
    model = DQN("MlpPolicy", vec, learning_rate=1e-4, buffer_size=100_000, batch_size=128,
                gamma=0.99, train_freq=4, target_update_interval=2000,
                exploration_fraction=0.1, exploration_final_eps=0.02, seed=seed, verbose=0)
    t0 = time.time()
    model.learn(total_timesteps=timesteps)
    train_s = time.time() - t0

    # 测试
    n_test = int(((df["timestamp"] >= test_start) & (df["timestamp"] <= test_end)).sum())
    def make_test():
        return make_env(df, test_start, test_end, feature_mean, feature_std, reward_params, interval)
    ev = DummyVecEnv([make_test]); ev = VecFrameStack(ev, n_stack=4)
    obs = ev.reset(); actions = []; last = 0
    for k in range(n_test):
        if k % interval == 0:
            a_arr, _ = model.predict(obs, deterministic=True)
            last = int(a_arr[0])
        actions.append(last)
        obs, r, done, info = ev.step([last])
    res = g.simulate(df, test_start, test_end, actions=actions,
                     ma_window=60, atr_window=14, k_width=4.0, target_ret=0.015,
                     min_steps=5, max_steps=200, reanchor_min_gap=168, reanchor_dev=0.25,
                     volume_per_grid=0.01, fee_rate=0.001, initial_base_ratio=0.5,
                     pricetick=0.01, band_mode="daily")
    st = res["stats"]
    return {
        "total_return": st["total_return"], "max_dd": st["max_dd"], "sharpe": st["sharpe"],
        "calmar": st["calmar"], "end_nav": st["end_nav"],
        "trades": res["n_trades"], "reanchor": res["reanchor_count"], "switch": res["switch_count"],
        "action0_ratio": float(pd.Series(actions).eq(0).mean()), "train_s": round(train_s, 1),
    }, actions


def run_variant(variant, df, args):
    reward_params = VARIANTS[variant]
    print(f"\n===== {variant}: {reward_params} =====", flush=True)
    rows = []
    for f, (train_start, train_end, test_start, test_end) in enumerate(FOLDS, 1):
        feature_mean, feature_std = g.get_feature_stats(
            df, train_start, train_end, ma_window=60, atr_window=14, k_width=4.0,
            target_ret=0.015, min_steps=5, max_steps=200, pricetick=0.01, band_mode="daily",
        )
        interval = int(args.intervals.split(",")[0])
        t0 = time.time()
        m, actions = train_and_eval(df, train_start, train_end, test_start, test_end,
                                    feature_mean, feature_std, reward_params, args.timesteps, args.seed, interval)
        m = {k: (round(v, 6) if isinstance(v, (int, float, np.floating)) else v) for k, v in m.items()}
        m.update({
            "variant": variant, "fold": f, "test_start": test_start, "test_end": test_end,
            "strategy": f"DQN门控({interval}h)",
        })
        rows.append(m)
        print(f"  fold {f}: ret {m['total_return']*100:.2f}% dd {m['max_dd']*100:.2f}% "
              f"sharpe {m['sharpe']:.3f} switch {m['switch']} trades {m['trades']} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # 增量写 CSV
    new = pd.DataFrame(rows)
    csv = Path(args.result_csv)
    if csv.exists():
        old = pd.read_csv(csv)
        old = old[old["variant"] != variant]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(csv, index=False)
    print(f"{variant} 已写入 {csv}", flush=True)
    return rows


def run_baselines(df, args):
    """基准只跑一次，写入 CSV（variant='baseline'）。"""
    rows = []
    for f, (_, _, test_start, test_end) in enumerate(FOLDS, 1):
        for mode, label in [("no_gate", "永不撤单动态带"), ("ma_gate", "MA门控动态带")]:
            res = simulate_rule(df, test_start, test_end, mode)
            st = res["stats"]
            rows.append({
                "variant": "baseline", "fold": f, "test_start": test_start, "test_end": test_end,
                "strategy": label,
                "total_return": st["total_return"], "max_dd": st["max_dd"], "sharpe": st["sharpe"],
                "calmar": st["calmar"], "end_nav": st["end_nav"],
                "trades": res["n_trades"], "reanchor": res["reanchor_count"], "switch": res["switch_count"],
                "action0_ratio": np.nan, "train_s": 0.0,
            })
    csv = Path(args.result_csv)
    if csv.exists():
        old = pd.read_csv(csv)
        old = old[old["variant"] != "baseline"]
        rows = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    else:
        rows = pd.DataFrame(rows)
    rows.to_csv(csv, index=False)
    print("baselines 已写入", csv)


def main():
    args = parse_args()
    df = pd.read_csv(args.data, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    run_baselines(df, args)

    if args.variant == "all":
        for v in ["R0", "R1", "R2", "R3", "R4"]:
            run_variant(v, df, args)
    else:
        run_variant(args.variant, df, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
