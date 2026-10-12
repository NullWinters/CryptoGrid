"""七币种 DQN 门控策略统一训练与回测。

币种：BTC, ETH, BNB, PAXG, SOL, TRX, XLM
配置：A1 账户状态（position_ratio, cash_ratio），1h 决策，R0 奖励，100k 步，seed 42。
网格：日频等价动态带（方案 A），每格名义金额约 GRID_NOTIONAL USDT。
模型统一保存至 models/，结果保存至 data/multi_coin_dqn_results.csv。
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

COINS = [
    # (coin, pricetick, grid_notional_usdt)
    ("BTC", 0.01, 1000.0),
    ("ETH", 0.01, 1000.0),
    ("BNB", 0.01, 1000.0),
    ("PAXG", 0.01, 1000.0),
    ("SOL", 0.001, 1000.0),
    ("TRX", 0.0001, 1000.0),
    ("XLM", 0.0001, 1000.0),
]

TRAIN_START = "2022-01-01"
TRAIN_END = "2025-12-31 23:00:00"
TEST_START = "2026-01-01"
TEST_END = "2026-09-30 23:00:00"

MODEL_DIR = PROJECT_ROOT / "models"
RESULT_CSV = PROJECT_ROOT / "data" / "multi_coin_dqn_results.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--coins", default="BTC,ETH,BNB,PAXG,SOL,TRX,XLM")
    p.add_argument("--model-dir", default=str(MODEL_DIR))
    p.add_argument("--result-csv", default=str(RESULT_CSV))
    return p.parse_args()


def make_env(df, start, end, feature_mean, feature_std, pricetick, volume_per_grid, *, eval_mode=False):
    return g.GridGateEnv(
        df, start, end,
        feature_mean=feature_mean, feature_std=feature_std,
        ma_window=60, atr_window=14, k_width=4.0, target_ret=0.015,
        min_steps=5, max_steps=200, reanchor_min_gap=168, reanchor_dev=0.25,
        volume_per_grid=volume_per_grid, fee_rate=0.001, initial_base_ratio=0.5,
        pricetick=pricetick, band_mode="daily",
        switch_penalty=0.0 if eval_mode else 0.05,
        decision_interval=1,
        asym_neg=1.0, lambda_down=0.0, lambda_dd=0.0, asymmetric_switch=False,
        use_account_state=True,
        account_features=["position_ratio", "cash_ratio"],
    )


def simulate_rule(df, start, end, pricetick, volume_per_grid, mode):
    return g.simulate(
        df, start, end, mode=mode,
        ma_window=60, atr_window=14, k_width=4.0, target_ret=0.015,
        min_steps=5, max_steps=200, reanchor_min_gap=168, reanchor_dev=0.25,
        volume_per_grid=volume_per_grid, fee_rate=0.001, initial_base_ratio=0.5,
        pricetick=pricetick, band_mode="daily",
    )


def train_eval(coin, pricetick, volume_per_grid, args):
    df = pd.read_csv(PROJECT_ROOT / "data" / f"{coin.lower()}_usdt_1h.csv", parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    feature_mean, feature_std = g.get_feature_stats(
        df, TRAIN_START, TRAIN_END, ma_window=60, atr_window=14, k_width=4.0,
        target_ret=0.015, min_steps=5, max_steps=200, pricetick=pricetick, band_mode="daily",
    )

    model_path = Path(args.model_dir) / f"dqn_gate_{coin.lower()}_a1_1h.zip"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    def make_train():
        return make_env(df, TRAIN_START, TRAIN_END, feature_mean, feature_std,
                        pricetick, volume_per_grid)
    vec = DummyVecEnv([make_train]); vec = VecFrameStack(vec, n_stack=4)
    model = DQN("MlpPolicy", vec, learning_rate=1e-4, buffer_size=100_000, batch_size=128,
                gamma=0.99, train_freq=4, target_update_interval=2000,
                exploration_fraction=0.1, exploration_final_eps=0.02, seed=args.seed, verbose=0)
    model.learn(total_timesteps=args.timesteps)
    model.save(model_path)
    train_s = time.time() - t0

    n_test = int(((df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)).sum())
    def make_test():
        return make_env(df, TEST_START, TEST_END, feature_mean, feature_std,
                        pricetick, volume_per_grid, eval_mode=True)
    ev = DummyVecEnv([make_test]); ev = VecFrameStack(ev, n_stack=4)
    obs = ev.reset(); actions = []
    for k in range(n_test):
        a_arr, _ = model.predict(obs, deterministic=True)
        a = int(a_arr[0]); actions.append(a)
        obs, r, done, info = ev.step([a])

    rl = g.simulate(df, TEST_START, TEST_END, actions=actions,
                    ma_window=60, atr_window=14, k_width=4.0, target_ret=0.015,
                    min_steps=5, max_steps=200, reanchor_min_gap=168, reanchor_dev=0.25,
                    volume_per_grid=volume_per_grid, fee_rate=0.001, initial_base_ratio=0.5,
                    pricetick=pricetick, band_mode="daily")
    st = rl["stats"]

    return {
        "coin": coin,
        "model": str(model_path),
        "train_s": round(train_s, 1),
        "action0_ratio": float(pd.Series(actions).eq(0).mean()),
        "rl_ret": st["total_return"],
        "rl_dd": st["max_dd"],
        "rl_sharpe": st["sharpe"],
        "rl_calmar": st["calmar"],
        "rl_end_nav": st["end_nav"],
        "rl_trades": rl["n_trades"],
        "rl_reanchor": rl["reanchor_count"],
        "rl_switch": rl["switch_count"],
    }


def run_coin(coin, pricetick, grid_notional, args):
    df = pd.read_csv(PROJECT_ROOT / "data" / f"{coin.lower()}_usdt_1h.csv", parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    start_price = float(df[df["timestamp"] >= TRAIN_START]["close"].iloc[0])
    volume_per_grid = grid_notional / start_price

    print(f"\n===== {coin}: start_price={start_price:.4f}, vol_per_grid={volume_per_grid:.6g}, tick={pricetick} =====", flush=True)
    row = train_eval(coin, pricetick, volume_per_grid, args)

    # 基准
    for mode, label in [("no_gate", "no_gate"), ("ma_gate", "ma_gate")]:
        res = simulate_rule(df, TEST_START, TEST_END, pricetick, volume_per_grid, mode)
        st = res["stats"]
        row[f"{label}_ret"] = st["total_return"]
        row[f"{label}_dd"] = st["max_dd"]
        row[f"{label}_sharpe"] = st["sharpe"]
        row[f"{label}_calmar"] = st["calmar"]
        row[f"{label}_end_nav"] = st["end_nav"]
        row[f"{label}_trades"] = res["n_trades"]
        row[f"{label}_reanchor"] = res["reanchor_count"]
        row[f"{label}_switch"] = res["switch_count"]
    bh = simulate_rule(df, TEST_START, TEST_END, pricetick, volume_per_grid, "no_gate")["bh_stats"]
    row["bh_ret"] = bh["total_return"]
    row["bh_dd"] = bh["max_dd"]
    row["bh_sharpe"] = bh["sharpe"]
    row["bh_end_nav"] = bh["end_nav"]

    print(f"  RL: ret {row['rl_ret']*100:.2f}% dd {row['rl_dd']*100:.2f}% sharpe {row['rl_sharpe']:.3f} "
          f"switch {row['rl_switch']} trades {row['rl_trades']}", flush=True)
    print(f"  MA: ret {row['ma_gate_ret']*100:.2f}% dd {row['ma_gate_dd']*100:.2f}% | "
          f"no_gate: ret {row['no_gate_ret']*100:.2f}% dd {row['no_gate_dd']*100:.2f}% | "
          f"BH: ret {row['bh_ret']*100:.2f}% dd {row['bh_dd']*100:.2f}%", flush=True)
    return row


def main():
    args = parse_args()
    coins = [c.strip() for c in args.coins.split(",") if c.strip()]
    config = {c.upper(): (tick, notion) for c, tick, notion in COINS}

    rows = []
    for coin in coins:
        tick, notion = config[coin.upper()]
        row = run_coin(coin.upper(), tick, notion, args)
        rows.append(row)
        new = pd.DataFrame(rows)
        new.to_csv(args.result_csv, index=False)
        print(f"{coin} 已写入 {args.result_csv}", flush=True)

    print("\n===== 全部完成 =====")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
