"""从 Binance 官方公开数据站 data.binance.vision 拉取现货 1h OHLCV 到 data/{coin}_usdt_1h.csv。

api.binance.com 在本环境被地域封锁（451），本脚本改用官方镜像：历史月份用月度 zip，
当月用日度 zip（T+1 发布，最新一根 bar 为昨日 23:00）。

用法:
    uv run python scripts/fetch_binance_vision.py                # 七币种
    uv run python scripts/fetch_binance_vision.py --coins BTC,ETH
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

COINS = ["BTC", "ETH", "BNB", "PAXG", "SOL", "TRX", "XLM"]
BASE = "https://data.binance.vision/data/spot"
COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _fetch_zip(url: str) -> pd.DataFrame | None:
    """下载并解析 klines zip（无表头，12 列），404 返回 None。"""
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            payload = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    with zipfile.ZipFile(BytesIO(payload)) as zf:
        with zf.open(zf.namelist()[0]) as f:
            raw = pd.read_csv(f, header=None)
    raw = raw.iloc[:, :6]
    # 2025 年起的新文件 open_time 用微秒，统一归一到毫秒
    if raw[0].max() > 1e14:
        raw[0] = raw[0] // 1000
    return raw


def download_coin(coin: str, start: str = "2021-01-01") -> pd.DataFrame:
    symbol = f"{coin}USDT"
    start_ts = pd.Timestamp(start)
    today = datetime.now(timezone.utc).date()
    frames: list[pd.DataFrame] = []

    # 历史完整月份
    for y in range(start_ts.year, today.year + 1):
        for m in range(1, 13):
            month = pd.Timestamp(year=y, month=m, day=1)
            if month < start_ts.replace(day=1) or month >= pd.Timestamp(today).replace(day=1, hour=0):
                continue
            url = f"{BASE}/monthly/klines/{symbol}/1h/{symbol}-1h-{y}-{m:02d}.zip"
            df = _fetch_zip(url)
            if df is None:
                print(f"{coin} {y}-{m:02d}: 404 跳过")
                continue
            frames.append(df)
            print(f"{coin} {y}-{m:02d}: {len(df)} 行", flush=True)

    # 当月日度（T+1 发布）
    d = today.replace(day=1)
    while d < today:
        url = f"{BASE}/daily/klines/{symbol}/1h/{symbol}-1h-{d.isoformat()}.zip"
        df = _fetch_zip(url)
        if df is None:
            break  # 未发布的日度文件即为当前边界
        frames.append(df)
        print(f"{coin} {d.isoformat()}: {len(df)} 行", flush=True)
        d += timedelta(days=1)

    out = pd.concat(frames, ignore_index=True)
    out.columns = COLS
    ts = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
    out = out[(ts >= start_ts.tz_localize("UTC"))].copy()
    out["timestamp"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    for c in COLS[1:]:
        out[c] = out[c].astype(float)
    return (
        out.drop_duplicates(subset="timestamp", keep="first")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--coins", default=",".join(COINS))
    p.add_argument("--start", default="2021-01-01")
    args = p.parse_args()

    for coin in (c.strip().upper() for c in args.coins.split(",") if c.strip()):
        df = download_coin(coin, args.start)
        # 落盘前自检：等距小时序列，缺口数应为 0 或个位数
        idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp"]))
        gaps = int((idx.to_series().diff().dropna() != pd.Timedelta(hours=1)).sum())
        out_path = PROJECT_ROOT / "data" / f"{coin.lower()}_usdt_1h.csv"
        df.to_csv(out_path, index=False)
        print(f"saved {len(df)} rows, gaps={gaps}, {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]} -> {out_path}\n", flush=True)


if __name__ == "__main__":
    sys.exit(main())
