"""使用 ccxt 从 Binance 拉取 ETH/USDT 的 OHLCV 并缓存为本地 CSV，支持任意时间周期。

用法:
    uv run python scripts/fetch_ohlcv.py --timeframe 1h
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def fetch_ohlcv(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    max_candles_per_request: int = 1000,
) -> pd.DataFrame:
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})

    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    since_ms = exchange.parse8601(f"{start_date}T00:00:00Z")
    end_ms = exchange.parse8601(f"{end_date}T23:59:59Z")

    all_rows: list[list] = []
    while since_ms <= end_ms:
        rows = exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since_ms, limit=max_candles_per_request
        )
        if not rows:
            break
        all_rows.extend(rows)
        since_ms = rows[-1][0] + timeframe_ms
        print(f"fetched {len(rows)} rows, latest: {exchange.iso8601(rows[-1][0])}")

    if not all_rows:
        raise RuntimeError("no data fetched")

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])

    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    df = df[(ts >= start_ts) & (ts <= end_ts)].copy()

    fmt = "%Y-%m-%d" if timeframe.endswith("d") else "%Y-%m-%d %H:%M:%S"
    df["timestamp"] = ts.dt.strftime(fmt)

    df = df.drop_duplicates(subset="timestamp", keep="first").sort_values("timestamp").reset_index(drop=True)
    return df


def main() -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description="拉取 OHLCV 并缓存为 CSV")
    parser.add_argument("--exchange", default="binanceus")
    parser.add_argument("--symbol", default="ETH/USDT")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default=today)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "eth_usdt_1d.csv"),
        help="输出 CSV 路径",
    )
    args = parser.parse_args()

    df = fetch_ohlcv(args.exchange, args.symbol, args.timeframe, args.start, args.end)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"saved {len(df)} rows -> {out_path}")
    print(df.head(3).to_string(index=False))
    print(df.tail(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
