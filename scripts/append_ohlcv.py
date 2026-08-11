"""使用 ccxt 从 Binance 拉取 ETH/USDT 的 OHLCV 并追加到本地 CSV。

根据已有 CSV 的最后一条时间戳，拉取缺失区间并追加去重。

用法:
    uv run python scripts/append_ohlcv.py --timeframe 1d
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def append_ohlcv(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    csv_path: Path,
    max_candles_per_request: int = 1000,
) -> None:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    last_ts = df["timestamp"].max()

    fmt = "%Y-%m-%d" if timeframe.endswith("d") else "%Y-%m-%d %H:%M:%S"
    last_str = last_ts.strftime(fmt)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    now_ms = exchange.milliseconds()
    if last_ts.timestamp() * 1000 + timeframe_ms > now_ms:
        print(f"{csv_path.name}: 已是最新（最后 {last_str}），无需追加")
        return

    new = _fetch_ohlcv(exchange, symbol, timeframe, last_str, today, max_candles_per_request)
    if new.empty:
        print(f"{csv_path.name}: 无新增数据（最后 {last_str}）")
        return

    new["timestamp"] = pd.to_datetime(new["timestamp"]).dt.tz_localize(None)
    new = new[new["timestamp"] > last_ts].copy()
    if new.empty:
        print(f"{csv_path.name}: 无新增数据（最后 {last_str}）")
        return
    combined = pd.concat([df, new], ignore_index=True)
    combined = (
        combined.drop_duplicates(subset="timestamp", keep="first")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    combined["timestamp"] = combined["timestamp"].dt.strftime(fmt)
    combined.to_csv(csv_path, index=False)
    print(f"{csv_path.name}: 追加 {len(new)} 行，共 {len(combined)} 行")
    print(combined.tail(3).to_string(index=False))


def _fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_str: str,
    end_date: str,
    max_candles_per_request: int = 1000,
) -> pd.DataFrame:

    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    since_ms = exchange.parse8601(f"{since_str}T00:00:00Z")
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
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[df["timestamp"] < pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)].copy()
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S" if not timeframe.endswith("d") else "%Y-%m-%d")
    return df.drop_duplicates(subset="timestamp", keep="first").sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="追加缺失 OHLCV 到 CSV")
    parser.add_argument("--exchange", default="binanceus")
    parser.add_argument("--symbol", default="ETH/USDT")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "eth_usdt_1d.csv"),
        help="输出 CSV 路径",
    )
    args = parser.parse_args()

    append_ohlcv(args.exchange, args.symbol, args.timeframe, Path(args.output))


if __name__ == "__main__":
    sys.exit(main())
