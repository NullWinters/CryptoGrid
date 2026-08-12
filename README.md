# CryptoGrid

加密货币网格交易法策略研究。

## 仓库结构

```
notebook/   # 研究 notebook（简单/动态/门控/动态带网格、波动率、行情分类）
scripts/    # 数据拉取、回测、参数搜索、组合优化脚本
data/       # 多币种日线 OHLCV 与 vnpy 数据实验室
pic/        # 回测与组合优化图片
策略报告.pdf # 策略报告
```

## 脚本

| 脚本 | 用途 |
|---|---|
| `fetch_ohlcv.py` / `append_ohlcv.py` | ccxt 拉取/追加 OHLCV 到 data/ |
| `dynamic_grid_backtest.py` | 日频动态网格回测（`--print-json` 供搜索） |
| `search_grid_params.sh` | 网格参数搜索（xargs 并行，按 Sharpe 取优） |
| `portfolio_optimize.py` | 风险平价组合优化（ETH/TRX 网格 + PAXG 持有） |
| `predict_regime.py` | XGBoost 预测市场震荡/单边 |

## 快速开始

```bash
# 拉取数据 → 回测 → 参数搜索 → 组合优化
uv run python scripts/fetch_ohlcv.py --symbol ETH/USDT --timeframe 1d
uv run python scripts/dynamic_grid_backtest.py --data data/eth_usdt_1d.csv --print-json
bash scripts/search_grid_params.sh
uv run python scripts/portfolio_optimize.py
```

## 策略报告

见 [策略报告.pdf](策略报告.pdf)：网格交易法在 ETH/TRX 上的回测与三资产风险平价组合优化。
