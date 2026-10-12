# CryptoGrid

加密货币网格交易法策略研究。

## 仓库结构

```
notebook/   # 研究 notebook，分 回测/ 与 数据探索/ 两个子目录
scripts/    # 数据拉取、回测、参数搜索、DQN 训练/消融、组合优化、上界实验、报告生成脚本
data/       # 多币种日线与小时线 OHLCV、vnpy 数据实验室、回测运行数据
models/     # 七币种 DQN 门控模型
pic/        # 回测与组合优化图片
时频策略报告.pdf # 七币种 DQN 门控策略风险平价组合报告
```

## 脚本

| 脚本                                 | 用途                                                   |
| ------------------------------------ | ------------------------------------------------------ |
| `fetch_ohlcv.py` / `append_ohlcv.py` | ccxt 拉取/追加 OHLCV 到 data/                          |
| `fetch_binance_vision.py`            | 从 Binance 官方数据站拉取七币种现货 1h OHLCV          |
| `dynamic_grid_backtest.py`           | 日频动态网格回测（`--print-json` 供搜索）              |
| `search_grid_params.sh`              | 网格参数搜索（xargs 并行，按 Sharpe 取优）             |
| `predict_regime.py`                  | XGBoost 预测市场震荡/单边                              |
| `dqn_gate_env.py`                    | DQN 门控 Gym 环境、网格撮合模拟器、特征/动态带构建     |
| `vnpy_hourly_grid_backtest.py`       | vnpy 小时频动态带回测，支持静态动作注入与 DQN 在线推理 |
| `walk_forward_decision_interval.py`  | Walk-forward 验证 1h/4h 决策频率稳健性                 |
| `reward_ablation.py`                 | 奖励函数消融实验（R0~R4）                              |
| `account_state_ablation.py`          | 账户状态消融实验（A0~A4）                              |
| `multi_coin_dqn.py`                  | 七币种 DQN 门控策略训练与回测                          |
| `portfolio_optimize.py`              | 日频三资产风险平价组合优化（ETH/TRX 网格 + PAXG 持有） |
| `portfolio_optimize_7coins.py`       | 七币种 DQN 门控策略季度风险平价组合回测（cvxopt 优化） |
| `window_mi_research.py` 系列         | 互信息-滞后研究，生成并执行 `window_research.ipynb`    |
| `grid_upper_bound.py`                | 网格收益上界实验引擎，总变差 TV 上界验证               |
| `generate_report_pdf.py`             | 生成《时频策略报告》HTML 并调用 Chrome 打印为 PDF      |

## 快速开始

```bash
# 拉取数据 → 回测 → 参数搜索 → 日频组合优化
uv run python scripts/fetch_binance_vision.py            # 七币种现货 1h（官方镜像）
uv run python scripts/fetch_ohlcv.py --symbol ETH/USDT --timeframe 1d
uv run python scripts/dynamic_grid_backtest.py --data data/eth_usdt_1d.csv --print-json
bash scripts/search_grid_params.sh
uv run python scripts/portfolio_optimize.py

# 七币种 DQN 门控策略训练与回测（模型保存至 models/）
uv run python scripts/multi_coin_dqn.py

# 七币种季度风险平价组合回测（结果保存至 data/portfolio_7coins_*）
uv run python scripts/portfolio_optimize_7coins.py

# 网格收益上界实验自检（内联数据，无需 data/）
uv run python scripts/grid_upper_bound.py

# 生成《时频策略报告》PDF
uv run python scripts/generate_report_pdf.py
```

## 策略报告

- [时频策略报告.pdf](时频策略报告.pdf)：七币种 DQN 门控策略风险平价组合的策略说明、回测结果、关键指标与风险分析。
