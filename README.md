# CryptoGrid

加密货币网格交易策略研究：数据拉取、回测、参数搜索与策略报告。

- 网格交易法的 **alpha 来源于震荡行情**，在明显单边行情中跑不过持有，但可有效减小回撤
- 策略详情与完整回测结果见根目录 [策略报告.pdf](策略报告.pdf)

## 仓库结构

```
notebook/   # 研究 notebook：简单/动态/MA门控/动态带/XGB门控网格、波动率、行情分类
scripts/    # 数据拉取、回测、参数搜索、PDF 报告生成脚本
data/       # OHLCV 缓存（btc/eth/sol 日线、eth 小时线）与 vnpy 数据实验室
pic/        # 策略报告图片（训练/验证集净值、区间、回撤）
策略报告.pdf # 策略完整报告（简体中文）
```

## 快速开始

```bash
# 1. 拉取日线数据（示例：ETH）
uv run python scripts/fetch_ohlcv.py --symbol ETH/USDT --timeframe 1d

# 2. 回测（默认参数，2021-04-01 ~ 2025-12-31 训练集）
uv run python scripts/dynamic_grid_backtest.py

# 3. 参数搜索（1500 组网格并行，按 Sharpe 取优）
bash scripts/search_grid_params.sh

# 4. 生成策略报告 PDF（HTML + headless Chrome，需安装 google-chrome）
uv run python scripts/generate_pdf_report.py
```

## 脚本说明

### `dynamic_grid_backtest.py` — 日频动态网格回测

动态网格（动态上下界 + 动态层数 + 均线门控）回测，**无前视偏差**：滚动窗口仅 ffill、无 bfill；`--start auto` 时自动从数据起始后 `max(ma,gate,atr)` 个交易日开始（保证预热）。

```bash
uv run python scripts/dynamic_grid_backtest.py \
  --data data/eth_usdt_1d.csv --start auto --end 2025-12-31 \
  --ma-window 10 --gate-window 90 --k-width 2 --target-ret 0.03 --atr-window 7 \
  --print-json          # 输出单行 JSON（供参数搜索解析）
  --plot pic --plot-prefix train_   # 生成净值/区间/回撤 3 张图
```

主要参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--ma-window` | 60 | 中枢均线周期（动态上下界 mid） |
| `--gate-window` | 60 | 门控均线周期：`close<MA(gate)` 撤单观望 |
| `--k-width` | 4.0 | 通道半宽 = K × ATR |
| `--target-ret` | 0.015 | 单格目标收益率（间距 = mid × target_ret） |
| `--min-steps / --max-steps` | 5 / 200 | 动态层数上下限 |
| `--reanchor-gap / --reanchor-dev` | 7 / 0.25 | 重锚定冷却期 / 偏离中枢阈值 |
| `--start` | auto | `auto` 或显式日期（如 2026-01-01） |
| `--end` | 2025-12-31 | 回测结束日期 |
| `--print-json` | — | 输出单行 JSON 指标 |
| `--plot DIR --plot-prefix P` | — | 生成报告图片到 DIR |

输出指标：总收益、年化收益、年化波动、最大回撤（含持续天数）、Sharpe、胜率（配对）、换手率、成交笔数、盈利/亏损天数、重锚定次数。

### `search_grid_params.sh` — 参数搜索

网格扫描并输出 Top 8 + 最佳参数（按 Sharpe）。`START=auto` 时每组从各自最早可回测日回测到 `END`，故各组区间长度不同，仅比较 Sharpe 等年化指标。

```bash
bash scripts/search_grid_params.sh                      # 默认 1500 组
END=2022-12-31 METRIC=return_dd PARALLEL=4 bash scripts/search_grid_params.sh
MA_WINDOWS="20 30 60" GATE_WINDOWS="20 60" bash scripts/search_grid_params.sh
```

默认网格（环境变量可覆盖）：`MA{10,20,30,60,90} × GATE{10,20,30,60,90} × K{2,3,4,5} × RET{0.005,0.01,0.015,0.02,0.03} × ATR{7,14,21}`，`MAX_STEPS=300`。已过滤回撤 >80% 的崩溃样本。

### 其他脚本

- `fetch_ohlcv.py` / `append_ohlcv.py` — 用 ccxt 拉取/追加 OHLCV 到 CSV
- `predict_regime.py` — XGBoost 预测未来市场震荡/单边（多币种）
- `generate_pdf_report.py` — 用 HTML + headless Chrome 生成根目录「策略报告.pdf」（含 6 张图、简体中文）

## 数据口径

- 年化按 **365 个交易日**换算；0.1% 单边手续费；现货**无复权**；基准 = 简单持有
- 数据来源：Binance（ccxt），缓存于 `data/` 下 CSV

## 策略报告

完整策略说明、回测结果、关键指标与风险分析见 [策略报告.pdf](策略报告.pdf)。核心结论：

- 训练集（2021-04 ~ 2025-12，震荡偏多）：收益 **+46.29%**、回撤 **-14.58%**、Sharpe 0.84
- 验证集（2026-01 ~ 2026-08，单边下跌）：收益 **-2.07%**（持有 -37.95%）、回撤 **-3.78%**（持有 -53.33%）
- 单边行情跑不过持有，但回撤控制显著优于持有
