# 项目规则

加密货币网格交易法策略研究，纯 Python 脚本 + Jupyter 研究，无测试/lint/CI 配置。

## 目录约定

- 数据文件统一保存在 `data/`，Jupyter Notebook 统一保存在 `notebook/`，Python 脚本统一保存在 `scripts/`，图片统一保存在 `pic/`。
- `models/` 存放已训练的 DQN 模型（`.zip`，已纳入 git）；`data/` 与 `*.html` 被 `.gitignore` 忽略，`data/` 内容需用 `scripts/fetch_ohlcv.py` / `append_ohlcv.py` 重新拉取，不要假设其存在。
- 在 notebook 中直接展示图片，不要额外保存到 `pic/`。

## 运行环境

- Python >= 3.13，使用 uv 管理，所有脚本从仓库根目录以 `uv run python scripts/<name>.py` 运行。
- 脚本内部用 `Path(__file__).resolve().parent.parent` 定位仓库根，不依赖调用时的 cwd。
- 仓库无 pytest/ruff/mypy/Makefile/CI；验证方式就是直接运行脚本并检查其输出 CSV/JSON。

## 数据约定

- OHLCV 文件命名为 `{coin}_usdt_{timeframe}.csv`（小写币种，如 `eth_usdt_1d.csv`、`btc_usdt_1h.csv`），列为 `timestamp,open,high,low,close,volume`。
- 使用 duckdb 查阅 `data/` 下的表数据（如 CSV）进行校验。

## 脚本约定与陷阱

- 脚本间复用模块靠 `sys.path.insert(0, str(PROJECT_ROOT))` 后 `import scripts.dqn_gate_env as g`；例外：`portfolio_optimize.py` 插入的是 `PROJECT_ROOT/"scripts"` 并直接 `import dynamic_grid_backtest as dgb`，改导入方式时注意两者不同。
- `dynamic_grid_backtest.py` 与 `vnpy_hourly_grid_backtest.py` 在 `import vnpy.alpha` 前会向 `sys.modules` 注入 `alphalens/talib/torch/lightgbm` 空模块，以支持无头运行；不要删除这段兼容代码。
- 回测设计强制无前视：技术指标只用 `rolling` + `ffill`（禁止 `bfill`），且 START 前预热天数必须 >= `max(ma_window, gate_window, atr_window)`，不足会报错退出。修改指标计算时须保持此约束。
- `search_grid_params.sh` 通过 xargs 并行调用 `dynamic_grid_backtest.py --print-json`，参数可用环境变量覆盖（`DATA`、`MA_WINDOWS`、`METRIC`、`PARALLEL` 等）；其聚合排序用系统 `python3` 仅标准库。

## 策略与模型约定

- DQN 门控策略的规范配置：A1 账户状态（`position_ratio`/`cash_ratio`）、1h 决策、R0 奖励、100k 步、seed 42、4 折 walk-forward；统一训练/回测入口为 `multi_coin_dqn.py`，特征由 `dqn_gate_env.get_feature_stats` 在训练区间统计。
- 训练/测试区间常量在各脚本中重复：训练 `2022-01-01 ~ 2025-12-31 23:00:00`，测试 `2026-01-01 ~ 2026-07-31 23:00:00`；改动需同步多个脚本。
- 七币种模型命名 `dqn_gate_{coin}_a1_1h.zip`（币种小写），保存在 `models/`。
- `generate_report_pdf.py` 依赖 `data/portfolio_7coins_*` 与 `data/multi_coin_dqn_results.csv`，并调用系统 `google-chrome --headless` 打印 PDF；缺少数据或 Chrome 会失败。