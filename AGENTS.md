# 项目规则

加密货币网格交易法策略研究，纯 Python 脚本 + Jupyter 研究，无 pytest/lint/mypy/Makefile/CI；验证方式就是直接运行脚本并检查其输出 CSV/JSON。

## 目录与数据

- 数据文件统一保存在 `data/`，Jupyter Notebook 统一保存在 `notebook/`，Python 脚本统一保存在 `scripts/`，图片统一保存在 `pic/`。
- `data/` 与 `*.html` 被 `.gitignore` 忽略；`data/` 内容需用 `scripts/fetch_binance_vision.py`（七币种 1h，走 data.binance.vision 官方镜像，api.binance.com 本地被地域封锁）或 `scripts/fetch_ohlcv.py` / `append_ohlcv.py`（ccxt）重新拉取，不要假设其存在。`models/` 下的 `.zip` 已纳入 git。
- OHLCV 命名为 `{coin}_usdt_{timeframe}.csv`（小写币种，如 `eth_usdt_1d.csv`、`btc_usdt_1h.csv`），列为 `timestamp,open,high,low,close,volume`；用 duckdb 查 `data/` 下的 CSV 做校验。
- 在 notebook 中直接展示图片，不要额外保存到 `pic/`。

## 运行环境

- Python >= 3.13，使用 uv 管理，所有脚本从仓库根目录以 `uv run python scripts/<name>.py` 运行。
- 脚本内部用 `Path(__file__).resolve().parent.parent` 定位仓库根，不依赖调用时的 cwd。

## 脚本约定与陷阱

- 复用模块靠 `sys.path.insert(0, str(PROJECT_ROOT))` 后 `import scripts.dqn_gate_env as g`（多数训练/回测脚本）。
- 例外：`portfolio_optimize.py` 插入的是 `PROJECT_ROOT/"scripts"` 并直接 `import dynamic_grid_backtest as dgb`，且在导入它之前先 `import torch` / `import scipy`；这是为了先加载真实 torch，避免 `dynamic_grid_backtest` 注入的 stub 遮蔽 scipy 所需符号。改动导入顺序会静默破坏 scipy 优化。
- `dynamic_grid_backtest.py` 与 `vnpy_hourly_grid_backtest.py` 在 `import vnpy.alpha` 前向 `sys.modules` 注入 `alphalens/talib/lightgbm` 空模块；前者还注入 `torch` 及其子模块。这是无头运行兼容代码，不要删除。
- 回测强制无前视：技术指标只用 `rolling` + `ffill`（禁止 `bfill`），且 START 前预热天数必须 >= `max(ma_window, gate_window, atr_window)`，不足会报错退出。修改指标计算时须保持此约束。
- `search_grid_params.sh` 通过 xargs 并行调用 `dynamic_grid_backtest.py --print-json`（单行 JSON 契约），参数可用环境变量覆盖（`DATA`、`MA_WINDOWS`、`METRIC`、`PARALLEL` 等）；聚合排序用系统 `python3`，仅标准库。
- `window_mi_research.py` / `window_mi_research_multi.py` 只是 notebook 生成器：先运行脚本写 `notebook/window_research.ipynb`，再 `jupyter nbconvert --to notebook --execute --inplace notebook/window_research.ipynb` 才会真正执行出图。

## 策略与模型约定

- DQN 门控规范配置：A1 账户状态（`position_ratio`/`cash_ratio`）、1h 决策、R0 奖励、100k 步、seed 42；统一训练/回测入口为 `multi_coin_dqn.py`（单次 train/test 划分），消融与决策频率稳健性用 4 折 walk-forward（`account_state_ablation.py` / `reward_ablation.py` / `walk_forward_decision_interval.py`）。特征由 `dqn_gate_env.get_feature_stats` 在训练区间统计。
- 七币种模型命名 `dqn_gate_{coin}_a1_1h.zip`（币种小写），保存在 `models/`。
- 周期常量有两套，改动需同步多个脚本，勿混用：
  - 小时频/七币种（`multi_coin_dqn.py`、`portfolio_optimize_7coins.py`）：训练 `2022-01-01 ~ 2025-12-31 23:00:00`，测试/验证 `2026-01-01 ~ 2026-09-30 23:00:00`；消融与决策频率稳健性（`account_state_ablation.py` / `reward_ablation.py` / `walk_forward_decision_interval.py`）用 4 折 walk-forward，折内测试期截至 `2025-12-31`。
  - 日频三资产 `portfolio_optimize.py`：训练 `2021-05-07 ~ 2025-12-31`，验证 `2026-01-01 ~ 2026-08-11`。
- `generate_report_pdf.py` 依赖 `data/portfolio_7coins_*` 与 `data/multi_coin_dqn_results.csv`，并调用系统 `google-chrome --headless` 打印 PDF；缺少数据或 Chrome 会失败。
