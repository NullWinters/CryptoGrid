# 项目规则

## 目录约定

- 数据文件统一保存在 `data/`
- Jupyter Notebook（.ipynb）统一保存在 `notebook/`
- Python 脚本统一保存在 `scripts/`
- 图片文件统一保存在 `pic/`

## 数据检查

- 使用 duckdb 查阅 `data/` 下的表数据（如 CSV）进行校验。

## Notebook 图片

- 在 notebook 中直接展示图片（如 `IPython.display.display`），不要额外保存到 `pic/`。
