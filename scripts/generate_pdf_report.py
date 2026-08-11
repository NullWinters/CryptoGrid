"""使用 HTML + headless Chrome 生成「策略报告.pdf」（简体中文，紧凑排版）。

报告内容：动态网格（动态上下界 + 动态层数 + 均线门控）交易策略的完整报告，
遵循模板六节：数据描述 / 策略说明 / 回测结果 / 关键指标 / 风险分析 / 小结。

方案说明：
- 用 HTML + 内联 CSS 排版（@page 打印样式，Noto Sans CJK SC 中文字体）
- 图片以 base64 内联，HTML 自包含可离线
- 调用系统 google-chrome --headless=new --print-to-pdf 渲染 PDF

用法：
    uv run python scripts/generate_pdf_report.py

输出：仓库根目录 策略报告.pdf
"""

from __future__ import annotations

import base64
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIC = PROJECT_ROOT / "pic"
OUT = PROJECT_ROOT / "策略报告.pdf"

IMG_NAV_TR = PIC / "train_grid_nav.png"
IMG_BAND_TR = PIC / "train_grid_band.png"
IMG_DD_TR = PIC / "train_grid_drawdown.png"
IMG_NAV_VAL = PIC / "val_grid_nav.png"
IMG_BAND_VAL = PIC / "val_grid_band.png"
IMG_DD_VAL = PIC / "val_grid_drawdown.png"


def b64(path: Path) -> str:
    """读取图片并返回 data URI（base64 内联）。"""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


CSS = """
@page { size: A4; margin: 1.5cm 1.6cm 1.6cm 1.6cm; }
@page :first { margin-top: 2.0cm; }
* { box-sizing: border-box; }
body {
  font-family: "Noto Sans CJK SC", "Noto Serif CJK SC", sans-serif;
  font-size: 9.5pt; line-height: 1.55; color: #1a1a1a; margin: 0;
}
h1.title { font-size: 19pt; text-align: center; margin: 0 0 2pt 0; }
p.subtitle { font-size: 8.5pt; text-align: center; color: #777; margin: 0 0 14pt 0; }
h2 { font-size: 12.5pt; margin: 12pt 0 4pt 0; border-left: 4px solid #1f77b4;
     padding-left: 7px; page-break-after: avoid; }
h3 { font-size: 10.5pt; margin: 8pt 0 3pt 0; color: #333; page-break-after: avoid; }
p { margin: 0 0 5pt 0; text-align: justify; text-indent: 2em; }
img { max-width: 100%; display: block; }
img.single { width: 100%; margin: 0 auto; }
.caption { font-size: 8pt; text-align: center; color: #888; margin: 1pt 0 8pt 0; }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 4pt 0 2pt 0; }
th, td { border: 0.5px solid #b0b0b0; padding: 3pt 6pt; text-align: center; }
th { background: #1f77b4; color: #fff; font-weight: normal; }
tbody tr:nth-child(even) { background: #f2f6fa; }
.tblcap { font-size: 8pt; text-align: center; color: #888; margin: 1pt 0 10pt 0; }
.risk li { margin-bottom: 3pt; }
.section { page-break-before: auto; }
"""


def build_html() -> str:
    def fig(src: str, cap: str) -> str:
        return f'<img class="single" src="{src}"/><div class="caption">{cap}</div>'

    rows = [
        ("总收益", "46.29%", "-2.07%", "51.06%", "-37.95%"),
        ("年化收益", "9.73%", "-3.39%", "—", "—"),
        ("年化波动", "10.10%", "3.06%", "—", "—"),
        ("最大回撤", "-14.58%", "-3.78%", "-79.28%", "-53.33%"),
        ("Sharpe", "0.84", "-1.10", "—", "—"),
        ("胜率（配对）", "66.31%", "22.22%", "—", "—"),
        ("换手率", "43.17", "0.87", "—", "—"),
        ("成交笔数", "1509", "39", "—", "—"),
        ("盈利/亏损天数", "916/799", "101/104", "—", "—"),
        ("重锚定次数", "95", "7", "—", "—"),
    ]
    tbl = "".join(
        f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td>"
        f"<td>{r[3]}</td><td>{r[4]}</td></tr>"
        for r in rows
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><style>{CSS}</style></head><body>

<h1 class="title">策略报告</h1>
<p class="subtitle">ETH/USDT 动态网格（动态上下界 + 动态层数 + 均线门控）· 2026-08</p>

<h2>一、数据描述</h2>
<p>标的为 ETH/USDT 现货日线数据，通过 ccxt 从 Binance 拉取并缓存于 data/eth_usdt_1d.csv，
区间 2021-01-01 ~ 2026-08-11，共 2049 根 K 线。现货无除权事件，价格未复权。
回测分两段：训练集 2021-04-01 ~ 2025-12-31（1736 个交易日），验证集 2026-01-01 ~ 2026-08-11（223 个交易日）。
年化按 365 个交易日换算；交易成本为单边 0.1% 手续费，不含滑点；基准为简单持有（Buy &amp; Hold）。</p>

<h2>二、策略说明</h2>
<p>动态网格交易法：网格上下界随价格中枢与波动率漂移——中枢 mid=MA(10)，半宽 half=2×ATR(7)，
区间为 [mid−half, mid+half]；层数随波动自适应，单格目标收益率 3%（间距=mid×3%，层数上限 300）。
以 close 与 MA(90) 比较做门控：跌破 MA(90) 判定下跌行情，撤单观望（保留底仓）；行情回暖时以当前价重锚定重建网格。
重锚定择时为：初始建仓、价格出界、偏离中枢超 25%（7 天冷却期）、门控恢复。
仓位为初始 50% 底仓 + 每格 1 ETH，卖出不超过持仓、买入不超过可用现金。
参数经 1500 组训练集网格搜索按夏普最优选取。网格法的收益本质来自震荡行情的低买高卖，
在明显单边行情中跑不过持有，但可有效减小回撤。</p>

<h2>三、回测结果</h2>
<h3>训练集（2021-04-01 ~ 2025-12-31，震荡偏多行情）</h3>
<p>策略总收益 46.29%，年化 9.73%，最大回撤 -14.58%（持续 113 天），夏普 0.84，
配对胜率 66.31%，换手率 43.17，成交 1509 笔（买 757 / 卖 752），盈利/亏损天数 916/799，重锚定 95 次。
对比基准：持有总收益 51.06%、最大回撤 -79.28%。策略收益略逊于持有，但回撤从 -79.28% 收窄至 -14.58%，
波动率大幅降低。下图依次为净值曲线、动态网格区间与回撤。</p>
{fig(b64(IMG_NAV_TR), "图 1 训练集净值曲线（策略 vs Buy&Hold）")}
{fig(b64(IMG_BAND_TR), "图 2 训练集动态网格区间（价格与上下界）")}
{fig(b64(IMG_DD_TR), "图 3 训练集净值与回撤（最大回撤 -14.58%）")}

<h3>验证集（2026-01-01 ~ 2026-08-11，单边下跌行情）</h3>
<p>策略总收益 -2.07%，年化 -3.39%，最大回撤 -3.78%（持续 157 天），夏普 -1.10，
配对胜率 22.22%，换手率 0.87，成交仅 39 笔（买 21 / 卖 18），盈利/亏损天数 101/104，重锚定 7 次。
对比基准：持有总收益 -37.95%、最大回撤 -53.33%。验证集处于单边下跌，策略未能盈利但大幅跑赢持有——
门控（close&lt;MA90）长期撤单观望，仅以底仓承受下跌，回撤控制在 -3.78%。</p>
{fig(b64(IMG_NAV_VAL), "图 4 验证集净值曲线（策略 vs Buy&Hold）")}
{fig(b64(IMG_BAND_VAL), "图 5 验证集动态网格区间（价格与上下界）")}
{fig(b64(IMG_DD_VAL), "图 6 验证集净值与回撤（最大回撤 -3.78%）")}

<h2>四、关键指标</h2>
<table>
<thead><tr><th>指标</th><th>训练集</th><th>验证集</th><th>训练集 Buy&amp;Hold</th><th>验证集 Buy&amp;Hold</th></tr></thead>
<tbody>{tbl}</tbody>
</table>
<div class="tblcap">表 1 训练集 / 验证集关键指标与持有基准对照</div>

<h2>五、风险分析</h2>
<ul class="risk">
<li><b>最大回撤时段：</b>训练集最大回撤 -14.58% 出现在 2024 年末段（持续 113 天）；验证集最大回撤 -3.78%
（持续 157 天）对应 2026 年单边下跌，门控长期撤单使净值缓慢回撤。</li>
<li><b>门控滞后：</b>MA(90) 存在滞后，验证集 2026 年 1 月初 GRID 段（约 19 天）已现 -5.9% 下行，门控后才停机。</li>
<li><b>单边行情短板：</b>网格法是系统性短板——下跌时持续接盘（由门控减轻但不能消除），上涨时被价格甩开存在踏空成本。</li>
<li><b>崩溃样本：</b>极端参数下曾出现回撤超 80% 的崩溃样本，已从参数搜索中剔除。</li>
<li><b>统计意义：</b>验证集成交仅 39 笔，胜率与换手率统计意义较弱。</li>
</ul>

<h2>六、小结</h2>
<p>网格交易法的收益（alpha）来源于震荡行情：在价格中枢附近反复低买高卖积累差价。
训练集（震荡偏多）实现 +46.29% 收益，波动率 10.10%，回撤仅 -14.58%；验证集（单边下跌）证实其在
明显单边行情中跑不过持有——策略仍亏损 -2.07%，但将回撤控制在 -3.78%（持有为 -53.33%），
波动率 3.06%，防御价值显著。策略适用于以低回撤为核心诉求、认可「震荡赚差价、单边不跟涨」特征的投资者。
需注意：参数由训练集 1500 组网格搜索按夏普择优，存在过拟合风险，验证集结果已转亏；
且结果仅反映 2021-01 ~ 2026-08 这一历史区间。</p>

</body></html>
"""


def main() -> int:
    chrome = shutil.which("google-chrome") or shutil.which("google-chrome-stable") or shutil.which("chromium")
    if not chrome:
        print("未找到 google-chrome / chromium，请先安装", file=sys.stderr)
        return 1

    html = build_html()
    with tempfile.TemporaryDirectory(prefix="grid_report_") as td:
        html_path = Path(td) / "report.html"
        html_path.write_text(html, encoding="utf-8")
        cmd = [
            chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={OUT}", str(html_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    print(f"已生成: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
