"""生成《时频策略报告》HTML 并调用 Chrome 打印为 PDF（页脚仅居中页码）。"""
from __future__ import annotations

import base64, io, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Noto Sans CJK SC", "axes.unicode_minus": False,
                     "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
HTML_PATH = ROOT / "时频策略报告.html"
PDF_PATH = ROOT / "时频策略报告.pdf"
COINS = ["BTC", "ETH", "BNB", "PAXG", "SOL", "TRX", "XLM"]
GROUPS = {"BTC": "核心蓝筹组", "ETH": "核心蓝筹组", "BNB": "核心蓝筹组",
          "SOL": "核心蓝筹组", "TRX": "事件驱动组", "XLM": "事件驱动组", "PAXG": "避险标的"}

nav = pd.read_csv(DATA / "portfolio_7coins_nav.csv", parse_dates=["timestamp"]).set_index("timestamp").iloc[:, 0]
nav_by_coin = pd.read_csv(DATA / "portfolio_7coins_nav_by_coin.csv", parse_dates=["timestamp"]).set_index("timestamp")
weights = pd.read_csv(DATA / "portfolio_7coins_weights.csv", parse_dates=["date"])
trades = pd.read_csv(DATA / "portfolio_7coins_trades.csv", parse_dates=["date"])
segments = pd.read_csv(DATA / "portfolio_7coins_segments.csv")
eq_nav = pd.read_csv(DATA / "portfolio_7coins_equalweight_dqn_nav.csv", parse_dates=["timestamp"]).set_index("timestamp").iloc[:, 0]
bh_nav = pd.read_csv(DATA / "portfolio_7coins_equalweight_bh_nav.csv", parse_dates=["timestamp"]).set_index("timestamp").iloc[:, 0]
metrics = json.load(open(DATA / "portfolio_7coins_metrics.json"))
coin_df = pd.read_csv(DATA / "multi_coin_dqn_results.csv")

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def dd(x): return x / x.cummax() - 1

# 图1 净值
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(nav.index, nav.values, lw=1.8, color="#1f77b4", label="风险平价 DQN 组合")
ax.plot(eq_nav.index, eq_nav.values, lw=1.2, color="#d62728", label="等权 DQN 组合")
ax.plot(bh_nav.index, bh_nav.values, lw=1.1, color="#7f7f7f", ls="--", label="等权 Buy&Hold")
ax.axhline(100_000, color="#333", ls=":", lw=0.8)
ax.set_title("组合净值对比（2026-01 ~ 2026-09）"); ax.set_ylabel("净值 (USDT)")
ax.legend(loc="lower left", fontsize=8); fig.tight_layout()
img_nav = fig_to_base64(fig); plt.close(fig)

# 图2 回撤
fig, ax = plt.subplots(figsize=(9, 3.6))
for series, label, color in [(nav, "风险平价 DQN 组合", "#1f77b4"),
                              (eq_nav, "等权 DQN 组合", "#d62728"),
                              (bh_nav, "等权 Buy&Hold", "#7f7f7f")]:
    d = dd(series) * 100
    ax.plot(series.index, d, lw=1.2, color=color, label=f"{label} (max {d.min():.2f}%)")
ax.axhline(0, color="#333", ls=":", lw=0.8)
ax.set_title("回撤对比"); ax.set_ylabel("回撤 (%)")
ax.legend(loc="lower left", fontsize=8); fig.tight_layout()
img_dd = fig_to_base64(fig); plt.close(fig)

# 图3 权重
w = weights.set_index("date")
fig, ax = plt.subplots(figsize=(9, 3.6))
w[COINS].plot(kind="bar", stacked=True, colormap="tab10", ax=ax, rot=0)
ax.set_title("风险平价权重演变（季度调仓）"); ax.set_ylabel("权重"); ax.set_xlabel("调仓日期")
ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in w.index])
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7); fig.tight_layout()
img_weights = fig_to_base64(fig); plt.close(fig)

# 图4 分币种 NAV（双子图）
seg = segments.copy(); seg["label"] = seg["segment"].astype(str) + "段\n" + seg["end"].str[5:10]
nav_cols = [f"end_nav_{c}" for c in COINS]
melt = seg.melt(id_vars="label", value_vars=nav_cols, var_name="coin", value_name="end_nav")
melt["coin"] = melt["coin"].str.replace("end_nav_", "")
piv = melt.pivot(index="label", columns="coin", values="end_nav")[COINS]
fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
xpos = np.arange(len(piv))
for ax, cols, sub_title in [
    (axes[0], ["PAXG", "TRX"], "PAXG / TRX"),
    (axes[1], ["BTC", "ETH", "BNB", "SOL", "XLM"], "BTC / ETH / BNB / SOL / XLM"),
]:
    for c in cols:
        ax.plot(xpos, piv[c].values, marker="o", lw=1.2, label=c)
    ax.set_xticks(xpos)
    ax.set_xticklabels(piv.index)
    ax.set_ylabel("NAV (USDT)")
    ax.set_title(sub_title, fontsize=10)
    ax.legend(fontsize=7)
fig.suptitle("各季度段末分币种 NAV 构成", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
img_seg = fig_to_base64(fig); plt.close(fig)

# 图5 单币种 DQN vs MA
coin_df["coin"] = coin_df["coin"].astype(str)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
x = np.arange(len(coin_df)); wd = 0.35
axes[0].bar(x - wd/2, coin_df["rl_ret"]*100, wd, color="#1f77b4", label="DQN门控")
axes[0].bar(x + wd/2, coin_df["ma_gate_ret"]*100, wd, color="#d62728", label="MA门控")
axes[0].axhline(0, color="#333", lw=0.7); axes[0].set_xticks(x); axes[0].set_xticklabels(coin_df["coin"])
axes[0].set_title("各币种收益：DQN vs MA"); axes[0].set_ylabel("%"); axes[0].legend(fontsize=7)
axes[1].bar(x - wd/2, coin_df["rl_dd"]*100, wd, color="#1f77b4", label="DQN门控")
axes[1].bar(x + wd/2, coin_df["ma_gate_dd"]*100, wd, color="#d62728", label="MA门控")
axes[1].axhline(0, color="#333", lw=0.7); axes[1].set_xticks(x); axes[1].set_xticklabels(coin_df["coin"])
axes[1].set_title("各币种回撤：DQN vs MA（负值越大回撤越深）"); axes[1].set_ylabel("%"); axes[1].legend(fontsize=7)
fig.tight_layout(); img_coin = fig_to_base64(fig); plt.close(fig)

# 分析数据
m_rp, m_eq, m_bh = metrics["risk_parity"], metrics["equal_weight_dqn"], metrics["equal_weight_buyhold"]
d = dd(nav); peak = nav.cummax(); trough = d.idxmin(); peak_time = peak.loc[:trough].idxmax()
daily_ret = pd.DataFrame({"rp": nav.resample("1D").last().pct_change(),
                          "eq": eq_nav.resample("1D").last().pct_change(),
                          "bh": bh_nav.resample("1D").last().pct_change()}).dropna()
corr_eq = daily_ret["rp"].corr(daily_ret["eq"]); corr_bh = daily_ret["rp"].corr(daily_ret["bh"])
hourly_ret = nav.pct_change().dropna(); neg = hourly_ret < 0
max_neg_run = cur = 0; max_run_start = cur_start = None
for t, v in neg.items():
    if v:
        if cur == 0: cur_start = t
        cur += 1
        if cur > max_neg_run: max_neg_run, max_run_start = cur, cur_start
    else:
        cur = 0
def group_w(row):
    s = {g: 0.0 for g in ["核心蓝筹组", "事件驱动组", "避险标的"]}
    for c in COINS: s[GROUPS[c]] += row[c]
    return s
w0 = group_w(weights.iloc[0]); w1 = group_w(weights.iloc[-1])
n_win_ret = int((coin_df["rl_ret"] > coin_df["ma_gate_ret"]).sum())
n_win_dd = int((coin_df["rl_dd"] > coin_df["ma_gate_dd"]).sum())

def pct(v): return f"{v*100:.2f}%"

sections_html = f"""
<div class="keep"><h3>3.1 净值曲线</h3><img src="data:image/png;base64,{img_nav}"></div>
<div class="keep"><h3>3.2 回撤曲线</h3><img src="data:image/png;base64,{img_dd}"></div>
<div class="keep"><h3>3.3 各季度段末分币种 NAV 构成</h3><img src="data:image/png;base64,{img_seg}"></div>
<div class="keep"><h3>3.4 权重演变</h3><img src="data:image/png;base64,{img_weights}"></div>
<div class="keep"><h3>3.5 单币种 DQN 门控 vs MA 门控</h3><img src="data:image/png;base64,{img_coin}"></div>
"""

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<style>
@page {{ size: A4; margin: 2cm 1.8cm; }}
body {{ font-family: 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', sans-serif; color:#1a1a1a; line-height:1.75; font-size:12px; }}
h1 {{ font-size:22px; text-align:center; margin:0 0 6px 0; }}
h2 {{ font-size:16px; border-bottom:2px solid #1f77b4; padding-bottom:4px; margin:24px 0 10px 0; color:#1f77b4; page-break-after:avoid; break-after:avoid; }}
h3 {{ font-size:13px; margin:14px 0 6px 0; page-break-after:avoid; break-after:avoid; }}
p {{ margin:6px 0; text-indent:2em; }}
.meta, .small {{ text-indent:0; }}
table {{ border-collapse:collapse; width:100%; margin:10px 0; font-size:11px; page-break-inside:avoid; break-inside:avoid; }}
th,td {{ border:1px solid #ccc; padding:5px 8px; text-align:center; }}
th {{ background:#f0f5fa; }}
img {{ width:100%; margin:8px 0; page-break-inside:avoid; break-inside:avoid; }}
.keep {{ page-break-inside:avoid; break-inside:avoid; margin:0; }}
.meta {{ text-align:center; color:#666; font-size:11px; margin-bottom:16px; }}
.small {{ font-size:11px; color:#444; }}
</style></head><body>

<h1>七币种 DQN 门控策略风险平价组合</h1>
<div class="meta">策略报告 · 2026-01-01 ~ 2026-09-30 · 初始资金 100,000 USDT</div>

<h2>一、数据描述</h2>
<p>标的池为 BTC、ETH、BNB、SOL、TRX、XLM、PAXG 七币种，USDT 计价，Binance 现货小时级 OHLCV，样本区间 2021-01-01 ~ 2026-10-11。策略模型训练区间 2022-01-01 ~ 2025-12-31；组合回测区间 2026-01-01 ~ 2026-09-30。价格不做复权，现货无资金费率。</p>
<table>
<tr><th>分组</th><th>币种</th><th>角色</th></tr>
<tr><td>核心蓝筹组</td><td>BTC、ETH、BNB、SOL</td><td>承担加密市场长期风险溢价，提供趋势弹性</td></tr>
<tr><td>事件驱动组</td><td>TRX、XLM</td><td>高波动、高弹性，受事件与生态催化驱动</td></tr>
<tr><td>避险标的</td><td>PAXG</td><td>黄金代币，低波动，降低组合尾部风险</td></tr>
</table>

<h2>二、策略说明</h2>
<p>七个币种各自运行一个“DQN 门控动态带网格策略”：网格交易负责在震荡行情中低买高卖，DQN 门控负责在趋势或下跌行情中撤单观望、控制回撤。组合层面，每季度末使用风险平价（Equal Risk Contribution）方法，把总资金重新分配到七个策略上：低波动、低相关资产获得更高权重，单资产权重上限 40%。调仓成本按 0.1% 单边计。</p>

<h2>三、回测结果</h2>
{sections_html}

<h2>四、关键指标</h2>
<table>
<tr><th>策略</th><th>总收益</th><th>年化收益</th><th>波动率</th><th>最大回撤</th><th>Sharpe</th><th>Calmar</th><th>期末NAV</th></tr>
<tr><td>风险平价 DQN 组合</td><td>{pct(m_rp['total_return'])}</td><td>{pct(m_rp['annual_return'])}</td><td>{pct(m_rp['return_std'])}</td><td>{pct(m_rp['max_dd'])}</td><td>{m_rp['sharpe']:.3f}</td><td>{m_rp['calmar']:.3f}</td><td>{m_rp['end_nav']:,.0f}</td></tr>
<tr><td>等权 DQN 组合</td><td>{pct(m_eq['total_return'])}</td><td>{pct(m_eq['annual_return'])}</td><td>{pct(m_eq['return_std'])}</td><td>{pct(m_eq['max_dd'])}</td><td>{m_eq['sharpe']:.3f}</td><td>{m_eq['calmar']:.3f}</td><td>{m_eq['end_nav']:,.0f}</td></tr>
<tr><td>等权 Buy&amp;Hold</td><td>{pct(m_bh['total_return'])}</td><td>{pct(m_bh['annual_return'])}</td><td>{pct(m_bh['return_std'])}</td><td>{pct(m_bh['max_dd'])}</td><td>{m_bh['sharpe']:.3f}</td><td>{m_bh['calmar']:.3f}</td><td>{m_bh['end_nav']:,.0f}</td></tr>
</table>
<p class="small">* 年化按 8760 小时自然日口径；无风险利率取 0。</p>

<h4>4.1 权重与分组</h4>
<p>期初权重：PAXG {weights.iloc[0]['PAXG']*100:.2f}%、TRX {weights.iloc[0]['TRX']*100:.2f}%、BTC {weights.iloc[0]['BTC']*100:.2f}%、BNB {weights.iloc[0]['BNB']*100:.2f}%、ETH {weights.iloc[0]['ETH']*100:.2f}%、XLM {weights.iloc[0]['XLM']*100:.2f}%、SOL {weights.iloc[0]['SOL']*100:.2f}%。</p>
<p>期末权重：TRX {weights.iloc[-1]['TRX']*100:.2f}%、PAXG {weights.iloc[-1]['PAXG']*100:.2f}%、BNB {weights.iloc[-1]['BNB']*100:.2f}%、BTC {weights.iloc[-1]['BTC']*100:.2f}%、XLM {weights.iloc[-1]['XLM']*100:.2f}%、ETH {weights.iloc[-1]['ETH']*100:.2f}%、SOL {weights.iloc[-1]['SOL']*100:.2f}%。</p>
<p>分组权重：期初 核心蓝筹 {w0['核心蓝筹组']*100:.2f}% / 事件驱动 {w0['事件驱动组']*100:.2f}% / 避险 {w0['避险标的']*100:.2f}%；期末 核心蓝筹 {w1['核心蓝筹组']*100:.2f}% / 事件驱动 {w1['事件驱动组']*100:.2f}% / 避险 {w1['避险标的']*100:.2f}%。</p>

<h4>4.2 成本与换手</h4>
<p>两次季度调仓，累计调仓费 {trades['fee'].sum():.2f} USDT，占初始资金 {trades['fee'].sum()/100000*100:.3f}%。单币种层面，DQN 门控相对 MA 门控：收益胜 {n_win_ret}/7，回撤更小 {n_win_dd}/7。</p>

<h2>五、风险分析</h2>
<p><b>最大回撤发生时段</b>：组合最大回撤 {m_rp['max_dd']*100:.2f}%，回撤起点约 {peak_time.strftime('%Y-%m-%d %H:%M')}，低点约 {trough.strftime('%Y-%m-%d %H:%M')}。</p>
<p><b>连续亏损尾部</b>：小时级最长连续亏损约 {max_neg_run} 小时，起点约 {max_run_start.strftime('%Y-%m-%d %H:%M') if max_run_start is not None else '—'}。</p>
<p><b>与基准的相关性</b>：风险平价组合日收益与等权 DQN 组合相关系数 {corr_eq:.3f}，与等权 Buy&amp;Hold 相关系数 {corr_bh:.3f}。说明组合波动来源与等权 DQN 接近，但显著低于 B&amp;H。</p>
<p><b>集中度风险</b>：PAXG + TRX 权重合计最高约 {weights[['PAXG', 'TRX']].sum(axis=1).max()*100:.2f}%，若两者同时趋势性下跌，组合防守优势将明显削弱。</p>
<p><b>模型/参数风险</b>：DQN 门控模型为 seed 42 单种子；风险平价参数（λ=0.98、单资产上限 40%）未经训练集回放校准。</p>

<h2>六、小结</h2>
<p>本策略在 2026 年 1 至 9 月的行情里实现了 {pct(m_rp['total_return'])} 的总收益和 {pct(m_rp['max_dd'])} 的最大回撤，优于等权 DQN 组合（{pct(m_eq['total_return'])}，回撤 {pct(m_eq['max_dd'])}）和等权 Buy&amp;Hold（{pct(m_bh['total_return'])}，回撤 {pct(m_bh['max_dd'])}）。上半年加密市场整体下跌，组合依靠单币种 DQN 门控减少下跌段亏损、季度风险平价把资金压向低波动资产，把回撤控制在相对较浅的水平；三季度市场转涨，防守姿态让出部分弹性，单币种层面 DQN 门控相对 MA 门控收益胜 {n_win_ret}/7、回撤更小 {n_win_dd}/7，组合仍保持正收益且回撤未再加深。收益来源主要来自两层设计：<b>DQN 门控在单币种上减少下跌段亏损，风险平价在组合层把资金压向低波动资产</b>。适合希望参与加密市场、但对回撤敏感、愿意放弃部分上涨弹性的稳健型投资者；不建议作为趋势牛市中追求高弹性的单一全仓策略。</p>

</body></html>"""

HTML_PATH.write_text(html, encoding="utf-8")

# Chrome 转 PDF（无页眉页脚）
tmp = ROOT / "时频策略报告_tmp.pdf"
r = subprocess.run(["google-chrome", "--headless", "--disable-gpu", "--no-sandbox",
                    f"--print-to-pdf={tmp}", "--no-pdf-header-footer", str(HTML_PATH)],
                   capture_output=True, text=True)
if r.returncode != 0:
    raise RuntimeError(r.stderr[:1000])

# 叠加居中页码
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
reader = PdfReader(str(tmp)); writer = PdfWriter(); w_pt, h_pt = A4
for i, page in enumerate(reader.pages):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont("Helvetica", 10)
    c.drawCentredString(w_pt/2, 20, str(i+1))
    c.save(); buf.seek(0)
    overlay = PdfReader(buf).pages[0]
    page.merge_page(overlay)
    writer.add_page(page)
with open(PDF_PATH, "wb") as f:
    writer.write(f)
tmp.unlink()
print("HTML:", HTML_PATH)
print("PDF:", PDF_PATH, "pages:", len(reader.pages))
