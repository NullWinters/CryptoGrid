#!/usr/bin/env bash
# 动态网格参数搜索：网格扫描 dynamic_grid_backtest.py，按 Sharpe 取最优（xargs 并行）。
#
# 默认网格（可用环境变量覆盖）：
#   MA_WINDOWS="10 20 30 60 90"  GATE_WINDOWS="10 20 30 60 90"
#   K_WIDTHS="2 3 4 5"  TARGET_RETS="0.005 0.01 0.015 0.02 0.03"
#   ATR_WINDOWS="7 14 21"  MAX_STEPS=300
#   DATA=...  START=auto  END=...  METRIC=sharpe  PARALLEL=8
#
# 说明：START=auto 时每组参数从各自最早可回测日（数据起始后 max(ma,gate,atr) 个交易日）
# 回测到 END，因此各组回测区间长度不同；按 Sharpe 排序取优，不比较收益绝对值。
#
# 用法：
#   bash scripts/search_grid_params.sh
#   METRIC=return_dd bash scripts/search_grid_params.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="uv run python $SCRIPT_DIR/dynamic_grid_backtest.py"

MA_WINDOWS="${MA_WINDOWS:-10 20 30 60 90}"
GATE_WINDOWS="${GATE_WINDOWS:-10 20 30 60 90}"
K_WIDTHS="${K_WIDTHS:-2 3 4 5}"
TARGET_RETS="${TARGET_RETS:-0.005 0.01 0.015 0.02 0.03}"
ATR_WINDOWS="${ATR_WINDOWS:-7 14 21}"
MAX_STEPS="${MAX_STEPS:-300}"
DATA="${DATA:-$SCRIPT_DIR/../data/eth_usdt_1d.csv}"
START="${START:-auto}"
END="${END:-2025-12-31}"
METRIC="${METRIC:-sharpe}"
PARALLEL="${PARALLEL:-8}"
LAB_ROOT="${LAB_ROOT:-$(mktemp -d /tmp/grid_search_XXXXXX)}"

# 输出行格式：参数 指标JSON（TSV）
emit_job() {
    local ma="$1" gate="$2" kw="$3" tr="$4" atr="$5"
    local lab="$LAB_ROOT/lab_${ma}_${gate}_${kw}_${tr}_${atr}"
    mkdir -p "$lab"
    local out
    out="$($PY \
        --data "$DATA" --start "$START" --end "$END" \
        --ma-window "$ma" --gate-window "$gate" --k-width "$kw" \
        --target-ret "$tr" --atr-window "$atr" --max-steps "$MAX_STEPS" \
        --lab "$lab" --print-json 2>/dev/null)" || {
        echo "FAIL ma=$ma gate=$gate K=$kw ret=$tr atr=$atr" >> "$LAB_ROOT/fail.log"
        return 0
    }
    printf 'ma=%s\tgate=%s\tK=%s\tret=%s\tatr=%s\t%s\n' "$ma" "$gate" "$kw" "$tr" "$atr" "$out"
}

export PY DATA START END MAX_STEPS LAB_ROOT
export -f emit_job

echo "搜索网格: MA∈{$MA_WINDOWS} GATE∈{$GATE_WINDOWS} K∈{$K_WIDTHS} RET∈{$TARGET_RETS} ATR∈{$ATR_WINDOWS}"
echo "区间: $START ~ $END (START=auto 各组取各自最早可回测日) | 目标指标: $METRIC | 并行度: $PARALLEL"
echo "开始搜索 $(date '+%H:%M:%S') ..."

# 生成全部组合，xargs 并行执行（个别任务失败不中断整体）
rm -f "$LAB_ROOT/results.tsv" "$LAB_ROOT/fail.log"
trap 'rm -rf "$LAB_ROOT"' EXIT

for ma in $MA_WINDOWS; do
  for gate in $GATE_WINDOWS; do
    for kw in $K_WIDTHS; do
      for tr in $TARGET_RETS; do
        for atr in $ATR_WINDOWS; do
          printf '%s %s %s %s %s\n' "$ma" "$gate" "$kw" "$tr" "$atr"
        done
      done
    done
  done
done \
| xargs -P "$PARALLEL" -n 5 bash -c 'emit_job "$@"' _ \
>> "$LAB_ROOT/results.tsv"

echo "搜索完成 $(date '+%H:%M:%S')，共 $(wc -l < "$LAB_ROOT/results.tsv") 组有效结果，失败 $(wc -l < "$LAB_ROOT/fail.log" 2>/dev/null || echo 0) 组"

# 聚合排序（系统 python3 仅用标准库 json）
python3 - "$METRIC" "$LAB_ROOT/results.tsv" "$DATA" "$START" "$END" "$MAX_STEPS" <<'EOF'
import json, sys

metric, path, data, start, end, max_steps = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
rows = []
for line in open(path):
    line = line.strip()
    if not line or "\t" not in line:
        continue
    params, js = line.rsplit("\t", 1)
    try:
        d = json.loads(js)
    except Exception:
        continue
    # 合理性过滤：剔除崩溃/失真样本（回撤过大、资金归零、收益异常高）
    if d.get("max_dd", 0) < -0.80:
        continue
    if d.get("end_balance", 0) <= 0:
        continue
    if d.get("total_return", 0) > 10 or d.get("total_return", 0) < -0.99:
        continue
    rows.append((params, d))

if not rows:
    print("无有效结果")
    sys.exit(1)

def keyf(r):
    d = r[1]
    if metric == "total_return":
        return d["total_return"]
    if metric == "return_dd":
        return d["total_return"] / max(abs(d["max_dd"]), 1e-9)
    return d["sharpe"]

rows.sort(key=keyf, reverse=True)
print(f"\n========== Top 8（目标指标: {metric}）==========")
print(f"{'参数':<36} {'起始日':<11} {'总收益':>8} {'年化':>7} {'回撤':>8} {'Sharpe':>7} {'胜率':>6} {'笔数':>6}")
for params, d in rows[:8]:
    print(f"{params:<36} {d['start']:<11} {d['total_return']*100:>7.2f}% {d['annual_return']*100:>6.2f}% "
          f"{d['max_dd']*100:>7.2f}% {d['sharpe']:>7.2f} "
          f"{d['win_rate']*100 if d['win_rate'] else float('nan'):>6.1f}% {d['trades']:>6}")

best_params, best = rows[0]
fields = best_params.split("\t")
cmd = (f"uv run python scripts/dynamic_grid_backtest.py "
       f"--ma-window {fields[0].split('=')[1]} "
       f"--gate-window {fields[1].split('=')[1]} "
       f"--k-width {fields[2].split('=')[1]} "
       f"--target-ret {fields[3].split('=')[1]} "
       f"--atr-window {fields[4].split('=')[1]} "
       f"--max-steps {max_steps} "
       f"--data {data} --start {best['start']} --end {end}")
print(f"\n最佳参数: {best_params}  (起始日 {best['start']})")
print(f"可复现命令:\n  {cmd}")
EOF
