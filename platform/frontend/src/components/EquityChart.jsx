import { useEffect, useRef } from "react";
import { LineStyle, createChart } from "lightweight-charts";

const toTime = (t) => Math.floor(Number(t) / 1000);

const withAlpha = (hex, alpha) => (hex?.startsWith("#") && hex.length === 7 ? `${hex}${alpha}` : hex);

const toValue = (v, unit, denom) => (unit === "pct" ? (Number(v) / denom) * 100 : Number(v));

function maxDrawdownBand(points) {
  let peak = null;
  let peakTime = null;
  let best = 0;
  let band = null;
  for (const point of points) {
    const equity = Number(point.equity);
    if (!Number.isFinite(equity)) continue;
    if (peak === null || equity > peak) {
      peak = equity;
      peakTime = point.t;
    }
    const drawdown = peak > 0 ? (peak - equity) / peak : 0;
    if (drawdown > best) {
      best = drawdown;
      band = { peakTime, troughTime: point.t };
    }
  }
  return best > 0 ? band : null;
}

export default function EquityChart({ points, totalInvested, maxDrawdown, offlineRanges, unit, colors }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const strategyRef = useRef(null);
  const benchmarkRef = useRef(null);
  const bandRef = useRef(null);

  const list = points ?? [];
  const ranges = offlineRanges ?? [];
  const empty = list.length === 0;

  useEffect(() => {
    // autoSize tracks the container's size, so no manual resize handling is needed
    const chart = createChart(containerRef.current, {
      autoSize: true,
      layout: { background: { color: colors.bg }, textColor: colors.fg },
      grid: { vertLines: { color: colors.border }, horzLines: { color: colors.border } },
      rightPriceScale: { borderColor: colors.border },
      timeScale: { borderColor: colors.border },
    });
    chartRef.current = chart;

    // 回撤带先创建，保证绘制在净值曲线下方
    const band = chart.addAreaSeries({
      priceScaleId: "",
      lineWidth: 0,
      lineColor: "transparent",
      topColor: withAlpha(colors.danger, "1A"),
      bottomColor: withAlpha(colors.danger, "1A"),
      lastValueVisible: false,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
      autoscaleInfoProvider: () => ({ priceRange: { minValue: 0, maxValue: 1 } }),
    });
    band.priceScale().applyOptions({ scaleMargins: { top: 0, bottom: 0 } });
    bandRef.current = band;

    strategyRef.current = chart.addAreaSeries({
      lineWidth: 2,
      priceLineVisible: false,
      topColor: withAlpha(colors.strategy, "2E"),
      bottomColor: withAlpha(colors.strategy, "00"),
    });
    benchmarkRef.current = chart.addLineSeries({
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    return () => {
      strategyRef.current = null;
      benchmarkRef.current = null;
      bandRef.current = null;
      chartRef.current = null;
      chart.remove();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    chartRef.current?.applyOptions({
      layout: { background: { color: colors.bg }, textColor: colors.fg },
      grid: { vertLines: { color: colors.border }, horzLines: { color: colors.border } },
      rightPriceScale: { borderColor: colors.border },
      timeScale: { borderColor: colors.border },
    });
    strategyRef.current?.applyOptions({
      lineColor: colors.strategy,
      topColor: withAlpha(colors.strategy, "2E"),
      bottomColor: withAlpha(colors.strategy, "00"),
    });
    benchmarkRef.current?.applyOptions({ color: colors.benchmark });
    bandRef.current?.applyOptions({
      topColor: withAlpha(colors.danger, "1A"),
      bottomColor: withAlpha(colors.danger, "1A"),
    });
  }, [colors.bg, colors.fg, colors.border, colors.strategy, colors.benchmark, colors.danger]);

  useEffect(() => {
    const denom = Number(totalInvested) || 1;
    const valid = list.filter((p) => Number.isFinite(Number(p.equity)));
    strategyRef.current?.setData(
      valid.map((p) => ({ time: toTime(p.t), value: toValue(p.equity, unit, denom) })),
    );
    benchmarkRef.current?.setData(
      valid
        .filter((p) => p.benchmark !== null && p.benchmark !== undefined && Number(p.benchmark) > 0)
        .map((p) => ({ time: toTime(p.t), value: toValue(p.benchmark, unit, denom) })),
    );

    const band = maxDrawdownBand(valid);
    bandRef.current?.setData(
      band && band.peakTime !== band.troughTime
        ? [
            { time: toTime(band.peakTime), value: 1 },
            { time: toTime(band.troughTime), value: 1 },
          ]
        : [],
    );
  }, [list, unit, totalInvested]);

  return (
    <div>
      {!empty && (
        <div className="drawdown-note">最大回撤 {(Number(maxDrawdown) * 100).toFixed(2)}%</div>
      )}
      <div ref={containerRef} className="chart-canvas equity" />
      {!empty && ranges.length > 0 && (
        <div className="offline-note">
          {ranges.map(([start, end]) => (
            <div key={`${start}-${end}`}>
              离线补齐：{new Date(Number(start)).toLocaleString()} - {new Date(Number(end)).toLocaleString()}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
