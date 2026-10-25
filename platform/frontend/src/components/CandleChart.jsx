import { useEffect, useRef } from "react";
import { createChart } from "lightweight-charts";

const toTime = (t) => Math.floor(Number(t) / 1000);

const toBar = (d) => ({
  time: toTime(d.t),
  open: Number(d.o),
  high: Number(d.h),
  low: Number(d.l),
  close: Number(d.c),
});

const withAlpha = (hex, alpha) => (hex?.startsWith("#") && hex.length === 7 ? `${hex}${alpha}` : hex);

const toVolume = (d, up, down) => ({
  time: toTime(d.t),
  value: Number(d.v ?? 0),
  color: withAlpha(Number(d.c) >= Number(d.o) ? up : down, "66"),
});

export default function CandleChart({ candles, liveRef, up, down, bg, fg, border }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef(null);
  const volumeRef = useRef(null);

  useEffect(() => {
    // autoSize tracks the container's size, so no manual resize handling is needed
    const chart = createChart(containerRef.current, {
      autoSize: true,
      layout: { background: { color: bg }, textColor: fg },
      grid: { vertLines: { color: border }, horzLines: { color: border } },
      rightPriceScale: { borderColor: border },
      timeScale: { borderColor: border },
    });
    chartRef.current = chart;
    seriesRef.current = chart.addCandlestickSeries({
      upColor: up,
      downColor: down,
      borderUpColor: up,
      borderDownColor: down,
      wickUpColor: up,
      wickDownColor: down,
    });
    // 为底部成交量留出空间，避免柱体与 K 线重叠
    seriesRef.current.priceScale().applyOptions({ scaleMargins: { top: 0.06, bottom: 0.22 } });
    const volume = chart.addHistogramSeries({
      priceScaleId: "",
      priceFormat: { type: "volume" },
      lastValueVisible: false,
      priceLineVisible: false,
    });
    volume.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
    volumeRef.current = volume;
    return () => {
      seriesRef.current = null;
      volumeRef.current = null;
      chartRef.current = null;
      chart.remove();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 配色与主题原地更新，不重建图表、不重新载入数据
  useEffect(() => {
    chartRef.current?.applyOptions({
      layout: { background: { color: bg }, textColor: fg },
      // 默认网格线为浅灰，暗色下过亮，改用主题边框色
      grid: { vertLines: { color: border }, horzLines: { color: border } },
      rightPriceScale: { borderColor: border },
      timeScale: { borderColor: border },
    });
    seriesRef.current?.applyOptions({
      upColor: up,
      downColor: down,
      borderUpColor: up,
      borderDownColor: down,
      wickUpColor: up,
      wickDownColor: down,
    });
  }, [up, down, bg, fg, border]);

  useEffect(() => {
    const list = candles ?? [];
    seriesRef.current?.setData(list.map(toBar));
    volumeRef.current?.setData(list.map((d) => toVolume(d, up, down)));
  }, [candles, up, down]);

  useEffect(() => {
    if (!liveRef) return;
    liveRef.current = (d) => {
      seriesRef.current?.update(toBar(d));
      volumeRef.current?.update(toVolume(d, up, down));
    };
    return () => {
      liveRef.current = null;
    };
  }, [liveRef, up, down]);

  return <div ref={containerRef} className="chart-canvas" />;
}
