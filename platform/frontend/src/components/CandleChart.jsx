import { useEffect, useRef } from "react";
import { createChart } from "lightweight-charts";

const toBar = (d) => ({
  time: Math.floor(d.t / 1000),
  open: Number(d.o),
  high: Number(d.h),
  low: Number(d.l),
  close: Number(d.c),
});

export default function CandleChart({ candles, liveRef, up, down, bg, fg, border }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef(null);

  useEffect(() => {
    // autoSize tracks the container's size, so no manual resize handling is needed
    const chart = createChart(containerRef.current, { autoSize: true });
    chartRef.current = chart;
    seriesRef.current = chart.addCandlestickSeries();
    return () => {
      seriesRef.current = null;
      chartRef.current = null;
      chart.remove();
    };
  }, []);

  // 配色与主题原地更新，不重建图表、不重新载入数据
  useEffect(() => {
    chartRef.current?.applyOptions({
      layout: { background: { color: bg }, textColor: fg },
      // 默认网格线为浅灰，暗色下过亮，改用主题边框色
      grid: { vertLines: { color: border }, horzLines: { color: border } },
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
    seriesRef.current?.setData((candles ?? []).map(toBar));
  }, [candles]);

  useEffect(() => {
    if (!liveRef) return;
    liveRef.current = (d) => seriesRef.current?.update(toBar(d));
    return () => {
      liveRef.current = null;
    };
  }, [liveRef]);

  return <div ref={containerRef} style={{ width: "100%", height: 400 }} />;
}
