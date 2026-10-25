import { useLayoutEffect, useRef } from "react";

export default function TradesList({ trades }) {
  const scrollRef = useRef(null);
  const prevHeight = useRef(0);
  const prevTop = useRef(0);
  const atTop = useRef(true);

  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (prevHeight.current > 0) {
      if (atTop.current) {
        el.scrollTop = 0;
      } else {
        el.scrollTop = prevTop.current + (el.scrollHeight - prevHeight.current);
      }
    }
    prevHeight.current = el.scrollHeight;
    prevTop.current = el.scrollTop;
  }, [trades]);

  function onScroll() {
    const el = scrollRef.current;
    if (!el) return;
    atTop.current = el.scrollTop <= 8;
    prevTop.current = el.scrollTop;
  }

  if (!trades || trades.length === 0) {
    return <div className="empty-state">暂无最新成交</div>;
  }

  return (
    <div
      className="trades-scroll"
      ref={scrollRef}
      onScroll={onScroll}
      tabIndex={0}
      role="region"
      aria-label="最新市场成交"
    >
      <table className="table">
        <thead>
          <tr>
            <th>时间</th>
            <th>方向</th>
            <th className="num">价格</th>
            <th className="num">数量</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((t, i) => {
            const buy = t.side === "B";
            return (
              <tr key={`${t.time}-${t.tid ?? i}`}>
                <td className="mono">{new Date(t.time).toLocaleTimeString()}</td>
                <td className={buy ? "up" : "down"}>{buy ? "买" : "卖"}</td>
                <td className={`num ${buy ? "up" : "down"}`}>{t.px}</td>
                <td className={`num ${buy ? "up" : "down"}`}>{t.sz}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
