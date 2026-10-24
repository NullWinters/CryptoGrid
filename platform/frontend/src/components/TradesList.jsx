export default function TradesList({ trades, up, down, muted }) {
  if (!trades || trades.length === 0) {
    return <div style={{ color: muted, fontSize: 13, padding: 8 }}>暂无最新成交</div>;
  }

  return (
    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
      <thead>
        <tr style={{ color: muted, textAlign: "right" }}>
          <th style={{ textAlign: "left", fontWeight: 500, padding: "2px 6px" }}>时间</th>
          <th style={{ fontWeight: 500, padding: "2px 6px" }}>价格</th>
          <th style={{ fontWeight: 500, padding: "2px 6px" }}>数量</th>
        </tr>
      </thead>
      <tbody>
        {trades.map((t, i) => (
          <tr key={i} style={{ color: t.side === "B" ? up : down }}>
            <td style={{ padding: "2px 6px", fontFamily: "monospace" }}>
              {new Date(t.time).toLocaleTimeString()}
            </td>
            <td style={{ padding: "2px 6px", fontFamily: "monospace", textAlign: "right" }}>{t.px}</td>
            <td style={{ padding: "2px 6px", fontFamily: "monospace", textAlign: "right" }}>{t.sz}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
