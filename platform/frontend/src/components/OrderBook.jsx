const cell = { textAlign: "right", padding: "1px 8px" };

// 只渲染一侧；levels 已按最优价在前，直接从上到下排列
export default function OrderBook({ levels, color, muted }) {
  const rows = (levels ?? []).slice(0, 20);
  if (rows.length === 0) return <div style={{ color: muted, fontSize: 13 }}>暂无订单簿数据</div>;

  return (
    <table style={{ width: "100%", fontFamily: "monospace", fontSize: 13, borderCollapse: "collapse" }}>
      <thead>
        <tr style={{ color: muted }}>
          <th style={cell}>价格</th>
          <th style={cell}>数量</th>
          <th style={cell}>订单数</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((l, i) => (
          <tr key={i} style={{ color }}>
            <td style={cell}>{l.px}</td>
            <td style={cell}>{l.sz}</td>
            <td style={cell}>{l.n}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
