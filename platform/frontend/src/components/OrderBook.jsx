// 只渲染一侧；levels 已按最优价在前，直接从上到下排列
export default function OrderBook({ levels, side = "buy", label }) {
  const rows = (levels ?? []).slice(0, 20);
  if (rows.length === 0) return <div className="empty-state">暂无订单簿数据</div>;

  return (
    <div className="table-scroll" tabIndex={0} role="region" aria-label={label ?? "订单簿"}>
      <table className={`table ${side === "buy" ? "book-buy" : "book-sell"}`}>
        <thead>
          <tr>
            <th className="num">价格</th>
            <th className="num">数量</th>
            <th className="num">订单数</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((level, i) => (
            <tr key={`${level.px}-${i}`}>
              <td className="num px">{level.px}</td>
              <td className="num">{level.sz}</td>
              <td className="num">{level.n}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
